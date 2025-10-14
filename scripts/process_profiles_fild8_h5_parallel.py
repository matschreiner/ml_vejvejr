#!/usr/bin/env python3
"""
Batch process multiple fild8 profile files to daily HDF5 with parallel parsing.

This script parallelizes the CPU-intensive parsing step while serializing
HDF5 writes to avoid file corruption.

Usage
-----
process_profiles_fild8_h5_parallel.py <input_dir> <output_dir> [--workers N]

Author: Carlos Peralta (parallelized version)
Date: 2025-01-XX
"""
import os
import sys
import h5py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import argparse
import gzip
import fcntl
from contextlib import contextmanager

DEPTHS = 15  # number of vertical layers


@contextmanager
def file_lock(file_path):
    """
    Context manager for file locking to ensure safe concurrent HDF5 writes.
    """
    lock_file = Path(str(file_path) + '.lock')
    lock_fd = None
    try:
        lock_fd = open(lock_file, 'w')
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if lock_fd:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
            lock_fd.close()
            try:
                lock_file.unlink()
            except:
                pass


def parse_raw(filename):
    """
    Parse the raw fild8 file and extract temperature profiles.
    
    Returns
    -------
    timestamp  : datetime      (already shifted –2 h if desired)
    station_ids : list[str]    (station identifiers)
    temps      : ndarray shape (n_station, DEPTHS)  float32
    """
    stations = []
    road_stretch = []
    station_names = {}
    line_count = 0
    start_found = False
    single_column_line = None
    temp_profiles = {}

    with open(filename, 'r') as file:
        # Skip the header line
        next(file)
        lines = file.readlines()

        for i, line in enumerate(lines, 1):
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            cols = line.split()

            # First part: collect stations
            if len(cols) == 3:
                road_stretch.append("-".join([cols[0], cols[1], cols[2]]))
                if cols[0] == '0':  # Lines with "0 NUMBER 0"
                    if cols[1] == '10000':
                        start_found = True
                    if start_found:
                        station_number = cols[1]
                        stations.append(int(station_number))

                        # Process station name
                        base_station = station_number[:-2]  # First 4 digits
                        sensor_number = station_number[-2:]  # Last 2 digits

                        # Add to station_names dictionary
                        if base_station not in station_names:
                            station_names[base_station] = []
                        if sensor_number not in station_names[base_station]:
                            station_names[base_station].append(sensor_number)

                        line_count += 1

            # Check for single column transition
            if i < len(lines) - 1:
                next_line = lines[i].strip()
                if len(line.split()) == 1 and len(next_line.split()) == 1:
                    try:
                        float(line)
                        float(next_line)
                        single_column_line = i + 1
                        break
                    except ValueError:
                        continue

    # Calculate number of stations
    num_stations = len(road_stretch)
    # Number of layers in each profile (typically 15)
    num_layers = 15

    # Initialize profiles for each station
    for station_idx in range(num_stations):
        temp_profiles[station_idx] = [0] * num_layers

    # Start collecting temperature profiles by layer
    if single_column_line:
        data_lines = [line.strip() for line in lines[single_column_line-2:] if line.strip()]

        # Process data by layers
        for layer_idx in range(num_layers):
            for station_idx in range(num_stations):
                line_index = layer_idx * num_stations + station_idx
                if line_index < len(data_lines):
                    try:
                        temp_value = float(data_lines[line_index])
                        temp_profiles[station_idx][layer_idx] = temp_value
                    except (ValueError, IndexError):
                        continue

    # Extract timestamp from filename
    filename_base = Path(filename).stem
    date_str = filename_base.split('_')[-1]  # Get YYYYMMDDHH part
    
    # Parse timestamp and apply 2-hour shift
    timestamp = datetime.strptime(date_str, '%Y%m%d%H')
    timestamp = timestamp - timedelta(hours=2)  # subtract 2h since data is from 2h ago
    
    # Convert station identifiers to strings
    station_ids = road_stretch
    
    # Convert temperature profiles to numpy array
    temp_matrix = np.zeros((num_stations, num_layers), dtype='float32')
    for station_idx in range(num_stations):
        if station_idx in temp_profiles:
            temp_matrix[station_idx, :] = temp_profiles[station_idx]
        else:
            temp_matrix[station_idx, :] = np.nan
    
    return timestamp, station_ids, temp_matrix, filename


def parse_file_wrapper(gz_file):
    """
    Wrapper function for parallel processing: decompress and parse a .gz file.
    Returns parsed data or None if error.
    """
    try:
        # Decompress to temporary file
        raw_file = Path(gz_file).stem
        
        with gzip.open(gz_file, 'rb') as f_in:
            with open(raw_file, 'wb') as f_out:
                f_out.write(f_in.read())
        
        # Parse the raw file
        timestamp, station_ids, temp_matrix, filename = parse_raw(raw_file)
        
        # Clean up temporary file
        os.remove(raw_file)
        
        print(f"✓ Parsed {Path(gz_file).name}: {len(station_ids)} stations, timestamp {timestamp}")
        
        return {
            'timestamp': timestamp,
            'station_ids': station_ids,
            'temp_matrix': temp_matrix,
            'source_file': gz_file,
            'success': True
        }
    except Exception as e:
        print(f"✗ Error parsing {gz_file}: {e}")
        return {
            'source_file': gz_file,
            'success': False,
            'error': str(e)
        }


def ensure_datasets(h5f, n_depth):
    """
    Create the datasets the first time the file is opened.
    """
    if "dynamic" in h5f:  # already initialised
        return

    # Create main groups
    static_group = h5f.create_group("static")
    dynamic_group = h5f.create_group("dynamic")

    # Static group structure (placeholders for future use)
    static_group.create_group("shadow")
    static_group.create_group("height")
    static_group.create_group("orientation")
    static_group.create_group("location")

    # Dynamic group: temperature profiles
    temp_profile_group = dynamic_group.create_group("temp_profile")

    # Create resizable datasets
    temp_data = temp_profile_group.create_dataset(
        "temperature",
        shape=(0, 0, n_depth),
        maxshape=(None, None, n_depth),
        dtype='float32',
        chunks=(1, 1000, n_depth),
        compression="gzip",
        compression_opts=4
    )

    timestamps_data = temp_profile_group.create_dataset(
        "timestamps",
        shape=(0,),
        maxshape=(None,),
        dtype='float64',
        chunks=(1000,)
    )

    timestamps_iso = temp_profile_group.create_dataset(
        "timestamps_iso",
        shape=(0,),
        maxshape=(None,),
        dtype=h5py.string_dtype(encoding='utf-8'),
        chunks=(1000,)
    )

    depths = temp_profile_group.create_dataset(
        "depths",
        data=np.arange(n_depth, dtype='float32'),
        dtype='float32'
    )

    station_ids = temp_profile_group.create_dataset(
        "station_ids",
        shape=(0,),
        maxshape=(None,),
        dtype=h5py.string_dtype(encoding='utf-8'),
        chunks=(1000,)
    )

    # Add metadata
    temp_data.attrs['units'] = 'Kelvin'
    temp_data.attrs['description'] = 'Temperature profiles by depth'
    temp_data.attrs['dimensions'] = 'time, station, depth'
    temp_data.attrs['missing_value'] = np.nan
    temp_data.attrs['valid_range'] = [200.0, 350.0]
    temp_data.attrs['dtype'] = 'float32'

    timestamps_data.attrs['units'] = 'seconds since 1970-01-01'
    timestamps_data.attrs['description'] = 'Unix timestamps'

    timestamps_iso.attrs['description'] = 'ISO format timestamps for reference'

    depths.attrs['units'] = 'meters'
    depths.attrs['description'] = 'Depth below surface'
    depths.attrs['note'] = 'Depth indices: 0=surface, 14=deepest'
    depths.attrs['dtype'] = 'float32'

    station_ids.attrs['description'] = 'Station identifiers'
    station_ids.attrs['format'] = 'String identifiers for measurement stations'

    # Add file metadata
    h5f.attrs['created'] = datetime.now().isoformat()
    h5f.attrs['description'] = 'Vertical temperature profiles and meteorological data (daily)'
    h5f.attrs['source'] = 'Converted from fild8 raw files'
    h5f.attrs['contact'] = 'Add your contact information here'
    h5f.attrs['memory_optimized'] = True
    h5f.attrs['dtype_temperature'] = 'float32'
    h5f.attrs['packing_strategy'] = 'daily'


def append_hour(h5_path, ts, stations, temps):
    """
    Append one hour worth of data to the HDF5 file with file locking.
    
    Returns
    -------
    bool : True if data was written, False if already existed
    """
    with file_lock(h5_path):
        with h5py.File(h5_path, "a") as f:
            ensure_datasets(f, temps.shape[1])
            g = f["dynamic/temp_profile"]
            tds = g["temperature"]
            
            # Check if this timestamp already exists (idempotent operation)
            existing_timestamps = g["timestamps"][:]
            ts_unix = ts.timestamp()
            if ts_unix in existing_timestamps:
                return False
            
            # -------------------------------- station axis --------------------
            new_stations = []
            if len(g["station_ids"]) == 0:
                # First time - add all stations
                st_map = {s: i for i, s in enumerate(stations)}
                new_stations = stations
            else:
                # Build mapping from existing stations
                st_map = {s.decode(): i for i, s in enumerate(g["station_ids"][:])}
                for s in stations:
                    if s not in st_map:
                        new_stations.append(s)
                        st_map[s] = len(st_map)
            
            if new_stations:  # extend station axis
                old_n_stations = len(g["station_ids"])
                new_n_stations = len(st_map)
                g["station_ids"].resize(new_n_stations, axis=0)
                g["station_ids"][old_n_stations:] = [s.encode() for s in new_stations]
                tds.resize((tds.shape[0], new_n_stations, tds.shape[2]))
            
            # -------------------------------- time axis -----------------------
            tds.resize(tds.shape[0] + 1, axis=0)
            g["timestamps"].resize(g["timestamps"].shape[0] + 1, axis=0)
            g["timestamps_iso"].resize(g["timestamps_iso"].shape[0] + 1, axis=0)

            time_idx = tds.shape[0] - 1
            tds[time_idx, :, :] = np.nan  # pre-fill with NaN
            
            # Fill in the temperature data
            for i, (s, vec) in enumerate(zip(stations, temps)):
                if s in st_map:
                    station_idx = st_map[s]
                    tds[time_idx, station_idx, :] = vec
            
            g["timestamps"][time_idx] = ts_unix
            g["timestamps_iso"][time_idx] = str(ts).encode()
            
            return True


def process_batch(input_dir, output_dir, workers=None):
    """
    Process all .gz files in input_dir using parallel parsing.
    
    Parameters
    ----------
    input_dir : Path
        Directory containing fild8_*.gz files
    output_dir : Path
        Directory for output HDF5 files
    workers : int, optional
        Number of worker processes (default: cpu_count())
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if workers is None:
        workers = cpu_count()
    
    # Find all .gz files
    gz_files = sorted(input_dir.glob("fild8_*.gz"))
    
    if not gz_files:
        print(f"No fild8_*.gz files found in {input_dir}")
        return
    
    print(f"Found {len(gz_files)} files to process")
    print(f"Using {workers} worker processes for parsing")
    print("=" * 60)
    
    # Parallel parsing phase
    print("\n[Phase 1/2] Parsing files in parallel...")
    with Pool(processes=workers) as pool:
        results = pool.map(parse_file_wrapper, gz_files)
    
    # Separate successful and failed results
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\nParsing complete: {len(successful)} successful, {len(failed)} failed")
    
    if failed:
        print("\nFailed files:")
        for f in failed:
            print(f"  - {Path(f['source_file']).name}: {f['error']}")
    
    # Group by day for sequential HDF5 writing
    print("\n[Phase 2/2] Writing to HDF5 files (grouped by day)...")
    
    by_day = {}
    for result in successful:
        day_tag = result['timestamp'].strftime("%Y%m%d")
        if day_tag not in by_day:
            by_day[day_tag] = []
        by_day[day_tag].append(result)
    
    print(f"Data spans {len(by_day)} unique day(s)")
    
    # Write each day's data
    written_count = 0
    skipped_count = 0
    
    for day_tag in sorted(by_day.keys()):
        h5_file = output_dir / f"road_temp_{day_tag}.h5"
        day_results = by_day[day_tag]
        
        print(f"\n[Day {day_tag}] Writing {len(day_results)} hour(s) to {h5_file.name}")
        
        for result in sorted(day_results, key=lambda x: x['timestamp']):
            was_written = append_hour(
                h5_file,
                result['timestamp'],
                result['station_ids'],
                result['temp_matrix']
            )
            
            if was_written:
                print(f"  ✓ {result['timestamp']:%Y-%m-%d %H:%M}")
                written_count += 1
            else:
                print(f"  ⊘ {result['timestamp']:%Y-%m-%d %H:%M} (already exists)")
                skipped_count += 1
    
    print("\n" + "=" * 60)
    print(f"Processing complete!")
    print(f"  Written: {written_count} hours")
    print(f"  Skipped: {skipped_count} hours (already existed)")
    print(f"  Failed:  {len(failed)} files")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Batch process fild8 files to HDF5 with parallel parsing'
    )
    parser.add_argument('input_dir', type=str, help='Directory containing fild8_*.gz files')
    parser.add_argument('output_dir', type=str, help='Output directory for HDF5 files')
    parser.add_argument('--workers', type=int, default=None,
                       help=f'Number of worker processes (default: {cpu_count()})')
    
    args = parser.parse_args()
    
    process_batch(args.input_dir, args.output_dir, args.workers)


if __name__ == "__main__":
    main()
