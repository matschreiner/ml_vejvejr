#!/usr/bin/env python3
"""
Convert one fild8 profile file to (or append into) daily HDF5.

Usage
-----
process_profiles_fild8_h5.py  fild8_YYYYMMDDHH  <out_dir>

The script is idempotent: running it twice on the same raw file
does not duplicate data inside the HDF5 container.

Author: Carlos Peralta
Date: 2025-07-02
"""
import os
import sys
import h5py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

DEPTHS = 15  # number of vertical layers

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
    
    print(f"Parsed {filename}: {len(station_ids)} stations, timestamp {timestamp}")
    
    return timestamp, station_ids, temp_matrix

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

    # Dynamic group structure
    temp_profile_group = dynamic_group.create_group("temp_profile")
    dynamic_group.create_group("radiation")  # Placeholder
    dynamic_group.create_group("wind")       # Placeholder
    dynamic_group.create_group("prec")       # Placeholder

    # Create resizable datasets
    temp_profile_group.create_dataset(
        "temperature",
        shape=(0, 0, n_depth),
        maxshape=(None, None, n_depth),
        chunks=(24, 100, n_depth),  # good for daily writes
        dtype="f4",
        compression="gzip", 
        compression_opts=6, 
        shuffle=True, 
        fletcher32=True,
        fillvalue=np.nan
    )
    
    temp_profile_group.create_dataset(
        "timestamps", 
        shape=(0,), 
        maxshape=(None,), 
        dtype="f8"
    )
    
    temp_profile_group.create_dataset(
        "timestamps_iso", 
        shape=(0,), 
        maxshape=(None,),
        dtype=h5py.string_dtype()
    )
    
    temp_profile_group.create_dataset(
        "station_ids", 
        shape=(0,), 
        maxshape=(None,),
        dtype=h5py.string_dtype()
    )
    
    temp_profile_group.create_dataset(
        "depths", 
        data=np.arange(n_depth, dtype="f4")
    )

    # Add attributes
    temp_data = temp_profile_group["temperature"]
    temp_data.attrs['units'] = 'Kelvin'
    temp_data.attrs['description'] = 'Temperature profiles by depth'
    temp_data.attrs['dimensions'] = 'time, station, depth'
    temp_data.attrs['missing_value'] = np.nan
    temp_data.attrs['valid_range'] = [200.0, 350.0]
    temp_data.attrs['dtype'] = 'float32'

    timestamps_data = temp_profile_group["timestamps"]
    timestamps_data.attrs['units'] = 'seconds since 1970-01-01'
    timestamps_data.attrs['description'] = 'Unix timestamps'

    timestamps_iso = temp_profile_group["timestamps_iso"]
    timestamps_iso.attrs['description'] = 'ISO format timestamps for reference'

    depths = temp_profile_group["depths"]
    depths.attrs['units'] = 'meters'
    depths.attrs['description'] = 'Depth below surface'
    depths.attrs['note'] = 'Depth indices: 0=surface, 14=deepest'
    depths.attrs['dtype'] = 'float32'

    station_ids = temp_profile_group["station_ids"]
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
    Append one hour worth of data to the HDF5 file.
    """
    with h5py.File(h5_path, "a") as f:
        ensure_datasets(f, temps.shape[1])
        g = f["dynamic/temp_profile"]
        tds = g["temperature"]
        
        # Check if this timestamp already exists (idempotent operation)
        existing_timestamps = g["timestamps"][:]
        ts_unix = ts.timestamp()
        if ts_unix in existing_timestamps:
            print(f"  Timestamp {ts} already exists in {h5_path}, skipping...")
            return
        
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

def main():
    if len(sys.argv) != 3:
        print("Usage: process_profiles_fild8_h5.py <raw_file> <output_dir>")
        sys.exit(1)
        
    raw_file = Path(sys.argv[1])  # fild8_YYYYMMDDHH
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        ts, station_ids, temp_mat = parse_raw(raw_file)
        day_tag = ts.strftime("%Y%m%d")
        h5_file = out_dir / f"road_temp_{day_tag}.h5"

        append_hour(h5_file, ts, station_ids, temp_mat)
        print(f"→  wrote hour {ts:%Y-%m-%d %H:%M} to {h5_file}")
        
        # Print some statistics
        file_size = h5_file.stat().st_size / (1024 * 1024)  # MB
        print(f"   File size: {file_size:.2f} MB")
        print(f"   Stations: {len(station_ids)}")
        print(f"   Temperature matrix shape: {temp_mat.shape}")
        
    except FileNotFoundError:
        print(f"File {raw_file} not found. Please check the filename and path.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
