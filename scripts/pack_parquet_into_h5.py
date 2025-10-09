#!/usr/bin/env python3
"""
Temperature Profile Converter: Parquet to HDF5 (Monthly Packing)
====

Converts vertical temperature profile data from parquet files to HDF5 format.
Organizes data by month with hierarchical structure for static and dynamic variables.
Uses streaming processing to minimize memory usage.

Usage:
    python convert_temperature_profiles_monthly.py

Author: Carlos Peralta (Modified for monthly packing)
Date: 2025-07-02
"""

import pandas as pd
import h5py
import numpy as np
from datetime import datetime
import os
import glob
from pathlib import Path
import argparse
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TemperatureProfileConverter:
    """
    Memory-optimized converter class for temperature profile data from parquet to HDF5
    Modified to pack data by month instead of year

    File Structure:
    ====
    road_temp_YYYY_MM.h5
    ├── static/
    │   ├── shadow/    # Placeholder for shadow data
    │   └── height/    # Placeholder for station heights  
    └── dynamic/
        ├── temp_profile/
        │   ├── temperature  # [time, station, depth] array
        │   ├── station_ids  # Station identifier strings
        │   ├── timestamps   # Unix timestamps
        │   └── depths    # Depth values in meters
        ├── radiation/    # Placeholder
        ├── wind/    # Placeholder
        └── prec/    # Placeholder
    """

    def __init__(self, input_dir, output_dir):
        """
        Initialize converter

        Args:
            input_dir (str): Directory containing parquet files
            output_dir (str): Directory for output HDF5 files
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    def get_parquet_files_by_month(self, year, month):
        """Get all parquet files for a specific year and month"""
        # Format month with leading zero
        month_str = f"{month:02d}"
        pattern = f"road_temp_{year}{month_str}*.parquet"
        files = list(self.input_dir.glob(pattern))
        return sorted(files)

    def get_available_months(self):
        """Get all available year-month combinations from parquet files"""
        parquet_files = list(self.input_dir.glob("road_temp_????????.parquet"))
        months = set()
        for file in parquet_files:
            # Extract year-month from filename like "road_temp_20210731.parquet"
            try:
                date_part = file.stem.split('_')[-1]  # Get "20210731"
                if len(date_part) >= 6:
                    year = int(date_part[:4])  # Get "2021"
                    month = int(date_part[4:6])  # Get "07"
                    months.add((year, month))
            except (ValueError, IndexError):
                logger.warning(f"Could not extract year-month from filename: {file}")
        return sorted(months)

    def _scan_files_for_metadata(self, parquet_files):
        """
        Scan all files to get complete metadata without loading all data
        Returns: (all_stations, all_timestamps, depth_columns)
        """
        logger.info("Scanning files for metadata...")
        all_stations = set()
        all_timestamps = set()
        depth_cols = None
        
        for i, file in enumerate(parquet_files):
            if i % 5 == 0:  # More frequent logging for smaller monthly batches
                logger.info(f"  Scanning file {i+1}/{len(parquet_files)}: {file.name}")
            
            # Read only the columns we need for metadata
            df_meta = pd.read_parquet(file, columns=['station_id', 'timestamp'])
            all_stations.update(df_meta['station_id'].unique())
            all_timestamps.update(df_meta['timestamp'].unique())
            
            # Get depth columns from first file
            if depth_cols is None:
                full_cols = pd.read_parquet(file, columns=None).columns
                depth_cols = [col for col in full_cols if col.startswith('depth_')]
            
            # Clear memory
            del df_meta
        
        return sorted(all_stations), sorted(all_timestamps), depth_cols

    def create_hdf5_structure(self, year, month, stations, timestamps, depth_cols):
        """Create the HDF5 file structure for a given year-month with known dimensions"""
        filename = self.output_dir / f"road_temp_{year}_{month:02d}.h5"

        with h5py.File(filename, 'w') as f:
            # Create main groups
            static_group = f.create_group("static")
            dynamic_group = f.create_group("dynamic")

            # Static group structure (placeholders for future use)
            static_group.create_group("shadow")
            static_group.create_group("height")

            # Dynamic group structure
            temp_profile_group = dynamic_group.create_group("temp_profile")
            dynamic_group.create_group("radiation")  # Placeholder
            dynamic_group.create_group("wind")    # Placeholder
            dynamic_group.create_group("prec")    # Placeholder

            # Create datasets with optimal chunking and compression
            n_stations = len(stations)
            n_times = len(timestamps)
            n_depths = len(depth_cols)
            
            # Optimal chunk shape for time series data (adjusted for monthly data)
            chunk_shape = (min(50, n_times), min(10, n_stations), n_depths)

            # Temperature data - using float32 to save memory
            temp_data = temp_profile_group.create_dataset(
                'temperature', 
                shape=(n_times, n_stations, n_depths),
                dtype='f4',  # float32
                compression='gzip',
                compression_opts=6,
                chunks=chunk_shape,
                fillvalue=np.nan,
                shuffle=True,
                fletcher32=True
            )

            # Station IDs
            station_ids = temp_profile_group.create_dataset(
                'station_ids',
                data=[s.encode('utf-8') for s in stations],
                dtype=h5py.string_dtype()
            )

            # Timestamps - using float64 for precision
            timestamps_data = temp_profile_group.create_dataset(
                'timestamps',
                data=[pd.Timestamp(ts).timestamp() for ts in timestamps],
                dtype='f8'  # Keep float64 for timestamp precision
            )

            # ISO timestamps for reference
            timestamps_iso = temp_profile_group.create_dataset(
                'timestamps_iso',
                data=[str(ts).encode('utf-8') for ts in timestamps],
                dtype=h5py.string_dtype()
            )

            # Depths - using float32
            depths = temp_profile_group.create_dataset(
                'depths',
                data=np.arange(n_depths, dtype='f4'),
                dtype='f4'
            )

            # Add comprehensive attributes
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
            f.attrs['created'] = datetime.now().isoformat()
            f.attrs['year'] = year
            f.attrs['month'] = month
            f.attrs['description'] = 'Vertical temperature profiles and meteorological data (monthly)'
            f.attrs['source'] = 'Converted from parquet files'
            f.attrs['contact'] = 'Add your contact information here'
            f.attrs['memory_optimized'] = True
            f.attrs['dtype_temperature'] = 'float32'
            f.attrs['packing_strategy'] = 'monthly'

        return filename

    def convert_month(self, year, month, chunk_size=5):
        """
        Convert all parquet files for a specific month to HDF5 using streaming approach

        Args:
            year (int): Year to process
            month (int): Month to process (1-12)
            chunk_size (int): Number of files to process in each streaming chunk
        """
        logger.info(f"Processing {year}-{month:02d} with streaming approach...")

        # Get parquet files for the month
        parquet_files = self.get_parquet_files_by_month(year, month)
        if not parquet_files:
            logger.warning(f"No parquet files found for {year}-{month:02d}")
            return None

        logger.info(f"Found {len(parquet_files)} parquet files for {year}-{month:02d}")

        # Scan files for complete metadata
        all_stations, all_timestamps, depth_cols = self._scan_files_for_metadata(parquet_files)
        
        logger.info(f"  Total stations: {len(all_stations)}")
        logger.info(f"  Total timestamps: {len(all_timestamps)}")
        logger.info(f"  Depth columns: {len(depth_cols)}")

        # Create HDF5 structure with known dimensions
        hdf5_file = self.create_hdf5_structure(year, month, all_stations, all_timestamps, depth_cols)

        # Stream process files
        self._stream_process_files(parquet_files, hdf5_file, all_stations, all_timestamps, depth_cols, chunk_size)

        logger.info(f"Successfully created: {hdf5_file}")
        return hdf5_file

    def _stream_process_files(self, parquet_files, hdf5_file, all_stations, all_timestamps, depth_cols, chunk_size):
        """
        Stream process files in chunks, writing directly to HDF5
        """
        # Create lookup dictionaries
        station_to_idx = {station: idx for idx, station in enumerate(all_stations)}
        time_to_idx = {ts: idx for idx, ts in enumerate(all_timestamps)}
        
        n_times = len(all_timestamps)
        n_stations = len(all_stations)
        n_depths = len(depth_cols)

        logger.info("Starting streaming processing...")
        
        # Process files in chunks
        for chunk_start in range(0, len(parquet_files), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(parquet_files))
            chunk_files = parquet_files[chunk_start:chunk_end]
            
            logger.info(f"  Processing chunk {chunk_start//chunk_size + 1}/{(len(parquet_files)-1)//chunk_size + 1} "
                       f"({len(chunk_files)} files)")
            
            # Load and process chunk
            chunk_dfs = []
            for file in chunk_files:
                df = pd.read_parquet(file)
                # Convert temperature columns to float32 immediately
                for col in depth_cols:
                    if col in df.columns:
                        df[col] = df[col].astype('float32')
                chunk_dfs.append(df)
            
            # Combine chunk
            chunk_df = pd.concat(chunk_dfs, ignore_index=True)
            
            # Clear individual DataFrames from memory
            del chunk_dfs
            
            # Sort chunk data
            chunk_df = chunk_df.sort_values(['timestamp', 'station_id'])
            
            # Write chunk data to HDF5
            self._write_chunk_to_hdf5(hdf5_file, chunk_df, station_to_idx, time_to_idx, depth_cols)
            
            # Clear chunk from memory
            del chunk_df
            
            logger.info(f"    Chunk {chunk_start//chunk_size + 1} completed and memory cleared")

    def _write_chunk_to_hdf5(self, hdf5_file, chunk_df, station_to_idx, time_to_idx, depth_cols):
        """
        Write a chunk of data directly to HDF5 file
        """
        with h5py.File(hdf5_file, 'a') as f:
            temp_dataset = f['dynamic/temp_profile/temperature']
            
            # Process each row in the chunk
            for _, row in chunk_df.iterrows():
                try:
                    time_idx = time_to_idx[row['timestamp']]
                    station_idx = station_to_idx[row['station_id']]
                    
                    # Extract temperature values as float32
                    temp_values = np.array([row[col] for col in depth_cols], dtype='float32')
                    
                    # Write to HDF5 dataset
                    temp_dataset[time_idx, station_idx, :] = temp_values
                    
                except KeyError as e:
                    logger.warning(f"Skipping row due to missing key: {e}")
                    continue

    def get_file_info(self, hdf5_file):
        """Get information about the created HDF5 file"""
        with h5py.File(hdf5_file, 'r') as f:
            temp_data = f['dynamic/temp_profile/temperature']
            file_size = os.path.getsize(hdf5_file) / (1024**2)  # MB for monthly files
            
            info = {
                'file_size_mb': round(file_size, 2),
                'shape': temp_data.shape,
                'dtype': temp_data.dtype,
                'compression': temp_data.compression,
                'chunks': temp_data.chunks,
                'n_stations': len(f['dynamic/temp_profile/station_ids']),
                'n_timestamps': len(f['dynamic/temp_profile/timestamps']),
                'n_depths': temp_data.shape[2]
            }
            
            return info

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Convert temperature profile parquet files to HDF5 (Monthly Packing)')
    parser.add_argument('--input-dir', '-i', 
                       default='/data/projects/glatmodel/obs/fild8/road_profiles_daily/',
                       help='Input directory containing parquet files')
    parser.add_argument('--output-dir', '-o',
                       default='/data/projects/glatmodel/obs/fild8/hdf5_monthly/',
                       help='Output directory for HDF5 files')
    parser.add_argument('--year', '-y', type=int,
                       help='Specific year to process')
    parser.add_argument('--month', '-m', type=int,
                       help='Specific month to process (1-12, requires --year)')
    parser.add_argument('--chunk-size', '-c', type=int, default=5,
                       help='Number of parquet files to process at once (reduced for monthly processing)')
    parser.add_argument('--list-months', '-l', action='store_true',
                       help='List available year-month combinations and exit')

    args = parser.parse_args()

    # Initialize converter
    try:
        converter = TemperatureProfileConverter(args.input_dir, args.output_dir)
    except FileNotFoundError as e:
        logger.error(e)
        return 1

    # List available months if requested
    if args.list_months:
        months = converter.get_available_months()
        print(f"Available year-month combinations: {len(months)}")
        for year, month in months:
            print(f"  {year}-{month:02d}")
        return 0

    # Process specific month or all months
    if args.year and args.month:
        # Process specific month
        hdf5_file = converter.convert_month(args.year, args.month, args.chunk_size)
        if hdf5_file:
            info = converter.get_file_info(hdf5_file)
            logger.info(f"File info: {info}")
    elif args.year:
        # Process all months for a specific year
        months = converter.get_available_months()
        year_months = [(y, m) for y, m in months if y == args.year]
        logger.info(f"Processing all months for year {args.year}: {len(year_months)} months")
        for year, month in year_months:
            try:
                hdf5_file = converter.convert_month(year, month, args.chunk_size)
                if hdf5_file:
                    info = converter.get_file_info(hdf5_file)
                    logger.info(f"Month {year}-{month:02d} - File info: {info}")
            except Exception as e:
                logger.error(f"Failed to process month {year}-{month:02d}: {e}")
                continue
    else:
        # Process all available months
        months = converter.get_available_months()
        logger.info(f"Processing all available months: {len(months)} total")
        for year, month in months:
            try:
                hdf5_file = converter.convert_month(year, month, args.chunk_size)
                if hdf5_file:
                    info = converter.get_file_info(hdf5_file)
                    logger.info(f"Month {year}-{month:02d} - File info: {info}")
            except Exception as e:
                logger.error(f"Failed to process month {year}-{month:02d}: {e}")
                continue

    logger.info("Conversion completed!")
    return 0

if __name__ == '__main__':
    exit(main())
