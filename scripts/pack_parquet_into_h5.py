#!/usr/bin/env python3
"""
Temperature Profile Converter: Parquet to HDF5
===============================================

Converts vertical temperature profile data from parquet files to HDF5 format.
Organizes data by year with hierarchical structure for static and dynamic variables.

Usage:
    python convert_temperature_profiles.py

Author: Carlos Peralta
Date: 2025-07-01
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TemperatureProfileConverter:
    """
    Converter class for temperature profile data from parquet to HDF5

    File Structure:
    ===============
    road_temp_YYYY.h5
    ├── static/
    │   ├── shadow/          # Placeholder for shadow data
    │   └── height/          # Placeholder for station heights  
    └── dynamic/
        ├── temp_profile/
        │   ├── temperature  # [time, station, depth] array
        │   ├── station_ids  # Station identifier strings
        │   ├── timestamps   # Unix timestamps
        │   └── depths       # Depth values in meters
        ├── radiation/       # Placeholder
        ├── wind/            # Placeholder
        └── prec/            # Placeholder
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

    def get_parquet_files_by_year(self, year):
        """Get all parquet files for a specific year"""
        pattern = f"road_temp_{year}????.parquet"
        files = list(self.input_dir.glob(pattern))
        return sorted(files)

    def get_available_years(self):
        """Get all available years from parquet files"""
        parquet_files = list(self.input_dir.glob("road_temp_????????.parquet"))
        years = set()
        for file in parquet_files:
            # Extract year from filename like "road_temp_20210731.parquet"
            try:
                date_part = file.stem.split('_')[-1]  # Get "20210731"
                year = int(date_part[:4])  # Get "2021"
                years.add(year)
            except (ValueError, IndexError):
                logger.warning(f"Could not extract year from filename: {file}")
        return sorted(years)

    def create_hdf5_structure(self, year):
        """Create the HDF5 file structure for a given year"""
        filename = self.output_dir / f"road_temp_{year}.h5"

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
            dynamic_group.create_group("wind")       # Placeholder
            dynamic_group.create_group("prec")       # Placeholder

            # Add file metadata
            f.attrs['created'] = datetime.now().isoformat()
            f.attrs['year'] = year
            f.attrs['description'] = 'Vertical temperature profiles and meteorological data'
            f.attrs['source'] = 'Converted from parquet files'
            f.attrs['contact'] = 'Add your contact information here'

        return filename

    def convert_year(self, year, chunk_size=None):
        """
        Convert all parquet files for a year to HDF5

        Args:
            year (int): Year to process
            chunk_size (int): Process data in chunks to manage memory usage
        """
        logger.info(f"Processing year {year}...")

        # Get parquet files for the year
        parquet_files = self.get_parquet_files_by_year(year)
        if not parquet_files:
            logger.warning(f"No parquet files found for year {year}")
            return None

        logger.info(f"Found {len(parquet_files)} parquet files for {year}")

        # Create HDF5 structure
        hdf5_file = self.create_hdf5_structure(year)

        # Process files in chunks if specified
        if chunk_size and len(parquet_files) > chunk_size:
            self._convert_year_chunked(parquet_files, hdf5_file, chunk_size)
        else:
            self._convert_year_all_at_once(parquet_files, hdf5_file)

        logger.info(f"Successfully created: {hdf5_file}")
        return hdf5_file

    def _convert_year_all_at_once(self, parquet_files, hdf5_file):
        """Convert all files at once (for smaller datasets)"""
        # Read all parquet files
        dfs = []
        for file in parquet_files:
            logger.info(f"  Reading {file.name}...")
            df = pd.read_parquet(file)
            dfs.append(df)

        # Combine and process
        combined_df = pd.concat(dfs, ignore_index=True)
        self._write_temperature_data(hdf5_file, combined_df)

    def _convert_year_chunked(self, parquet_files, hdf5_file, chunk_size):
        """Convert files in chunks (for large datasets)"""
        logger.info(f"Processing in chunks of {chunk_size} files...")

        all_dfs = []
        for i in range(0, len(parquet_files), chunk_size):
            chunk_files = parquet_files[i:i+chunk_size]
            logger.info(f"  Processing chunk {i//chunk_size + 1}/{(len(parquet_files)-1)//chunk_size + 1}")

            chunk_dfs = []
            for file in chunk_files:
                df = pd.read_parquet(file)
                chunk_dfs.append(df)

            chunk_combined = pd.concat(chunk_dfs, ignore_index=True)
            all_dfs.append(chunk_combined)

        # Combine all chunks
        combined_df = pd.concat(all_dfs, ignore_index=True)
        self._write_temperature_data(hdf5_file, combined_df)

    def _write_temperature_data(self, hdf5_file, df):
        """Write temperature data to HDF5 file"""
        # Sort and prepare data
        df = df.sort_values(['timestamp', 'station_id'])

        # Get metadata
        unique_stations = sorted(df['station_id'].unique())
        unique_timestamps = sorted(df['timestamp'].unique())
        depth_cols = [col for col in df.columns if col.startswith('depth_')]
        n_depths = len(depth_cols)

        logger.info(f"  Stations: {len(unique_stations)}")
        logger.info(f"  Timestamps: {len(unique_timestamps)}")
        logger.info(f"  Depths: {n_depths}")

        with h5py.File(hdf5_file, 'a') as f:
            temp_group = f['dynamic/temp_profile']

            n_stations = len(unique_stations)
            n_times = len(unique_timestamps)

            # Create datasets with optimal chunking and compression
            chunk_shape = (min(100, n_times), min(10, n_stations), n_depths)

            temp_data = temp_group.create_dataset(
                'temperature', 
                shape=(n_times, n_stations, n_depths),
                dtype='f4',
                compression='gzip',
                compression_opts=6,
                chunks=chunk_shape,
                fillvalue=np.nan,
                shuffle=True,
                fletcher32=True  # Add checksum for data integrity
            )

            station_ids = temp_group.create_dataset(
                'station_ids',
                data=[s.encode('utf-8') for s in unique_stations],
                dtype=h5py.string_dtype()
            )

            timestamps_data = temp_group.create_dataset(
                'timestamps',
                data=[pd.Timestamp(ts).timestamp() for ts in unique_timestamps],
                dtype='f8'
            )

            # Convert timestamps to readable format for reference
            timestamps_iso = temp_group.create_dataset(
                'timestamps_iso',
                data=[str(ts).encode('utf-8') for ts in unique_timestamps],
                dtype=h5py.string_dtype()
            )

            depths = temp_group.create_dataset(
                'depths',
                data=np.arange(n_depths, dtype='f4'),
                dtype='f4'
            )

            # Create lookup dictionaries for efficient indexing
            station_to_idx = {station: idx for idx, station in enumerate(unique_stations)}
            time_to_idx = {ts: idx for idx, ts in enumerate(unique_timestamps)}

            # Fill temperature data efficiently
            logger.info("  Writing temperature data...")
            temp_array = np.full((n_times, n_stations, n_depths), np.nan, dtype='f4')

            for _, row in df.iterrows():
                time_idx = time_to_idx[row['timestamp']]
                station_idx = station_to_idx[row['station_id']]
                temp_values = [row[col] for col in depth_cols]
                temp_array[time_idx, station_idx, :] = temp_values

            # Write array to dataset
            temp_data[:] = temp_array

            # Add comprehensive attributes
            temp_data.attrs['units'] = 'Kelvin'
            temp_data.attrs['description'] = 'Temperature profiles by depth'
            temp_data.attrs['dimensions'] = 'time, station, depth'
            temp_data.attrs['missing_value'] = np.nan
            temp_data.attrs['valid_range'] = [200.0, 350.0]  # Reasonable temperature range

            timestamps_data.attrs['units'] = 'seconds since 1970-01-01'
            timestamps_data.attrs['description'] = 'Unix timestamps'

            timestamps_iso.attrs['description'] = 'ISO format timestamps for reference'

            depths.attrs['units'] = 'meters'
            depths.attrs['description'] = 'Depth below surface'
            depths.attrs['note'] = 'Depth indices: 0=surface, 14=deepest'

            station_ids.attrs['description'] = 'Station identifiers'
            station_ids.attrs['format'] = 'String identifiers for measurement stations'

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Convert temperature profile parquet files to HDF5')
    parser.add_argument('--input-dir', '-i', 
                       default='/data/projects/glatmodel/obs/fild8/road_profiles_daily/',
                       help='Input directory containing parquet files')
    parser.add_argument('--output-dir', '-o',
                       default='/data/projects/glatmodel/obs/fild8/hdf5_yearly/',
                       help='Output directory for HDF5 files')
    parser.add_argument('--year', '-y', type=int,
                       help='Specific year to process (if not specified, processes all available years)')
    parser.add_argument('--chunk-size', '-c', type=int, default=50,
                       help='Number of parquet files to process at once (for memory management)')
    parser.add_argument('--list-years', '-l', action='store_true',
                       help='List available years and exit')

    args = parser.parse_args()

    # Initialize converter
    try:
        converter = TemperatureProfileConverter(args.input_dir, args.output_dir)
    except FileNotFoundError as e:
        logger.error(e)
        return 1

    # List available years if requested
    if args.list_years:
        years = converter.get_available_years()
        print(f"Available years: {years}")
        return 0

    # Process specific year or all years
    if args.year:
        converter.convert_year(args.year, args.chunk_size)
    else:
        years = converter.get_available_years()
        logger.info(f"Processing all available years: {years}")
        for year in years:
            try:
                converter.convert_year(year, args.chunk_size)
            except Exception as e:
                logger.error(f"Failed to process year {year}: {e}")
                continue

    logger.info("Conversion completed!")
    return 0

if __name__ == '__main__':
    exit(main())
