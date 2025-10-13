#!/usr/bin/env python3
"""
HDF5 Temperature Profile Explorer
=================================

Utility for inspecting and extracting data from yearly HDF5 files
created by the temperature profile converter.

Usage:
    python explore_temperature_hdf5.py --file road_temp_2021.h5

Author: Carlos Peralta
Date: 2025-07-01
"""

import h5py
import numpy as np
import argparse
import pandas as pd

def print_hdf5_structure(h5file):
    def print_structure(name, obj):
        indent = "  " * name.count('/')
        if isinstance(obj, h5py.Group):
            print(f"{indent}{name}/ (Group)")
        else:
            print(f"{indent}{name} (Dataset: {obj.shape}, {obj.dtype})")
    print("\nHDF5 File Structure:")
    h5file.visititems(print_structure)

def summarize_temperature_data(h5file):
    temp_path = 'dynamic/temp_profile/temperature'
    if temp_path not in h5file:
        print("No temperature data found in file.")
        return
    temp = h5file[temp_path][:]
    stations = [s.decode('utf-8') for s in h5file['dynamic/temp_profile/station_ids'][:]]
    timestamps = h5file['dynamic/temp_profile/timestamps'][:]
    depths = h5file['dynamic/temp_profile/depths'][:]
    print(f"\nTemperature array shape: {temp.shape}")
    print(f"Number of stations: {len(stations)}")
    print(f"Number of timestamps: {len(timestamps)}")
    print(f"Number of depths: {len(depths)}")
    print(f"Temperature range: {np.nanmin(temp):.2f} - {np.nanmax(temp):.2f} K")
    print(f"First 3 stations: {stations[:3]}")
    print(f"First 3 timestamps: {[pd.to_datetime(ts, unit='s') for ts in timestamps[:3]]}")
    print(f"Depths: {depths}")

def extract_profile(h5file, station_id, timestamp_idx=0):
    stations = [s.decode('utf-8') for s in h5file['dynamic/temp_profile/station_ids'][:]]
    timestamps = h5file['dynamic/temp_profile/timestamps'][:]
    depths = h5file['dynamic/temp_profile/depths'][:]
    temp = h5file['dynamic/temp_profile/temperature'][:]
    if station_id not in stations:
        print(f"Station {station_id} not found.")
        return
    station_idx = stations.index(station_id)
    profile = temp[timestamp_idx, station_idx, :]
    print(f"\nTemperature profile for station {station_id} at time {pd.to_datetime(timestamps[timestamp_idx], unit='s')}:")
    for i, t in enumerate(profile):
        print(f"  Depth {depths[i]:.2f} m: {t:.2f} K")

def main():
    parser = argparse.ArgumentParser(description="Explore HDF5 temperature profile files")
    parser.add_argument('--file', '-f', required=True, help='Path to HDF5 file')
    parser.add_argument('--station', '-s', help='Station ID to extract profile for')
    parser.add_argument('--time-idx', '-t', type=int, default=0, help='Time index for profile extraction')
    args = parser.parse_args()

    with h5py.File(args.file, 'r') as h5file:
        print_hdf5_structure(h5file)
        summarize_temperature_data(h5file)
        if args.station:
            extract_profile(h5file, args.station, args.time_idx)

if __name__ == '__main__':
    main()
