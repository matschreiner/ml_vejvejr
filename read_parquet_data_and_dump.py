"""
Test using the parquet data.
This example does not take into account the temporal
order...

"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import glob
import os
from datetime import datetime

def load_parquet_files(pattern, station_id):
    """Load temperature profiles from parquet files matching pattern"""
    files = sorted(glob.glob(pattern))
    if not files:
        raise ValueError(f"No files found matching pattern: {pattern}")
    
    all_profiles = []
    depth_cols = [f'depth_{i}' for i in range(15)]
    
    for file_path in files:
        print(f"Loading {file_path}...")
        
        try:
            df = pd.read_parquet(file_path)
            
            # Get station_id if not specified
            if station_id is None:
                station_id = df['station_id'].iloc[0]
                print(f"Using station: {station_id}")
            
            # Filter for specific station
            station_data = df[df['station_id'] == station_id]
            
            if len(station_data) == 0:
                print(f"  No data for station {station_id} in {file_path}")
                continue
            
            # Extract temperature profiles
            profiles = station_data[depth_cols].values
            
            all_profiles.append(profiles)
            
            print(f"  Loaded {len(profiles)} samples")
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    if not all_profiles:
        raise ValueError(f"No valid data found for station {station_id}")
    combined_profiles = np.vstack(all_profiles)
    print(f"Total profiles loaded: {len(combined_profiles)}")
    print(f"Temperature range: {combined_profiles.min():.2f} - {combined_profiles.max():.2f} K")
    
    return combined_profiles, station_id

def convert_parquet_to_ml_format(
    input_parquet_pattern,
    target_parquet_pattern, 
    station_id=None,
    output_file='road_temp_data.npz'
):
    """
    Convert road temperature parquet files to ML training format.
    
    Parameters:
    -----------
    input_parquet_pattern : str
        Pattern to match input parquet files (e.g., 'data/road_temp_202203*.parquet')
    target_parquet_pattern : str  
        Pattern to match target parquet files (e.g., 'data/road_temp_202204*.parquet')
    station_id : str, optional
        Specific station ID to use. If None, uses the first available station.
    output_file : str
        Path to save the converted NPZ file
        
    Returns:
    --------
    dict with keys: 'input', 'target'
    """
    
    
    # Load input and target data
    print("=== Loading Input Data ===")
    input_profiles, used_station_id = load_parquet_files(
        input_parquet_pattern, station_id
    )

    print("\n=== Loading Target Data ===")
    target_profiles, _ = load_parquet_files(
        target_parquet_pattern, used_station_id
    )
    
    # Ensure same number of samples. The code seems to assume this somehow?
    min_samples = min(len(input_profiles), len(target_profiles))
    print(f"Using {min_samples} samples for training")
    input_profiles = input_profiles[:min_samples]
    target_profiles = target_profiles[:min_samples]
    
    print(f"\n=== Data Processing ===")
    
    # normalize to [0,1] range
    input_scaler = MinMaxScaler()
    input_data = input_scaler.fit_transform(input_profiles)
    
    # convert to Celsius and standardize
    #target_celsius = target_profiles - 273.15  # K to C
    target_scaler = StandardScaler()
    target_data = target_scaler.fit_transform(target_profiles)
    
    print(f"Input data - Shape: {input_data.shape}, Range: [{input_data.min():.3f}, {input_data.max():.3f}]")
    print(f"Target data - Shape: {target_data.shape}, Range: [{target_data.min():.3f}, {target_data.max():.3f}]")
    
    # Save data
    np.savez_compressed(
        output_file,
        input=input_data,
        target=target_data
    )
    
    print(f"\nData saved to: {output_file}")
    
    return {
        'input': input_data,
        'target': target_data, 
        'input_scaler': input_scaler,
        'target_scaler': target_scaler
    }

def load_converted_data(npz_file):
    """Load and inspect converted ML data"""
    data = np.load(npz_file, allow_pickle=True)
    
    print(f"Loaded data from: {npz_file}")
    print(f"Arrays: {list(data.keys())}")
    
    input_data = data['input']
    target_data = data['target']
    
    print(f"\\nData shapes:")
    print(f"  Input: {input_data.shape}")
    print(f"  Target: {target_data.shape}")
    
    return input_data, target_data

# Example usage:
if __name__ == "__main__":
    data_path="/data/projects/glatmodel/obs/fild8/road_profiles_daily"
    # Convert data from consecutive periods
    # Input: March 2022 data, Target: 2022 03 11 data
    result = convert_parquet_to_ml_format(
        input_parquet_pattern=os.path.join(data_path,'road_temp_2022030*.parquet'),
        target_parquet_pattern=os.path.join(data_path,'road_temp_20220311.parquet'),
        station_id='0-100000-0',  # or None to auto-select
        output_file='road_temp_march_april_2022.npz'
    )
    
    # Load and inspect converted data
    #input_data, target_data = load_converted_data('road_temp_march_april_2022.npz')
