"""
Read parquet data and create proper time series for temperature profile prediction.
This version creates sequential temporal data where we predict profile at time t+1 
from profile at time t.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import glob
import os
from datetime import datetime, timedelta

def load_parquet_files_timeseries(pattern, station_id, max_depth=15):
    """
    Load temperature profiles from parquet files and organize as time series.
    
    Parameters:
    -----------
    pattern : str
        Pattern to match parquet files (e.g., 'data/road_temp_2022*.parquet')
    station_id : str
        Specific station ID to extract
    max_depth : int
        Number of depth levels to use (default 15, can use 10 to compare with original)
    
    Returns:
    --------
    pandas.DataFrame with columns: timestamp, depth_0, depth_1, ..., depth_n
    """
    files = sorted(glob.glob(pattern))
    if not files:
        raise ValueError(f"No files found matching pattern: {pattern}")
    
    all_data = []
    depth_cols = [f'depth_{i}' for i in range(max_depth)]
    
    for file_path in files:
        print(f"Loading {file_path}...")
        
        try:
            df = pd.read_parquet(file_path)
            
            # Filter for specific station
            station_data = df[df['station_id'] == station_id].copy()
            if len(station_data) == 0:
                print(f"  No data for station {station_id} in {file_path}")
                continue
            
            # Ensure we have datetime information
            if 'timestamp' not in station_data.columns:
                # Try to extract from filename or create synthetic datetime
                print(f"  Warning: No datetime column found in {file_path}")
                continue
            
            # Select only the depth columns we need
            available_depths = [col for col in depth_cols if col in station_data.columns]
            if len(available_depths) < max_depth:
                print(f"  Warning: Only {len(available_depths)} depth columns available")
                # Pad with the last available depth if needed
                for i in range(len(available_depths), max_depth):
                    station_data[f'depth_{i}'] = station_data[available_depths[-1]]
            
            # Keep datetime and depth columns
            cols_to_keep = ['timestamp'] + depth_cols
            station_data = station_data[cols_to_keep]
            
            all_data.append(station_data)
            print(f"  Loaded {len(station_data)} samples")
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    if not all_data:
        raise ValueError(f"No valid data found for station {station_id}")
    
    # Combine all data and sort by datetime
    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data = combined_data.sort_values('timestamp').reset_index(drop=True)
    
    print(f"Total profiles loaded: {len(combined_data)}")
    print(f"Time range: {combined_data['timestamp'].min()} to {combined_data['timestamp'].max()}")
    
    return combined_data

def create_timeseries_dataset(
    parquet_pattern,
    station_id,
    output_file='road_temp_timeseries.npz',
    max_depth=10,
    time_step_hours=1
):
    """
    Create time series dataset for temperature profile prediction.
    
    Parameters:
    -----------
    parquet_pattern : str
        Pattern to match parquet files
    station_id : str
        Station ID to use
    output_file : str
        Output NPZ file path
    max_depth : int
        Number of depth levels to use
    time_step_hours : int
        Hours between consecutive predictions (1 = predict next hour)
    
    Returns:
    --------
    dict with 'input', 'target' arrays
    """
    
    print("=== Loading Time Series Data ===")
    df = load_parquet_files_timeseries(parquet_pattern, station_id, max_depth)
    
    depth_cols = [f'depth_{i}' for i in range(max_depth)]
    profiles = df[depth_cols].values
    
    print(f"\\n=== Creating Time Series Pairs ===")
    print(f"Original profiles shape: {profiles.shape}")
    
    # Create input-target pairs for time series prediction
    # Input: profile at time t, Target: profile at time t+time_step
    input_profiles = []
    target_profiles = []
    
    for i in range(len(profiles) - time_step_hours):
        input_profiles.append(profiles[i])
        target_profiles.append(profiles[i + time_step_hours])
    
    input_profiles = np.array(input_profiles)
    target_profiles = np.array(target_profiles)
    
    print(f"Time series pairs created:")
    print(f"  Input shape: {input_profiles.shape}")
    print(f"  Target shape: {target_profiles.shape}")
    print(f"  Time step: {time_step_hours} hours")
    
    # Normalize data
    print(f"\\n=== Data Normalization ===")
    
    # Use MinMaxScaler to normalize to [0,1] range
    scaler = MinMaxScaler()
    
    # Fit scaler on input data and transform both input and target
    input_normalized = scaler.fit_transform(input_profiles)
    target_normalized = scaler.transform(target_profiles)
    
    print(f"Input data - Range: [{input_normalized.min():.3f}, {input_normalized.max():.3f}]")
    print(f"Target data - Range: [{target_normalized.min():.3f}, {target_normalized.max():.3f}]")
    print(f"Original temperature range: [{profiles.min():.2f}, {profiles.max():.2f}] °C")
    
    # Save data
    np.savez_compressed(
        output_file,
        input=input_normalized,
        target=target_normalized,
        # Save scaler parameters for later use
        scaler_min=scaler.data_min_,
        scaler_scale=scaler.scale_,
        original_temp_range=[profiles.min(), profiles.max()],
        time_step_hours=time_step_hours
    )
    
    print(f"\\nTime series data saved to: {output_file}")
    
    return {
        'input': input_normalized,
        'target': target_normalized,
        'scaler': scaler,
        'original_profiles': profiles
    }

def load_timeseries_data(npz_file):
    """Load and inspect time series data"""
    data = np.load(npz_file, allow_pickle=True)
    
    print(f"Loaded time series data from: {npz_file}")
    print(f"Arrays: {list(data.keys())}")
    
    input_data = data['input']
    target_data = data['target']
    
    print(f"\\nData shapes:")
    print(f"  Input: {input_data.shape}")
    print(f"  Target: {target_data.shape}")
    
    if 'time_step_hours' in data:
        print(f"  Time step: {data['time_step_hours']} hours")
    
    if 'original_temp_range' in data:
        temp_range = data['original_temp_range']
        print(f"  Original temperature range: [{temp_range[0]:.2f}, {temp_range[1]:.2f}] K")
    
    return input_data, target_data

# Example usage:
if __name__ == "__main__":
    data_path = "/data/projects/glatmodel/obs/fild8/road_profiles_daily"
    
    # Create time series dataset from multiple days/months
    result = create_timeseries_dataset(
        parquet_pattern=os.path.join(data_path, 'road_temp_202203*.parquet'),
        station_id='0-100000-0',  # Station ID
        output_file='road_temp_timeseries.npz',
        max_depth=15,  # Number of layers
        time_step_hours=1  # Predict 1 hour ahead
    )
    
    # Load and inspect the created dataset
    input_data, target_data = load_timeseries_data('road_temp_timeseries.npz')
