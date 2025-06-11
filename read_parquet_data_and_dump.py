"""
Read parquet data and create proper time series for temperature profile prediction.
This version creates sequential temporal data where we predict profile at time t+1 
from profile at time t.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
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
        Number of depth levels to use (default 10, since your model expects 10)

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

            # Use 'timestamp' column (corrected from 'datetime')
            if 'timestamp' not in station_data.columns:
                print(f"  Warning: No timestamp column found in {file_path}")
                print(f"  Available columns: {list(station_data.columns)}")
                continue

            # Select only the depth columns we need
            available_depths = [col for col in depth_cols if col in station_data.columns]
            if len(available_depths) < max_depth:
                print(f"  Warning: Only {len(available_depths)} depth columns available")
                # Pad with the last available depth if needed
                for i in range(len(available_depths), max_depth):
                    station_data[f'depth_{i}'] = station_data[available_depths[-1]]

            # Keep timestamp and depth columns
            cols_to_keep = ['timestamp'] + depth_cols
            station_data = station_data[cols_to_keep]
            station_data[depth_cols] = station_data[depth_cols] - 273.15 #convert to Celsius

            all_data.append(station_data)
            print(f"  Loaded {len(station_data)} samples")

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

    if not all_data:
        raise ValueError(f"No valid data found for station {station_id}")

    # Combine all data and sort by timestamp
    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data = combined_data.sort_values('timestamp').reset_index(drop=True)

    print(f"Total profiles loaded: {len(combined_data)}")
    print(f"Time range: {combined_data['timestamp'].min()} to {combined_data['timestamp'].max()}")

    return combined_data

def plot_training_data_example(df, max_depth=15, time_step_hours=1, n_examples=3):
    """
    Plot examples of what the model is being trained to predict.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with timestamp and depth columns
    max_depth : int
        Number of depth levels
    time_step_hours : int
        Hours between input and target
    n_examples : int
        Number of example pairs to plot
    """
    depth_cols = [f'depth_{i}' for i in range(max_depth)]
    depths = np.arange(max_depth)  # Depth indices (0 to 9)

    fig, axes = plt.subplots(n_examples, 1, figsize=(12, 4*n_examples))
    if n_examples == 1:
        axes = [axes]

    # Select random time points for examples
    valid_indices = range(len(df) - time_step_hours)
    example_indices = np.random.choice(valid_indices, n_examples, replace=False)

    for i, idx in enumerate(example_indices):
        # Get input and target profiles
        input_profile = df.iloc[idx][depth_cols].values
        target_profile = df.iloc[idx + time_step_hours][depth_cols].values

        input_time = df.iloc[idx]['timestamp']
        target_time = df.iloc[idx + time_step_hours]['timestamp']

        # Plot profiles
        axes[i].plot(depths, input_profile, 'b-o', label=f'Input (t={input_time})', linewidth=2, markersize=6)
        axes[i].plot(depths, target_profile, 'r-s', label=f'Target (t={target_time})', linewidth=2, markersize=6)

        axes[i].set_xlabel('Depth Level')
        axes[i].set_ylabel('Temperature (°C)')
        axes[i].set_title(f'Training Example {i+1}: Predict {time_step_hours}h ahead')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].invert_yaxis()  # Deeper depths at bottom

    plt.tight_layout()
    plt.savefig('training_examples.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Training visualization saved as 'training_examples.png'")

def plot_time_series_overview(df, max_depth=15, depth_to_plot=0):
    """
    Plot time series overview showing temperature evolution at one depth.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with timestamp and depth columns
    max_depth : int
        Number of depth levels
    depth_to_plot : int
        Which depth level to plot (0 = surface)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Plot 1: Time series at selected depth
    depth_col = f'depth_{depth_to_plot}'
    ax1.plot(df['timestamp'], df[depth_col], 'b-', linewidth=1)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title(f'Temperature Time Series at Depth {depth_to_plot} (Surface)')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Temperature profile heatmap over time (sample of data)
    # Take every Nth sample to make heatmap readable
    sample_every = max(1, len(df) // 200)  # Show ~200 time points
    sampled_df = df.iloc[::sample_every]

    depth_cols = [f'depth_{i}' for i in range(max_depth)]
    temp_matrix = sampled_df[depth_cols].values.T  # Transpose for proper orientation

    im = ax2.imshow(temp_matrix, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
    ax2.set_xlabel('Time Index (sampled)')
    ax2.set_ylabel('Depth Level')
    ax2.set_title('Temperature Profile Heatmap Over Time')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Temperature (°C)')

    plt.tight_layout()
    plt.savefig('timeseries_overview.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Time series overview saved as 'timeseries_overview.png'")

def create_timeseries_dataset(
    parquet_pattern,
    station_id,
    output_file='road_temp_timeseries.npz',
    max_depth=15,
    time_step_hours=1,
    plot_examples=True
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
    plot_examples : bool
        Whether to create visualization plots

    Returns:
    --------
    dict with 'input', 'target' arrays
    """

    print("=== Loading Time Series Data ===")
    df = load_parquet_files_timeseries(parquet_pattern, station_id, max_depth)

    if plot_examples:
        print("\n=== Creating Visualizations ===")
        plot_time_series_overview(df, max_depth)
        plot_training_data_example(df, max_depth, time_step_hours)

    depth_cols = [f'depth_{i}' for i in range(max_depth)]
    profiles = df[depth_cols].values

    print(f"\n=== Creating Time Series Pairs ===")
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
    print(f"\n=== Data Normalization ===")

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

    print(f"\nTime series data saved to: {output_file}")

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

    print(f"\nData shapes:")
    print(f"  Input: {input_data.shape}")
    print(f"  Target: {target_data.shape}")

    if 'time_step_hours' in data:
        print(f"  Time step: {data['time_step_hours']} hours")

    if 'original_temp_range' in data:
        temp_range = data['original_temp_range']
        print(f"  Original temperature range: [{temp_range[0]:.2f}, {temp_range[1]:.2f}] °C")

    return input_data, target_data

if __name__ == "__main__":
    data_path = "/data/projects/glatmodel/obs/fild8/road_profiles_daily"

    # Create time series dataset from multiple days/months
    result = create_timeseries_dataset(
        parquet_pattern=os.path.join(data_path, 'road_temp_2022030*.parquet'),
        station_id='0-100000-0',  # Station ID
        output_file='road_temp_timeseries.npz',
        max_depth=15,  # number of layers
        time_step_hours=1,  # Predict 1 hour ahead
        plot_examples=True  # Create visualization plots
    )

    # Load and inspect the created dataset
    input_data, target_data = load_timeseries_data('road_temp_timeseries.npz')
