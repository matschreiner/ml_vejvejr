import os
import numpy as np


def main():
    # Generate sequential temperature profiles (time series)
    n_timesteps = 101  # Generate 101 timesteps so we can use 100 input-target pairs
    n_depths = 15 # number of layers
    
    # Initialize first temperature profile
    temp_profiles = np.zeros((n_timesteps, n_depths))
    temp_profiles[0] = np.random.rand(n_depths)  # Random initial state
    
    # Generate temporal evolution with some physics-inspired dynamics
    W = np.random.randn(n_depths, n_depths) * 0.01  # Transition matrix (small values for stability). Using 0.1 is very unstable for more layers
    
    for t in range(1, n_timesteps):
        # Temperature evolution: current state + linear transformation + noise
        temp_profiles[t] = (temp_profiles[t-1] + 
                           temp_profiles[t-1].dot(W) + 
                           0.05 * np.random.randn(n_depths))
        
    # Save as input-target pairs for time series prediction
    # Input: t=0 to t=99, Target: t=1 to t=100
    os.makedirs("data", exist_ok=True)
    np.savez(
        "data/test_data.npz",
        input=temp_profiles[:-1],   # Shape: (100, 10) - timesteps 0 to 99
        target=temp_profiles[1:],   # Shape: (100, 10) - timesteps 1 to 100
    )
    
    print(f"Generated time series data:")
    print(f"Input shape: {temp_profiles[:-1].shape}")
    print(f"Target shape: {temp_profiles[1:].shape}")
    print(f"Temperature range: [{temp_profiles.min():.3f}, {temp_profiles.max():.3f}]")


if __name__ == "__main__":
    main()
