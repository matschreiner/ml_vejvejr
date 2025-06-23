import numpy as np
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.input_data, self.target_data = self.load_data(path)

    def __len__(self):
        # For time series, we have n-1 valid pairs from n timesteps
        return len(self.input_data)

    def __getitem__(self, idx):
        # include time of day

        return {
            #"tod": torch.tensor ... #placeholder for later
            "input": torch.tensor(self.input_data[idx], dtype=torch.float32),
            "target": torch.tensor(self.target_data[idx], dtype=torch.float32),  # Already offset in data generation, so not using idx+1 here
        }

    def load_data(self, path):
        """
        Load time series data from an NPZ file containing 'input' and 'target' arrays.
        
        For time series prediction:
        - Input: temperature profiles at time t
        - Target: temperature profiles at time t+1
        
        Args:
            path (str): Path to the NPZ file
        
        Returns:
            tuple: (input_data, target_data) where both are numpy arrays
            with shape (n_samples, depth). input_data[i] corresponds to time t,
            target_data[i] corresponds to time t+1.
        """
        data = np.load(path)
        input_data = data['input']   # Shape: (n_samples, n_depths) - timesteps t
        target_data = data['target'] # Shape: (n_samples, n_depths) - timesteps t+1
        
        print(f"Loaded time series data - Input: {input_data.shape}, Target: {target_data.shape}")
        return input_data, target_data
