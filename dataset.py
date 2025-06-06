import numpy as np
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.input_data, self.target_data = self.load_data(path)

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return {
            "input": torch.tensor(self.input_data[idx], dtype=torch.float32),
            "target": torch.tensor(self.target_data[idx], dtype=torch.float32),
        }

    def load_data(self, path):
        """
        Load data from an NPZ file containing 'input' and 'target' arrays.
        
        Args:
            path (str): Path to the NPZ file
            
        Returns:
            tuple: (input_data, target_data) where both are numpy arrays
                   with shape (n_samples, depth)
        """
        data = np.load(path)
        input_data = data['input']
        target_data = data['target']
        
        print(f"Loaded NPZ data - Input: {input_data.shape}, Target: {target_data.shape}")
        
        # Ensure both arrays have the same number of samples
        #assert input_data.shape[0] == target_data.shape[0], \
        #    f"Input and target must have same number of samples: {input_data.shape[0]} vs {target_data.shape[0]}"
        
        return input_data, target_data
