import numpy as np
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.data = self.load_data(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input": torch.tensor(self.data[idx], dtype=torch.float32),
            "target": torch.tensor(self.data[idx + 1], dtype=torch.float32),
        }

    def load_data(self, path):
        # This function should take a path or whatever arguments, and create some kind of array on disk
        # or in memory that can be indexed with data[idx]
        #
        # Its dimension should be n_samples x depth.
        # so if you have 1000 datapoints, data[0] should give you the temperature profile of the first
        data = ...
        return data


#  class Dataset(torch.utils.data.Dataset):
#      def __init__(
#          self,
#      ):
#          # This should be instantiated such that the getitem function can access all the necessary
#          # data for a single sample. This could be done by loading all the data into memory or reading
#          # it from disk on demand.
#          ...
#
#      def __len__(self):
#          # This function is necessary for the training framework to work
#          return len(self.data)
#
#      def __getitem__(self, idx):
#          # This function should be able to fetch all the necessary data for a single sample
#          # identified by an idx. So with 100 stations with 1_000 samples each we have a dataset
#          # of length 100_000.
#
#          temp_profile = ...
#          station_id = ...
#          zenith = ...
#          azimuth = ...
#          forcing = ...
#          temp_profile_delta = ...
#          shadow = ...
#
#          # All this data will be returned in a dictionary such that the model can access it
#
#          return {
#              "temp_profile": torch.tensor(temp_profile),  # array, temperature profile at
#              "station_id": torch.tensor(station_id),  # int, unique station id
#              "zenith": torch.tensor(zenith),  # float, zenith
#              "azimuth": torch.tensor(azimuth),  # float, azimuth
#              "height_grid": self.height_grid,  # grid with height information
#              "forcing": torch.tensor(forcing),  # forcing variables
#              "temp_profile_delta": temp_profile_delta,  #  array, temperature profile delta
#              "shadow": torch.tensor(shadow),  # shadow information
#          }
