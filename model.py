import numpy as np
import pytorch_lightning as pl
import torch


class MLP(torch.nn.Module):
    def __init__(self, dim_input, dim_output):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim_input, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, dim_output),
        )

    def forward(self, x):
        return self.net(x)


class TempProfileModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.losses = []

    def training_step(self, batch, _):
        target_hat = self.forward(batch)
        target = batch["target"]

        loss = torch.nn.functional.mse_loss(target_hat, target)
        self.log("train_loss", loss, prog_bar=True)

        self.losses.append(loss.item())

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class Model(TempProfileModel):
    def __init__(self, dim_temp):
        super().__init__()
        self.mlp = MLP(dim_input=dim_temp, dim_output=dim_temp)

    def forward(self, batch):
        # This function takes a batch of data and processes it through the model and
        # outputs the model prediction.
        return self.mlp(batch["input"])


#  class Model(TempProfileModel):
#      def __init__(self):
#          super().__init__()
#          # The Embbedding module takes a unique ID and embeds it into a vector of a given dimension.
#          # This embedding is optimized during training to best represent each station, such that
#          # the model can represent features that we haven't thought of or are not providing.
#          self.station_embedding = torch.nn.Embedding(n_locations, dim_station_embedding)
#
#          # The CNN Module an image processing module that can extract features from images, and the
#          # idea is that it should extract features from the height grid aronud the station.
#          # The output from the network is flattened such that it can be concatenated with the rest of the
#          # features
#          # The dimensions of the network should be tuned together with the size of the surorunding grid.
#          self.cnn = CNN()  # Convolutional Neural Network for image processing
#
#          # Maybe there could also be some interaction between the shadow/zenith/azimuth, not sure)
#
#          self.readout = MLP(
#              input_dim=dim_station_embedding
#              + dim_cnn_output
#              + dim_input_temp
#              + dim_forcing
#              + dim_other_features,
#              output_dim=dim_input_temp,  # depends on what we want to learn with the model
#          )
#
#      def forward(self, batch):
#          # This function takes a batch of data and processes it through the model and
#          # outputs the model prediction.
#          temp_profile = batch["temp_profile"]
#          station_embedding = self.station_embedding(batch["station_id"])
#          height_grid_embedding = self.cnn(batch["height_grid"].unsqueeze(1))
#
#          # Here we combing all the featuers into a single input vector for the readout module.
#          combined_input = torch.cat(
#              [
#                  station_embedding,
#                  height_grid_embedding,
#                  batch["forcing"],
#                  temp_profile,
#                  other_features,
#              ],
#              dim=-1,
#          )
#
#          return self.readout(combined_input)
#
#      def training_step(self, batch, _):
#          # This function is called during training, it receives a batch and should be able to
#          # calculate a loss signal that the optimizer can use to update the model parameters.
#          # This can be calculated both from the whole delta x profile or just from the delta of
#          # the surface temperature.
#
#          x_phys, loc, zenith, azimuth, y = batch
#          output = self.forward(x_phys, loc, zenith, azimuth)
#          loss = torch.nn.functional.mse_loss(output, y)
#          self.log("train_loss", loss)
#          return loss
#
#  class CNN(torch.nn.Module):
#      def __init__(self, input_channels=1):
#          super().__init__()
#          self.net = torch.nn.Sequential(
#              torch.nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
#              torch.nn.ReLU(),
#              torch.nn.MaxPool2d(2),
#              torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
#              torch.nn.ReLU(),
#              torch.nn.MaxPool2d(2),
#              torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
#              torch.nn.ReLU(),
#              torch.nn.MaxPool2d(2),
#              torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
#              torch.nn.ReLU(),
#              torch.nn.MaxPool2d(2),
#          )
#
#      def forward(self, x):
#          return self.net(x).flatten()
