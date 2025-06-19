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
        self.val_losses = []

    def training_step(self, batch, _):
        target_hat = self.forward(batch)
        target = batch["target"]

        loss = torch.nn.functional.mse_loss(target_hat, target)
        self.log("train_loss", loss, prog_bar=True)

        self.losses.append(loss.item())

        return loss

    def validation_step(self, batch, _):
        loss = self.training_step(batch, _)
        self.val_losses.append(loss.item())
        return loss
        

    def test_step(self, batch, _):
        return self.training_step(batch, _)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class Model(TempProfileModel):
    def __init__(self, dim_temp):
        super().__init__()
        self.mlp = MLP(dim_input=dim_temp, dim_output=dim_temp)

    def forward(self, batch):
        # This function takes a batch of data and processes it through the model and
        # outputs the model prediction.

        # input_ = concat batch['input'] batch['tod'] #this is a place holder for later
        # in the above example on would concat the input and tod to create extended features
        return self.mlp(batch["input"])
