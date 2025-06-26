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
        target_hat = self.forward(batch)
        target = batch["target"]
        val_loss = torch.nn.functional.mse_loss(target_hat, target)
        self.log("val_loss", val_loss, prog_bar=True)
        self.val_losses.append(val_loss.item())
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

class Model(TempProfileModel):
    def __init__(self, dim_temp):
        super().__init__()
        self.mlp = MLP(dim_input=dim_temp, dim_output=dim_temp)

    def forward(self, batch):
        return self.mlp(batch["input"])
