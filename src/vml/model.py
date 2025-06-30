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
        self.losses = []      # Per-epoch average train loss
        self.val_losses = []  # Per-epoch average val loss

    def on_train_epoch_start(self):
        self._epoch_train_losses = []

    def training_step(self, batch, _):
        target_hat = self.forward(batch)
        target = batch["target"]
        loss = torch.nn.functional.mse_loss(target_hat, target)
        self.log("train_loss", loss, prog_bar=True)
        self._epoch_train_losses.append(loss.item())
        return loss

    def on_train_epoch_end(self):
        avg_loss = np.mean(self._epoch_train_losses)
        self.losses.append(avg_loss)

    def on_validation_epoch_start(self):
        self._epoch_val_losses = []

    def validation_step(self, batch, _):
        target_hat = self.forward(batch)
        target = batch["target"]
        val_loss = torch.nn.functional.mse_loss(target_hat, target)
        self.log("val_loss", val_loss, prog_bar=True)
        self._epoch_val_losses.append(val_loss.item())
        return val_loss

    def on_validation_epoch_end(self):
        avg_val_loss = np.mean(self._epoch_val_losses)
        self.val_losses.append(avg_val_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

class Model(TempProfileModel):
    def __init__(self, dim_temp):
        super().__init__()
        self.mlp = MLP(dim_input=dim_temp, dim_output=dim_temp)

    def forward(self, batch):
        return self.mlp(batch["input"])
