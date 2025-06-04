import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset import Dataset
from model import Model


def main(args):
    depth = 10
    model = Model(depth)
    dataset = Dataset(args.path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    trainer = pl.Trainer(max_epochs=50)
    trainer.fit(model, dataloader)
    plot_losses(model)


def plot_losses(model):
    plt.plot(model.losses)
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the temperature profile model.")
    parser.add_argument("path", type=str)
    main(parser.parse_args())
