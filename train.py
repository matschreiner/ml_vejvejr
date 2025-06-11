import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset import Dataset  # Updated import
from model import Model  # Use the fixed model


def main(args):
    depth = 15 #number of layers
    model = Model(depth)
    dataset = Dataset(args.path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    trainer = pl.Trainer(max_epochs=200) #using 50 worked okay for 10 layers but not for 15
    trainer.fit(model, dataloader)
    plot_losses(model)


def plot_losses(model):
    if model.losses:
        plt.figure(figsize=(10, 6))
        plt.plot(model.losses)
        plt.xlabel("Training Steps")
        plt.ylabel("MSE Loss")
        plt.title("Time Series Prediction Training Loss")
        plt.grid(True)
        plt.show()
        
        print(f"Final loss: {model.losses[-1]:.6f}")
        print(f"Training completed with {len(model.losses)} steps")
    else:
        print("No losses recorded during training")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the temperature profile time series model.")
    parser.add_argument("path", type=str, help="Path to the NPZ data file")
    main(parser.parse_args())
