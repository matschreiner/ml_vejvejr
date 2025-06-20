import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import Dataset
from model import Model

def main(args):
    depth = 15
    model = Model(depth)
    train_dataset = Dataset(args.train_path)
    val_dataset = Dataset(args.val_path)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    trainer = pl.Trainer(max_epochs=args.epochs)
    trainer.fit(model, train_loader, val_loader)
    plot_losses(model)

def plot_losses(model):
    if hasattr(model, "losses") and model.losses:
        plt.figure(figsize=(10, 6))
        plt.plot(model.losses, label="Train Loss")
        if hasattr(model, "val_losses") and model.val_losses:
            plt.plot(model.val_losses, label="Validation Loss")
        plt.xlabel("Training Steps")
        plt.ylabel("MSE Loss")
        plt.title("Time Series Prediction Loss")
        plt.legend()
        plt.grid(True)
        plt.show()
        print(f"Final train loss: {model.losses[-1]:.6f}")
        if hasattr(model, "val_losses") and model.val_losses:
            print(f"Final val loss: {model.val_losses[-1]:.6f}")
        print(f"Training completed with {len(model.losses)} steps")
    else:
        print("No losses recorded during training")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the temperature profile time series model.")
    parser.add_argument("train_path", type=str, help="Path to the training NPZ data file")
    parser.add_argument("val_path", type=str, help="Path to the validation NPZ data file")
    parser.add_argument("epochs", type=int, help="Number of training epochs", default=50)
    main(parser.parse_args())
