import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset import Dataset
from model import Model

from model_analysis import analyze_model_performance, plot_training_loss

def main(args):
    depth = 15 #10
    model = Model(depth)
    dataset = Dataset(args.path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    trainer = pl.Trainer(max_epochs=200) # 50)
    trainer.fit(model, dataloader)
    plot_losses(model)

    # adding the more comprehensive analysis here
    #print("\nAnalyzing model performance...")
    #analyze_model_performance(model, dataset)


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
