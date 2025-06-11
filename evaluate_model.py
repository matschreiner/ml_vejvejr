"""
Evaluate and visualize trained temperature profile prediction model.
This script loads a trained model and shows its predictions vs actual targets.
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from sklearn.metrics import mean_squared_error, mean_absolute_error

from dataset import Dataset
from model import Model


def load_trained_model(checkpoint_path, depth=15):
    """
    Load a trained model from checkpoint.
    
    Parameters:
    -----------
    checkpoint_path : str
        Path to the model checkpoint (.ckpt file)
    depth : int
        Model depth dimension
    
    Returns:
    --------
    Trained model
    """
    model = Model.load_from_checkpoint(checkpoint_path, dim_temp=depth)
    model.eval()  # Set to evaluation mode
    return model

def evaluate_model(model, dataset, n_samples=100):
    """
    Evaluate model on dataset and return predictions vs targets.
    
    Parameters:
    -----------
    model : trained Model
        The trained PyTorch Lightning model
    dataset : Dataset
        Dataset to evaluate on
    n_samples : int
        Number of samples to evaluate (for speed)
    
    Returns:
    --------
    dict with predictions, targets, inputs, and metrics
    """
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    all_predictions = []
    all_targets = []
    all_inputs = []
    
    model.eval()
    with torch.no_grad():
        sample_count = 0
        for batch in dataloader:
            if sample_count >= n_samples:
                break
                
            # Get model predictions
            predictions = model(batch)
            
            # Store results
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(batch["target"].cpu().numpy())
            all_inputs.append(batch["input"].cpu().numpy())
            
            sample_count += len(batch["input"])
    
    # Concatenate all results
    predictions = np.vstack(all_predictions)[:n_samples]
    targets = np.vstack(all_targets)[:n_samples]
    inputs = np.vstack(all_inputs)[:n_samples]
    
    # Calculate metrics
    mse = mean_squared_error(targets.flatten(), predictions.flatten())
    mae = mean_absolute_error(targets.flatten(), predictions.flatten())
    
    print(f"Model Evaluation Results:")
    print(f"  Samples evaluated: {len(predictions)}")
    print(f"  Mean Squared Error: {mse:.6f}")
    print(f"  Mean Absolute Error: {mae:.6f}")
    print(f"  Root Mean Squared Error: {np.sqrt(mse):.6f}")
    
    return {
        'predictions': predictions,
        'targets': targets,
        'inputs': inputs,
        'mse': mse,
        'mae': mae,
        'rmse': np.sqrt(mse)
    }

def plot_prediction_examples(results, n_examples=6, max_depth=15):
    """
    Plot examples of model predictions vs actual targets.
    
    Parameters:
    -----------
    results : dict
        Results from evaluate_model()
    n_examples : int
        Number of examples to plot
    max_depth : int
        Number of depth levels
    """
    predictions = results['predictions']
    targets = results['targets']
    inputs = results['inputs']
    
    depths = np.arange(max_depth)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Select random examples
    example_indices = np.random.choice(len(predictions), n_examples, replace=False)
    
    for i, idx in enumerate(example_indices):
        ax = axes[i]
        
        # Plot input, target, and prediction
        ax.plot(depths, inputs[idx], 'b-o', label='Input (t)', linewidth=2, markersize=6)
        ax.plot(depths, targets[idx], 'g-s', label='Actual (t+1)', linewidth=2, markersize=6)
        ax.plot(depths, predictions[idx], 'r--^', label='Predicted (t+1)', linewidth=2, markersize=6)
        
        ax.set_xlabel('Depth Level')
        ax.set_ylabel('Temperature (normalized)')
        ax.set_title(f'Prediction Example {i+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()  # Deeper depths at bottom
    
    plt.tight_layout()
    plt.savefig('model_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Prediction examples saved as 'model_predictions.png'")

def plot_prediction_accuracy(results, max_depth=15):
    """
    Plot prediction accuracy analysis.
    
    Parameters:
    -----------
    results : dict
        Results from evaluate_model()
    max_depth : int
        Number of depth levels
    """
    predictions = results['predictions']
    targets = results['targets']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Scatter plot: Predicted vs Actual
    ax1.scatter(targets.flatten(), predictions.flatten(), alpha=0.5, s=1)
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax1.set_xlabel('Actual Temperature')
    ax1.set_ylabel('Predicted Temperature')
    ax1.set_title('Predicted vs Actual (All Depths)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Error distribution
    errors = predictions - targets
    ax2.hist(errors.flatten(), bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Prediction Error')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Prediction Error Distribution')
    ax2.grid(True, alpha=0.3)
    
    # 3. Error by depth level
    depth_errors = []
    for depth in range(max_depth):
        depth_errors.append(np.abs(errors[:, depth]))
    
    ax3.boxplot(depth_errors, labels=[f'D{i}' for i in range(max_depth)])
    ax3.set_xlabel('Depth Level')
    ax3.set_ylabel('Absolute Error')
    ax3.set_title('Prediction Error by Depth Level')
    ax3.grid(True, alpha=0.3)
    
    # 4. Time series of predictions (first few samples)
    n_time_samples = min(50, len(predictions))
    time_indices = np.arange(n_time_samples)
    
    # Plot surface temperature (depth 0) over time
    ax4.plot(time_indices, targets[:n_time_samples, 0], 'g-', label='Actual', linewidth=2)
    ax4.plot(time_indices, predictions[:n_time_samples, 0], 'r--', label='Predicted', linewidth=2)
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Surface Temperature (normalized)')
    ax4.set_title('Surface Temperature: Actual vs Predicted')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Prediction accuracy analysis saved as 'prediction_accuracy.png'")

def denormalize_predictions(results, npz_file):
    """
    Convert normalized predictions back to original temperature scale.
    
    Parameters:
    -----------
    results : dict
        Results from evaluate_model()
    npz_file : str
        Path to the NPZ file containing scaler information
    
    Returns:
    --------
    dict with denormalized predictions and targets
    """
    # Load scaler information
    data = np.load(npz_file, allow_pickle=True)
    
    if 'scaler_min' in data and 'scaler_scale' in data:
        scaler_min = data['scaler_min']
        scaler_scale = data['scaler_scale']
        
        # Denormalize: X_original = (X_normalized / scale) + min
        predictions_orig = results['predictions'] / scaler_scale + scaler_min
        targets_orig = results['targets'] / scaler_scale + scaler_min
        inputs_orig = results['inputs'] / scaler_scale + scaler_min
        
        print("Predictions denormalized to original temperature scale")
        print(f"Temperature range - Predictions: [{predictions_orig.min():.2f}, {predictions_orig.max():.2f}] °C")
        print(f"Temperature range - Targets: [{targets_orig.min():.2f}, {targets_orig.max():.2f}] °C")
        
        return {
            'predictions': predictions_orig,
            'targets': targets_orig,
            'inputs': inputs_orig
        }
    else:
        print("No scaler information found in NPZ file")
        return results

def main(checkpoint_path, dataset_path, n_samples=100):
    """
    Main evaluation function.
    
    Parameters:
    -----------
    checkpoint_path : str
        Path to trained model checkpoint
    dataset_path : str
        Path to dataset NPZ file
    n_samples : int
        Number of samples to evaluate
    """
    print("=== Loading Model and Dataset ===")
    
    # Load trained model
    model = load_trained_model(checkpoint_path)
    print(f"Loaded model from: {checkpoint_path}")
    
    # Load dataset
    dataset = Dataset(dataset_path)
    print(f"Loaded dataset from: {dataset_path}")
    
    print("\\n=== Evaluating Model ===")
    results = evaluate_model(model, dataset, n_samples)
    
    print("\\n=== Creating Visualizations ===")
    plot_prediction_examples(results)
    plot_prediction_accuracy(results)
    
    # Try to denormalize if scaler info is available
    try:
        denorm_results = denormalize_predictions(results, dataset_path)
        print("\\n=== Plotting Denormalized Results ===")
        plot_prediction_examples(denorm_results, n_examples=6)
    except Exception as e:
        print(f"Could not denormalize results: {e}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained temperature profile model")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint (.ckpt file)")
    parser.add_argument("dataset", type=str, help="Path to dataset NPZ file")
    parser.add_argument("--n_samples", type=int, default=6, help="Number of samples to evaluate")
    
    args = parser.parse_args()
    
    results = main(args.checkpoint, args.dataset, args.n_samples)
