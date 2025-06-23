import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

def analyze_model_performance(model, dataset, batch_size=32, num_samples_to_show=5):
    """
    Comprehensive analysis of model performance with training and prediction plots
    
    Args:
        model: Trained PyTorch Lightning model
        dataset: Your Dataset instance
        batch_size: Batch size for evaluation
        num_samples_to_show: Number of sample profiles to display
    """
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Collect predictions and targets
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            predictions = model.forward(batch)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(batch["target"].cpu().numpy())
    
    # Concatenate all batches
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Create comprehensive plots
    create_analysis_plots(model.losses, targets, predictions, num_samples_to_show)
    
    return predictions, targets

def create_analysis_plots(training_losses, actual_data, predicted_data, num_samples=5):
    """
    Create comprehensive plots for training analysis and prediction comparison
    """
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Training Loss Curve
    plt.subplot(2, 4, 1)
    plt.plot(training_losses, 'b-', linewidth=2, alpha=0.8)
    plt.title('Training Loss Curve', fontsize=12, fontweight='bold')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss (MSE)')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # 2. Prediction vs Actual Scatter Plot
    plt.subplot(2, 4, 2)
    plt.scatter(actual_data.flatten(), predicted_data.flatten(), alpha=0.5, s=15)
    
    # Perfect prediction line
    min_val = min(actual_data.min(), predicted_data.min())
    max_val = max(actual_data.max(), predicted_data.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.title('Predicted vs Actual', fontsize=12, fontweight='bold')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Calculate and display R²
    correlation_matrix = np.corrcoef(actual_data.flatten(), predicted_data.flatten())
    r_squared = correlation_matrix[0, 1] ** 2
    plt.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. Sample Temperature Profiles
    plt.subplot(2, 4, 3)
    depth_points = np.arange(actual_data.shape[1])
    
    for i in range(min(num_samples, actual_data.shape[0])):
        plt.plot(depth_points, actual_data[i], 'o-', label=f'Actual {i+1}', 
                linewidth=2, markersize=4, alpha=0.8)
        plt.plot(depth_points, predicted_data[i], 's--', label=f'Pred {i+1}', 
                linewidth=2, markersize=3, alpha=0.7)
    
    plt.title('Temperature Profiles', fontsize=12, fontweight='bold')
    plt.xlabel('Depth Index')
    plt.ylabel('Temperature')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # 4. Residuals Plot
    plt.subplot(2, 4, 4)
    residuals = predicted_data - actual_data
    plt.scatter(actual_data.flatten(), residuals.flatten(), alpha=0.5, s=15)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.title('Residuals Plot', fontsize=12, fontweight='bold')
    plt.xlabel('Actual Values')
    plt.ylabel('Residuals')
    plt.grid(True, alpha=0.3)
    
    # 5. Error Distribution
    plt.subplot(2, 4, 5)
    plt.hist(residuals.flatten(), bins=30, alpha=0.7, edgecolor='black')
    plt.title('Error Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    mean_error = np.mean(residuals)
    plt.axvline(mean_error, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_error:.3f}')
    plt.legend()
    
    # 6. MAE by Depth
    plt.subplot(2, 4, 6)
    mae_by_depth = np.mean(np.abs(residuals), axis=0)
    plt.bar(depth_points, mae_by_depth, alpha=0.7, edgecolor='black')
    plt.title('MAE by Depth', fontsize=12, fontweight='bold')
    plt.xlabel('Depth Index')
    plt.ylabel('Mean Absolute Error')
    plt.grid(True, alpha=0.3)
    
    # 7. Loss Convergence (zoomed)
    plt.subplot(2, 4, 7)
    if len(training_losses) > 20:
        plt.plot(training_losses[-20:], 'g-', linewidth=2, alpha=0.8)
        plt.title('Loss Convergence (Last 20)', fontsize=12, fontweight='bold')
    else:
        plt.plot(training_losses, 'g-', linewidth=2, alpha=0.8)
        plt.title('Loss Convergence', fontsize=12, fontweight='bold')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    # 8. Performance Metrics Summary
    plt.subplot(2, 4, 8)
    plt.axis('off')
    
    # Calculate metrics
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))
    mbe = np.mean(residuals)
    std_error = np.std(residuals)
    
    metrics_text = f"""Performance Metrics:
    
R² Score: {r_squared:.4f}
MAE: {mae:.4f}
RMSE: {rmse:.4f}
Mean Bias: {mbe:.4f}
Std Error: {std_error:.4f}

Final Loss: {training_losses[-1]:.4f}
Total Samples: {actual_data.shape[0]}
Depth Points: {actual_data.shape[1]}"""
    
    plt.text(0.1, 0.9, metrics_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed summary
    print("\\n" + "="*50)
    print("MODEL PERFORMANCE ANALYSIS")
    print("="*50)
    print(f"R² Score: {r_squared:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Root Mean Square Error: {rmse:.4f}")
    print(f"Mean Bias Error: {mbe:.4f}")
    print(f"Standard Deviation of Errors: {std_error:.4f}")
    print(f"Final Training Loss: {training_losses[-1]:.4f}")
    print(f"Total Training Steps: {len(training_losses)}")
    print("="*50)

# Example usage function to add to your train.py
def enhanced_main(args):
    """
    Enhanced main function with comprehensive analysis
    """
    depth = 10
    model = Model(depth)
    dataset = Dataset(args.path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Train the model
    trainer = pl.Trainer(max_epochs=50)
    trainer.fit(model, dataloader)
    
    # Analyze performance
    print("\\nAnalyzing model performance...")
    predictions, targets = analyze_model_performance(model, dataset)
    
    return model, predictions, targets

# Simple function to just plot training loss
def plot_training_loss(model, save_path=None):
    """
    Simple function to plot just the training loss
    """
    plt.figure(figsize=(10, 6))
    plt.plot(model.losses, 'b-', linewidth=2)
    plt.title('Training Loss Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss (MSE)')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Final loss: {model.losses[-1]:.6f}")
    print(f"Initial loss: {model.losses[0]:.6f}")
    print(f"Loss reduction: {((model.losses[0] - model.losses[-1]) / model.losses[0] * 100):.2f}%")

print("Created model_analysis.py with comprehensive plotting functions!")
print("\nTo use in your code, add this to your train.py:")
print("""
from model_analysis import analyze_model_performance, plot_training_loss

# After training your model:
analyze_model_performance(model, dataset)

# Or for just the training loss:
plot_training_loss(model)
""")
