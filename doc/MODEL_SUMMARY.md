# ML Vejvejr - Model Summary

## Overview

This repository contains machine learning models for predicting road temperature profiles using time-series data from road weather stations. The project focuses on predicting temperature at different depths beneath the road surface, which is crucial for road maintenance and winter road management.

## Project Purpose

The models predict temperature profiles at time `t+1` based on observations at time `t`. The data consists of temperature measurements at 15 different depth levels beneath the road surface, collected from Danish road weather stations (`fild8` files in the `glatfoere` model output).

## Models

### 1. MLP-based Temperature Profile Model (Primary)

**Location:** `src/vml/model.py`

#### Architecture

The primary model is a Multi-Layer Perceptron (MLP) that predicts the complete temperature profile for the next hour:

```python
Model(TempProfileModel)
└── MLP
    ├── Linear(dim_input=15, 64)
    ├── ReLU
    ├── Linear(64, 32)
    ├── ReLU
    └── Linear(32, dim_output=15)
```

**Key Components:**

- **Input:** Temperature profile at time `t` (15 depth levels)
- **Output:** Temperature profile at time `t+1` (15 depth levels)
- **Hidden layers:** 64 → 32 neurons with ReLU activation
- **Loss function:** Mean Squared Error (MSE)
- **Optimizer:** Adam (learning rate: 0.001)
- **Framework:** PyTorch Lightning

#### Base Class: TempProfileModel

Located at `src/vml/model.py:19-56`, this is a PyTorch Lightning module providing:

- **Training loop management** with automatic loss tracking per epoch
- **Validation loop management** with validation loss tracking
- **Loss history tracking:**
  - `self.losses`: List of per-epoch training losses
  - `self.val_losses`: List of per-epoch validation losses
- **Configurable optimizer:** Adam with lr=0.001

The base class implements:
- `training_step()`: Computes MSE loss between predicted and target profiles
- `validation_step()`: Computes validation MSE loss
- `on_train_epoch_end()`: Aggregates training losses per epoch
- `on_validation_epoch_end()`: Aggregates validation losses per epoch
- `configure_optimizers()`: Returns Adam optimizer

### 2. LSTM Model for Road Temperature (Alternative Implementation)

**Location:** `src/vml/simple_lstm_pytorch_troad.py`

This was a quick-and-dirty attempt to implement an LSTM-based model that uses multiple meteorological variables:
- T2m  (2m temperature)
- Td2m (2m dew point temperature)
to predict TROAD (road surface temperature).

#### Architecture

```python
LSTMModel
├── LSTM(input_size=n_features, hidden_size=128, batch_first=True)
└── Fully Connected(128, 1)
```

**Key Features:**

- **Input variables:** TROAD (road temperature), T2m (2m air temperature), Td2m (2m dewpoint)
- **Sequence length:** 24 hours (configurable, can be up to 168 hours/1 week)
- **Prediction horizon:** 1 hour ahead
- **Hidden size:** 128 neurons
- **Batch size:** 32
- **Data normalization:** MinMaxScaler to [0, 1] range
- **Early stopping:** Patience of 5 epochs
- **Checkpoint saving:** Saves best model based on validation loss

#### Training Configuration

- **Data split:** 80% training, 10% validation, 10% testing
- **Epochs:** 20 (with early stopping)
- **Criterion:** MSE Loss
- **Optimizer:** Adam (default PyTorch learning rate)

#### Data Source

- Loads data from SQLite databases (`OBSTABLE_{variable}_{year}.sqlite`)
- Optimizes performance with SQLite pragmas (synchronous=OFF, journal_mode=MEMORY)
- Filters data for specific stations (e.g., station 503100)

## Data Pipeline

### Input Data Format

**Source:** Parquet files containing road temperature profiles

Example data structure from `/data/projects/glatmodel/obs/fild8/road_profiles_daily`:

```
timestamp           station_id  depth_0  depth_1  depth_2  ...  depth_14
2022-08-22 00:00:00 0-100000-0  290.15   290.269  290.454  ...  286.99
```

- **Format:** One parquet file per day
- **Temperature units:** Kelvin (converted to Celsius in processing)
- **Temporal resolution:** Hourly (24 profiles per station per day)
- **Depth levels:** 15 measurements at different depths

### Data Processing Pipeline

**Script:** `scripts/generate_data_from_profiles.py`

#### Key Functions

1. **`load_parquet_files_timeseries()`** (`line 14-94`)
   - Loads and merges multiple parquet files
   - Filters data for specific station IDs
   - Converts temperatures from Kelvin to Celsius
   - Sorts by timestamp to create temporal sequence
   - Handles missing depth columns by padding with last available depth

2. **`create_timeseries_dataset()`** (`line 191-289`)
   - Creates input-target pairs: `(profile_t, profile_t+1)`
   - Normalizes data using MinMaxScaler to [0, 1] range
   - Saves to compressed NPZ format with:
     - `input`: Normalized profiles at time t
     - `target`: Normalized profiles at time t+1
     - `scaler_min`, `scaler_scale`: For denormalization
     - `time_step_hours`: Time offset (default: 1 hour)
   - Generates visualization plots for quality control

3. **Visualization functions:**
   - `plot_training_data_example()`: Shows input-target profile pairs
   - `plot_time_series_overview()`: Time series and heatmap visualizations

### Dataset Class

**Location:** `src/vml/dataset.py`

```python
class Dataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        return {
            "input": torch.tensor(self.input_data[idx]),   # Profile at time t
            "target": torch.tensor(self.target_data[idx])  # Profile at time t+1
        }
```

- Loads NPZ files containing preprocessed time-series data
- Returns PyTorch tensors for input and target profiles
- Supports batch loading via DataLoader

## Training

### Training Script

**Location:** `scripts/train.py`

**Usage:**
```bash
python scripts/train.py <train_path> <val_path> <epochs>
```

**Configuration:**
- Batch size: 32
- Depth levels: 15
- Shuffle: True for training, False for validation
- Framework: PyTorch Lightning Trainer

**Output:**
- Model checkpoints in `lightning_logs/`
- Training/validation loss plot: `train_valid_losses.png`
- Loss curves plotted on log scale

### Example Workflow

1. **Generate training data:**
```bash
python scripts/generate_data_from_profiles.py \
    2021,2022,2023 \  # years
    0-100000-0 \       # station ID
    train \            # data type
    data/ \            # output path
    /path/to/profiles  # input path
```

2. **Train model:**
```bash
python scripts/train.py \
    data/road_temp_training_2021-2023.npz \
    data/road_temp_training_2024.npz \
    50
```

3. **Evaluate model:**
```bash
python src/vml/evaluate_model.py \
    lightning_logs/version_X/checkpoints/epoch=Y.ckpt \
    data/road_temp_training_2024.npz \
    --n_samples 100
```

## Model Evaluation

### Evaluation Script

**Location:** `src/vml/evaluate_model.py`

#### Metrics Computed

- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**

#### Denormalization

The `denormalize_predictions()` function (line 210-248) converts normalized predictions back to original Celsius scale using scaler parameters saved in the NPZ file:

```python
X_original = (X_normalized / scale) + min
```

## Data Characteristics

### Temperature Profile Structure

The 15 depth levels represent temperature measurements from the road surface down to deeper layers:
- `depth_0`: Road surface temperature
- `depth_1` to `depth_13`: Subsurface temperatures at increasing depths
- `depth_14`: Deepest measurement level

Temperature generally:
- Varies most at the surface (depth_0)
- Shows decreasing variation with depth
- Exhibits thermal inertia (slower changes at deeper levels)

### Temporal Characteristics

- **Diurnal cycle:** Strong daily patterns in surface temperature
- **Thermal lag:** Deeper layers lag behind surface temperature changes
- **Seasonal variation:** Long-term trends in temperature ranges

## Model Performance Considerations

### Strengths of MLP Approach

1. **Simplicity:** Fast training and inference
2. **Direct mapping:** Learns direct profile-to-profile transformation
3. **No sequence requirement:** Can predict from single time step
4. **Captures spatial patterns:** Learns relationships between depth levels

### Limitations of MLP Approach

1. **Limited temporal context:** Uses only previous hour
2. **No long-term dependencies:** Cannot capture multi-day weather patterns
3. **Stateless:** Doesn't maintain memory of past predictions

### Strengths of LSTM Approach

1. **Temporal memory:** Uses 24-168 hours of history
2. **Multiple features:** Incorporates air temperature and dewpoint
3. **Sequential learning:** Better for capturing weather trends
4. **Long-term dependencies:** Can learn weekly patterns

### Potential Improvements

1. **Hybrid architecture:** Combine LSTM temporal features with spatial profile modeling
2. **Attention mechanisms:** Focus on relevant time steps
3. **Weather forecasts:** Include predicted meteorological variables
4. **Station metadata:** Incorporate location, elevation, surface type
5. **Multi-task learning:** Predict multiple horizons simultaneously (1h, 3h, 6h, 12h)

## File Structure

```
ml_vejvejr/
├── src/vml/
│   ├── model.py                          # MLP model definition
│   ├── dataset.py                        # PyTorch Dataset class
│   ├── evaluate_model.py                 # Evaluation and visualization
│   ├── simple_lstm_pytorch_troad.py      # LSTM alternative model
│   └── plot_daily_profiles.py            # Profile visualization utilities
├── scripts/
│   ├── train.py                          # Training script
│   ├── generate_data_from_profiles.py    # Data preprocessing pipeline
│   └── explore_h5_data.py               # Data exploration tools
├── shell_scripts/
│   ├── run_model.sh                      # Training automation
│   └── run_eval.sh                       # Evaluation automation
└── data/                                 # Generated datasets (NPZ files)
```

## Dependencies

Main libraries (from `requirements.txt` analysis):
- **PyTorch:** Neural network framework
- **PyTorch Lightning:** Training orchestration
- **NumPy:** Numerical operations
- **Pandas:** Data manipulation
- **scikit-learn:** Data preprocessing (MinMaxScaler, metrics)
- **Matplotlib:** Visualization

## Research Context

The temperature profiles are used to:

1. **Predict road surface conditions:** Ice formation, frost depth
2. **Winter road maintenance:** Optimize salting and plowing operations
3. **Road safety:** Anticipate dangerous conditions
4. **Resource optimization:** Efficient use of de-icing materials

The models predict temperature evolution which is critical for understanding:
- Surface freezing conditions
- Heat storage in the road structure
- Effectiveness of preventive treatments
- Timing of maintenance interventions

## Future Development Areas

Based on the codebase structure:

1. **Multi-station modeling:** Currently trained per station; could leverage spatial relationships
2. **Ensemble methods:** Combine MLP and LSTM predictions
3. **Uncertainty quantification:** Probabilistic predictions for risk assessment
4. **Real-time deployment:** Integration with operational forecasting systems
5. **Transfer learning:** Pre-train on multiple stations, fine-tune for specific locations
6. **Feature engineering:** Road material properties, traffic density, shading
