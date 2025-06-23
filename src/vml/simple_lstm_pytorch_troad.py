"""
TROAD prediction using an LSTM NN.
Based on tensorflow tutorial found online.
Converted tensorflow references to pytorch.
Reads observation data from OBSTABLE for 
a given station and makes prediction.

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import sqlite3
import os

def load_and_merge_data_optimized(variables, year):
    """
    Load data from SQLite databases and merge into a single dataframe.
    Each variable is in a different sqlite file. The function
    merges all of them in a single dataframe.

    Args:
    variables: List of meteorological variables to load
    year: Year of data to load

    Returns:
    DataFrame with merged data from all variables
    """
    dataframes = []
    for variable in variables:
        conn = sqlite3.connect(os.path.join(DB_PATH, f'OBSTABLE_{variable}_{year}.sqlite'))
        # Optimize SQLite performance
        conn.execute('PRAGMA synchronous = OFF')
        conn.execute('PRAGMA journal_mode = MEMORY')
        query = f"SELECT valid_dttm, SID, lat, lon, {variable} FROM SYNOP"
        for chunk in pd.read_sql_query(query, conn, chunksize=10000):
            dataframes.append(chunk)
        conn.close()

    # Merge all dataframes
    full_df = pd.concat(dataframes, ignore_index=True)
    merged_df = full_df.groupby(['valid_dttm', 'SID', 'lat', 'lon']).first().reset_index()
    return merged_df
# Load the dataset
#df_orig = pd.read_csv('data/weather_data_lo_barnechea_hourly.csv')
# the data is in degC so no need to convert it
DB_PATH = "/media/cap/extra_work/road_model/OBSTABLE"
DB_PATH = "/data/projects/glatmodel/verification/harp/OBSTABLE"
variables = ['TROAD', 'T2m', 'Td2m'] #, 'D10m', 'S10m']
year = 2023
print(f"Loading the data for {year}")
df = load_and_merge_data_optimized(variables, year)
df["date_time"] = pd.to_datetime(df["valid_dttm"],unit="s")

#Select one of the stations
sel_station = 503100
df=df[df.SID == sel_station]
df.drop('valid_dttm', axis=1, inplace=True)


# Sort the DataFrame by date_time
df.sort_values(by='date_time', inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"Selected station {sel_station}")
print(df.head(10))

# Selecting relevant columns
#numeric_cols = ['tempC', 'humidity', 'pressure', 'precipMM', 'uvIndex', 'windspeedKmph', 'winddirDegree']
numeric_cols = variables
data_selected = df[numeric_cols]
data_selected = data_selected.fillna(data_selected.mean())

# Splitting the dataset into training, validation, and testing sets
SPLIT = 0.8
train_size = int(len(data_selected) * SPLIT)
val_size = int(len(data_selected) * (1-SPLIT)//2)
test_size = len(data_selected) - train_size - val_size

print(f"Details of the {SPLIT*100} % split")
print(f"Training: {train_size} h ({train_size/24} days)")

data_train = data_selected[:train_size]
data_val = data_selected[train_size:train_size + val_size]
data_test = data_selected[-test_size:]

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_train_normalized = scaler.fit_transform(data_train)
data_val_normalized = scaler.transform(data_val)
data_test_normalized = scaler.transform(data_test)


# Save the scaler parameters to a text file
# this is for converting temperature from normalized units
with open('scaler_params.txt', 'w') as file:
    file.write('scale:' + ','.join(map(str, scaler.scale_)) + '\n')
    file.write('min:' + ','.join(map(str, scaler.min_)) + '\n')
    file.write('data_min:' + ','.join(map(str, scaler.data_min_)) + '\n')
    file.write('data_max:' + ','.join(map(str, scaler.data_max_)) + '\n')
    file.write('data_range:' + ','.join(map(str, scaler.data_range_)) + '\n')





# Create sequences
def create_sequences(input_data, n_steps, fut_hours, out_feat_index):
  X, y = [], []
  for i in range(len(input_data) - n_steps - fut_hours):
      end_ix = i + n_steps
      out_end_ix = end_ix + fut_hours
      if out_end_ix > len(input_data):
          break
      seq_x, seq_y = input_data[i:end_ix, :], input_data[out_end_ix - 1, out_feat_index]
      X.append(seq_x)
      y.append(seq_y)
  return np.array(X), np.array(y)

# Creating sequences
#output_feature = 'tempC'
output_feature = 'TROAD'
out_feat_index = numeric_cols.index(output_feature)
fut_hours = 1 #prediction one hour ahead
n_steps = 168 #168 = 1 week or 24*7, this represents the hour of history used for each prediction in the LSTM
n_steps = 24 # a day
n_inputs = len(data_selected.columns)

X_train, y_train = create_sequences(data_train_normalized, n_steps, fut_hours, out_feat_index)
X_val, y_val = create_sequences(data_val_normalized, n_steps, fut_hours, out_feat_index)
X_test, y_test = create_sequences(data_test_normalized, n_steps, fut_hours, out_feat_index)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_val = torch.FloatTensor(X_val)
y_val = torch.FloatTensor(y_val)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

# Create DataLoaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

BATCH = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False)

# Define the LSTM model
class LSTMModel(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
      super(LSTMModel, self).__init__()
      self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
      self.fc = nn.Linear(hidden_size, output_size)
      
  def forward(self, x):
      lstm_out, _ = self.lstm(x)
      return self.fc(lstm_out[:, -1, :])

# Instantiate the model
input_size = X_train.shape[2]
hidden_size = 128
output_size = 1
model = LSTMModel(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
print("Training the model")
# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
  train_losses = []
  val_losses = []
  best_val_loss = float('inf')
  patience = 5
  counter = 0
  
  for epoch in range(epochs):
      model.train()
      train_loss = 0.0
      for inputs, targets in train_loader:
          optimizer.zero_grad()
          outputs = model(inputs)
          loss = criterion(outputs.squeeze(), targets)
          loss.backward()
          optimizer.step()
          train_loss += loss.item()
      
      train_loss /= len(train_loader)
      train_losses.append(train_loss)
      
      model.eval()
      val_loss = 0.0
      with torch.no_grad():
          for inputs, targets in val_loader:
              outputs = model(inputs)
              loss = criterion(outputs.squeeze(), targets)
              val_loss += loss.item()
      
      val_loss /= len(val_loader)
      val_losses.append(val_loss)
      
      print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
      
      if val_loss < best_val_loss:
          best_val_loss = val_loss
          counter = 0
          torch.save(model.state_dict(), 'best_model.pth')
          # Save both the model and scaler
          torch.save({
              'model_state_dict': model.state_dict(),
              'scaler': scaler,
              'numeric_cols': numeric_cols,
              'output_feature': output_feature
          }, 'model_checkpoint.pth')
      else:
          counter += 1
          if counter >= patience:
              print("Early stopping")
              break
  
  return train_losses, val_losses

# Train the model
EPOCHS = 20
train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS)
print("Created LSTM model")
# print the model summary
sequence_length = X_train.shape[1]  # This is the sequence length
from torchinfo import summary
summary(model, input_size=(BATCH, sequence_length, input_size))

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Model Training History')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()



# Evaluate the model on test data
model.load_state_dict(torch.load('best_model.pth'))
#model.load_state_dict(torch.load('model_checkpoint.pth'))
model.eval()
test_loss = 0.0
predictions = []
actuals = []

with torch.no_grad():
  for inputs, targets in test_loader:
      outputs = model(inputs)
      loss = criterion(outputs.squeeze(), targets)
      test_loss += loss.item()
      predictions.extend(outputs.squeeze().tolist())
      actuals.extend(targets.tolist())

test_loss /= len(test_loader)
print(f'Test RMSE: {np.sqrt(test_loss):.3f}')



# Load the scaler parameters from the text file
loaded_scaler_params = {}
with open('scaler_params.txt', 'r') as file:
    for line in file:
        key, value = line.strip().split(':')
        loaded_scaler_params[key] = np.array([float(i) for i in value.split(',')])

# Create a new scaler instance and set its parameters
inference_scaler = MinMaxScaler()
inference_scaler.scale_ = loaded_scaler_params['scale']
inference_scaler.min_ = loaded_scaler_params['min']
inference_scaler.data_min_ = loaded_scaler_params['data_min']
inference_scaler.data_max_ = loaded_scaler_params['data_max']
inference_scaler.data_range_ = loaded_scaler_params['data_range']

# After making predictions, we need to map the timestamps
# Get the corresponding dates for the test set
test_dates = df['date_time'].iloc[-len(actuals):]

# Convert predictions and actuals back to original scale
# We're only predicting TROAD (index 0), so we need to create arrays with the right shape
def denormalize_single_feature(normalized_values, scaler, feature_index):
    """Convert normalized values back to original scale for a single feature"""
    # Create a dummy array with the same number of features as the original data
    dummy_array = np.zeros((len(normalized_values), len(scaler.scale_)))
    # Put the normalized values in the correct feature column
    dummy_array[:, feature_index] = normalized_values
    # Inverse transform
    denormalized = scaler.inverse_transform(dummy_array)
    # Return only the feature we care about
    return denormalized[:, feature_index]

# Convert back to original units
actuals_original = denormalize_single_feature(actuals, inference_scaler, out_feat_index)
predictions_original = denormalize_single_feature(predictions, inference_scaler, out_feat_index)

# Plot predictions vs actual with dates (in original units)
plt.figure(figsize=(12, 6))
plt.plot(test_dates, actuals_original, color='red', label='Real data')
plt.plot(test_dates, predictions_original, color='blue', label='Predicted data', alpha=0.6)
plt.title('Road Temperature Prediction')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')  # Now showing original units
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig("troad_prediction.png")
plt.show()

# For the subset plot (in original units)
subset_size = 2000
plt.figure(figsize=(12, 6))
plt.plot(test_dates[-subset_size:], actuals_original[-subset_size:],
         color='red', label='Real data')
plt.plot(test_dates[-subset_size:], predictions_original[-subset_size:],
         color='blue', label='Predicted data', alpha=0.6)
plt.title('Road Temperature Prediction (Subset)')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')  # Now showing original units
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig("troad_prediction_subset.png")
plt.show()

# Update RMSE calculation to show error in original units
test_rmse_original = np.sqrt(np.mean((actuals_original - predictions_original)**2))
print(f'Test RMSE in original units: {test_rmse_original:.3f} °C')


## Plot predictions vs actual with dates
#plt.figure(figsize=(12, 6))
#plt.plot(test_dates, actuals, color='red', label='Real data')
#plt.plot(test_dates, predictions, color='blue', label='Predicted data', alpha=0.6)
#plt.title('Road Temperature Prediction')
#plt.xlabel('Date')
#plt.ylabel('Temperature')
#plt.xticks(rotation=45)
#plt.grid(True, linestyle='--', alpha=0.7)
#plt.legend()
#plt.tight_layout()  # Prevents label cutoff
#plt.savefig("troad_prediction.png")
#plt.show()
#
## For the subset plot
#subset_size = 2000
#plt.figure(figsize=(12, 6))
#plt.plot(test_dates[-subset_size:], actuals[-subset_size:],
#         color='red', label='Real data')
#plt.plot(test_dates[-subset_size:], predictions[-subset_size:],
#         color='blue', label='Predicted data', alpha=0.6)
#plt.title('Road Temperature Prediction (Subset)')
#plt.xlabel('Date')
#plt.ylabel('Temperature')
#plt.xticks(rotation=45)
#plt.grid(True, linestyle='--', alpha=0.7)
#plt.legend()
#plt.tight_layout()
#plt.savefig("troad_prediction_subset.png")
#plt.show()
#
## Plot predictions vs actual
#plt.figure(figsize=(10, 6))
#plt.plot(actuals, color='red', label='Real data')
#plt.plot(predictions, color='blue', label='Predicted data', alpha=0.6)
#plt.title('Data Prediction')
#plt.xlabel('Time')
#plt.ylabel('Value')
#plt.legend()
#plt.show()
#
## Plot a subset of the predictions
#subset_size = 2000
#plt.figure(figsize=(10, 6))
#plt.plot(actuals[-subset_size:], color='red', label='Real data')
#plt.plot(predictions[-subset_size:], color='blue', label='Predicted data', alpha=0.6)
#plt.title('Data Prediction (Subset)')
#plt.xlabel('Time')
#plt.ylabel('Value')
#plt.legend()
#plt.show()
#
