import os
import pandas as pd
import rasterio
import numpy as np
import torch
from datetime import datetime, timedelta
from lib.lstm_definition import LSTM
from lib.data_scaler import DataScaler

# ---------------------------
# Settings
# ---------------------------
data_folder = "data"  # folder containing subfolders for each variable
model_folder = "models" # folder containing the trained LSTM model and scalers
dynamic_vars = ["pr", "rsds", "tas"]
static_vars = ["elevation_mn", "slope_mn", "eastness_mn", "northness_mn", "tpi_mn"]
target_var = "swe"
partition_size = 64      # partition size
H, W = 10080, 43200      # dimensions of the input maps

# Date range
warmup_date = datetime(1979, 10, 1)
start_date = datetime(1980, 10, 1)
end_date   = datetime(2020, 9, 30)

# Load the LSTM model
num_features = len(dynamic_vars) + len(static_vars) + 1  # dynamic vars + static vars + previous SWE
hidden_size = 5  # example
lstm_model = LSTM(input_size=num_features, hidden_size=hidden_size)
lstm_model.load_state_dict(
    torch.load(os.path.join(model_folder, "lstm_model.pt"), weights_only=True)
    )
lstm_model.eval()

# Load data scalers
scalers = DataScaler()
scalers.load_transformers(model_folder)
trans_zero_swe = scalers.get_transformed_zero()
target_transformer = scalers.get_transformers().get('target')

# ---------------------------
# Loop over dates
# ---------------------------
current_date = warmup_date
hidden_states = [None] * (H//partition_size * W//partition_size)  # one hidden state per partition
swe_prev = np.zeros((H, W), dtype=np.float32) # initialize SWE at 0 for warmup

while current_date <= end_date:
    day_str = current_date.strftime("%d_%m_%Y")

    # Read input variables for this day
    variable_maps = []
    for var in dynamic_vars:
        var_file = os.path.join(data_folder, var, f"CHELSA_{var}_{day_str}_V.2.1.tif")
        with rasterio.open(var_file) as src:
            var_map = src.read(1)  # shape H x W
            if var_map.shape != (H, W):
                raise ValueError(f"Variable {var} on {day_str} has shape {var_map.shape}, expected {(H, W)}")
            variable_maps.append(var_map)

    # Stack variables: shape [num_vars, H, W]
    dynamic_stacked = np.stack(variable_maps, axis=0)

    # Add previous SWE to input stack: shape [num_vars+1, H, W]
    dynamic_stacked = np.concatenate([swe_prev[np.newaxis, :, :], dynamic_stacked], axis=0)

    # ---------------------------
    # Loop over partitions
    # ---------------------------
    for i in range(0, H, partition_size):
        for j in range(0, W, partition_size):
            dynamic_partition = dynamic_stacked[:, i:i+partition_size, j:j+partition_size]
            ph, pw = dynamic_partition.shape[1], dynamic_partition.shape[2]

            # Flatten pixels: [num_pixels, num_features]
            dynamic_partition_flat = dynamic_partition.reshape(len(dynamic_vars) + 1, -1).T

            # Transform the dynamic variables using the scalers
            dynamic_partition_df = pd.DataFrame(dynamic_partition_flat, columns=['prev_swe'] + dynamic_vars)
            dynamic_partition_tensor = scalers.transform_dataframe(dynamic_partition_df, var_group='dynamic')

            # LSTM input: [batch_size=num_pixels, seq_len=1, num_features]
            batch_size = dynamic_partition_tensor.shape[0]
            lstm_inputs = dynamic_partition_tensor.reshape(batch_size, 1, len(dynamic_vars) + 1)

            # Gather previous hidden state, run the LSTM model, and update hidden state
            with torch.no_grad():
                prev_hidden = hidden_states[(i//partition_size) * (W//partition_size) + (j//partition_size)]
                pred, hidden_state = lstm_model(lstm_inputs, min_val=trans_zero_swe, hidden_state=prev_hidden)
                hidden_states[(i//partition_size) * (W//partition_size) + (j//partition_size)] = hidden_state

            # Reshape back to partition map
            pred_map = pred.cpu().numpy().reshape(ph, pw)
            swe_prev[i:i+ph, j:j+pw] = pred_map

    # ---------------------------
    # Save predicted SWE for current day
    # ---------------------------
    if current_date >= start_date:
        
        # Inverse transform the predictions and snap small values to zero
        inv_trans_preds = target_transformer.inverse_transform(pd.DataFrame(swe_prev.flatten(), columns=['swe']))
        inv_trans_preds = np.where(inv_trans_preds < 0.1, 0, inv_trans_preds).reshape(H, W).astype(np.float32)

        # Save as GeoTIFF
        out_file = os.path.join(data_folder, target_var, f"SWE_pred_{day_str}.tif")
        with rasterio.open(
            out_file,
            'w',
            driver='GTiff',
            height=H,
            width=W,
            count=1,
            dtype=np.float32,
            crs=src.crs,
            transform=src.transform
        ) as dst:
            dst.write(inv_trans_preds, 1)

    print(f"Day {day_str} predicted and saved.")

    # Next day
    current_date += timedelta(days=1)