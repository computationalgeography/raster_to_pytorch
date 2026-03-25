import os
import rasterio
import numpy as np
import torch
from datetime import datetime, timedelta
from lib.lstm_definition import LSTM

# ---------------------------
# Settings
# ---------------------------
data_folder = "data"  # folder containing subfolders for each variable
model_folder = "models"  # folder containing the saved LSTM model
input_vars = ["hurs", "pr", "rsds", "sfcWind", "tas", "tasmax", "tasmin"]
target_var = "SWE"
partition_size = 64      # partition size
seq_len = 1              # sequence length

# Date range
warmup_date = datetime(1979, 10, 1)
start_date = datetime(1980, 10, 1)
end_date   = datetime(2020, 9, 30)

# Load the LSTM model
num_features = len(input_vars) + 1  # input vars + previous SWE
hidden_size = 5  # example
lstm_model = LSTM(input_size=num_features, hidden_size=hidden_size)
lstm_model.load_state_dict(
    torch.load(os.path.join(model_folder, "lstm_model.pt"), weights_only=True)
    )
lstm_model.eval()

# ---------------------------
# Loop over dates
# ---------------------------
current_date = warmup_date
hidden_states = [None] * (H//partition_size * W//partition_size)  # one hidden state per partition
swe_prev = np.zeros((H, W), dtype=np.float32) # initialize SWE at 0 for warmup

while current_date <= end_date:
    day_str = current_date.strftime("%d_%m_%Y")

    # Read input variables for this day
    partitions = []
    for var in input_vars:
        var_file = os.path.join(data_folder, var, f"CHELSA_{var}_{day_str}_V.2.1.tif")
        with rasterio.open(var_file) as src:
            partitions.append(src.read(1))  # shape H x W

    # Stack variables: shape [num_vars, H, W]
    stacked = np.stack(partitions, axis=0)
    H, W = stacked.shape[1], stacked.shape[2]

    # Add previous SWE to input stack: shape [num_vars+1, H, W]
    stacked = np.concatenate([swe_prev[np.newaxis, :, :], stacked], axis=0)

    # ---------------------------
    # Loop over partitions
    # ---------------------------
    for i in range(0, H, partition_size):
        for j in range(0, W, partition_size):
            block = stacked[:, i:i+partition_size, j:j+partition_size]
            ph, pw = block.shape[1], block.shape[2]

            # Flatten pixels: [num_pixels, num_features]
            block_flat = block.reshape(num_features, -1).T  # [num_pixels, num_features]

            # LSTM input: [batch_size=num_pixels, seq_len=1, num_features]
            batch_size = block_flat.shape[0]
            block_tensor = torch.tensor(block_flat, dtype=torch.float32).reshape(batch_size, seq_len, num_features)

            # Gather previous hidden state, run the LSTM model, and update hidden state
            with torch.no_grad():
                prev_hidden = hidden_states[(i//partition_size) * (W//partition_size) + (j//partition_size)]
                pred, hidden_state = lstm_model(block_tensor, min_val=0.0, hidden_state=prev_hidden)
                hidden_states[(i//partition_size) * (W//partition_size) + (j//partition_size)] = hidden_state

            # Reshape back to partition map
            pred_map = pred.cpu().numpy().reshape(ph, pw)
            swe_prev[i:i+ph, j:j+pw] = pred_map

    # ---------------------------
    # Save predicted SWE for current day
    # ---------------------------
    if current_date >= start_date:
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
            dst.write(swe_prev, 1)

    print(f"Day {day_str} predicted and saved.")

    # Next day
    current_date += timedelta(days=1)