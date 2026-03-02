import os
import rasterio
import numpy as np
import torch
from datetime import datetime, timedelta

# ---------------------------
# Settings
# ---------------------------
data_folder = "data"  # folder containing subfolders for each variable
input_vars = ["hurs", "pr", "rsds", "sfcWind", "tas", "tasmax", "tasmin"]
target_var = "SWE"
partition_size = 64      # partition size
seq_len = 1              # sequence length
device = "cuda" if torch.cuda.is_available() else "cpu"

# Date range
start_date = datetime(1981, 4, 19)
end_date   = datetime(2020, 4, 21)  # example 3 days

# Initial SWE map for the first day
swe_start_file = os.path.join(data_folder, target_var, "SWE_start.tif")
with rasterio.open(swe_start_file) as src:
    swe_prev = src.read(1)  # initial SWE

# Dummy LSTM model (replace with trained model)
num_features = len(input_vars) + 1  # input vars + previous SWE
hidden_size = 5  # example
lstm = torch.nn.LSTM(input_size=num_features, hidden_size=hidden_size, batch_first=True).to(device)
linear = torch.nn.Linear(hidden_size, 1).to(device)  # map hidden to SWE

# ---------------------------
# Loop over dates
# ---------------------------
current_date = start_date
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
    stacked = np.concatenate([stacked, swe_prev[np.newaxis, :, :]], axis=0)

    # Prepare output SWE map
    swe_pred_full = np.zeros((H, W), dtype=np.float32)

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
            block_tensor = torch.tensor(block_flat, dtype=torch.float32).reshape(batch_size, seq_len, num_features).to(device)

            # Run LSTM
            with torch.no_grad():
                output, _ = lstm(block_tensor)  # [batch_size, seq_len, hidden_size]
                pred = linear(output[:, 0, :])  # [batch_size, 1]

            # Reshape back to partition map
            pred_map = pred.cpu().numpy().reshape(ph, pw)
            swe_pred_full[i:i+ph, j:j+pw] = pred_map

    # ---------------------------
    # Save predicted SWE for current day
    # ---------------------------
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
        dst.write(swe_pred_full, 1)

    # Update SWE for next day
    swe_prev = swe_pred_full.copy()
    print(f"Day {day_str} predicted and saved.")

    # Next day
    current_date += timedelta(days=1)