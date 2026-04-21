import os
import sys
import pandas as pd
import rasterio
import numpy as np
import torch
from datetime import datetime, timedelta
from pathlib import Path
from lib.lstm_definition import LSTM

import lib.data_scaler as ds      # load real module
sys.modules["data_scaler"] = ds   # alias old name ? new module
from lib.data_scaler import DataScaler  # normal usage

# ---------------------------
# Settings
# ---------------------------
dynamic_data_folder = "/projects/prjs1930/pilot_data"   # "data"
static_data_folder = "/projects/prjs1930/pilot_data/DEM"   # "data"
model_folder = "models"

scalers_path_model_folder = Path(model_folder)

dynamic_vars = ["pr", "rsds", "tas"]
static_vars = ["elevation_mn", "slope_mn", "eastness_mn", "northness_mn", "tpi_mn"]
target_var = "swe"

all_vars = dynamic_vars + static_vars

partition_size_y = 64
partition_size_x = 256

H, W = 10080, 43200

# Date range
warmup_date = datetime(1979, 10, 1)   # initital date 
start_date = datetime(1979, 10, 5)    # end of warmup
end_date   = datetime(1979, 10, 15)
# end_date   = datetime(2020, 9, 30)

# ---------------------------
# Model
# ---------------------------
# num_features = len(dynamic_vars) + len(static_vars) + 1  # + previous SWE
num_features = len(all_vars) + 1  # + previous SWE

hidden_size = 5
lstm_model = LSTM(input_size=num_features, hidden_size=hidden_size)

# lstm_model.load_state_dict(
#     torch.load(os.path.join(model_folder, "lstm_model.pt"), weights_only=True)
# )

lstm_model.eval()

# ---------------------------
# Scalers
# ---------------------------
scalers = DataScaler()
scalers.load_transformers(scalers_path_model_folder)

trans_zero_swe = scalers.get_transformed_zero()
target_transformer = scalers.get_transformers().get('target')

# ---------------------------
# Init state
# ---------------------------
swe_prev = np.zeros((H, W), dtype=np.float32)

num_blocks_y = (H + partition_size_y - 1) // partition_size_y
num_blocks_x = (W + partition_size_x - 1) // partition_size_x

hidden_states = [None] * (num_blocks_y * num_blocks_x)   # one hidden state per partition


# ---------------------------
# read static vars
# ---------------------------

variable_maps_static = []

for var in static_vars:
    var_file_static = os.path.join(static_data_folder, f"{var}.tif")

    with rasterio.open(var_file_static) as src:
        var_map_static = src.read(1)

        print(f"reading {var} ...")

        if var_map_static.shape != (H, W):
            raise ValueError(f"{var} shape mismatch {var_map_static.shape}")

        variable_maps_static.append(var_map_static)

static_stacked = np.stack(variable_maps_static, axis=0)


# ---------------------------
# Loop over dates
# ---------------------------


current_date = warmup_date

while current_date <= end_date:

    day_str = current_date.strftime("%d_%m_%Y")
    print(f"Processing {day_str}")

    # ---------------------------
    # Load dynamic variables
    # ---------------------------
    variable_maps = []

    for var in dynamic_vars:
        var_file = os.path.join(dynamic_data_folder, var, f"CHELSA_{var}_{day_str}_V.2.1.tif")

        with rasterio.open(var_file) as src:
            var_map = src.read(1)

            if var_map.shape != (H, W):
                raise ValueError(f"{var} shape mismatch {var_map.shape}")

            variable_maps.append(var_map)

    # Stack variables: shape [num_vars, H, W]
    dynamic_stacked = np.stack(variable_maps, axis=0)

    # add previous SWE: shape [num_vars+1, H, W]
    dynamic_stacked = np.concatenate(
        [swe_prev[np.newaxis, :, :], dynamic_stacked],
        axis=0
    )

    # ---------------------------
    # Partition loop
    # ---------------------------
    for i in range(0, H, partition_size_y):
        row_end = min(i + partition_size_y, H)

        for j in range(0, W, partition_size_x):
            col_end = min(j + partition_size_x, W)

            idx = (i // partition_size_y) * num_blocks_x + (j // partition_size_x)

            # extract partitions
            dynamic_partition = dynamic_stacked[:, i:row_end, j:col_end]
            static_partition = static_stacked[:, i:row_end, j:col_end]

            ph, pw = dynamic_partition.shape[1], dynamic_partition.shape[2]

            # flatten correctly
            dynamic_flat = dynamic_partition.reshape(len(dynamic_vars) + 1, -1).T
            static_flat = static_partition.reshape(len(static_vars), -1).T

            # DataFrames
            df_dynamic = pd.DataFrame(dynamic_flat, columns=['swe_prev'] + dynamic_vars)             #columns = ['swe_prev', 'pr', 'rsds', 'tas']
            df_static = pd.DataFrame(static_flat, columns=static_vars)

            # scaling
            scaled_dynamic = scalers.transform_dataframe(df_dynamic, var_group='dynamic')
            scaled_static = scalers.transform_dataframe(df_static, var_group='static')

            # combine correctly
            scaled_all_vars = np.concatenate([scaled_dynamic, scaled_static], axis=1)   #  ['swe', 'pr', 'rsds', 'tas', 'static_vars']

            # LSTM input
            batch_size = scaled_all_vars.shape[0] # batch_size is the number of pixels in the map

            lstm_inputs = torch.tensor(
                scaled_all_vars, dtype=torch.float32
            ).reshape(batch_size, 1, num_features)

            # hidden state
            prev_hidden = hidden_states[idx]

            # forward pass
            with torch.no_grad():
                pred, hidden_state = lstm_model(
                    lstm_inputs,
                    min_val=trans_zero_swe,
                    hidden_state=prev_hidden
                )

            hidden_states[idx] = hidden_state

            # reshape prediction
            pred_map = pred.cpu().numpy().reshape(ph, pw)

            # write back safely
            swe_prev[i:row_end, j:col_end] = pred_map

    # ---------------------------
    # Save output after warmup
    # ---------------------------
    if current_date >= start_date:

        inv = target_transformer.inverse_transform(
            pd.DataFrame(swe_prev.flatten(), columns=['swe'])
        )

        inv = np.where(inv < 0.1, 0, inv)
        inv = inv.reshape(H, W).astype(np.float32)

        out_file = os.path.join(dynamic_data_folder, target_var, f"SWE_pred_{day_str}.tif")

        with rasterio.open(
            out_file,
            'w',
            driver='GTiff',
            height=H,
            width=W,
            count=1,
            dtype=np.float32
        ) as dst:
            dst.write(inv, 1)

    print(f"{day_str} done")

    current_date += timedelta(days=1)