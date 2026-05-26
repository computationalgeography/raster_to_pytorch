import os


#os.environ['OMP_NUM_THREADS'] = '1'
#os.environ['MKL_NUM_THREADS'] = '1'
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#os.environ['NUMEXPR_NUM_THREADS'] = '1'



import sys
import pandas as pd
import rasterio
import numpy as np
import torch
from datetime import datetime, timedelta
from pathlib import Path
#from lib.lstm_definition import LSTM
from lib.lstm_definition import LSTM_MultiLayer
from lib.data_scaler import DataScaler

#from memory_profiler import profile

#    torch.set_num_interop_threads(1)
#    torch.set_num_threads(n)

import time

#@profile
def run_inference():
    # ---------------------------
    # Settings
    # ---------------------------
    dynamic_data_folder = "/projects/prjs1930/pilot_data"   # "data"
    static_data_folder = "/projects/prjs1930/pilot_data/DEM"   # "data"
    model_folder = "models_0"

    scalers_path_model_folder = Path(model_folder)

    dynamic_vars = ["pr", "rsds", "tas"]
    static_vars = ["elevation_mn", "slope_mn", "eastness_mn", "northness_mn", "tpi_mn"]
    target_var = "swe"

    all_vars = dynamic_vars + static_vars

    partition_size_y = 200
    partition_size_x = 300

    H, W = 10080, 43200
    
#    partition_size_y = H
#    partition_size_x = W

    # Date range
    warmup_date = datetime(1979, 10, 1)   # initital date 
    start_date = datetime(1979, 10, 5)    # end of warmup
    end_date   = datetime(1979, 10, 10)
#    end_date   = datetime(1979, 10, 15)
    # end_date   = datetime(2020, 9, 30)

    # ---------------------------
    # Model
    # ---------------------------
    # num_features = len(dynamic_vars) + len(static_vars) + 1  # + previous SWE
    num_features = len(all_vars) + 1  # + previous SWE

#    hidden_size = 5 #128
#    lstm_model = LSTM(input_size=num_features, hidden_size=hidden_size)
    
    hidden_size = 128     #64  #32   #64 #128   # hybrid: 128, LSTM: 64
    num_layers = 1      # hybrid: 1, LSTM: 2
    lstm_model = LSTM_MultiLayer(input_size=num_features, hidden_size=hidden_size, num_layers=num_layers)
    
#    torch.set_num_interop_threads(1) 



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

    # ---------------------------
    # Init state
    # ---------------------------
    swe_prev = np.full((H, W), fill_value=trans_zero_swe, dtype=np.float32)

    num_blocks_y = (H + partition_size_y - 1) // partition_size_y
    num_blocks_x = (W + partition_size_x - 1) // partition_size_x

    hidden_states = [None] * (num_blocks_y * num_blocks_x)   # one hidden state per partition

    start_read_static_var = time.perf_counter()
    
    # ---------------------------
    # read static vars
    # ---------------------------

    variable_maps_static = []

    for var in static_vars:
        var_file_static = os.path.join(static_data_folder, f"{var}.tif")

        with rasterio.open(var_file_static) as src:
            var_map_static = src.read(1)
            
            profile = src.profile.copy()

            print(f"reading {var} ...")

            if var_map_static.shape != (H, W):
                raise ValueError(f"{var} shape mismatch {var_map_static.shape}")

            variable_maps_static.append(var_map_static)

    static_stacked = np.stack(variable_maps_static, axis=0)
    
    end_read_static_var = time.perf_counter()

    print("Elapsed time:", end_read_static_var - start_read_static_var, "seconds")


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

        # add previous SWE: shape [1, H, W]
        target_stacked = swe_prev[np.newaxis, :, :]

        # ---------------------------
        # Partition loop
        # ---------------------------
        
        for i in range(0, H, partition_size_y):
            row_end = min(i + partition_size_y, H)

            for j in range(0, W, partition_size_x):
                col_end = min(j + partition_size_x, W)
                
                start_partition = time.perf_counter()

                idx = (i // partition_size_y) * num_blocks_x + (j // partition_size_x)

                # extract partitions
                dynamic_partition = dynamic_stacked[:, i:row_end, j:col_end]
                static_partition = static_stacked[:, i:row_end, j:col_end]
                target_partition = target_stacked[:, i:row_end, j:col_end]

                ph, pw = dynamic_partition.shape[1], dynamic_partition.shape[2]
                
                start_panda_scaling_reshape = time.perf_counter()

                # flatten correctly
                dynamic_flat = dynamic_partition.reshape(len(dynamic_vars), -1).T
                static_flat = static_partition.reshape(len(static_vars), -1).T
                target_flat = target_partition.reshape(1, -1).T


                # DataFrames
                df_dynamic = pd.DataFrame(dynamic_flat, columns= dynamic_vars)             #columns = ['pr', 'rsds', 'tas']
                df_static = pd.DataFrame(static_flat, columns=static_vars)
                # df_target = pd.DataFrame(target_flat, columns=['swe'])


                # scaling
                scaled_dynamic = scalers.transform_dataframe(df_dynamic, var_group='dynamic')
                scaled_static = scalers.transform_dataframe(df_static, var_group='static')
                # scaled_target = scalers.transform_dataframe(df_target, var_group='target')
                
                
                batch_size = dynamic_flat.shape[0]

                scaled_all_vars = np.empty((batch_size, num_features), dtype=np.float32)


                # combine correctly
                # scaled_all_vars = np.concatenate([scaled_target, scaled_dynamic, scaled_static], axis=1)   #  ['swe', 'pr', 'rsds', 'tas', 'static_vars']
#                scaled_all_vars = np.concatenate([target_flat, scaled_dynamic, scaled_static], axis=1)   #  ['swe', 'pr', 'rsds', 'tas', 'static_vars']

                scaled_all_vars[:, 0:1] = target_flat
                scaled_all_vars[:, 1:1+len(dynamic_vars)] = scaled_dynamic
                scaled_all_vars[:, 1+len(dynamic_vars):] = scaled_static
                
                end_panda_scaling_reshape = time.perf_counter()

                print("Elapsed time panda scaling reshape:", end_panda_scaling_reshape - start_panda_scaling_reshape, "seconds")

#                print("scaled_dynamic shape:", scaled_dynamic.shape)
#                print("scaled_static shape:", scaled_static.shape)
#                print("scaled_all_vars shape:", scaled_all_vars.shape)
#                print("num_features expected:", num_features)


                # LSTM input
                batch_size = scaled_all_vars.shape[0] # batch_size is the number of pixels in the map

                lstm_inputs = torch.tensor(
                    scaled_all_vars, dtype=torch.float32
                ).reshape(batch_size, 1, num_features)

                # hidden state
                prev_hidden = hidden_states[idx]
                
                start_predic_lstm = time.perf_counter()

                # forward pass
                with torch.no_grad():
                    pred, hidden_state = lstm_model(
                        lstm_inputs,
                        min_val=trans_zero_swe,
                        hidden=prev_hidden
                    )
                    
                end_predic_lstm = time.perf_counter()

                print("Elapsed time predic lstm in each partition:", end_predic_lstm - start_predic_lstm, "seconds")

                hidden_states[idx] = hidden_state

                # reshape prediction
                pred_map = pred.cpu().numpy().reshape(ph, pw)

                # write back safely
                swe_prev[i:row_end, j:col_end] = pred_map
                
                
                end_partition = time.perf_counter()

                print("Elapsed time all partition:", end_partition - start_partition, "seconds")
                

        # ---------------------------
        # Save output after warmup
        # ---------------------------
        if current_date >= start_date:
        
            start_write = time.perf_counter()

            inv = scalers.inverse_transform_target(swe_prev)

            inv = np.where(inv < 0.1, 0, inv)
            inv = inv.reshape(H, W).astype(np.float32)

            out_dir = os.path.join(dynamic_data_folder, target_var)
            os.makedirs(out_dir, exist_ok=True)

            out_file = os.path.join(out_dir, f"SWE_pred_{day_str}.tif")

#            with rasterio.open(
#                out_file,
#                'w',
#                driver='GTiff',
#                height=H,
#                width=W,
#                count=1,
#                dtype=np.float32
#            ) as dst:
#                dst.write(inv, 1)

            profile.update(
            dtype=rasterio.float32,
            count=1,
            compress="ZSTD"
            )
        
            with rasterio.open(out_file, 'w', **profile) as dst:
              dst.write(inv, 1)
              
            end_wirte = time.perf_counter()

            print("Elapsed time write:", end_wirte - start_write, "seconds")

              

        print(f"{day_str} done")

        current_date += timedelta(days=1)
        
        
#import os
#import time
#from datetime import datetime, timedelta
#from pathlib import Path
#
#import numpy as np
#import pandas as pd
#import rasterio
#import torch
#
#from lib.lstm_definition import LSTM_MultiLayer
#from lib.data_scaler import DataScaler


# =========================================================
# MAIN
# =========================================================

def run_inference_with_mask():

    # =====================================================
    # SETTINGS
    # =====================================================
    
    print("inside run_inference_with_mask")

    dynamic_data_folder = "/projects/prjs1930/pilot_data"
    static_data_folder = "/projects/prjs1930/pilot_data/DEM"

    model_folder = "models_0" # "models_20260510"

    scalers_path_model_folder = Path(model_folder)

    mask_file = "/projects/prjs1930/pilot_data/inference_mask.tif"

    dynamic_vars = ["pr", "rsds", "tas"]

    static_vars = [
        "elevation_mn",
        "slope_mn",
        "eastness_mn",
        "northness_mn",
        "tpi_mn"
    ]

    target_var = "swe"

    all_vars = dynamic_vars + static_vars

    # =====================================================
    # TILE SIZE
    # =====================================================

    partition_size_y = 200
    partition_size_x = 300

    # =====================================================
    # DOMAIN SIZE
    # =====================================================

    H, W = 10080, 43200

    # =====================================================
    # DATE RANGE
    # =====================================================

    warmup_date = datetime(1979, 10, 1)

    start_date = datetime(1979, 10, 5)     #datetime(1979, 10, 5)

    end_date = datetime(1979, 10, 30)      #datetime(1979, 10, 10)

    # =====================================================
    # MODEL
    # =====================================================

    num_features = len(all_vars) + 1

    hidden_size = 128
    num_layers = 1

    lstm_model = LSTM_MultiLayer(
        input_size=num_features,
        hidden_size=hidden_size,
        num_layers=num_layers
    )

    # -----------------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------------

    model_path = os.path.join(model_folder, "lstm_model.pt")

    lstm_model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )

    lstm_model.eval()

    # =====================================================
    # SCALERS
    # =====================================================

    scalers = DataScaler()

    scalers.load_transformers(
        scalers_path_model_folder
    )

    trans_zero_swe = scalers.get_transformed_zero()

    # =====================================================
    # READ MASK
    # =====================================================

    print("Reading mask...")

    with rasterio.open(mask_file) as src:

        global_mask = src.read(1).astype(bool)

    if global_mask.shape != (H, W):

        raise ValueError(
            f"Mask shape mismatch: {global_mask.shape}"
        )

    print("Mask loaded")


    # =====================================================
    # TILE COUNT
    # =====================================================

    num_blocks_y = (H + partition_size_y - 1) // partition_size_y

    num_blocks_x = (W + partition_size_x - 1) // partition_size_x

    # -----------------------------------------------------
    # Hidden state per tile
    # -----------------------------------------------------

    hidden_states = [None] * (num_blocks_y * num_blocks_x)

    # =====================================================
    # READ STATIC VARIABLES
    # =====================================================

    print("Reading static variables...")

    start_static = time.perf_counter()

    variable_maps_static = []

    profile = None

    for var in static_vars:

        var_file_static = os.path.join(
            static_data_folder,
            f"{var}.tif"
        )

        print(f"Reading {var} ...")

        with rasterio.open(var_file_static) as src:

            var_map_static = src.read(1)

            profile = src.profile.copy()
            
            NODATA_VALUE = src.nodata if src.nodata is not None else -9999.0

        if var_map_static.shape != (H, W):

            raise ValueError(
                f"{var} shape mismatch {var_map_static.shape}"
            )

        variable_maps_static.append(var_map_static)

    static_stacked = np.stack(
        variable_maps_static,
        axis=0
    )

    end_static = time.perf_counter()

    print(
        "Static variables loaded in:",
        end_static - start_static,
        "seconds"
    )
    
    # =====================================================
    # INITIAL SWE
    # =====================================================

    swe_prev = np.full(
        (H, W),
        fill_value=NODATA_VALUE,
        dtype=np.float32
    )

    # initialize ONLY valid pixels

    swe_prev[global_mask] = trans_zero_swe

    # =====================================================
    # TIME LOOP
    # =====================================================

    current_date = warmup_date

    while current_date <= end_date:

        total_day_start = time.perf_counter()

        day_str = current_date.strftime("%d_%m_%Y")

        print("\n===================================")
        print(f"Processing {day_str}")
        print("===================================\n")

        # =================================================
        # LOAD DYNAMIC VARIABLES
        # =================================================

        dynamic_start = time.perf_counter()

        variable_maps = []

        for var in dynamic_vars:

            var_file = os.path.join(
                dynamic_data_folder,
                var,
                f"CHELSA_{var}_{day_str}_V.2.1.tif"
            )

            print(f"Reading {var} ...")

            with rasterio.open(var_file) as src:

                var_map = src.read(1)

            if var_map.shape != (H, W):

                raise ValueError(
                    f"{var} shape mismatch {var_map.shape}"
                )

            variable_maps.append(var_map)

        dynamic_stacked = np.stack(
            variable_maps,
            axis=0
        )

        dynamic_end = time.perf_counter()

        print(
            "Dynamic variables loaded in:",
            dynamic_end - dynamic_start,
            "seconds"
        )

        # =================================================
        # PREVIOUS SWE
        # =================================================

        target_stacked = swe_prev[np.newaxis, :, :]

        # =================================================
        # TILE LOOP
        # =================================================

        for i in range(0, H, partition_size_y):

            row_end = min(i + partition_size_y, H)

            for j in range(0, W, partition_size_x):

                col_end = min(j + partition_size_x, W)

                tile_start = time.perf_counter()

                idx = (
                    (i // partition_size_y)
                    * num_blocks_x
                    + (j // partition_size_x)
                )

                # =========================================
                # TILE MASK
                # =========================================

                tile_mask = global_mask[
                    i:row_end,
                    j:col_end
                ]

                # skip fully invalid tile

                if not np.any(tile_mask):

                    continue

                valid_idx = np.where(
                    tile_mask.ravel()
                )[0]

                # =========================================
                # EXTRACT TILES
                # =========================================

                dynamic_partition = dynamic_stacked[
                    :,
                    i:row_end,
                    j:col_end
                ]

                static_partition = static_stacked[
                    :,
                    i:row_end,
                    j:col_end
                ]

                target_partition = target_stacked[
                    :,
                    i:row_end,
                    j:col_end
                ]

                ph, pw = tile_mask.shape

                # =========================================
                # FLATTEN
                # =========================================

                dynamic_flat = dynamic_partition.reshape(
                    len(dynamic_vars),
                    -1
                ).T

                static_flat = static_partition.reshape(
                    len(static_vars),
                    -1
                ).T

                target_flat = target_partition.reshape(
                    1,
                    -1
                ).T

                # =========================================
                # VALID PIXELS ONLY
                # =========================================

                dynamic_valid = dynamic_flat[valid_idx]

                static_valid = static_flat[valid_idx]

                target_valid = target_flat[valid_idx]

                # =========================================
                # DATAFRAMES
                # =========================================

                df_dynamic = pd.DataFrame(
                    dynamic_valid,
                    columns=dynamic_vars
                )

                df_static = pd.DataFrame(
                    static_valid,
                    columns=static_vars
                )

                # =========================================
                # SCALING
                # =========================================

                scaled_dynamic = scalers.transform_dataframe(
                    df_dynamic,
                    var_group='dynamic'
                )

                scaled_static = scalers.transform_dataframe(
                    df_static,
                    var_group='static'
                )

                # =========================================
                # BUILD MODEL INPUT
                # =========================================

                batch_size = len(valid_idx)

                scaled_all_vars = np.empty(
                    (batch_size, num_features),
                    dtype=np.float32
                )

                # previous SWE

                scaled_all_vars[:, 0:1] = target_valid

                # dynamic vars

                scaled_all_vars[
                    :,
                    1:1 + len(dynamic_vars)
                ] = scaled_dynamic

                # static vars

                scaled_all_vars[
                    :,
                    1 + len(dynamic_vars):
                ] = scaled_static

                # =========================================
                # TORCH INPUT
                # =========================================

                lstm_inputs = torch.from_numpy(
                    scaled_all_vars
                ).reshape(
                    batch_size,
                    1,
                    num_features
                )

                prev_hidden = hidden_states[idx]

                # =========================================
                # LSTM INFERENCE
                # =========================================

                with torch.no_grad():

                    pred, hidden_state = lstm_model(
                        lstm_inputs,
                        min_val=trans_zero_swe,
                        hidden=prev_hidden
                    )

                # detach hidden states

                hidden_states[idx] = (
                    hidden_state[0].detach(),
                    hidden_state[1].detach()
                )

                # =========================================
                # SCATTER BACK
                # =========================================

                pred_valid = pred.cpu().numpy().ravel()

                pred_tile = np.full(
                    ph * pw,
                    NODATA_VALUE,
                    dtype=np.float32
                )

                pred_tile[valid_idx] = pred_valid

                pred_tile = pred_tile.reshape(ph, pw)

                swe_prev[
                    i:row_end,
                    j:col_end
                ] = pred_tile

                tile_end = time.perf_counter()

                print(
                    f"Tile ({i}:{row_end}, {j}:{col_end}) "
                    f"done in "
                    f"{tile_end - tile_start:.2f} s"
                )

        # =================================================
        # SAVE OUTPUT
        # =================================================

        if current_date >= start_date:

            print("\nWriting output...")

            write_start = time.perf_counter()

            # ---------------------------------------------
            # Create output initialized as nodata
            # ---------------------------------------------

            inv = np.full(
                swe_prev.shape,
                NODATA_VALUE,
                dtype=np.float32
            )

            # ---------------------------------------------
            # inverse transform ONLY valid pixels
            # ---------------------------------------------

            valid_pixels = (
                swe_prev != NODATA_VALUE
            )

            inv[valid_pixels] = (
                scalers.inverse_transform_target(
                    swe_prev[valid_pixels]
                )
            )

            # ---------------------------------------------
            # Threshold snow
            # ---------------------------------------------

            low_snow_mask = (
                (inv < 0.1)
                &
                (inv != NODATA_VALUE)
            )

            inv[low_snow_mask] = 0

            # ---------------------------------------------
            # output directory
            # ---------------------------------------------

            out_dir = os.path.join(
                dynamic_data_folder,
                target_var
            )

            os.makedirs(out_dir, exist_ok=True)

            out_file = os.path.join(
                out_dir,
                f"SWE_pred_{day_str}.tif"
            )

            # ---------------------------------------------
            # update profile
            # ---------------------------------------------

            profile.update(
                dtype=rasterio.float32,
                count=1,
                compress="ZSTD",
                nodata=NODATA_VALUE
            )

            # ---------------------------------------------
            # write raster
            # ---------------------------------------------

            with rasterio.open(
                out_file,
                'w',
                **profile
            ) as dst:

                dst.write(inv, 1)

            write_end = time.perf_counter()

            print(
                "Write completed in:",
                write_end - write_start,
                "seconds"
            )

        total_day_end = time.perf_counter()

        print(
            f"\n{day_str} completed in "
            f"{total_day_end - total_day_start:.2f} seconds"
        )

        current_date += timedelta(days=1)