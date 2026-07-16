import os


os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import torch

num_threads = int(os.environ["SLURM_CPUS_PER_TASK"])

torch.set_num_threads(num_threads)
torch.set_num_interop_threads(1)

import sys
import pandas as pd
import rasterio
import numpy as np
#import psutil
#import gc

from datetime import datetime, timedelta
from pathlib import Path
#from lib.lstm_definition import LSTM
from lib.lstm_definition import LSTM_MultiLayer
from lib.data_scaler import DataScaler

from rasterio.windows import Window


#from memory_profiler import profile

#    torch.set_num_interop_threads(1)
#    torch.set_num_threads(n)

import time

import numpy as np
import rasterio


def create_balanced_col_blocks(mask_file, n_tasks):
    """
    Create column blocks with approximately equal number of valid mask pixels.

    Parameters
    ----------
    mask_file : str
        Path to mask raster
    n_tasks : int
        Number of parallel tasks

    Returns
    -------
    col_blocks : list of tuples
        [(start_col, end_col), ...]
    """

    # Read mask
    with rasterio.open(mask_file) as src:
        mask = src.read(1)

    H, W = mask.shape

    # Workload per column
    col_load = np.sum(mask == 1, axis=0)

    # cumulative workload
    cumulative = np.cumsum(col_load)

    total_load = cumulative[-1]
    target_load = total_load / n_tasks

    col_blocks = []

    start = 0

    for i in range(n_tasks):

        if i == n_tasks - 1:
            end = W
        else:
            end = np.searchsorted(
                cumulative,
                (i + 1) * target_load
            )

        col_blocks.append((start, end))
        start = end

    return col_blocks



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
    start_date = datetime(1979, 10, 30)    # end of warmup
    end_date   = datetime(1979, 12, 30)
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

    model_folder = "models_0" #"models_20260510" #"models_0"

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

    start_date = datetime(1979, 10, 3)     #datetime(1979, 10, 5)

    end_date = datetime(1979, 10, 5)      #datetime(1979, 10, 10)

    # =====================================================
    # MODEL
    # =====================================================

    num_features = len(all_vars) + 1

    hidden_size = 128 #64    
    num_layers = 1    #2    

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
        


def run_inference_multi_node():

    # =====================================================
    # DOMAIN SIZE
    # =====================================================
    H, W = 10080, 43200

    partition_size_y = 1000 #2500   #1000   #200
    partition_size_x = 1000 #2500   #1000   #300

    # =====================================================
    # SLURM SETUP
    # =====================================================
    n_tasks = 8 #int(os.environ["SLURM_NTASKS"])
    rank = 4 #int(os.environ["SLURM_PROCID"])
    
#    n_cols = int(np.sqrt(n_tasks))
#    n_rows = int(np.ceil(n_tasks / n_cols))

    print("get_num_threads: ", torch.get_num_threads())
    print("get_num_interop_threads: ", torch.get_num_interop_threads())

    n_cols = n_tasks
    n_rows = 1

    row_id = rank // n_cols
    col_id = rank % n_cols

    row_blocks = np.array_split(np.arange(H), n_rows)
    col_blocks = np.array_split(np.arange(W), n_cols)

    H_start = row_blocks[row_id][0]
    H_end   = row_blocks[row_id][-1] + 1

    W_start = col_blocks[col_id][0]
    W_end   = col_blocks[col_id][-1] + 1

    H_task = H_end - H_start
    W_task = W_end - W_start

    # =====================================================
    # PATHS
    # =====================================================
    dynamic_data_folder = "/projects/prjs1930/pilot_data"
    static_data_folder = "/projects/prjs1930/pilot_data/DEM"
    mask_file = "/projects/prjs1930/pilot_data/inference_mask.tif"
    model_folder = "lstm_models" #"models_20260510" #"models_0"
    
    scalers_path_model_folder = Path(model_folder)

    dynamic_vars = ["pr", "rsds", "tas"]
    static_vars = ["elevation_mn", "slope_mn", "eastness_mn", "northness_mn", "tpi_mn"]

    target_var = "swe"

    all_vars = dynamic_vars + static_vars
    num_features = len(all_vars) + 1

    # =====================================================
    # MODEL
    # =====================================================
    
    hidden_size = 32    
    num_layers = 2    

    lstm_model = LSTM_MultiLayer(
        input_size=num_features,
        hidden_size=hidden_size,
        num_layers=num_layers
    )
    
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
    # WINDOW (CRITICAL)
    # =====================================================
    window = Window(
        col_off=W_start,
        row_off=H_start,
        width=W_task,
        height=H_task
    )

    # =====================================================
    # MASK (LOCAL ONLY)
    # =====================================================
    with rasterio.open(mask_file) as src:
        local_mask = src.read(1, window=window).astype(bool)

    # =====================================================
    # STATIC DATA (READ ONLY ONCE PER NODE)
    # =====================================================
    static_maps = []

    NODATA_VALUE = -9999.0
    profile = None

    for var in static_vars:

        path = os.path.join(static_data_folder, f"{var}.tif")

        with rasterio.open(path) as src:
            arr = src.read(1, window=window)
            arr = arr*src.scales[0] + src.offsets[0]  # apply scale and offset
            profile = src.profile.copy()
            original_transform = profile["transform"]
            # if src.nodata is not None:
            #     NODATA_VALUE = src.nodata

        static_maps.append(arr)

    static_stacked = np.stack(static_maps, axis=0)

    # =====================================================
    # STATE ARRAYS
    # =====================================================
    swe_prev = np.full((H_task, W_task), NODATA_VALUE, dtype=np.float32)
    swe_prev[local_mask] = trans_zero_swe

    num_blocks_y = (H_task + partition_size_y - 1) // partition_size_y
    num_blocks_x = (W_task + partition_size_x - 1) // partition_size_x

    hidden_states = [None] * (num_blocks_y * num_blocks_x)

    # =====================================================
    # TIME LOOP
    # =====================================================
    warmup_date = datetime(1979, 10, 1)
    start_date = datetime(1980, 10, 1)    #(1979, 10, 3)
    end_date = datetime(1982, 10, 1)     #(1979, 10, 5)

    current_date = warmup_date

    while current_date <= end_date:

        day_str = current_date.strftime("%d_%m_%Y")

        print(f"\nRank {rank} processing {day_str}")

        # =================================================
        # DYNAMIC VARIABLES (WINDOW READ)
        # =================================================
        dynamic_maps = []

        for var in dynamic_vars:

            path = os.path.join(
                dynamic_data_folder,
                var,
                f"CHELSA_{var}_{day_str}_V.2.1.tif"
            )

            with rasterio.open(path) as src:
                arr = src.read(1, window=window)
                arr = arr*src.scales[0] + src.offsets[0]  # apply scale and offset

            dynamic_maps.append(arr)

        dynamic_stacked = np.stack(dynamic_maps, axis=0)

        target_stacked = swe_prev[np.newaxis, :, :]

        # =================================================
        # TILE LOOP
        # =================================================
        for i in range(0, H_task, partition_size_y):

            row_end = min(i + partition_size_y, H_task)

            for j in range(0, W_task, partition_size_x):

                col_end = min(j + partition_size_x, W_task)

                tile_mask = local_mask[i:row_end, j:col_end]

                if not np.any(tile_mask):
                    continue

                valid_idx = np.where(tile_mask.ravel())[0]

                # -----------------------------------------
                # TILE DATA (ALREADY LOCAL)
                # -----------------------------------------
                dyn = dynamic_stacked[:, i:row_end, j:col_end]
                stat = static_stacked[:, i:row_end, j:col_end]
                targ = target_stacked[:, i:row_end, j:col_end]

                ph, pw = tile_mask.shape

                dyn = dyn.reshape(len(dynamic_vars), -1).T
                stat = stat.reshape(len(static_vars), -1).T
                targ = targ.reshape(1, -1).T

                dyn = dyn[valid_idx]
                stat = stat[valid_idx]
                targ = targ[valid_idx]

                # -----------------------------------------
                # SCALE
                # -----------------------------------------
                df_dyn = pd.DataFrame(dyn, columns=dynamic_vars)
                df_stat = pd.DataFrame(stat, columns=static_vars)

                dyn_s = scalers.transform_dataframe(df_dyn, "dynamic")
                stat_s = scalers.transform_dataframe(df_stat, "static")

                batch = len(valid_idx)

                x = np.empty((batch, num_features), dtype=np.float32)
                x[:, 0:len(dynamic_vars)] = dyn_s
                x[:, len(dynamic_vars):-1] = stat_s
                x[:, -1] = targ.ravel()

                x = torch.from_numpy(x).reshape(batch, 1, num_features)

                idx = (i // partition_size_y) * num_blocks_x + (j // partition_size_x)

                prev_hidden = hidden_states[idx]

                with torch.no_grad():
                    pred, h = lstm_model(
                        x,
                        min_val=trans_zero_swe,
                        hidden=prev_hidden
                    )

                hidden_states[idx] = (h[0].detach(), h[1].detach())

                pred = pred.cpu().numpy().ravel()

                out_tile = np.full(ph * pw, NODATA_VALUE, dtype=np.float32)
                out_tile[valid_idx] = pred
                out_tile = out_tile.reshape(ph, pw)

                swe_prev[i:row_end, j:col_end] = out_tile

        # =================================================
        # WRITE OUTPUT
        # =================================================
        if current_date >= start_date:

            inv = np.full((H_task, W_task), NODATA_VALUE, dtype=np.float32)

            valid = ~np.isnan(swe_prev)
            inv[valid] = scalers.inverse_transform_target(swe_prev[valid])

            out_dir = os.path.join(dynamic_data_folder, target_var + "_lstm")
            os.makedirs(out_dir, exist_ok=True)

            out_file = os.path.join(out_dir, f"SWE_{day_str}_r{rank}.tif")

            transform = rasterio.windows.transform(window, original_transform)

            profile.update(
                height=H_task,
                width=W_task,
                transform=transform,
                dtype=rasterio.float32,
                count=1,
                nodata=NODATA_VALUE
            )

            with rasterio.open(out_file, "w", **profile) as dst:
                dst.write(inv, 1)

        current_date += timedelta(days=1)
        

def run_inference_multi_node_hybrid_model():

    # =====================================================
    # DOMAIN SIZE
    # =====================================================
    H, W = 10080, 43200

    partition_size_y = 1000   #2500 #1000  #2500   #1000   #200
    partition_size_x = 1000   #2500 #1000  #2500   #1000   #300
    
    # =====================================================
    # PATHS
    # =====================================================
    dynamic_data_folder = "/projects/prjs1930/pilot_data"
    static_data_folder = "/projects/prjs1930/pilot_data/DEM"
    mask_file = "/projects/prjs1930/pilot_data/inference_mask.tif"
    model_folder = "hybrid_models"      #"models_0" #"models_20260510" #"models_0"
    

    # =====================================================
    # SLURM SETUP
    # =====================================================
#    n_tasks =   16 #8 # n_tasks = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
#    rank =  12 #6 # int(os.environ["SLURM_ARRAY_TASK_ID"])
    n_tasks = int(os.environ["SLURM_ARRAY_TASK_COUNT"])   # 16 #8
    rank =   int(os.environ["SLURM_ARRAY_TASK_ID"]) #     12 #6
    
    
#    n_cols = int(np.sqrt(n_tasks))
#    n_rows = int(np.ceil(n_tasks / n_cols))

    # print("get_num_threads: ", torch.get_num_threads())
    # print("get_num_interop_threads: ", torch.get_num_interop_threads())

    # def debug_mem(tag=None):
    #     try:
    #         p = psutil.Process(os.getpid())
    #         mem = p.memory_info().rss / (1024 ** 2)
    #         vm = psutil.virtual_memory()
    #         print(f"[MEM] {tag or ''} RSS={mem:.1f}MB, VM%={vm.percent}%")
    #     except Exception as e:
    #         print("[MEM] psutil not available:", e)

    # debug_mem("after thread setup")

    n_cols = n_tasks
    n_rows = 1

    row_id = rank // n_cols
    col_id = rank % n_cols
    
    # ----- uniform band witdth for tasks -----

#    row_blocks = np.array_split(np.arange(H), n_rows)
#    col_blocks = np.array_split(np.arange(W), n_cols)
#
#    H_start = row_blocks[row_id][0]
#    H_end   = row_blocks[row_id][-1] + 1
#
#    W_start = col_blocks[col_id][0]
#    W_end   = col_blocks[col_id][-1] + 1
#
#    H_task = H_end - H_start
#    W_task = W_end - W_start
    
    # ----- End: uniform band witdth for tasks -----
    
    # ----- adjustable band witdth for tasks -----
    
    col_blocks = create_balanced_col_blocks(
    mask_file,
    n_tasks)

    W_start, W_end = col_blocks[rank]
    
    H_start = 0
    H_end = H

    H_task = H_end - H_start
    W_task = W_end - W_start
    
    # ----- End adjustable band withdth for taks -----  
    

#    print(f"Task window rows: {H_start}:{H_end} (H_task={H_task}), cols: {W_start}:{W_end} (W_task={W_task})")
#    print(f"Partition size: y={partition_size_y}, x={partition_size_x}")
    
    
    with rasterio.open(mask_file) as src:
        mask = src.read(1)
        
    print(
    "rank:", rank,
    "W range:", W_start, W_end,
    "pixels:", H_task*W_task,
    "valid pixels:",
    np.sum(mask[H_start:H_end, W_start:W_end])
    )


    scalers_path_model_folder = Path(model_folder)

    dynamic_vars = ["pr", "rsds", "tas", "era5_swe"]
    static_vars = ["elevation_mn", "slope_mn", "eastness_mn", "northness_mn", "tpi_mn"]

    target_var = "swe"

    all_vars = dynamic_vars + static_vars
    num_features = len(all_vars) + 1

    # =====================================================
    # MODEL
    # =====================================================
    
    hidden_size = 128 #64    
    num_layers = 1    #2    

    lstm_model = LSTM_MultiLayer(
        input_size=num_features,
        hidden_size=hidden_size,
        num_layers=num_layers
    )
    
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
    # WINDOW (CRITICAL)
    # =====================================================
    window = Window(
        col_off=W_start,
        row_off=H_start,
        width=W_task,
        height=H_task
    )

    # =====================================================
    # MASK (LOCAL ONLY)
    # =====================================================
    with rasterio.open(mask_file) as src:
        local_mask = src.read(1, window=window).astype(bool)

    # print("local_mask shape:", local_mask.shape, "sum(valid)=", np.count_nonzero(local_mask))
    # print("local_mask nbytes:", local_mask.nbytes)
    # debug_mem("after mask read")

    # =====================================================
    # STATIC DATA (READ ONLY ONCE PER NODE)
    # =====================================================
    static_maps = []

    NODATA_VALUE = -9999.0 #-32768
    profile = None

    for var in static_vars:

        path = os.path.join(static_data_folder, f"{var}.tif")

        with rasterio.open(path) as src:
            arr = src.read(1, window=window)
            arr = arr*src.scales[0] + src.offsets[0]  # apply scale and offset
            profile = src.profile.copy()
            original_transform = profile["transform"]
            # if src.nodata is not None:
            #     NODATA_VALUE = src.nodata

        static_maps.append(arr)

    static_stacked = np.stack(static_maps, axis=0)

    # print("static_stacked shape:", static_stacked.shape, "nbytes:", static_stacked.nbytes)
    # debug_mem("after reading static")

    # =====================================================
    # STATE ARRAYS
    # =====================================================
    swe_prev = np.full((H_task, W_task), NODATA_VALUE, dtype=np.float32) # np.full((H_task, W_task), np.nan, dtype=np.float32)
    swe_prev[local_mask] = trans_zero_swe

    # Print the NODATA_VALUE and transformed zero for debugging
    # print("transformed zero SWE:", trans_zero_swe)
    # print("swe_prev initialized shape:", swe_prev.shape, "nbytes:", swe_prev.nbytes)
    # print("swe_prev sample min,max:", float(np.nanmin(swe_prev)), float(np.nanmax(swe_prev)))
    # debug_mem("after swe_prev init")

    # Inverse transorm swe_prev to check the initial values in original scale
#    inv_swe_prev = np.full_like(swe_prev, NODATA_VALUE) #np.full_like(swe_prev, np.nan)
#    valid = ~np.isnan(swe_prev)
#    inv_swe_prev[valid] = scalers.inverse_transform_target(swe_prev[valid])
#    print("inverse transformed swe_prev sample min,max:", float(np.nanmin(inv_swe_prev)), float(np.nanmax(inv_swe_prev)))

    num_blocks_y = (H_task + partition_size_y - 1) // partition_size_y
    num_blocks_x = (W_task + partition_size_x - 1) // partition_size_x

    hidden_states = [None] * (num_blocks_y * num_blocks_x)

    # print("num_blocks_y, num_blocks_x:", num_blocks_y, num_blocks_x)
    # print("hidden_states length:", len(hidden_states))
    # debug_mem("before time loop")

    # =====================================================
    # TIME LOOP
    # =====================================================
    warmup_date = datetime(1979, 10, 1)
    start_date = datetime(1980, 10, 1)   #(1979, 10, 3)    #(1979, 10, 3) (1980, 10, 1)
    end_date = datetime(1982, 10, 1)     #(1979, 10, 5)   #(1979, 10, 5) (1982, 10, 1)

    current_date = warmup_date

    while current_date <= end_date:

        day_str = current_date.strftime("%d_%m_%Y")

        print(f"Rank {rank} processing {day_str}")

        # debug_mem(f"start day {day_str}")

        # =================================================
        # DYNAMIC VARIABLES (WINDOW READ)
        # =================================================
        dynamic_maps = []

        for var in dynamic_vars:

            if var == "era5_swe":
                filename = f"sd_ERA5-Land_{day_str}.tif"
                path = os.path.join(dynamic_data_folder, "sd", filename)
        
            else:
                filename = f"CHELSA_{var}_{day_str}_V.2.1.tif"
                path = os.path.join(dynamic_data_folder, var, filename)
        
            with rasterio.open(path) as src:
                arr = src.read(1, window=window)
                scale_factor = src.scales[0] * (1000 if var == "era5_swe" else 1) # convert to era5 to mm
                arr = arr*scale_factor + src.offsets[0]  # apply scale and offset


            dynamic_maps.append(arr)

        dynamic_stacked = np.stack(dynamic_maps, axis=0)
        target_stacked = swe_prev[np.newaxis, :, :]

        # if current_date == warmup_date:
        #     print("dynamic_stacked shape:", dynamic_stacked.shape, "nbytes:", dynamic_stacked.nbytes)
        #     print("target_stacked shape:", target_stacked.shape)
        #     debug_mem("after dynamic stack (first day)")

        # =================================================
        # TILE LOOP
        # =================================================
        for i in range(0, H_task, partition_size_y):

            row_end = min(i + partition_size_y, H_task)

            for j in range(0, W_task, partition_size_x):

                col_end = min(j + partition_size_x, W_task)

                tile_mask = local_mask[i:row_end, j:col_end]

                if not np.any(tile_mask):
                    continue

                valid_idx = np.where(tile_mask.ravel())[0]

                # -----------------------------------------
                # TILE DATA (ALREADY LOCAL)
                # -----------------------------------------
                dyn = dynamic_stacked[:, i:row_end, j:col_end]
                stat = static_stacked[:, i:row_end, j:col_end]
                targ = target_stacked[:, i:row_end, j:col_end]

                ph, pw = tile_mask.shape

                dyn = dyn.reshape(len(dynamic_vars), -1).T
                stat = stat.reshape(len(static_vars), -1).T
                targ = targ.reshape(1, -1).T

                dyn = dyn[valid_idx]
                stat = stat[valid_idx]
                targ = targ[valid_idx]

                # -----------------------------------------
                # SCALE
                # -----------------------------------------
                df_dyn = pd.DataFrame(dyn, columns=dynamic_vars)
                df_stat = pd.DataFrame(stat, columns=static_vars)

                dyn_s = scalers.transform_dataframe(df_dyn, "dynamic")
                stat_s = scalers.transform_dataframe(df_stat, "static")

                batch = len(valid_idx)

                x = np.empty((batch, num_features), dtype=np.float32)
#                x[:, 0] = targ.ravel()
#                x[:, 1:1+len(dynamic_vars)] = dyn_s
#                x[:, 1+len(dynamic_vars):] = stat_s
                
                x[:, 0:len(dynamic_vars)] = dyn_s
                x[:, len(dynamic_vars):-1] = stat_s
                x[:, -1] = targ.ravel()

                x = torch.from_numpy(x).reshape(batch, 1, num_features)

                idx = (i // partition_size_y) * num_blocks_x + (j // partition_size_x)

                prev_hidden = hidden_states[idx]

                with torch.no_grad():
                    pred, h = lstm_model(
                        x,
                        min_val=trans_zero_swe,
                        hidden=prev_hidden
                    )

                hidden_states[idx] = (h[0].detach(), h[1].detach())

                pred = pred.cpu().numpy().ravel()

                # if hasattr(h[0], 'shape'):
                #     print("hidden h0 shape:", tuple(h[0].shape), "h1 shape:", tuple(h[1].shape))

                out_tile = np.full(ph * pw, np.nan, dtype=np.float32)
                out_tile[valid_idx] = pred
                out_tile = out_tile.reshape(ph, pw)

                swe_prev[i:row_end, j:col_end] = out_tile

                # print("updated swe_prev region min,max:", float(np.nanmin(swe_prev[i:row_end, j:col_end])), float(np.nanmax(swe_prev[i:row_end, j:col_end])))


        # =================================================
        # WRITE OUTPUT
        # =================================================
        if current_date >= start_date:

            inv = np.full((H_task, W_task), NODATA_VALUE, dtype=np.float32)

            valid = swe_prev != NODATA_VALUE      # valid = ~np.isnan(swe_prev)
            inv[valid] = scalers.inverse_transform_target(swe_prev[valid])

#            print("output stats: valid pixels", np.count_nonzero(valid), " min,max:", float(np.nanmin(inv[valid])) if np.count_nonzero(valid) else None, float(np.nanmax(inv[valid])) if np.count_nonzero(valid) else None)

            out_dir = os.path.join(dynamic_data_folder, target_var)
            os.makedirs(out_dir, exist_ok=True)

            out_file = os.path.join(out_dir, f"SWE_{day_str}_r{rank}.tif")

            transform = rasterio.windows.transform(window, original_transform)

            profile.update(
                height=H_task,
                width=W_task,
                transform=transform,
                dtype=rasterio.float32,
                count=1,
                nodata=NODATA_VALUE
            )

            with rasterio.open(out_file, "w", **profile) as dst:
                dst.write(inv, 1)

            try:
                fsize = os.path.getsize(out_file) / (1024 ** 2)
                print(f"Wrote output {out_file}, size={fsize:.2f} MB")
            except Exception:
                print("Wrote output", out_file)

        current_date += timedelta(days=1)
        

#def run_inference_multi_node_hybrid_model_debug():
#
#    # =====================================================
#    # Debug variables
#    # =====================================================
#
#    debug_col = 22541
#    debug_row = 4577
#     
#    enable_debug_pixel = True
#     
#    debug_time = "14_10_1980"
#
#    # =====================================================
#    # DOMAIN SIZE
#    # =====================================================
#    H, W = 10080, 43200
#
#    partition_size_y = 1000  #2500   #1000   #200
#    partition_size_x = 1000  #2500   #1000   #300
#
#    # =====================================================
#    # SLURM SETUP
#    # =====================================================
#    n_tasks = 8 #int(os.environ["SLURM_NTASKS"])
#    rank = 4 #int(os.environ["SLURM_PROCID"])
#    
##    n_cols = int(np.sqrt(n_tasks))
##    n_rows = int(np.ceil(n_tasks / n_cols))
#
#    print("get_num_threads: ", torch.get_num_threads())
#    print("get_num_interop_threads: ", torch.get_num_interop_threads())
#
#    def debug_mem(tag=None):
#        try:
#            p = psutil.Process(os.getpid())
#            mem = p.memory_info().rss / (1024 ** 2)
#            vm = psutil.virtual_memory()
#            print(f"[MEM] {tag or ''} RSS={mem:.1f}MB, VM%={vm.percent}%")
#        except Exception as e:
#            print("[MEM] psutil not available:", e)
#
#    debug_mem("after thread setup")
#
#    n_cols = n_tasks
#    n_rows = 1
#
#    row_id = rank // n_cols
#    col_id = rank % n_cols
#
#    row_blocks = np.array_split(np.arange(H), n_rows)
#    col_blocks = np.array_split(np.arange(W), n_cols)
#
#    H_start = row_blocks[row_id][0]
#    H_end   = row_blocks[row_id][-1] + 1
#
#    W_start = col_blocks[col_id][0]
#    W_end   = col_blocks[col_id][-1] + 1
#
#    H_task = H_end - H_start
#    W_task = W_end - W_start
#
#    print(f"Task window rows: {H_start}:{H_end} (H_task={H_task}), cols: {W_start}:{W_end} (W_task={W_task})")
#    print(f"Partition size: y={partition_size_y}, x={partition_size_x}")
#
#    # =====================================================
#    # PATHS
#    # =====================================================
#    dynamic_data_folder = "/projects/prjs1930/pilot_data"
#    static_data_folder = "/projects/prjs1930/pilot_data/DEM"
#    mask_file = "/projects/prjs1930/pilot_data/inference_mask.tif"
#    model_folder = "hybrid_models"      #"models_0" #"models_20260510" #"models_0"
#
#    scalers_path_model_folder = Path(model_folder)
#
#    dynamic_vars = ["pr", "rsds", "tas", "era5_swe"]
#    static_vars = ["elevation_mn", "slope_mn", "eastness_mn", "northness_mn", "tpi_mn"]
#
#    target_var = "swe"
#
#    all_vars = dynamic_vars + static_vars
#    num_features = len(all_vars) + 1
#
#    # =====================================================
#    # MODEL
#    # =====================================================
#    
#    hidden_size = 128 #64    
#    num_layers = 1    #2    
#
#    lstm_model = LSTM_MultiLayer(
#        input_size=num_features,
#        hidden_size=hidden_size,
#        num_layers=num_layers
#    )
#    
#    model_path = os.path.join(model_folder, "lstm_model.pt")
#
#    lstm_model.load_state_dict(
#        torch.load(model_path, map_location="cpu")
#    )
#    lstm_model.eval()
#
#    # =====================================================
#    # SCALERS
#    # =====================================================
#
#    scalers = DataScaler()
#
#    scalers.load_transformers(
#        scalers_path_model_folder
#    )
#
#    trans_zero_swe = scalers.get_transformed_zero()
#
#    # =====================================================
#    # WINDOW (CRITICAL)
#    # =====================================================
#    window = Window(
#        col_off=W_start,
#        row_off=H_start,
#        width=W_task,
#        height=H_task
#    )
#
#    # =====================================================
#    # MASK (LOCAL ONLY)
#    # =====================================================
#    with rasterio.open(mask_file) as src:
#        local_mask = src.read(1, window=window).astype(bool)
#
#    print("local_mask shape:", local_mask.shape, "sum(valid)=", np.count_nonzero(local_mask))
#    print("local_mask nbytes:", local_mask.nbytes)
#    debug_mem("after mask read")
#
#    # =====================================================
#    # STATIC DATA (READ ONLY ONCE PER NODE)
#    # =====================================================
#    static_maps = []
#
#    NODATA_VALUE = -9999.0 #-32768
#    profile = None
#
#    for var in static_vars:
#
#        path = os.path.join(static_data_folder, f"{var}.tif")
#
#        with rasterio.open(path) as src:
#            arr = src.read(1, window=window)
#            arr = arr*src.scales[0] + src.offsets[0]  # apply scale and offset
#            profile = src.profile.copy()
#            original_transform = profile["transform"]
#            # if src.nodata is not None:
#            #     NODATA_VALUE = src.nodata
#
#        static_maps.append(arr)
#
#    static_stacked = np.stack(static_maps, axis=0)
#
#    print("static_stacked shape:", static_stacked.shape, "nbytes:", static_stacked.nbytes)
#    debug_mem("after reading static")
#
#    # =====================================================
#    # STATE ARRAYS
#    # =====================================================
#    swe_prev = np.full((H_task, W_task), np.nan, dtype=np.float32)
#    swe_prev[local_mask] = trans_zero_swe
#
#    # Print the NODATA_VALUE and transformed zero for debugging
#    print("transformed zero SWE:", trans_zero_swe)
#    print("swe_prev initialized shape:", swe_prev.shape, "nbytes:", swe_prev.nbytes)
#    print("swe_prev sample min,max:", float(np.nanmin(swe_prev)), float(np.nanmax(swe_prev)))
#    debug_mem("after swe_prev init")
#
#    # Inverse transorm swe_prev to check the initial values in original scale
#    inv_swe_prev = np.full_like(swe_prev, np.nan)
#    valid = ~np.isnan(swe_prev)
#    inv_swe_prev[valid] = scalers.inverse_transform_target(swe_prev[valid])
#    print("inverse transformed swe_prev sample min,max:", float(np.nanmin(inv_swe_prev)), float(np.nanmax(inv_swe_prev)))
#
#    num_blocks_y = (H_task + partition_size_y - 1) // partition_size_y
#    num_blocks_x = (W_task + partition_size_x - 1) // partition_size_x
#
#    hidden_states = [None] * (num_blocks_y * num_blocks_x)
#
#    print("num_blocks_y, num_blocks_x:", num_blocks_y, num_blocks_x)
#    print("hidden_states length:", len(hidden_states))
#    debug_mem("before time loop")
#
#    # =====================================================
#    # TIME LOOP
#    # =====================================================
#    warmup_date = datetime(1980, 10, 1)   #1979, 10, 1)
#    start_date = datetime(1980, 10, 5)    #(1979, 10, 3)
#    end_date = datetime(1980, 10, 15)     #(1979, 10, 5)
#
#    current_date = warmup_date
#
#    while current_date <= end_date:
#
#        day_str = current_date.strftime("%d_%m_%Y")
#
#        print(f"\nRank {rank} processing {day_str}")
#
#        debug_mem(f"start day {day_str}")
#
#        # =================================================
#        # DYNAMIC VARIABLES (WINDOW READ)
#        # =================================================
#        dynamic_maps = []
#
#        for var in dynamic_vars:
#
#            if var == "era5_swe":
#                filename = f"sd_ERA5-Land_{day_str}.tif"
#                path = os.path.join(dynamic_data_folder, "sd", filename)
#        
#            else:
#                filename = f"CHELSA_{var}_{day_str}_V.2.1.tif"
#                path = os.path.join(dynamic_data_folder, var, filename)
#        
#            with rasterio.open(path) as src:
#                arr = src.read(1, window=window)
#                scale_factor = src.scales[0] * (1000 if var == "era5_swe" else 1) # convert to era5 to mm
#                arr = arr*scale_factor + src.offsets[0]  # apply scale and offset
#
#
#            dynamic_maps.append(arr)
#
#        dynamic_stacked = np.stack(dynamic_maps, axis=0)
#        target_stacked = swe_prev[np.newaxis, :, :]
#
#        if current_date == warmup_date:
#            print("dynamic_stacked shape:", dynamic_stacked.shape, "nbytes:", dynamic_stacked.nbytes)
#            print("target_stacked shape:", target_stacked.shape)
#            debug_mem("after dynamic stack (first day)")
#
#        # =================================================
#        # TILE LOOP
#        # =================================================
#        for i in range(0, H_task, partition_size_y):
#
#            row_end = min(i + partition_size_y, H_task)
#
#            for j in range(0, W_task, partition_size_x):
#
#                col_end = min(j + partition_size_x, W_task)
#
#                tile_mask = local_mask[i:row_end, j:col_end]
#
#                if not np.any(tile_mask):
#                    continue
#
#                valid_idx = np.where(tile_mask.ravel())[0]
#                
#                
#                is_debug_tile = False
#                local_idx = None
#                local_r = None
#                local_c = None
#                
#                
#                if enable_debug_pixel:
#                    if i <= debug_row < row_end and j <= debug_col < col_end and day_str == debug_time:
#                        is_debug_tile = True
#                        
#                        ph_debug, pw_debug = tile_mask.shape
#                
#                        local_r = debug_row - i
#                        local_c = debug_col - j
#                
#                        local_idx = local_r * pw_debug + local_c
#                
#                        print("\n================ DEBUG TILE ================")
#                        print("global (row,col):", debug_row, debug_col)
#                        print("local (r,c):", local_r, local_c)
#                        print("flat index:", local_idx)
#
#                # -----------------------------------------
#                # TILE DATA (ALREADY LOCAL)
#                # -----------------------------------------
#                dyn = dynamic_stacked[:, i:row_end, j:col_end]
#                stat = static_stacked[:, i:row_end, j:col_end]
#                targ = target_stacked[:, i:row_end, j:col_end]
#
#                ph, pw = tile_mask.shape
#
#                dyn = dyn.reshape(len(dynamic_vars), -1).T
#                stat = stat.reshape(len(static_vars), -1).T
#                targ = targ.reshape(1, -1).T
#
#                dyn = dyn[valid_idx]
#                stat = stat[valid_idx]
#                targ = targ[valid_idx]
#
#                # -----------------------------------------
#                # SCALE
#                # -----------------------------------------
#                df_dyn = pd.DataFrame(dyn, columns=dynamic_vars)
#                df_stat = pd.DataFrame(stat, columns=static_vars)
#
#                dyn_s = scalers.transform_dataframe(df_dyn, "dynamic")
#                stat_s = scalers.transform_dataframe(df_stat, "static")
#
#                batch = len(valid_idx)
#
#                x = np.empty((batch, num_features), dtype=np.float32)
##                x[:, 0] = targ.ravel()
##                x[:, 1:1+len(dynamic_vars)] = dyn_s
##                x[:, 1+len(dynamic_vars):] = stat_s
#
#                x[:, 0:len(dynamic_vars)] = dyn_s
#                x[:, len(dynamic_vars):-1] = stat_s
#                x[:, -1] = targ.ravel()
#                
#                if is_debug_tile:
#
#                    print("\n--- DEBUG INPUT CHECK ---")
#                
#                    print("dyn_s:", dyn_s[local_idx])
#                    print("stat_s:", stat_s[local_idx])
#                    print("target:", targ.ravel()[local_idx])
#                
#                    print("FULL LSTM INPUT VECTOR:")
#                    print(x[local_idx, 0, :])
#                    
#
#                x = torch.from_numpy(x).reshape(batch, 1, num_features)
#
#                idx = (i // partition_size_y) * num_blocks_x + (j // partition_size_x)
#
#                prev_hidden = hidden_states[idx]
#
#                with torch.no_grad():
#                    pred, h = lstm_model(
#                        x,
#                        min_val=trans_zero_swe,
#                        hidden=prev_hidden
#                    )
#
#                hidden_states[idx] = (h[0].detach(), h[1].detach())
#
#                pred = pred.cpu().numpy().ravel()
#
#                # if hasattr(h[0], 'shape'):
#                #     print("hidden h0 shape:", tuple(h[0].shape), "h1 shape:", tuple(h[1].shape))
#
#                out_tile = np.full(ph * pw, np.nan, dtype=np.float32)
#                out_tile[valid_idx] = pred
#                out_tile = out_tile.reshape(ph, pw)
#
#                swe_prev[i:row_end, j:col_end] = out_tile
#    
#
#                # print("updated swe_prev region min,max:", float(np.nanmin(swe_prev[i:row_end, j:col_end])), float(np.nanmax(swe_prev[i:row_end, j:col_end])))
#                
#                if is_debug_tile:
#
#                    print("\n--- DEBUG OUTPUT CHECK ---")
#                
#                    print("prediction:", pred[valid_idx.tolist().index(local_idx)] if local_idx in valid_idx else "NOT VALID PIXEL")
#                
#                    print("previous SWE:", swe_prev[i + local_r, j + local_c])
#
#
#        # =================================================
#        # WRITE OUTPUT
#        # =================================================
#        if current_date >= start_date:
#
#            inv = np.full((H_task, W_task), NODATA_VALUE, dtype=np.float32)
#
#            valid = ~np.isnan(swe_prev)
#            inv[valid] = scalers.inverse_transform_target(swe_prev[valid])
#
#            print("output stats: valid pixels", np.count_nonzero(valid), " min,max:", float(np.nanmin(inv[valid])) if np.count_nonzero(valid) else None, float(np.nanmax(inv[valid])) if np.count_nonzero(valid) else None)
#
#            out_dir = os.path.join(dynamic_data_folder, target_var)
#            os.makedirs(out_dir, exist_ok=True)
#
#            out_file = os.path.join(out_dir, f"SWE_{day_str}_r{rank}.tif")
#
#            transform = rasterio.windows.transform(window, original_transform)
#
#            profile.update(
#                height=H_task,
#                width=W_task,
#                transform=transform,
#                dtype=rasterio.float32,
#                count=1,
#                nodata=NODATA_VALUE
#            )
#
#            with rasterio.open(out_file, "w", **profile) as dst:
#                dst.write(inv, 1)
#
#            try:
#                fsize = os.path.getsize(out_file) / (1024 ** 2)
#                print(f"Wrote output {out_file}, size={fsize:.2f} MB")
#            except Exception:
#                print("Wrote output", out_file)
#
#        current_date += timedelta(days=1)

def run_inference_multi_node_hybrid_model_with_saving_states():

    # =====================================================
    # DOMAIN SIZE
    # =====================================================
    H, W = 10080, 43200

    partition_size_y = 1000   #2500 #1000  #2500   #1000   #200
    partition_size_x = 1000   #2500 #1000  #2500   #1000   #300
    
    # =====================================================
    # PATHS
    # =====================================================
    dynamic_data_folder = "/projects/prjs1930/pilot_data"
    static_data_folder = "/projects/prjs1930/pilot_data/DEM"
    mask_file = "/projects/prjs1930/pilot_data/inference_mask.tif"
    model_folder = "hybrid_models"      #"models_0" #"models_20260510" #"models_0"
    

    # =====================================================
    # SLURM SETUP
    # =====================================================
#    n_tasks =   16 #8 # n_tasks = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
#    rank =  12 #6 # int(os.environ["SLURM_ARRAY_TASK_ID"])
    n_tasks = int(os.environ["SLURM_ARRAY_TASK_COUNT"])   # 16 #8
    rank =   int(os.environ["SLURM_ARRAY_TASK_ID"]) #     12 #6
    
    
#    n_cols = int(np.sqrt(n_tasks))
#    n_rows = int(np.ceil(n_tasks / n_cols))

    # print("get_num_threads: ", torch.get_num_threads())
    # print("get_num_interop_threads: ", torch.get_num_interop_threads())

    # def debug_mem(tag=None):
    #     try:
    #         p = psutil.Process(os.getpid())
    #         mem = p.memory_info().rss / (1024 ** 2)
    #         vm = psutil.virtual_memory()
    #         print(f"[MEM] {tag or ''} RSS={mem:.1f}MB, VM%={vm.percent}%")
    #     except Exception as e:
    #         print("[MEM] psutil not available:", e)

    # debug_mem("after thread setup")

    n_cols = n_tasks
    n_rows = 1

    row_id = rank // n_cols
    col_id = rank % n_cols
    
    # ----- uniform band witdth for tasks -----

#    row_blocks = np.array_split(np.arange(H), n_rows)
#    col_blocks = np.array_split(np.arange(W), n_cols)
#
#    H_start = row_blocks[row_id][0]
#    H_end   = row_blocks[row_id][-1] + 1
#
#    W_start = col_blocks[col_id][0]
#    W_end   = col_blocks[col_id][-1] + 1
#
#    H_task = H_end - H_start
#    W_task = W_end - W_start
    
    # ----- End: uniform band witdth for tasks -----
    
    # ----- adjustable band witdth for tasks -----
    
    col_blocks = create_balanced_col_blocks(
    mask_file,
    n_tasks)

    W_start, W_end = col_blocks[rank]
    
    H_start = 0
    H_end = H

    H_task = H_end - H_start
    W_task = W_end - W_start
    
    # ----- End adjustable band withdth for taks -----  
    

#    print(f"Task window rows: {H_start}:{H_end} (H_task={H_task}), cols: {W_start}:{W_end} (W_task={W_task})")
#    print(f"Partition size: y={partition_size_y}, x={partition_size_x}")
    
    
    with rasterio.open(mask_file) as src:
        mask = src.read(1)
        
    print(
    "rank:", rank,
    "W range:", W_start, W_end,
    "pixels:", H_task*W_task,
    "valid pixels:",
    np.sum(mask[H_start:H_end, W_start:W_end])
    )


    scalers_path_model_folder = Path(model_folder)

    dynamic_vars = ["pr", "rsds", "tas", "era5_swe"]
    static_vars = ["elevation_mn", "slope_mn", "eastness_mn", "northness_mn", "tpi_mn"]

    target_var = "swe"

    all_vars = dynamic_vars + static_vars
    num_features = len(all_vars) + 1
    
    # save states 
    checkpoint_interval_days = 365  # checkpoint once per year
    checkpoint_dir = os.path.join(dynamic_data_folder, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_r{rank}.pt")


    # =====================================================
    # MODEL
    # =====================================================
    
    hidden_size = 128 #64    
    num_layers = 1    #2    

    lstm_model = LSTM_MultiLayer(
        input_size=num_features,
        hidden_size=hidden_size,
        num_layers=num_layers
    )
    
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
    # WINDOW (CRITICAL)
    # =====================================================
    window = Window(
        col_off=W_start,
        row_off=H_start,
        width=W_task,
        height=H_task
    )

    # =====================================================
    # MASK (LOCAL ONLY)
    # =====================================================
    with rasterio.open(mask_file) as src:
        local_mask = src.read(1, window=window).astype(bool)

    # print("local_mask shape:", local_mask.shape, "sum(valid)=", np.count_nonzero(local_mask))
    # print("local_mask nbytes:", local_mask.nbytes)
    # debug_mem("after mask read")

    # =====================================================
    # STATIC DATA (READ ONLY ONCE PER NODE)
    # =====================================================
    static_maps = []

    NODATA_VALUE = -9999.0 #-32768
    profile = None

    for var in static_vars:

        path = os.path.join(static_data_folder, f"{var}.tif")

        with rasterio.open(path) as src:
            arr = src.read(1, window=window)
            arr = arr*src.scales[0] + src.offsets[0]  # apply scale and offset
            profile = src.profile.copy()
            original_transform = profile["transform"]
            # if src.nodata is not None:
            #     NODATA_VALUE = src.nodata

        static_maps.append(arr)

    static_stacked = np.stack(static_maps, axis=0)

    # print("static_stacked shape:", static_stacked.shape, "nbytes:", static_stacked.nbytes)
    # debug_mem("after reading static")

    # =====================================================
    # STATE ARRAYS
    # =====================================================
    swe_prev = np.full((H_task, W_task), NODATA_VALUE, dtype=np.float32) # np.full((H_task, W_task), np.nan, dtype=np.float32)
    swe_prev[local_mask] = trans_zero_swe

    # Print the NODATA_VALUE and transformed zero for debugging
    # print("transformed zero SWE:", trans_zero_swe)
    # print("swe_prev initialized shape:", swe_prev.shape, "nbytes:", swe_prev.nbytes)
    # print("swe_prev sample min,max:", float(np.nanmin(swe_prev)), float(np.nanmax(swe_prev)))
    # debug_mem("after swe_prev init")

    # Inverse transorm swe_prev to check the initial values in original scale
#    inv_swe_prev = np.full_like(swe_prev, NODATA_VALUE) #np.full_like(swe_prev, np.nan)
#    valid = ~np.isnan(swe_prev)
#    inv_swe_prev[valid] = scalers.inverse_transform_target(swe_prev[valid])
#    print("inverse transformed swe_prev sample min,max:", float(np.nanmin(inv_swe_prev)), float(np.nanmax(inv_swe_prev)))

    num_blocks_y = (H_task + partition_size_y - 1) // partition_size_y
    num_blocks_x = (W_task + partition_size_x - 1) // partition_size_x

    hidden_states = [None] * (num_blocks_y * num_blocks_x)

    # print("num_blocks_y, num_blocks_x:", num_blocks_y, num_blocks_x)
    # print("hidden_states length:", len(hidden_states))
    # debug_mem("before time loop")

    # =====================================================
    # TIME LOOP
    # =====================================================
    warmup_date = datetime(1979, 10, 1)
    start_date = datetime(1980, 10, 1)   #(1979, 10, 3)    #(1979, 10, 3) (1980, 10, 1)
    end_date = datetime(1981, 12, 1)       #(1982, 10, 1)   #(1979, 10, 5)   #(1979, 10, 5) (1982, 10, 1)

    current_date = warmup_date
    
    # save states
    if os.path.exists(checkpoint_file):
        print(f"Rank {rank}: found checkpoint at {checkpoint_file}, attempting to resume...")
        try:
            ckpt = torch.load(checkpoint_file, map_location="cpu")
            current_date = ckpt["current_date"] + timedelta(days=1)
            swe_prev = ckpt["swe_prev"]
            hidden_states = ckpt["hidden_states"]
            print(f"Rank {rank}: resuming simulation from {current_date.strftime('%d_%m_%Y')} "
                  f"(last checkpointed day: {ckpt['current_date'].strftime('%d_%m_%Y')})")
        except Exception as e:
            print(f"Rank {rank}: failed to load checkpoint ({e}); starting from {current_date.strftime('%d_%m_%Y')} instead")
    else:
        print(f"Rank {rank}: no checkpoint found, starting simulation from {current_date.strftime('%d_%m_%Y')}")


    while current_date <= end_date:

        day_str = current_date.strftime("%d_%m_%Y")

        print(f"Rank {rank} processing {day_str}")

        # debug_mem(f"start day {day_str}")

        # =================================================
        # DYNAMIC VARIABLES (WINDOW READ)
        # =================================================
        dynamic_maps = []

        for var in dynamic_vars:

            if var == "era5_swe":
                filename = f"sd_ERA5-Land_{day_str}.tif"
                path = os.path.join(dynamic_data_folder, "sd", filename)
        
            else:
                filename = f"CHELSA_{var}_{day_str}_V.2.1.tif"
                path = os.path.join(dynamic_data_folder, var, filename)
        
            with rasterio.open(path) as src:
                arr = src.read(1, window=window)
                scale_factor = src.scales[0] * (1000 if var == "era5_swe" else 1) # convert to era5 to mm
                arr = arr*scale_factor + src.offsets[0]  # apply scale and offset


            dynamic_maps.append(arr)

        dynamic_stacked = np.stack(dynamic_maps, axis=0)
        target_stacked = swe_prev[np.newaxis, :, :]

        # if current_date == warmup_date:
        #     print("dynamic_stacked shape:", dynamic_stacked.shape, "nbytes:", dynamic_stacked.nbytes)
        #     print("target_stacked shape:", target_stacked.shape)
        #     debug_mem("after dynamic stack (first day)")

        # =================================================
        # TILE LOOP
        # =================================================
        for i in range(0, H_task, partition_size_y):

            row_end = min(i + partition_size_y, H_task)

            for j in range(0, W_task, partition_size_x):

                col_end = min(j + partition_size_x, W_task)

                tile_mask = local_mask[i:row_end, j:col_end]

                if not np.any(tile_mask):
                    continue

                valid_idx = np.where(tile_mask.ravel())[0]

                # -----------------------------------------
                # TILE DATA (ALREADY LOCAL)
                # -----------------------------------------
                dyn = dynamic_stacked[:, i:row_end, j:col_end]
                stat = static_stacked[:, i:row_end, j:col_end]
                targ = target_stacked[:, i:row_end, j:col_end]

                ph, pw = tile_mask.shape

                dyn = dyn.reshape(len(dynamic_vars), -1).T
                stat = stat.reshape(len(static_vars), -1).T
                targ = targ.reshape(1, -1).T

                dyn = dyn[valid_idx]
                stat = stat[valid_idx]
                targ = targ[valid_idx]

                # -----------------------------------------
                # SCALE
                # -----------------------------------------
                df_dyn = pd.DataFrame(dyn, columns=dynamic_vars)
                df_stat = pd.DataFrame(stat, columns=static_vars)

                dyn_s = scalers.transform_dataframe(df_dyn, "dynamic")
                stat_s = scalers.transform_dataframe(df_stat, "static")

                batch = len(valid_idx)

                x = np.empty((batch, num_features), dtype=np.float32)
#                x[:, 0] = targ.ravel()
#                x[:, 1:1+len(dynamic_vars)] = dyn_s
#                x[:, 1+len(dynamic_vars):] = stat_s
                
                x[:, 0:len(dynamic_vars)] = dyn_s
                x[:, len(dynamic_vars):-1] = stat_s
                x[:, -1] = targ.ravel()

                x = torch.from_numpy(x).reshape(batch, 1, num_features)

                idx = (i // partition_size_y) * num_blocks_x + (j // partition_size_x)

                prev_hidden = hidden_states[idx]

                with torch.no_grad():
                    pred, h = lstm_model(
                        x,
                        min_val=trans_zero_swe,
                        hidden=prev_hidden
                    )

                hidden_states[idx] = (h[0].detach(), h[1].detach())

                pred = pred.cpu().numpy().ravel()

                # if hasattr(h[0], 'shape'):
                #     print("hidden h0 shape:", tuple(h[0].shape), "h1 shape:", tuple(h[1].shape))

                out_tile = np.full(ph * pw, np.nan, dtype=np.float32)
                out_tile[valid_idx] = pred
                out_tile = out_tile.reshape(ph, pw)

                swe_prev[i:row_end, j:col_end] = out_tile

                # print("updated swe_prev region min,max:", float(np.nanmin(swe_prev[i:row_end, j:col_end])), float(np.nanmax(swe_prev[i:row_end, j:col_end])))


        # =================================================
        # WRITE OUTPUT
        # =================================================
        if current_date >= start_date:

            inv = np.full((H_task, W_task), NODATA_VALUE, dtype=np.float32)

            valid = swe_prev != NODATA_VALUE      # valid = ~np.isnan(swe_prev)
            inv[valid] = scalers.inverse_transform_target(swe_prev[valid])

#            print("output stats: valid pixels", np.count_nonzero(valid), " min,max:", float(np.nanmin(inv[valid])) if np.count_nonzero(valid) else None, float(np.nanmax(inv[valid])) if np.count_nonzero(valid) else None)

            out_dir = os.path.join(dynamic_data_folder, target_var)
            os.makedirs(out_dir, exist_ok=True)

            out_file = os.path.join(out_dir, f"SWE_{day_str}_r{rank}.tif")

            transform = rasterio.windows.transform(window, original_transform)

            profile.update(
                height=H_task,
                width=W_task,
                transform=transform,
                dtype=rasterio.float32,
                count=1,
                nodata=NODATA_VALUE
            )

            with rasterio.open(out_file, "w", **profile) as dst:
                dst.write(inv, 1)

            try:
                fsize = os.path.getsize(out_file) / (1024 ** 2)
                print(f"Wrote output {out_file}, size={fsize:.2f} MB")
            except Exception:
                print("Wrote output", out_file)
                
        # Save checkpoint every checkpoint_interval_days
        days_since_warmup = (current_date - warmup_date).days

        if days_since_warmup > 0 and days_since_warmup % checkpoint_interval_days == 0:
            checkpoint_tmp_file = checkpoint_file + ".tmp"
            torch.save(
                {
                    "current_date": current_date,
                    "swe_prev": swe_prev,
                    "hidden_states": hidden_states,
                },
                checkpoint_tmp_file,
            )
            os.replace(checkpoint_tmp_file, checkpoint_file)
            print(f"Rank {rank}: saved checkpoint at {day_str} -> {checkpoint_file}")

        current_date += timedelta(days=1)
        
    # Remove checkpoint file after successful completion
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"Rank {rank}: simulation completed, removed checkpoint {checkpoint_file}")
