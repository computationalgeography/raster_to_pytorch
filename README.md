# raster_to_pytorch

This repository provides inference scripts for running a trained hybrid machine-learning model based on a Long Short-Term Memory (LSTM) neural network to predict snow water equivalent (SWE) across the Northern Hemisphere.

It also provides Python utilities for reading and preprocessing raster maps in GeoTIFF (`.tif`) format for machine-learning applications.

The Hybrid SWE model uses high-resolution meteorological and topographical data, together with ERA5-Land SWE, as predictor variables to generate daily SWE predictions.

The inference workflow supports parallel execution on high-performance computing (HPC) clusters. To reduce memory requirements, large raster domains are divided into blocks and processed in smaller partitions.

## Features

- **Raster reading:** Load GeoTIFF raster files into NumPy arrays or PyTorch tensors.
- **Data preprocessing:** Handle missing or invalid raster values and interpolate missing data where required.
- **Spatial cropping:** Extract specific regions of interest from larger raster maps.
- **LSTM data preparation:** Prepare raster sequences as time-series inputs for LSTM-based models.
- **Hybrid SWE inference:** Generate SWE predictions using a trained Hybrid SWE model.
- **Memory-efficient processing:** Divide raster blocks into smaller partitions to reduce memory consumption during inference.
- **Parallel processing:** Distribute raster processing across multiple tasks and compute nodes on an HPC cluster.
- **GeoTIFF output:** Save predicted SWE values as raster files.

## Requirements

Running the inference workflow requires Python and the libraries used by the raster-processing utilities and trained model.

The required dependencies include:

- NumPy
- Rasterio and/or GDAL
- PyTorch

Ensure that the required packages and compatible versions are installed before running the inference script.

## Input Data

The inference script reads the predictor variables required by the trained Hybrid SWE model from GeoTIFF (`.tif`) raster files.

The input data include high-resolution meteorological and topographical variables, as well as ERA5-Land SWE.

Ensure that all required input variables are available and that the raster files meet the spatial and temporal requirements of the trained model.

## Trained Model

The trained Hybrid SWE model is required to perform inference.

Ensure that the model files and any associated configuration or preprocessing parameters are available before starting the inference workflow.

If the trained model is stored separately from this repository, download or copy it to the appropriate location and update the model path in the inference configuration.

The model architecture, input variable ordering, preprocessing steps, and other settings must be consistent with those used during training.

## Running Inference

Configure the paths to the input raster data, trained model, and output directory in the inference script or its configuration files.

Run the inference script using the following command:

```bash
python script/inference_multi_node_hybrid_model_save_states.py
```

The script reads the predictor rasters, prepares the inputs required by the model, performs inference, and writes the predicted SWE values to GeoTIFF output files.

## Parallel Processing and Memory Management

The inference workflow supports parallel execution on an HPC cluster. The raster domain is divided into blocks, which are assigned to individual processing tasks.

To reduce memory consumption, each block is further divided into smaller partitions. The model performs inference on these partitions, allowing large raster datasets to be processed without loading the entire domain into memory at once.

The block size, partition size, and parallel execution settings should be configured according to the available computational resources and the requirements of the inference workflow.

## Running on an HPC Cluster with Slurm

For large-scale inference, the workflow can be submitted to an HPC cluster using Slurm.

### 1. Configure the Slurm script

Before submitting the job, review the Slurm submission script and update the settings as required. These include:

- The number of compute nodes and CPUs per task.
- Memory allocation and wall-clock time.
- Paths to the trained model, input data, and output directory.
- Parallel processing configuration.

Ensure that the required software environment and dependencies are available on the compute nodes.

### 2. Submit the job

Submit the Slurm script using the following command:

```bash
sbatch slurm_inference_multi_node_array_saving_states_final.slurm
```

Monitor the job status and inspect the Slurm output and error logs to check the progress of the inference workflow.

## Output Data

The inference script saves the predicted SWE values as GeoTIFF (`.tif`) raster files.

Each processing block is saved as a separate GeoTIFF file, with a corresponding identifier assigned to the block. These files contain the predicted SWE values for their respective spatial regions.

**The complete SWE map must be reconstructed by merging the output raster files from all processing blocks.** Ensure that the blocks are correctly aligned and that their spatial reference information is preserved during merging.