# raster_to_pytorch

A Python package for reading `.tif` raster maps, preprocessing them, and preparing data for LSTM models. This repository provides tools for loading raster maps, handling missing data, cropping regions of interest, and performing inference using trained LSTM models.

---

## Features

- **Raster reading**: Load `.tif` files as NumPy or PyTorch tensors.
- **Data cleaning**: Interpolate missing or invalid values.
- **Cropping**: Extract specific regions from larger raster maps.
- **LSTM data preparation**: Convert raster sequences into time-series datasets suitable for LSTM models.
- **Inference scripts**: Run predictions on new raster maps using trained models.

---