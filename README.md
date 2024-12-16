# l2 wind direction: README

This repository provides tools for predicting wind direction and estimating uncertainty from SAR images acquired by Sentinel-1 (S1), RADARSAT Constellation Mission (RCM), and RADARSAT-2 (RS2) satellites. It utilizes a trained machine learning model built with PyTorch Lightning and Hydra-Zen

# Features

- Prediction Pipeline: Load trained models and make predictions on new data.

- Hydra-Zen Integration: Flexible configuration management.

- Output Options: Generate predictions as a pandas DataFrame or an xarray Dataset.

- Uncertainty Estimation: Predict both wind direction and associated uncertainty.

- Georeferencing Support: Adjust predictions for geographical reference using dataset metadata.

# Installation

## Clone the repository:
```bash
git clone https://github.com/jean2262/l2winddir.git
cd l2winddir
```

# Usage

## Command-Line Interface

Run predictions using the predict.py script:

```bash
python predict.py --model_path <path_to_model> --data_path <path_to_data> --eval <True/False>
```

- Arguments:

    - ```--model_path```: Path to the trained model's directory.

    - ```--data_path```: Path to the data file (NetCDF or xarray-compatible format).

    - ```--eval```: If True, outputs a pandas DataFrame; otherwise, modifies and returns the input xarray Dataset.

## Example

### Command-Line
```bash
python predict.py --model_path "/path/to/model" --data_path "/path/to/dataset.nc" --eval True
```

### Programmatic Usage

You can use the make_prediction function directly in Python scripts:
```python
from predict import make_prediction

model_path = "/path/to/model"
data_path = "/path/to/dataset.nc"
result = make_prediction(model_path=model_path, data_path=data_path, eval=True)

print(result)
```

# License

This project is licensed under the MIT License. See the LICENSE file for details.