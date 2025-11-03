# Time Series Forecasting Repository

## Overview
This repository is designed for teaching time series forecasting using Python. It includes tools for:
- Loading multiple time series datasets
- Preprocessing data (splitting, normalization, decomposition)
- Selecting and training models (including ARIMA and other ML models)
- Evaluating model performance

## Features
- **ARIMA Integration**: Includes ARIMA implementation via Python and R integration.
- **Preprocessing**: Tools for data normalization, decomposition, and splitting.
- **Model Selection**: Support for various time series forecasting models, including:
  - **ARMA Model**: A statistical model combining autoregressive and moving average components, suitable for stationary time series.
  - **ARIMA Model**: An extension of the ARMA model that includes differencing to handle non-stationary time series data.
  - **RNN**: A neural network model designed for sequential data, processing one element at a time while maintaining a memory of previous elements.
  - **GRU**: A variant of RNNs that uses gating mechanisms to efficiently capture dependencies in sequential data.
  - **LSTM**: A type of recurrent neural network capable of learning long-term dependencies in sequential data.
- **Training and Testing**: Classes for training and evaluating models.

## Repository Structure
```
.
├── data_loading
│   └── data_loader.py       # Data loading utilities
├── preprocessing
│   └── preprocessor.py      # Preprocessing utilities
├── modeling
│   └── model_selector.py    # Model selection utilities
├── training
│   └── trainer.py           # Training utilities
├── testing
│   └── tester.py            # Testing utilities
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd time_series_practice
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Prerequisites

Before installing the dependencies, ensure that R is installed on your system. You can install R using the following commands:

1. Update the package list:
   ```bash
   sudo apt update
   ```
2. Install R:
   ```bash
   sudo apt install r-base
   ```
3. Verify the installation:
   ```bash
   R --version
   ```

## Usage
1. Use the `DataLoader` class to load your time series data.
2. Preprocess the data using the `Preprocessor` class.
3. Select a model using the `ModelSelector` class.
4. Train the model using the `Trainer` class.
5. Evaluate the model using the `Tester` class.

## Dependencies
- Python 3.8+
- pandas
- numpy
- scikit-learn
- statsmodels
- rpy2
- torch

## License
This project is licensed under the MIT License.