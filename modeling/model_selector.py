import pandas as pd
from rpy2.robjects import r, pandas2ri
import torch.nn as nn


class ModelSelector:
    """
    A class for selecting and initializing time series forecasting models, including ARMA, ARIMA, LSTM, RNN, and GRU models.
    """

    def __init__(self):
        pass

    def arima_model(self, data, order):
        """
        Fit an ARIMA model using R's implementation via rpy2.

        Args:
            data (pandas.Series): The time series data.
            order (tuple): The (p, d, q) order of the ARIMA model.

        Returns:
            rpy2.robjects.vectors.ListVector: Fitted ARIMA model from R.
        """
        pandas2ri.activate()
        r_data = pandas2ri.py2rpy(data)
        r_arima = r['forecast::auto.arima']
        return r_arima(r_data, order=order)

    def arma_model(self, data, order):
        """
        Fit an ARMA model using R's implementation via rpy2.

        Args:
            data (pandas.Series): The time series data.
            order (tuple): The (p, q) order of the ARMA model.

        Returns:
            rpy2.robjects.vectors.ListVector: Fitted ARMA model from R.
        """
        pandas2ri.activate()
        r_data = pandas2ri.py2rpy(data)
        r_arma = r['forecast::Arima']
        return r_arma(r_data, order=(order[0], 0, order[1]))

    def lstm_model(self, input_dim, hidden_dim, output_dim, num_layers):
        """
        Initialize an LSTM model using PyTorch.

        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of hidden units.
            output_dim (int): Number of output features.
            num_layers (int): Number of LSTM layers.

        Returns:
            torch.nn.Module: Initialized LSTM model.
        """

        class LSTM(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
                super(LSTM, self).__init__()
                self.hidden_dim = hidden_dim

                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_dim, output_dim)

            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.fc(out[:, -1, :])
                return out

        return LSTM(input_dim, hidden_dim, output_dim, num_layers)

    def rnn_model(self, input_dim, hidden_dim, output_dim, num_layers):
        """
        Initialize an RNN model using PyTorch.

        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of hidden units.
            output_dim (int): Number of output features.
            num_layers (int): Number of RNN layers.

        Returns:
            torch.nn.Module: Initialized RNN model.
        """

        class RNN(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
                super(RNN, self).__init__()
                self.hidden_dim = hidden_dim

                self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_dim, output_dim)

            def forward(self, x):
                out, _ = self.rnn(x)
                out = self.fc(out[:, -1, :])
                return out

        return RNN(input_dim, hidden_dim, output_dim, num_layers)

    def gru_model(self, input_dim, hidden_dim, output_dim, num_layers):
        """
        Initialize a GRU model using PyTorch.

        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of hidden units.
            output_dim (int): Number of output features.
            num_layers (int): Number of GRU layers.

        Returns:
            torch.nn.Module: Initialized GRU model.
        """

        class GRU(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
                super(GRU, self).__init__()
                self.hidden_dim = hidden_dim

                self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_dim, output_dim)

            def forward(self, x):
                out, _ = self.gru(x)
                out = self.fc(out[:, -1, :])
                return out

        return GRU(input_dim, hidden_dim, output_dim, num_layers)