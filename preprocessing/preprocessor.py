class Preprocessor:
    """
    A class for preprocessing time series data, including splitting, normalization, and decomposition.
    """

    def __init__(self):
        pass

    def split_data(self, data, train_size=0.8):
        """
        Split the dataset into training and testing sets.

        Args:
            data (pandas.DataFrame): The dataset to split.
            train_size (float): Proportion of the data to use for training.

        Returns:
            tuple: Training and testing datasets.
        """
        train_size = int(len(data) * train_size)
        train, test = data[:train_size], data[train_size:]
        return train, test

    def normalize(self, data):
        """
        Normalize the dataset.

        Args:
            data (pandas.DataFrame): The dataset to normalize.

        Returns:
            pandas.DataFrame: Normalized dataset.
        """
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(data)
        return normalized_data

    def decompose(self, data, model='additive', period=1):
        """
        Decompose the time series data.

        Args:
            data (pandas.Series): The time series data to decompose.
            model (str): Type of decomposition ('additive' or 'multiplicative').
            period (int): Seasonal period.

        Returns:
            statsmodels.tsa.seasonal.DecomposeResult: Decomposition result.
        """
        from statsmodels.tsa.seasonal import seasonal_decompose
        return seasonal_decompose(data, model=model, period=period)