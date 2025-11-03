import pandas as pd
from statsmodels import datasets


class DataLoader:
    """
    A class for loading multiple time series datasets.
    """

    def __init__(self):
        pass

    def load_builtin_dataset(self, dataset_name):
        """
        Load a built-in time series dataset from statsmodels.

        Args:
            dataset_name (str): Name of the dataset to load (e.g., 'airline', 'sunspots').

        Returns:
            pandas.DataFrame or pandas.Series: Loaded dataset.
        """
        if dataset_name not in datasets.__all__:
            raise ValueError(f"Dataset '{dataset_name}' is not available in statsmodels.")

        data = datasets.get_rdataset(dataset_name).data
        return data