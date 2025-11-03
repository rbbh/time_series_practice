class DataLoader:
    """
    A class for loading multiple time series datasets.
    """

    def __init__(self):
        pass

    def load_csv(self, file_path):
        """
        Load a time series dataset from a CSV file.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            pandas.DataFrame: Loaded dataset.
        """
        import pandas as pd
        return pd.read_csv(file_path)

    def load_excel(self, file_path, sheet_name=0):
        """
        Load a time series dataset from an Excel file.

        Args:
            file_path (str): Path to the Excel file.
            sheet_name (int or str): Sheet name or index to load.

        Returns:
            pandas.DataFrame: Loaded dataset.
        """
        import pandas as pd
        return pd.read_excel(file_path, sheet_name=sheet_name)

    def load_from_database(self, connection_string, query):
        """
        Load a time series dataset from a database.

        Args:
            connection_string (str): Database connection string.
            query (str): SQL query to fetch the data.

        Returns:
            pandas.DataFrame: Loaded dataset.
        """
        import pandas as pd
        from sqlalchemy import create_engine
        engine = create_engine(connection_string)
        return pd.read_sql(query, engine)