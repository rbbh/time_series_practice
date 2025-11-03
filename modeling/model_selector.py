class ModelSelector:
    """
    A class for selecting and initializing time series forecasting models, including ARIMA and other ML models.
    """

    def __init__(self):
        pass

    def arima_model(self, data, order):
        """
        Fit an ARIMA model to the data.

        Args:
            data (pandas.Series): The time series data.
            order (tuple): The (p, d, q) order of the ARIMA model.

        Returns:
            statsmodels.tsa.arima.model.ARIMAResults: Fitted ARIMA model.
        """
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(data, order=order)
        return model.fit()

    def integrate_r_arima(self, data, order):
        """
        Use R's ARIMA implementation via rpy2.

        Args:
            data (pandas.Series): The time series data.
            order (tuple): The (p, d, q) order of the ARIMA model.

        Returns:
            rpy2.robjects.vectors.ListVector: Fitted ARIMA model from R.
        """
        from rpy2.robjects import r, pandas2ri
        pandas2ri.activate()
        r_data = pandas2ri.py2rpy(data)
        r_arima = r['forecast::auto.arima']
        return r_arima(r_data, order=order)

    def other_ml_model(self, model_name, **kwargs):
        """
        Initialize other machine learning models for time series forecasting.

        Args:
            model_name (str): Name of the ML model (e.g., 'RandomForest').
            kwargs: Additional parameters for the model.

        Returns:
            object: Initialized ML model.
        """
        if model_name == 'RandomForest':
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(**kwargs)
        else:
            raise ValueError(f"Model {model_name} is not supported.")