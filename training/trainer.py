class Trainer:
    """
    A class for training time series forecasting models.
    """

    def __init__(self):
        pass

    def train(self, model, train_data, **kwargs):
        """
        Train the given model on the training data.

        Args:
            model (object): The model to train.
            train_data (pandas.DataFrame or pandas.Series): The training data.
            kwargs: Additional parameters for the training process.

        Returns:
            object: Trained model.
        """
        if hasattr(model, 'fit'):
            return model.fit(train_data, **kwargs)
        else:
            raise ValueError("The provided model does not have a 'fit' method.")