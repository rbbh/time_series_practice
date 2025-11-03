class Tester:
    """
    A class for testing and evaluating time series forecasting models.
    """

    def __init__(self):
        pass

    def evaluate(self, model, test_data, metric):
        """
        Evaluate the model on the test data using the specified metric.

        Args:
            model (object): The trained model to evaluate.
            test_data (pandas.DataFrame or pandas.Series): The test data.
            metric (callable): A function to compute the evaluation metric.

        Returns:
            float: Computed metric value.
        """
        if hasattr(model, 'predict'):
            predictions = model.predict(test_data)
            return metric(test_data, predictions)
        else:
            raise ValueError("The provided model does not have a 'predict' method.")