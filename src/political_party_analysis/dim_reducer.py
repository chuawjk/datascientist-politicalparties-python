import pandas as pd


class DimensionalityReducer:
    """Class to model a dimensionality reduction method for the given dataset.
    1. Write a function to convert the high dimensional data to 2 dimensional.
    """

    def __init__(self, data: pd.DataFrame, n_components: int = 2):
        self.n_components = n_components
        self.data = data

    ##### YOUR CODE GOES HERE #####
    def transform(self):
        # TODO: If a PCA model doesn't exist, create one, fit it and transform the data
        # TODO: Else just transform the data
        # TODO: Print the variance explained ratios to assess goodness of fit
        pass
