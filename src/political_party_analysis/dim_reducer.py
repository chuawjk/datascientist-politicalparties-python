import pandas as pd
from sklearn.decomposition import PCA


class DimensionalityReducer:
    """Class to model a dimensionality reduction method for the given dataset.
    1. Write a function to convert the high dimensional data to 2 dimensional.
    """

    def __init__(self, model: str, data: pd.DataFrame, n_components: int = 2):
        self.n_components = n_components
        self.data = data
        if model == "PCA":
            self.model = PCA(n_components=self.n_components)
        else:
            raise ValueError(f"Model {model} not supported")

    ##### YOUR CODE GOES HERE #####
    def transform(self) -> pd.DataFrame:
        """
        Transform the data to the specified number of dimensions using PCA.
        """
        self.model.fit(self.data)
        reduced_dim_data = pd.DataFrame(
            self.model.transform(self.data),
            columns=[f"dim_{i}" for i in range(self.n_components)],
        )
        return reduced_dim_data
