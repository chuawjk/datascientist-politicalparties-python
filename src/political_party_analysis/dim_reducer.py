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
        # If there is no existing model, create one and fit it to the data
        if not hasattr(self.model, "explained_variance_ratio_"):
            self.model.fit(self.data)
            print(
                f"Fitted new dimensionality reduction model. Explained variance ratio: {self.model.explained_variance_ratio_.tolist()}"
            )

        reduced_dim_data = pd.DataFrame(
            self.model.transform(self.data),
            columns=[f"dim_{i}" for i in range(self.n_components)],
        )
        return reduced_dim_data
