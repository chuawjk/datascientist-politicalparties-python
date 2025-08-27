import pandas as pd
from sklearn.mixture import GaussianMixture


class DensityEstimator:
    """Class to estimate Density/Distribution of the given data.
    1. Write a function to model the distribution of the political party dataset
    2. Write a function to randomly sample 10 parties from this distribution
    3. Map the randomly sampled 10 parties back to the original higher dimensional
    space as per the previously used dimensionality reduction technique.
    """

    def __init__(self, data: pd.DataFrame, dim_reducer, high_dim_feature_names, n_components: int = 5):
        self.data = data
        self.dim_reducer_model = dim_reducer.model
        self.feature_names = high_dim_feature_names
        self.n_components = n_components
        self.distribution_model = GaussianMixture(n_components=self.n_components)

    ##### YOUR CODE GOES HERE #####
    def fit(self):
        self.distribution_model.fit(self.data)

    def sample_parties(self, n_samples: int = 10):
        self.sampled_parties = self.distribution_model.sample(n_samples)

    def map_to_high_dim(self):
        reduced_dim_samples, _ = self.sample_parties()
        high_dim_samples = self.dim_reducer_model.inverse_transform(reduced_dim_samples)
        high_dim_samples_df = pd.DataFrame(high_dim_samples, columns=self.feature_names)

    def run_analysis(self):
        self.fit()
        self.sample_parties()
        self.map_to_high_dim()
