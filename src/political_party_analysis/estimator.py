from typing import Tuple

import numpy as np
import pandas as pd


class DensityEstimator:
    """Class to estimate Density/Distribution of the given data.
    1. Write a function to model the distribution of the political party dataset
    2. Write a function to randomly sample 10 parties from this distribution
    3. Map the randomly sampled 10 parties back to the original higher dimensional
    space as per the previously used dimensionality reduction technique.
    """

    # NOTE: We should fit a Gaussian Mixture Model. This satisties the following requirements (which we can infer from
    # visualization.py):
    # 1. Put data into several groups
    # 2. Define means and covariances for each group
    # 3. Ability to sample synthetic data from the model

    # TODO: Add arg to specify number of GMM components
    def __init__(self, data: pd.DataFrame, dim_reducer, high_dim_feature_names):
        self.data = data
        self.dim_reducer_model = dim_reducer.model
        self.feature_names = high_dim_feature_names

    ##### YOUR CODE GOES HERE #####
    def model_distribution(self) -> pd.Series:
        # TODO: If a GMM model doesn't exist, create one, fit it and predict the labels
        # TODO: Else just predict the labels
        pass

    def sample_parties(self, n_samples: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        # TODO: If a GMM model doesn't exist, raise an error
        # TODO: Sample from the model, and return the sampled parties and their labels
        pass

    def map_to_high_dim(self, sampled_parties: np.ndarray) -> pd.DataFrame:
        # TODO: if a fitted dimensionality reducer doesn't exist, raise an error
        # TODO: Inverse transform the sampled parties back to the original high dimensional space
        pass
