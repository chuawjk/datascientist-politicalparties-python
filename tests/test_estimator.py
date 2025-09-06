import pytest

import numpy as np
import pandas as pd

from political_party_analysis.dim_reducer import DimensionalityReducer
from political_party_analysis.estimator import DensityEstimator

# TODO: Initialize some high dimensional data
# TODO: Initialize dimensionality reducer

# TODO: Create fixture that to perform dimensionality reduction
# TODO: Returns dimensionality reducer, high dimensional data, and transformed low dimensional data

# TODO: Write test for model_distribution
# TODO: Check that function returns predicted labels with shape (input_data.shape[0],), and correct number of unique labels

# TODO: Write test for sample_parties
# TODO: Check that first returned value is a numpy array of shape (num_samples, num_reduced_dims)
# TODO: Check that the second returned value is a numpy array of shape (num_groups), with the correct number of unique labels

# TODO: Write test for map_to_high_dim
# TODO: Check that function returns a pandas DataFrame with shape (num_samples, num_high_dim_features)
