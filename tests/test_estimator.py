from typing import Tuple
import pytest

import numpy as np
import pandas as pd

from political_party_analysis.dim_reducer import DimensionalityReducer
from political_party_analysis.estimator import DensityEstimator


@pytest.fixture
def mock_dim_reducer_and_data() -> Tuple[DimensionalityReducer, pd.DataFrame, pd.DataFrame]:
    data = pd.DataFrame(
        data={
            "col1": [-1.225, 0, 1.225],
            "col2": [-1.175, -0.1, 1.257],
            "col3": [-1.019, -0.340, 1.359],
        },
        index=[0, 1, 2],
    )
    data.index.name = "id"

    dim_reducer = DimensionalityReducer("PCA", data)
    transformed_data = dim_reducer.transform()

    return dim_reducer, data, transformed_data


def test_model_distribution(mock_dim_reducer_and_data: Tuple[DimensionalityReducer, pd.DataFrame, pd.DataFrame]):
    dim_reducer, data, transformed_data = mock_dim_reducer_and_data
    estimator = DensityEstimator(transformed_data, dim_reducer, data.columns, n_components=2)
    pred_labels = estimator.model_distribution()
    assert len(pred_labels) == data.shape[0]
    assert pred_labels.nunique() == 2


def test_sample_parties(mock_dim_reducer_and_data: Tuple[DimensionalityReducer, pd.DataFrame, pd.DataFrame]):
    dim_reducer, data, transformed_data = mock_dim_reducer_and_data
    estimator = DensityEstimator(transformed_data, dim_reducer, data.columns, n_components=2)
    _ = estimator.model_distribution()
    sampled_parties, sampled_labels = estimator.sample_parties(10)
    assert sampled_parties.shape == (10, 2)
    assert len(np.unique(sampled_labels)) == 2


def test_map_to_high_dim(mock_dim_reducer_and_data: Tuple[DimensionalityReducer, pd.DataFrame, pd.DataFrame]):
    dim_reducer, data, transformed_data = mock_dim_reducer_and_data
    estimator = DensityEstimator(transformed_data, dim_reducer, data.columns, n_components=2)
    _ = estimator.model_distribution()
    sampled_parties, _ = estimator.sample_parties(10)
    high_dim_samples = estimator.map_to_high_dim(sampled_parties)
    assert high_dim_samples.shape == (10, 3)
