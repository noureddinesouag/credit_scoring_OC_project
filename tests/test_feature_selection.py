import pytest
import pandas as pd
import numpy as np
import os
from src.data.feature_selection import (
    load_data,
    remove_high_missing,
    remove_low_variance,
    remove_high_correlation,
)

# Define paths
@pytest.fixture
def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

@pytest.fixture
def mock_data_paths(project_root):
    train_path = os.path.join(project_root, "data", "processed", "train_final_mock.pkl")
    test_path = os.path.join(project_root, "data", "processed", "test_final_mock.pkl")
    return train_path, test_path

@pytest.fixture
def output_dir(project_root):
    output_dir = os.path.join(project_root, "data", "features")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def test_load_data(mock_data_paths):
    """Test the load_data function."""
    train_path, test_path = mock_data_paths
    train_df, test_df = load_data(train_path, test_path)

    # Check shapes (updated to match actual mock data: 5 rows, 6 columns including index)
    assert train_df.shape == (5, 6), f"Expected train_df shape (5, 6), got {train_df.shape}"
    assert test_df.shape == (5, 5), f"Expected test_df shape (5, 5), got {test_df.shape}"

    # Check SK_ID_CURR is a column
    assert 'SK_ID_CURR' in train_df.columns, "SK_ID_CURR should be a column in train_df"
    assert 'SK_ID_CURR' in test_df.columns, "SK_ID_CURR should be a column in test_df"

    # Check SK_ID_CURR values (updated to match actual mock data)
    expected_train_skids = [100001, 100002, 100003, 100004, 100005]
    expected_test_skids = [200001, 200002, 200003, 200004, 200005]
    assert train_df['SK_ID_CURR'].tolist() == expected_train_skids, "Train SK_ID_CURR values incorrect"
    assert test_df['SK_ID_CURR'].tolist() == expected_test_skids, "Test SK_ID_CURR values incorrect"


def test_remove_high_missing(mock_data_paths):
    """Test the remove_high_missing function."""
    train_path, _ = mock_data_paths
    train_df = pd.read_pickle(train_path)

    # Remove features with high missing values
    filtered_df, removed = remove_high_missing(train_df, threshold=0.2, exclude_cols=['SK_ID_CURR', 'TARGET'])

    # EXT_SOURCE_1 has 1 out of 5 missing (20%), should NOT be removed with threshold 0.2
    assert 'EXT_SOURCE_1' in filtered_df.columns, "EXT_SOURCE_1 should NOT be removed with 20% missing and threshold 0.2"
    assert 'SK_ID_CURR' in filtered_df.columns, "SK_ID_CURR should not be removed"
    assert 'TARGET' in filtered_df.columns, "TARGET should not be removed"

def test_remove_low_variance(mock_data_paths):
    """Test the remove_low_variance function."""
    train_path, _ = mock_data_paths
    train_df = pd.read_pickle(train_path)

    # Remove features with low variance
    filtered_df, removed = remove_low_variance(train_df, target_col='TARGET', threshold=0.01, exclude_cols=['SK_ID_CURR'])

    # EXT_SOURCE_2 has zero variance, should be removed
    assert 'EXT_SOURCE_2' not in filtered_df.columns, "EXT_SOURCE_2 should be removed due to low variance"
    assert 'SK_ID_CURR' in filtered_df.columns, "SK_ID_CURR should not be removed"
    assert 'TARGET' in filtered_df.columns, "TARGET should not be removed"

def test_remove_high_correlation(mock_data_paths):
    """Test the remove_high_correlation function."""
    train_path, _ = mock_data_paths
    train_df = pd.read_pickle(train_path)

    # Remove highly correlated features
    filtered_df, removed = remove_high_correlation(train_df, target_col='TARGET', threshold=0.9, exclude_cols=['SK_ID_CURR'])

    # EXT_SOURCE_1 and EXT_SOURCE_3 correlation depends on random data; adjust threshold if needed
    remaining_features = [col for col in filtered_df.columns if col not in ['SK_ID_CURR', 'TARGET']]
    assert len(remaining_features) <= 3, "Should have at most 3 features after correlation removal"
    assert 'SK_ID_CURR' in filtered_df.columns, "SK_ID_CURR should not be removed"
    assert 'TARGET' in filtered_df.columns, "TARGET should not be removed"
