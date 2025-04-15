import pandas as pd
import numpy as np
import os

def create_mock_data():
    """Create mock train and test data for testing."""
    # Mock training data
    np.random.seed(42)
    train_data = {
        'SK_ID_CURR': [100001, 100002, 100003, 100004, 100005],
        'TARGET': [0, 1, 0, 1, 0],
        'EXT_SOURCE_1': [0.5, 0.6, np.nan, 0.8, 0.9],  # Some missing values
        'EXT_SOURCE_2': [0.7, 0.7, 0.7, 0.7, 0.7],  # Low variance
        'EXT_SOURCE_3': [0.3, 0.4, 0.5, 0.6, 0.7],  # Correlated with EXT_SOURCE_1
        'NAME_EDUCATION_TYPE_Higher education': [1, 0, 1, 0, 1]  # Feature with spaces
    }
    train_df = pd.DataFrame(train_data)

    # Mock test data
    test_data = {
        'SK_ID_CURR': [200001, 200002, 200003, 200004, 200005],
        'EXT_SOURCE_1': [0.4, 0.5, 0.6, 0.7, 0.8],
        'EXT_SOURCE_2': [0.7, 0.7, 0.7, 0.7, 0.7],
        'EXT_SOURCE_3': [0.2, 0.3, 0.4, 0.5, 0.6],
        'NAME_EDUCATION_TYPE_Higher education': [0, 1, 0, 1, 0]
    }
    test_df = pd.DataFrame(test_data)

    # Create output directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    processed_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # Save mock data
    train_df.to_pickle(os.path.join(processed_dir, "train_final_mock.pkl"))
    test_df.to_pickle(os.path.join(processed_dir, "test_final_mock.pkl"))
    print("Mock data created successfully!")

if __name__ == "__main__":
    create_mock_data()