import os
import mlflow
import mlflow.lightgbm
import pandas as pd
import numpy as np
import warnings
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Suppress feature name warnings
warnings.filterwarnings("ignore", category=UserWarning)

# File paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
folder_root = os.path.abspath(os.path.join(current_dir, "../"))
test_path = os.path.join(project_root, "data/features/test_selected.pkl")
output_dir = os.path.join(project_root, "data/predictions")
models_dir = os.path.join(folder_root, "models")
os.makedirs(output_dir, exist_ok=True)

# Load test data
if not os.path.exists(test_path):
    raise FileNotFoundError(f"File not found: {test_path}")
test_df = pd.read_pickle(test_path)
logger.info(f"Loaded test_df with shape: {test_df.shape}")
logger.info(f"Test features: {list(test_df.columns)}")

# Extract SK_ID_CURR (check if it's a column or the index)
if 'SK_ID_CURR' in test_df.columns:
    sk_id_curr = test_df['SK_ID_CURR'].copy()
else:
    sk_id_curr = test_df.index.copy()
    test_df = test_df.reset_index()  # Move SK_ID_CURR from index to column if needed
    if 'SK_ID_CURR' in test_df.columns:
        sk_id_curr = test_df['SK_ID_CURR'].copy()
    else:
        raise ValueError("SK_ID_CURR not found in test_df columns or index")

# Exclude SK_ID_CURR from test features
test_features_df = test_df.drop(columns=['SK_ID_CURR'], errors='ignore')
logger.info(f"Test features after dropping SK_ID_CURR: {list(test_features_df.columns)}")

# Load MLflow runs to get the optimal threshold
runs = mlflow.search_runs(order_by=["metrics.business_cost_optimized DESC"])
if runs.empty:
    raise ValueError("No MLflow runs found.")
valid_runs = runs[runs["metrics.business_cost_optimized"].notna()]
if valid_runs.empty:
    raise ValueError("No MLflow runs found with valid business_cost_optimized metric.")
best_threshold = float(valid_runs.iloc[0]["params.optimal_threshold"])
logger.info(f"Loaded optimal threshold from MLflow: {best_threshold:.2f}")

# Load model and scaler from local files (fallback to MLflow if not found)
model_path = os.path.join(models_dir, "lightgbm_model.pkl")
scaler_path = os.path.join(models_dir, "scaler.pkl")
if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    logger.info(f"Loaded model from '{model_path}'.")
    logger.info(f"Loaded scaler from '{scaler_path}'.")
else:
    logger.info("Local model/scaler not found; attempting MLflow fallback.")
    latest_run_id = valid_runs.iloc[0]["run_id"]
    model_uri = f"runs:/{latest_run_id}/model"
    scaler_uri = f"runs:/{latest_run_id}/scaler"
    model = mlflow.lightgbm.load_model(model_uri)
    scaler = mlflow.sklearn.load_model(scaler_uri)
    logger.info(f"Loaded model from MLflow run '{latest_run_id}'.")
    logger.info(f"Loaded scaler from MLflow run '{latest_run_id}'.")

# Get model's expected feature names
train_features = model.feature_name_
logger.info(f"Training features from model: {train_features}")

# Scale the test data using the original feature names
X_test_scaled = scaler.transform(test_features_df)
logger.info(f"Scaled test data with shape: {X_test_scaled.shape}")

# Apply feature name mapping to match model's expected names
feature_name_mapping = {
    'NAME_EDUCATION_TYPE_Higher education': 'NAME_EDUCATION_TYPE_Higher_education',
    'NAME_EDUCATION_TYPE_Secondary / secondary special': 'NAME_EDUCATION_TYPE_Secondary_/_secondary_special'
}
test_features_df = test_features_df.rename(columns=feature_name_mapping)
logger.info(f"Renamed feature names in test data using explicit mapping")
logger.info(f"Test features after renaming: {list(test_features_df.columns)}")

# Validate feature names (excluding SK_ID_CURR)
test_features = list(test_features_df.columns)
if set(test_features) != set(train_features):
    missing_in_test = set(train_features) - set(test_features)
    extra_in_test = set(test_features) - set(train_features)
    raise ValueError(f"Feature mismatch: Missing in test: {missing_in_test}, Extra in test: {extra_in_test}")

# Predict probabilities and labels
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred_labels = (y_pred_proba >= best_threshold).astype(int)

# Save predictions with probabilities and labels, including SK_ID_CURR
output_df = pd.DataFrame({
    'SK_ID_CURR': test_df.index,
    'TARGET_proba': y_pred_proba,
    'TARGET': y_pred_labels
})
output_path = os.path.join(output_dir, "test_predictions_with_labels.csv")
output_df.to_csv(output_path, index=False)
logger.info(f"Test predictions (proba + labels) saved to '{output_path}'.")