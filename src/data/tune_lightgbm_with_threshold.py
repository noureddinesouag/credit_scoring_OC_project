from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer
import os
import mlflow
import mlflow.lightgbm
import lightgbm as lgb
from utils import business_cost, business_scorer
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings
import joblib
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Suppress feature name warnings
warnings.filterwarnings("ignore", category=UserWarning)

# File paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
folder_root = os.path.abspath(os.path.join(current_dir, "../"))
train_path = os.path.join(project_root, "data/features/train_selected.pkl")
test_path = os.path.join(project_root, "data/features/test_selected.pkl")
output_dir = os.path.join(project_root, "data/predictions")
models_dir = os.path.join(folder_root, "models")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Load training data
if not os.path.exists(train_path):
    raise FileNotFoundError(f"File not found: {train_path}")
train_df = pd.read_pickle(train_path)
logger.info(f"Loaded train_df with shape: {train_df.shape}")

# Exclude both TARGET and SK_ID_CURR from features
X = train_df.drop(columns=['TARGET', 'SK_ID_CURR'])
y = train_df['TARGET']
logger.info(f"Training features: {list(X.columns)}")

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Set up cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define and train the model
model = lgb.LGBMClassifier(class_weight='balanced', n_jobs=-1)
model.fit(X_scaled, y, feature_name=X.columns.tolist())
logger.info(f"Model trained with features: {model.feature_name_}")

# Evaluate with default threshold (0.5)
auc_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
business_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring=business_scorer)
logger.info(f"Default LightGBM - AUC: {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")
logger.info(f"Default LightGBM - Business Cost (threshold 0.5): {-business_scores.mean():.4f}")

# Threshold optimization
thresholds = np.arange(0.2, 0.51, 0.05)
best_threshold = 0.5
best_cost = business_scores.mean()

logger.info("Optimizing threshold...")
for threshold in thresholds:
    scorer = make_scorer(business_cost, needs_proba=True, greater_is_better=False, threshold=threshold)
    scores = cross_val_score(model, X_scaled, y, cv=cv, scoring=scorer)
    mean_cost = scores.mean()
    logger.info(f"Threshold {threshold:.2f}: Business Cost = {-mean_cost:.4f}")
    if mean_cost > best_cost:  # Less negative is better
        best_cost = mean_cost
        best_threshold = threshold

logger.info(f"Optimal threshold: {best_threshold:.2f}")
logger.info(f"Optimized business cost: {-best_cost:.4f}")

# Save model, scaler, and threshold locally
model_path = os.path.join(models_dir, "lightgbm_model.pkl")
scaler_path = os.path.join(models_dir, "scaler.pkl")
threshold_path = os.path.join(models_dir, "best_threshold.txt")
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
with open(threshold_path, 'w') as f:
    f.write(str(best_threshold))
logger.info(f"Model saved to '{model_path}'.")
logger.info(f"Scaler saved to '{scaler_path}'.")
logger.info(f"Threshold saved to '{threshold_path}'.")

# Load and predict initial probabilities on test set
if os.path.exists(test_path):
    test_df = pd.read_pickle(test_path)
    logger.info(f"Loaded test_df with shape: {test_df.shape}")
    logger.info(f"Test features: {list(test_df.columns)}")
    
    # Ensure test features match training features (excluding SK_ID_CURR)
    test_features = test_df.drop(columns=['SK_ID_CURR'], errors='ignore')
    if set(test_features.columns) != set(X.columns):
        missing_in_test = set(X.columns) - set(test_features.columns)
        extra_in_test = set(test_features.columns) - set(X.columns)
        raise ValueError(f"Feature mismatch: Missing in test: {missing_in_test}, Extra in test: {extra_in_test}")
    
    X_test_scaled = scaler.transform(test_features)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Create output DataFrame with SK_ID_CURR and predictions
    output_df = pd.DataFrame({'SK_ID_CURR': test_df['SK_ID_CURR'], 'TARGET_proba': y_pred_proba})
    output_path = os.path.join(output_dir, "test_predictions_initial.csv")
    output_df.to_csv(output_path, index=False)
    logger.info(f"Initial test predictions saved to '{output_path}'.")
else:
    logger.info("Test file not found; skipping test prediction.")

# Log with MLflow
with mlflow.start_run(run_name="LightGBM_balanced_with_threshold"):
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("n_jobs", -1)
    mlflow.log_param("optimal_threshold", best_threshold)
    mlflow.log_metric("business_cost_default", business_scores.mean())
    mlflow.log_metric("business_cost_optimized", best_cost)
    mlflow.log_metric("mean_auc", auc_scores.mean())
    mlflow.log_metric("std_auc", auc_scores.std())
    mlflow.lightgbm.log_model(model, "model", input_example=X_scaled[:5])
    mlflow.sklearn.log_model(scaler, "scaler")