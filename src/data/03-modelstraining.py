import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import os
import time

# File paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
train_path = os.path.join(project_root, "data/features/train_selected.pkl")

# Load data with selected features
if not os.path.exists(train_path):
    raise FileNotFoundError(f"File not found: {train_path}")
train_df = pd.read_pickle(train_path)
print(f"Loaded train_df with shape: {train_df.shape}")
X = train_df.drop('TARGET', axis=1)
y = train_df['TARGET']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Set up cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define business cost function
def business_cost(y_true, y_pred_proba, threshold=0.5, fn_cost=10, fp_cost=1, **kwargs):
    y_pred = (y_pred_proba >= threshold).astype(int)
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return -(fn * fn_cost + fp * fp_cost)

business_scorer = make_scorer(business_cost, needs_proba=True, greater_is_better=False)

# Train multiple models and track with MLflow
models = {
    'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=1000),
    'RandomForest': RandomForestClassifier(n_estimators=50, class_weight='balanced', n_jobs=-1),  # Reduced n_estimators, added n_jobs
    'LightGBM': lgb.LGBMClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, scale_pos_weight=10, n_jobs=-1)
}

mlflow.set_experiment("credit-scoring")

for model_name, model in models.items():
    print(f"Starting {model_name} evaluation...")
    start_time = time.time()
    with mlflow.start_run(run_name=model_name):
        # Log model parameters
        params = model.get_params()
        for param, value in params.items():
            mlflow.log_param(param, value)
        
        # Evaluate model with cross-validation
        print(f"  Running AUC cross-validation for {model_name}")
        auc_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc', verbose=1)
        print(f"  Running business cost cross-validation for {model_name}")
        business_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring=business_scorer, verbose=1)
        
        # Log metrics
        mlflow.log_metric("mean_auc", np.mean(auc_scores))
        mlflow.log_metric("std_auc", np.std(auc_scores))
        mlflow.log_metric("mean_business_cost", np.mean(business_scores))
        
        # Train on full dataset and log model
        print(f"  Training {model_name} on full dataset")
        model.fit(X_scaled, y)
        if model_name == 'LightGBM':
            mlflow.lightgbm.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")
        
        # Print results
        elapsed_time = time.time() - start_time
        print(f"{model_name} - AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
        print(f"{model_name} - Business Cost: {np.mean(business_scores):.4f}")
        print(f"{model_name} - Completed in {elapsed_time:.2f} seconds")