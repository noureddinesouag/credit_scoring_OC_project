from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import os
import time
import logging
import numpy as np
app = FastAPI(title="Credit Risk Prediction API")

# Configure logging
logging.basicConfig(filename='api.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# File paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
model_path = os.path.join(project_root, "src", "models", "lightgbm_model.pkl")
threshold_path = os.path.join(project_root, "src", "models", "best_threshold.txt")
scaler_path = os.path.join(project_root, "src", "models", "scaler.pkl")
train_selected_path = os.path.join(project_root, "data", "features", "train_selected.pkl")

# Feature name mapping to handle inconsistencies
feature_name_mapping = {
    'NAME_EDUCATION_TYPE_Higher education': 'NAME_EDUCATION_TYPE_Higher_education',
    'NAME_EDUCATION_TYPE_Secondary / secondary special': 'NAME_EDUCATION_TYPE_Secondary_/_secondary_special'
}

# Load the training data to get the cleaned feature names
try:
    train_selected = pd.read_pickle(train_selected_path)
    # Exclude TARGET and SK_ID_CURR
    train_selected = train_selected.rename(columns=feature_name_mapping)  # Apply mapping to match model
    EXPECTED_FEATURES = [col for col in train_selected.columns if col not in ['TARGET', 'SK_ID_CURR']]
except Exception as e:
    raise Exception(f"Error loading train_selected.pkl: {e}")

# Load the model, scaler, and threshold
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(threshold_path, 'r') as f:
        threshold = float(f.read())
except Exception as e:
    raise Exception(f"Error loading model, scaler, or threshold: {e}")

# Validate that the model expects the same features
model_features = model.feature_name_
if set(model_features) != set(EXPECTED_FEATURES):
    print("Model's expected features:", model_features)
    print("EXPECTED_FEATURES:", EXPECTED_FEATURES)
    print("Number of model features:", len(model_features))
    print("Number of EXPECTED_FEATURES:", len(EXPECTED_FEATURES))
    raise Exception("Model features do not match expected features from train_selected.pkl")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Credit Risk Prediction API"}

@app.post("/predict/")
async def predict(data: dict):
    start_time = time.time()
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([data])
        # Extract SK_ID_CURR if present
        sk_id_curr = None
        if 'SK_ID_CURR' in input_df.columns:
            sk_id_curr = input_df['SK_ID_CURR'].iloc[0]
            input_df = input_df.drop(columns=['SK_ID_CURR'])
        # Check for missing features (before renaming, using original names)
        missing_features = [f for f in EXPECTED_FEATURES if f not in input_df.columns and f not in feature_name_mapping.values()]
        if missing_features:
            raise HTTPException(status_code=400, detail=f"Missing features: {missing_features}")
        # Check for extra features
        extra_features = [f for f in input_df.columns if f not in EXPECTED_FEATURES and f not in feature_name_mapping]
        if extra_features:
            input_df = input_df.drop(columns=extra_features)
        # Reorder columns to match scaler expectations (original names)
        input_df = input_df[[col for col in scaler.feature_names_in_ if col in input_df.columns]]
        # Scale the input data with original feature names
        input_scaled = scaler.transform(input_df)
        # Create a DataFrame with scaled data and rename features for the model
        input_scaled_df = pd.DataFrame(input_scaled, columns=input_df.columns)
        input_scaled_df = input_scaled_df.rename(columns=feature_name_mapping)
        # Reorder columns to match model expectations (cleaned names)
        input_scaled_df = input_scaled_df[EXPECTED_FEATURES]
        # Make prediction
        proba = model.predict_proba(input_scaled_df)[:, 1]
        prediction = (proba >= threshold).astype(int)
        response_time = time.time() - start_time
        logging.info(f"Prediction successful - Response time: {response_time:.2f}s")
        # Include SK_ID_CURR in the response if it was provided
        response = {
            "probability": float(proba[0]),
            "prediction": int(prediction[0]),
            "threshold": threshold
        }
        if sk_id_curr is not None:
            response["SK_ID_CURR"] = int(sk_id_curr) if isinstance(sk_id_curr, (int, np.integer)) else sk_id_curr
        return response
    except Exception as e:
        response_time = time.time() - start_time
        logging.error(f"Prediction failed - Error: {str(e)} - Response time: {response_time:.2f}s")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)