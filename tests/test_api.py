import requests
import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
test_df = os.path.join(project_root, "data/features/test_selected.pkl")
test_df = pd.read_pickle(test_df)
sample_row = test_df.iloc[22].to_dict()  # Include SK_ID_CURR in the input
print(sample_row)

response = requests.post("http://localhost:8000/predict/", json=sample_row)
json_response = response.json()

if json_response['probability'] > 0.5:
    prediction = "Default"
else:
    prediction = "No Default"

print(prediction)