import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import joblib

# File paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
predictions_path = os.path.join(project_root, "data/predictions/test_predictions_with_labels.csv")
features_path = os.path.join(project_root, "data/features/test_selected.pkl")
model_path = os.path.join(project_root, "src/models/lightgbm_model.pkl")

# Load data
@st.cache_data
def load_data():
    # Load predictions
    predictions = pd.read_csv(predictions_path)
    # st.write("Predictions columns:", predictions.columns.tolist())  # Debugging
    # Load features and reset index to make SK_ID_CURR a column
    features = pd.read_pickle(features_path)
    features = features.reset_index()  # Convert index (SK_ID_CURR) to a column
    # st.write("Features columns after reset_index:", features.columns.tolist())  # Debugging

    # Ensure SK_ID_CURR is in features
    if 'SK_ID_CURR' not in features.columns:
        raise KeyError("SK_ID_CURR not found in features DataFrame after reset_index")

    # Merge predictions with features
    data = features.merge(predictions, on="SK_ID_CURR", how="inner")
    # st.write("Merged data shape:", data.shape)  # Debugging
    # st.write("Merged data columns:", data.columns.tolist())  # Debugging
    return data

@st.cache_resource
def load_model():
    return joblib.load(model_path)

# Sidebar filter
st.sidebar.title("⚙️ Filter Options")
risk_filter = st.sidebar.selectbox(
    "Select risk category to filter:",
    options=["All", "Low Risk (<30%)", "Moderate Risk (30%-70%)", "High Risk (≥70%)"]
)

# Load data with error handling
try:
    data = load_data()
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Apply selected filter
if risk_filter == "Low Risk (<30%)":
    data = data[data["TARGET_proba"] < 0.3]
elif risk_filter == "Moderate Risk (30%-70%)":
    data = data[(data["TARGET_proba"] >= 0.3) & (data["TARGET_proba"] < 0.7)]
elif risk_filter == "High Risk (≥70%)":
    data = data[data["TARGET_proba"] >= 0.7]

# Streamlit app
st.title("Credit Scoring Dashboard")
st.markdown("This dashboard helps customer relationship managers understand client credit risk and explore client information.")

# Client selection
st.header("Select a Client")
client_ids = data["SK_ID_CURR"].astype(str).tolist()

# Check if client_ids is empty
if not client_ids:
    st.error("No clients available to display. Please check the data.")
    st.stop()

# Set the first client as the default selection
selected_client = st.selectbox(
    "Choose a client (SK_ID_CURR):",
    client_ids,
    index=0  # Default to the first client in the list
)

# Ensure selected_client is not None before proceeding
if selected_client is None:
    st.warning("Please select a client to continue.")
    st.stop()

# Convert selected_client to int and filter data
client_data = data[data["SK_ID_CURR"] == int(selected_client)]

# 1. Visualize Score and Interpretation
st.header("Credit Score and Risk Interpretation")
score = client_data["TARGET_proba"].iloc[0] * 100  # Convert to percentage
st.subheader(f"Default Probability: {score:.1f}%")

# Gauge chart for score
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=score,
    title={'text': "Risk of Default (%)"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range': [0, 30], 'color': "green"},
            {'range': [30, 70], 'color': "orange"},
            {'range': [70, 100], 'color': "red"}
        ],
        'threshold': {
            'line': {'color': "black", 'width': 4},
            'value': 50
        }
    }
))
st.plotly_chart(fig)

# Risk interpretation
if score < 30:
    st.success("**Low Risk**: This client is likely to repay the loan.")
elif score < 70:
    st.warning("**Moderate Risk**: This client may require further review.")
else:
    st.error("**High Risk**: This client has a high likelihood of defaulting.")

# 2. Descriptive Information
st.header("Client Descriptive Information")
key_features = ["DAYS_BIRTH", "AMT_INCOME_TOTAL", "AMT_CREDIT", "DAYS_EMPLOYED", "NAME_FAMILY_STATUS_Married", "NAME_EDUCATION_TYPE_Higher education"]
client_info = client_data[key_features].transpose().reset_index()
age = client_data["DAYS_BIRTH"].iloc[0] // 365
income = client_data["AMT_INCOME_TOTAL"].iloc[0]
credit = client_data["AMT_CREDIT"].iloc[0]
employment = client_data["DAYS_EMPLOYED"].iloc[0] // 365

col1, col2, col3, col4 = st.columns(4)
col1.metric("Age (Years)", f"{age}")
col2.metric("Annual Income", f"${income:,.0f}")
col3.metric("Loan Amount", f"${credit:,.0f}")
col4.metric("Years Employed", f"{employment}")

# 3. Comparison with All Clients
st.header("Compare with All Clients")
comparison_feature = st.selectbox("Select a feature to compare:", key_features, index=1)
comparison_value = client_data[comparison_feature].iloc[0]
all_values = data[comparison_feature]

# Histogram for comparison
fig = px.histogram(
    data,
    x=comparison_feature,
    title=f"Distribution of {comparison_feature.replace('_', ' ')}",
    labels={comparison_feature: comparison_feature.replace('_', ' ')},
    nbins=50
)
fig.add_vline(x=comparison_value, line_dash="dash", line_color="red", annotation_text="Selected Client", annotation_position="top")
fig.update_layout(showlegend=False)
st.plotly_chart(fig)

# Summary stats
avg_value = all_values.mean()
st.write(f"**Average {comparison_feature.replace('_', ' ')} across all clients:** {avg_value:,.2f}")
st.write(f"**Selected client's {comparison_feature.replace('_', ' ')}:** {comparison_value:,.2f}")

# 4. Similar Clients
st.header("👥 Similar Clients")
numeric_cols = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "DAYS_EMPLOYED"]
client_vector = client_data[numeric_cols].values[0]
data["similarity"] = data[numeric_cols].apply(lambda row: np.linalg.norm(row - client_vector), axis=1)

similar_clients = data.sort_values("similarity").head(10).drop("similarity", axis=1)
st.dataframe(similar_clients[["SK_ID_CURR", "TARGET_proba", *numeric_cols]])

# Custom CSS
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', sans-serif;
        }
        .stButton>button {
            background-color: #2C3E50;
            color: white;
            border-radius: 10px;
            padding: 8px 16px;
        }
        .stSelectbox>div>div>div>div {
            font-size: 16px;
        }
        .block-container {
            padding-top: 2rem;
        }
        .css-1d391kg {
            background-color: #F5F7FA;
        }
    </style>
""", unsafe_allow_html=True)