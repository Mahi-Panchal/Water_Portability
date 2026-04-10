import streamlit as st
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Page Configuration ---
st.set_page_config(
    page_title=" Water Potability Predictor",
    page_icon="💧",
    layout="centered"
)

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007BFF;
        color: white;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 24px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Data Loading & Model Training ---
@st.cache_resource
def train_lgbm_model():
    # Load data
    try:
        df = pd.read_csv('water_potability.csv')
    except FileNotFoundError:
        return None, None

    # Preprocessing
    df['ph'] = df['ph'].fillna(df['ph'].mean())
    df['Sulfate'] = df['Sulfate'].fillna(df['Sulfate'].mean())
    df['Trihalomethanes'] = df['Trihalomethanes'].fillna(df['Trihalomethanes'].mean())

    X = df.drop('Potability', axis=1)
    y = df['Potability']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train LightGBM
    model = LGBMClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

model, scaler = train_lgbm_model()

# --- Dashboard Header ---
st.title("💧 Water Potability Predictor")
st.markdown("Enter the water quality parameters below to check if the water is **Potable (Safe to Drink)**.")
st.divider()

if model is None:
    st.error("Error: 'water_potability.csv' not found. Please ensure the dataset is in the repository.")
else:
    # --- Input Fields ---
    col1, col2 = st.columns(2)

    with col1:
        ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
        hardness = st.number_input("Hardness", value=196.0)
        solids = st.number_input("Solids (ppm)", value=20000.0)
        chloramines = st.number_input("Chloramines (ppm)", value=7.1)
        sulfate = st.number_input("Sulfate (mg/L)", value=333.0)

    with col2:
        conductivity = st.number_input("Conductivity (μS/cm)", value=426.0)
        organic_carbon = st.number_input("Organic Carbon (ppm)", value=14.0)
        trihalomethanes = st.number_input("Trihalomethanes (μg/L)", value=66.0)
        turbidity = st.number_input("Turbidity (NTU)", value=4.0)

    # --- Prediction Logic ---
    st.divider()
    if st.button("Analyze Water Quality"):
        # Prepare input
        features = np.array([[ph, hardness, solids, chloramines, sulfate, 
                              conductivity, organic_carbon, trihalomethanes, turbidity]])
        
        # Scale and Predict
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled)[0][1]

        # Display Results
        if prediction[0] == 1:
            st.success(f"### Result: Potable (Safe)")
            st.markdown(f"**Confidence Level:** {probability*100:.2f}%")
            st.balloons()
        else:
            st.error(f"### Result: Not Potable (Unsafe)")
            st.markdown(f"**Confidence Level:** {(1-probability)*100:.2f}%")

