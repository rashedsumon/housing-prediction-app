import os
import streamlit as st
import pandas as pd
import pickle
from model import train_and_save_model

# Set up page configurations
st.set_page_config(page_title="Housing Price Predictor", page_icon="🏠", layout="centered")

# Ensure the model exists before trying to load it
MODEL_PATH = "housing_model.pkl"
if not os.path.exists(MODEL_PATH):
    with st.spinner("Model file missing. Running training pipeline to build the model..."):
        train_and_save_model()

# Load the trained model and features metadata
with open(MODEL_PATH, 'rb') as f:
    artifacts = pickle.load(f)

model = artifacts['model']
model_features = artifacts['features']

# --- UI Setup ---
st.title("🏠 Housing Price Prediction AI")
st.write("Adjust the parameters below to estimate the market value of a house.")
st.markdown("---")

st.header("✨ Property Specifications")

# Create a clean 2-column layout for inputs
col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Total Area (sq ft)", min_value=100, max_value=20000, value=4500, step=50)
    bedrooms = st.slider("Number of Bedrooms", min_value=1, max_value=6, value=3)
    bathrooms = st.slider("Number of Bathrooms", min_value=1, max_value=4, value=2)
    stories = st.slider("Number of Stories/Floors", min_value=1, max_value=4, value=2)
    parking = st.slider("Parking Spaces / Garages", min_value=0, max_value=3, value=1)
    furnishing = st.selectbox("Furnishing Status", ["Fully Furnished", "Semi-Furnished", "Unfurnished"])

with col2:
    mainroad = st.radio("Connected to Main Road?", ["Yes", "No"])
    guestroom = st.radio("Has a Guestroom?", ["Yes", "No"])
    basement = st.radio("Has a Basement?", ["Yes", "No"])
    hotwater = st.radio("Has Hot Water Heating?", ["Yes", "No"])
    ac = st.radio("Has Air Conditioning?", ["Yes", "No"])
    prefarea = st.radio("Located in a Preferred Area?", ["Yes", "No"])

# --- Processing the User Inputs ---
if st.button("💰 Predict Housing Price", use_container_width=True):
    # Map Yes/No to 1/0
    input_dict = {
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'stories': stories,
        'mainroad': 1 if mainroad == "Yes" else 0,
        'guestroom': 1 if guestroom == "Yes" else 0,
        'basement': 1 if basement == "Yes" else 0,
        'hotwaterheating': 1 if hotwater == "Yes" else 0,
        'airconditioning': 1 if ac == "Yes" else 0,
        'parking': parking,
        'prefarea': 1 if prefarea == "Yes" else 0,
    }
    
    # Map furnishing status back to One-Hot columns
    # 'furnishingstatus_semi-furnished' and 'furnishingstatus_unfurnished'
    input_dict['furnishingstatus_semi-furnished'] = 1 if furnishing == "Semi-Furnished" else 0
    input_dict['furnishingstatus_unfurnished'] = 1 if furnishing == "Unfurnished" else 0
    
    # Convert input to dataframe and match the precise feature order from training
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[model_features]
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    # Output display
    st.markdown("---")
    st.success(f"### 💵 Estimated Property Value: **${prediction:,.2f}**")