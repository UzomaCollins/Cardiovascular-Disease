import numpy as np
import pandas as pd
import streamlit as st
import warnings
from keras.models import load_model  # type: ignore # Import Keras's load_model for model loading

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Title of the app
st.title("Cardiovascular Disease Prediction")

# Description
st.write("""
This app predicts whether a person has **Cardiovascular Disease** based on various health metrics. 
Please fill out the following details to get a prediction.
""")

# Input fields for user data
age = st.slider("Age (in years)", 18, 100, 50)
height = st.number_input("Height (in cm)", min_value=100, max_value=250, value=170)
weight = st.number_input("Weight (in kg)", min_value=30.0, max_value=200.0, value=70.0)
gender = st.selectbox("Gender", options=["Male", "Female"])
ap_hi = st.number_input("Systolic Blood Pressure (ap_hi)", min_value=80, max_value=200, value=120)
ap_lo = st.number_input("Diastolic Blood Pressure (ap_lo)", min_value=50, max_value=130, value=80)
cholesterol = st.selectbox("Cholesterol Level", options=["Normal", "Above Normal", "Well Above Normal"])
gluc = st.selectbox("Glucose Level", options=["Normal", "Above Normal", "Well Above Normal"])
smoke = st.selectbox("Do you smoke?", options=["No", "Yes"])
alco = st.selectbox("Do you consume alcohol?", options=["No", "Yes"])
active = st.selectbox("Are you physically active?", options=["No", "Yes"])

# Map categorical inputs to numerical values
gender_map = {"Male": 2, "Female": 1}
cholesterol_map = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}
gluc_map = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}
binary_map = {"No": 0, "Yes": 1}

# Convert inputs into the required format for the model
features = {
    "age": age * 365,  # Convert age to days as required
    "height": height,
    "weight": weight,
    "gender": gender_map[gender],
    "ap_hi": ap_hi,
    "ap_lo": ap_lo,
    "cholesterol": cholesterol_map[cholesterol],
    "gluc": gluc_map[gluc],
    "smoke": binary_map[smoke],
    "alco": binary_map[alco],
    "active": binary_map[active],
}

# Convert the feature dictionary to a dataframe
features_df = pd.DataFrame([features])

# Ensure the input data has 16 columns if the model expects it
required_columns = 16
current_columns = features_df.shape[1]

if current_columns < required_columns:
    for i in range(current_columns, required_columns):
        features_df[f"extra_{i}"] = 0  # Add extra columns as needed

# Load the trained model using Keras
model = load_model(r"C:\Users\COLLINS\Desktop\Cadio Vascular\my_model.keras")

# Make prediction
if st.button("Predict Cardiovascular Disease"):
    prediction = model.predict(features_df)
    if prediction == 1:
        st.error("The model predicts that the person **has** Cardiovascular Disease.")
    else:
        st.success("The model predicts that the person **does not have** Cardiovascular Disease.")
