# app.py
import streamlit as st
import pandas as pd
import pickle
import os

# -----------------------------
# 1. Page Configuration
# -----------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered",
)

st.title("🏠 House Price Predictor App")
st.write("Enter the details of the house to predict its price.")

# -----------------------------
# 2. Load Model
# -----------------------------
MODEL_PATH = "house_price_model1.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    st.success("✅ Model loaded successfully!")
except FileNotFoundError:
    st.error("❌ Model file not found! Make sure it's in the repo.")
    st.stop()
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
    st.stop()

# -----------------------------
# 3. User Input
# -----------------------------
st.sidebar.header("Enter House Details")

size = st.sidebar.number_input("Size (in sq.ft)", min_value=300, max_value=10000, value=1500)
bedrooms = st.sidebar.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
age = st.sidebar.number_input("Age of House (in years)", min_value=0, max_value=100, value=10)

# -----------------------------
# 4. Prediction
# -----------------------------
if st.button("Predict Price"):
    input_data = pd.DataFrame([[size, bedrooms, age]], columns=["size", "bedrooms", "age"])
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Predicted House Price: ${prediction:,.2f}")
