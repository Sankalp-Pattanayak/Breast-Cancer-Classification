import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open('model.pkl', 'rb') as file:
    grid_search = pickle.load(file)

# Set page title and icon
st.set_page_config(page_title="Breast Cancer Prediction",
                   page_icon="🩺", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .stApp { background-color: #f5f5f5; }
    .main-title { color: #333366; text-align: center; font-size: 32px; font-weight: bold; }
    .sub-title { color: #444444; text-align: center; font-size: 18px; }
    .prediction-box { padding: 20px; border-radius: 10px; text-align: center; font-size: 20px; font-weight: bold; }
    .benign { background-color: #d4edda; color: #155724; }
    .malignant { background-color: #f8d7da; color: #721c24; }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-title">🔬 Breast Cancer Prediction App</h1>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-title">Enter the required values below and click "Predict" to see the results.</p>', unsafe_allow_html=True)

# Sidebar for input fields
st.sidebar.header("Enter Feature Values")
input_data = []
feature_names = [
    "Mean Radius", "Mean Texture", "Mean Perimeter", "Mean Area", "Mean Smoothness",
    "Mean Compactness", "Mean Concavity", "Mean Concave Points", "Mean Symmetry", "Mean Fractal Dimension",
    "Radius Error", "Texture Error", "Perimeter Error", "Area Error", "Smoothness Error",
    "Compactness Error", "Concavity Error", "Concave Points Error", "Symmetry Error", "Fractal Dimension Error",
    "Worst Radius", "Worst Texture", "Worst Perimeter", "Worst Area", "Worst Smoothness",
    "Worst Compactness", "Worst Concavity", "Worst Concave Points", "Worst Symmetry", "Worst Fractal Dimension"
]

default_values = [
    13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766,
    0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023,
    15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259
]

for i, feature in enumerate(feature_names):
    input_data.append(st.sidebar.number_input(
        feature, value=default_values[i]))

# Prediction button
if st.sidebar.button("🔍 Predict"):
    # Convert input data to numpy array and reshape for single prediction
    input_data_npArray = np.asarray(input_data).reshape(1, -1)

    # Predict using the loaded model
    prediction = grid_search.predict(input_data_npArray)

    # Display result in the main page
    if prediction[0] == 0:
        st.markdown(
            '<div class="prediction-box malignant">⚠️ The Breast Cancer is Malignant.</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="prediction-box benign">✅ The Breast Cancer is Benign.</div>', unsafe_allow_html=True)

# Footer
st.markdown("<br><hr><p style='text-align:center;'>Made by Sankalp</p>",
            unsafe_allow_html=True)
