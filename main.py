import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open('model.pkl', 'rb') as file:
    grid_search = pickle.load(file)

# Title and description
st.title("Breast Cancer Prediction App")
st.write("""
This app predicts whether breast cancer is **Malignant** or **Benign** based on input data.
Please enter the values below:
""")

# Input fields for the features
input_data = []
input_data.append(st.number_input("Mean Radius", value=13.54))
input_data.append(st.number_input("Mean Texture", value=14.36))
input_data.append(st.number_input("Mean Perimeter", value=87.46))
input_data.append(st.number_input("Mean Area", value=566.3))
input_data.append(st.number_input("Mean Smoothness", value=0.09779))
input_data.append(st.number_input("Mean Compactness", value=0.08129))
input_data.append(st.number_input("Mean Concavity", value=0.06664))
input_data.append(st.number_input("Mean Concave Points", value=0.04781))
input_data.append(st.number_input("Mean Symmetry", value=0.1885))
input_data.append(st.number_input("Mean Fractal Dimension", value=0.05766))
input_data.append(st.number_input("Radius Error", value=0.2699))
input_data.append(st.number_input("Texture Error", value=0.7886))
input_data.append(st.number_input("Perimeter Error", value=2.058))
input_data.append(st.number_input("Area Error", value=23.56))
input_data.append(st.number_input("Smoothness Error", value=0.008462))
input_data.append(st.number_input("Compactness Error", value=0.0146))
input_data.append(st.number_input("Concavity Error", value=0.02387))
input_data.append(st.number_input("Concave Points Error", value=0.01315))
input_data.append(st.number_input("Symmetry Error", value=0.0198))
input_data.append(st.number_input("Fractal Dimension Error", value=0.0023))
input_data.append(st.number_input("Worst Radius", value=15.11))
input_data.append(st.number_input("Worst Texture", value=19.26))
input_data.append(st.number_input("Worst Perimeter", value=99.7))
input_data.append(st.number_input("Worst Area", value=711.2))
input_data.append(st.number_input("Worst Smoothness", value=0.144))
input_data.append(st.number_input("Worst Compactness", value=0.1773))
input_data.append(st.number_input("Worst Concavity", value=0.239))
input_data.append(st.number_input("Worst Concave Points", value=0.1288))
input_data.append(st.number_input("Worst Symmetry", value=0.2977))
input_data.append(st.number_input("Worst Fractal Dimension", value=0.07259))

# Prediction button
if st.button("Predict"):
    # Convert input data to numpy array
    input_data_npArray = np.asarray(input_data)

    # Reshape for single prediction
    input_data_reshaped = input_data_npArray.reshape(1, -1)

    # Predict using the loaded model
    prediction = grid_search.predict(input_data_reshaped)

    # Display result
    if prediction[0] == 0:
        st.success("The Breast Cancer is **Malignant**.")
    else:
        st.success("The Breast Cancer is **Benign**.")
