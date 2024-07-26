# app.ipynb

# Importing necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import load_breast_cancer

# Load the saved model, scaler, and selector
with open('breast_cancer_model.pkl', 'rb') as model_file:
    mlp = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('selector.pkl', 'rb') as selector_file:
    selector = pickle.load(selector_file)

# Streamlit app
def main():
    st.title("Breast Cancer Prediction")

    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    if st.checkbox("Show Raw Data"):
        st.write(df)

    st.sidebar.header("User Input Parameters")
    def user_input_features():
        data = {}
        for feature in df.columns[:-1]:
            data[feature] = st.sidebar.slider(feature, float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))
        return pd.DataFrame(data, index=[0])

    user_input = user_input_features()
    st.write(user_input)

    user_input_scaled = scaler.transform(user_input)
    user_input_selected = selector.transform(user_input_scaled)
    user_prediction = mlp.predict(user_input_selected)
    st.write("Prediction:", "Malignant" if user_prediction == 1 else "Benign")

if __name__ == '__main__':
    main()
