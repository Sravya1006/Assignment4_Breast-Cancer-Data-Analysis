# Importing necessary libraries for the application, data handling, and machine learning
import streamlit as st
import pandas as pd
import pickle
from sklearn.datasets import load_breast_cancer

# Load the saved model, scaler, and selector from disk
# This ensures we are using the same model and preprocessing tools that were trained and saved previously
with open('breast_cancer_model.pkl', 'rb') as model_file:
    mlp = pickle.load(model_file)  # Load the trained Multi-Layer Perceptron model

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)  # Load the scaler for data normalization

with open('selector.pkl', 'rb') as selector_file:
    selector = pickle.load(selector_file)  # Load the feature selector for dimensionality reduction

# Main function for the Streamlit app
def main():
    st.title("Breast Cancer Prediction App")  # Title of the application

    # Load the Breast Cancer dataset for reference and to obtain feature ranges
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target  # Append target to use for demonstration if needed

    # Option to display raw dataset within the app
    if st.checkbox("Show Raw Data"):
        st.subheader("Breast Cancer Dataset")
        st.write(df)  # This shows the complete dataset on demand for users interested in the raw data

    st.sidebar.header("Input Parameters")  # Sidebar for input parameters section

    # Function to capture user inputs for each feature using sliders
    def user_input_features():
        inputs = {}
        for feature in df.columns[:-1]:  # Loop through each feature excluding the target
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
            mean_val = float(df[feature].mean())
            # Create a slider for each feature in the sidebar with the feature's range and default value as the mean
            inputs[feature] = st.sidebar.slider(feature, min_val, max_val, mean_val)
        return pd.DataFrame(inputs, index=[0])

    # Collect user input using the function defined above
    user_input = user_input_features()
    st.subheader("User Input Parameters")
    st.write(user_input)  # Display the user's inputted parameters

    # Preprocess the input data using the loaded scaler and selector
    user_input_scaled = scaler.transform(user_input)  # Normalize the input data
    user_input_selected = selector.transform(user_input_scaled)  # Reduce dimensions based on the trained selector

    # Make prediction based on the preprocessed input
    prediction = mlp.predict(user_input_selected)
    prediction_proba = mlp.predict_proba(user_input_selected)[0][1]  # Probability of being malignant

    # Display the prediction and the probability of the prediction
    st.subheader("Prediction")
    result = "Malignant" if prediction[0] == 1 else "Benign"
    st.write(f"Prediction for the input data: **{result}**")  # Output the prediction as Malignant or Benign

    st.subheader("Prediction Probability")
    st.write(f"Probability of being malignant: **{prediction_proba:.4f}**")  # Show the probability of malignancy

# Conditional to ensure that the main function is run only when this script is executed directly
if __name__ == '__main__':
    main()
