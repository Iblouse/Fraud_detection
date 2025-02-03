import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn
import os

def load_model():
    """
    Load a trained model from MLflow. Handles both local and cloud environments.

    Returns:
        model: The loaded machine learning model.
    """
    # Check if running locally or on Streamlit Cloud
    if os.getenv('STREAMLIT_ENV') == 'cloud':
        # Use the MLflow 'runs:' URI for cloud deployment
        model_uri = "runs:/c0d2e266d07945819b98e5a00161980d/model"
    else:
        # Use the full local path to the model for local testing
        model_uri = "file:///Users/ibrahimabarry/Documents/Fraud_detection/MLOps/mlruns/212254493020907020/c0d2e266d07945819b98e5a00161980d/artifacts/model"

    # Load the model from MLflow
    model = mlflow.sklearn.load_model(model_uri)
    return model

def predict_fraud(model, data):
    """
    Predict whether transactions are fraudulent or not using the given model.

    Args:
        model: The machine learning model used for making predictions.
        data (DataFrame): The input data on which to make predictions.

    Returns:
        list: A list of predictions, where "fraud" indicates a fraudulent transaction and 
              "not fraud" indicates a non-fraudulent transaction.
    """
    prediction = model.predict(data)
    return ["fraud" if pred == 1 else "not fraud" for pred in prediction]

# Set the title of the Streamlit app
st.title("Fraud Detection App")

# Load the pre-trained model
model = load_model()

# Section to upload a CSV file for making predictions
st.header("Upload a CSV File")
uploaded_file = st.file_uploader("Choose a CSV file (You can use sample data at https://github.com/Iblouse/Fraud_detection/tree/main/data)", type="csv")

if uploaded_file is not None:
    # Read the uploaded CSV file into a DataFrame
    input_data = pd.read_csv(uploaded_file)
    
    # Make predictions on the input data
    predictions = predict_fraud(model, input_data)
    
    # Add the predictions to the input data
    input_data['prediction'] = predictions
    
    # Display the input data with predictions
    st.write(input_data)
    
    # Provide an option to download the predictions as a CSV file
    st.download_button("Download Predictions", input_data.to_csv(index=False), file_name="predictions.csv")
