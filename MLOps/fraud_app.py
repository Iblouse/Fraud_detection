import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn

def load_model():
    """
    Load a trained model from a URI specified in a text file.

    The URI is read from the 'model_uri.txt' file, and the model is loaded using 
    MLflow's sklearn module.

    Returns:
        model: The loaded machine learning model.
    """
    with open('model_uri.txt', 'r') as file:
        model_uri = file.read().strip()
    return mlflow.sklearn.load_model(model_uri)

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
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

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
