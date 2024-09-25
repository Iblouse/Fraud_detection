# Fraud Detection

## Project Description and Overview

The **Fraud Detection** project aims to build a machine learning model capable of detecting fraudulent transactions from a given dataset. The project involves various stages, including Exploratory Data Analysis (EDA), model development, and deployment. The model is deployed using MLOps practices with tools like MLflow and Streamlit to ensure that it is production-ready and easily accessible.

The primary goal is to identify transactions that are likely to be fraudulent so that they can be flagged for further investigation. The project demonstrates a complete workflow from data exploration to model deployment.

## Table of Contents

- [Project Description and Overview](#project-description-and-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

## Features

- **Exploratory Data Analysis (EDA):** Understand the dataset through visualization and statistical analysis.
- **Model Development:** Multiple machine learning models are trained and evaluated to identify the best-performing model.
- **MLOps Integration:** Modularized Python scripts for training, evaluating, and deploying models using MLflow and Streamlit.
- **Model Deployment:** Deploy the best model with Streamlit, allowing users to upload a CSV file and get fraud detection predictions.

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/Fraud_detection.git
   ```
   
2. **Navigate to the project directory:**
   ```bash
   cd Fraud_detection
   ```

3. **Create a virtual environment:**
   ```bash
   python -m venv fraud_venv
   ```

4. **Activate the virtual environment:**
   - On Windows:
     ```bash
     fraud_venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source fraud_venv/bin/activate
     ```

5. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Jupyter Notebook

To explore the data and develop models, you can open and run the `fraud_detection.ipynb` notebook:

1. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

2. **Open the `fraud_detection.ipynb` file** from the notebook interface.

### Running the Streamlit App

To run the Streamlit app for fraud detection:

1. **Ensure MLflow tracking server is running:**
   ```bash
   mlflow ui
   ```

2. **Run the `main.py` script** to train and save the best model:
   ```bash
   python MLOps/main.py
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run MLOps/fraud_app.py
   ```

4. **Upload a CSV file** in the Streamlit interface to get predictions for fraudulent transactions.

## Project Structure

```plaintext
Fraud_detection/
│
├── data/
│   ├── your_dataset.csv
│   └── ... (other datasets)
│
├── MLOps/
│   ├── data_processing.py         # Contains functions for loading and preprocessing data
│   ├── model_training.py          # Contains functions for training and evaluating models
│   ├── mlflow_utils.py            # Contains functions for logging models and metrics with MLflow
│   ├── main.py                    # Main script to train the model and log to MLflow
│   ├── fraud_app.py                     # Streamlit app for fraud detection
│   └── model_uri.txt              # File to store the URI of the best model
│
├── modelfraud.ipynb          # Jupyter notebook for EDA and model development
├── requirements.txt               # Required Python packages
└── README.md                      # This file
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
