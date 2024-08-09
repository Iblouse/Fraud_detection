from data_processing import load_data, preprocess_data
from model_training import get_models, balance_data, train_models, evaluate_models
from mlflow_utils import log_model_with_mlflow
from sklearn.model_selection import train_test_split

# Load and preprocess data
file_path = "fraud_data.csv"

# Load the dataset from a CSV file
data = load_data(file_path)

# Preprocess the data (e.g., handle missing values, encode categorical variables)
preprocessor, X, y = preprocess_data(data)

# Split data into training and testing sets before balancing the classes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11, stratify=y)

# Balance the training data to address class imbalance (e.g., using SMOTE)
X_train_balanced, y_train_balanced = balance_data(X_train, y_train)

# Get a list of machine learning models to train
models = get_models()

# Train each model on the balanced training data
trained_models = train_models(models, preprocessor, X_train_balanced, y_train_balanced)

# Evaluate the trained models on the test data to find the best one based on recall
evaluation_results = evaluate_models(trained_models, X_test, y_test)
best_model_name = max(evaluation_results, key=lambda k: evaluation_results[k]['recall'])
best_model = trained_models[best_model_name]

# Log the best model with MLflow and save all metrics and artifacts
try:
    model_uri = log_model_with_mlflow(best_model, best_model_name, X_test, y_test)

    # Save the model URI to a file for later use in the Streamlit app
    with open('model_uri.txt', 'w') as f:
        f.write(model_uri)

    print(f"The best model is {best_model_name} with a recall score of {evaluation_results[best_model_name]['recall']:.4f}")
    print(f"Model saved in MLflow at: {model_uri}")
except Exception as e:
    print(f"Failed to log model with MLflow: {e}")
