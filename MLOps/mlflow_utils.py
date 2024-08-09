import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, recall_score, precision_score, f1_score, accuracy_score
import pandas as pd
import numpy as np

def log_model_with_mlflow(model, model_name, X_test, y_test):
    """
    Logs a trained model and its evaluation metrics to MLflow, along with visual artifacts 
    such as the confusion matrix and ROC curve.

    Parameters:
    model: The trained model to be logged.
    model_name (str): A name to identify the model.
    X_test (array-like): The input features of the test set.
    y_test (array-like): The true labels of the test set.

    Returns:
    str: The URI where the model is logged in MLflow.
    """

    try:
        # Set the MLflow experiment; creates it if it doesn't exist
        mlflow.set_experiment("fraud_detection")

        # Start an MLflow run to log the model and metrics
        with mlflow.start_run():
            # Log the trained model using MLflow's sklearn module
            mlflow.sklearn.log_model(model, "model")
            mlflow.log_params({"model": model_name})

            # Predict using the test set and calculate evaluation metrics
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            recall = recall_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)

            # Log the calculated metrics to MLflow
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("auc", auc)

            # Create and log the confusion matrix as an image artifact
            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'], 
                                    columns=['Predicted Negative', 'Predicted Positive'])
            plt.figure(figsize=(10, 7))
            sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
            plt.title(f"Confusion Matrix for {model_name}")
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.savefig("confusion_matrix.png")
            plt.close()
            mlflow.log_artifact("confusion_matrix.png")

            # Generate and log the ROC curve as an image artifact
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.figure(figsize=(10, 7))
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for {model_name}')
            plt.legend(loc='best')
            plt.savefig("roc_curve.png")
            plt.close()
            mlflow.log_artifact("roc_curve.png")

            # Save and return the URI where the model is stored in MLflow
            model_uri = mlflow.get_artifact_uri("model")
        return model_uri

    except mlflow.exceptions.MlflowException as e:
        print(f"MLflowException: {e}")
        raise e

    except Exception as e:
        print(f"Exception: {e}")
        raise e
