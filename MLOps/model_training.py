import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

def get_models():
    """
    Returns a dictionary of machine learning models to be used in the pipeline.
    
    The models include:
    - Logistic Regression
    - Random Forest Classifier
    - Bagging Classifier
    - Decision Tree Classifier
    - Gaussian Naive Bayes
    - Gradient Boosting Classifier

    Returns:
        dict: A dictionary with model names as keys and instantiated models as values.
    """
    return {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=11),
        'RandomForest': RandomForestClassifier(random_state=11),
        'BaggingClassifier': BaggingClassifier(random_state=11),
        'DecisionTree': DecisionTreeClassifier(random_state=11),
        'NaiveBayes': GaussianNB(),
        'GradientBoosting': GradientBoostingClassifier(random_state=11)
    }

def balance_data(X_train, y_train):
    """
    Balances the training dataset using RandomOverSampler to handle class imbalance.
    
    Args:
        X_train (pd.DataFrame or np.array): The training feature set.
        y_train (pd.Series or np.array): The training labels.

    Returns:
        tuple: Resampled training feature set and labels.
    """
    ros = RandomOverSampler(random_state=11)
    return ros.fit_resample(X_train, y_train)

def train_models(models, preprocessor, X_train, y_train):
    """
    Trains the provided models using the preprocessed training data.
    
    Args:
        models (dict): A dictionary containing the machine learning models to train.
        preprocessor (sklearn.pipeline.Pipeline): A preprocessing pipeline to apply to the data.
        X_train (pd.DataFrame or np.array): The training feature set.
        y_train (pd.Series or np.array): The training labels.

    Returns:
        dict: A dictionary containing the trained models with the model names as keys.
    """
    trained_models = {}
    for name, model in models.items():
        # Create a pipeline that first applies the preprocessor and then fits the model
        clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        # Fit the model on the training data
        clf.fit(X_train, y_train)
        # Store the trained model in the dictionary
        trained_models[name] = clf
    return trained_models

def evaluate_models(models, X_test, y_test):
    """
    Evaluates the performance of each model on the test data using various metrics.
    
    Metrics calculated include:
    - Recall
    - Precision
    - F1-score
    - Accuracy
    
    Args:
        models (dict): A dictionary containing the trained models.
        X_test (pd.DataFrame or np.array): The test feature set.
        y_test (pd.Series or np.array): The test labels.

    Returns:
        dict: A dictionary containing the evaluation metrics for each model.
    """
    results = {}
    for name, model in models.items():
        # Predict the labels for the test set
        y_pred = model.predict(X_test)
        # Calculate evaluation metrics
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        # Store the results in the dictionary
        results[name] = {
            'recall': recall,
            'precision': precision,
            'f1_score': f1,
            'accuracy': accuracy
        }
    return results
