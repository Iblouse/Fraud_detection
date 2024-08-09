import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def load_data(file_path):
    """
    Load the dataset from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file containing the dataset.

    Returns:
    pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """
    return pd.read_csv(file_path)

def preprocess_data(data):
    """
    Preprocess the dataset by encoding ordinal and categorical columns, 
    and scaling numerical columns.

    Parameters:
    data (pd.DataFrame): The dataset to be preprocessed.

    Returns:
    tuple: A tuple containing the preprocessor object, the preprocessed features, 
           and the target variable 'is_fraud'.
    """
    
    # Define the columns by type: ordinal, categorical, and numerical
    ordinal_cols = ['day_of_week', 'time_of_day', 'season', 'age_group']
    categorical_cols = ['category', 'gender', 'region']
    numerical_cols = ['amt', 'city_pop']
    
    # Define the order of categories for ordinal columns
    ordinal_mappings = [
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],  # Days of the week in order
        ['late night', 'early morning', 'late morning', 'early afternoon', 'late afternoon', 'evening', 'night'],  # Times of day in order
        ['Spring', 'Summer', 'Fall', 'Winter'],  # Seasons in order
        ['Under-18', '19-35', '36-50', '51-65', '66-plus']  # Age groups in order
    ]
    
    # Create a preprocessor that applies appropriate transformations to each column type
    preprocessor = ColumnTransformer(
        transformers=[
            ('ord', OrdinalEncoder(categories=ordinal_mappings), ordinal_cols),  # Apply ordinal encoding
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),  # Apply one-hot encoding
            ('num', StandardScaler(), numerical_cols)  # Apply standard scaling
        ])
    
    # Return the preprocessor, features (excluding 'is_fraud'), and the target variable 'is_fraud'
    return preprocessor, data.drop(columns=['is_fraud', 'trans_num']), data['is_fraud']

