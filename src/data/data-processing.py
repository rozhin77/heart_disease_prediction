import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os

def get_data_path():
    """Return the path to the data file."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base_dir, 'data', 'framingham.csv')

def load_and_preprocess_data(file_path=None):
    """
    Load and preprocess the heart disease dataset.
    
    Parameters:
    -----------
    file_path : str, optional
        Path to the CSV file. If None, uses the default path.
        
    Returns:
    --------
    X : pandas.DataFrame
        Preprocessed features
    y : pandas.Series
        Target variable
    """
    if file_path is None:
        file_path = get_data_path()
        
    # Load data
    data = pd.read_csv(file_path)
    
    # Split features and target
    X = data.drop(['TenYearCHD'], axis=1)
    y = data['TenYearCHD']
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Feature scaling
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X, y
