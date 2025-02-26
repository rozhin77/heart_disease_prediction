import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

def get_model_path():
    """Return the path where models should be saved."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base_dir, 'models', 'best_heart_disease_model.joblib')

def train_or_load_model(X=None, y=None, force_train=False):
    """
    Train a new model or load an existing one.
    
    Parameters:
    -----------
    X : pandas.DataFrame, optional
        Features for training
    y : pandas.Series, optional
        Target variable for training
    force_train : bool, default=False
        If True, trains a new model even if one exists
        
    Returns:
    --------
    model : sklearn estimator
        Trained model
    """
    model_path = get_model_path()
    
    # Try to load existing model
    if not force_train and os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            print('مدل از فایل بارگذاری شد')
            return model
        except Exception as e:
            print(f'خطا در بارگذاری مدل: {str(e)}')
            print('آموزش مدل جدید...')
    else:
        print('آموزش مدل جدید...')
    
    # Check if we have data for training
    if X is None or y is None:
        raise ValueError("برای آموزش مدل جدید، داده‌های X و y باید ارائه شوند")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'دقت مدل: {accuracy:.4f}')
    
    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model performance.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test target
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Return results
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix
    }
