from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import pandas as pd
import numpy as np

def train_models(X_train, y_train):
    """
    Train multiple models and return a dictionary of trained models.
    """
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
    return trained_models

def evaluate_models(models, X_test, y_test):
    """
    Evaluate models and return a dataframe of metrics.
    """
    results = []
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec
        })
        print(f"--- {name} ---")
        print(classification_report(y_test, y_pred))
        
    return pd.DataFrame(results)

def get_best_model(models, X_test, y_test, metric='Accuracy'):
    """
    Select the best model based on a metric.
    """
    # Simply running evaluate and picking top
    results_df = evaluate_models(models, X_test, y_test)
    best_row = results_df.sort_values(by=metric, ascending=False).iloc[0]
    best_model_name = best_row['Model']
    print(f"Best Model: {best_model_name} with {metric}: {best_row[metric]:.4f}")
    return models[best_model_name]

import joblib
import os

def save_model(model, filename):
    """
    Save the trained model to a file.
    """
    if not os.path.exists('models'):
        os.makedirs('models')
    filepath = os.path.join('models', filename)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filename):
    """
    Load a trained model from a file.
    """
    filepath = os.path.join('models', filename)
    if os.path.exists(filepath):
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model
    else:
        print(f"Model file not found: {filepath}")
        return None

if __name__ == "__main__":
    # Test stub
    pass
