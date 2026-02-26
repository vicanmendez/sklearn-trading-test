from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, log_loss, classification_report)
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def train_models(X_train, y_train):
    """
    Train multiple models using Pipelines (Scaler + Classifier) and return a dictionary of trained pipelines.
    """
    models = {
        'LogisticRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000))
        ]),
        'RandomForest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        'GradientBoosting': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(n_estimators=100, random_state=42))
        ])
    }
    
    trained_models = {}
    for name, pipeline in models.items():
        print(f"Training {name} with Pipeline...")
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline
        
    return trained_models

def evaluate_models(models, X_test, y_test):
    """
    Evaluate models and return a dataframe of metrics.
    Includes Log Loss (menor = menos error) and F1-Score.
    """
    results = []
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Log Loss requiere probabilidades; solo disponible si el modelo las soporta
        try:
            y_proba = model.predict_proba(X_test)
            ll = log_loss(y_test, y_proba)
        except AttributeError:
            ll = float('nan')
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'LogLoss': ll   # Menor es mejor
        })
    
    results_df = pd.DataFrame(results)
    
    # Tabla comparativa
    print("\n" + "=" * 65)
    print(" COMPARATIVA DE MODELOS")
    print("=" * 65)
    display_df = results_df.copy()
    for col in ['Accuracy', 'Precision', 'Recall', 'F1']:
        display_df[col] = display_df[col].map(lambda x: f"{x:.4f}")
    display_df['LogLoss'] = display_df['LogLoss'].map(
        lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A"
    )
    print(display_df.to_string(index=False))
    print("  * LogLoss: MENOR = MEJOR (menos error probabilístico)")
    print("=" * 65)
    
    # Detalle por modelo
    for name, model in models.items():
        y_pred = model.predict(X_test)
        print(f"\n--- {name} ---")
        print(classification_report(y_test, y_pred, zero_division=0))
        
    return results_df

def get_best_model(models, X_test, y_test, metric='Precision'):
    """
    Select the best model based on a metric.
    - Para Accuracy, Precision, Recall, F1: mayor es mejor (ascending=False)
    - Para LogLoss: menor es mejor (ascending=True)
    """
    results_df = evaluate_models(models, X_test, y_test)
    
    # LogLoss se minimiza; el resto se maximiza
    ascending = (metric == 'LogLoss')
    best_row = results_df.sort_values(by=metric, ascending=ascending).iloc[0]
    best_model_name = best_row['Model']
    
    direction = "(menor=mejor)" if ascending else "(mayor=mejor)"
    print(f"\n✅ Mejor Modelo: {best_model_name} | {metric}: {best_row[metric]:.4f} {direction}")
    return models[best_model_name]

import joblib
import os

def save_model(model, filepath):
    """
    Save the trained model to a file.
    filepath can be a full path (e.g. /abs/path/models/bot_1.pkl)
    or just a filename (e.g. 'best_model.pkl'), in which case it saves to models/.
    """
    # If filepath is just a bare filename (no directory), prepend models/
    if not os.path.dirname(filepath):
        filepath = os.path.join('models', filepath)
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")
    return filepath

def load_model(filepath):
    """
    Load a trained model from a file.
    filepath can be a full path or a bare filename (looked up in models/).
    """
    if not os.path.dirname(filepath):
        filepath = os.path.join('models', filepath)
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
