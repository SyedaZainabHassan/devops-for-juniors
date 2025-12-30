import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def create_sample_dataset():
    """Create a sample classification dataset"""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=10,
        n_clusters_per_class=1,
        random_state=42
    )
    return X, y

def train_model():
    """Train a simple ML model"""
    print("Creating sample dataset...")
    X, y = create_sample_dataset()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("Training Random Forest model...")
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, accuracy

if __name__ == "__main__":
    print("Python ML Container - Sample Classification Task")
    print("=" * 50)
    model, accuracy = train_model()
    print(f"\nTraining completed successfully!")
    print(f"Final accuracy: {accuracy:.4f}")
