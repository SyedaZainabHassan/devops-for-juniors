import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def create_regression_dataset():
    """Create a sample regression dataset"""
    X, y = make_regression(
        n_samples=1000,
        n_features=10,
        noise=0.1,
        random_state=42
    )
    return X, y

def build_neural_network(input_dim):
    """Build a simple neural network for regression"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train_tensorflow_model():
    """Train a TensorFlow model"""
    print("TensorFlow Version:", tf.version.VERSION)
    print("Creating regression dataset...")
    
    # Create dataset
    X, y = create_regression_dataset()
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Building neural network...")
    model = build_neural_network(X_train.shape[1])
    
    print("Model Architecture:")
    model.summary()
    
    print("\nTraining model...")
    history = model.fit(
        X_train_scaled, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    return model, history

if __name__ == "__main__":
    print("TensorFlow ML Container - Neural Network Regression")
    print("=" * 55)
    model, history = train_tensorflow_model()
    print("\nTensorFlow model training completed successfully!")
