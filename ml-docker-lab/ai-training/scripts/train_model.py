#!/usr/bin/env python3

import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse

class AIModelTrainer:
    def __init__(self, config_path=None):
        self.config = self.load_config(config_path)
        self.model = None
        self.history = None
        self.scaler = StandardScaler()
        
    def load_config(self, config_path):
        """Load training configuration"""
        default_config = {
            "model_name": "neural_classifier",
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "validation_split": 0.2,
            "random_state": 42,
            "save_model": True,
            "save_plots": True
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def create_synthetic_dataset(self):
        """Create a synthetic dataset for training"""
        print("Creating synthetic dataset...")
        
        # Generate synthetic data
        np.random.seed(self.config['random_state'])
        n_samples = 10000
        n_features = 20
        
        # Create features with different patterns for each class
        X = np.random.randn(n_samples, n_features)
        
        # Create three classes with different characteristics
        y = np.zeros(n_samples)
        
        # Class 0: High values in first 5 features
        mask_0 = np.random.choice(n_samples, n_samples//3, replace=False)
        X[mask_0, :5] += 2
        y[mask_0] = 0
        
        # Class 1: High values in middle 5 features
        remaining = np.setdiff1d(np.arange(n_samples), mask_0)
        mask_1 = np.random.choice(remaining, len(remaining)//2, replace=False)
        X[mask_1, 5:10] += 2
        y[mask_1] = 1
        
        # Class 2: High values in last 5 features
        mask_2 = np.setdiff1d(remaining, mask_1)
        X[mask_2, 10:15] += 2
        y[mask_2] = 2
        
        return X, y.astype(int)
    
    def build_model(self, input_dim, num_classes):
        """Build neural network model"""
        print("Building neural network model...")
        
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            keras.layers.Dense(64, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            keras.layers.Dense(32, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        optimizer = keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self):
        """Train the AI model"""
        print("Starting AI model training...")
        print("=" * 50)
        
        # Create dataset
        X, y = self.create_synthetic_dataset()
        print(f"Dataset shape: {X.shape}")
        print(f"Number of classes: {len(np.unique(y))}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=self.config['random_state'],
            stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Build model
        self.model = self.build_model(X_train.shape[1], len(np.unique(y)))
        
        print("\nModel Architecture:")
        self.model.summary()
        
        # Define callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train model
        print(f"\nTraining for {self.config['epochs']} epochs...")
        self.history = self.model.fit(
            X_train_scaled, y_train,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            validation_split=self.config['validation_split'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        print("\nEvaluating model...")
        test_loss, test_accuracy = self.model.evaluate(X_test_scaled, y_test, verbose=0)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Generate predictions for detailed analysis
        y_pred = self.model.predict(X_test_scaled)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes))
        
        # Save results
        self.save_results(test_accuracy, test_loss)
        
        if self.config['save_plots']:
            self.create_visualizations(y_test, y_pred_classes)
        
        if self.config['save_model']:
            self.save_model()
        
        return test_accuracy, test_loss
    
    def create_visualizations(self, y_true, y_pred):
        """Create and save training visualizations"""
        print("Creating visualizations...")
        
        # Training history plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy plot
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss plot
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title('Confusion Matrix')
        axes[1, 0].set_xlabel('Predicted Label')
        axes[1, 0].set_ylabel('True Label')
        
        # Class distribution
        unique, counts = np.unique(y_true, return_counts=True)
        axes[1, 1].bar(unique, counts, alpha=0.7, label='True')
        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
        axes[1, 1].bar(unique_pred, counts_pred, alpha=0.7, label='Predicted')
        axes[1, 1].set_title('Class Distribution')
        axes[1, 1].set_xlabel('Class')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('/app/models/training_results.png', dpi=300, bbox_inches='tight')
        print("Visualizations saved to /app/models/training_results.png")
    
    def save_model(self):
        """Save the trained model"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"/app/models/{self.config['model_name']}_{timestamp}"
        
        # Save model
        self.model.save(f"{model_path}.h5")
        
        # Save scaler
        import joblib
        joblib.dump(self.scaler, f"{model_path}_scaler.pkl")
        
        print(f"Model saved to {model_path}.h5")
        print(f"Scaler saved to {model_path}_scaler.pkl")
    
    def save_results(self, accuracy, loss):
        """Save training results"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "test_accuracy": float(accuracy),
            "test_loss": float(loss),
            "tensorflow_version": tf.__version__
        }
        
        with open('/app/models/training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("Results saved to /app/models/training_results.json")

def main():
    parser = argparse.ArgumentParser(description='Train AI model in Docker')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    print("AI Model Training in Docker Container")
    print("=" * 40)
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
    print()
    
    # Initialize trainer
    trainer = AIModelTrainer(args.config)
    
    # Override config with command line arguments
    if args.epochs:
        trainer.config['epochs'] = args.epochs
    if args.batch_size:
        trainer.config['batch_size'] = args.batch_size
    
    # Train model
    try:
        accuracy, loss = trainer.train()
        print(f"\nTraining completed successfully!")
        print(f"Final Test Accuracy: {accuracy:.4f}")
        print(f"Final Test Loss: {loss:.4f}")
        
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
