import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import shutil

# Ensure glow is in path
sys.path.append(os.getcwd())

from glow.active_learning.models import GatingClassifier

def train_and_plot():
    # Load data
    data_path = "data/real_training_data.parquet"
    print(f"Loading data from {data_path}...")

    if not os.path.exists(data_path):
        print("Error: Data file not found.")
        return

    # Load Full Dataset (Fits in memory check: 20M rows * ~40 bytes < 1GB)
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} samples.")
    print("Columns:", df.columns.tolist())

    # Pre-shuffling (Pandas sample with frac=1 shuffles rows)
    print("Shuffling data...")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Prepare Inputs
    # Features: Log_FSC, Log_SSC
    # Note: df already has Log_FSC, Log_SSC calculated.
    X = df[['Log_FSC', 'Log_SSC']].to_numpy()
    
    # Label: 0 or 1
    y = df['Label'].to_numpy()
    
    # Metadata for GatingClassifier
    dataset_ids = df['DatasetID'].to_numpy()
    cell_types = df['CellType'].to_numpy()

    # Split Train/Val (80/20)
    print("Splitting data 80/20...")
    X_train, X_val, y_train, y_val, id_train, id_val, ct_train, ct_val = train_test_split(
        X, y, dataset_ids, cell_types, test_size=0.2, random_state=42
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")

    # ---------------------------------------------------------
    # Model Training
    # ---------------------------------------------------------
    print("Initializing model...")
    # Using default architecture as defined in class
    model = GatingClassifier(random_state=42)
    model.mlp.verbose = True  # Let sklearn print progress
    model.mlp.early_stopping = True
    model.mlp.n_iter_no_change = 5

    print("Starting training (Standard fit)...")
    
    # model.fit expects: X, y, dataset_ids, cell_types
    model.fit(
        X_train, 
        y_train, 
        dataset_ids=id_train, 
        cell_types=ct_train
    )

    print("Training complete.")

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------
    print("Evaluating on Validation Set...")
    y_pred = model.predict(X_val, dataset_ids=id_val, cell_types=ct_val)
    
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {acc:.4%}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['Debris', 'Live']))

    # ---------------------------------------------------------
    # Save Model
    # ---------------------------------------------------------
    output_path = "data/gating_model_full.pkl"
    print(f"Saving model to {output_path}...")
    # Using joblib via pickle logic inside internal or external (joblib is standard for sklearn)
    # The user asked to save to data/
    with open(output_path, 'wb') as f:
        joblib.dump(model, f)
        
    print("Model saved.")

    # ---------------------------------------------------------
    # Plotting Loss Curve
    # ---------------------------------------------------------
    if hasattr(model.mlp, 'loss_curve_'):
        print("Generating loss curve...")
        plt.figure(figsize=(10, 6))
        plt.plot(model.mlp.loss_curve_, label='Training Loss')
        
        # If early stopping was used, validation scores might be available in validation_scores_
        if hasattr(model.mlp, 'validation_scores_') and model.mlp.validation_scores_:
             plt.plot(model.mlp.validation_scores_, label='Validation Score (Accuracy)')
             plt.legend()

        plt.title('MLP Training Loss Curve')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plot_path = "data/training_loss_curve.png"
        plt.savefig(plot_path)
        print(f"Saved loss curve to {plot_path}")
    else:
        print("Warning: Loss curve not found in model attributes.")

if __name__ == "__main__":
    train_and_plot()
