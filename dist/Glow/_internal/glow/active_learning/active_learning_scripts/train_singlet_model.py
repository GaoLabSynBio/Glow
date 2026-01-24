import pandas as pd
import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from glow.active_learning.models import SingletClassifier

def train_singlet_model(data_path, output_path, subsample=None):
    print(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    df = pd.read_parquet(data_path)

    # Drop NAs
    df = df.dropna()

    print(f"Loaded {len(df)} samples.")

    if subsample and len(df) > subsample:
        print(f"Subsampling to {subsample} samples...")
        df = df.sample(n=subsample, random_state=42)

    # Features: Log_FSC_A, Log_FSC_H
    X = df[['Log_FSC_A', 'Log_FSC_H']].values
    y = df['Label'].values
    dataset_ids = df['DatasetID'].values
    cell_types = df['CellType'].values if 'CellType' in df.columns else None

    if cell_types is None:
        print("Warning: CellType column not found in data. Training without cell type info.")

    # Split Train/Val (80/20)
    print("Splitting data 80/20...")
    if cell_types is not None:
        X_train, X_val, y_train, y_val, id_train, id_val, ct_train, ct_val = train_test_split(
            X, y, dataset_ids, cell_types, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, X_val, y_train, y_val, id_train, id_val = train_test_split(
            X, y, dataset_ids, test_size=0.2, random_state=42, stratify=y
        )
        ct_train = None
        ct_val = None

    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")

    # Initialize model
    print("Initializing model...")
    model = SingletClassifier(hidden_layer_sizes=(64, 32), random_state=42)

    # Configure MLP for monitoring and early stopping
    model.mlp.verbose = True
    model.mlp.early_stopping = True
    model.mlp.n_iter_no_change = 5

    print("Starting training...")
    # Passing dataset_ids is crucial for DensityCentering
    model.fit(X_train, y_train, dataset_ids=id_train, cell_types=ct_train)
    print("Training complete.")

    # Evaluation
    print("Evaluating on Validation Set...")
    y_pred = model.predict(X_val, dataset_ids=id_val, cell_types=ct_val)

    acc = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {acc:.4%}")

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['Doublet', 'Singlet']))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    # Save Model
    print(f"\nSaving model to {output_path}...")
    model.save(output_path)
    print("Model saved.")

    # Plotting Loss Curve
    if hasattr(model.mlp, 'loss_curve_'):
        print("Generating loss curve...")
        plt.figure(figsize=(10, 6))
        plt.plot(model.mlp.loss_curve_, label='Training Loss')

        # If early stopping was used, validation scores might be available in validation_scores_
        if hasattr(model.mlp, 'validation_scores_') and model.mlp.validation_scores_:
             plt.plot(model.mlp.validation_scores_, label='Validation Score (Accuracy)')

        plt.title('Singlet Classifier Training Loss Curve')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Determine plot path
        plot_dir = os.path.dirname(output_path)
        if not plot_dir:
            plot_dir = "."
        plot_path = os.path.join(plot_dir, "singlet_training_loss_curve.png")

        plt.savefig(plot_path)
        print(f"Saved loss curve to {plot_path}")
    else:
        print("Warning: Loss curve not found in model attributes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Singlet Classifier")
    parser.add_argument("--data", type=str, default="data/singlet_training_data.parquet", help="Path to input parquet data")
    parser.add_argument("--output", type=str, default="data/singlet_model_full.pkl", help="Path to save model")
    parser.add_argument("--subsample", type=int, default=1000000, help="Number of samples to train on (for speed)")

    args = parser.parse_args()
    train_singlet_model(args.data, args.output, args.subsample)
