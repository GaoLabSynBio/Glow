import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Ensure glow is in path
sys.path.append(os.getcwd())

from glow.active_learning.models import GatingClassifier

def train_and_plot():
    # Load data
    data_path = "data/synthetic_training_data.parquet"
    print(f"Loading data from {data_path}...")
    
    if not os.path.exists(data_path):
        print("Error: Data file not found.")
        return

    df = pd.read_parquet(data_path)
    X = df[["Log_FSC", "Log_SSC"]].values
    y = df["Label"].values

    print(f"Loaded {len(X)} samples. Starting training...")

    # Initialize model
    # Using verbose=True to see progress in terminal
    model = GatingClassifier(hidden_layer_sizes=(64, 32))
    
    # We need to set verbose on the internal MLP to see output during fit
    model.pipeline.named_steps['mlp'].verbose = True

    # Fit
    model.fit(X, y)

    # Extract loss curve
    mlp_step = model.pipeline.named_steps['mlp']
    loss_curve = mlp_step.loss_curve_

    print(f"Training completed in {len(loss_curve)} epochs.")
    print(f"Final Loss: {loss_curve[-1]:.6f}")
    
    # Score
    print("Calculating final accuracy...")
    acc = model.score(X, y)
    print(f"Final Accuracy: {acc:.4%}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_curve) + 1), loss_curve, marker='o', linestyle='-', markersize=4)
    plt.title('Neural Network Training Loss (MLPClassifier)')
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    output_plot = 'training_loss_curve.png'
    plt.savefig(output_plot)
    print(f"Saved loss curve to {output_plot}")

    # Save model
    model_path = "data/gating_model_full.pkl"
    model.save(model_path)
    print(f"Saved trained model to {model_path}")

if __name__ == "__main__":
    train_and_plot()
