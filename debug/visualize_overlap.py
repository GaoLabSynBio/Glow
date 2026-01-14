import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

# Ensure glow is in path
sys.path.append(os.getcwd())

from glow.active_learning.models import GatingClassifier, DensityCenteringTransformer

def visualize_dataset_overlap():
    data_path = "data/real_training_data.parquet"
    if not os.path.exists(data_path):
        print("Data not found")
        return

    # Load small subset
    df = pd.read_parquet(data_path).sample(50000, random_state=42)
    
    # Pick one dataset ID that has decent count
    counts = df['DatasetID'].value_counts()
    target_id = counts.index[0]
    
    subset = df[df['DatasetID'] == target_id].copy()
    print(f"Visualizing DatasetID {target_id} with {len(subset)} samples.")

    X = subset[['Log_FSC', 'Log_SSC']].to_numpy()
    y = subset['Label'].to_numpy()
    
    # Apply Centering
    centerer = DensityCenteringTransformer()
    X_centered = centerer.transform(X) # No ID needed since single dataset
    
    # Calculate Separation
    live = X_centered[y == 1]
    debris = X_centered[y == 0]
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(debris[:, 0], debris[:, 1], c='red', s=1, alpha=0.5, label='Debris')
    plt.scatter(live[:, 0], live[:, 1], c='blue', s=1, alpha=0.5, label='Live')
    plt.title("Original (Centered) Features")
    plt.xlabel("Centered Log FSC")
    plt.ylabel("Centered Log SSC")
    plt.legend()
    
    # Histogram of Distance from Center
    plt.subplot(1, 2, 2)
    dist_live = np.linalg.norm(live, axis=1)
    dist_debris = np.linalg.norm(debris, axis=1)
    
    plt.hist(dist_live, bins=50, color='blue', alpha=0.5, density=True, label='Live')
    plt.hist(dist_debris, bins=50, color='red', alpha=0.5, density=True, label='Debris')
    plt.title("Distance from Mode (0,0)")
    plt.xlabel("Euclidean Distance")
    plt.legend()
    
    plt.savefig("debug/overlap_check.png")
    print("Saved plot to debug/overlap_check.png")

if __name__ == "__main__":
    visualize_dataset_overlap()
