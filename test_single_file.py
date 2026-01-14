import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure glow is in path
sys.path.append(os.getcwd())

from glow.active_learning.models import GatingClassifier

def test_model_on_file(file_path, model_path="data/gating_model_full.pkl", output_image="test_prediction.png"):
    """
    Tests the trained GatingClassifier on a specific CSV file and plots side-by-side comparison.

    Args:
        file_path (str): Path to the .csv.gz file.
        model_path (str): Path to the saved model pickle.
        output_image (str): Path to save the resulting plot.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # ---------------------------------------------------------
    # Preprocessing (Replicating logic from generate_real_data_parquet.py)
    # ---------------------------------------------------------
    # 1. Identify Columns
    fsc_col_candidates = ['FSC 488/10-A', 'FSC-A', 'FSC 488/10-A']
    ssc_col_candidates = ['SSC 488/10-A', 'SSC-A', 'SSC 488/10-A']
    label_col = 'cells'

    fsc_col = next((c for c in fsc_col_candidates if c in df.columns), None)
    ssc_col = next((c for c in ssc_col_candidates if c in df.columns), None)

    # Fuzzy fallback
    if not fsc_col or not ssc_col:
        for c in df.columns:
            if 'FSC' in c and '-A' in c: fsc_col = c
            if 'SSC' in c and '-A' in c: ssc_col = c

    if not fsc_col or not ssc_col or label_col not in df.columns:
        print(f"Missing essential columns. FSC: {fsc_col}, SSC: {ssc_col}, Label: {label_col in df.columns}")
        return

    # 2. Filter valid positive data for Log
    valid_mask = (df[fsc_col] > 0) & (df[ssc_col] > 0)
    clean_df = df.loc[valid_mask].copy()

    if len(clean_df) == 0:
        print("No valid positive data found.")
        return

    print(f"Processed {len(clean_df)} events.")

    # 3. Log Transform
    clean_df['Log_FSC'] = np.log10(clean_df[fsc_col])
    clean_df['Log_SSC'] = np.log10(clean_df[ssc_col])

    X = clean_df[['Log_FSC', 'Log_SSC']].to_numpy()
    y_true = clean_df[label_col].astype(int).to_numpy()

    # 4. Metadata Extraction (DatasetID & CellType)
    # Since we are testing a single file, we treat it as one dataset ID (e.g., 0)
    # This ensures DensityCentering uses the mode of THIS file to normalize.
    dataset_ids = np.zeros(len(clean_df), dtype=int)

    # Infer cell type from folder name
    folder_name = os.path.basename(os.path.dirname(file_path))
    cell_type = folder_name.lower().split('_')[0] # simplistic extraction logic adjusting as needed
    # The generation script used full folder_name.lower(). Let's stick to that for consistency.
    cell_type = folder_name.lower()

    print(f"Inferred Cell Type: {cell_type}")
    cell_types = np.array([cell_type] * len(clean_df))

    # ---------------------------------------------------------
    # Inference
    # ---------------------------------------------------------
    print("Loading model...")
    model = GatingClassifier.load(model_path)

    print("Predicting...")
    # Passing dataset_ids=zeros allows the model to center THIS file specifically
    y_pred = model.predict(X, dataset_ids=dataset_ids, cell_types=cell_types)

    # Calculate accuracy on this file
    accuracy = np.mean(y_pred == y_true)
    print(f"Accuracy on this file: {accuracy:.2%}")

    # ---------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------
    print("Generating plot...")
    plt.figure(figsize=(12, 6))

    # Downsample for plotting if too large (faster rendering)
    if len(clean_df) > 50000:
        plot_indices = np.random.choice(len(clean_df), 50000, replace=False)
        X_plot = X[plot_indices]
        y_true_plot = y_true[plot_indices]
        y_pred_plot = y_pred[plot_indices]
    else:
        X_plot = X
        y_true_plot = y_true
        y_pred_plot = y_pred

    # Plot 1: Ground Truth
    plt.subplot(1, 2, 1)

    # Plot Debris (0) first, then Live (1) on top mostly
    mask_true_0 = (y_true_plot == 0)
    mask_true_1 = (y_true_plot == 1)

    plt.scatter(X_plot[mask_true_0, 0], X_plot[mask_true_0, 1],
                c='gray', s=1, alpha=0.5, label='Debris')
    plt.scatter(X_plot[mask_true_1, 0], X_plot[mask_true_1, 1],
                c='blue', s=1, alpha=0.5, label='Live')

    plt.title(f"Ground Truth\n(File: {os.path.basename(file_path)})")
    plt.xlabel("Log FSC")
    plt.ylabel("Log SSC")
    # Legend with larger markers
    lgnd = plt.legend()
    for handle in lgnd.legend_handles:
        handle.set_sizes([20.0])

    # Plot 2: Model Prediction
    plt.subplot(1, 2, 2)

    mask_pred_0 = (y_pred_plot == 0)
    mask_pred_1 = (y_pred_plot == 1)

    plt.scatter(X_plot[mask_pred_0, 0], X_plot[mask_pred_0, 1],
                c='red', s=1, alpha=0.5, label='Predicted Debris')
    plt.scatter(X_plot[mask_pred_1, 0], X_plot[mask_pred_1, 1],
                c='green', s=1, alpha=0.5, label='Predicted Live')

    plt.title(f"Model Prediction\nAccuracy: {accuracy:.2%} | Cell: {cell_type}")
    plt.xlabel("Log FSC")
    plt.ylabel("Log SSC")
    lgnd = plt.legend()
    for handle in lgnd.legend_handles:
        handle.set_sizes([20.0])

    plt.tight_layout()
    plt.savefig(output_image, dpi=150)
    print(f"Comparison plot saved to {output_image}")

if __name__ == "__main__":
    # Example usage with the file found via grep in previous turn
    sample_file = "/home/xwz/Projects/test/Glow/training/labeled_data/hug1n/mzhang23-2025-11-21_HUG1N_mCherry_GFP-20251121-1102_D5_2173.csv.gz"
    if len(sys.argv) > 1:
        sample_file = sys.argv[1]

    test_model_on_file(sample_file)
