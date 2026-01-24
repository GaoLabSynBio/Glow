import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse

# Allow import from relative path when running as script
if __name__ == "__main__":
    sys.path.append(os.getcwd())
    from glow.active_learning.models import GatingClassifier
else:
    from .models import GatingClassifier

def plot_classification_result(dataset_id, data_path="data/synthetic_training_data.parquet", model_path="data/gating_model_full.pkl", output_file=None):
    """
    Plots the MLP classification classification vs Ground Truth for a specific dataset.
    Args:
        dataset_id (int): The ID of the dataset to visualize (0-999).
        data_path (str): Path to the parquet data file.
        model_path (str): Path to the saved model pickle.
        output_file (str): Path to save the plot. If None, shows the plot.
    """
    print(f"Loading data for Dataset {dataset_id} from {data_path}...")
    if not os.path.exists(data_path):
        # try relative path if running from subdir
        if os.path.exists(os.path.join("..", "..", data_path)):
             data_path = os.path.join("..", "..", data_path)
        else:
            raise FileNotFoundError(f"Data file not found: {data_path}")

    try:
        # Check if we can use filters (pyarrow required, usually present)
        # Check for DatasetID, if likely to exist
        try:
             df = pd.read_parquet(data_path, filters=[('DatasetID', '=', dataset_id)])
        except Exception:
             # If filters fail (e.g. column missing in schema causing error before fallback), just load all
             df = pd.read_parquet(data_path)
             if 'DatasetID' in df.columns:
                 df = df[df['DatasetID'] == dataset_id]
             else:
                 print("Warning: 'DatasetID' column not found, using all data.")

    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if df.empty:
        print(f"No data found for DatasetID {dataset_id}")
        return

    print(f"Loaded {len(df)} events.")

    # Load Model (Ensure class is importable)
    if not os.path.exists(model_path):
         if os.path.exists(os.path.join("..", "..", model_path)):
             model_path = os.path.join("..", "..", model_path)

    print(f"Loading model from {model_path}...")
    try:
        model = GatingClassifier.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Predict
    # Ensure columns exist
    if "Log_FSC" not in df.columns:
        # Note: model expects log1p now as per train_model.py
        if "FSC-A" in df.columns:
            df["Log_FSC"] = np.log1p(df["FSC-A"])
        else:
            print("Error: FSC-A column missing")
            return

        if "SSC-A" in df.columns:
            df["Log_SSC"] = np.log1p(df["SSC-A"])
        else:
            print("Error: SSC-A column missing")
            return

    X = df[["Log_FSC", "Log_SSC"]].values

    # Predict
    y_pred = model.predict(X)

    # Ground Truth
    has_ground_truth = False
    acc_text = "N/A"

    if "TrueLabel" in df.columns:
        y_true_raw = df["TrueLabel"].values
        # Check if all are Unknown
        if np.all(y_true_raw == 'Unknown'):
            acc_text = "N/A (Unknown Labels)"
            has_ground_truth = True # We have the column, but it's all unknown
        else:
            # Convert 'Live' to 1, others to 0
            y_true = np.zeros(len(y_true_raw), dtype=int)
            y_true[y_true_raw == 'Live'] = 1
            has_ground_truth = True

            acc = np.mean(y_pred == y_true)
            acc_text = f"{acc:.2%}"

    print(f"Dataset {dataset_id} Accuracy: {acc_text}")

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Common settings
    fs = 12
    alpha = 0.6
    s = 5

    # 1. MLP Prediction
    ax1 = axes[0]
    colors_pred = np.where(y_pred == 1, 'blue', 'red')

    ax1.scatter(df["FSC-A"], df["SSC-A"], c=colors_pred, s=s, alpha=alpha, label='Prediction')
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_title(f"MLP Prediction (Dataset #{dataset_id})\nAccuracy: {acc_text}", fontsize=fs)
    ax1.set_xlabel("FSC-A (Linear)", fontsize=fs)
    ax1.set_ylabel("SSC-A (Linear)", fontsize=fs)
    ax1.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D
    legend_1 = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="blue", label="Pred: In-Gate (Live)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="red", label="Pred: Out-Gate")
    ]
    ax1.legend(handles=legend_1, loc='upper left')

    # 2. Ground Truth
    ax2 = axes[1]
    if has_ground_truth:
        label_colors = {'Live': 'green', 'Debris': 'black', 'Outliers': 'orange', 'Unknown': 'gray'}
        unique_labels = df['TrueLabel'].unique()

        for lbl in unique_labels:
            mask = df['TrueLabel'] == lbl
            # Convert category to string if needed
            lbl_str = str(lbl)
            c = label_colors.get(lbl_str, 'gray')
            ax2.scatter(df.loc[mask, "FSC-A"], df.loc[mask, "SSC-A"], c=c, s=s, alpha=alpha, label=lbl_str)
    else:
        ax2.text(0.5, 0.5, "No Ground Truth Available", ha='center', va='center', transform=ax2.transAxes)

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_title(f"Ground Truth (Dataset #{dataset_id})", fontsize=fs)
    ax2.set_xlabel("FSC-A (Linear)", fontsize=fs)
    ax2.set_ylabel("SSC-A (Linear)", fontsize=fs)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize MLP Classification vs Ground Truth")
    parser.add_argument("dataset_id", type=int, nargs="?", default=42, help="ID of the dataset to visualize (default: 42)")
    parser.add_argument("--data", type=str, default="data/synthetic_training_data.parquet", help="Path to data parquet file")
    parser.add_argument("--model", type=str, default="data/gating_model_full.pkl", help="Path to model pickle file")
    parser.add_argument("--output", "-o", type=str, default="mlp_vis_check.png", help="Output filename (default: mlp_vis_check.png)")

    args = parser.parse_args()

    plot_classification_result(args.dataset_id, data_path=args.data, model_path=args.model, output_file=args.output)
