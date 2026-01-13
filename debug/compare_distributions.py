import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def compare_distributions():
    real_path = "data/real_data.parquet"
    synth_path = "data/synthetic_training_data.parquet"

    print(f"Loading {real_path}...")
    try:
        df_real = pd.read_parquet(real_path)
        print(f"Loaded Real Data: {df_real.shape}")
    except Exception as e:
        print(f"Error loading real data: {e}")
        return

    print(f"Loading {synth_path}...")
    try:
        df_synth = pd.read_parquet(synth_path)
        print(f"Loaded Synth Data: {df_synth.shape}")
    except Exception as e:
        print(f"Error loading synth data: {e}")
        return

    features = ["FSC-A", "SSC-A"]

    # Statistics
    print("\n--- Statistics (Linear Scale) ---")
    for feat in features:
        print(f"\nFeature: {feat}")
        print("Real Data:")
        print(df_real[feat].describe())
        print("Synthetic Data:")
        print(df_synth[feat].describe())

    # Log Statistics
    print("\n--- Statistics (Log1p Scale) ---")
    for feat in features:
        feat_log = f"Log_{feat}"
        # Calculate log if not present
        real_log = np.log1p(df_real[feat])
        synth_log = np.log1p(df_synth[feat])

        print(f"\nFeature: {feat_log}")
        print("Real Data:")
        print(real_log.describe())
        print("Synthetic Data:")
        print(synth_log.describe())

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for i, feat in enumerate(features):
        # Linear Plot
        ax_lin = axes[0, i]
        ax_lin.hist(df_synth[feat], bins=50, alpha=0.5, label='Synthetic', density=True, color='blue')
        ax_lin.hist(df_real[feat], bins=50, alpha=0.5, label='Real', density=True, color='red')
        ax_lin.set_title(f"{feat} (Linear)")
        ax_lin.legend()

        # Log Plot
        ax_log = axes[1, i]
        real_log = np.log1p(df_real[feat])
        synth_log = np.log1p(df_synth[feat])

        ax_log.hist(synth_log, bins=50, alpha=0.5, label='Synthetic', density=True, color='blue')
        ax_log.hist(real_log, bins=50, alpha=0.5, label='Real', density=True, color='red')
        ax_log.set_title(f"{feat} (Log1p)")
        ax_log.legend()

    output_file = "debug/distribution_comparison.png"
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"\nPlot saved to {output_file}")

if __name__ == "__main__":
    compare_distributions()
