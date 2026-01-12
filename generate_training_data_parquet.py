import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm

# Ensure we can import glow
sys.path.append(os.getcwd())

from glow.active_learning.simulation import generate_synthetic_data

def generate_large_dataset(n_datasets=1000, n_samples_per_dataset=5000, output_path="data/synthetic_training_data.parquet"):
    print(f"Generating {n_datasets} datasets with {n_samples_per_dataset} samples each...")

    all_dfs = []

    for i in tqdm(range(n_datasets)):
        # Generate single dataset
        df = generate_synthetic_data(n_samples=n_samples_per_dataset)

        # Add DatasetID column
        df["DatasetID"] = i

        # Optimize types to save memory
        df["DatasetID"] = df["DatasetID"].astype(np.int32)
        df["TrueLabel"] = df["TrueLabel"].astype("category")

        all_dfs.append(df)

    print("Concatenating...")
    full_df = pd.concat(all_dfs, ignore_index=True)

    # Ensure data directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Saving to {output_path}...")
    full_df.to_parquet(output_path, index=False)

    print(f"Done! Shape: {full_df.shape}")
    print(f"Columns: {full_df.columns.tolist()}")

if __name__ == "__main__":
    generate_large_dataset()
