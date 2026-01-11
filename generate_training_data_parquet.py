import os
import sys
import pandas as pd
import numpy as np

# Ensure glow is in path
sys.path.append(os.getcwd())

from glow.active_learning.simulation import generate_synthetic_data

OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "synthetic_training_data.parquet")
N_DATASETS = 1000
N_POINTS = 5000

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

all_dfs = []
print(f"Generating {N_DATASETS} datasets of {N_POINTS} points...")

for i in range(N_DATASETS):
    with open(os.devnull, 'w') as fnull:
        original_stdout = sys.stdout
        sys.stdout = fnull
        try:
            df = generate_synthetic_data(N_POINTS)
        finally:
            sys.stdout = original_stdout

    df['Label'] = (df['TrueLabel'] == 'Live').astype(int)
    df['DatasetID'] = i
    
    df["Log_FSC"] = np.log10(df["FSC-A"] + 1e-1)
    df["Log_SSC"] = np.log10(df["SSC-A"] + 1e-1)
    
    keep_cols = ["DatasetID", "FSC-A", "SSC-A", "Log_FSC", "Log_SSC", "Label", "TrueLabel"]
    
    # Optimize types for memory
    df_small = df[keep_cols].copy()
    df_small['DatasetID'] = df_small['DatasetID'].astype(np.int16)
    df_small['Label'] = df_small['Label'].astype(np.int8)
    df_small['FSC-A'] = df_small['FSC-A'].astype(np.float32)
    df_small['SSC-A'] = df_small['SSC-A'].astype(np.float32)
    df_small['Log_FSC'] = df_small['Log_FSC'].astype(np.float32)
    df_small['Log_SSC'] = df_small['Log_SSC'].astype(np.float32)
    # TrueLabel is string/category, keep as is or cast to category
    df_small['TrueLabel'] = df_small['TrueLabel'].astype('category')
    
    all_dfs.append(df_small)
    
    if (i+1) % 100 == 0:
        print(f"  Generated {i+1}/{N_DATASETS} datasets")

print("Concatenating...")
final_df = pd.concat(all_dfs, ignore_index=True)

print(f"Saving {len(final_df)} rows to {OUTPUT_FILE}...")
final_df.to_parquet(OUTPUT_FILE, index=False)
print("Done.")
