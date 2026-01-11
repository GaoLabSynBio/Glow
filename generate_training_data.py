import os
import sys
import pandas as pd
import numpy as np

# Ensure glow is in path
sys.path.append(os.getcwd())

from glow.active_learning.simulation import generate_synthetic_data

OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "synthetic_training_data.csv")
N_DATASETS = 1000
N_POINTS = 5000

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

all_dfs = []
print(f"Generating {N_DATASETS} datasets of {N_POINTS} points...")

for i in range(N_DATASETS):
    # Suppress print output from generate_synthetic_data using a context manager or just let it print if checking progress?
    # The function prints to stdout. 1000 datasets * 4 lines = 4000 lines. 
    # That's a lot of spam. I should redirect stdout temporarily.
    
    # Simple stdout capture
    with open(os.devnull, 'w') as fnull:
        original_stdout = sys.stdout
        sys.stdout = fnull
        try:
            df = generate_synthetic_data(N_POINTS)
        finally:
            sys.stdout = original_stdout

    # Process labels
    # In-gate: Live, Out-gate: Debris, Outliers
    df['Label'] = (df['TrueLabel'] == 'Live').astype(int)
    df['DatasetID'] = i
    
    # Add Log columns for convenience
    df["Log_FSC"] = np.log10(df["FSC-A"] + 1e-1)
    df["Log_SSC"] = np.log10(df["SSC-A"] + 1e-1)
    
    # Select columns
    keep_cols = ["DatasetID", "FSC-A", "SSC-A", "Log_FSC", "Log_SSC", "Label", "TrueLabel"]
    all_dfs.append(df[keep_cols])
    
    if (i+1) % 100 == 0:
        print(f"  Generated {i+1}/{N_DATASETS} datasets")

final_df = pd.concat(all_dfs, ignore_index=True)
print(f"Saving {len(final_df)} rows to {OUTPUT_FILE}...")
final_df.to_csv(OUTPUT_FILE, index=False)
print("Done.")
