import pandas as pd
import numpy as np
import os
import glob
import sys
from tqdm import tqdm

def load_all_labeled_data(
    base_path="/home/xwz/Projects/test/Glow/training/labeled_data/",
    output_path="data/real_training_data.parquet"
):
    print(f"Scanning {base_path}...")

    # Define cell types to look for based on folder names
    # User mentioned: colo, hek, hug1n, snu
    # We can just look at all subdirectories that contain csv.gz files

    all_files = glob.glob(os.path.join(base_path, "*", "*.csv.gz"))
    if not all_files:
        print("No .csv.gz files found.")
        return

    print(f"Found {len(all_files)} files total.")

    dfs = []
    dataset_counter = 0

    # Mapping for consistent column names
    # Based on: filter found in 'zcat ...' output earlier
    fsc_col_candidates = ['FSC 488/10-A', 'FSC-A', 'FSC 488/10-A']
    ssc_col_candidates = ['SSC 488/10-A', 'SSC-A', 'SSC 488/10-A']
    label_col = 'cells'

    # We need to ensure we handle the cell type mapping correctly if the model expects specific strings.
    # The prompt implies using folder names as cell type names.

    for file_path in tqdm(all_files):
        try:
            # Extract metadata
            folder_name = os.path.basename(os.path.dirname(file_path))
            cell_type = folder_name.lower() # e.g. 'colo', 'hek'

            # Load CSV
            df = pd.read_csv(file_path)

            # Identify columns
            fsc_col = next((c for c in fsc_col_candidates if c in df.columns), None)
            ssc_col = next((c for c in ssc_col_candidates if c in df.columns), None)

            if not fsc_col or not ssc_col:
                # Fallback fuzzy search if exact match fails
                for c in df.columns:
                    if 'FSC' in c and '-A' in c: fsc_col = c
                    if 'SSC' in c and '-A' in c: ssc_col = c

            if not fsc_col or not ssc_col or label_col not in df.columns:
                print(f"Skipping {file_path}: Missing FSC/SSC/cells columns.")
                continue

            # Process Data
            # 1. Log transform features (with simple epsilon/cutoff for log)
            # Flow values can be negative or zero sometimes after compensation? usually raw is > 0.
            # We assume raw linear data here.

            # Filter valid positive data for Log
            valid_mask = (df[fsc_col] > 0) & (df[ssc_col] > 0)
            clean_df = df.loc[valid_mask].copy()

            if len(clean_df) == 0:
                continue

            clean_df['Log_FSC'] = np.log10(clean_df[fsc_col])
            clean_df['Log_SSC'] = np.log10(clean_df[ssc_col])

            # 2. Labels
            # 'cells' column is Boolean. True -> Live (1), False -> Debris (0)
            clean_df['Label'] = clean_df[label_col].astype(int)

            # 3. Metadata
            clean_df['DatasetID'] = dataset_counter
            clean_df['CellType'] = cell_type

            # Select only needed columns to save memory
            subset = clean_df[['Log_FSC', 'Log_SSC', 'Label', 'DatasetID', 'CellType']]

            # Optimize types
            subset['DatasetID'] = subset['DatasetID'].astype(np.int32)
            subset['Label'] = subset['Label'].astype(np.int8)
            subset['Log_FSC'] = subset['Log_FSC'].astype(np.float32)
            subset['Log_SSC'] = subset['Log_SSC'].astype(np.float32)

            dfs.append(subset)
            dataset_counter += 1

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if not dfs:
        print("No valid data extracted.")
        return

    print("Concatenating all datasets...")
    full_df = pd.concat(dfs, ignore_index=True)

    print(f"Total samples: {len(full_df)}")
    print(f"breakdown by CellType:\n{full_df['CellType'].value_counts()}")

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    full_df.to_parquet(output_path, index=False)
    print(f"Saved training data to {output_path}")

if __name__ == "__main__":
    load_all_labeled_data()
