import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

def generate_singlet_data(
    base_path="/home/xwz/Projects/test/Glow/training/labeled_data/",
    output_path="data/singlet_training_data.parquet"
):
    print(f"Scanning {base_path} for .csv.gz files...")
    all_files = glob.glob(os.path.join(base_path, "*", "*.csv.gz"))

    if not all_files:
        print("No .csv.gz files found.")
        return

    print(f"Found {len(all_files)} files total.")

    dfs = []
    dataset_counter = 0

    # Column candidates
    fsc_a_candidates = ['FSC 488/10-A', 'FSC-A']
    fsc_h_candidates = ['FSC 488/10-H', 'FSC-H']
    label_col = 'singlets'

    for file_path in tqdm(all_files):
        try:
            # Metadata
            folder_name = os.path.basename(os.path.dirname(file_path))

            # Load
            df = pd.read_csv(file_path)

            # Find columns
            fsc_a = next((c for c in fsc_a_candidates if c in df.columns), None)
            fsc_h = next((c for c in fsc_h_candidates if c in df.columns), None)

            # Fuzzy match fallback
            if not fsc_a:
                fsc_a = next((c for c in df.columns if 'FSC' in c and '-A' in c), None)
            if not fsc_h:
                fsc_h = next((c for c in df.columns if 'FSC' in c and '-H' in c), None)

            if not fsc_a or not fsc_h:
                # print(f"Skipping {file_path}: Missing FSC-A/H columns.")
                continue

            if label_col not in df.columns:
                # print(f"Skipping {file_path}: Missing 'singlets' label.")
                continue

            # Filter valid
            valid_mask = (df[fsc_a] > 0) & (df[fsc_h] > 0)

            # Filter live cells if 'cells' column exists
            if 'cells' in df.columns:
                valid_mask = valid_mask & (df['cells'].astype(bool))

            clean_df = df.loc[valid_mask].copy()

            if len(clean_df) == 0:
                continue

            # Log Transform
            clean_df['Log_FSC_A'] = np.log10(clean_df[fsc_a])
            clean_df['Log_FSC_H'] = np.log10(clean_df[fsc_h])

            # Label
            # 'singlets' might be boolean or int. Convert to 1 (Singlet), 0 (Doublet)
            clean_df['Label'] = clean_df[label_col].astype(int)

            clean_df['DatasetID'] = dataset_counter
            clean_df['Filename'] = os.path.basename(file_path)
            clean_df['CellType'] = folder_name.lower()

            subset = clean_df[['Log_FSC_A', 'Log_FSC_H', 'Label', 'DatasetID', 'CellType']]

            # Optimize types
            subset['DatasetID'] = subset['DatasetID'].astype(np.int32)
            subset['Label'] = subset['Label'].astype(np.int8)
            subset['Log_FSC_A'] = subset['Log_FSC_A'].astype(np.float32)
            subset['Log_FSC_H'] = subset['Log_FSC_H'].astype(np.float32)

            dfs.append(subset)
            dataset_counter += 1

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if not dfs:
        print("No valid data extracted.")
        return

    print("Concatenating...")
    full_df = pd.concat(dfs, ignore_index=True)

    print(f"Total samples: {len(full_df)}")
    print(f"Class Balance: {full_df['Label'].value_counts().to_dict()}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    full_df.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    generate_singlet_data()
