import sys
import os
import pandas as pd
import numpy as np

# Add workspace root to path
sys.path.append(os.getcwd())

from glow.data_loading.loader import DataLoader

def main():
    source_dir = '/home/xwz/Projects/Flow/flow-data/dhy_20251001_fcs/'
    output_file = 'data/real_data.parquet'
    
    print(f"Scanning {source_dir}...")
    loader = DataLoader()
    paths = loader.scan_directory(source_dir)
    
    if not paths:
        print("No FCS files found.")
        return

    print(f"Found {len(paths)} files.")
    
    # Prepare metadata
    meta = loader.prepare_fcs_files(paths)
    
    print("Loading into Cytoflow Experiment...")
    experiment = loader.load_as_cytoflow_experiment(meta)
    
    if experiment is None:
        print("Failed to load experiment.")
        return

    df = experiment.data
    df = df.reset_index(drop=True)
    
    print(f"Loaded DataFrame with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    rename_map = {
        'FSC 488/10-A': 'FSC-A',
        'SSC 488/10-A': 'SSC-A'
    }
    
    actual_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    if not actual_rename:
         print("Warning: Standard FSC/SSC columns not found to rename.")
         print("Attempting fuzzy match...")
         for col in df.columns:
             if 'FSC' in col and '-A' in col:
                 actual_rename[col] = 'FSC-A'
             if 'SSC' in col and '-A' in col:
                 actual_rename[col] = 'SSC-A'
    
    if actual_rename:
         print(f"Renaming columns: {actual_rename}")
         df = df.rename(columns=actual_rename)
    else:
        print("ERROR: Could not identify FSC-A and SSC-A columns.")
        return

    if 'TrueLabel' not in df.columns:
        df['TrueLabel'] = 'Unknown'
        
    if 'Sample' in df.columns:
        df['Sample'] = df['Sample'].astype(str)
        unique_samples = sorted(df['Sample'].unique())
        print(f"Found samples: {unique_samples}")
        
        sample_to_id = {s: i for i, s in enumerate(unique_samples)}
        df['DatasetID'] = df['Sample'].map(sample_to_id)
        print(f"Mapped {len(unique_samples)} samples to DatasetIDs.")
    else:
        print("Warning: No 'Sample' column found. Assigning DatasetID=0 to all.")
        df['DatasetID'] = 0
        
    df['DatasetID'] = df['DatasetID'].astype(int)

    print(f"Saving to {output_file}...")
    try:
        df.to_parquet(output_file, index=False)
        print("Done.")
    except Exception as e:
        print(f"Error saving parquet: {e}")

if __name__ == "__main__":
    main()
