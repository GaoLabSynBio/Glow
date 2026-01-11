import pandas as pd
import numpy as np
from glow.analysis.positive import calculate_positive_stats

def test_positive_stats():
    # Load the test data
    csv_path = "./tests/test_data/experiment_full_population.csv.gz"
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Mock Experiment
    class MockExperiment:
        def __init__(self, data):
            self.data = data
            self.conditions = {'Sample': 'Sample'} # Adjust if Sample col exists
            # Create a simple mapping if Sample column exists, else 1 sample
            if 'Sample' not in self.data.columns and 'sample' not in self.data.columns:
                 self.data['Sample'] = 'TestSample'
                 
    experiment = MockExperiment(df)
    
    # Pick a channel - finding a numeric one
    exclude = ['cells', 'singlets', 'Sample', 'sample', 'Cell_Line', 'Replicate', 'Experiment_Name', 'Event Info']
    channels = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    
    if not channels:
        print("No numeric channels found.")
        return

    target_channel = channels[0]
    print(f"Testing on channel: {target_channel}")
    
    # ----------------------------------------------------
    # DIAGNOSTIC: Check Live/Singlet logic manually
    # ----------------------------------------------------
    total_events = len(df)
    
    # Check 'cells' column
    if 'cells' in df.columns:
        n_cells = df['cells'].sum()
        print(f"Column 'cells': {n_cells}/{total_events} ({(n_cells/total_events)*100:.2f}%)")
    else:
        print("Column 'cells' not found.")

    # Check 'singlets' column
    if 'singlets' in df.columns:
        n_singlets = df['singlets'].sum()
        print(f"Column 'singlets': {n_singlets}/{total_events} ({(n_singlets/total_events)*100:.2f}%)")
    else:
        print("Column 'singlets' not found.")

    # Check Intersection
    if 'cells' in df.columns and 'singlets' in df.columns:
        n_valid = (df['cells'] & df['singlets']).sum()
        print(f"Intersection (cells & singlets): {n_valid}/{total_events} ({(n_valid/total_events)*100:.2f}%)")
        print(f"Singlets % (of Live): {(n_valid/n_cells)*100:.2f}%")

    # ----------------------------------------------------
    # Run Tool Logic
    # ----------------------------------------------------
    
    # Data stats
    vals = df[target_channel].values

    min_v, max_v = vals.min(), vals.max()
    mean_v = vals.mean()
    
    print(f"Range: {min_v} - {max_v}, Mean: {mean_v}")
    
    # Test Thresholds
    # 1. Min - Should be near 100%
    t_min = min_v - 1
    res_min = calculate_positive_stats(experiment, target_channel, t_min)
    print(f"Threshold {t_min} (All > T): \n{res_min}")
    
    # 2. Max - Should be 0%
    t_max = max_v + 1
    res_max = calculate_positive_stats(experiment, target_channel, t_max)
    print(f"Threshold {t_max} (All > T): \n{res_max}")
    
    # 3. Mean - Should be mixed
    t_mean = mean_v
    res_mean = calculate_positive_stats(experiment, target_channel, t_mean)
    print(f"Threshold {t_mean} (All > T): \n{res_mean}")

if __name__ == "__main__":
    test_positive_stats()
