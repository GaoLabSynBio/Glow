import cytoflow
import pandas as pd
import numpy as np
import logging
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from glow.compensation.manager import CompensationManager

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from glow.data_loading.loader import DataLoader

def run_test():
    print("Loading Files with Glow Loader...")
    
    loader = DataLoader()
    
    # Prepare preview metadata manually
    # tests/test_complex_upload/no_stain.fcs -> Unstained
    # tests/test_complex_upload/GFP_only_C10.fcs -> GFP_Control
    # tests/test_complex_upload/BFP_only.fcs -> BFP_Control
    
    meta_list = [
        {'original_path': 'tests/test_complex_upload/no_stain.fcs', 'display_name': 'Unstained', 'current_path': 'tests/test_complex_upload/no_stain.fcs', 'condition': 'Control'},
        {'original_path': 'tests/test_complex_upload/GFP_only_C10.fcs', 'display_name': 'GFP_Control', 'current_path': 'tests/test_complex_upload/GFP_only_C10.fcs', 'condition': 'Control'},
        {'original_path': 'tests/test_complex_upload/BFP_only.fcs', 'display_name': 'BFP_Control', 'current_path': 'tests/test_complex_upload/BFP_only.fcs', 'condition': 'Control'}
    ]
    
    experiment = loader.load_as_cytoflow_experiment(meta_list)
    
    if experiment is None:
        print("Loader failed.")
        return

    print(f"Loaded {len(experiment.data)} events.")
    
    # 2. Setup Manager
    manager = CompensationManager()
    channels = ['GFP-A', 'TagBFP-A'] 
    
    unstained_spec = {'Sample': 'Unstained'}
    controls_spec = {
        'GFP-A': {'Sample': 'GFP_Control'},
        'TagBFP-A': {'Sample': 'BFP_Control'}
    }
    
    # 3. Compute Matrix
    print("Computing Matrix...")
    matrix = manager.compute_spillover_matrix(experiment, unstained_spec, controls_spec, channels)
    print("\nSpillover Matrix:")
    print(matrix)
    
    # 4. Apply Compensation
    print("Applying Compensation...")
    comp_df = manager.apply_compensation(experiment.data, channels, rename=True)
    
    # 5. Check Results (GFP -> BFP Spillover)
    # We look at the GFP Control. 
    # Ideally: comp_TagBFP-A should be flat (same median as Unstained).
    
    # Get Unstained Median BFP
    mask_u = comp_df['Sample'] == 'Unstained'
    u_bfp = comp_df.loc[mask_u, 'comp_TagBFP-A'].median()
    
    # Get GFP Control Median BFP (Compensated)
    # Filter for high GFP events 
    mask_g_pos = (comp_df['Sample'] == 'GFP_Control') & (comp_df['GFP-A'] > 1e5)
    if mask_g_pos.sum() == 0:
        print("Warning: No GFP+ events found > 1e5")
        g_bfp = 0
    else:
        g_bfp = comp_df.loc[mask_g_pos, 'comp_TagBFP-A'].median()
        
    # Also check the "Negative" population of the GFP Control
    # To see if the baseline was shifted?
    mask_g_neg = (comp_df['Sample'] == 'GFP_Control') & (comp_df['GFP-A'] < 1e4) # Low GFP
    if mask_g_neg.sum() > 0:
        g_neg_bfp = comp_df.loc[mask_g_neg, 'comp_TagBFP-A'].median()
    else:
        g_neg_bfp = 0
    
    print(f"\n[Compensated Stats]")
    print(f"Unstained BFP Median:       {u_bfp:.2f}")
    print(f"GFP-Control (Pos) BFP Median: {g_bfp:.2f}")
    print(f"GFP-Control (Neg) BFP Median: {g_neg_bfp:.2f}")
    
    diff = g_bfp - u_bfp
    print(f"Difference (Pos - Unstained): {diff:.2f}")

    # Inspect the internal logic values by checking logs or hacking
    # We can check specific values here
    u_raw = experiment.data.loc[experiment.data['Sample']=='Unstained', channels].median()
    g_raw = experiment.data.loc[experiment.data['Sample']=='GFP_Control']
    
    print("\n[Raw Stats]")
    print("Unstained Medians:\n", u_raw)
    
    q_lo = g_raw['GFP-A'].quantile(0.2)
    g_neg = g_raw[g_raw['GFP-A'] <= q_lo][channels].median()
    print("GFP_Control Low (Internal Neg):\n", g_neg)

    # Internal Negative logic check
    internal_neg_bfp = g_neg['TagBFP-A']
    diff_internal = abs(g_bfp - internal_neg_bfp)
    print(f"Difference (Pos - Internal Neg): {diff_internal:.2f}")

    if abs(diff) < 2000:
        print("SUCCESS: Compensated to Unstained Baseline.")
    elif diff_internal < 25000: 
        print(f"SUCCESS: Compensated close to Sample-Specific Background ({internal_neg_bfp:.0f}).")
        print("Note: The shift from Unstained (15k) to Sample Neg (50k) is preserved.")
    elif g_bfp < u_bfp - 2000:
        print("FAIL: OVERCOMPENSATION (Dip > 2000 below Unstained)")
    else:
        print(f"FAIL: UNDERCOMPENSATION (Pos {g_bfp:.0f} != Neg {internal_neg_bfp:.0f})")
    
    # Check Max values
    max_gfp = experiment.data['GFP-A'].max()
    print(f"Max GFP: {max_gfp}")
    
    # Check Pos Stats used for Slope
    # Recalculate manually to see what Manager saw
    # Mocking logic
    gfp_tube = experiment.data[experiment.data['Sample']=='GFP_Control']
    q_hi = gfp_tube['GFP-A'].quantile(0.95)
    print(f"Q95 GFP: {q_hi}")
    
    top_5 = gfp_tube[gfp_tube['GFP-A'] >= q_hi]
    print(f"Top 5% GFP Median: {top_5['GFP-A'].median()}")
    print(f"Top 5% BFP Median: {top_5['TagBFP-A'].median()}")
    
    internal_low = gfp_tube[gfp_tube['GFP-A'] <= gfp_tube['GFP-A'].quantile(0.2)]
    print(f"Low 20% BFP Median: {internal_low['TagBFP-A'].median()}")
    
    # Calculate Slope manually
    num = top_5['TagBFP-A'].median() - internal_low['TagBFP-A'].median()
    denom = top_5['GFP-A'].median() - internal_low['GFP-A'].median()
    slope = num / denom
    print(f"Calculated Slope: {slope}")

if __name__ == "__main__":
    run_test()
