
import cytoflow
import pandas as pd
import numpy as np
from glow.gating.manager import GatingManager

def test_gating():
    print("Loading data...")
    tube = cytoflow.Tube(file='data/01-Well-A1.fcs')
    op = cytoflow.ImportOp(tubes=[tube])
    experiment = op.apply()
    
    print("Channels:", experiment.channels)
    
    # Verify data stats
    print(experiment.data[['FSC-A', 'SSC-A']].describe())
    
    gater = GatingManager()
    
    print("Attempting Auto Gating (Density)...")
    try:
        # Simulate empty experiment
        empty_data = experiment.data.iloc[:0]
        empty_exp = experiment.clone()
        empty_exp.data = empty_data
        
        gated_exp = gater.auto_gate_live_cells(
            empty_exp, 
            x_channel="FSC-A", 
            y_channel="SSC-A", 
            method="density",
            keep=0.9,
            sigma=2.0
        )
        print("Density Gating (Empty) Success")
    except Exception as e:
        print(f"Density Gating (Empty) Failed: {e}")

    print("Attempting Auto Gating (All NaN)...")
    try:
        # Simulate All NaN
        nan_data = experiment.data.copy()
        nan_data['FSC-A'] = np.nan
        nan_exp = experiment.clone()
        nan_exp.data = nan_data
        
        gated_exp = gater.auto_gate_live_cells(
            nan_exp, 
            x_channel="FSC-A", 
            y_channel="SSC-A", 
            method="density",
            keep=0.9,
            sigma=2.0
        )
        print("Density Gating (NaN) Success")
    except Exception as e:
        print(f"Density Gating (NaN) Failed: {e}")


    print("Attempting Auto Gating (GMM)...")
    try:
        gated_exp = gater.auto_gate_live_cells(
            experiment, 
            x_channel="FSC-A", 
            y_channel="SSC-A", 
            method="gmm",
            num_components=2
        )
        print("GMM Gating Success")
    except Exception as e:
        print(f"GMM Gating Failed: {e}")

if __name__ == "__main__":
    test_gating()
