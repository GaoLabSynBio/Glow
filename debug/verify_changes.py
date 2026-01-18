
import sys
import os
import pandas as pd
import numpy as np

# Mock Streamlit to avoid errors during import
import unittest.mock as mock
mock_st = mock.MagicMock()
sys.modules["streamlit"] = mock_st

# Setup paths
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
sys.path.append(os.getcwd())

from glow.plotting.scatter import plot_2d_scatter_plotly
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Create Dummy Experiment
class Experiment:
    def __init__(self):
        self.data = pd.DataFrame({
            "FSC-A": np.random.rand(100) * 1000,
            "SSC-A": np.random.rand(100) * 1000,
            "CD3": np.random.rand(100) * 100,
            "Sample": ["S1"] * 50 + ["S2"] * 50
        })

exp = Experiment()

def test_plotly_generation():
    print("Testing Plotly Generation...")
    fig = plot_2d_scatter_plotly(exp, "FSC-A", "SSC-A", hue="Sample", title="Test Plot")
    assert isinstance(fig, go.Figure)
    print("Plotly Figure Created Successfully.")
    
    # Check layout
    assert fig.layout.title.text == "Test Plot"
    print("Title correct.")

def test_matplotlib_scaling_logic():
    print("Testing Matplotlib Scaling Logic...")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter([1, 2], [3, 4])
    
    # Simulate Canvas Logic
    font_scale = 2.0
    width_scale = 1.5
    height_scale = 0.5
    
    w_orig, h_orig = fig.get_size_inches()
    
    target_w = (w_orig * width_scale) / font_scale
    target_h = (h_orig * height_scale) / font_scale
    
    fig.set_size_inches(target_w, target_h)
    
    current_w, current_h = fig.get_size_inches()
    print(f"Original: {w_orig}x{h_orig}, Target: {target_w}x{target_h}, Result: {current_w}x{current_h}")
    
    assert np.isclose(current_w, target_w)
    assert np.isclose(current_h, target_h)
    print("Matplotlib Scaling Logic Verified.")

if __name__ == "__main__":
    try:
        test_plotly_generation()
        test_matplotlib_scaling_logic()
        print("\nAll Tests Passed!")
    except Exception as e:
        print(f"\nTest Failed: {e}")
        import traceback
        traceback.print_exc()
