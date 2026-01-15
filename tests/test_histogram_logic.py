import unittest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os

# Adjust path to import glow
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from glow.analysis.positive import get_channel_data_for_samples, calculate_positive_stats
from glow.analysis.stats import get_summary_statistics

class MockExperiment:
    def __init__(self, data):
        self.data = data
        # Simulate that channels property might exist but we shouldn't use it in stats if it crashes
        # But here we just mock data.

    @property
    def conditions(self):
        # Simulate the crash typical of Cytoflow when metadata is out of sync
        raise KeyError("Metadata missing 'cells'")

class TestHistogramIntergration(unittest.TestCase):
    def setUp(self):
        # Create a dataframe with mixed sample types (some might read as int, some string?)
        # Standard flow data often has 'Sample' as a string column
        # Samples: S1 (normal), S2 (high dynamic range), S3 (negative/zero only)
        self.df = pd.DataFrame({
            'Sample': ['S1', 'S1', 'S2', 'S2', 'S3', 'S3', 'S4'],
            'GFP-A': [10.0, 100.0, 1000.0, 10000.0, -50.0, 0.0, np.nan],
            'cells': [True] * 7,
            'singlets': [True] * 7
        })
        self.experiment = MockExperiment(self.df)

    def test_end_to_end_data_flow(self):
        """Simulate the flow from Stats generation -> UI Selection -> Data Retrieval -> Plotting"""

        # 1. UI: Get Summary Stats to populate sample list
        stats_df = get_summary_statistics(self.experiment)
        sample_list = stats_df['Sample'].unique().tolist()

        print(f"\nUI Sample List: {sample_list}")

        # Verify samples are found
        self.assertIn('S1', sample_list)
        self.assertIn('S3', sample_list)

        # 2. UI: User selects 'S2' (High value) and 'S3' (Negative/Zero) and S4 (NaN)
        selected_samples = ['S2', 'S3', 'S4']
        curr_channel = 'GFP-A'

        # 3. UI: Fetch Data
        data_map = get_channel_data_for_samples(self.experiment, curr_channel, selected_samples)

        # Verify Data Map
        self.assertIn('S2', data_map)
        self.assertIn('S3', data_map)
        self.assertIn('S4', data_map)

        # Check values consistency
        s4_vals = data_map['S4']
        print(f"Data S4: {s4_vals}")

        # 4. UI Logic: Calculate Range for Slider
        all_values = []
        for s in data_map:
            # We must handle NaNs here if the UI does filtering
            # But the CURRENT UI code effectively trusts np.min/max

            # The UI flattens list then calls min/max (which propagates NaNs)
            all_values.extend(data_map[s])

        # If we have NaNs, np.min returns NaN
        # We need to simulate the FIX I'm about to make.
        # Clean NaNs
        all_values = [x for x in all_values if not np.isnan(x)]

        raw_min = float(np.min(all_values))
        raw_max = float(np.max(all_values))

        print(f"Raw Range: {raw_min} to {raw_max}")
        self.assertEqual(raw_min, -50.0)
        self.assertEqual(raw_max, 10000.0)


        # 5. UI Logic: Log Scale Filtering
        use_log = True
        pos_vals = [v for v in all_values if v > 0]

        # Check what happens if we force log
        if not pos_vals:
            use_log = False

        valid_min = float(np.min(pos_vals)) if pos_vals else raw_min
        valid_max = float(np.max(pos_vals)) if pos_vals else raw_max

        print(f"Log Range: {valid_min} to {valid_max}")
        self.assertEqual(valid_min, 1000.0)

        # 6. Plotly Object Creation (Smoke Test)
        fig = go.Figure()
        for s in selected_samples:
            vals = data_map[s]
            # UI Code: if use_log: vals = vals[vals > 0]
            if use_log:
                vals = vals[vals > 0]

            if len(vals) > 0:
                fig.add_trace(go.Histogram(x=vals, name=s))

        # If S3 was all negative/zero, it should not be in the plot if log is True
        # checking trace names
        trace_names = [t.name for t in fig.data]
        print(f"Traces in Figure: {trace_names}")

        self.assertIn('S2', trace_names)
        self.assertNotIn('S3', trace_names) # S3 (-50, 0) filtered out by log logic

        # 7. Verify Layout Range logic
        layout_range = None
        if use_log:
            if valid_min > 0:
                layout_range = [np.log10(valid_min), np.log10(valid_max)]

        print(f"Layout Range: {layout_range}")
        self.assertAlmostEqual(layout_range[0], 3.0) # log10(1000)
        self.assertAlmostEqual(layout_range[1], 4.0) # log10(10000)

if __name__ == '__main__':
    unittest.main()
