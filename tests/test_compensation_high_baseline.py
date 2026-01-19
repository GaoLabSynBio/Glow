import unittest
import pandas as pd
import numpy as np
import logging
from glow.compensation.manager import CompensationManager

# Configure logging to see output during tests if needed
logging.basicConfig(level=logging.DEBUG)

class TestCompensationHighBaseline(unittest.TestCase):
    def setUp(self):
        self.manager = CompensationManager()
        self.channels = ['FITC-A', 'PE-A']
        self.baseline_val = 10000.0
        
        # 1. Simulate Data with High Baseline
        # Unstained: Gaussian around 10,000
        self.df_unstained = pd.DataFrame({
            'FITC-A': np.random.normal(self.baseline_val, 100, 1000),
            'PE-A': np.random.normal(self.baseline_val, 100, 1000),
            'Sample': 'Unstained'
        })
        
        # FITC Single Stain:
        # FITC High (True signal ~100,000 + Baseline)
        # PE Spillover = 20% of (FITC_Observed - Baseline)
        # Observed PE = Baseline + 0.2 * (True_FITC)
        
        # Let's construct it carefully:
        # True FITC Signal = Random(100,000)
        true_fitc = np.random.normal(100000, 5000, 1000)
        
        obs_fitc = true_fitc + self.baseline_val
        # PE sees 20% of the True FITC signal + Baseline
        obs_pe = (true_fitc * 0.2) + self.baseline_val
        
        self.df_fitc = pd.DataFrame({
            'FITC-A': obs_fitc,
            'PE-A': obs_pe,
            'Sample': 'FITC_Control'
        })
        
        # PE Single Stain: (Assume no spill to FITC for simplicity, or 0% spill)
        true_pe = np.random.normal(100000, 5000, 1000)
        obs_pe_2 = true_pe + self.baseline_val
        obs_fitc_2 = self.baseline_val # No spill
        
        self.df_pe = pd.DataFrame({
            'FITC-A': obs_fitc_2,
            'PE-A': obs_pe_2,
            'Sample': 'PE_Control'
        })
        
        # Combine into Experiment-like structure
        self.full_df = pd.concat([self.df_unstained, self.df_fitc, self.df_pe], ignore_index=True)
        
        # Mock Experiment object (needs to have .data attribute)
        class MockExperiment:
            def __init__(self, data):
                self.data = data
        self.experiment = MockExperiment(self.full_df)
        
        self.unstained_spec = {'Sample': 'Unstained'}
        self.controls_spec = {
            'FITC-A': {'Sample': 'FITC_Control'},
            'PE-A': {'Sample': 'PE_Control'}
        }

    def test_matrix_calculation_and_application(self):
        # 1. Compute Matrix
        matrix = self.manager.compute_spillover_matrix(
            self.experiment, 
            self.unstained_spec, 
            self.controls_spec, 
            self.channels
        )
        
        print("\nComputed Spillover Matrix:")
        print(matrix)
        
        # Check FITC -> PE spillover (should be approx 0.2)
        spill_fitc_pe = matrix.loc['FITC-A', 'PE-A']
        self.assertAlmostEqual(spill_fitc_pe, 0.2, delta=0.02, msg=f"Spillover extraction failed. Got {spill_fitc_pe} expected 0.2")
        
        # Check integrity of baseline storage
        self.assertIsNotNone(self.manager.baseline, "Baseline was not stored in manager")
        base_fitc = self.manager.baseline['FITC-A']
        self.assertAlmostEqual(base_fitc, 10000, delta=200, msg="Baseline estimation incorrect")

        # 2. Apply Compensation
        # We define "correct" as:
        # - Unstained events remain centered at ~10,000 (NOT 8,000)
        # - FITC events have PE centered at ~10,000
        
        comp_df = self.manager.apply_compensation(self.full_df, self.channels, rename=True)
        
        # A. Check Unstained
        # In original, Unstained PE mean is ~10,000
        # In compensated, it MUST be ~10,000. 
        # If naive compensation happened: 10000 - (10000*0.2) = 8000.
        
        mask_u = comp_df['Sample'] == 'Unstained'
        mean_pe_unstained = comp_df.loc[mask_u, 'comp_PE-A'].mean()
        
        print(f"\nUnstained PE Mean (Compensated): {mean_pe_unstained}")
        
        # Allow small tolerance
        self.assertTrue(9800 < mean_pe_unstained < 10200, 
                        f"Unstained Baseline shifted significantly! Value: {mean_pe_unstained}. Expected ~10000.")

        # B. Check FITC Control (Compensated)
        # It should now have PE ~ 10000 (spillover removed)
        mask_f = comp_df['Sample'] == 'FITC_Control'
        mean_pe_fitc = comp_df.loc[mask_f, 'comp_PE-A'].mean()
        
        print(f"FITC Control PE Mean (Compensated): {mean_pe_fitc}")
        
        self.assertTrue(9500 < mean_pe_fitc < 10500, 
                        f"Spillover not correctly removed. Value: {mean_pe_fitc}. Expected ~10000.")

if __name__ == '__main__':
    unittest.main()
