import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class CompensationManager:
    def __init__(self):
        self.spillover_matrix = None
        self.channels = []

    def compute_spillover_matrix(self, experiment, unstained_spec, controls_spec, channels):
        """
        Compute pairwise spillover based on single-stain controls.
        
        Args:
            experiment: The Cytoflow Experiment object (containing all data including controls)
            unstained_spec: Dict matching metadata to identify unstained sample (e.g. {'name': 'Blank.fcs'})
            controls_spec: Dict mapping channel to sample spec (e.g. {'FITC-A': {'name': 'FITC_Only.fcs'}})
            channels: List of channels to compensate.
        
        Returns:
            pd.DataFrame: Spillover matrix (row=source, col=detector)
        """
        # Note: In a full Cytoflow implementation, we might use AutofluorescenceOp or BleedthroughLinearOp.
        # However, for flexibility and since we are operating in a specific UI flow, 
        # we will implement the standard "Slope Method" for spectral compensation/spillover calculation
        # which is mathematically equivalent to what most Cytometry software does for bead/cell controls.
        
        matrix = pd.DataFrame(np.eye(len(channels)), index=channels, columns=channels)

        # 1. Get Unstained Stats
        unstained_data = self._get_data_subset(experiment, unstained_spec)
        if unstained_data is None or len(unstained_data) < 100:
            raise ValueError("Unstained control has insufficient events.")
            
        unstained_medians = unstained_data[channels].median()

        # 2. Compute Coefficients per Channel
        for primer_ch in channels:
            if primer_ch not in controls_spec:
                continue # Should be caught by UI validation
                
            spec = controls_spec[primer_ch]
            pos_data = self._get_data_subset(experiment, spec)
            
            if pos_data is None or len(pos_data) < 100:
                logger.warning(f"Control for {primer_ch} missing or small. Skipping.")
                continue

            pos_medians = pos_data[channels].median()
            
            # Primary signal delta (Signal - Background)
            # This is the formatted intensity in the primary detector
            primary_delta = pos_medians[primer_ch] - unstained_medians[primer_ch]
            
            if primary_delta <= 1e-4: # Avoid division by zero
                logger.warning(f"Control for {primer_ch} has no signal over background.")
                continue

            # Calculate spillover into OTHER detectors
            for detector_ch in channels:
                if primer_ch == detector_ch:
                    continue
                    
                secondary_delta = pos_medians[detector_ch] - unstained_medians[detector_ch]
                
                # Slope = Secondary / Primary
                # This represents % spillover from Primer -> Detector
                spillover = secondary_delta / primary_delta
                
                # Values < 0 are physically impossible for spillover (unless background subtract issue), usually clamped to 0
                if spillover < 0:
                    spillover = 0
                
                # Matrix concept: 
                # Observed = Matrix * True
                # We are building the inverted matrix (compensation matrix) directly? 
                # No, usually we build the Spillover Matrix (S) where S[row=source, col=detector]
                # Then we invert it to get Compensation Matrix.
                
                # Rows = Source Fluorophore (The Control We Ran)
                # Cols = Detector (Where we see signal)
                matrix.loc[primer_ch, detector_ch] = spillover

        self.spillover_matrix = matrix
        self.channels = channels
        return matrix

    def get_compensation_matrix(self):
        """
        Invert the spillover matrix to get the compensation matrix.
        """
        if self.spillover_matrix is None:
            return None
        try:
            # Linear Algebra inverse
            comp_matrix = pd.DataFrame(
                np.linalg.inv(self.spillover_matrix.values), 
                index=self.spillover_matrix.index, 
                columns=self.spillover_matrix.columns
            )
            return comp_matrix
        except np.linalg.LinAlgError:
            logger.error("Singular matrix - cannot compensate.")
            return None

    def apply_compensation(self, df, channels, rename=True):
        """
        Apply compensation to dataframe.
        """
        comp_matrix = self.get_compensation_matrix()
        if comp_matrix is None:
            return df
        
        # We only work on the selected channels
        # Data Matrix (N x C)
        raw_data = df[channels].values
        
        # Comp Matrix (C x C)
        # Corrected = Raw * CompMatrix
        # (N x C) = (N x C) dot (C x C)
        # Check alignment:
        # matrix is (Source x Detector). Inverse is (Detector x TrueSource)?
        # Standard: True = Inv(S) * Observed^T ... 
        # Let's align with pandas dot result.
        
        compensated_data = raw_data.dot(comp_matrix.values) # This assumes comp_matrix is correctly oriented
        
        # With pandas:
        # df_comp = df[channels].dot(comp_matrix)
        
        comp_df = df.copy()
        
        for i, ch in enumerate(channels):
            new_col = f"comp_{ch}" if rename else ch
            comp_df[new_col] = compensated_data[:, i]
            
        return comp_df

    def _get_data_subset(self, experiment, spec):
        """
        Extract dataframe subset matching the spec.
        Spec example: {'Composite Name': 'Tube_001.fcs'}
        """
        # Assumption: Glow experiment data is a pandas DF in experiment.data
        # And metadata columns identify the samples.
        # Usually metadata keys like 'name' or 'filename' are columns in the big dataframe 
        # OR they are handled via MultiIndex.
        # Glow's current loader merges everything.
        
        df = experiment.data
        mask = pd.Series(True, index=df.index)
        
        for key, value in spec.items():
            if key in df.columns:
                mask = mask & (df[key] == value)
            elif key in experiment.metadata:
                # If it's experiment-wide metadata, it might not differentiate rows unless split
                pass
        
        subset = df[mask]
        if subset.empty:
            # Try partial matching on string columns if exact match failed?
            # For now, simplistic.
            return None
        return subset
