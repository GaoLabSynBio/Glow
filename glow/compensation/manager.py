import pandas as pd
import numpy as np
import logging
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)

class CompensationManager:
    def __init__(self):
        self.spillover_matrix = None
        self.channels = []
        self.baseline = None # Store baseline (autofluorescence/offset) for correction

    def compute_spillover_matrix(self, experiment, unstained_spec, controls_spec, channels):
        """
        Compute pairwise spillover using Median Ratio.
        This reimplements cytoflow.BleedthroughLinearOp estimation logic adaptation for 
        in-memory experiments where control files are not reloadable.
        
        Args:
            experiment: The Cytoflow Experiment object (containing all data including controls)
            unstained_spec: Dict identifying unstained sample. 
            controls_spec: Dict mapping channel to sample spec.
            channels: List of channels to compensate.
        
        Returns:
            pd.DataFrame: Spillover matrix (row=source, col=detector)
        """
        # Initialize identity matrix
        matrix = pd.DataFrame(np.eye(len(channels)), index=channels, columns=channels)
        
        # We need the unstained control for background subtraction?
        # Cytoflow BleedthroughLinearOp does NOT explicitly subtract Unstained unless 
        # an AutofluorescenceOp was applied previously in the history.
        # However, curve_fit with intercept=0 assumes background is 0.
        # If we don't subtract background, and background is non-zero, slope will be biased.
        # Cytoflow docs say: "The module can also estimate... spillover... usually after correcting for autofluorescence".
        # If we stick to pure BleedthroughLinearOp logic, we just fit the slope on the raw control data?
        # But if raw data has offset, slope is wrong.
        # Let's subtract the Median Unstained Intensity from the Control Data before fitting.
        # This is a safe hybrid approach.

        # 1. Get Unstained Stats for Background Subtraction
        # Required for correct slope estimation via fit-through-origin
        # If available, this provides the "Negative" anchor (0,0 point)
        unstained_medians = None
        if unstained_spec:
            try:
                unstained_data = self._get_data_subset(experiment, unstained_spec)
                if unstained_data is not None and len(unstained_data) > 100:
                    unstained_medians = unstained_data[channels].median()
            except Exception:
                logger.warning("Could not calculate background from unstained control.")
                
        # Store baseline for application phase
        # This is critical for data with high offsets (e.g. 10^4) to prevent "overcompensation" 
        # where the offset itself is compensated down.
        self.baseline = unstained_medians

        for from_channel in channels:
            if from_channel not in controls_spec:
                continue

            # Get Control Data
            spec = controls_spec[from_channel]
            tube_data = self._get_data_subset(experiment, spec)

            if tube_data is None or len(tube_data) < 100:
                logger.warning(f"Control for {from_channel} missing/small. Skipping.")
                continue
            
            # Work on a copy
            tube_data = tube_data.copy()

            # METHOD: Median Ratio with Dynamic Negative Reference
            # Overcompensation often occurs when "Unstained" background is lower than the actual 
            # autofluorescence of the single-stain tube (e.g. different cells, different prep).
            # To be robust, we preferentially use the INTERNAL negative population of the control tube.
            
            # 1. Analyze Distributions
            # Calculate percentiles to find dynamic range
            q_lo = tube_data[from_channel].quantile(0.20)
            q_hi = tube_data[from_channel].quantile(0.95)
            
            internal_neg_medians = None
            use_internal_neg = False
            
            
            # Logic: Prefer Internal Negative to account for tube-specific autofluorescence.
            # Only fall back to Unstained if the tube appears to be "All Positive" (Internal Low is very bright)
            # or if the tube has no events.
            
            neg_medians = None
            use_internal = False
            
            lower_subset = tube_data[tube_data[from_channel] <= q_lo]
            
            # Can we find a valid internal negative?
            if len(lower_subset) > 10:
                internal_neg = lower_subset[channels].median()
                
                if unstained_medians is not None:
                    # Comparison check
                    # Ratio of Internal Low / Unstained
                    # 1.0 = Match
                    # 1.2 = Slight AF shift (Common) -> Should use Internal
                    # 10.0 = Likely Positive (Beads) -> Should use Unstained
                    
                    # We use a permissive threshold (e.g. 5.0) to allow for significant AF variation
                    # without triggering the "All Positive" fallback.
                    # This is key to fixing the overcompensation artifact caused by subtracting 
                    # a too-low Unstained baseline from a higher-AF tube.
                    
                    # Safe-guard for zero-div
                    denom = unstained_medians[from_channel] if unstained_medians[from_channel] > 1 else 1.0
                    ratio = internal_neg[from_channel] / denom
                    
                    if ratio < 5.0:
                        neg_medians = internal_neg
                        use_internal = True
                    else:
                        logger.info(f"Control {from_channel}: Internal Low is {ratio:.1f}x brighter than Unstained. Assuming all-positive. Using Unstained.")
                        neg_medians = unstained_medians
                else:
                    neg_medians = internal_neg
            else:
                 neg_medians = unstained_medians

            if neg_medians is None:
                # Fallback purely
                 neg_medians = tube_data[channels].min()

            # 2. Determine Positive Population (P)
            # Take top 5%
            pos_subset = tube_data[tube_data[from_channel] >= q_hi]
            
            if len(pos_subset) < 10:
                logger.warning(f"Not enough positive events in control {from_channel}")
                continue
                
            pos_medians = pos_subset[channels].median()
            
            # 3. Calculate Slopes
            # Delta = Pos - Neg
            delta_primary = pos_medians[from_channel] - neg_medians[from_channel]
            
            if delta_primary <= 1.0:
                 # Avoid div zero
                 continue

            for to_channel in channels:
                if from_channel == to_channel:
                    matrix.loc[from_channel, to_channel] = 1.0
                    continue
                
                delta_secondary = pos_medians[to_channel] - neg_medians[to_channel]
                
                # Robustness: If secondary delta is negative (Pos < Neg), it implies No Spillover (or noise).
                # Clamp to 0.
                if delta_secondary < 0:
                    delta_secondary = 0
                
                slope = delta_secondary / delta_primary
                
                matrix.loc[from_channel, to_channel] = slope

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
        Apply compensation to dataframe using the computed matrix.
        Corrects for baseline offset if available.
        """
        comp_matrix = self.get_compensation_matrix()
        if comp_matrix is None:
            return df
        
        # Only extract selected channels
        raw_data = df[channels].values
        
        # Baseline Correction Strategy:
        # If we have a known background/offset (self.baseline) which is high (e.g. 10^4),
        # we must subtract it before applying the matrix, otherwise the offset itself (10^4)
        # gets "compensated", causing values to drop significantly below the baseline level.
        # Logic: 
        # 1. Delta = Raw - Baseline
        # 2. CompDelta = Delta * InvMatrix
        # 3. Final = CompDelta + Baseline
        
        if self.baseline is not None:
            try:
                # Ensure we have baseline values for all requested channels
                # Re-index to match 'channels' list order
                missing = [c for c in channels if c not in self.baseline.index]
                if missing:
                    raise KeyError(f"Baseline missing for channels: {missing}")
                
                b_vec = self.baseline[channels].values
                
                # Check for NaNs
                if np.isnan(b_vec).any():
                     raise ValueError("Baseline contains NaNs")

                logger.info(f"Applying baseline correction. Vector: {b_vec}")

                # Subtract Baseline
                data_for_comp = raw_data - b_vec
                
                # Apply Matrix
                comp_data = data_for_comp.dot(comp_matrix.values)
                
                # Add Baseline Back
                compensated_data = comp_data + b_vec
                
            except Exception as e:
                logger.error(f"Baseline correction failed: {e}. Falling back to standard compensation.")
                print(f"ERROR: Baseline correction failed: {e}") # Print to visible console
                compensated_data = raw_data.dot(comp_matrix.values)
        else:
            logger.warning("No baseline available for correction. Using standard compensation (risk of overcompensation).")
            print("WARNING: No baseline available. using standard compensation.")
            # Fallback to standard
            compensated_data = raw_data.dot(comp_matrix.values) 
        
        comp_df = df.copy()
        for i, ch in enumerate(channels):
            # Use 'comp_' prefix by default (snake_case)
            # Renaming to 'Comp-' is handled by the UI Analysis tab
            new_col = f"comp_{ch}" if rename else ch
            comp_df[new_col] = compensated_data[:, i]
            
        return comp_df

    def _get_data_subset(self, experiment, spec):
        """
        Helper to retrieve data for a specific sample/control.
        """
        if 'Sample' not in spec:
            return None
        
        sample_name = spec['Sample']
        if experiment is None or experiment.data is None:
            return None
            
        # Filter dataframe by Sample column
        if 'Sample' in experiment.data.columns:
            return experiment.data[experiment.data['Sample'] == sample_name]
        
        return None

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
