import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class CompensationManager:
    def __init__(self):
        self.spillover_matrix = None
        self.channels = []
        self.baseline = None # Store baseline (autofluorescence/offset) for correction

    def compute_spillover_matrix(self, experiment, unstained_spec, controls_spec, channels):
        """
        Compute pairwise spillover using standard Median Ratio method.
        Robustly determines Positive and Negative populations for slope calculation.
        """
        # Initialize identity matrix
        matrix = pd.DataFrame(np.eye(len(channels)), index=channels, columns=channels)

        # Retrieve Unstained Reference (if available) as a fallback negative
        unstained_medians = None
        if unstained_spec:
            try:
                unstained_data = self._get_data_subset(experiment, unstained_spec)
                if unstained_data is not None and len(unstained_data) > 100:
                    unstained_medians = unstained_data[channels].median()
            except Exception:
                logger.warning("Could not calculate background from unstained control.")

        # Baseline is NOT stored or used for correction in standard mode 
        # to strictly follow standard compensation math (Raw * InvMatrix).
        self.baseline = None 

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

            # METHOD: Standard Median Ratio
            # Slope = (Pos_Signal - Neg_Signal) / (Pos_Primary - Neg_Primary)
            
            # 1. Determine Negative Reference
            # Priority: Internal Negative (bottom 10%) > Unstained Control
            q_lo = tube_data[from_channel].quantile(0.10)
            neg_subset = tube_data[tube_data[from_channel] <= q_lo]
            
            neg_medians = None
            if len(neg_subset) > 50:
                # Use internal negative if sufficient events exist
                neg_medians = neg_subset[channels].median()
            elif unstained_medians is not None:
                # Fallback to unstained if tube is "all positive" (e.g. bright beads)
                neg_medians = unstained_medians
            else:
                 # Last resort: minimums
                 neg_medians = tube_data[channels].min()

            # 2. Determine Positive Population
            # Top 2% (robust max)
            q_hi = tube_data[from_channel].quantile(0.98)
            pos_subset = tube_data[tube_data[from_channel] >= q_hi]

            if len(pos_subset) < 10:
                logger.warning(f"Not enough positive events in control {from_channel}")
                continue

            pos_medians = pos_subset[channels].median()

            # 3. Calculate Slopes
            # Delta = Pos - Neg
            delta_primary = pos_medians[from_channel] - neg_medians[from_channel]

            if delta_primary <= 1.0:
                 # Primary signal too weak or invalid
                 continue

            for to_channel in channels:
                if from_channel == to_channel:
                    matrix.loc[from_channel, to_channel] = 1.0
                    continue

                delta_secondary = pos_medians[to_channel] - neg_medians[to_channel]
                
                # Standard compensation calculates the ratio of the fluorescence in the detector channel
                # to the fluorescence in the primary channel.
                # If delta_secondary is negative, it implies the positive population is dimmer than 
                # the negative/unstained in this channel, which is physically impossible for spillover 
                # (unless noise). We clamp to 0.
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
        Apply compensation to dataframe using valid standard matrix multiplication.
        Assumes data is already properly scaled.
        """
        comp_matrix = self.get_compensation_matrix()
        if comp_matrix is None:
            return df

        # Only extract selected channels
        raw_data = df[channels].values
        
        # Standard Compensation: Data * Inverse(Spillover)
        # Any negative values resulting from Matrix Inverse applied to data with offsets
        # are mathematically correct in the context of compensation algebra.
        # Visualization should handle them (e.g. Biexponential/Logicle).
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
