from typing import List
from .strategies import (
    CompensationManager,
    GatingStrategy,
    GMMGating,
    DensityGating,
    DBSCANGating,
    MLPGating,
    MLPSingletGating,
    PolygonGating,
    OutlierDetector
)

# -----------------------------------------------------------------------------
# Gating Workflow
# -----------------------------------------------------------------------------

class GatingWorkflow:
    """
    Manages a Standard Gating Hierarchy.
    Typical Flow: Live/Dead -> Singlets -> Downstream
    """
    def __init__(self):
        self.steps: List[GatingStrategy] = []

    def add_step(self, strategy: GatingStrategy):
        self.steps.append(strategy)

    def run(self, experiment):
        """
        Runs specific steps in order.
        Note: Cytoflow operations often just add conditions to the object.
        So we pass the experiment through the chain.
        """
        current_exp = experiment
        for strategy in self.steps:
            # Handle potential tuple returns from strategies that return models
            result = strategy.apply(current_exp)
            if isinstance(result, tuple):
                current_exp = result[0]
                # If we need to store the model, we should handle it here
            else:
                current_exp = result
        return current_exp

# -----------------------------------------------------------------------------
# Gating Manager (Facade/Compatibility Layer)
# -----------------------------------------------------------------------------

class GatingManager:
    """
    Central facade for gating operations.
    Maintains backward compatibility with existing pipeline code while using
    the new hierarchy.
    """
    def __init__(self):
        self.compensation_manager = CompensationManager()
        # The workflow instance could be used for building complex pipelines
        self.workflow = GatingWorkflow()

    def apply_compensation(self, experiment, spillover_matrix_file=None):
        return self.compensation_manager.apply_compensation(experiment, spillover_matrix_file)

    def _resolve_channel(self, experiment, channel_name: str, variants: List[str] = None) -> str:
        """
        Helper to resolve channel names flexibly (case-insensitive, variants).
        """
        if channel_name in experiment.channels:
            return channel_name

        lower_map = {c.lower(): c for c in experiment.channels}
        
        if channel_name.lower() in lower_map:
            return lower_map[channel_name.lower()]

        if variants:
            for v in variants:
                if v in experiment.channels:
                    return v
                if v.lower() in lower_map:
                    return lower_map[v.lower()]
        
        # Fallback: Try to find something that looks like the requested channel
        # e.g. "FSC-A" -> look for "FSC" and "A" or "Area"
        # This is a bit of a heuristic for common flow cytometry naming
        base = channel_name.upper()
        candidates = []
        if "FSC" in base or "FORWARD" in base:
            if "A" in base or "AREA" in base:
                candidates = ["FSC-Area", "FS-Area", "Forward Scatter-Area", "FSC-A", "FS-A"]
            elif "H" in base or "HEIGHT" in base:
                candidates = ["FSC-Height", "FS-Height", "Forward Scatter-Height", "FSC-H", "FS-H"]
            elif "W" in base or "WIDTH" in base:
                candidates = ["FSC-Width", "FS-Width", "Forward Scatter-Width", "FSC-W", "FS-W"]
        elif "SSC" in base or "SIDE" in base:
             if "A" in base or "AREA" in base:
                candidates = ["SSC-Area", "SS-Area", "Side Scatter-Area", "SSC-A", "SS-A"]
             elif "H" in base or "HEIGHT" in base:
                candidates = ["SSC-Height", "SS-Height", "Side Scatter-Height", "SSC-H", "SS-H"]
             elif "W" in base or "WIDTH" in base:
                candidates = ["SSC-Width", "SS-Width", "Side Scatter-Width", "SSC-W", "SS-W"]

        for c in candidates:
            if c in experiment.channels:
                return c
            if c.lower() in lower_map:
                return lower_map[c.lower()]

        return channel_name

    def remove_duplicates(self, experiment, fsc_h_channel="FSC-H", fsc_a_channel="FSC-A", linearity_threshold=0.1, method="gmm", cell_type="colo"):
        """
        Removes doublets/duplicates.
        """
        # Resolve channels
        fsc_h_channel = self._resolve_channel(experiment, fsc_h_channel, ["FSC-Height", "FS-H", "FS-Height"])
        fsc_a_channel = self._resolve_channel(experiment, fsc_a_channel, ["FSC-Area", "FS-A", "FS-Area"])

        if method == "gmm":
            strategy = GMMGating(name="_temp_singlet_mask_auto",
                                 channels=[fsc_a_channel, fsc_h_channel],
                                 num_components=1)
        elif method == "mlp":
            strategy = MLPSingletGating(name="_temp_singlet_mask_auto",
                                        channels=[fsc_a_channel, fsc_h_channel],
                                        cell_type=cell_type)
        else:
            raise ValueError(f"Unknown singlet gating method: {method}")

        return strategy.apply(experiment)

    def auto_gate_live_cells(self, experiment, x_channel="FSC-A", y_channel="SSC-A", method="density", num_components=2, keep=0.9, sigma=2.0, outlier_low=0.01, outlier_high=0.99, return_model=False, use_outlier_removal=False, eps=0.05, min_samples=50, cell_type="colo"):
        """
        Uses AutoGatingStrategy (GMM, Density).
        method: 'gmm', 'density', or 'smart' (legacy alias for density + outliers).
        """
        # Resolve channels
        x_channel = self._resolve_channel(experiment, x_channel, ["FSC-Area", "FS-A", "FS-Area"])
        y_channel = self._resolve_channel(experiment, y_channel, ["SSC-Area", "SS-A", "SS-Area"])

        outlier_detector = None

        # Handle 'smart' alias or explicit request
        if method == 'smart':
            method = 'density'
            use_outlier_removal = True

        if use_outlier_removal:
            outlier_detector = OutlierDetector(method='isolation_forest',
                                               outlier_low=outlier_low,
                                               outlier_high=outlier_high)

        if method == 'dbscan':
            strategy = DBSCANGating(name="_temp_live_labels",
                                    channels=[x_channel, y_channel],
                                    eps=eps,
                                    min_samples=min_samples,
                                    outlier_detector=outlier_detector)
        elif method == 'gmm':
            strategy = GMMGating(name="_temp_live_labels",
                                 channels=[x_channel, y_channel],
                                 num_components=num_components,
                                 return_model=return_model,
                                 outlier_detector=outlier_detector)
        elif method == 'density':
            strategy = DensityGating(name="_temp_live_labels",
                                     channels=[x_channel, y_channel],
                                     keep=keep,
                                     sigma=sigma,
                                     outlier_detector=outlier_detector)
        elif method == 'mlp':
            strategy = MLPGating(name="_temp_live_labels",
                                 channels=[x_channel, y_channel],
                                 outlier_detector=outlier_detector,
                                 cell_type=cell_type)
        else:
            raise ValueError(f"Unknown gating method: {method}")

        return strategy.apply(experiment)

    def apply_manual_gate(self, experiment, name, x_channel, y_channel, vertices):
        """
        Uses ManualGatingStrategy (Polygon).
        """
        x_channel = self._resolve_channel(experiment, x_channel)
        y_channel = self._resolve_channel(experiment, y_channel)

        strategy = PolygonGating(name=name,
                                 channels=[x_channel, y_channel],
                                 vertices=vertices)
        return strategy.apply(experiment)

    def apply_density_gate(self, experiment, name, x_channel, y_channel, keep=0.9, sigma=2.0):
        """
        Uses AutoGatingStrategy (Density implementation).
        """
        x_channel = self._resolve_channel(experiment, x_channel)
        y_channel = self._resolve_channel(experiment, y_channel)

        strategy = DensityGating(name=name,
                                 channels=[x_channel, y_channel],
                                 keep=keep,
                                 sigma=sigma)
        return strategy.apply(experiment)

    def filter_by_condition(self, experiment, condition_name, value):
        """
        Filters the experiment.
        """
        pass
