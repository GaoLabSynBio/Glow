from typing import List, Optional
from .gating_strategies import (
    CompensationManager,
    GatingStrategy,
    GMMGating,
    DensityGating,
    DBSCANGating,
    PolygonGating,
    RangeGating,
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

    def remove_duplicates(self, experiment, fsc_h_channel="FSC-H", fsc_a_channel="FSC-A", linearity_threshold=0.1):
        """
        Removes doublets/duplicates.
        Uses GMMGating (1 component) effectively acting as a Singlet Finder on the diagonal.
        """
        strategy = GMMGating(name="Singlets",
                             channels=[fsc_a_channel, fsc_h_channel],
                             num_components=1)
        return strategy.apply(experiment)

    def auto_gate_live_cells(self, experiment, x_channel="FSC-A", y_channel="SSC-A", method="density", num_components=2, keep=0.9, sigma=2.0, outlier_low=0.01, outlier_high=0.99, return_model=False, use_outlier_removal=False, eps=0.05, min_samples=50):
        """
        Uses AutoGatingStrategy (GMM, Density).
        method: 'gmm', 'density', or 'smart' (legacy alias for density + outliers).
        """
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
            strategy = DBSCANGating(name="AutoLiveGate",
                                    channels=[x_channel, y_channel],
                                    eps=eps,
                                    min_samples=min_samples,
                                    outlier_detector=outlier_detector)
        elif method == 'gmm':
            strategy = GMMGating(name="AutoLiveGate",
                                 channels=[x_channel, y_channel],
                                 num_components=num_components,
                                 return_model=return_model,
                                 outlier_detector=outlier_detector)
        elif method == 'density':
            strategy = DensityGating(name="AutoLiveGate",
                                     channels=[x_channel, y_channel],
                                     keep=keep,
                                     sigma=sigma,
                                     outlier_detector=outlier_detector)
        else:
            raise ValueError(f"Unknown gating method: {method}")

        return strategy.apply(experiment)

    def apply_manual_gate(self, experiment, name, x_channel, y_channel, vertices):
        """
        Uses ManualGatingStrategy (Polygon).
        """
        strategy = PolygonGating(name=name,
                                 channels=[x_channel, y_channel],
                                 vertices=vertices)
        return strategy.apply(experiment)

    def apply_density_gate(self, experiment, name, x_channel, y_channel, keep=0.9, sigma=2.0):
        """
        Uses AutoGatingStrategy (Density implementation).
        """
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
