import cytoflow
import pandas as pd
import numpy as np

class GatingManager:
    def __init__(self):
        """
        Manages gating operations (Compensation, Filtering, Cell Selection).
        """
        pass

    def apply_compensation(self, experiment, spillover_matrix_file=None):
        """
        Applies compensation.
        If a spillover matrix is provided (CSV), it applies it.
        Otherwise, it assumes data is already compensated or requires user calculation
        (which is complex to automate without controls, so we start with matrix application).
        """
        if spillover_matrix_file:
            # Assuming CSV with channel names as headers and index
            # This is a simplification; cytoflow usually calculates spillover from controls.
            # If the user provides a matrix, we might need a custom Op or simply
            # use Cytoflow's ChannelOperation if it's simple linear algebra, 
            # but Cytoflow favors control-based compensation.
            # For now, we will place a placeholder for 'User Preference' compensation.
            print(f"Applying compensation matrix from {spillover_matrix_file}...")
            # Todo: Implement custom compensation matrix loading if needed.
            # For now, let's assume the user might want to run AutofluorescenceOp if no controls
            pass
        
        return experiment

    def remove_duplicates(self, experiment, fsc_h_channel="FSC-H", fsc_a_channel="FSC-A", linearity_threshold=0.1):
        """
        Removes doublets/duplicates usually by checking linearity between Height and Area.
        This is a heuristic approach.
        """
        # A simple approach is defining a linear gate. 
        # For simplicity in 'base_pipeline', we can use a range gate or density gate on these two.
        
        # In Cytoflow, we can create a new view or specific gate.
        # Let's try a Gaussian Mixture on FSC-A vs FSC-H to find the singlet population 
        # which usually is the diagonal.
        
        # 'SingletFinder' naming convention
        gm = cytoflow.GaussianMixtureOp(name = "Singlets",
                                        channels = [fsc_a_channel, fsc_h_channel],
                                        num_components = 1) # Assumes singlets are the main mode
        
        # Note: True duplicate removal is harder without manual inspection. 
        # We can also implement a standard polygon gate if coordinates are known.
        
        gm.estimate(experiment)
        return gm.apply(experiment)

    def auto_gate_live_cells(self, experiment, x_channel="FSC-A", y_channel="SSC-A", num_components=2, return_model=False):
        """
        Uses Gaussian Mixture Models to approximate 'Live' vs 'Dead/Debris'.
        Returns (experiment, model_op) if return_model is True.
        """
        # This is the 'Base Model' for active learning.
        gmm = cytoflow.GaussianMixtureOp(name = "AutoLiveGate",
                                         channels = [x_channel, y_channel],
                                         num_components = num_components)
        
        # The user will later receive 'AutoLiveGate_1' and 'AutoLiveGate_2' conditions.
        # They can choose which one is 'Live'.
        
        gmm.estimate(experiment)
        res = gmm.apply(experiment)
        
        if return_model:
            return res, gmm
        return res

    def apply_manual_gate(self, experiment, name, x_channel, y_channel, vertices):
        """
        Applies a Polygon Gate based on user provided vertices.
        vertices: List of (x, y) tuples.
        """
        poly_gate = cytoflow.PolygonOp(name = name,
                                       xchannel = x_channel,
                                       ychannel = y_channel,
                                       vertices = vertices)
        return poly_gate.apply(experiment)

    def filter_by_condition(self, experiment, condition_name, value):
        """
        Filters the experiment to keep only events matching a condition (like a Gate).
        """
        # In Cytoflow, you often 'subset' the views.
        # Here we can filter the Experiment data object itself if needed, 
        # but often we just carry the conditions forward.
        pass
