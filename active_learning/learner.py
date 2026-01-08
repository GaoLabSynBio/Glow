import numpy as np

class ActiveLearner:
    def __init__(self, base_model=None):
        self.model = base_model
        self.queried_points = []
        
    def select_uncertain_samples(self, experiment, n_samples=10):
        """
        Selects samples from the experiment where the model is least certain.
        For a GMM, this might be points with probability ~0.5.
        """
        # Placeholder logic: random selection if no model
        # Real logic: get probabilities, find those near decision boundary
        pass

    def update_model(self, experiment, new_labels):
        """
        Updates the internal model based on new user labels.
        """
        pass
