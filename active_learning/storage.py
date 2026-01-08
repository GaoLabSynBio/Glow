import json
import os
import datetime
import numpy as np

class ActiveLearningLogger:
    def __init__(self, storage_dir="active_learning_data"):
        self.storage_dir = storage_dir
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

    def log_gate_feedback(self, sample_name, x_channel, y_channel, gate_type, gate_params, user_decision):
        """
        Logs user interaction for active learning.
        
        Args:
            sample_name (str): ID of the sample.
            x_channel (str): X axis channel name.
            y_channel (str): Y axis channel name.
            gate_type (str): 'GaussianMixture' or 'Polygon'.
            gate_params (dict): Parameters of the gate (e.g., GMM means/covars or Polygon vertices).
            user_decision (str): 'accepted' or 'manual_override'.
        """
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "sample_name": sample_name,
            "channels": [x_channel, y_channel],
            "gate_type": gate_type,
            "gate_params": self._serialize(gate_params),
            "user_decision": user_decision
        }
        
        filename = f"{self.storage_dir}/feedback_{datetime.datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(filename, "a") as f:
            f.write(json.dumps(entry) + "\n")
            
    def _serialize(self, obj):
        """Helper to handle numpy types for JSON"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return obj
