import json
import os
import datetime
import numpy as np

class ActiveLearningLogger:
    def __init__(self, storage_dir="data/active_learning_data"):
        self.storage_dir = storage_dir
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

    def log_gate_feedback(self, sample_name, cell_line, x_channel, y_channel, gate_type, gate_params, user_decision):
        """
        Logs user interaction for active learning.

        Args:
            sample_name (str): ID of the sample.
            cell_line (str): The cell line identifier from metadata.
            x_channel (str): X axis channel name.
            y_channel (str): Y axis channel name.
            gate_type (str): 'GaussianMixture' or 'Polygon'.
            gate_params (dict): Parameters of the gate (e.g., GMM means/covars or Polygon vertices).
            user_decision (str): 'accepted' or 'manual_override'.
        """
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "sample_name": sample_name,
            "cell_line": cell_line,
            "channels": [x_channel, y_channel],
            "gate_type": gate_type,
            "gate_params": self._serialize(gate_params),
            "user_decision": user_decision
        }

        filename = f"{self.storage_dir}/feedback_{datetime.datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(filename, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except (PermissionError, OSError) as e:
            # Silently fail or log to stderr if feedback cannot be saved to avoid blocking the UI
            print(f"Active Learning Logging Failed: {e}")

    def load_feedback(self):
        """
        Loads all feedback from the storage directory.
        Returns a list of dicts.
        """
        all_feedback = []
        if not os.path.exists(self.storage_dir):
            return all_feedback

        for filename in sorted(os.listdir(self.storage_dir)):
            if filename.startswith("feedback_") and filename.endswith(".jsonl"):
                full_path = os.path.join(self.storage_dir, filename)
                with open(full_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            try:
                                all_feedback.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
        return all_feedback

    def _serialize(self, obj):
        """Helper to handle numpy types for JSON"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return obj
