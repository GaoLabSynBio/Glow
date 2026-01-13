import json
import os
import datetime
import numpy as np
import pandas as pd

class ActiveLearningLogger:
    def __init__(self, storage_dir="data/active_learning_data"):
        self.storage_dir = storage_dir
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

    def log_gate_feedback(self, sample_name, cell_line, x_channel, y_channel, gate_type, gate_params, user_decision):
        """
        Logs user interaction for active learning.
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
            print(f"Active Learning Logging Failed: {e}")

    def save_labeled_batch(self, X, y, source="user_labeling"):
        """
        Saves a batch of labeled data (features + labels) to a Parquet file.
        Useful for accumulating training data for the neural network.
        
        Args:
            X (np.ndarray): N x 2 array of features (Log_FSC, Log_SSC).
            y (np.ndarray): N array of labels (0/1).
            source (str): Metadata describing where this data came from.
        """
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.storage_dir, f"labeled_batch_{timestamp}.parquet")
        
        df = pd.DataFrame(X, columns=["Log_FSC", "Log_SSC"])
        df["Label"] = y
        df["Source"] = source
        df["Timestamp"] = datetime.datetime.now().isoformat()
        
        try:
            df.to_parquet(filename, index=False)
            print(f"Saved labeled batch to {filename} ({len(df)} samples)")
            return filename
        except Exception as e:
            print(f"Failed to save labeled batch: {e}")
            return None

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
    
    def load_all_labeled_data(self):
        """
        Loads and concatenates all saving labeled batches.
        Returns X, y arrays.
        """
        dfs = []
        if not os.path.exists(self.storage_dir):
             return None, None
             
        for filename in sorted(os.listdir(self.storage_dir)):
            if filename.startswith("labeled_batch_") and filename.endswith(".parquet"):
                try:
                    df = pd.read_parquet(os.path.join(self.storage_dir, filename))
                    dfs.append(df)
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
                    
        if not dfs:
            return None, None
            
        full_df = pd.concat(dfs, ignore_index=True)
        X = full_df[["Log_FSC", "Log_SSC"]].values
        y = full_df["Label"].values
        return X, y

    def _serialize(self, obj):
        """Helper to handle numpy types for JSON"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return obj
