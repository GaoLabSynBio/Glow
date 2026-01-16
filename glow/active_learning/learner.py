import os
import shutil
import numpy as np
import pandas as pd
from .storage import ActiveLearningLogger
from .models import GatingClassifier, SingletClassifier

class ActiveLearner:
    """
    Manages the Active Learning process:
    1. Loads the Neural Network model.
    2. Queries for uncertain samples.
    3. Updates the model with new user feedback.
    """
    def __init__(self, storage_dir="data/active_learning_data", model_path="data/gating_model_full.pkl", model_type="gating"):
        self.storage_dir = storage_dir
        self.model_path = model_path
        self.model_type = model_type

        # Select Class
        if model_type == 'singlet':
            self.model_class = SingletClassifier
        else:
            self.model_class = GatingClassifier

        # Ensure backup exists for reset functionality
        self.backup_path = self.model_path + ".base"
        if os.path.exists(self.model_path) and not os.path.exists(self.backup_path):
            print(f"Creating base backup at {self.backup_path}")
            shutil.copy2(self.model_path, self.backup_path)

        self.logger = ActiveLearningLogger(storage_dir)
        self.model = self._load_or_init_model()

    def _load_or_init_model(self):
        """Loads model from disk or creates valid fallback if missing."""
        if os.path.exists(self.model_path):
            try:
                print(f"Loading Active Learning model ({self.model_type}) from {self.model_path}")
                return self.model_class.load(self.model_path)
            except Exception as e:
                print(f"Error loading model: {e}")

        print(f"Initializing new/empty {self.model_class.__name__}.")
        return self.model_class()

    def reset_model(self):
        """
        Resets the model to the base state (from .base backup).
        Returns:
            bool: Success status.
        """
        if not os.path.exists(self.backup_path):
            print("Cannot reset: Base backup file not found.")
            return False

        print(f"Resetting model from {self.backup_path}...")
        try:
            # Restore file
            shutil.copy2(self.backup_path, self.model_path)
            # Reload object
            self.model = self.model_class.load(self.model_path)
            return True
        except Exception as e:
            print(f"Error resetting model: {e}")
            return False

    def predict(self, df):
        """
        Predicts labels for a dataframe.
        """
        X = self._extract_features(df)
        return self.model.predict(X)

    def query_uncertainty(self, df, n_instances=10):
        """
        Selects the top N most uncertain instances for labeling.
        """
        X = self._extract_features(df)
        uncertainties = self.model.uncertainty(X)
        query_indices = np.argsort(uncertainties)[::-1][:n_instances]
        return df.iloc[query_indices], query_indices

    def teach(self, X_new, y_new):
        """
        Updates the model with new labeled data.
        """
        if len(X_new) == 0:
            return

        print(f"Teaching model with {len(X_new)} new samples...")

        # 1. Update Model (Incremental)
        try:
            self.model.partial_fit(X_new, y_new)
        except Exception as e:
            print(f"Partial fit failed ({e}).")
            return

        # 2. Save Model checkpoint
        self.model.save(self.model_path)

        # 3. Log/Save this training batch
        self.logger.save_labeled_batch(X_new, y_new, source=f"active_learning_feedback_{self.model_type}")

    def _extract_features(self, df):
        """Helper to safely get Features based on model type"""
        if self.model_type == 'singlet':
            # Expecting Log_FSC_A, Log_FSC_H usually, or just 2 cols
            # If explicit columns exist, use them
            if "Log_FSC_A" in df.columns and "Log_FSC_H" in df.columns:
                 return df[["Log_FSC_A", "Log_FSC_H"]].values

            # Fallback: Just take the first 2 columns if not named specifically?
            # Or try to construct from standard names if passed
            pass

        # Default / Gating behavior
        if "Log_FSC" in df.columns:
             return df[["Log_FSC", "Log_SSC"]].values

        # Generic Fallback: Use values of dataframe directly if 2 columns
        if df.shape[1] == 2:
            return df.values

        raise ValueError(f"Dataframe missing required feature columns for {self.model_type}.")
