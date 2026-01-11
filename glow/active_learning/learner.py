import os
import numpy as np
import pandas as pd
from .storage import ActiveLearningLogger
from .models import GatingClassifier

class ActiveLearner:
    """
    Manages the Active Learning process:
    1. Loads the Neural Network model.
    2. Queries for uncertain samples.
    3. Updates the model with new user feedback.
    """
    def __init__(self, storage_dir="data/active_learning_data", model_path="data/gating_model_full.pkl"):
        self.storage_dir = storage_dir
        self.model_path = model_path
        self.logger = ActiveLearningLogger(storage_dir)
        self.model = self._load_or_init_model()
        
    def _load_or_init_model(self):
        """Loads model from disk or creates valid fallback if missing."""
        if os.path.exists(self.model_path):
            try:
                print(f"Loading Active Learning model from {self.model_path}")
                return GatingClassifier.load(self.model_path)
            except Exception as e:
                print(f"Error loading model: {e}")
        
        print("Initializing new/empty GatingClassifier.")
        return GatingClassifier()

    def predict(self, df):
        """
        Predicts labels for a dataframe.
        Args:
            df (pd.DataFrame): Must contain ['Log_FSC', 'Log_SSC'] or transformable columns.
        Returns:
            np.ndarray: Predicted labels (0/1)
        """
        X = self._extract_features(df)
        return self.model.predict(X)

    def query_uncertainty(self, df, n_instances=10):
        """
        Selects the top N most uncertain instances for labeling.
        Strategy: Uncertainty Sampling (Margin).
        
        Returns:
            pd.DataFrame: Subset of df containing the uncertain instances.
            np.ndarray: Indices of these instances in the original df.
        """
        X = self._extract_features(df)
        uncertainties = self.model.uncertainty(X)
        
        # Get indices of top N uncertainties
        # argsort assumes ascending, so we take the end (largest uncertainty)
        # However, uncertainty() returns 1.0 (High Uncert) to 0.0 (Low).
        # So we sort usually.
        # sort descending
        query_indices = np.argsort(uncertainties)[::-1][:n_instances]
        
        return df.iloc[query_indices], query_indices

    def teach(self, X_new, y_new):
        """
        Updates the model with new labeled data.
        
        Args:
            X_new (np.ndarray): N x 2 features.
            y_new (np.ndarray): Labels.
        """
        if len(X_new) == 0:
            return

        print(f"Teaching model with {len(X_new)} new samples...")
        
        # 1. Update Model (Incremental)
        try:
            self.model.partial_fit(X_new, y_new)
        except Exception as e:
            print(f"Partial fit failed ({e}), attempting full retrain context if available (not implemented).")
            # In a real system, we might reload all old data + new data here.
            # For now, simplistic partial_fit.
            return

        # 2. Save Model checkpoint
        self.model.save(self.model_path)
        
        # 3. Log/Save this training batch for future full-retraining
        self.logger.save_labeled_batch(X_new, y_new, source="active_learning_feedback")
    
    def retrain_full(self):
        """
        Reloads ALL stored labeled data (initial + feedback) and fully retrains the model.
        Useful if partial_fit diverges or forgetting occurs.
        """
        print("Starting full retraining...")
        # 1. Load Initial Data (if we want to keep it frozen or mix it)
        # Usually we would have a 'seed' dataset path. 
        # For this setup, we assume load_all_labeled_data gets everything relevant stored in storage_dir.
        # OR we combine 'synthetic_training_data.parquet' + 'active_learning_data/*.parquet'.
        
        # Load accumulated feedback
        X_feed, y_feed = self.logger.load_all_labeled_data()
        
        if X_feed is None:
            print("No feedback data found to retrain on.")
            return

        print(f"Retraining on {len(X_feed)} accumulated feedback samples.")
        # Re-init model to fresh state? Or fine tune?
        # Usually full retrain = fresh state.
        self.model = GatingClassifier() 
        self.model.fit(X_feed, y_feed)
        self.model.save(self.model_path)
        print("Full retraining complete.")

    def _extract_features(self, df):
        """Helper to safely get Log_FSC/SSC"""
        if "Log_FSC" not in df.columns:
            # Try to compute on the fly if raw exists
            if "FSC-A" in df.columns:
                return np.column_stack([
                    np.log10(df["FSC-A"] + 1e-1),
                    np.log10(df["SSC-A"] + 1e-1)
                ])
            else:
                 raise ValueError("Dataframe missing required columns: Log_FSC, Log_SSC or FSC-A, SSC-A")
        
        return df[["Log_FSC", "Log_SSC"]].values
