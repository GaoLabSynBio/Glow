import numpy as np
import joblib
import os
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

class DensityCenteringTransformer(BaseEstimator, TransformerMixin):
    """
    Centers data based on the highest density region (mode) of the distribution.
    This allows the classifier to be invariant to location shifts of the main population.
    """
    def __init__(self, bin_size=50):
        self.bin_size = bin_size
        self.reference_mode = None
        self.dataset_modes = {}

    def set_reference_mode(self, mode):
        """Sets a fixed mode to use for transformation (e.g. from the full dataset during feedback loop)."""
        self.reference_mode = np.array(mode)

    def _find_mode(self, X):
        if X.shape[0] < 5:
            if self.reference_mode is not None:
                return self.reference_mode
            return np.mean(X, axis=0)

        H, xedges, yedges = np.histogram2d(X[:, 0], X[:, 1], bins=self.bin_size)
        idx = np.unravel_index(np.argmax(H), H.shape)
        x_c = (xedges[idx[0]] + xedges[idx[0]+1]) / 2.0
        y_c = (yedges[idx[1]] + yedges[idx[1]+1]) / 2.0
        # Only return the center of the mode
        return np.array([x_c, y_c])

    def fit(self, X, y=None):
        return self

    def transform(self, X, dataset_ids=None):
        X = np.asarray(X)
        if dataset_ids is None:
            if self.reference_mode is not None:
                return X - self.reference_mode
            mode = self._find_mode(X)
            return X - mode

        X_out = np.zeros_like(X)
        dataset_ids = np.asarray(dataset_ids)
        unique_ids = np.unique(dataset_ids)
        for uid in unique_ids:
            mask = (dataset_ids == uid)
            X_sub = X[mask]
            
            key = str(uid)
            
            if X_sub.shape[0] >= 50:
                mode = self._find_mode(X_sub)
                self.dataset_modes[key] = mode
            elif key in self.dataset_modes:
                mode = self.dataset_modes[key]
            else:
                if self.reference_mode is not None:
                    mode = self.reference_mode
                else:
                    mode = self._find_mode(X_sub)
            
            X_out[mask] = X_sub - mode

        return X_out

class GatingClassifier:
    """
    A neural network classifier for distinguishing In-Gate (Live) vs Out-Gate (Debris/Outlier) events.
    Architecture:
    - Input: 2 Features (Log_FSC, Log_SSC)
    - Preprocessing: DensityCentering -> StandardScaler
    - Network: MLP with 3 hidden layers (128, 64, 32 neurons)
    - Activation: ReLU
    - Output: Binary classification probability
    """

    def __init__(self, hidden_layer_sizes=(128, 64, 32), random_state=42):
        self.centerer = DensityCenteringTransformer()
        self.scaler = StandardScaler()
        self.mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            alpha=0.0001,      # L2 regularization
            batch_size='auto', # Minibatch optimization
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True, # Important for active learning to prevent overfitting
            validation_fraction=0.1,
            random_state=random_state,
            verbose=False
        )
        self.is_fitted = False

    def set_context_mode(self, mode):
        """Pass context (dataset mode) to the centerer."""
        self.centerer.set_reference_mode(mode)

    def fit(self, X, y, dataset_ids=None):
        """
        Train the model.
        Args:
            X: Features (Log_FSC, Log_SSC)
            y: Labels
            dataset_ids: Optional, array of dataset identifiers for correct per-dataset centering.
        """
        # 1. Center
        X_centered = self.centerer.transform(X, dataset_ids=dataset_ids)

        # 2. Scale
        X_scaled = self.scaler.fit_transform(X_centered)

        # 3. Train
        self.mlp.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def partial_fit(self, X, y, classes=None, dataset_ids=None):
        """
        Incremental update of the model.
        """
        if classes is None:
            classes = np.array([0, 1])

        # 1. Center
        X_centered = self.centerer.transform(X, dataset_ids=dataset_ids)

        # 2. Scale (Partial fit scaler)
        self.scaler.partial_fit(X_centered)
        X_scaled = self.scaler.transform(X_centered)

        # 3. Update MLP
        self.mlp.partial_fit(X_scaled, y, classes=classes)
        self.is_fitted = True
        return self

    def predict(self, X, dataset_ids=None):
        """Returns class predictions (0 or 1)."""
        if not self.is_fitted:
            # Fallback for unfitted predictions (e.g. init)
            pass

        X_centered = self.centerer.transform(X, dataset_ids=dataset_ids)
        X_scaled = self.scaler.transform(X_centered)
        return self.mlp.predict(X_scaled)

    def predict_proba(self, X, dataset_ids=None):
        """
        Returns probability of class 1 (Live).
        """
        X_centered = self.centerer.transform(X, dataset_ids=dataset_ids)
        X_scaled = self.scaler.transform(X_centered)
        probs = self.mlp.predict_proba(X_scaled)
        return probs[:, 1]

    def score(self, X, y, dataset_ids=None):
        """Returns accuracy."""
        X_centered = self.centerer.transform(X, dataset_ids=dataset_ids)
        X_scaled = self.scaler.transform(X_centered)
        return self.mlp.score(X_scaled, y)

    def uncertainty(self, X, dataset_ids=None):
        """
        Calculates uncertainty metric.
        """
        probs = self.predict_proba(X, dataset_ids=dataset_ids) # P(Live)
        uncertainty = 1.0 - np.abs(probs - 0.5) * 2.0
        return uncertainty

    def save(self, filepath):
        """Save model to disk."""
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # Save the whole object to preserve scaler state and configuration
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath):
        """Load model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        loaded = joblib.load(filepath)

        if isinstance(loaded, cls):
            return loaded
        elif hasattr(loaded, 'predict'):
            # Legacy Pipeline support
            instance = cls()
            instance.mlp = loaded # Assume loaded is the pipeline (or wrap it)
            # Disable our manual scaling/centering
            instance.centerer.transform = lambda x, **k: x
            instance.scaler.transform = lambda x: x
            instance.scaler.fit = lambda x: x
            # Mark as fitted
            instance.is_fitted = True
            return instance

        return loaded
