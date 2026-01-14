import numpy as np
import pandas as pd
import cytoflow
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# -----------------------------------------------------------------------------
# 1. Compensation Management
# -----------------------------------------------------------------------------

class CompensationManager:
    """
    Manages compensation operations, applying spillover matrices or other
    adjustments to the flow cytometry data.
    """
    def apply_compensation(self, experiment, spillover_matrix_file=None):
        """
        Applies compensation.
        """
        if spillover_matrix_file:
            print(f"Applying compensation matrix from {spillover_matrix_file}...")
            # Note: Cytoflow favors control-based compensation.
            # This is a placeholder for where manual matrix application would happen.
            pass
        return experiment

# -----------------------------------------------------------------------------
# 2. Gating Strategies Hierarchy
# -----------------------------------------------------------------------------

class GatingStrategy(ABC):
    """
    Abstract base class for all gating strategies.
    Defines the contract for applying a gate to an experiment.
    """
    def __init__(self, name: str, channels: List[str]):
        """
        Args:
            name: The name of the gate/condition to be created.
            channels: The list of channels involved in the gating.
        """
        self.name = name
        self.channels = channels

    def validate_dimensions(self, required_dim: int):
        """Helper to validate channel count."""
        if len(self.channels) != required_dim:
            raise ValueError(f"Strategy {self.__class__.__name__} requires {required_dim} channels, got {len(self.channels)}")

    @abstractmethod
    def apply(self, experiment):
        """
        Applies the gate to the experiment.
        """
        pass

# --- Level 2: Broad Categories ---

class OutlierDetector:
    """
    Handles outlier detection and removal to provide clean data for gating estimation.
    """
    def __init__(self, method: str = 'isolation_forest', outlier_low: float = 0.01, outlier_high: float = 0.99):
        self.method = method
        self.outlier_low = outlier_low
        self.outlier_high = outlier_high
        self.use_isolation_forest = (method == 'isolation_forest') and SKLEARN_AVAILABLE

    def get_mask(self, experiment, channels: List[str]):
        """
        Returns a boolean mask (Series) where True = Keep (Inlier), False = Remove (Outlier).
        """
        data = experiment.data
        x, y = channels[0], channels[1] if len(channels) > 1 else channels[0]

        mask = None
        if self.use_isolation_forest and len(data) > 20:
            # Isolation Forest Strategy
            contam_est = 1.0 - (self.outlier_high - self.outlier_low)
            contam_est = max(0.001, min(0.4, contam_est))

            X_iso = data[[x, y]].values

            if len(X_iso) > 20000:
                idx = np.random.choice(len(X_iso), 20000, replace=False)
                X_train = X_iso[idx]
            else:
                X_train = X_iso

            clf = IsolationForest(contamination=contam_est, n_jobs=-1, random_state=42)
            clf.fit(X_train)

            preds = clf.predict(X_iso)
            mask = (preds == 1) # 1 = inlier
        else:
            # Quantile Box Strategy
            x_low, x_high = data[x].quantile([self.outlier_low, self.outlier_high])
            if len(channels) > 1:
                y_low, y_high = data[y].quantile([self.outlier_low, self.outlier_high])
                mask = (data[x] >= x_low) & (data[x] <= x_high) & \
                       (data[y] >= y_low) & (data[y] <= y_high)
            else:
                mask = (data[x] >= x_low) & (data[x] <= x_high)

        # Ensure Series format
        if not isinstance(mask, (pd.Series, pd.DataFrame)):
            mask = pd.Series(mask, index=data.index)

        return mask

    def get_clean_subset(self, experiment, channels: List[str]):
        data = experiment.data
        mask = self.get_mask(experiment, channels)

        clean_data = data[mask]

        # Check if we removed everything
        if len(clean_data) < 10:
             print("Warning: Outlier removal resulted in too few events. Using full dataset.")
             return experiment

        try:
            # Create subset experiment for estimation
            est_data_reset = clean_data.reset_index(drop=True)
            est_data_reset['Sample'] = 'Subset'
            # Create fresh Experiment
            est_exp = cytoflow.Experiment(data=est_data_reset, metadata={'Sample': 'category'})
            return est_exp
        except Exception as e:
            print(f"OutlierDetector Warning: {e}. Falling back to full experiment.")
            return experiment

class AutomaticGating(GatingStrategy):
    """
    Base class for automated, data-driven gating strategies.
    Examples: GMM, K-Means, Density-based.
    Supports optional OutlierDetector to clean data before estimation.
    """
    def __init__(self, name: str, channels: List[str], outlier_detector: Optional[OutlierDetector] = None):
        super().__init__(name, channels)
        self.outlier_detector = outlier_detector

    def get_estimation_data(self, experiment):
        """
        Returns the dataset to be used for estimating the gate parameters.
        If an outlier detector is configured, it returns the cleaned subset.
        """
        if self.outlier_detector:
            return self.outlier_detector.get_clean_subset(experiment, self.channels)
        return experiment

class ManualGating(GatingStrategy):
    """
    Base class for manual, user-defined gating strategies.
    Examples: Polygon, Rectangle, Range/Threshold.
    """
    pass

# --- Level 3: Specific Implementations of Automatic Gating ---

class GMMGating(AutomaticGating):
    """
    Gaussian Mixture Model gating.
    Can be used for clustering, live/dead discrimination, or singlet finding.
    Dimension: N (usually 1 or 2).
    """
    def __init__(self, name: str, channels: List[str], num_components=2, return_model=False, outlier_detector: Optional[OutlierDetector] = None):
        super().__init__(name, channels, outlier_detector)
        self.num_components = num_components
        self.return_model = return_model

    def apply(self, experiment):
        gmm = cytoflow.GaussianMixtureOp(name=self.name,
                                         channels=self.channels,
                                         num_components=self.num_components)

        # Estimate on (potentially cleaned) data
        estimation_data = self.get_estimation_data(experiment)
        gmm.estimate(estimation_data)

        res = gmm.apply(experiment)

        if self.return_model:
            return res, gmm
        return res

class DensityGating(AutomaticGating):
    """
    Density-based gating (kernel density estimation).
    Selects the high-density region of the plot.
    Dimension: 2.
    """
    def __init__(self, name: str, channels: List[str], keep: float = 0.9, sigma: float = 1.0, outlier_detector: Optional[OutlierDetector] = None):
        super().__init__(name, channels, outlier_detector)
        self.validate_dimensions(2)
        self.keep = keep
        self.sigma = sigma

    def apply(self, experiment):
        density_op = cytoflow.DensityGateOp(name=self.name,
                                            xchannel=self.channels[0],
                                            ychannel=self.channels[1],
                                            keep=self.keep,
                                            sigma=self.sigma)

        estimation_data = self.get_estimation_data(experiment)
        density_op.estimate(estimation_data)

        return density_op.apply(experiment)


class DBSCANGating(AutomaticGating):
    """
    DBSCAN-based gating.
    Identifies clusters based on density and proximity.
    Dimension: N (usually 2).
    """
    def __init__(self, name: str, channels: List[str], eps: float = 0.05, min_samples: int = 50, outlier_detector: Optional[OutlierDetector] = None):
        super().__init__(name, channels, outlier_detector)
        self.eps = eps
        self.min_samples = min_samples

    def apply(self, experiment):
        from sklearn.cluster import DBSCAN
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.preprocessing import MinMaxScaler

        MAX_SAMPLES = 20000

        # Get estimation data (potentially cleaned)
        estimation_data = self.get_estimation_data(experiment)

        # Get full data for prediction later
        full_data = experiment.data[self.channels].values

        # Determine if we need to downsample for the clustering step
        X = estimation_data.data[self.channels].values
        n_samples = len(X)

        scaler = MinMaxScaler()

        if n_samples > MAX_SAMPLES:
            # Downsample for DBSCAN
            indices = np.random.choice(n_samples, MAX_SAMPLES, replace=False)
            X_subset = X[indices]
            X_scaled_subset = scaler.fit_transform(X_subset) # Fit scaler on subset
            
            # Scale min_samples relative to downsampling
            ratio = MAX_SAMPLES / n_samples
            adjusted_min_samples = max(3, int(self.min_samples * ratio))

            # Run DBSCAN on subset
            db = DBSCAN(eps=self.eps, min_samples=adjusted_min_samples).fit(X_scaled_subset)
            labels_subset = db.labels_

            # Train KNN classifier to propagate labels
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_scaled_subset, labels_subset)

            # Predict for ALL data (scaled with the same scaler)
            # Use chunks to avoid memory spike during prediction if massive
            X_full_scaled = scaler.transform(full_data)
            labels_full = knn.predict(X_full_scaled)

            # Note: KNN will predict a label for everything, even outliers from DBSCAN (-1).
            # This is generally acceptable for "gating" as we want a boundary,
            # and -1 points (outliers) will form a "noise" class region.

        else:
            # Run on full estimation data
            X_scaled = scaler.fit_transform(X)
            db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(X_scaled)
            labels_est = db.labels_

            # If estimation_data IS experiment data, we are done
            # But if estimation_data is a subset (outlier removal), we need to propagate or map back
            # For simplicity, if we have a mismatch, we use KNN as well

            if len(X) != len(full_data):
                knn = KNeighborsClassifier(n_neighbors=1)
                knn.fit(X_scaled, labels_est)
                X_full_scaled = scaler.transform(full_data)
                labels_full = knn.predict(X_full_scaled)
            else:
                labels_full = labels_est

        # Clone experiment
        res = experiment.clone()
        res.data[self.name] = labels_full

        return res

# --- Level 3: Specific Implementations of Manual Gating ---

class PolygonGating(ManualGating):
    """
    Manual Polygon Gate.
    Dimension: 2.
    """
    def __init__(self, name: str, channels: List[str], vertices: List[Tuple[float, float]]):
        super().__init__(name, channels)
        self.validate_dimensions(2)
        self.vertices = vertices

    def apply(self, experiment):
        poly_gate = cytoflow.PolygonOp(name=self.name,
                                       xchannel=self.channels[0],
                                       ychannel=self.channels[1],
                                       vertices=self.vertices)
        return poly_gate.apply(experiment)

class RangeGating(ManualGating):
    """
    Manual Range/Threshold Gate (1D).
    Dimension: 1.
    """
    def __init__(self, name: str, channel: str, low: Optional[float] = None, high: Optional[float] = None):
        super().__init__(name, [channel])
        self.low = low
        self.high = high

    def apply(self, experiment):
        range_op = cytoflow.RangeOp(name=self.name,
                                    channel=self.channels[0],
                                    low=self.low,
                                    high=self.high)
        return range_op.apply(experiment)

class MLPGating(AutomaticGating):
    """
    MLP-based gating using a pre-trained neural network.
    """
    def __init__(self, name: str, channels: List[str], model_path: str = "data/gating_model_full.pkl", outlier_detector: Optional[OutlierDetector] = None, cell_type: str = "colo"):
        super().__init__(name, channels, outlier_detector)
        self.model_path = model_path
        self.cell_type = cell_type

    def apply(self, experiment):
        from glow.active_learning.models import GatingClassifier
        import os
        
        if not os.path.exists(self.model_path):
             # Fallback or Error? 
             # Check if we can find it in the generic path
             self.model_path = "data/gating_model_full.pkl"
             if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"MLP Model not found at {self.model_path}")

        clf = GatingClassifier.load(self.model_path)
        
        data = experiment.data
        
        # Prepare Features: Model expects 2 columns, usually Log transformed.
        xs = data[self.channels[0]].values
        ys = data[self.channels[1]].values
        
        # Heuristic Log Transform if data seems linear (max > 10)
        # The model 'gating_model_full.pkl' was trained on Log data.
        if np.max(xs) > 10:
            xs = np.log10(xs)
        if np.max(ys) > 10:
            ys = np.log10(ys)
            
        X = np.column_stack([xs, ys])
        
        # Prepare Metadata for Cell-Type Aware Inference
        # 1. Dataset ID: We treat the current experiment as a single batch (ID=0)
        #    so the DensityCenteringTransformer calculates the mode of THIS specific data.
        dataset_ids = np.zeros(len(X), dtype=int)
        
        # 2. Cell Type: Broadcast user selection
        cell_types = np.array([self.cell_type] * len(X))
        
        # Predict
        preds = clf.predict(X, dataset_ids=dataset_ids, cell_types=cell_types)
        
        # Result
        res = experiment.clone()
        
        # Map 0/1 to meaningful labels if desired, or keep as is.
        # User UI expects 'AutoLiveGate' to split population.
        # 1 = Live, 0 = Debris usually.
        # We'll cast to string or int to match expectations of categorical gate
        res.data[self.name] = preds 
        
        return res
