import numpy as np
import joblib
import os
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class GatingClassifier:
    """
    A neural network classifier for distinguishing In-Gate (Live) vs Out-Gate (Debris/Outlier) events.
    
    Architecture:
    - Input: 2 Features (Log_FSC, Log_SSC)
    - Preprocessing: StandardScaler (Zero mean, Unit variance)
    - Network: MLP with 2 hidden layers (64, 32 neurons)
    - Activation: ReLU
    - Output: Binary classification probability
    """
    
    def __init__(self, hidden_layer_sizes=(64, 32), random_state=42):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(
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
            ))
        ])
        
    def fit(self, X, y):
        """
        Train the model.
        
        Args:
            X (np.ndarray or pd.DataFrame): Feature matrix (N x 2). 
                                            Expected to be [Log_FSC, Log_SSC].
            y (np.ndarray or pd.Series): Labels (0, 1).
        """
        self.pipeline.fit(X, y)
        return self

    def partial_fit(self, X, y, classes=None):
        """
        Incremental update of the model. 
        Note: StandardScaler partial_fit needs to be handled carefully if data distribution shifts wildly, 
        but here we assume it's relatively stable log-training data.
        """
        if classes is None:
            classes = np.array([0, 1])

        # 1. Update Scaler
        scaler = self.pipeline.named_steps['scaler']
        scaler.partial_fit(X)
        
        # 2. Transform Data
        X_scaled = scaler.transform(X)
        
        # 3. Update MLP
        mlp = self.pipeline.named_steps['mlp']
        mlp.partial_fit(X_scaled, y, classes=classes)
        
        return self
        
    def predict(self, X):
        """Returns class predictions (0 or 1)."""
        return self.pipeline.predict(X)
        
    def predict_proba(self, X):
        """
        Returns probability of class 1 (Live).
        Useful for uncertainty sampling in Active Learning.
        """
        # predict_proba returns [prob_0, prob_1]
        probs = self.pipeline.predict_proba(X)
        return probs[:, 1]
    
    def score(self, X, y):
        """Returns accuracy."""
        return self.pipeline.score(X, y)

    def uncertainty(self, X):
        """
        Calculates uncertainty metric (Entropy or Least Confidence) based on prediction probability.
        Here we use simple margin uncertainty: 1 - |prob - 0.5|*2
        Range: 0 (Confidence) to 1 (Uncertain).
        """
        probs = self.predict_proba(X) # P(Live)
        # Map 0.5 -> 1.0, 0.0/1.0 -> 0.0
        uncertainty = 1.0 - np.abs(probs - 0.5) * 2.0
        return uncertainty

    def save(self, filepath):
        """Save model to disk."""
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        joblib.dump(self.pipeline, filepath)
        
    @classmethod
    def load(cls, filepath):
        """Load model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        instance = cls()
        instance.pipeline = joblib.load(filepath)
        return instance
