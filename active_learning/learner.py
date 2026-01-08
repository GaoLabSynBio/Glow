import json
import os
import numpy as np
from .storage import ActiveLearningLogger

class ActiveLearner:
    def __init__(self, storage_dir="active_learning_data", model_filename="model_weights.json"):
        self.storage_dir = storage_dir
        self.model_path = os.path.join(storage_dir, model_filename)
        self.logger = ActiveLearningLogger(storage_dir)
        # Knowledge base structure: { cell_line: { 'Live': vertices, 'Singlet': vertices } }
        self.knowledge_base = {} 
        self.load_weights()

    def train(self):
        """
        Rebuilds knowledge base from stored feedback logs.
        Strategy: Use the LATEST accepted or manual_override gate for each cell_line.
        This updates the in-memory knowledge base.
        """
        feedback_history = self.logger.load_feedback()
        
        # Sort by timestamp to ensure we process chronologically
        feedback_history.sort(key=lambda x: x.get('timestamp', ''))

        for entry in feedback_history:
            cell_line = entry.get('cell_line')
            if not cell_line:
                continue
            
            # Normalize cell line name (trim whitespace)
            cell_line = cell_line.strip()
            
            gate_params = entry.get('gate_params')
            vertices = gate_params
            # Handle potential nested dict structure from GMM or others if we standardize on Polygon here for learning
            if isinstance(gate_params, dict) and 'vertices' in gate_params:
                 vertices = gate_params['vertices']
            
            if not isinstance(vertices, list):
                continue

            num_vertices = len(vertices)
            
            inferred_type = None
            # Live cells = 5 points, Singlets = 4 points.
            if num_vertices == 5:
                inferred_type = 'Live'
            elif num_vertices == 4:
                inferred_type = 'Singlet'
            
            if inferred_type:
                if cell_line not in self.knowledge_base:
                    self.knowledge_base[cell_line] = {}
                self.knowledge_base[cell_line][inferred_type] = vertices
        
        print(f"Training complete. Knowledge base updated for {len(self.knowledge_base)} cell lines.")
        self.save_weights()

    def get_known_cell_lines(self):
        """
        Returns a sorted list of unique cell lines currently in the knowledge base.
        Useful for UI dropdowns.
        """
        return sorted(list(self.knowledge_base.keys()))

    def save_weights(self):
        """
        Saves the current knowledge base (weights) to a JSON file.
        """
        try:
            with open(self.model_path, 'w') as f:
                json.dump(self.knowledge_base, f, indent=2)
            print(f"Model weights saved to {self.model_path}")
        except Exception as e:
            print(f"Error saving weights: {e}")

    def load_weights(self):
        """
        Loads the knowledge base (weights) from a JSON file if it exists.
        """
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'r') as f:
                    self.knowledge_base = json.load(f)
                print(f"Model weights loaded from {self.model_path}")
            except Exception as e:
                print(f"Error loading weights: {e}")
                self.knowledge_base = {}
        else:
            print("No existing model weights found. Starting fresh.")
            self.knowledge_base = {}

    def predict(self, cell_line, gate_category):
        """
        Predicts vertices for a given cell_line and gate category ('Live' or 'Singlet').
        Returns list of (x,y) tuples or None if unknown.
        
        Args:
            cell_line (str): The cell line identifier.
            gate_category (str): 'Live' or 'Singlet'.
        """
        if cell_line in self.knowledge_base:
            val = self.knowledge_base[cell_line].get(gate_category)
            if val:
                return val
        return None
