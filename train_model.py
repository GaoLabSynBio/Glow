import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import shutil

# Ensure glow is in path
sys.path.append(os.getcwd())

from glow.active_learning.models import GatingClassifier
from glow.data_loading.loader import ParquetBatchLoader

def train_and_plot():
    # Load data
    data_path = "data/synthetic_training_data.parquet"
    print(f"Loading data from {data_path}...")

    if not os.path.exists(data_path):
        print("Error: Data file not found.")
        return

    # Use Lazy Loader to get Row Groups
    temp_loader = ParquetBatchLoader(data_path)
    total_groups = temp_loader.num_row_groups
    indices = list(range(total_groups))
    
    # Shuffle and Split 80/20
    random.seed(42)  # For reproducibility
    random.shuffle(indices)
    split_idx = int(0.8 * total_groups)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    print(f"Total Row Groups: {total_groups}")
    print(f"Training on {len(train_indices)} groups, Validating on {len(val_indices)} groups")

    # ---------------------------------------------------------
    # Model Training
    # ---------------------------------------------------------
    print("Initializing model with new architecture...")
    model = GatingClassifier()
    model.mlp.verbose = False 
    model.mlp.early_stopping = False # We handle it manually

    print("Starting training (Incremental Learning using partial_fit)...")

    train_loader = ParquetBatchLoader(
        data_path,
        columns=['FSC-A', 'SSC-A', 'TrueLabel', 'DatasetID'],
        row_group_indices=train_indices
    )
    
    val_loader = ParquetBatchLoader(
        data_path,
        columns=['FSC-A', 'SSC-A', 'TrueLabel', 'DatasetID'],
        row_group_indices=val_indices
    )

    # Hyperparameters
    n_epochs = 20
    patience = 3
    classes = np.array([0, 1])
    
    best_val_acc = 0.0
    patience_counter = 0
    best_model_path = "data/gating_model_best.pkl"
    
    train_losses = []
    val_accuracies = []

    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch+1}/{n_epochs}")
        
        # Training Loop
        batch_loss_sum = 0
        batch_count = 0
        
        for table in train_loader:
            batch_count += 1
            # Simple progress 
            if batch_count % 10 == 0:
                print(f"  [Train] Processing batch {batch_count}/{len(train_indices)}...", end='\r')

            fsc = table['FSC-A'].to_numpy()
            ssc = table['SSC-A'].to_numpy()
            y_raw = table['TrueLabel'].to_numpy().astype(str)
            dataset_ids = table['DatasetID'].to_numpy()

            y = np.zeros(len(y_raw), dtype=int)
            y[y_raw == 'Live'] = 1

            X = np.log1p(np.column_stack((fsc, ssc)))

            model.partial_fit(X, y, classes=classes, dataset_ids=dataset_ids)
            
            if hasattr(model.mlp, 'loss_'):
                batch_loss_sum += model.mlp.loss_

        # Just to clear the carriage return line
        print(f"  [Train] Processing batch {batch_count}/{len(train_indices)} - Done.    ")

        avg_train_loss = batch_loss_sum / batch_count if batch_count > 0 else 0
        train_losses.append(avg_train_loss)
        print(f"  [Train] Avg Loss: {avg_train_loss:.4f}")
        
        # Validation Loop
        val_acc_sum = 0
        val_total_samples = 0
        
        # We can re-use val_loader because __iter__ resets iterator
        v_batch_count = 0
        for table in val_loader:
            v_batch_count += 1
            
            fsc = table['FSC-A'].to_numpy()
            ssc = table['SSC-A'].to_numpy()
            y_raw = table['TrueLabel'].to_numpy().astype(str)
            dataset_ids = table['DatasetID'].to_numpy()

            y = np.zeros(len(y_raw), dtype=int)
            y[y_raw == 'Live'] = 1
            
            X = np.log1p(np.column_stack((fsc, ssc)))
            
            # Score returns mean accuracy
            # To get global accuracy, weigh by sample count
            acc = model.score(X, y, dataset_ids=dataset_ids)
            samples = len(y)
            val_acc_sum += acc * samples
            val_total_samples += samples
            
        val_acc = val_acc_sum / val_total_samples if val_total_samples > 0 else 0
        val_accuracies.append(val_acc)
        print(f"  [Val]   Accuracy: {val_acc:.4%}")
        
        # Early Stopping Check
        if val_acc > best_val_acc:
            print(f"  > Improvement found! ({best_val_acc:.4%} -> {val_acc:.4%}) Saving model...")
            best_val_acc = val_acc
            patience_counter = 0
            model.save(best_model_path)
            # Make a copy as the active model
            shutil.copy2(best_model_path, "data/gating_model_full.pkl")
        else:
            patience_counter += 1
            print(f"  > No improvement. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # ---------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-o', label='Train Loss')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.grid(True)
    plt.legend()
    
    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, 'g-o', label='Val Accuracy')
    plt.title('Validation Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_loss_curve.png')
    print("Saved training plots to training_loss_curve.png")

if __name__ == "__main__":
    train_and_plot()
