import argparse
import os
import sys

# Ensure we can import modules from the current package context if run directly
# Logic: If this script is run, add parent dir to sys.path or use relative imports if run as module
# Ideally correct usage: python -m base_pipeline.pipeline
from .data_loader import DataLoader
from .gating import GatingManager
from active_learning.learner import ActiveLearner

def run_base_pipeline(file_list_path=None, input_dir=None, file_paths=None, rename_strategy='folder_name', output_dir='processed_fcs', cell_line=None):
    """
    Main entry point for the base pipeline logic.
    """
    loader = DataLoader()
    
    # 1. Load Paths
    final_paths = []
    if file_list_path:
        final_paths.extend(loader.parse_file_paths_from_text(file_list_path))
    if input_dir:
        final_paths.extend(loader.scan_directory(input_dir))
    if file_paths:
        final_paths.extend(file_paths)
        
    if not final_paths:
        print("Error: No valid file paths provided.")
        return None

    # 2. Rename & Prepare
    print("Preparing files (renaming and organizing)...")
    # Using 'prepare_fcs_files' as defined in data_loader.py (wrapper)
    metadata = loader.prepare_fcs_files(final_paths, rename_strategy=rename_strategy, output_dir=output_dir)
    
    # 3. Import to Cytoflow
    print("Loading files into Cytoflow Experiment...")
    experiment = loader.load_as_cytoflow_experiment(metadata)
    
    if experiment is None:
        print("Failed to load experiment.")
        return None

    # 4. Gating Strategy
    gater = GatingManager()
    learner = ActiveLearner()

    # Apply Learned Knowledge if available
    if cell_line:
        print(f"Checking for learned gates for cell line: {cell_line}...")
        
        # Singlet Gate (Learned)
        singlet_vertices = learner.predict(cell_line, "Singlet")
        if singlet_vertices:
            print("Applying learned Singlet gate...")
            # Note: channel names here are hardcoded for now, but should ideally come from learned metadata too or user config
            experiment = gater.apply_manual_gate(experiment, "LearnedSinglet", "FSC-A", "FSC-H", singlet_vertices)
        else:
             # Fallback to auto
             # print("Attempting to filter duplicates (Singlet finding)...")
             # experiment = gater.remove_duplicates(experiment)
             pass

        # Live Gate (Learned)
        live_vertices = learner.predict(cell_line, "Live")
        if live_vertices:
            print("Applying learned Live gate...")
            experiment = gater.apply_manual_gate(experiment, "LearnedLive", "FSC-A", "SSC-A", live_vertices)
            print("Learned Live gate applied.")
            # Skip Auto Gating if learned gate is applied? 
            # Or maybe we return both and let user decide in UI.
            # For pipeline automation, we prefer the learned one.
            return experiment

    # B. Auto Live Gating (Fallback)
    print("Applying Automatic Live/Dead Gating (GMM)...")
    # Assuming FSC-A and SSC-A are present. This might fail if channels differ.
    try:
        experiment = gater.auto_gate_live_cells(experiment, x_channel="FSC-A", y_channel="SSC-A")
        print("Auto gating applied. Check 'AutoLiveGate' condition.")
    except Exception as e:
        print(f"Warning: Auto gating failed (check channel names?): {e}")

    print("Pipeline execution finished.")
    return experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Glow Base Pipeline")
    parser.add_argument("--file_list", help="Path to a text file containing list of FCS file paths")
    parser.add_argument("--input_dir", help="Directory to recursively search for FCS files")
    parser.add_argument("--output_dir", default="processed_fcs", help="Directory to save renamed files")
    parser.add_argument("--rename", default="folder_name", choices=["folder_name", "keep"], help="Renaming strategy")
    
    args = parser.parse_args()
    
    if args.file_list or args.input_dir:
        run_base_pipeline(file_list_path=args.file_list, input_dir=args.input_dir, output_dir=args.output_dir, rename_strategy=args.rename)
    else:
        print("Please provide --file_list or --input_dir to run via CLI.")
