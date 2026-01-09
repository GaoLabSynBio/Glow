# Glow
A user-friendly flow analysis pipeline with pretty UI and Active Learning capabilities.

## Overview
Glow is designed to streamline flow cytometry data analysis. It provides an interactive interface for loading FCS files, managing metadata, and performing gating (Live/Dead and Singlets). It features an **Active Learning** system that learns from user feedback to improve automated gating suggestions over time based on Cell Line characteristics.

## Installation

1. Clone the repository.
2. Create the environment:
   ```bash
   conda env create -f environment.yml
   conda activate glow_env
   ```
   *(Or install dependencies listed in `active_learning` and `base_pipeline` manually).*

## Usage

### 1. Starting the Application
Run the Streamlit UI:
```bash
streamlit run ui/app.py
```

### 2. Workflow
1.  **File Loading**:
    *   Select "Enter Directory Path" to scan a folder of `.fcs` files.
    *   Click "Generate Preview" to edit metadata (Sample Name, Replicate, **Cell Line**, etc.).
    *   Confirm to load files.

2.  **Live/Dead Gating**:
    *   The app attempts to Auto-Gate using a Gaussian Mixture Model (GMM).
    *   **Accept Auto**: If the suggestion is good, click "✅ Accept Auto-Gate". This logs the "Auto" parameters as correct for this Cell Line.
    *   **Manual Gate**: Click "❌ Reject / Manual Gate".
        *   Click roughly **5 points** on the plot around your population.
        *   Confirm the selection. This logs the custom polygon for training.

3.  **Singlet Gating**:
    *   After Live gating, the app shows the Singlet discrimination step.
    *   Similar to above, accept the Auto-Gate or define a manual **4-point** polygon.

### 3. Active Learning
The system records your "Accept" or "Manual Override" decisions. 
*   **Live Cells**: Learned as a 5-point polygon.
*   **Singlets**: Learned as a 4-point polygon.
*   Future analysis of the same **Cell Line** will prioritize these learned gates over generic auto-gating.

## Known Issues / Limitations
*   **Real-time Learning Update**: The Active Learning model currently updates its weights based on logs but may not immediately apply learned gates to the *current* running session if the metadata was just entered. You may need to restart the pipeline or ensure the model training trigger runs to see the effect on new files.
*   **Concurrent Learning**: If multiple users or sessions are providing feedback simultaneously, the "latest" feedback might overwrite previous weights for a specific cell line during the training step.

## Repository Architecture

The project is organized into modular components to separate core logic, user interface, and data management.

```text
Glow/
├── glow/                       # Core Python Package
│   ├── base_pipeline/          # Backend logic (Cytoflow wrappers, Gating algos)
│   └── active_learning/        # Learning system (Feedback logging, Model weights)
├── ui/                         # Streamlit Frontend
│   ├── app.py                  # Main Application Entry Point
│   └── modules/                # Modular UI Components
│       ├── ui_file_loader.py   # File scanning & loading widgets
│       ├── ui_metadata.py      # Metadata parsing & merging widgets
│       └── ui_gating.py        # Gating visuals & interaction logic
├── tests/                      # Unit and integration tests
├── data/                       # Data storage (Active Learning logs, processed files)
└── debug/                      # Prototyping notebooks
```
*   **Cell Line Naming**: Ensure consistent naming of Cell Lines (case-sensitive handling is normalized, but typos will create new entries).