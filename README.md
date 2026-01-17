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
   *(Or install dependencies listed in `glow/data_loading` and `glow/gating` manually).*

## Usage

### 1. Starting the Application
Run the Streamlit UI:
```bash
streamlit run ui/app.py
```

### 2. Workflow
1.  **File Loading**:
    *   Select "Enter Directory Path" to scan a folder of `.fcs` files.
    *   **Or select "Upload Files / Drag & Drop Folder"** to upload from your local machine.
    *   Click "Generate Preview" to edit metadata (Sample Name, Replicate, **Cell Line**, etc.).
    *   Confirm to load files.
    *   The loader can also automatically detect cell line info from file names.

2.  **Live/Dead Gating**:
    *   The app attempts to Auto-Gate using one of the options: **a pre-trained MLP model**, Density-based selection, Gaussian Mixture Model (GMM), or DBSCAN-based clustering.
    *   **Accept Auto**: If the suggestion is good, click "✅ Accept Gate". This logs the live and dead cells.
    *   **Manual Gate**: Click "❌ Reject / Manual Gate".
        *   Click roughly **5 points** on the plot around your population.
        *   Confirm the selection. This logs the custom polygon for selection.

3.  **Singlet Gating**:
    *   After Live gating, the app discriminates Singlets vs Doublets.
    *   **Default Strategy**: Uses either **a pre-trained MLP model** or Gaussian Mixture Model (GMM) on user-selected channels (e.g., FSC-A vs FSC-H).
    *   **Accept Auto**: If the suggestion is good, click "✅ Accept Singlets". This logs the singlet and doublet cells.
    *   **Manual Detection**: Revert to manual polygon selection if the auto detection fails.

4.  **Final Data Settings (Export)**:
    *   Once Singlets are accepted, the export options appear.
    *   **Channel Selection**: You can modify this to include/exclude specific channels.
    *   **Export**: Save the "Gated Population" (Live + Singlets) or generate a full annotated dataset.

### 3. Active Learning
The system uses a hybrid logging and interactive learning approach.

*   **Model Architecture**: A **Multi-Layer Perceptron (MLP)** classifier.
    *   **Features**: Log-Scale Scatter Channels + One-Hot Encoded **Cell Line**.
    *   **Preprocessing**: Density Centering (mode alignment) + Standard Scaling.

*   **Feedback Mechanisms**:
    1.  **Gate Acceptance (Passive)**:
        *   When you click "✅ Accept Auto/Manual Gate", the system **logs** the parameters to disk.
        *   *Note: This does NOT update the running model immediately. These logs are used for offline retraining / batch updates.*
    2.  **Point Labeling (Active/Immediate)**:
        *   **Find Confusion**: Click "🔍 Find Confusion" to see points where the model is uncertain.
        *   **Interactive Training**: Select points on the plot (via Box Select or Confusion Highlight) and click **"✅ In Gate"** or **"🚫 Not In Gate"**.
        *   **Effect**: This triggers an immediate **`partial_fit`** on the neural network, updating the decision boundary instantly for the current session.

## Known Issues / Limitations

*   **Concurrent Learning**: If multiple users or sessions are providing feedback simultaneously, the "latest" feedback might overwrite previous weights for a specific cell line during the training step.

## Repository Architecture

The project is organized into modular components to separate core logic, user interface, and data management.

```text
Glow/
├── glow/                       # Core Python Package
│   ├── active_learning/        # Learning System (Feedback logging, Model Weights)
│   ├── analysis/               # Statistical Analysis Logic
│   ├── canvas/                 # Canvas Plotting Utilities
│   ├── data_loading/           # File Loading & Metadata Parsing
│   ├── gating/                 # Gating Strategies & Boundary Computation
│   ├── pipeline/               # Pipeline Runner
│   └── plotting/               # Plotting Logic (Matplotlib/Plotly)
├── ui/                         # Streamlit Frontend
│   ├── app.py                  # Main Application Entry Point
│   ├── analysis/               # Analysis Tab Views
│   ├── canvas/                 # Canvas UI Components
│   ├── gating/                 # Gating UI Components
│   └── plot/                   # Plotting UI Components
├── tests/                      # Unit and Integration Tests
├── data/                       # Data Storage
└── debug/                      # Prototyping Notebooks & Debugging Scripts
```
*   **Cell Line Naming**: Ensure consistent naming of Cell Lines (case-sensitive handling is normalized, but typos will create new entries).