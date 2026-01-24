<img src="assets/glow_icon_breath.gif" width="300" alt="Glow Icon" />

A user-friendly flow analysis pipeline with pretty UI and Active Learning capabilities.

## Overview

Glow is designed to streamline flow cytometry data analysis. It provides an interactive interface for loading FCS files, managing metadata, and performing gating (Live/Dead and Singlets). It features an **Active Learning** system that learns from user feedback to improve automated gating suggestions over time based on Cell Line characteristics.

### Installation

### Linux

1. Clone the repository.
2. Create the environment:
   ```bash
   conda env create -f environment.yml
   conda activate glow_env
   ```

   *(Or install dependencies listed in `glow/data_loading` and `glow/gating` manually).*

### Windows & Mac

For Windows and Mac users, a standalone app will be provided—no manual installation required. Please check the releases or contact the maintainers for the latest installer.

## Usage

### 1. Starting the Application

#### Linux

Run the Streamlit UI:

```bash
streamlit run ui/app.py
```

#### Windows

Run the provided `.exe` application (no command line required).

#### Mac

Run the provided app bundle (no command line required).

### 2. Workflow

1. **File Loading & Metadata**:

   * Select "Enter Directory Path" to scan a folder of `.fcs` files.
   * **Or select "Upload Files / Drag & Drop Folder"** to upload from your local machine.
   * For metadata, you have two options:
     - **Upload your metadata spreadsheet** (e.g., CSV or Excel) and the app will read it.
     - **Manually input all metadata** in the UI.
   * When entering metadata, make sure to provide **Cell Line** and **Experiment Name** (these are important for distinguishing conditions; the same condition for different samples should have consistent names).
   * Click "Generate Preview" to review and edit metadata (Sample Name, Replicate, **Cell Line**, etc.).
   * Confirm to load files.
   * The loader can also automatically detect cell line info from file names.
2. **Live/Dead Gating**:

   * The app attempts to Auto-Gate using one of the options: **a pre-trained MLP model**, Density-based selection, Gaussian Mixture Model (GMM), or DBSCAN-based clustering on the left side.
   * **Accept Auto**: If the suggestion is good, click "✅ Accept Gate". This logs the live and dead cells.
   * **Manual Gate**: Click "❌ Reject / Manual Gate".
     * Click roughly **5 points** on the plot around your population.
     * Confirm the selection. This logs the custom polygon for selection. There could be a lag in manual selection, be patient.
3. **Singlet Gating**:

   * After Live gating, the app discriminates Singlets vs Doublets.
   * **Default Strategy**: Uses either **a pre-trained MLP model** or Gaussian Mixture Model (GMM) on user-selected channels (e.g., FSC-A vs FSC-H) on the left side.
   * **Accept Auto**: If the suggestion is good, click "✅ Accept Singlets". This logs the singlet and doublet cells.
   * **Manual Detection**: Revert to manual polygon selection if the auto detection fails.
4. **Final Data Settings (Export)**:

   * Once Singlets are accepted, the export options appear.
   * **Channel Selection**: You can modify this to include/exclude specific channels.
   * **Export**: Save the "Gated Population" (Live + Singlets) or generate a full annotated dataset. You can also change the names of file to download here.

 5. **Analysis**:

   * After exporting or selecting your data, you can perform further analysis within the app:
   
   * **Transfection Efficiency Gate**: Specialized gating for transfection efficiency assessment. You can do gating for binary or just binary.
   * **2D Quadratic Gating**: Use quadratic boundaries for more complex population separation in 2D plots.
   * Additional analysis tools and visualizations are available to support your experimental needs.

6. **Plotting**:

   * Glow supports a variety of plotting options to help visualize your data:
   
   * **1D Stack Plot**: Visualize distributions across samples or conditions in a stacked format.
   * **2D Running Means**: Display trends and relationships between two variables using running means in 2D plots.
   * **Bar Plots**: Summarize and compare data across groups or conditions with customizable bar plots.
   * Additional plotting features are available for in-depth data exploration.

### 3. Canvas Customization

Glow provides a canvas feature where you can customize the appearance of your plots, all plots are able to added to canvas to tune or directly downloaded as svg to be further modified in illustrator easily:

* Change fonts to match your presentation or publication style.
* Select different color palettes for better visualization and accessibility.
* Adjust other visual elements to tailor the output to your needs.
