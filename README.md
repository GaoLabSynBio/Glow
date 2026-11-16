<img src="assets/glow_icon_breath.svg" width="300" alt="Glow Icon" />

A flow analysis pipeline with a clean user-friendly UI and active learning capabilities.
(We are actively collecting data from various machines and updating this repository for compatibility. If you encounter any data issues, please open an issue on GitHub.)

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

You can check out our demo video through youtube link https://youtu.be/d6pQYoGVG68

### 1. Starting the Application

#### Linux

Run the Streamlit UI:

```bash
streamlit run ui/app.py
```

#### Windows

Run the provided `.exe` application (no command line required).

#### Mac

Run the provided Electron app bundle (no command line required). 

Please ensure the .app file is moved into your Applications folder. 

### 2. Workflow

The sidebar works as a stepper: **Gating → Analysis → Plot → Canvas**. Finished steps get a ✓,
and steps that are not available yet are marked with the reason (for example, "load files first").
"Start new experiment" at the bottom of the sidebar clears everything, after a confirmation.

**Units.** High-resolution data are divided by 10,000 once, at load time. Every tab shows the
same scaled values, and exported CSVs are written back in raw units.

1. **File Loading & Metadata**:

   * Select "Enter Directory Path" and click "Scan Directory" to scan a folder of `.fcs` files,
     **or** select "Upload Files / Drag & Drop Folder" to upload from your machine.
   * Click "Generate Preview" to open the metadata editor. Every field has a default, so you
     can confirm right away:
     - **Experiment Name** defaults to the file name (`D1`, `E2`, …).
     - **Replicate** is inferred: rows that share an Experiment Name are numbered 1, 2, 3, …
       (it is re-inferred when you edit names).
     - **Cell Line** is guessed from the path when a known name appears; otherwise it is left blank,
       which means one population.
     - Auto-filled values are marked, and a "Fill down" helper applies one value to all rows.
   * A metadata spreadsheet (CSV / Excel) can be imported from the collapsed bulk-upload section.
   * Click "Confirm and Load". A progress bar shows the files being prepared and read.
   * Metadata stays editable: an **"Edit metadata"** expander on the Gating, Analysis and Plot
     tabs relabels samples **without re-gating**. Renaming a cell line keeps its gates.
2. **Live/Dead Gating**:

   * The app auto-gates with **a pre-trained MLP model**, density-based selection, a Gaussian
     Mixture Model (GMM) or DBSCAN. Pick the method on the left. The toolbar above the plot
     sets the scale, colormap, interaction mode and representative file.
   * **Accept**: if the suggestion looks right, click "Accept Auto-Gate" (or "Accept Gate for
     <cell line>").
   * **Manual gate**: click "Adjust gate manually", click roughly **5 points** around your
     population, then "Apply Manual Gate".
   * "Redo" buttons let you redo the gate for one cell line or for all of them.
3. **Singlet Gating**:

   * Singlets vs doublets are detected with **a pre-trained MLP model** or a GMM on the chosen
     channels (e.g., FSC-A vs FSC-H). The card layout matches the Live/Dead step.
   * Click "Accept Singlets", or "Adjust gate manually" to draw the gate yourself.
4. **Gating complete**:

   * **Channels to analyze**: a multiselect of the fluorescence channels shown in Analysis and
     Plot. Structural channels are included automatically.
   * **Export** (optional): download the gated population (Live + Singlets) or a full annotated
     dataset as a ZIP, in raw units.
   * Optionally run compensation, then click "Finalize Gating & Proceed to Analysis".
5. **Analysis**:

   * **Summary table**: Experiment Name, Replicate and your condition columns come first, then the
     metrics. The file-level `Sample` column is hidden by default. "Group by experiment" shows
     mean ± SD across replicates, and "Download CSV" saves the table.
   * **Add New Gate** opens the gate creator in one card, with parent population, mode and
     channel on one row:
     - **1D Histogram**: click on the plot (or type a value) to set the threshold, or use the
       **P95 / P99 / P99.9** quick-picks computed from a chosen control sample. A live
       "% in gate" table for the overlaid samples updates before you save.
     - **2D Quadrant / Polygon** gates for two-channel populations.
6. **Plotting**:

   * Choose the plot type at the top: **1D Histogram**, **2D Scatter**, **Bar Plot** or
     **Pairwise Plot**.
   * Histogram, scatter and bar plots preview **live** on a subsample of up to 50k events as you
     change settings. "Render full data" draws the final plot from every event, for the canvas
     or export.
   * Optional per-plot event filters live in the collapsed **"Filter events"** expander. A chip
     shows which filters are active.
   * If no Experiment Name was entered, the plot says so and points you to "Edit metadata".

### 3. Canvas

Every plot has an "Add to Canvas" button. The Canvas tab has two panes:

* **Left**: a thumbnail list of your plots. Click a plot to select it.
* **Right**: the selected plot at full size, with its controls beside it: width/height,
  font/zoom, dot size, font family, colormap, and per-sample or per-category colours.
  "↑ Move up" / "↓ Move down" reorder plots, and "Delete" asks for confirmation.
* Download the selected plot as PNG, SVG or PDF. SVG and PDF stay editable in Illustrator.
* **"Export all (ZIP)"** downloads every plot (PNG/SVG/PDF, numbered in canvas order) plus one
  multi-page PDF of all plots.

## Acknowledgements

* **Eerik Kaseniit**: For developing the foundational pipeline that served as the base for Glow's upgrades.
* **Gao Lab Members**: Special thanks to all Gao Lab members for advice, feedback, and for sharing the testing data used to refine the models.
