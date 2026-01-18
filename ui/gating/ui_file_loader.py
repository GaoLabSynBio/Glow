import streamlit as st
import os
import re
import pandas as pd
from .ui_metadata import render_metadata_upload_section

# Helper for Cell Line Detection
def detect_cell_line_from_path(path):
    if not path: return "COLO"
    filename = os.path.basename(path)
    dirname = os.path.basename(os.path.dirname(path))
    full_str = f"{dirname}_{filename}"
    match = re.search(r'(?:cell[_\-]?line|cl)[_\-]?([a-zA-Z0-9]+)', full_str, re.IGNORECASE)
    if match: return match.group(1).upper()
    known_lines = ["colo", "hek", "hug1n", "snu", "b16", "cho", "hela", "jurkat"]
    for line in known_lines:
        if line.lower() in full_str.lower(): return line.upper().upper()
    return "COLO"

# Mock Experiment Class to emulate Cytoflow structure for downstream compatibility
class MockExperiment:
    def __init__(self, data):
        self.data = data
        # Infer channels (numeric columns not in exclusion list)
        exclude_cols = ['cells', 'singlets', 'Is_Live', 'Is_Singlet', 'LiveGate', 'SingletGate', 'Sample', 'sample']
        self.channels = [c for c in data.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(data[c])]
        # Preserve metadata columns as conditions for plotting and filtering
        self.conditions = {c: "category" for c in data.columns if c not in self.channels}

    def clone(self):
        return MockExperiment(self.data.copy())
    def __repr__(self):
        return f"<MockExperiment: {len(self.data)} events>"

def clear_experiment_state():
    """Clears all experiment-related data from session state to ensure fresh loading."""
    for key in ['base_experiment', 'gated_experiment', 'analysis_experiment',
                'live_gate_params', 'singlet_gate_params', 'fixed_gate_params',
                'live_gate_stats', 'singlet_gate_stats', 'gate_accepted',
                'doublets_removed', 'positive_gates', 'threshold_gate',
                'selected_channels', 'persistent_selected_channels']:
        if key in st.session_state:
            del st.session_state[key]

def render_file_loader(DataLoader, CYTOFLOW_AVAILABLE):
    """
    Renders the File Loading section
    """
    st.subheader("1. File Loading")


    upload_option = st.radio("Choose Input Method", ("Enter Directory Path", "Upload Files / Drag & Drop Folder", "Upload Processed CSV"))

    loaded_files = []

    if upload_option == "Upload Processed CSV":
        st.info("Upload a previously processed CSV file (Full or Gated population) to skip gating. Supports .csv, .csv.gz, and .zip")
        uploaded_csv = st.file_uploader("Upload CSV", type=["csv", "gz", "zip"])

        if uploaded_csv is not None:
             if st.button("Load CSV"):
                try:
                    with st.spinner("Loading CSV..."):
                        df = None
                        # Handle ZIP
                        if uploaded_csv.name.endswith('.zip'):
                            import zipfile
                            with zipfile.ZipFile(uploaded_csv) as z:
                                csv_files = [n for n in z.namelist() if n.endswith('.csv')]
                                if not csv_files:
                                    st.error("No CSV file found inside ZIP archive.")
                                    return
                                # Assume first CSV is the data
                                with z.open(csv_files[0]) as f:
                                    df = pd.read_csv(f)
                        else:
                            # Auto-detect compression based on extension
                            comp_arg = 'gzip' if uploaded_csv.name.endswith('.gz') else 'infer'
                            df = pd.read_csv(uploaded_csv, compression=comp_arg)

                        # Data Integrity Check: Ensure vital columns exist (if we expect FCS-like data)
                        # We at least need some numeric columns to be useful.
                        # If the user expects re-importable data, checks for 'cells' and 'singlets' are good but optional (as per logic below).
                        if df is not None:
                             if df.shape[1] < 2:
                                 st.error("Uploaded CSV appears to have insufficient columns/data.")
                                 return
                             
                             # Check for empty dataframe
                             if df.empty:
                                 st.error("Uploaded CSV is empty.")
                                 return

                        # Clear any existing experiment state before loading new CSV
                        clear_experiment_state()

                        st.session_state.base_experiment = MockExperiment(df)

                        # Handle Gating Logic based on columns
                        if 'cells' in df.columns and 'singlets' in df.columns:
                            # Do NOT filter rows, just create the experiment with the full data
                            # The analysis module will handle the 'cells' and 'singlets' boolean columns
                            st.session_state.gated_experiment = MockExperiment(df)

                            valid_count = (df['cells'] & df['singlets']).sum()
                            st.info(f"Loaded {len(df)} events. Identified {valid_count} valid events (Live & Singlet).")
                        else:
                            # Assume data is already the target population
                            st.warning("'cells' or 'singlets' annotations not found. Assuming imported data is the target population.")
                            st.session_state.gated_experiment = MockExperiment(df)

                        st.session_state.loaded_files = [f"CSV: {uploaded_csv.name}"]
                        st.session_state.gating_complete = True # Auto-complete since we are skipping
                        st.success("Loaded CSV successfully. Gating marked as complete.")

                except Exception as e:
                    st.error(f"Error loading CSV: {e}")

    elif upload_option == "Enter Directory Path":
        input_dir = st.text_input("Enter absolute path to directory containing .fcs files", "/home/xwz/Projects/Flow/flow-data/dhy_20251001_fcs/")

        if st.button("Scan Directory"):
            if CYTOFLOW_AVAILABLE:
                loader = DataLoader()
                if os.path.isdir(input_dir):
                    file_paths = loader.scan_directory(input_dir)
                    st.session_state.scanned_paths = file_paths
                    st.write(f"Found {len(file_paths)} files in {input_dir}")
                else:
                    st.error(f"Directory not found: {input_dir}")
            else:
                st.warning("Cytoflow not installed. showcasing UI logic.")

        if 'scanned_paths' in st.session_state and st.session_state.scanned_paths:
            if CYTOFLOW_AVAILABLE:
                loader = DataLoader()
                # Shared logic with File Upload -> Refactor ideally, but mirroring here for now
                # Step 1: Generate Preview
                if 'preview_meta' not in st.session_state:
                    st.session_state.preview_meta = []

                if st.button("Generate Preview from Directory"):
                    meta = loader.generate_preview_metadata(st.session_state.scanned_paths, rename_strategy='folder_name')
                    for m in meta:
                        m['Cell_Line'] = detect_cell_line_from_path(m.get('original_path'))
                    st.session_state.preview_meta = meta

                if st.session_state.preview_meta:
                    st.subheader("Edit Metadata Structure")

                    # --- Metadata Upload ---
                    render_metadata_upload_section(key_prefix="dir")

                    # 1. Define fields user wants
                    default_cols = ["Replicate", "Cell_Line", "Experiment_Name", "Timepoint", "Concentration"]
                    st.write("Current Metadata Fields:")

                    # User custom fields manager
                    if 'custom_metadata_fields' not in st.session_state:
                         st.session_state.custom_metadata_fields = []

                    with st.expander("Manage Custom Metadata Columns"):
                         c1, c2 = st.columns([3, 1])
                         with c1:
                              new_custom_csv = st.text_input("Add custom column names (comma separated)",
                                                            value=", ".join(st.session_state.custom_metadata_fields),
                                                            help="Example: Part, Promoter, Strain")
                         with c2:
                              if st.button("Update Columns"):
                                   # Parse CSV
                                   if new_custom_csv:
                                        new_fields = [f.strip() for f in new_custom_csv.split(",") if f.strip()]
                                        st.session_state.custom_metadata_fields = new_fields
                                   else:
                                        st.session_state.custom_metadata_fields = []
                                   st.rerun()

                    # Dynamic options
                    preview_df_cols = pd.DataFrame(st.session_state.preview_meta).columns.tolist() if st.session_state.preview_meta else []
                    extra_opts = [c for c in preview_df_cols if c not in default_cols and c not in ['display_name', 'condition', 'original_path']]

                    all_selectable = default_cols + extra_opts + st.session_state.custom_metadata_fields
                    # Remove duplicates while preserving order
                    seen = set()
                    all_selectable = [x for x in all_selectable if not (x in seen or seen.add(x))]

                    cols_to_use = st.multiselect("Select columns to annotate:",
                                                options=all_selectable,
                                                default=["Replicate", "Cell_Line", "Experiment_Name"] + ([extra_opts[0]] if extra_opts else []))

                    # 2. Prepare DataFrame
                    df_preview = pd.DataFrame(st.session_state.preview_meta)

                    # Ensure selected columns exist
                    for col in cols_to_use:
                        if col not in df_preview.columns:
                            df_preview[col] = "" # Default to empty string

                    # Fix types for Editor (Streamlit is strict about types)
                    # Force string type for TextColumn fields to avoid Arrow/Streamlit errors with NaN/Float
                    for col in ["Experiment_Name", "Cell_Line", "Replicate", "condition", "display_name", "original_path"]:
                        if col in df_preview.columns:
                            df_preview[col] = df_preview[col].fillna("").astype(str)

                    # Reorder/Select columns for editor
                    display_cols = ['display_name', 'condition'] + cols_to_use + ['original_path']

                    # 3. Enhanced Editor
                    # Removed constrained SelectboxColumn for Cell_Line to allow new inputs.
                    edited_df = st.data_editor(df_preview[display_cols],
                                                column_config={
                                                    "display_name": st.column_config.TextColumn("Sample Name", required=True),
                                                    "condition": st.column_config.TextColumn("Folder (Auto)", disabled=True),
                                                    "original_path": st.column_config.TextColumn("Original Path", disabled=True),
                                                    "Replicate": st.column_config.TextColumn("Replicate #"),
                                                    "Experiment_Name": st.column_config.TextColumn("Experiment Name"),
                                                    "Cell_Line": st.column_config.TextColumn("Cell Line"),
                                                    "Timepoint": st.column_config.NumberColumn("Time (h)", format="%d h")
                                                },
                                                width="stretch",
                                                num_rows="fixed")

                    if st.button("Confirm and Load Directory files"):
                        with st.spinner("Processing files..."):
                            clear_experiment_state()
                            # Auto-Uppercase Manual Input for Cell_Line
                            if 'Cell_Line' in edited_df.columns:
                                edited_df['Cell_Line'] = edited_df['Cell_Line'].astype(str).str.upper()
                            final_meta = edited_df.to_dict('records')
                            loaded_files = loader.process_files(final_meta, output_dir="processed_fcs_ui")
                            st.session_state.loaded_files = loaded_files
                            st.success(f"Successfully loaded {len(loaded_files)} files with metadata.")
                            st.write("Loaded Metadata Sample:", pd.DataFrame(loaded_files).head())

    elif upload_option == "Upload Files / Drag & Drop Folder":
        st.info("You can drag and drop a folder containing FCS files directly into the box below. All files in the folder will be uploaded.")

        col_up1, col_up2 = st.columns([1, 1])
        with col_up1:
            upload_folder_name = st.text_input("Name this Upload Batch (will be used as Folder Name)", value="My_Experiment_1")

        uploaded_files = st.file_uploader("Upload FCS Files or Folder", type=['fcs'], accept_multiple_files=True)

        if uploaded_files:
            if st.button("Process Uploaded Files"):
                if CYTOFLOW_AVAILABLE:
                    loader = DataLoader()
                    with st.spinner("Saving uploaded files..."):
                        saved_paths = loader.save_uploaded_files(uploaded_files, target_dir="data/uploaded_raw", folder_name=upload_folder_name)

                    st.session_state.uploaded_paths = saved_paths
                    st.success(f"Saved {len(saved_paths)} files to folder '{upload_folder_name}'.")
                else:
                     st.warning("Cytoflow/Glow backend not available.")

        if 'uploaded_paths' in st.session_state and st.session_state.uploaded_paths:
            if CYTOFLOW_AVAILABLE:
                loader = DataLoader()

                if 'preview_meta' not in st.session_state:
                     st.session_state.preview_meta = []

                if st.button("Generate Preview from Uploads"):
                     meta = loader.generate_preview_metadata(st.session_state.uploaded_paths, rename_strategy='folder_name')
                     for m in meta:
                        m['Cell_Line'] = detect_cell_line_from_path(m.get('original_path'))
                     st.session_state.preview_meta = meta

                if st.session_state.preview_meta:
                    st.subheader("Edit Metadata Structure")
                    render_metadata_upload_section(key_prefix="upload")

                    default_cols = ["Replicate", "Cell_Line", "Experiment_Name", "Timepoint", "Concentration"]
                    preview_df_cols = pd.DataFrame(st.session_state.preview_meta).columns.tolist() if st.session_state.preview_meta else []
                    extra_opts = [c for c in preview_df_cols if c not in default_cols and c not in ['display_name', 'condition', 'original_path']]

                    cols_to_use = st.multiselect("Select columns to annotate:",
                                                options=default_cols + extra_opts + ["Custom1", "Custom2"],
                                                default=["Replicate", "Cell_Line", "Experiment_Name"] + ([extra_opts[0]] if extra_opts else []),
                                                key="upload_cols")

                    df_preview = pd.DataFrame(st.session_state.preview_meta)
                    for col in cols_to_use:
                        if col not in df_preview.columns:
                            df_preview[col] = ""

                    for col in ["Experiment_Name", "Cell_Line", "Replicate", "condition", "display_name", "original_path"]:
                        if col in df_preview.columns:
                            df_preview[col] = df_preview[col].fillna("").astype(str)

                    display_cols = ['display_name', 'condition'] + cols_to_use + ['original_path']

                    edited_df = st.data_editor(df_preview[display_cols],
                                                column_config={
                                                    "display_name": st.column_config.TextColumn("Sample Name", required=True),
                                                    "condition": st.column_config.TextColumn("Folder (Auto)", disabled=True),
                                                    "original_path": st.column_config.TextColumn("Original Path", disabled=True),
                                                    "Replicate": st.column_config.TextColumn("Replicate #"),
                                                    "Experiment_Name": st.column_config.TextColumn("Experiment Name"),
                                                    "Cell_Line": st.column_config.TextColumn("Cell Line"),
                                                    "Timepoint": st.column_config.NumberColumn("Time (h)", format="%d h")
                                                },
                                                width="stretch",
                                                num_rows="fixed",
                                                key="upload_editor")

                    if st.button("Confirm and Load Uploaded files"):
                        with st.spinner("Processing files..."):
                            clear_experiment_state()
                            # Auto-Uppercase Manual Input for Cell_Line
                            if 'Cell_Line' in edited_df.columns:
                                edited_df['Cell_Line'] = edited_df['Cell_Line'].astype(str).str.upper()
                            final_meta = edited_df.to_dict('records')
                            loaded_files = loader.process_files(final_meta, output_dir="processed_fcs_ui")
                            st.session_state.loaded_files = loaded_files
                            st.success(f"Successfully loaded {len(loaded_files)} files with metadata.")

    # Access loaded files from session state
    if 'loaded_files' in st.session_state:
        loaded_files = st.session_state.loaded_files

    return loaded_files

