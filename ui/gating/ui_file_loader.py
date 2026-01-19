import streamlit as st
import os
import re
import pandas as pd
from .ui_metadata import render_metadata_upload_section

# Helper for Cell Line Detection
def detect_cell_line_from_path(path):
    if not path: return "HEK"
    filename = os.path.basename(path)
    dirname = os.path.basename(os.path.dirname(path))
    full_str = f"{dirname}_{filename}"
    match = re.search(r'(?:cell[_\-]?line|cl)[_\-]?([a-zA-Z0-9]+)', full_str, re.IGNORECASE)
    if match: return match.group(1).upper()
    known_lines = ["colo", "hek", "hug1n", "snu", "b16", "cho", "hela", "jurkat"]
    for line in known_lines:
        if line.lower() in full_str.lower(): return line.upper().upper()
    return "HEK"

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
                    st.session_state.working_df_dir = pd.DataFrame(meta)
                    # Reset
                    if 'dir_cols_selected' in st.session_state:
                         del st.session_state['dir_cols_selected']
                    if 'dir_custom_cols' in st.session_state:
                         del st.session_state['dir_custom_cols']
                    if 'dir_new_col_input' in st.session_state:
                         del st.session_state['dir_new_col_input']

                if st.session_state.preview_meta:
                    st.subheader("Edit Metadata Structure")

                    if 'working_df_dir' not in st.session_state:
                         st.session_state.working_df_dir = pd.DataFrame(st.session_state.preview_meta)

                    # Sync back to preview_meta so upload uses current state
                    if 'working_df_dir' in st.session_state:
                         st.session_state.preview_meta = st.session_state.working_df_dir.to_dict('records')

                    # --- Metadata Upload ---
                    render_metadata_upload_section(key_prefix="dir", reset_keys=['working_df_dir'])

                    # 1. Custom Fields Persistence
                    if 'dir_custom_cols' not in st.session_state:
                         st.session_state.dir_custom_cols = []

                    default_cols = ["Replicate", "Cell_Line", "Experiment_Name", "Timepoint", "Concentration"]
                    preview_df_cols = pd.DataFrame(st.session_state.preview_meta).columns.tolist() if st.session_state.preview_meta else []
                    extra_opts = [c for c in preview_df_cols if c not in default_cols and c not in ['display_name', 'condition', 'original_path']]

                    # 2. Add New Column Interface
                    def add_dir_custom_column():
                         new_col = st.session_state.get("dir_new_col_input", "").strip()
                         if new_col:
                              if new_col not in st.session_state.dir_custom_cols:
                                   st.session_state.dir_custom_cols.append(new_col)
                              
                              current = st.session_state.get("dir_cols_selected", [])
                              if new_col not in current:
                                   st.session_state.dir_cols_selected = current + [new_col]
                              st.session_state.dir_new_col_input = ""

                    c1, c2 = st.columns([0.7, 0.3])
                    with c1:
                         st.text_input("Add Custom Column", key="dir_new_col_input", on_change=add_dir_custom_column, placeholder="Type column name and press Enter")
                    with c2:
                         st.write("")
                         st.write("")
                         st.button("Add", key="dir_add_btn", on_click=add_dir_custom_column)

                    # 3. Available Options
                    all_selectable = default_cols + extra_opts + st.session_state.dir_custom_cols
                    seen = set()
                    all_selectable = [x for x in all_selectable if not (x in seen or seen.add(x))]

                    # Sanitize
                    if 'dir_cols_selected' in st.session_state:
                         st.session_state.dir_cols_selected = [c for c in st.session_state.dir_cols_selected if c in all_selectable]

                    # 4. Defaults
                    if 'dir_cols_selected' not in st.session_state:
                         default_selection = ["Replicate", "Cell_Line", "Experiment_Name"] + ([extra_opts[0]] if extra_opts else [])
                         st.session_state.dir_cols_selected = [c for c in default_selection if c in all_selectable]

                    cols_to_use = st.multiselect("Select columns to annotate:",
                                                options=all_selectable,
                                                key="dir_cols_selected")

                    # 5. Prepare DataFrame from Persistent State
                    df_preview = st.session_state.working_df_dir


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
                                                num_rows="fixed",
                                                key="dir_editor")

                    # Persist edits
                    if edited_df is not None:
                         st.session_state.working_df_dir[edited_df.columns] = edited_df

                    if st.button("Confirm and Load Directory files"):
                        with st.spinner("Processing files..."):
                            clear_experiment_state()
                            # Auto-Uppercase Manual Input for Cell_Line
                            if 'Cell_Line' in edited_df.columns:
                                edited_df['Cell_Line'] = edited_df['Cell_Line'].astype(str).str.upper()
                            final_meta = edited_df.to_dict('records')

                            # Local Sanitization Fix: Propagate clean keys immediately
                            sanitized_meta_list = []
                            for m in final_meta:
                                clean_m = {}
                                for k, v in m.items():
                                    clean_k = re.sub(r'[^0-9a-zA-Z_]', '_', str(k))
                                    if clean_k and clean_k[0].isdigit():
                                         clean_k = "_" + clean_k
                                    if not clean_k: clean_k = "col_" + str(hash(k))
                                    clean_m[clean_k] = v
                                sanitized_meta_list.append(clean_m)

                            loaded_files = loader.process_files(sanitized_meta_list, output_dir="processed_fcs_ui")
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
                     meta = loader.generate_preview_metadata(st.session_state.uploaded_paths)
                     st.session_state.preview_meta = meta
                     st.session_state.working_df = pd.DataFrame(meta)
                     # Do NOT reset custom columns or selection to allow persistence
                     if "new_col_input" in st.session_state:
                         del st.session_state["new_col_input"]

                if st.session_state.preview_meta:
                    st.subheader("Edit Metadata Structure")
                    if 'working_df' not in st.session_state:
                         st.session_state.working_df = pd.DataFrame(st.session_state.preview_meta)

                    # Sync back to preview_meta so upload uses current state
                    if 'working_df' in st.session_state:
                         st.session_state.preview_meta = st.session_state.working_df.to_dict('records')

                    render_metadata_upload_section(key_prefix="upload", reset_keys=['working_df'])

                    # 1. Manage Custom Columns Persistence
                    if 'user_custom_cols' not in st.session_state:
                        st.session_state.user_custom_cols = []

                    default_cols = ["Replicate", "Cell_Line", "Experiment_Name", "Timepoint", "Concentration"]
                    preview_df_cols = pd.DataFrame(st.session_state.preview_meta).columns.tolist() if st.session_state.preview_meta else []
                    extra_opts = [c for c in preview_df_cols if c not in default_cols and c not in ['display_name', 'condition', 'original_path']]

                    # 2. Add New Column Interface
                    def add_custom_column():
                        new_col = st.session_state.get("new_col_input", "").strip()
                        if new_col:
                            # Add to custom columns list if not present
                            if new_col not in st.session_state.user_custom_cols:
                                st.session_state.user_custom_cols.append(new_col)
                            
                            # Add to selection list if not present
                            current_selection = st.session_state.get("upload_cols_selected", [])
                            if new_col not in current_selection:
                                st.session_state.upload_cols_selected = current_selection + [new_col]
                            
                            # Clear input
                            st.session_state.new_col_input = ""

                    c1, c2 = st.columns([0.7, 0.3])
                    with c1:
                        st.text_input("Add Custom Column", key="new_col_input", on_change=add_custom_column, placeholder="Type column name and press Enter")
                    with c2:
                        st.write("") # Spacer
                        st.write("")
                        st.button("Add", on_click=add_custom_column)

                    # 3. Define Available Options
                    available_options = default_cols + extra_opts + st.session_state.user_custom_cols
                    # Deduplicate while preserving order
                    seen = set()
                    available_options = [x for x in available_options if not (x in seen or seen.add(x))]

                    # Sanitize existing selection to avoid errors if options changed
                    if 'upload_cols_selected' in st.session_state:
                         st.session_state.upload_cols_selected = [c for c in st.session_state.upload_cols_selected if c in available_options]

                    # 4. Initialize Selection Default (First Run Only)
                    if 'upload_cols_selected' not in st.session_state:
                        default_selection = ["Replicate", "Cell_Line", "Experiment_Name"] + ([extra_opts[0]] if extra_opts else [])
                        # Ensure defaults are valid options
                        st.session_state.upload_cols_selected = [c for c in default_selection if c in available_options]

                    # 5. Multiselect for Display/Removal
                    cols_to_use = st.multiselect("Selected Metadata Columns:",
                                                options=available_options,
                                                key="upload_cols_selected")

                    df_preview = st.session_state.working_df
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

                    # Persist edits to working_df so they survive column changes
                    if edited_df is not None:
                         st.session_state.working_df[edited_df.columns] = edited_df

                    if st.button("Confirm and Load Uploaded files"):
                        with st.spinner("Processing files..."):
                            clear_experiment_state()
                            # Auto-Uppercase Manual Input for Cell_Line
                            if 'Cell_Line' in edited_df.columns:
                                edited_df['Cell_Line'] = edited_df['Cell_Line'].astype(str).str.upper()
                            final_meta = edited_df.to_dict('records')

                            # Local Sanitization Fix: Propagate clean keys immediately
                            sanitized_meta_list = []
                            for m in final_meta:
                                clean_m = {}
                                for k, v in m.items():
                                    clean_k = re.sub(r'[^0-9a-zA-Z_]', '_', str(k))
                                    if clean_k and clean_k[0].isdigit():
                                         clean_k = "_" + clean_k
                                    if not clean_k: clean_k = "col_" + str(hash(k))
                                    clean_m[clean_k] = v
                                sanitized_meta_list.append(clean_m)

                            loaded_files = loader.process_files(sanitized_meta_list, output_dir="processed_fcs_ui")
                            st.session_state.loaded_files = loaded_files
                            st.success(f"Successfully loaded {len(loaded_files)} files with metadata.")

    # Access loaded files from session state
    if 'loaded_files' in st.session_state:
        loaded_files = st.session_state.loaded_files

    return loaded_files

