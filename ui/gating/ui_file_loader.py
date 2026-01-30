import streamlit as st
import os
import re
import pandas as pd
from .ui_metadata import render_metadata_upload_section

# Helper for Cell Line Detection
def detect_cell_line_from_path(path):
    if not path:
        return "HEK"
    filename = os.path.basename(path)
    dirname = os.path.basename(os.path.dirname(path))
    full_str = f"{dirname}_{filename}"
    match = re.search(r'(?:cell[_\-]?line|cl)[_\-]?([a-zA-Z0-9]+)', full_str, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    known_lines = ["colo", "hek", "hug1n", "snu", "b16", "cho", "hela", "jurkat"]
    for line in known_lines:
        if line.lower() in full_str.lower():
            return line.upper().upper()
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

def reset_singlet_gating(keep_live_gate=True):
    """Clears singlet-specific gating state."""
    keys = [
        'singlet_gate_params', 'singlet_gate_stats', 'doublets_removed',
        'cell_line_singlet_gates', 'pending_doublet_verts', 'doublet_verts_input',
        'doublet_manual_mode', 'show_doublet_gating'
    ]
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]

    # If keeping live gate, signal that we need to reconstruct the live-only experiment
    # to "undo" any singlet filtering that happened to gated_experiment
    if keep_live_gate and st.session_state.get('gate_accepted'):
        st.session_state.reconstruct_live_gate = True

def reset_live_gating():
    """Clears live gating state and singlet state."""
    reset_singlet_gating(keep_live_gate=False)

    keys = [
        'live_gate_params', 'live_gate_stats', 'gate_accepted',
        'cell_line_gates', 'last_gating_params', 'current_auto_boundary',
        'suggested_live_label', 'manual_vertices', 'pending_manual_vertices',
        'manual_live_verts_input', 'show_manual_gating', 'gated_experiment',
        'reconstruct_live_gate'
    ]
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]

def clear_experiment_state():
    """Clears all experiment-related data from session state to ensure fresh loading."""
    # Full wipe
    reset_live_gating()

    for key in ['base_experiment', 'analysis_experiment',
                'fixed_gate_params', 'positive_gates', 'threshold_gate',
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
                    st.subheader("Metadata Editing")

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

                    # 2. Configure View Options
                    # Logic for default selections
                    default_cols = ["Replicate", "Cell_Line", "Experiment_Name", "Timepoint", "Concentration"]
                    preview_df_cols = pd.DataFrame(st.session_state.preview_meta).columns.tolist() if st.session_state.preview_meta else []
                    extra_opts = [c for c in preview_df_cols if c not in default_cols and c not in ['display_name', 'condition', 'original_path']]

                    # Available options computation
                    all_selectable = default_cols + extra_opts + st.session_state.dir_custom_cols
                    seen = set()
                    all_selectable = [x for x in all_selectable if not (x in seen or seen.add(x))]

                    # Initialize Default Selection (First Run Only)
                    if 'dir_cols_selected' not in st.session_state:
                         default_selection = ["Experiment_Name", "Replicate", "Cell_Line"] + ([extra_opts[0]] if extra_opts else [])
                         st.session_state.dir_cols_selected = [c for c in default_selection if c in all_selectable]

                    # 5. Prepare DataFrame from Persistent State
                    # Ensure selected columns exist in the source of truth
                    # (We use the LAST known selection to prep the view)
                    current_view_cols = st.session_state.dir_cols_selected
                    for col in current_view_cols:
                        if col not in st.session_state.working_df_dir.columns:
                            st.session_state.working_df_dir[col] = ""

                    # Ensure types are string for text columns to prevent Streamlit errors
                    for col in ["Experiment_Name", "Cell_Line", "Replicate", "condition", "display_name", "original_path"]:
                        if col in st.session_state.working_df_dir.columns:
                            st.session_state.working_df_dir[col] = st.session_state.working_df_dir[col].fillna("").astype(str)

                    # Reorder/Select columns for editor
                    display_cols = ['display_name', 'condition'] + current_view_cols + ['original_path']

                    # Create a specific view for the editor
                    df_view = st.session_state.working_df_dir[display_cols].copy()

                    # 3. Enhanced Editor WRAPPED IN FORM
                    # Moving "Add Custom Column" INSIDE the form ensures pending edits are submitted when you click "Add".
                    with st.form("dir_metadata_form"):
                        # A. Add Custom Column UI (Safety: Submitting valid form saves data)
                        st.write("Add Custom Column (Clicking 'Add' saves current table data)")
                        c1, c2 = st.columns([0.7, 0.3])
                        new_col_name = c1.text_input("Add Custom Column", label_visibility="collapsed")
                        add_col_btn = c2.form_submit_button("Add New Column")

                        # B. Column Selector logic is tricky inside forms.
                        # We display the multiselect, but changes only apply on SUBMIT (Add Column or Confirm).
                        # This matches the user request for safety.
                        st.write("Select columns to annotate (Click 'Update View' or 'Load Files' to apply changes)")
                        cols_selected = st.multiselect("Select columns",
                                                    options=all_selectable,
                                                    default=st.session_state.dir_cols_selected,
                                                    label_visibility="collapsed")

                        update_view_btn = st.form_submit_button("Update View / Save Progress")

                        # C. The Editor
                        edited_df = st.data_editor(df_view,
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

                        confirm_dir = st.form_submit_button("Confirm and Load Directory Files")

                    # --- LOGIC HANDLING ---
                    # 1. First, SAVE ANY PENDING DATA if any button was clicked
                    if add_col_btn or update_view_btn or confirm_dir:
                         if edited_df is not None:
                              # Auto-populate Replicate if Experiment Name changed and Replicate did not
                              old_df = st.session_state.working_df_dir
                              if "Experiment_Name" in edited_df.columns and "Replicate" in edited_df.columns:
                                   for idx in edited_df.index:
                                        if idx in old_df.index:
                                             new_exp = str(edited_df.at[idx, "Experiment_Name"])
                                             old_exp = str(old_df.at[idx, "Experiment_Name"]) if "Experiment_Name" in old_df.columns else ""
                                             new_rep = str(edited_df.at[idx, "Replicate"])
                                             old_rep = str(old_df.at[idx, "Replicate"]) if "Replicate" in old_df.columns else ""

                                             if new_exp != old_exp and new_rep == old_rep:
                                                  edited_df.at[idx, "Replicate"] = new_exp

                              st.session_state.working_df_dir.update(edited_df)

                         # Save column selection
                         st.session_state.dir_cols_selected = cols_selected

                    # 2. Handle "Add Column" specifically
                    if add_col_btn and new_col_name:
                         cleaned_name = new_col_name.strip()
                         if cleaned_name and cleaned_name not in st.session_state.dir_custom_cols:
                              st.session_state.dir_custom_cols.append(cleaned_name)
                              if cleaned_name not in st.session_state.dir_cols_selected:
                                   st.session_state.dir_cols_selected.append(cleaned_name)
                         st.rerun() # Rerun to update the Editor View with new column

                    # 3. Handle "Update View"
                    if update_view_btn:
                         st.rerun()

                    # 4. Handle "Confirm"
                    if confirm_dir: # already saved data above

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
                                    if not clean_k:
                                        clean_k = "col_" + str(hash(k))
                                    clean_m[clean_k] = v
                                sanitized_meta_list.append(clean_m)

                            loaded_files = loader.process_files(sanitized_meta_list, output_dir="processed_fcs_ui")
                            st.session_state.loaded_files = loaded_files
                            st.success(f"Successfully loaded {len(loaded_files)} files with metadata.")
                            st.write("Loaded Metadata Example:", pd.DataFrame(loaded_files).head())

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
                     for m in meta:
                        m['Cell_Line'] = detect_cell_line_from_path(m.get('original_path'))
                     st.session_state.preview_meta = meta
                     st.session_state.working_df = pd.DataFrame(meta)
                     # Do NOT reset custom columns or selection to allow persistence
                     if "new_col_input" in st.session_state:
                         del st.session_state["new_col_input"]

                if st.session_state.preview_meta:
                    st.subheader("Metadata Editing")
                    if 'working_df' not in st.session_state:
                         st.session_state.working_df = pd.DataFrame(st.session_state.preview_meta)

                    # Sync back to preview_meta so upload uses current state
                    if 'working_df' in st.session_state:
                         st.session_state.preview_meta = st.session_state.working_df.to_dict('records')

                    render_metadata_upload_section(key_prefix="upload", reset_keys=['working_df'])

                    if 'user_custom_cols' not in st.session_state:
                        st.session_state.user_custom_cols = []

                    # 2. Configure View Options for Uploads
                    default_cols = ["Replicate", "Cell_Line", "Experiment_Name", "Timepoint", "Concentration"]
                    preview_df_cols = pd.DataFrame(st.session_state.preview_meta).columns.tolist() if st.session_state.preview_meta else []
                    extra_opts = [c for c in preview_df_cols if c not in default_cols and c not in ['display_name', 'condition', 'original_path']]

                    # 3. Define Available Options
                    available_options = default_cols + extra_opts + st.session_state.user_custom_cols
                    seen = set()
                    available_options = [x for x in available_options if not (x in seen or seen.add(x))]

                    # 4. Initialize Selection Default (First Run Only)
                    if 'upload_cols_selected' not in st.session_state:
                        default_selection = ["Experiment_Name", "Replicate", "Cell_Line"] + ([extra_opts[0]] if extra_opts else [])
                        st.session_state.upload_cols_selected = [c for c in default_selection if c in available_options]

                    # 5. Prepare View
                    current_view_cols = st.session_state.upload_cols_selected

                    # Sync Columns
                    for col in current_view_cols:
                        if col not in st.session_state.working_df.columns:
                            st.session_state.working_df[col] = ""

                    # Sync Types
                    for col in ["Experiment_Name", "Cell_Line", "Replicate", "condition", "display_name", "original_path"]:
                        if col in st.session_state.working_df.columns:
                             st.session_state.working_df[col] = st.session_state.working_df[col].fillna("").astype(str)

                    # Slice for Editor
                    display_cols = ['display_name', 'condition'] + current_view_cols + ['original_path']
                    df_view = st.session_state.working_df[display_cols].copy()

                    # WRAPPED IN FORM to prevent auto-refresh on every keystroke
                    with st.form("upload_metadata_form"):
                        # A. Add Custom Column UI (Safety: Submitting valid form saves data)
                        st.write("Add Custom Column (Clicking 'Add' saves current table data)")
                        c1, c2 = st.columns([0.7, 0.3])
                        new_col_name = c1.text_input("Add Custom Column", label_visibility="collapsed")
                        add_col_btn = c2.form_submit_button("Add New Column")

                        # B. Column Selector logic
                        st.write("Select columns to annotate (Click 'Update View' or 'Load Files' to apply changes)")
                        cols_selected = st.multiselect("Select columns",
                                                    options=available_options,
                                                    default=st.session_state.upload_cols_selected,
                                                    label_visibility="collapsed")

                        update_view_btn = st.form_submit_button("Update View / Save Progress")

                        # C. Editor
                        edited_df = st.data_editor(df_view,
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

                        confirm_upload = st.form_submit_button("Confirm and Load Uploaded files")

                    # --- LOGIC HANDLING ---
                    # 1. First, SAVE ANY PENDING DATA
                    if add_col_btn or update_view_btn or confirm_upload:
                         if edited_df is not None:
                              # Auto-populate Replicate if Experiment Name changed and Replicate did not
                              old_df = st.session_state.working_df
                              if "Experiment_Name" in edited_df.columns and "Replicate" in edited_df.columns:
                                   for idx in edited_df.index:
                                        if idx in old_df.index:
                                             new_exp = str(edited_df.at[idx, "Experiment_Name"])
                                             old_exp = str(old_df.at[idx, "Experiment_Name"]) if "Experiment_Name" in old_df.columns else ""
                                             new_rep = str(edited_df.at[idx, "Replicate"])
                                             old_rep = str(old_df.at[idx, "Replicate"]) if "Replicate" in old_df.columns else ""

                                             if new_exp != old_exp and new_rep == old_rep:
                                                  edited_df.at[idx, "Replicate"] = new_exp

                              st.session_state.working_df.update(edited_df)

                         st.session_state.upload_cols_selected = cols_selected

                    # 2. Handle "Add Column"
                    if add_col_btn and new_col_name:
                         cleaned_name = new_col_name.strip()
                         if cleaned_name and cleaned_name not in st.session_state.user_custom_cols:
                              st.session_state.user_custom_cols.append(cleaned_name)
                              if cleaned_name not in st.session_state.upload_cols_selected:
                                   st.session_state.upload_cols_selected.append(cleaned_name)
                         st.rerun()

                    # 3. Handle "Update View"
                    if update_view_btn:
                         st.rerun()

                    if confirm_upload:
                         # Data already saved above
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
                                    if not clean_k:
                                        clean_k = "col_" + str(hash(k))
                                    clean_m[clean_k] = v
                                sanitized_meta_list.append(clean_m)

                            loaded_files = loader.process_files(sanitized_meta_list, output_dir="processed_fcs_ui")
                            st.session_state.loaded_files = loaded_files
                            st.success(f"Successfully loaded {len(loaded_files)} files with metadata.")

    # Access loaded files from session state
    if 'loaded_files' in st.session_state:
        loaded_files = st.session_state.loaded_files

    return loaded_files

