import streamlit as st
import os
import pandas as pd
from .ui_metadata import render_metadata_upload_section

def render_file_loader(DataLoader, CYTOFLOW_AVAILABLE):
    """
    Renders the File Loading section
    """
    st.subheader("1. File Loading")
    
    # Renaming Preference
    rename_strategy = st.sidebar.radio("Renaming Strategy", ('folder_name', 'keep'))

    upload_option = st.radio("Choose Input Method", ("Enter Directory Path", "Upload File Paths List", "Upload Files / Drag & Drop Folder"))

    loaded_files = []

    if upload_option == "Enter Directory Path":
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
                    st.session_state.preview_meta = loader.generate_preview_metadata(st.session_state.scanned_paths, rename_strategy=rename_strategy)
                
                if st.session_state.preview_meta:
                    st.subheader("Edit Metadata Structure")
                    
                    # --- Metadata Upload ---
                    render_metadata_upload_section(key_prefix="dir")
                    
                    # 1. Define fields user wants
                    default_cols = ["Replicate", "Cell_Line", "Experiment_Name", "Timepoint", "Concentration"]
                    st.write("Current Metadata Fields:")
                    
                    # Dynamic options
                    preview_df_cols = pd.DataFrame(st.session_state.preview_meta).columns.tolist() if st.session_state.preview_meta else []
                    extra_opts = [c for c in preview_df_cols if c not in default_cols and c not in ['display_name', 'condition', 'original_path']]
                    
                    cols_to_use = st.multiselect("Select columns to annotate:", 
                                                options=default_cols + extra_opts + ["Custom1", "Custom2"],
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
                                                use_container_width=True,
                                                num_rows="fixed")
                    
                    if st.button("Confirm and Load Directory files"):
                        with st.spinner("Processing files..."):
                            final_meta = edited_df.to_dict('records')
                            loaded_files = loader.process_files(final_meta, output_dir="processed_fcs_ui")
                            st.session_state.loaded_files = loaded_files
                            st.success(f"Successfully loaded {len(loaded_files)} files with metadata.")
                            st.write("Loaded Metadata Sample:", pd.DataFrame(loaded_files).head())

    elif upload_option == "Upload File Paths List":
        st.info("Format: /path/to/file.fcs OR /path/to/file.fcs, NewName")
        path_file = st.file_uploader("Upload .txt with file paths", type=['txt'])
        if path_file:
            # Save temp to read
            with open("temp_paths.txt", "wb") as f:
                f.write(path_file.getbuffer())
                
            if CYTOFLOW_AVAILABLE:
                loader = DataLoader() # Re-instantiate
                file_paths = loader.parse_file_paths_from_text("temp_paths.txt")
                st.write(f"Found {len(file_paths)} paths.")
                
                # Step 1: Generate Preview
                if 'preview_meta' not in st.session_state:
                    st.session_state.preview_meta = []

                if st.button("Generate Preview"):
                    st.session_state.preview_meta = loader.generate_preview_metadata(file_paths, rename_strategy=rename_strategy)
                
                if st.session_state.preview_meta:
                    st.subheader("Edit Metadata Structure")
                    
                    # --- Metadata Upload ---
                    render_metadata_upload_section(key_prefix="list")

                    # 1. Define fields
                    # User requested: "Treatment" -> "Experiment_Name"
                    default_cols = ["Replicate", "Cell_Line", "Experiment_Name"]
                    
                    preview_df_cols = pd.DataFrame(st.session_state.preview_meta).columns.tolist() if st.session_state.preview_meta else []
                    extra_opts = [c for c in preview_df_cols if c not in default_cols and c not in ['display_name', 'condition', 'original_path', "Timepoint", "Concentration", "Treatment"]]

                    cols_to_use = st.multiselect("Select columns:", 
                                                options=default_cols + extra_opts + ["Timepoint", "Concentration", "Treatment"], # Kept Treatment just in case
                                                default=default_cols, key="txt_cols")
                                                
                    df_preview = pd.DataFrame(st.session_state.preview_meta)
                    
                    for col in cols_to_use:
                        if col not in df_preview.columns:
                            # Default value logic
                            if col == "Replicate":
                                df_preview[col] = "1"
                            else:
                                df_preview[col] = "" 
                    
                    # Fix types for Editor (Streamlit is strict about types)
                    for col in ["Experiment_Name", "Cell_Line", "Replicate", "condition", "display_name", "original_path"]:
                        if col in df_preview.columns:
                            df_preview[col] = df_preview[col].fillna("").astype(str)

                    display_cols = ['display_name', 'condition'] + cols_to_use + ['original_path']
                    
                    # User requested: Input a new cell line name (free text), copy paste support (standard).
                    # We remove the constrained SelectboxColumn for Cell_Line.
                    edited_df = st.data_editor(df_preview[display_cols],
                                                column_config={
                                                    "display_name": "Sample Name",
                                                    "original_path": st.column_config.TextColumn("Original Path", disabled=True),
                                                    "condition": st.column_config.TextColumn("Folder", disabled=True),
                                                },
                                                use_container_width=True, num_rows="fixed")
                    
                    # Step 2: Confirm and Process
                    if st.button("Confirm and Load files"):
                        with st.spinner("Processing files..."):
                            # Convert dataframe back to list of dicts
                            final_meta = edited_df.to_dict('records')
                            # Process (copy/rename)
                            loaded_files = loader.process_files(final_meta, output_dir="processed_fcs_ui")
                            st.session_state.loaded_files = loaded_files
                            st.success(f"Successfully loaded and renamed {len(loaded_files)} files.")

            else:
                st.warning("Cytoflow not installed. showcasing UI logic only.")
                st.dataframe(pd.DataFrame({'display_name': ['demo_file_1.fcs', 'demo_file_2.fcs'], 'path': ['/tmp/A/1.fcs', '/tmp/B/2.fcs']}))

    elif upload_option == "Upload Files / Drag & Drop Folder":
        st.info("Drag and drop your folder or files here. Browsers handle folder drops by uploading all containing files.")
        uploaded_files = st.file_uploader("Upload FCS Files", type=['fcs'], accept_multiple_files=True)
        
        if uploaded_files:
            if st.button("Process Uploaded Files"):
                if CYTOFLOW_AVAILABLE:
                    loader = DataLoader() 
                    with st.spinner("Saving uploaded files..."):
                        saved_paths = loader.save_uploaded_files(uploaded_files, target_dir="data/uploaded_raw")
                    
                    st.session_state.uploaded_paths = saved_paths
                    st.success(f"Saved {len(saved_paths)} files.")
                else:
                     st.warning("Cytoflow/Glow backend not available.")

        if 'uploaded_paths' in st.session_state and st.session_state.uploaded_paths:
            if CYTOFLOW_AVAILABLE:
                loader = DataLoader()
                
                if 'preview_meta' not in st.session_state:
                     st.session_state.preview_meta = []
                
                if st.button("Generate Preview from Uploads"):
                     st.session_state.preview_meta = loader.generate_preview_metadata(st.session_state.uploaded_paths, rename_strategy=rename_strategy)
                
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
                                                use_container_width=True,
                                                num_rows="fixed",
                                                key="upload_editor")
                    
                    if st.button("Confirm and Load Uploaded files"):
                        with st.spinner("Processing files..."):
                            final_meta = edited_df.to_dict('records')
                            loaded_files = loader.process_files(final_meta, output_dir="processed_fcs_ui")
                            st.session_state.loaded_files = loaded_files
                            st.success(f"Successfully loaded {len(loaded_files)} files with metadata.")
    
    # Access loaded files from session state
    if 'loaded_files' in st.session_state:
        loaded_files = st.session_state.loaded_files
        
    return loaded_files

