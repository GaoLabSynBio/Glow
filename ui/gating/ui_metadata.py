import streamlit as st
import pandas as pd
import numpy as np
from glow.data_loading.metadata_parser import parse_metadata_file

def render_metadata_upload_section(key_prefix="", reset_keys=None):
    """
    Helper to allow uploading a CSV/Excel and merging it into st.session_state.preview_meta
    """
    if reset_keys is None:
        reset_keys = []
    st.markdown("---")
    st.write("### Bulk Metadata Upload (CSV/Excel)")
    with st.expander("Upload Metadata File", expanded=True):
        uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx', 'xls'], key=f"{key_prefix}_uploader")

        if uploaded_file:
            try:
                # 1. Read Raw (lines)
                if uploaded_file.name.endswith('.csv'):
                    df_raw = pd.read_csv(uploaded_file, header=None)
                else:
                    df_raw = pd.read_excel(uploaded_file, header=None)

                st.write("#### 1. File Preview (Raw Content)")
                st.dataframe(df_raw.head(20), width="stretch")

                # 2. Configuration for Parsing
                st.write("#### 2. Parse Settings")
                col_cfg1, col_cfg2 = st.columns(2)

                header_row_idx = col_cfg1.number_input("Header Row Index (0-based)", min_value=0, value=0, key=f"{key_prefix}_header_idx")
                data_start_idx = col_cfg2.number_input("Data Start Row Index", min_value=header_row_idx+1, value=header_row_idx+1, key=f"{key_prefix}_data_idx")

                # 3. Process DataFrame
                # Set headers
                if header_row_idx < len(df_raw):
                    try:
                        df_parsed = parse_metadata_file(
                            uploaded_file=None,
                            df_raw=df_raw,
                            header_row_idx=header_row_idx,
                            data_start_idx=data_start_idx,
                            global_exp_name_coords=None
                        )
                    except ValueError as ve:
                        st.error(f"Parsing error: {ve}")
                        df_parsed = pd.DataFrame() # Fallback


                    st.write("#### 3. Parsed Metadata Preview")
                    st.dataframe(df_parsed.head(), width="stretch")

                    # --- NEW SECTION: Column Selection & Renaming ---
                    st.write("#### 4. Select & Rename Columns")

                    all_cols = list(df_parsed.columns)
                    # Filter out Untitled ones from default selection maybe?
                    default_cols = [c for c in all_cols if "Untitled" not in c]

                    selected_cols = st.multiselect("Select columns to include", all_cols, default=default_cols, key=f"{key_prefix}_sel_cols")

                    if not selected_cols:
                        st.warning("Please select at least one column to proceed.")
                        # Stop execution here if no columns selected to prevent errors below
                    else:
                        # Allow renaming
                        st.write("Rename columns (edit 'Target Name' column):")
                        renaming_data = [{"Source Column": c, "Target Name": c} for c in selected_cols]
                        renaming_df = pd.DataFrame(renaming_data)

                        mapping_df = st.data_editor(
                            renaming_df,
                            column_config={
                                "Source Column": st.column_config.TextColumn("Source", disabled=True),
                                "Target Name": st.column_config.TextColumn("Target Name (Edit here)")
                            },
                            hide_index=True,
                            key=f"{key_prefix}_renaming_editor",
                            width="stretch"
                        )

                        # Apply selection and renaming
                        column_map = dict(zip(mapping_df["Source Column"], mapping_df["Target Name"]))
                        df_to_merge = df_parsed[selected_cols].rename(columns=column_map)

                        st.write("#### 5. Final Data Preview")
                        st.dataframe(df_to_merge.head(), width="stretch")

                        # 6. Merge Logic
                        if 'preview_meta' not in st.session_state or not st.session_state.preview_meta:
                            st.warning("No file preview generated yet to merge with. Please scan files first.")
                        else:
                            df_preview = pd.DataFrame(st.session_state.preview_meta)

                            st.write("#### 6. Merge Options")
                            # Removed "Merge by Order" option as per user request
                            st.info("Select one or more columns from both datasets to join on. The order matters!")
                            col1, col2 = st.columns(2)

                            # Add option to use Row Index as a join key
                            file_opts = ["<Row Index>"] + list(df_preview.columns)
                            upload_opts = ["<Row Index>"] + list(df_to_merge.columns)

                            key_file = col1.multiselect("File Keys (in order)", file_opts, key=f"{key_prefix}_key_file")
                            key_upload = col2.multiselect("Upload Keys (matching file keys)", upload_opts, key=f"{key_prefix}_key_upload")

                            if st.button("Apply Merge by Key", key=f"{key_prefix}_btn_key"):
                                if not key_file or not key_upload:
                                    st.error("Please select at least one key column from both sides.")
                                elif len(key_file) != len(key_upload):
                                    st.error(f"Number of keys must match! File: {len(key_file)}, Upload: {len(key_upload)}")
                                else:
                                    try:
                                        # Prepare datasets for index matching if needed
                                        df_preview_merge = df_preview.copy()
                                        df_upload_merge = df_to_merge.copy()

                                        final_left_on = []
                                        final_right_on = []
                                        
                                        # Create composite keys for validation
                                        left_composite = pd.Series([""] * len(df_preview_merge))
                                        right_composite = pd.Series([""] * len(df_upload_merge))

                                        # Process Left Keys
                                        for k in key_file:
                                            if k == "<Row Index>":
                                                col_name = "__row_index_temp__"
                                                df_preview_merge[col_name] = df_preview_merge.index
                                                final_left_on.append(col_name)
                                                val = df_preview_merge.index.astype(str)
                                            else:
                                                col_name = f"__norm_key_{k}__"
                                                df_preview_merge[col_name] = df_preview_merge[k].astype(str).str.upper()
                                                final_left_on.append(col_name)
                                                val = df_preview_merge[col_name]
                                            left_composite = left_composite + val + "|"

                                        # Process Right Keys
                                        for k in key_upload:
                                            if k == "<Row Index>":
                                                col_name = "__row_index_temp__"
                                                df_upload_merge[col_name] = df_upload_merge.index
                                                final_right_on.append(col_name)
                                                val = df_upload_merge.index.astype(str)
                                            else:
                                                col_name = f"__norm_key_{k}__"
                                                df_upload_merge[col_name] = df_upload_merge[k].astype(str).str.upper()
                                                final_right_on.append(col_name)
                                                val = df_upload_merge[col_name]
                                            right_composite = right_composite + val + "|"
                                            
                                        # VALIDATION: Check coverage
                                        left_set = set(left_composite)
                                        right_set = set(right_composite)
                                        
                                        missing_in_upload = left_set - right_set
                                        extra_in_upload = right_set - left_set

                                        if missing_in_upload:
                                            st.warning(f"{len(missing_in_upload)} files (Left) do not have matching metadata in the uploaded file (Right). They will have empty values for new columns.")
                                            with st.expander("Show Missing Keys"):
                                                st.write(list(missing_in_upload))
                                        
                                        if extra_in_upload:
                                            st.info(f"{len(extra_in_upload)} metadata entries (Right) were not used (no matching file).")

                                        merged = df_preview_merge.merge(df_upload_merge, left_on=final_left_on, right_on=final_right_on, how='left', suffixes=('', '_upload'))

                                        # Clean up temp keys
                                        temp_cols = [c for c in merged.columns if c.startswith("__norm_key_") or c == "__row_index_temp__"]
                                        if temp_cols:
                                            merged.drop(columns=temp_cols, inplace=True)

                                        # Resolve suffixes
                                        for col in df_to_merge.columns:
                                            if f"{col}_upload" in merged.columns:
                                                merged[col] = merged[f"{col}_upload"].combine_first(merged[col])
                                                merged.drop(columns=[f"{col}_upload"], inplace=True)

                                        # Post-merge cleanup: Unify common columns (Space vs Underscore)
                                        # This prevents duplicate columns like "Experiment Name" and "Experiment_Name"
                                        unify_pairs = [("Experiment Name", "Experiment_Name"), ("Cell Line", "Cell_Line")]
                                        for source, target in unify_pairs:
                                            if source in merged.columns and target in merged.columns:
                                                # Treat empty strings as NaN for combination
                                                merged[target] = merged[target].replace("", np.nan)
                                                merged[source] = merged[source].replace("", np.nan)
                                                
                                                # Merge source into target (source values fill target gaps)
                                                merged[target] = merged[target].combine_first(merged[source])
                                                
                                                # Fill NaN back to "" if that was the convention, or let downstream handle it
                                                merged[target] = merged[target].fillna("")
                                                
                                                # Drop source column
                                                merged.drop(columns=[source], inplace=True)

                                        st.session_state.preview_meta = merged.to_dict('records')

                                        # Auto-select merged columns in the UI
                                        target_key = f"{key_prefix}_cols_selected"
                                        # Ensure the key exists (it should if the UI was rendered)
                                        if target_key in st.session_state:
                                             current_selected = st.session_state[target_key]
                                             # Add columns from the uploaded file to selection
                                             for col in df_to_merge.columns:
                                                  # Determine final column name in 'merged'
                                                  final_col = col
                                                  # Check unification overrides
                                                  if col == "Experiment Name" and "Experiment_Name" in merged.columns:
                                                       final_col = "Experiment_Name"
                                                  elif col == "Cell Line" and "Cell_Line" in merged.columns:
                                                       final_col = "Cell_Line"
                                                  
                                                  if final_col in merged.columns and final_col not in current_selected:
                                                       current_selected.append(final_col)
                                             
                                             st.session_state[target_key] = current_selected

                                        for k in reset_keys:
                                            if k in st.session_state:
                                                del st.session_state[k]
                                        st.success("Merged by key! Metadata updated.")
                                        st.rerun()
                                    except Exception as merge_err:
                                        st.error(f"Merge failed: {merge_err}")
                else:
                    st.error("Header row index out of bounds.")

            except Exception as e:
                st.error(f"Error reading/merging file: {e}")
                # Print traceback to terminal for debugging
                import traceback
                traceback.print_exc()
