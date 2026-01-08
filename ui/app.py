import streamlit as st
import os
import sys
import pandas as pd
import numpy as np

# Add parent directory to path to import pipeline modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from base_pipeline.data_loader import DataLoader
    from base_pipeline.gating import GatingManager
    import cytoflow
    CYTOFLOW_AVAILABLE = True
except ImportError as e:
    st.error(f"Cytoflow or pipeline modules not found: {e}. Running in Mock Mode.")
    CYTOFLOW_AVAILABLE = False

st.title("Glow Flow Analysis Pipeline")

st.sidebar.header("Configuration")

# Renaming Preference
rename_strategy = st.sidebar.radio("Renaming Strategy", ('folder_name', 'keep'))

# File Input
st.subheader("1. File Loading")
upload_option = st.radio("Choose Input Method", ("Enter Directory Path", "Upload File Paths List", "Upload FCS Files directly (Demo)"))

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
            # Shared logic with File Upload -> Refractor ideally, but mirroring here for now
             # Step 1: Generate Preview
            if 'preview_meta' not in st.session_state:
                 st.session_state.preview_meta = []

            if st.button("Generate Preview from Directory"):
                 st.session_state.preview_meta = loader.generate_preview_metadata(st.session_state.scanned_paths, rename_strategy=rename_strategy)
            
            if st.session_state.preview_meta:
                 st.subheader("Edit Metadata Structure")
                 
                 # 1. Define fields user wants
                 default_cols = ["Replicate", "Cell_Line", "Experiment_Name", "Timepoint", "Concentration"]
                 st.write("Current Metadata Fields:")
                 cols_to_use = st.multiselect("Select columns to annotate:", 
                                              options=default_cols + ["Custom1", "Custom2"],
                                              default=["Replicate", "Cell_Line", "Experiment_Name"])
                 
                 # 2. Prepare DataFrame
                 df_preview = pd.DataFrame(st.session_state.preview_meta)
                 
                 # Ensure selected columns exist
                 for col in cols_to_use:
                     if col not in df_preview.columns:
                         df_preview[col] = "" # Default to empty string
                 
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
                 
                 # 1. Define fields
                 # User requested: "Treatment" -> "Experiment_Name"
                 default_cols = ["Replicate", "Cell_Line", "Experiment_Name"]
                 cols_to_use = st.multiselect("Select columns:", 
                                              options=default_cols + ["Timepoint", "Concentration", "Treatment"], # Kept Treatment just in case
                                              default=default_cols, key="txt_cols")
                                              
                 df_preview = pd.DataFrame(st.session_state.preview_meta)
                 
                 for col in cols_to_use:
                     if col not in df_preview.columns:
                         # Default value logic
                         if col == "Replicate":
                             df_preview[col] = "1"
                         else:
                             df_preview[col] = "" 
                 
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

# Access loaded files from session state
if 'loaded_files' in st.session_state:
    loaded_files = st.session_state.loaded_files

# Gating Section showing the base functionality
st.subheader("2. Auto Gating & Selection")

if CYTOFLOW_AVAILABLE and 'loaded_files' in st.session_state and st.session_state.loaded_files:
     st.write("Configuring Gating...")
     
     # Load Base Experiment (Clean state)
     if 'base_experiment' not in st.session_state:
         with st.spinner("Loading Experiment Data..."):
             loader = DataLoader()
             st.session_state.base_experiment = loader.load_as_cytoflow_experiment(st.session_state.loaded_files)
     
     # We work with base_experiment for channel selection
     base_exp = st.session_state.base_experiment

     if base_exp:
         # Channel Selection
         st.write("Select Channels for Live/Dead Gating:")
         channels = base_exp.channels
         
         # Try to be smart about defaults
         default_x = "FSC 488/10-A" if "FSC 488/10-A" in channels else channels[0]
         default_y = "SSC 488/10-A" if "SSC 488/10-A" in channels else channels[1] if len(channels) > 1 else channels[0]

         x_channel = st.selectbox("X Channel (Forward Scatter)", options=channels, index=channels.index(default_x) if default_x in channels else 0)
         y_channel = st.selectbox("Y Channel (Side Scatter)", options=channels, index=channels.index(default_y) if default_y in channels else 0)

         if st.button("Run Auto-Gating"):
             gater = GatingManager()
             with st.spinner("Applying GMM Gating..."):
                 # ALWAYS apply to the CLEAN base_experiment to avoid column name collisions on re-runs
                 # Cytoflow operations return a NEW experiment object, they don't modify inplace.
                 gated_experiment = gater.auto_gate_live_cells(base_exp, x_channel=x_channel, y_channel=y_channel)
                 
                 st.success("Auto Gating Complete!")
                 
                 # Save the RESULTING experiment separately
                 st.session_state.gated_experiment = gated_experiment
         
         # If we have a gated experiment, show it (persisting the view)
         if 'gated_experiment' in st.session_state:
             st.divider()
             st.subheader("3. Verification & Active Learning")
             
             exp = st.session_state.gated_experiment
             df = exp.data.copy() # Safe copy
             
             import plotly.express as px
             import plotly.graph_objects as go
             import numpy as np
             from scipy.spatial import ConvexHull
             from scipy.stats import gaussian_kde

             # Scale Toggle
             st.write("### Gating Visualization")
             viz_col1, _ = st.columns([1, 4])
             with viz_col1:
                 scale_opt = st.radio("Scale", ["Linear", "Log"], index=1, horizontal=True)
                 is_log = (scale_opt == "Log")

             # Color mode is always Density (Heatmap) per user request
             
             try:
                # Prepare Coloring Logic - CONSTANT DENSITY
                color_col = None
                color_map = None
                continuous_color = False
                
                # --- Density Calculation (Fast 2D Histogram Method) ---
                try:
                    # Use log data for density calculation if log scale is selected
                    x_data = np.log10(df[x_channel]) if is_log else df[x_channel]
                    y_data = np.log10(df[y_channel]) if is_log else df[y_channel]
                    
                    # Remove infs/nans for calculation
                    mask = np.isfinite(x_data) & np.isfinite(y_data)
                    x_clean, y_clean = x_data[mask], y_data[mask]
                    
                    # Fast density via histogram
                    # 100x100 bins
                    H, xedges, yedges = np.histogram2d(x_clean, y_clean, bins=100)
                    
                    # Map density back to points
                    # Digitizing data points to find which bin they fall into
                    x_bins = np.digitize(x_clean, xedges) - 1
                    y_bins = np.digitize(y_clean, yedges) - 1
                    
                    # Clip to valid range
                    x_bins = np.clip(x_bins, 0, H.shape[0]-1)
                    y_bins = np.clip(y_bins, 0, H.shape[1]-1)
                    
                    # Assign density values
                    # We initialize a density column with min value
                    df['Density'] = 0.0
                    df.loc[mask, 'Density'] = H[x_bins, y_bins]
                    
                    color_col = 'Density'
                    continuous_color = True
                except Exception as e:
                    # st.warning(f"Density calculation failed: {e}. Falling back to solid color.") # suppress warning for cleanliness
                    color_col = None


                # --- Create Plot ---
                # DOWNSAMPLE for Performance & Visibility (SVG rendering)
                # Plotting massive points in SVG is slow, but fixes z-order issues.
                # We limit to 5000 points for the interactive plot.
                plot_df = df
                if len(plot_df) > 5000:
                    plot_df = plot_df.sample(n=5000, random_state=42)
                    st.caption(f"Plotting 5,000 representative events (out of {len(df)}) for smoother interaction.")
                
                # To match the user image: Viridis, small dots
                # REMOVED render_mode='webgl' to ensure Overlay traces appear ON TOP of scatter points
                fig = px.scatter(plot_df, x=x_channel, y=y_channel, color=color_col, 
                                title="Gating Result",
                                color_discrete_map=color_map,
                                color_continuous_scale='Viridis' if continuous_color else None,
                                opacity=0.8 if continuous_color else 0.6, 
                                log_x=is_log, log_y=is_log)
                
                # Improve marker style for density look
                if continuous_color:
                    fig.update_traces(marker=dict(size=6)) # Increased size for easier clicking
                    fig.update_layout(coloraxis_colorbar=dict(title="Count"))

                # Enable Click Selection by default (Shift+Click for multi)
                # dragmode='pan' disables the "rectangle" box select, so clicks are registered as point selections.
                fig.update_layout(clickmode='event+select', dragmode='pan')
                
                # Note: dragmode='lasso' removed per user request for "point selection"
                
                # --- Auto-Gating Visualization (Centroids & Ellipses) ---
                # ALWAYS overlay Auto-Gating if available
                if 'AutoLiveGate' in df.columns:
                    unique_labels = df['AutoLiveGate'].unique()
                    
                    # Heuristic: Identify the "Live" population
                    # The population with the higher Mean FSC (x_channel) is usually the cells, not debris.
                    # We calculate means for all available labels first
                    label_means = {}
                    for label in unique_labels:
                         subset = df[df['AutoLiveGate'] == label]
                         if not subset.empty and len(subset) > 5:
                             label_means[label] = subset[x_channel].mean()
                    
                    # Sort labels by mean X descending
                    if label_means:
                        sorted_labels = sorted(label_means.items(), key=lambda item: item[1], reverse=True)
                        # The "Live" gate is likely the first one (highest FSC)
                        live_label = sorted_labels[0][0]
                        
                        # Store for the "Accept" button
                        st.session_state['suggested_live_label'] = live_label

                        # Only plot this one
                        subset = df[df['AutoLiveGate'] == live_label]
                        label = live_label # for variable consistency

                        # Calculate Mean (Centroid) and Ellipse
                        # Handle Log-Space Geometry if needed
                        
                        center_x, center_y = None, None
                        ell_x, ell_y = None, None
                        
                        try:
                            if is_log:
                                # Perform calculations in LOG space
                                # Filter <= 0 for log
                                valid_mask = (subset[x_channel] > 0) & (subset[y_channel] > 0)
                                s_valid = subset.loc[valid_mask]
                                
                                if len(s_valid) > 5:
                                    log_x = np.log10(s_valid[x_channel])
                                    log_y = np.log10(s_valid[y_channel])
                                    
                                    # Geometric Mean (Mean in log space)
                                    l_mean_x = log_x.mean()
                                    l_mean_y = log_y.mean()
                                    
                                    # Covariance in log space
                                    cov = np.cov(log_x, log_y)
                                    lambda_, v = np.linalg.eig(cov)
                                    lambda_ = np.sqrt(lambda_)
                                    
                                    # USER REQ: "five points with lines in between"
                                    # We discretize the ellipse into a pentagon (5 points + closing point)
                                    theta = np.linspace(0, 2*np.pi, 6) 
                                    
                                    # Ellipse in log space
                                    l_ell_x = l_mean_x + 2 * lambda_[0] * v[0, 0] * np.cos(theta) + 2 * lambda_[1] * v[0, 1] * np.sin(theta)
                                    l_ell_y = l_mean_y + 2 * lambda_[0] * v[1, 0] * np.cos(theta) + 2 * lambda_[1] * v[1, 1] * np.sin(theta)
                                    
                                    # Transform back to linear for plotting
                                    center_x = 10**l_mean_x
                                    center_y = 10**l_mean_y
                                    ell_x = 10**l_ell_x
                                    ell_y = 10**l_ell_y
                            else:
                                # Linear space
                                center_x = subset[x_channel].mean()
                                center_y = subset[y_channel].mean()
                                
                                cov = np.cov(subset[x_channel], subset[y_channel])
                                lambda_, v = np.linalg.eig(cov)
                                lambda_ = np.sqrt(lambda_)
                                
                                # USER REQ: 5 points
                                theta = np.linspace(0, 2*np.pi, 6)
                                ell_x = center_x + 2 * lambda_[0] * v[0, 0] * np.cos(theta) + 2 * lambda_[1] * v[0, 1] * np.sin(theta)
                                ell_y = center_y + 2 * lambda_[0] * v[1, 0] * np.cos(theta) + 2 * lambda_[1] * v[1, 1] * np.sin(theta)

                            if center_x is not None:
                                # Add Centroid ("Big Dot")
                                fig.add_trace(go.Scatter(
                                    x=[center_x], y=[center_y], mode='markers',
                                    marker=dict(color='red', size=14, line=dict(color='white', width=2)), # Red for visibility
                                    name=f"Auto Gate Center"
                                ))
                            
                            if ell_x is not None:
                                fig.add_trace(go.Scatter(x=ell_x, y=ell_y, mode='lines+markers', 
                                                        line=dict(color='white', width=4, dash='solid'), # Very Thick White Line
                                                        marker=dict(size=8, color='white'), # Show vertices
                                                        name=f"Auto Gate"))
                        except Exception:
                            pass # geometry calc fail

                # --- Manual Gate Visualization ---
                if st.session_state.get('manual_vertices'):
                    verts = st.session_state.manual_vertices
                    path_x = [v[0] for v in verts] + [verts[0][0]]
                    path_y = [v[1] for v in verts] + [verts[0][1]]
                    
                    fig.add_trace(go.Scatter(x=path_x, y=path_y, mode="lines+markers", 
                                             line=dict(color="yellow", width=4), # Thicker for visibility over density
                                             marker=dict(size=6, color="yellow"),
                                             name="Manual Gate"))
                
                # --- Pending Gate Visualization (Immediate Feedback) ---
                if st.session_state.get('pending_manual_vertices'):
                    p_verts = st.session_state.pending_manual_vertices
                    p_path_x = [v[0] for v in p_verts] + [p_verts[0][0]]
                    p_path_y = [v[1] for v in p_verts] + [p_verts[0][1]]
                    
                    fig.add_trace(go.Scatter(x=p_path_x, y=p_path_y, mode="lines+markers", 
                                             line=dict(color="cyan", width=4, dash='dash'), 
                                             marker=dict(size=8, color="cyan"),
                                             name="New Selection (Pending)"))

                # Interactive Plot
                # Legend layout adjustment to prevent overlapping
                fig.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ))
                
                # Render Plot
                # Use selection_mode='points' for clicking individual cells
                selection = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points")
                
                # Check for NEW selection
                if selection and selection.get("selection"):
                     pts = selection["selection"]["points"]
                     count = len(pts)
                     
                     if count > 0:
                         # Extract new points
                         new_x = [p["x"] for p in pts]
                         new_y = [p["y"] for p in pts]
                         new_candidates = list(zip(new_x, new_y))
                         
                         # Get current pending points
                         current_pending = st.session_state.get('pending_manual_vertices', [])
                         updated_pending = list(current_pending)
                         
                         has_changes = False
                         for cand in new_candidates:
                             # Check if close to existing
                             is_close = False
                             for ex in current_pending:
                                  if np.isclose(cand[0], ex[0], atol=1e-5) and np.isclose(cand[1], ex[1], atol=1e-5):
                                      is_close = True
                                      break
                             if not is_close:
                                 updated_pending.append(cand)
                                 has_changes = True
                                 
                         # If we have 5 or more points, sort them to make a polygon
                         final_pending = updated_pending
                         
                         # User wanted 5 points generally. Let's sort if >=3
                         if len(updated_pending) >= 3:
                            try:
                                arr = np.array(updated_pending)
                                cx, cy = np.mean(arr[:,0]), np.mean(arr[:,1])
                                angles = np.arctan2(arr[:,1] - cy, arr[:,0] - cx)
                                indices = np.argsort(angles)
                                final_pending = [updated_pending[i] for i in indices]
                            except:
                                pass

                         if has_changes:
                             st.session_state.pending_manual_vertices = final_pending
                             # Update Text Area State
                             st.session_state.manual_live_verts_input = "\n".join([f"{v[0]}, {v[1]}" for v in final_pending])
                             st.rerun()

                # Confirmation UI
                if st.session_state.get('pending_manual_vertices'):
                    st.divider()
                    st.info(f"Selected {len(st.session_state.pending_manual_vertices)} points. Confirm to create gate?")
                    conf_col1, conf_col2 = st.columns(2)
                    if conf_col1.button("✅ Confirm New Gate"):
                        st.session_state.manual_vertices = st.session_state.pending_manual_vertices
                        del st.session_state.pending_manual_vertices
                        st.rerun()
                    
                    if conf_col2.button("❌ Discard Selection"):
                        del st.session_state.pending_manual_vertices
                        st.rerun()

                # Helper for logging
                # Check for module availability (in case created dynamically)
                try:
                    from active_learning.storage import ActiveLearningLogger
                    logger = ActiveLearningLogger()
                except ImportError:
                    logger = None
                    # st.warning("Active Learning module not found.") # Optional

                # Feedback Controls (Only show if no pending selection to avoid clutter)
                if not st.session_state.get('pending_manual_vertices'):
                    st.divider()
                    col1, col2 = st.columns(2)
                    
                    if col1.button("✅ Accept Auto-Gate"):
                        if logger:
                            # Log the acceptance
                            # We use a batch name since 'Sample' might be row-wise. 
                            # We log the configuration.
                            
                            # Determine Cell Line for logging
                            current_cell_line = "Unknown"
                            if st.session_state.get('loaded_files'):
                                lines = set()
                                for f in st.session_state.loaded_files:
                                    cl = f.get('Cell_Line') or f.get('Cell Line') or 'Unknown'
                                    lines.add(cl)
                                
                                if len(lines) == 1:
                                    current_cell_line = list(lines)[0]
                                else:
                                    current_cell_line = ",".join(list(lines))
                            
                            logger.log_gate_feedback(
                                sample_name="Batch_Experiment", 
                                cell_line=current_cell_line,
                                x_channel=x_channel, 
                                y_channel=y_channel, 
                                gate_type="GMM", 
                                gate_params="Auto", 
                                user_decision="accepted"
                            )
                        
                        # Apply the Auto-Gate (Filter the experiment)
                        if 'suggested_live_label' in st.session_state and CYTOFLOW_AVAILABLE:
                            label = st.session_state.suggested_live_label
                            exp = st.session_state.gated_experiment
                            
                            # Filter to keep only the Live Label population
                            # We modify the experiment in-place for the session
                            filtered_df = exp.data[exp.data['AutoLiveGate'] == label].copy()
                            exp.data = filtered_df
                            
                            st.session_state.gated_experiment = exp
                            st.session_state.gate_accepted = True
                            st.success("Auto-Gate Accepted! Data filtered to Live Cells.")
                            st.rerun()
                        else:
                            st.success("Feedback saved! (Mock mode)")
                        
                    if col2.button("❌ Reject / Manual Gate"):
                        st.session_state.show_manual_gating = True
                
                # Check for Gate Acceptance to show clean-up options
                if st.session_state.get('gate_accepted'):
                    st.success("Gate Applied.")
                
                # Manual Gating Interface
                if st.session_state.get('show_manual_gating'):
                    st.info("Manual Selection Mode: CLICK roughly 5 cells on the boundary of your population to define the gate (Hold Shift to Multi-Select).")
                    
                    # Convert current vertices to string for the text area
                    current_poly_str = ""
                    if st.session_state.get('manual_vertices'):
                        current_poly_str = "\n".join([f"{v[0]}, {v[1]}" for v in st.session_state.manual_vertices])
                    else:
                        current_poly_str = f"10000, 10000\n100000, 10000\n100000, 100000\n10000, 100000"
                        
                    vertices_text = st.text_area("Polygon Vertices (x, y per line)", current_poly_str, height=150, key="manual_live_verts_input")
                    
                    if st.button("Apply Manual Gate"):
                        # Parse
                        try:
                            vertices = []
                            for line in vertices_text.strip().split('\n'):
                                p = line.split(',')
                                vertices.append((float(p[0].strip()), float(p[1].strip())))
                            
                            gater = GatingManager()
                            # Apply to BASE experiment
                            manual_exp = gater.apply_manual_gate(base_exp, "ManualLiveGate", x_channel, y_channel, vertices)
                            st.session_state.gated_experiment = manual_exp
                            st.session_state.manual_vertices = vertices # Save for visualization
                            
                            # Determine Cell Line for logging
                            current_cell_line = "Unknown"
                            if st.session_state.get('loaded_files'):
                                lines = set()
                                for f in st.session_state.loaded_files:
                                    cl = f.get('Cell_Line') or f.get('Cell Line') or 'Unknown'
                                    lines.add(cl)
                                
                                if len(lines) == 1:
                                    current_cell_line = list(lines)[0]
                                else:
                                    current_cell_line = ",".join(list(lines))
                            
                            if logger:
                                logger.log_gate_feedback(
                                    sample_name="Batch_Experiment", 
                                    cell_line=current_cell_line,
                                    x_channel=x_channel, 
                                    y_channel=y_channel, 
                                    gate_type="Polygon", 
                                    gate_params=vertices, 
                                    user_decision="manual_override"
                                )
                            
                            st.session_state.gate_accepted = True
                            st.rerun()
                                
                        except Exception as e:
                            st.error(f"Invalid format: {e}")
                
                # --- Module 4: Cleanup & Export (Post-Gating) ---
                if st.session_state.get('gate_accepted'): 
                    st.divider()
                    st.header("4. Post-Gating Processing")
                    st.info("Gated population selected. Refine or export.")
                    
                    # Export Data (Always available)
                    csv = st.session_state.gated_experiment.data.to_csv(index=False).encode('utf-8')
                    
                    col_exp1, col_exp2 = st.columns([1, 1])
                    with col_exp1:
                         st.download_button("Download Gated CSV", data=csv, file_name="gated_live_cells.csv", mime="text/csv")
                    
                    st.divider()
                    st.subheader("4.1 Doublet Discrimination (Singlet Selection)")
                    
                    # Toggle for Doublet Gating
                    is_doublet_mode = st.checkbox("Show Doublet Check (FSC-A vs FSC-H)", value=st.session_state.get('show_doublet_gating', False))
                    if is_doublet_mode:
                        st.session_state.show_doublet_gating = True
                        
                        exp = st.session_state.gated_experiment
                        df_gated = exp.data
                        d_channels = exp.channels
                        
                        # Auto-guess channels
                        idx_a, idx_h = 0, 0
                        for i, c in enumerate(d_channels):
                            if "FSC" in c and ("-A" in c or "Area" in c): idx_a = i
                            if "FSC" in c and ("-H" in c or "Height" in c): idx_h = i
                        
                        dc1, dc2, dc3 = st.columns(3)
                        d_x = dc1.selectbox("Area Channel (X)", d_channels, index=idx_a, key="dA")
                        d_y = dc2.selectbox("Height Channel (Y)", d_channels, index=idx_h, key="dH")
                        d_scale = dc3.radio("Scale", ["Linear", "Log"], index=0, key="dScale") # Default Linear
                        d_is_log = (d_scale == "Log")

                        # --- Auto-Gate Calculation (Singlets - GMM on Linear Data) ---
                        try:
                            from sklearn.mixture import GaussianMixture
                            
                            # Always calculate on Linear for Singlets (Physical relationship is linear)
                            X_lin = df_gated[[d_x, d_y]].values
                            
                            # Fit GMM
                            gmm_lin = GaussianMixture(n_components=1, random_state=42)
                            gmm_lin.fit(X_lin)
                            
                            d_mean = gmm_lin.means_[0]
                            d_cov = gmm_lin.covariances_[0]
                            
                            # Generate Ellipse (2.5 Sigma)
                            lambda_d, v_d = np.linalg.eig(d_cov)
                            lambda_d = np.sqrt(lambda_d)
                            theta_d = np.linspace(0, 2*np.pi, 100)
                            ell_x = d_mean[0] + 2.5 * lambda_d[0] * v_d[0, 0] * np.cos(theta_d) + 2.5 * lambda_d[1] * v_d[0, 1] * np.sin(theta_d)
                            ell_y = d_mean[1] + 2.5 * lambda_d[0] * v_d[1, 0] * np.cos(theta_d) + 2.5 * lambda_d[1] * v_d[1, 1] * np.sin(theta_d)
                            
                            # --- Visualization ---
                            fig_d = go.Figure()
                            
                            # Use ScatterGL for overlaid points if manual, else Density
                            # User Req: "you also need density plot"
                            # We use Density Heatmap
                            
                            # Define viz_df (Downsampled)
                            subset_n = 5000
                            # FIX 1: Use random_state for stable plotting across reruns
                            viz_df = df_gated.sample(subset_n, random_state=42) if len(df_gated) > subset_n else df_gated

                            # For Density, we need 1D arrays
                            if d_is_log:
                                # Safe log
                                x_to_plot = np.log10(viz_df[d_x] + 1)
                                y_to_plot = np.log10(viz_df[d_y] + 1)
                                ell_x_plot = np.log10(ell_x + 1) if not np.any(ell_x <= -1) else ell_x # approximation
                                ell_y_plot = np.log10(ell_y + 1) if not np.any(ell_y <= -1) else ell_y
                            else:
                                x_to_plot = viz_df[d_x]
                                y_to_plot = viz_df[d_y]
                                ell_x_plot = ell_x
                                ell_y_plot = ell_y

                            # USER REQ: "density should be each cell but colored with density it has"
                            # We use Gaussian KDE to estimate density for each point -> color by density
                            from scipy.stats import gaussian_kde
                            
                            # Stack X and Y for KDE
                            # We use the PLOT coordinates (Linear or Log transformed) for visual density
                            xy = np.vstack([x_to_plot, y_to_plot])
                            
                            try:
                                z_density = gaussian_kde(xy)(xy)
                                # Sort to plot high density on top
                                idx_sort = z_density.argsort()
                                x_to_plot = x_to_plot.iloc[idx_sort] if hasattr(x_to_plot, 'iloc') else x_to_plot[idx_sort]
                                y_to_plot = y_to_plot.iloc[idx_sort] if hasattr(y_to_plot, 'iloc') else y_to_plot[idx_sort]
                                z_density = z_density[idx_sort]
                            except Exception:
                                # Fallback if KDE fails (singular matrix)
                                z_density = 'blue'
                            
                            # Use Scatter (SVG) instead of Scattergl for reliable individual point selection
                            # Scattergl can have issues with hit-testing on some systems/browsers and resets context
                            fig_d.add_trace(go.Scatter(
                                x=x_to_plot, y=y_to_plot,
                                mode='markers',
                                marker=dict(
                                    size=4, # Slightly larger for easier clicking 
                                    color=z_density, 
                                    colorscale='Viridis', 
                                    showscale=True,
                                    colorbar=dict(title='Density')
                                ),
                                name='Events',
                                # Ensure this trace handles selection
                                selected=dict(marker=dict(color='orange', size=6)), # Visual feedback on selection
                                unselected=dict(marker=dict(opacity=0.5))
                            ))
                            
                            # Auto-Gate Ellipse -> Converted to 4-Point Polygon (Quadrilateral)
                            # User Req: "auto selection for singlets should be 4 points connected by lines"
                            theta_poly = np.linspace(0, 2*np.pi, 5) # 5 points to close the loop (4 sides)
                            ell_x_poly = d_mean[0] + 2.5 * lambda_d[0] * v_d[0, 0] * np.cos(theta_poly) + 2.5 * lambda_d[1] * v_d[0, 1] * np.sin(theta_poly)
                            ell_y_poly = d_mean[1] + 2.5 * lambda_d[0] * v_d[1, 0] * np.cos(theta_poly) + 2.5 * lambda_d[1] * v_d[1, 1] * np.sin(theta_poly)
                            
                            ell_x_plot = np.log10(ell_x_poly + 1) if d_is_log else ell_x_poly
                            ell_y_plot = np.log10(ell_y_poly + 1) if d_is_log else ell_y_poly
                            
                            # Draw the Auto-Gate
                            fig_d.add_trace(go.Scatter(x=ell_x_plot, y=ell_y_plot, mode='lines+markers', 
                                                     line=dict(color='red', width=2), 
                                                     marker=dict(size=6, color='red'),
                                                     name="Auto Singlets"))
                            
                            # Manual Gate Pending
                            if st.session_state.get('pending_doublet_verts'):
                                 pv = st.session_state.pending_doublet_verts
                                 # Transform for display if needed
                                 if d_is_log:
                                     # Not implementing complex log drag yet, assume manual on linear for now or transform points
                                     # For simplicity in this iteration, let's keep manual selection logic on the raw data coordinates
                                     # but display might misalign if we transformed x_to_plot.
                                     # Let's stick to LINEAR coordinates for the axes if user selected Linear, 
                                     # if Log, we should probably plot log-log. 
                                     pass
                                 
                                 px_ = [v[0] for v in pv] + [pv[0][0]]
                                 py_ = [v[1] for v in pv] + [pv[0][1]]
                                 
                                 if d_is_log:
                                     px_ = np.log10(np.array(px_) + 1)
                                     py_ = np.log10(np.array(py_) + 1)
                                     
                                 fig_d.add_trace(go.Scatter(x=px_, y=py_, mode='lines+markers',
                                                          line=dict(color='cyan', width=3), name="Manual Selection"))
                            
                            # Interaction Mode State
                            is_manual_active = st.session_state.get('doublet_manual_mode', False)

                            # --- Calculate Auto-Zoom Range ---
                            # Focus on the GMM cluster to avoid manual zooming (User Req for start zoom)
                            zoom_margin = 4 # sigmas
                            x_min_lin = d_mean[0] - zoom_margin * np.sqrt(d_cov[0,0])
                            x_max_lin = d_mean[0] + zoom_margin * np.sqrt(d_cov[0,0])
                            y_min_lin = d_mean[1] - zoom_margin * np.sqrt(d_cov[1,1])
                            y_max_lin = d_mean[1] + zoom_margin * np.sqrt(d_cov[1,1])
                            
                            start_range_x = None
                            start_range_y = None
                            
                            if d_is_log:
                                # Avoid log of negative/zero
                                x_min_lin = max(0, x_min_lin)
                                y_min_lin = max(0, y_min_lin)
                                start_range_x = [np.log10(x_min_lin + 1), np.log10(x_max_lin + 1)]
                                start_range_y = [np.log10(y_min_lin + 1), np.log10(y_max_lin + 1)]
                            else:
                                start_range_x = [x_min_lin, x_max_lin]
                                start_range_y = [y_min_lin, y_max_lin]

                            # Axes
                            # User Req: "when i zoom in i want to keep the visual the same"
                            
                            # USER REQ: "background to be transparent"
                            
                            fig_d.update_layout(
                                title="Singlet Identification " + ("(Manual Mode)" if is_manual_active else "(Auto Mode)"), 
                                xaxis_title=d_x + (" (Log)" if d_is_log else ""), 
                                yaxis_title=d_y + (" (Log)" if d_is_log else ""),
                                dragmode='pan', # Set to 'pan' to allow easier clicking without zoom box interference
                                clickmode='event+select',
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='black'), # Ensure text is visible
                                margin=dict(l=20, r=20, t=40, b=20),
                                height=500,
                                xaxis=dict(range=start_range_x),
                                yaxis=dict(range=start_range_y),
                                uirevision='constant' # Keeps user zoom state on re-runs
                            )
                            fig_d.update_xaxes(showgrid=True, gridcolor='rgba(0,0,0,0.1)', linecolor='black', zeroline=False)
                            fig_d.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.1)', linecolor='black', zeroline=False)
                            
                            # Selection Interaction
                            # User req for Shift selection is handled by Plotly standard behavior with clickmode
                            sel_d = st.plotly_chart(fig_d, use_container_width=True, on_select="rerun", selection_mode="points", key="doublet_chart")
                            
                            # Counter for selection
                            if is_manual_active:
                                cnt = len(st.session_state.get('pending_doublet_verts', []))
                                st.markdown(f"**Selected Points:** {cnt} / 4")
                            
                            # Handle Manual Selection (Strict: Only when manual mode is ON)
                            # Implements ACCUMULATIVE selection (Server-Side) to allow creating polygon click-by-click
                            if is_manual_active and sel_d and sel_d.get("selection"):
                                 pts = sel_d["selection"]["points"]
                                 if len(pts) > 0:
                                      # Coords form Plotly are in Plot Space. 
                                      sx = np.array([p["x"] for p in pts])
                                      sy = np.array([p["y"] for p in pts])
                                      
                                      if d_is_log:
                                          # Inverse log transform if plot was log
                                          sx = 10**sx - 1
                                          sy = 10**sy - 1
                                      
                                      # Candidate points
                                      candidates = list(zip(sx, sy))
                                      
                                      # Retrieve current persistent points
                                      current_verts = st.session_state.get('pending_doublet_verts', [])
                                      updated_verts = list(current_verts) # Copy
                                      
                                      # Add ONLY new points (Deduplication)
                                      changed = False
                                      for c in candidates:
                                          is_close = False
                                          for ex in current_verts:
                                               # Check proximity (tolerant to float diffs)
                                               if np.isclose(c[0], ex[0], atol=1e-5) and np.isclose(c[1], ex[1], atol=1e-5):
                                                   is_close = True
                                                   break
                                          if not is_close:
                                              updated_verts.append(c)
                                              changed = True
                                        
                                      # Sort radially to ensure valid polygon (avoid bowtie) if >=3
                                      final_verts = updated_verts # Default
                                      if len(updated_verts) >= 3:
                                          arr_u = np.array(updated_verts)
                                          cx, cy = np.mean(arr_u[:,0]), np.mean(arr_u[:,1])
                                          angs = np.arctan2(arr_u[:,1]-cy, arr_u[:,0]-cx)
                                          idxs = np.argsort(angs)
                                          final_verts = [updated_verts[i] for i in idxs]
                                      
                                      # Update state if changed
                                      if changed:
                                          st.session_state.pending_doublet_verts = final_verts
                                          # SYNC TEXT AREA: Update the key storing the text area value
                                          st.session_state.doublet_verts_input = "\n".join([f"{v[0]}, {v[1]}" for v in final_verts])
                                          st.rerun()

                            # Actions
                            st.divider() # Separate controls
                            col_da, col_dr = st.columns(2)
                            
                            # Accept Auto
                            # Displayed prominently like Live/Dead gate
                            if col_da.button("✅ Accept Auto-Singlets", key="accept_singlets"):
                                if logger:
                                    # Determine Cell Line for logging
                                    current_cell_line = "Unknown"
                                    if st.session_state.get('loaded_files'):
                                        lines = set()
                                        for f in st.session_state.loaded_files:
                                            cl = f.get('Cell_Line') or f.get('Cell Line') or 'Unknown'
                                            lines.add(cl)
                                        if len(lines) == 1:
                                            current_cell_line = list(lines)[0]
                                        else:
                                            current_cell_line = ",".join(list(lines))

                                    logger.log_gate_feedback(
                                        sample_name="Batch_Experiment",
                                        cell_line=current_cell_line, 
                                        x_channel=d_x, 
                                        y_channel=d_y, 
                                        gate_type="GMM_Singlet", 
                                        gate_params="Auto", 
                                        user_decision="accepted"
                                    )

                                # Mahalanobis distance filter using the Auto-GMM
                                inv_cov = np.linalg.inv(d_cov)
                                diff = X_lin - d_mean
                                dist = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
                                mask = dist <= 2.5 # Sigma threshold
                                
                                exp.data = exp.data[mask]
                                st.session_state.gated_experiment = exp
                                st.session_state.doublets_removed = True
                                st.session_state.show_doublet_gating = False
                                st.success(f"Kept {sum(mask)} singlet events.")
                                st.rerun()

                            # Reject -> Enable Manual
                            if col_dr.button("❌ Reject / Manual Gate", key="reject_singlets"):
                                st.session_state.doublet_manual_mode = True
                                st.rerun()
                            
                            # Manual Gating Interface (Shown BELOW buttons if active)
                            if is_manual_active:
                                st.info("Manual Mode: Click **4 points** to define the gate. (Points are accumulated)")
                                
                                # Convert current vertices to string for the text area (Similar to Live/Dead gate)
                                current_poly_str_d = ""
                                if st.session_state.get('pending_doublet_verts'):
                                    current_poly_str_d = "\n".join([f"{v[0]}, {v[1]}" for v in st.session_state.pending_doublet_verts])
                                
                                vertices_text_d = st.text_area("Polygon Vertices (x, y per line)", current_poly_str_d, height=150, key="doublet_verts_input")
                                
                                col_m1, col_m2, col_m3 = st.columns(3)
                                
                                # Manual Apply Button
                                # Enable confirm if text area has content (allows manual editing)
                                # USER REQ: "only see the manual gate [button] after selection"
                                has_valid_gate = len(st.session_state.get('pending_doublet_verts', [])) >= 3 or (vertices_text_d.strip() != "")
                                
                                if has_valid_gate:
                                    if col_m1.button("✅ Confirm Manual Gate", key="confirm_manual_singlets"):

                                        parsed_verts = []
                                        try:
                                            # Parse the text area content
                                            lines = vertices_text_d.strip().split('\n')
                                            for line in lines:
                                                parts = line.split(',')
                                                if len(parts) >= 2:
                                                    parsed_verts.append((float(parts[0].strip()), float(parts[1].strip())))
                                            
                                            if len(parsed_verts) < 3:
                                                st.error("Gate must have at least 3 points.")
                                            else:
                                                # Log separation
                                                if logger:
                                                    # Determine Cell Line for logging
                                                    current_cell_line = "Unknown"
                                                    if st.session_state.get('loaded_files'):
                                                        lines = set()
                                                        for f in st.session_state.loaded_files:
                                                            cl = f.get('Cell_Line') or f.get('Cell Line') or 'Unknown'
                                                            lines.add(cl)
                                                        if len(lines) == 1:
                                                            current_cell_line = list(lines)[0]
                                                        else:
                                                            current_cell_line = ",".join(list(lines))

                                                    logger.log_gate_feedback(
                                                        sample_name="Batch_Experiment",
                                                        cell_line=current_cell_line, 
                                                        x_channel=d_x, 
                                                        y_channel=d_y, 
                                                        gate_type="Polygon_Singlet", 
                                                        gate_params=parsed_verts, 
                                                        user_decision="manual_override"
                                                    )

                                                from matplotlib.path import Path
                                                poly_path = Path(parsed_verts)
                                                mask = poly_path.contains_points(X_lin)
                                                
                                                exp.data = exp.data[mask]
                                                st.session_state.gated_experiment = exp
                                                st.session_state.doublets_removed = True
                                                st.session_state.show_doublet_gating = False
                                                st.session_state.doublet_manual_mode = False # Reset
                                                if 'pending_doublet_verts' in st.session_state:
                                                    del st.session_state.pending_doublet_verts
                                                st.success(f"Kept {sum(mask)} manual events.")
                                                st.rerun()

                                        except Exception as e:
                                            st.error(f"Error parsing vertices: {e}")
                                
                                # Buttons for Clear / Cancel / Undo (User Req)
                                if col_m2.button("🚫 Clear Points", key="clear_manual_singlets"):
                                     if 'pending_doublet_verts' in st.session_state:
                                            del st.session_state.pending_doublet_verts
                                     st.rerun()

                                if col_m3.button("🔙 Back to Auto / Cancel", key="cancel_manual_singlets"):
                                         st.session_state.doublet_manual_mode = False
                                         if 'pending_doublet_verts' in st.session_state:
                                            del st.session_state.pending_doublet_verts
                                         st.rerun()
                                    
                        except Exception as e:
                            st.write(e) # Debug

                    if st.session_state.get('doublets_removed'):
                         st.success("Doublets have been removed from the dataset.")

             except Exception as e:
                 st.write(e) # Debug


     else:
         st.error("Failed to load experiment.")
elif not CYTOFLOW_AVAILABLE:
    if st.button("Run Auto-Gating"):
        st.success("Mock Auto-Gating Complete!")
    else:
        st.error("Failed to create experiment.")
elif not CYTOFLOW_AVAILABLE:
    st.success("Mock Auto-Gating Complete!")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/FlowCytometry_scatter_plot.svg/1200px-FlowCytometry_scatter_plot.svg.png", caption="Mock Output", width=300)
else:
    st.warning("Please load files first.")

st.info("Active Learning Module to be integrated here.")
