import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
from .active_learning_ui import render_uncertainty_controls, get_uncertainty_trace, render_labeling_controls, render_reset_controls
from glow.canvas.functions import add_plot_to_canvas

def render_gating_section(DataLoader, GatingManager, ActiveLearningLogger, CYTOFLOW_AVAILABLE):
    # Gating Section showing the base functionality
    st.subheader("2. Auto Gating & Selection")

    if CYTOFLOW_AVAILABLE and 'loaded_files' in st.session_state and st.session_state.loaded_files:
        
        # Check if loaded from CSV (Bypass mode)
        is_csv_source = any(str(f).startswith("CSV:") for f in st.session_state.loaded_files)
        
        if is_csv_source:
            st.info("Data loaded from Processed CSV. Skipping Gating workflow.")
            
            if 'gated_experiment' in st.session_state:
                exp = st.session_state.gated_experiment
                st.write(f"Loaded Population: {len(exp.data)} events")
                
                # Show Density Plots of the Singlet Population
                if len(exp.data) > 0:
                    st.write("### Population Density")
                    # Filter for likely channel columns (numeric) excluding system flags
                    numeric_cols = exp.data.select_dtypes(include=['float', 'int']).columns.tolist()
                    exclude_cols = ['cells', 'singlets', 'Is_Live', 'Is_Singlet', 'LiveGate', 'SingletGate']
                    channels = [c for c in numeric_cols if c not in exclude_cols]
                    
                    if not channels:
                         channels = exp.data.columns.tolist() # Fallback

                    # Try to default to standard Scatter channels if available
                    default_x = "FSC 488/10-A" if "FSC 488/10-A" in channels else channels[0]
                    default_y = "SSC 488/10-A" if "SSC 488/10-A" in channels else channels[1] if len(channels) > 1 else channels[0]
                    
                    x_axis = st.selectbox("X Axis", channels, index=channels.index(default_x) if default_x in channels else 0, key="csv_x")
                    y_axis = st.selectbox("Y Axis", channels, index=channels.index(default_y) if default_y in channels else 0, key="csv_y")
                    
                    viz_col1, viz_col2, _ = st.columns([1, 1, 3])
                    with viz_col1:
                        scale_opt = st.radio("Scale", ["Linear", "Log"], index=1, horizontal=True, key="csv_scale")
                        is_log = (scale_opt == "Log")
                with viz_col1:
                    drag_tool = st.radio("Interaction", ["Pan", "Box Select"], index=0, horizontal=True, key="csv_interaction_mode")
                    drag_mode = "pan" if drag_tool == "Pan" else "select"
                    with viz_col2:
                         cmap_selected = st.selectbox("Colormap", ["Viridis", "Plasma"], index=0, key="csv_cmap")

                    # High performance plotting - Downsample for interaction speed if necessary, but density calc on full relevant set
                    df_viz = exp.data.copy()
                    
                    # --- Density Calculation (Fast 2D Histogram Method) ---
                    try:
                        color_col = None
                        
                        # Use log data for density calculation if log scale is selected
                        x_data = np.log10(df_viz[x_axis]) if is_log else df_viz[x_axis]
                        y_data = np.log10(df_viz[y_axis]) if is_log else df_viz[y_axis]
                        
                        # Remove infs/nans for calculation
                        mask = np.isfinite(x_data) & np.isfinite(y_data)
                        x_clean, y_clean = x_data[mask], y_data[mask]
                        
                        if len(x_clean) > 0:
                            # Fast density via histogram
                            # 100x100 bins
                            H, xedges, yedges = np.histogram2d(x_clean, y_clean, bins=100)
                            
                            # Map density back to points
                            x_bins = np.digitize(x_clean, xedges) - 1
                            y_bins = np.digitize(y_clean, yedges) - 1
                            
                            x_bins = np.clip(x_bins, 0, H.shape[0]-1)
                            y_bins = np.clip(y_bins, 0, H.shape[1]-1)
                            
                            # Assign density values
                            df_viz['Density'] = 0.0
                            df_viz.loc[mask, 'Density'] = H[x_bins, y_bins]
                            color_col = 'Density'
                        else:
                            color_col = None # Fallback

                    except Exception as e:
                        color_col = None

                    # Construct Plot
                    # Downsample for SVG rendering performance
                    if len(df_viz) > 5000:
                        df_plot = df_viz.sample(n=5000, random_state=42)
                        st.caption(f"Plotting 5,000 representative events (out of {len(df_viz)}) for smoother interaction.")
                    else:
                        df_plot = df_viz

                    fig = px.scatter(df_plot, x=x_axis, y=y_axis, color=color_col,
                                    title="Population Density (Singlets)",
                                    color_continuous_scale=cmap_selected,
                                    opacity=0.8,
                                    log_x=is_log, log_y=is_log,
                                    render_mode='svg')
                    
                    if color_col:
                        fig.update_traces(marker=dict(size=5))
                        fig.update_layout(coloraxis_colorbar=dict(title="Density"))

                    # Improved Plot Layout (White Background, Black Axes)
                    fig.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        xaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', color='black'),
                        yaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', color='black'),
                        font=dict(color="black"),
                        legend=dict(font=dict(color="black"))
                    )

                    st.plotly_chart(fig)

                    # Add to Canvas Button
                    if st.button(f"Add \"Population Density\" to Canvas", key="add_density_to_canvas"):
                        add_plot_to_canvas(fig, "Population Density")
                        st.success("Plot added to canvas!")
            
            return

        st.write("Configuring Gating...")
        
        # Load Base Experiment (Clean state)
        if 'base_experiment' not in st.session_state:
            with st.spinner("Loading Experiment Data..."):
                loader = DataLoader()
                st.session_state.base_experiment = loader.load_as_cytoflow_experiment(st.session_state.loaded_files)
        
        # We work with base_experiment for channel selection
        base_exp = st.session_state.base_experiment

        if base_exp:
            # --- Step 1 & 2: Live/Dead Gating ---
            is_live_gate_ready = st.session_state.get('gate_accepted', False)
            
            with st.expander("1 & 2: Live/Dead Gating Parameters", expanded=not is_live_gate_ready):
                # Channel Selection
                st.write("Select Channels for Live/Dead Gating:")
                channels = base_exp.channels
                
                # Try to be smart about defaults
                default_x = "FSC 488/10-A" if "FSC 488/10-A" in channels else channels[0]
                default_y = "SSC 488/10-A" if "SSC 488/10-A" in channels else channels[1] if len(channels) > 1 else channels[0]

                x_channel = st.selectbox("X Channel (Forward Scatter)", options=channels, index=channels.index(default_x) if default_x in channels else 0)
                y_channel = st.selectbox("Y Channel (Side Scatter)", options=channels, index=channels.index(default_y) if default_y in channels else 0)

                # --- Cell Line Detection ---
                cell_line_col = None
                unique_cell_lines = []
                
                # Inspect metadata for cell line information
                potential_cols = ['Cell Line', 'Cell_Line', 'cell_line', 'CellType', 'cell_type', 'Strain', 'strain']
                for c in potential_cols:
                    if c in base_exp.data.columns:
                        unique_vals = base_exp.data[c].dropna().astype(str).unique()
                        if len(unique_vals) > 1:
                            cell_line_col = c
                            unique_cell_lines = sorted(unique_vals)
                            break
                
                is_multi_line_meta = (cell_line_col is not None)
                
                active_cell_line = None
                working_exp = base_exp
                
                if is_multi_line_meta:
                    st.info(f"Detected multiple cell lines in '{cell_line_col}': {', '.join(unique_cell_lines)}")
                    gating_mode = "Per Cell Line"
                    
                    # Selector for current cell line
                    active_cell_line = st.selectbox("Select Cell Line to Gate", unique_cell_lines, key="cell_line_sel_top")
                    
                    # Filter Experiment for this cell line
                    import cytoflow
                    try:
                        subset_data = base_exp.data[base_exp.data[cell_line_col] == active_cell_line].copy()
                        subset_data.reset_index(drop=True, inplace=True)
                        
                        # Correctly rebuild metadata for Cytoflow
                        # 1. Start with original metadata if available
                        new_metadata = {}
                        for col in subset_data.columns:
                            if col in base_exp.metadata:
                                new_metadata[col] = base_exp.metadata[col]
                            else:
                                # Fallback for new columns or if base_exp.metadata is missing it
                                if subset_data[col].dtype.name in ['object', 'category']:
                                    new_metadata[col] = {'type': 'condition', 'class': 'category'}
                                else:
                                    new_metadata[col] = {'type': 'channel'}
                                    
                        working_exp = cytoflow.Experiment(data=subset_data, metadata=new_metadata)
                    except Exception as e:
                        st.error(f"Failed to create subset for {active_cell_line}: {e}")
                        working_exp = base_exp # Fallback
                else:
                    # Gating Mode Selection (Manual text fallback)
                    gating_mode = st.radio("Gating Mode", ["Single Cell Line", "Mixed Population (Cluster)"], horizontal=True, 
                                         help="Single: One homogeneous population. Mixed: Multiple, co-cultured populations (uses GMM/DBSCAN).")

                # Unique Key Suffix for Widgets
                # If per-cell-line, controls are independent.
                ukey = f"_{active_cell_line}" if active_cell_line else "_global"

                # Gating Method Selection
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    options = ["Density", "GMM", "DBSCAN"]
                    if gating_mode == "Mixed Population (Cluster)":
                        options = ["GMM", "DBSCAN"]
                    
                    gating_method = st.selectbox("Gating Method", options, index=0, key=f"method{ukey}")
                
                # Default values to prevent unbound errors
                eps = 0.05
                min_samples = 50

                with col_g2:
                    if gating_method == "DBSCAN":
                        eps = st.number_input("Epsilon (Radius)", 0.01, 1.0, 0.05, 0.01, format="%.2f", key=f"eps{ukey}")
                        min_samples = st.number_input("Min Samples", 5, 200, 50, 5, key=f"min_samples{ukey}")
                        # Defaults for others
                        num_comp = 2; keep_percent = 0.9; sigma_val = 2.0
                    elif gating_method == "Density":
                        keep_percent = st.slider("Keep Fraction", 0.1, 1.0, 0.9, 0.05, key=f"keep{ukey}")
                        sigma_val = 2.0
                        num_comp = 2 # Unused
                        min_samples = 50
                    else:
                        num_comp = st.number_input("Num Components", min_value=1, max_value=5, value=2, key=f"comp{ukey}")
                        keep_percent = 0.9 # Unused
                        sigma_val = 2.0 # Unused
                        min_samples = 50

                use_outliers = st.checkbox("Remove Outliers before Estimation", value=False, key=f"outliers{ukey}")

                # Boundary Percentile (User Request: Toggle bar 50-100, step 1)
                # This controls the contour tightness for the final Polygon gate
                boundary_percentile = st.slider("Boundary Confidence (Percentile)", 50, 100, 90, 1, key=f"bound{ukey}") 

            # --- Reactive Auto-Gating ---
            # Detect parameter changes to trigger re-calculation automatically.
            current_gating_params = {
                'x': x_channel,
                'y': y_channel,
                'mode': gating_mode,
                'cell_line': active_cell_line,
                'method': gating_method,
                'keep': keep_percent,
                'comp': num_comp,
                'sigma': sigma_val,
                'eps': eps,
                'min_samples': min_samples,
                'use_outliers': use_outliers,
                'working_exp_id': id(working_exp) # Ensure we re-run if the subset/file changes
            }
            
            # Key for storing last params needs to be unique per cell line too, 
            # or we just rely on the 'cell_line' field in the dict to trigger change.
            last_gating_params = st.session_state.get('last_gating_params', {})
            
            # Check if params changed OR if we don't have a result yet
            should_run_gating = (current_gating_params != last_gating_params) or ('gated_experiment' not in st.session_state)

            if should_run_gating:
                gater = GatingManager()
                with st.spinner(f"Applying {gating_method} Gating..."):
                    try:
                        # Apply to the WORKING experiment (subset or full)
                        gated_experiment = gater.auto_gate_live_cells(working_exp, 
                                                                    x_channel=x_channel, 
                                                                    y_channel=y_channel,
                                                                    method=gating_method.lower(),
                                                                    num_components=num_comp,
                                                                    keep=keep_percent,
                                                                    sigma=sigma_val,
                                                                    use_outlier_removal=use_outliers,
                                                                    eps=eps,
                                                                    min_samples=min_samples)
                        
                        # Update State
                        st.session_state.gated_experiment = gated_experiment
                        st.session_state.last_gating_params = current_gating_params
                    except Exception as e:
                        st.error(f"Gating failed: {e}")
            
            # If we have a gated experiment, show it (persisting the view)
            if 'gated_experiment' in st.session_state:
                st.divider()
                st.subheader("3. Verification & Active Learning")
                
                exp = st.session_state.gated_experiment
                render_uncertainty_controls(exp, x_channel, y_channel)
                render_reset_controls()
                df = exp.data.copy() # Safe copy

                # --- Representative File Selection (Request: Gate per cell line on one file) ---
                if active_cell_line and 'Sample' in df.columns:
                    unique_samples = sorted(df['Sample'].unique())
                    if len(unique_samples) > 1:
                        st.info(f"Multiple files detected for {active_cell_line}. Select one to visualize and define the gate.")
                        selected_sample = st.selectbox("Representative File", unique_samples, key=f"sample_sel{ukey}")
                        df = df[df['Sample'] == selected_sample].copy()

                # Scale Toggle
                st.write("### Gating Visualization")
                viz_col1, viz_col2, _ = st.columns([1, 1, 3])
                with viz_col1:
                    scale_opt = st.radio("Scale", ["Linear", "Log"], index=1, horizontal=True)
                    is_log = (scale_opt == "Log")
                with viz_col1:
                    drag_tool = st.radio("Interaction", ["Pan", "Box Select"], index=0, horizontal=True, key="gating_interaction_mode")
                    drag_mode = "pan" if drag_tool == "Pan" else "select"
                with viz_col2:
                    cmap_selected = st.selectbox("Colormap", ["Viridis", "Plasma"], index=0)

                # Color mode is always Density (Heatmap) per user request
                
                try:
                    # Prepare Coloring Logic
                    color_col = None
                    color_map = None
                    continuous_color = False
                    
                    if gating_mode == "Multiple Cell Lines" and "AutoLiveGate" in df.columns:
                        color_col = "AutoLiveGate"
                        continuous_color = False
                    else:
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
                    # Explicitly use render_mode='svg' to ensure correct z-stacking with the boundary lines
                    
                    # Add stable row id for selection mapping (works across traces)
                    plot_df = plot_df.copy()
                    plot_df['__row_id'] = plot_df.index.astype(str)

                    fig = px.scatter(plot_df, x=x_channel, y=y_channel, color=color_col, custom_data=['__row_id'], 
                                    title="Gating Result",
                                    color_discrete_map=color_map,
                                    color_continuous_scale=cmap_selected,
                                    opacity=0.8 if continuous_color else 0.6, 
                                    log_x=is_log, log_y=is_log,
                                    render_mode='svg')
                    
                    # Improve marker style for density look
                    if continuous_color:
                        fig.update_traces(marker=dict(size=6)) # Increased size for easier clicking
                        fig.update_layout(coloraxis_colorbar=dict(title="Count"))

                    # Improved Plot Layout (White Background)
                    fig.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        xaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', color='black', tickfont=dict(color='black')),
                        yaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', color='black', tickfont=dict(color='black')),
                        font=dict(color="black"),
                        legend=dict(font=dict(color="black"))
                    )

                    # Enable Click Selection by default (Shift+Click for multi)
                    # dragmode='pan' disables the "rectangle" box select, so clicks are registered as point selections.
                    fig.update_layout(clickmode='event+select', dragmode=drag_mode)
                    # Add Uncertainty Trace
                    unc_trace = get_uncertainty_trace(exp, x_channel, y_channel)
                    if unc_trace:
                        fig.add_trace(unc_trace)
                    
                    # Note: dragmode='lasso' removed per user request for "point selection"

                    # Initialize boundary variables to prevent UnboundLocalError
                    boundary_x, boundary_y = None, None
                    
                    # --- Show ALREADY ACCEPTED Gate for this Cell Line ---
                    if active_cell_line and 'cell_line_gates' in st.session_state and active_cell_line in st.session_state.cell_line_gates:
                        saved_gate = st.session_state.cell_line_gates[active_cell_line]
                        if saved_gate.get('type') == 'polygon':
                            sv = saved_gate['vertices']
                            sx = [v[0] for v in sv] + [sv[0][0]]
                            sy = [v[1] for v in sv] + [sv[0][1]]
                            fig.add_trace(go.Scatter(x=sx, y=sy, mode='lines',
                                                    line=dict(color='black', width=4),
                                                    name=f"Saved Gate ({active_cell_line})"))

                    # --- Auto-Gating Visualization (Centroids & Ellipses) ---
                    # ALWAYS overlay Auto-Gating if available
                    # Optimization: Do not re-compute contour if gate is already accepted (User Request)
                    # For Per-Cell-Line mode, we only skip if the GLOBAL gating is complete.
                    if 'AutoLiveGate' in df.columns and not st.session_state.get('gate_accepted'):
                        unique_labels = df['AutoLiveGate'].unique()
                        
                        # Reset stored boundary to prevent stale state
                        st.session_state.current_auto_boundary = None
                        
                        # Heuristic: Identify the "Live" population
                        # 1. Ignore Noise (-1)
                        # 2. Filter out insignificant clusters (< 1% count) to avoid outlier groups
                        # 3. Pick Highest Mean FSC from leftovers
                        
                        label_stats = []
                        total_events = len(df)
                        
                        for label in unique_labels:
                            # Skip standard noise label
                            if str(label) == '-1': 
                                continue
                                
                            subset = df[df['AutoLiveGate'] == label]
                            count = len(subset)
                            if count > 5:
                                mean_fsc = subset[x_channel].mean()
                                label_stats.append({
                                    'label': label,
                                    'count': count,
                                    'fraction': count / total_events,
                                    'mean_fsc': mean_fsc
                                })
                        
                        # Filter Candidates
                        # Only keep clusters with > 1% (0.01) abundance if possible
                        valid_candidates = [s for s in label_stats if s['fraction'] > 0.01]
                        
                        # If filtering removed everything (rare case), fallback to all
                        if not valid_candidates:
                            valid_candidates = label_stats
                        
                        if valid_candidates:
                            # Sort by Mean FSC Descending to find the "Rightmost" (Live) population
                            # ignoring debris (left) and assuming outliers (extreme right) are filtered by size
                            valid_candidates.sort(key=lambda x: x['mean_fsc'], reverse=True)
                            
                            # Heuristic Improvement: Prefer explicitly TRUE/True/1 labels
                            preferred_label = None
                            for lbl in unique_labels:
                                if str(lbl).lower() in ['true', '1', 'live', 'pos', 'positive']:
                                    preferred_label = lbl
                                    break
                            
                            if preferred_label is not None:
                                live_label = preferred_label
                            else:
                                # Pick the highest FSC one from valid candidates
                                live_label = valid_candidates[0]['label']

                            # Store for the "Accept" button
                            st.session_state['suggested_live_label'] = live_label

                            # --- Draw Boundaries for ALL Valid Clusters ---
                            # Iterate through candidates to show Context
                            try:
                                from glow.gating.boundary import compute_density_boundary
                                
                                # Use a color palette for non-live items
                                palette = ['cyan', 'magenta', 'orange', 'yellow', 'brown']
                                p_idx = 0
                                
                                for cand in valid_candidates:
                                    lbl = cand['label']
                                    is_target = (lbl == live_label)
                                    
                                    # User Request: Only show boundary for the Live population
                                    if not is_target:
                                        continue
                                    
                                    # Get Data
                                    subset = df[df['AutoLiveGate'] == lbl]
                                    subset_df = subset[[x_channel, y_channel]]
                                    
                                    bx, by = None, None
                                    
                                    # Calculate Boundary
                                    if is_log:
                                         valid_mask = (subset_df[x_channel] > 0) & (subset_df[y_channel] > 0)
                                         hull_points = subset_df[valid_mask].values
                                         if len(hull_points) > 10:
                                            log_points = np.log10(hull_points)
                                            # Use lower percentile for non-live to show "extent" rather than core
                                            pct = float(boundary_percentile) if is_target else 85
                                            res_x, res_y = compute_density_boundary(log_points, percentile=pct, smoothing=1.5)
                                            if res_x is not None:
                                                 bx, by = 10**res_x, 10**res_y
                                    else:
                                        valid_mask = np.isfinite(subset_df[x_channel]) & np.isfinite(subset_df[y_channel])
                                        hull_points = subset_df[valid_mask].values
                                        if len(hull_points) > 10:
                                             pct = float(boundary_percentile) if is_target else 85
                                             bx, by = compute_density_boundary(hull_points, percentile=pct, smoothing=1.5)
                                    
                                    # Determine Style
                                    if is_target:
                                        line_color = 'black'
                                        line_width = 4
                                        trace_name = "Suggested Live"
                                        # Save this one for the action
                                        boundary_x, boundary_y = bx, by 
                                    else:
                                        line_color = palette[p_idx % len(palette)]
                                        p_idx += 1
                                        line_width = 2
                                        trace_name = f"Cluster {lbl}"
                                    
                                    # Add Trace
                                    if bx is not None:
                                        fig.add_trace(go.Scatter(x=bx, y=by, mode='lines',
                                                                line=dict(color=line_color, width=line_width, dash='solid'),
                                                                name=trace_name))

                                if boundary_x is not None:
                                    # Save to session state for application (User Request)
                                    st.session_state.current_auto_boundary = list(zip(boundary_x, boundary_y))
                                    
                                    # Force boundaries to top
                                    data = list(fig.data)
                                    data.sort(key=lambda t: 1 if (t.mode == 'lines') else 0)
                                    fig.data = tuple(data)
                                    
                            except Exception as e:
                                print(f"Warning: Boundary computation failed: {e}")
                                # pass # Geometry fail

                    # --- Validation Plot (Live vs Non-Live) ---
                    # User Request: "separate scatter plot below... where live cells and non-live cells are colored"
                    if 'AutoLiveGate' in df.columns:
                        st.write("### Gating Classification")
                        
                        # Create validation dataframe (downsampled if needed)
                        # We use the same plot_df for consistency
                        # We explicitly convert the gate column to string to ensure discrete coloring
                        val_df = plot_df.copy()
                        val_df['AutoLiveGate'] = val_df['AutoLiveGate'].astype(str)
                        
                        # Map to meaningful labels
                        live_lbl = str(st.session_state.get('suggested_live_label', '1'))
                        
                        def get_label_desc(val):
                            return "Live Cells" if str(val) == live_lbl else "Debris / Dead"
                            
                        val_df['Cond_Label'] = val_df['AutoLiveGate'].apply(get_label_desc)
                        
                        # Define discrete color map
                        color_map_val = {
                            "Live Cells": "green",
                            "Debris / Dead": "red"
                        }

                        fig_val = px.scatter(val_df, x=x_channel, y=y_channel, 
                                            color='Cond_Label',
                                            title="MLP Classification (Points) vs. Computed Gate (Line)",
                                            color_discrete_map=color_map_val,
                                            opacity=0.6,
                                            log_x=is_log, log_y=is_log,
                                            render_mode='svg')
                        
                        # White background for validation plot too
                        fig_val.update_layout(
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            xaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', color='black', tickfont=dict(color='black')),
                            yaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', color='black', tickfont=dict(color='black')),
                            font=dict(color="black"),
                            legend=dict(font=dict(color="black"))
                        )
                        
                        # Draw the same boundary on this plot for verification
                        if boundary_x is not None:
                            fig_val.add_trace(go.Scatter(x=boundary_x, y=boundary_y, mode='lines', 
                                                            line=dict(color='black', width=2), # Black Line, Thinner
                                                            name="Computed Boundary"))
                            
                            # Force order on validation plot too
                            val_data = list(fig_val.data)
                            val_data.sort(key=lambda t: 1 if "Boundary" in (t.name or "") else 0)
                            fig_val.data = tuple(val_data)

                        st.plotly_chart(fig_val, use_container_width=True)

                        if st.button("Add \"Live Classification\" to Canvas", key="add_val_to_canvas"):
                            add_plot_to_canvas(fig_val, "Live Classification")
                            st.success("Plot added to canvas!")


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
                    selection = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points" if drag_mode=="pan" else "box", key=f"live_gating_plot_{len(df)}_{st.session_state.get('plot_id', 0)}")
                    render_labeling_controls(exp, x_channel, y_channel, selection, plot_df=plot_df)
                    
                    # Check for NEW selection
                    # Only treat as Manual Gating if explicitly in manual mode
                    if st.session_state.get('show_manual_gating', False) and selection and selection.get("selection"):
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
                    if ActiveLearningLogger:
                        try:
                            logger = ActiveLearningLogger()
                        except:
                            logger = None
                    else:
                        logger = None

                        # Feedback Controls (Only show if no pending selection to avoid clutter)
                    if not st.session_state.get('pending_manual_vertices'):
                        
                        st.divider()
                        col1, col2 = st.columns(2)
                        
                        btn_label = "✅ Accept Auto-Gate"
                        if active_cell_line:
                            btn_label = f"✅ Accept Gate for {active_cell_line}"

                        if col1.button(btn_label):
                            # --- Per Cell Line Logic ---
                            if active_cell_line:
                                if 'cell_line_gates' not in st.session_state:
                                     st.session_state.cell_line_gates = {}
                                
                                # Store the gate for this cell line
                                gate_data = {}
                                if st.session_state.get('current_auto_boundary'):
                                     # Optimization: Simplify polygon if too complex
                                     verts = st.session_state.current_auto_boundary
                                     if len(verts) > 150:
                                        step = len(verts) // 150
                                        if step > 1:
                                            verts = verts[::step]
                                            if verts[-1] != st.session_state.current_auto_boundary[-1]:
                                                 verts.append(st.session_state.current_auto_boundary[-1])

                                     gate_data = {
                                         'type': 'polygon',
                                         'vertices': verts,
                                         'channels': [x_channel, y_channel]
                                     }
                                elif 'suggested_live_label' in st.session_state:
                                     gate_data = {
                                         'type': 'label',
                                         'label': st.session_state.suggested_live_label,
                                         'auto_params': current_gating_params
                                     }
                                
                                st.session_state.cell_line_gates[active_cell_line] = gate_data
                                
                                # Check progress
                                all_lines = set(unique_cell_lines)
                                done_lines = set(st.session_state.cell_line_gates.keys())
                                missing = all_lines - done_lines
                                
                                if not missing:
                                    st.success("All cell lines gated! Processing final dataset...")
                                    st.session_state.live_gate_params = {
                                        'type': 'per_cell_line',
                                        'gates': st.session_state.cell_line_gates,
                                        'col': cell_line_col
                                    }
                                    st.session_state.gate_accepted = True
                                    st.rerun()
                                else:
                                    # Logic to auto-switch to next missing
                                    next_line = list(missing)[0]
                                    # Store a temporary success message in session state to show AFTER rerun
                                    st.session_state[f"success_msg_{active_cell_line}"] = f"✅ Gate saved for {active_cell_line}."
                                    st.rerun()
                                return

                        # Display previous successes (if any) to avoid duplicate messages on same page
                        if active_cell_line:
                            skey = f"success_msg_{active_cell_line}"
                            if skey in st.session_state:
                                st.success(st.session_state[skey])
                                # Optional: del st.session_state[skey] # Clear after showing?
                                
                        # Progress & Reset for Multi-Gating (Only for Per Cell Line mode)
                        if active_cell_line:
                            if 'cell_line_gates' in st.session_state:
                                done_lines = list(st.session_state.cell_line_gates.keys())
                                if done_lines:
                                    st.write(f"**Gated Lines:** {', '.join(done_lines)}")
                                    if active_cell_line in st.session_state.cell_line_gates:
                                        if st.button(f"🔄 Reset Gate for {active_cell_line}", key=f"reset{ukey}"):
                                            del st.session_state.cell_line_gates[active_cell_line]
                                            st.rerun()
                                
                                if st.button("🗑️ Reset ALL Cell Line Gates", type="secondary"):
                                    st.session_state.cell_line_gates = {}
                                    st.rerun()

                        # --- Original/Global Gating Logic (Only if NOT Per Cell Line) ---
                        if not active_cell_line:
                            if logger:
                                # Log the acceptance
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
                            # User Request: Use the Contour (Polygon) as the gate, not just the label
                            total_cells = len(base_exp.data)
                            
                            if st.session_state.get('current_auto_boundary') and CYTOFLOW_AVAILABLE:
                                verts = st.session_state.current_auto_boundary
                                
                                # Optimization: Simplify polygon if too complex to prevent hang
                                if len(verts) > 150:
                                    # Simple decimation: keep max ~150 points
                                    step = len(verts) // 150
                                    if step > 1:
                                        verts = verts[::step]
                                        # Ensure the last point connects back effectively or is included
                                        if verts[-1] != st.session_state.current_auto_boundary[-1]:
                                             verts.append(st.session_state.current_auto_boundary[-1])
                                
                                # Store params for Export
                                st.session_state.live_gate_params = {
                                    'type': 'polygon', 
                                    'vertices': verts, 
                                    'channels': [x_channel, y_channel]
                                }

                                gater = GatingManager()
                                
                                # Apply Manual Gate (Polygon) using the computed contour
                                # Apply to BASE experiment so we get a clean cut
                                exp = gater.apply_manual_gate(base_exp, "LiveBoundaryGate", x_channel, y_channel, verts)
                                
                                # Filter to keep only the cell inside the contour
                                if "LiveBoundaryGate" in exp.data.columns:
                                     # Filter in-place
                                     exp.data = exp.data[exp.data["LiveBoundaryGate"]].copy()
                                
                                kept_cells = len(exp.data)
                                percent = (kept_cells / total_cells) * 100 if total_cells > 0 else 0
                                st.session_state.live_gate_stats = f"{kept_cells} / {total_cells} ({percent:.1f}%)"

                                st.session_state.gated_experiment = exp
                                st.session_state.gate_accepted = True
                                st.session_state.doublets_removed = False # Reset downstream step
                                st.success(f"Auto-Gate Accepted! Applied Contour Gate. Live Cells: {st.session_state.live_gate_stats}")
                                st.rerun()

                            elif 'suggested_live_label' in st.session_state and CYTOFLOW_AVAILABLE:
                                label = st.session_state.suggested_live_label

                                # Store params for Export
                                st.session_state.live_gate_params = {
                                    'type': 'label',
                                    'label': label,
                                    'channel': 'AutoLiveGate'
                                }

                                exp = st.session_state.gated_experiment
                                
                                # Filter to keep only the Live Label population
                                # We modify the experiment in-place for the session
                                filtered_df = exp.data[exp.data['AutoLiveGate'] == label].copy()
                                exp.data = filtered_df
                                
                                kept_cells = len(exp.data)
                                percent = (kept_cells / total_cells) * 100 if total_cells > 0 else 0
                                st.session_state.live_gate_stats = f"{kept_cells} / {total_cells} ({percent:.1f}%)"

                                st.session_state.gated_experiment = exp
                                st.session_state.gate_accepted = True
                                st.session_state.doublets_removed = False # Reset downstream step
                                st.success(f"Auto-Gate Accepted! Proceeding to Singlet Detection... Live Cells: {st.session_state.live_gate_stats}")
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
                                total_cells = len(base_exp.data)
                                
                                # Apply to BASE experiment
                                manual_exp = gater.apply_manual_gate(base_exp, "ManualLiveGate", x_channel, y_channel, vertices)

                                # Store params for Export
                                st.session_state.live_gate_params = {
                                    'type': 'polygon',
                                    'vertices': vertices,
                                    'channels': [x_channel, y_channel]
                                }
                                
                                # Filter manually to match Auto behavior
                                if "ManualLiveGate" in manual_exp.data.columns:
                                     manual_exp.data = manual_exp.data[manual_exp.data["ManualLiveGate"]].copy()
                                
                                kept_cells = len(manual_exp.data)
                                percent = (kept_cells / total_cells) * 100 if total_cells > 0 else 0
                                st.session_state.live_gate_stats = f"{kept_cells} / {total_cells} ({percent:.1f}%)"

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
                                st.session_state.doublets_removed = False # Reset downstream step
                                st.rerun()
                                    
                            except Exception as e:
                                st.error(f"Invalid format: {e}")
                    
                    # --- Step 3: Singlet Detection ---
                    if st.session_state.get('gate_accepted'): 
                        st.divider()
                        
                        # Stats Display (Moved here per user request)
                        if st.session_state.get('live_gate_stats'):
                            st.success(f"Final Live Cells: {st.session_state.live_gate_stats}")
                            
                        st.header("3. Singlet Detection")
                        
                        # Automatically proceed to Singlet Detection
                        is_doublet_mode = not st.session_state.get('doublets_removed', False)
                        st.session_state.show_doublet_gating = is_doublet_mode
                        
                        # --- Per Cell Line Mode Detection for Singlets ---
                        active_line_singlets = None
                        if is_multi_line_meta:
                            st.info(f"Detected multiple cell lines. Process singlets per cell line.")
                            
                            # Selector for current cell line (independent from Step 1)
                            active_line_singlets = st.selectbox("Select Cell Line for Singlet Gating", unique_cell_lines, key="singlet_cell_line_sel")
                            
                            # Filter Base Experiment for this cell line
                            subset_data = base_exp.data[base_exp.data[cell_line_col] == active_line_singlets].copy()
                            subset_data.reset_index(drop=True, inplace=True)
                            
                            # Re-Apply the Live/Dead Gate previously saved for this cell line
                            if 'cell_line_gates' in st.session_state and active_line_singlets in st.session_state.cell_line_gates:
                                gate = st.session_state.cell_line_gates[active_line_singlets]
                                if gate['type'] == 'polygon':
                                    from matplotlib.path import Path
                                    v = gate['vertices']
                                    ch = gate['channels']
                                    path = Path(v)
                                    is_in = path.contains_points(subset_data[ch].values)
                                    df_gated = subset_data[is_in].copy()
                                elif gate['type'] == 'label':
                                    # Fallback if label
                                    df_gated = subset_data # logic simplified
                                else:
                                    df_gated = subset_data
                            else:
                                # Fallback if no specific gate found (shouldn't happen if Step 1 complete)
                                df_gated = subset_data
                        else:
                            exp = st.session_state.gated_experiment
                            if exp:
                                df_gated = exp.data
                            else:
                                df_gated = base_exp.data # Fallback

                        if is_doublet_mode:
                            st.session_state.show_doublet_gating = True
                            
                            d_channels = base_exp.channels # Use base channels for selection
                            
                            # Auto-guess channels
                            idx_a, idx_h = 0, 0
                            for i, c in enumerate(d_channels):
                                if "FSC" in c and ("-A" in c or "Area" in c): idx_a = i
                                if "FSC" in c and ("-H" in c or "Height" in c): idx_h = i
                            
                            # Unique key for singlet controls
                            sukey = f"_singlet_{active_line_singlets}" if active_line_singlets else "_singlet_global"
                            
                            dc1, dc2, dc3 = st.columns(3)
                            d_x = dc1.selectbox("Area Channel (X)", d_channels, index=idx_a, key=f"dA{sukey}")
                            d_y = dc2.selectbox("Height Channel (Y)", d_channels, index=idx_h, key=f"dH{sukey}")
                            d_scale = dc3.radio("Scale", ["Linear", "Log"], index=0, key=f"dScale{sukey}") # Default Linear
                            d_is_log = (d_scale == "Log")

                            # --- Representative File Selection for Singlets ---
                            if active_line_singlets and 'Sample' in df_gated.columns:
                                unique_s_samples = sorted(df_gated['Sample'].unique())
                                if len(unique_s_samples) > 1:
                                    st.info(f"Multiple files for {active_line_singlets}. Select representative for singlet gating.")
                                    rep_sample = st.selectbox("Representative File (Singlets)", unique_s_samples, key=f"s_sample_{sukey}")
                                    viz_df_full = df_gated[df_gated['Sample'] == rep_sample].copy()
                                else:
                                    viz_df_full = df_gated
                            else:
                                viz_df_full = df_gated

                            # --- Auto-Gate Calculation (Singlets - GMM on Linear Data) ---
                            try:
                                # Always calculate on Linear for Singlets (Physical relationship is linear)
                                X_lin_full = df_gated[[d_x, d_y]].values
                                
                                # Fit GMM (Fit on FULL gated subset for this cell line, or just representative? Let's do full subset for better stats)
                                gmm_lin = GaussianMixture(n_components=1, random_state=42)
                                gmm_lin.fit(X_lin_full)
                                
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
                                
                                # Define viz_df (Downsampled from viz_df_full)
                                subset_n = 5000
                                viz_df = viz_df_full.sample(subset_n, random_state=42) if len(viz_df_full) > subset_n else viz_df_full

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
                                                        line=dict(color='black', width=3), # User Request: Black full line
                                                        marker=dict(size=6, color='black'),
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
                                    xaxis=dict(range=start_range_x, color='black', tickfont=dict(color='black')),
                                    yaxis=dict(range=start_range_y, color='black', tickfont=dict(color='black')),
                                    legend=dict(font=dict(color="black")),
                                    uirevision='constant' # Keeps user zoom state on re-runs
                                )
                                fig_d.update_xaxes(showgrid=True, gridcolor='rgba(0,0,0,0.1)', linecolor='black', zeroline=False)
                                fig_d.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.1)', linecolor='black', zeroline=False)
                                
                                # Selection Interaction
                                # User req for Shift selection is handled by Plotly standard behavior with clickmode
                                sel_d = st.plotly_chart(fig_d, use_container_width=True, on_select="rerun", selection_mode="points", key="doublet_chart")

                                if st.button("Add \"Singlet Detection\" to Canvas", key="add_singlet_to_canvas"):
                                    add_plot_to_canvas(fig_d, "Singlet Detection")
                                    st.success("Plot added to canvas!")
                                
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
                                btn_label_s = "✅ Accept Auto-Singlets"
                                if active_line_singlets:
                                    btn_label_s = f"✅ Accept Singlets for {active_line_singlets}"

                                if col_da.button(btn_label_s, key=f"accept_singlets{sukey}"):
                                    # --- Per Cell Line Singlet Logic ---
                                    if active_line_singlets:
                                        if 'cell_line_singlet_gates' not in st.session_state:
                                            st.session_state.cell_line_singlet_gates = {}
                                        
                                        # Save GMM Params for this cell line
                                        st.session_state.cell_line_singlet_gates[active_line_singlets] = {
                                            'type': 'gmm_mahalanobis',
                                            'mean': d_mean,
                                            'cov': d_cov,
                                            'channels': [d_x, d_y],
                                            'threshold': 2.5
                                        }

                                        # Progress check
                                        all_lines = set(unique_cell_lines)
                                        done_lines = set(st.session_state.cell_line_singlet_gates.keys())
                                        missing = all_lines - done_lines

                                        if not missing:
                                            st.success("All cell line singlets gated!")
                                            st.session_state.singlet_gate_params = {
                                                'type': 'per_cell_line',
                                                'gates': st.session_state.cell_line_singlet_gates,
                                                'col': cell_line_col
                                            }
                                            st.session_state.doublets_removed = True
                                            st.rerun()
                                        else:
                                            next_line = list(missing)[0]
                                            st.session_state[f"success_msg_singlet_{active_line_singlets}"] = f"✅ Singlet gate saved for {active_line_singlets}."
                                            st.rerun()
                                        return

                                    # --- Global Singlet Logic ---
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

                                    # Store params for Export
                                    st.session_state.singlet_gate_params = {
                                        'type': 'gmm_mahalanobis',
                                        'mean': d_mean,
                                        'cov': d_cov,
                                        'channels': [d_x, d_y],
                                        'threshold': 2.5
                                    }
                                    
                                    # Calculate Stats
                                    total_input = len(exp.data)
                                    kept_singlets = sum(mask)
                                    pct = (kept_singlets / total_input) * 100 if total_input > 0 else 0
                                    st.session_state.singlet_gate_stats = f"{kept_singlets} / {total_input} ({pct:.1f}%)"
                                    
                                    exp.data = exp.data[mask]
                                    st.session_state.gated_experiment = exp
                                    st.session_state.doublets_removed = True
                                    st.session_state.show_doublet_gating = False
                                    st.success(f"Kept {sum(mask)} singlet events.")
                                    st.rerun()

                                # Reject -> Enable Manual
                                if col_dr.button("❌ Reject / Manual Gate", key=f"reject_singlets{sukey}"):
                                    st.session_state.doublet_manual_mode = True
                                    st.rerun()

                                # Progress Feedback for multi-singlets
                                if active_line_singlets:
                                    skey_s = f"success_msg_singlet_{active_line_singlets}"
                                    if skey_s in st.session_state:
                                        st.success(st.session_state[skey_s])
                                    
                                    if 'cell_line_singlet_gates' in st.session_state:
                                        done_s = list(st.session_state.cell_line_singlet_gates.keys())
                                        if done_s:
                                            st.write(f"**Gated Singlets:** {', '.join(done_s)}")
                                            if active_line_singlets in st.session_state.cell_line_singlet_gates:
                                                if st.button(f"🔄 Reset Singlet Gate for {active_line_singlets}", key=f"reset_s{active_line_singlets}"):
                                                    del st.session_state.cell_line_singlet_gates[active_line_singlets]
                                                    st.rerun()
                                            
                                            if st.button("🗑️ Reset ALL Singlet Gates", type="secondary", key="reset_all_s"):
                                                st.session_state.cell_line_singlet_gates = {}
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
                                        if col_m1.button("✅ Confirm Manual Gate", key=f"confirm_manual_singlets{sukey}"):

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
                                                    # --- Per Cell Line Manual Singlet Logic ---
                                                    if active_line_singlets:
                                                        if 'cell_line_singlet_gates' not in st.session_state:
                                                            st.session_state.cell_line_singlet_gates = {}
                                                        
                                                        st.session_state.cell_line_singlet_gates[active_line_singlets] = {
                                                            'type': 'polygon',
                                                            'vertices': parsed_verts,
                                                            'channels': [d_x, d_y]
                                                        }

                                                        # Progress check
                                                        all_lines = set(unique_cell_lines)
                                                        done_lines = set(st.session_state.cell_line_singlet_gates.keys())
                                                        missing = all_lines - done_lines

                                                        if not missing:
                                                            st.success("All cell line singlets gated!")
                                                            st.session_state.singlet_gate_params = {
                                                                'type': 'per_cell_line',
                                                                'gates': st.session_state.cell_line_singlet_gates,
                                                                'col': cell_line_col
                                                            }
                                                            st.session_state.doublets_removed = True
                                                            st.rerun()
                                                        else:
                                                            st.session_state[f"success_msg_singlet_{active_line_singlets}"] = f"✅ Manual singlet gate saved for {active_line_singlets}."
                                                            st.rerun()
                                                        return

                                                    # --- Global Manual Singlet Logic ---
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

                                                    # Use GatingManager for standard application + Filter
                                                    gater = GatingManager()
                                                    try:
                                                        gate_name = "ManualSingletGate"
                                                        total_input = len(exp.data)

                                                        exp = gater.apply_manual_gate(exp, gate_name, d_x, d_y, parsed_verts)

                                                        # Store params for Export
                                                        st.session_state.singlet_gate_params = {
                                                            'type': 'polygon',
                                                            'vertices': parsed_verts,
                                                            'channels': [d_x, d_y]
                                                        }
                                                        
                                                        # Cytoflow apply() adds the condition, we must explicitly filter for the UI validation
                                                        if gate_name in exp.data.columns:
                                                            exp.data = exp.data[exp.data[gate_name]].copy()
                                                        
                                                        kept_singlets = len(exp.data)
                                                        pct = (kept_singlets / total_input) * 100 if total_input > 0 else 0
                                                        st.session_state.singlet_gate_stats = f"{kept_singlets} / {total_input} ({pct:.1f}%)"

                                                        st.session_state.gated_experiment = exp
                                                        st.session_state.doublets_removed = True
                                                        st.session_state.show_doublet_gating = False
                                                        st.session_state.doublet_manual_mode = False # Reset
                                                        if 'pending_doublet_verts' in st.session_state:
                                                            del st.session_state.pending_doublet_verts
                                                        st.success(f"Manual singlet gate applied. Kept {len(exp.data)} events.")
                                                        st.rerun()
                                                    except Exception as e:
                                                        st.error(f"Failed to apply gate: {e}")

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
                            # Stats for Singlets
                            if st.session_state.get('singlet_gate_stats'):
                                st.success(f"Final Singlets: {st.session_state.singlet_gate_stats}")
                            else:
                                st.success("Doublets have been removed from the dataset.")
                            
                            st.divider()
                            st.header("4. Export Data")
                            
                            export_mode = st.radio("Export Mode", ["Gated Population (Live + Singlets)", "Full Population (Annotated)"], horizontal=True)
                            
                            # Determine Experiment Name for filename
                            import os
                            import io
                            experiment_name = "experiment"
                            if 'loaded_files' in st.session_state and st.session_state.loaded_files:
                                try:
                                    experiment_name = os.path.basename(os.path.dirname(st.session_state.loaded_files[0]))
                                except:
                                    pass

                            final_csv = None
                            file_name = "processed_data.csv.gz"

                            if export_mode == "Gated Population (Live + Singlets)":
                                df_gated = st.session_state.gated_experiment.data.copy()
                                
                                # Filter channels if requested
                                if st.session_state.get('selected_channels'):
                                    all_ch = st.session_state.base_experiment.channels
                                    to_drop = [c for c in all_ch if c not in st.session_state.selected_channels]
                                    df_gated = df_gated.drop(columns=[c for c in to_drop if c in df_gated.columns])

                                # Unify columns to represent the state of this data
                                # Since this is the gated subset, these conditions are by definition met (or ignored if not applied)
                                df_gated['cells'] = True
                                df_gated['singlets'] = True
                                
                                # Compress to GZIP
                                buf = io.BytesIO()
                                df_gated.to_csv(buf, index=False, compression='gzip')
                                final_csv = buf.getvalue()
                                
                                file_name = f"{experiment_name}_singlets.csv.gz"
                            else:
                                with st.spinner("Reconstructing full dataset annotation..."):
                                    # Clone base
                                    full_exp = st.session_state.base_experiment.clone()

                                    # Filter channels if requested
                                    if st.session_state.get('selected_channels'):
                                        all_ch = st.session_state.base_experiment.channels
                                        to_drop = [c for c in all_ch if c not in st.session_state.selected_channels]
                                        full_exp.data = full_exp.data.drop(columns=[c for c in to_drop if c in full_exp.data.columns])
                                        if hasattr(full_exp, 'channels'):
                                            full_exp.channels = st.session_state.selected_channels

                                    bgater = GatingManager()
                                    
                                    # 1. Re-Apply Live Gate (if exists)
                                    l_params = st.session_state.get('live_gate_params')
                                    # Do not modify full_exp.data directly before Cytoflow ops to avoid metadata mismatch errors

                                    if l_params:
                                        if l_params['type'] == 'polygon':
                                            full_exp = bgater.apply_manual_gate(full_exp, "LiveGate", l_params['channels'][0], l_params['channels'][1], l_params['vertices'])
                                        elif l_params['type'] == 'label':
                                            # Retrieve params
                                            gp = st.session_state.get('last_gating_params', {})
                                            if gp:
                                                # Re-run auto gating to get labels
                                                try:
                                                    # Fix: Ensure method is lower case string
                                                    method_name = gp.get('method', 'density').lower() if isinstance(gp.get('method'), str) else 'density'
                                                    
                                                    res_exp = bgater.auto_gate_live_cells(
                                                        full_exp,
                                                        x_channel=gp.get('x'),
                                                        y_channel=gp.get('y'),
                                                        method=method_name,
                                                        num_components=gp.get('comp', 2),
                                                        keep=gp.get('keep', 0.9),
                                                        sigma=gp.get('sigma', 2.0),
                                                        eps=gp.get('eps', 0.05),
                                                        min_samples=gp.get('min_samples', 50),
                                                        use_outlier_removal=gp.get('use_outliers', False)
                                                    )
                                                    full_exp = res_exp
                                                except Exception as e:
                                                    st.warning(f"Auto-gate reconstruction warning: {e}")
                                            else:
                                                st.warning("Could not reconstruct Auto Gate (params lost). Annotating all as Live.")

                                    # 2. Re-Apply Singlet Gate (if exists)
                                    s_params = st.session_state.get('singlet_gate_params')
                                    
                                    if s_params:
                                        if s_params['type'] == 'polygon':
                                            full_exp = bgater.apply_manual_gate(full_exp, "SingletGate", s_params['channels'][0], s_params['channels'][1], s_params['vertices'])
                                        
                                        # Note: GMM/Mahalanobis is calculated on DataFrame level below to avoid complex custom Cytoflow wrappers

                                    # 3. Final Assembly & Column Unification
                                    # Create a copy as DataFrame to safely add annotation columns
                                    df_export = full_exp.data.copy()

                                    # Create "cells" column (Live Cells)
                                    if 'LiveGate' in df_export.columns:
                                        df_export['cells'] = df_export['LiveGate']
                                    elif 'AutoLiveGate' in df_export.columns and l_params and l_params.get('type') == 'label':
                                        lbl = l_params['label']
                                        df_export['cells'] = (df_export['AutoLiveGate'] == lbl)
                                    else:
                                        # If no gate applied, assume everything is part of population or False? 
                                        # Usually False if we strive for intersection, but if no gate, maybe True?
                                        # Based on prompt, we want annotations. Defaulting to False (not live) or True (all live) depends on semantics.
                                        # Let's assume False for safety if gate missing, OR True if we consider "no gate = all cells".
                                        # Given loop structure, we only enter if l_params. If no l_params, likely no gate.
                                        df_export['cells'] = False # Default fail
                                    
                                    # Create "singlets" column
                                    if 'SingletGate' in df_export.columns:
                                        df_export['singlets'] = df_export['SingletGate']
                                    elif s_params and s_params.get('type') == 'gmm_mahalanobis':
                                         try:
                                            X = df_export[s_params['channels']].values
                                            d_mean = s_params['mean']
                                            inv_cov = np.linalg.inv(s_params['cov'])
                                            threshold = s_params.get('threshold', 2.5)
                                            
                                            diff = X - d_mean
                                            left = diff @ inv_cov
                                            dist_sq = np.sum(left * diff, axis=1)
                                            dist = np.sqrt(dist_sq)
                                            
                                            df_export['singlets'] = dist <= threshold
                                         except Exception as e:
                                             st.warning(f"Singlet calculation warning: {e}")
                                             df_export['singlets'] = False
                                    else:
                                        # Default False if no singlet gate defined
                                        df_export['singlets'] = False
                                    
                                    # Remove redundant LiveGate column as requested
                                    if 'LiveGate' in df_export.columns:
                                        del df_export['LiveGate']

                                    # Compress to GZIP
                                    buf = io.BytesIO()
                                    df_export.to_csv(buf, index=False, compression='gzip')
                                    final_csv = buf.getvalue()
                                    
                                    file_name = f"{experiment_name}_full_population.csv.gz"
                            
                            st.download_button("Download CSV (GZIP)", data=final_csv, file_name=file_name, mime="application/gzip")

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
