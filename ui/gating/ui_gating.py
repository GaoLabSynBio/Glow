import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
from .ui_active_learning import render_uncertainty_controls, get_uncertainty_trace, render_labeling_controls, render_reset_controls, get_active_learner
from glow.canvas.functions import add_plot_to_canvas
import matplotlib.pyplot as plt
import numpy as np

def create_canvas_figure(df, x_col, y_col, title, is_log=False, color_col=None, boundaries=None, cmap='plasma', color_map=None):
    try:
        fig, ax = plt.subplots(figsize=(6, 5))

        # Scatter
        if color_col and color_col in df.columns:
            if color_map:
                 # Explicit categorical mapping
                 grouped = df.groupby(color_col)
                 for name, group in grouped:
                     c = color_map.get(name, 'blue')
                     ax.scatter(group[x_col], group[y_col], c=c, label=str(name), s=10, alpha=0.6, edgecolors='none')
                 ax.legend(title=color_col)
            # Check if color_col is numeric for cmap
            elif np.issubdtype(df[color_col].dtype, np.number):
                 sc = ax.scatter(df[x_col], df[y_col], c=df[color_col], cmap=cmap, s=10, alpha=0.6, edgecolors='none')
                 plt.colorbar(sc, ax=ax, label=color_col)
            else:
                 # Categorical coloring fallback
                 unique_vals = df[color_col].unique()
                 for val in unique_vals:
                     mask = df[color_col] == val
                     ax.scatter(df.loc[mask, x_col], df.loc[mask, y_col], label=str(val), s=10, alpha=0.6, edgecolors='none')
                 ax.legend(title=color_col)
        else:
            ax.scatter(df[x_col], df[y_col], c='blue', s=10, alpha=0.6, edgecolors='none')

        # Boundaries
        if boundaries:
            for b in boundaries:
                verts = b.get('vertices')
                if not verts: continue

                bx = [v[0] for v in verts]
                by = [v[1] for v in verts]
                # close loop if strict polygon
                if len(bx) > 0 and (bx[0] != bx[-1] or by[0] != by[-1]):
                    bx.append(bx[0])
                    by.append(by[0])

                col = b.get('color', 'black')
                lw = b.get('width', 2)
                ax.plot(bx, by, color=col, linewidth=lw)

        if is_log:
            ax.set_xscale('log')
            ax.set_yscale('log')

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        return fig
    except Exception as e:
        print(f"Fig error: {e}")
        return plt.figure()


def render_gating_section(DataLoader, GatingManager, ActiveLearningLogger, CYTOFLOW_AVAILABLE):
    # Gating Section showing the base functionality
    st.subheader("2. Live Gating Setup")

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
                    exclude_cols = ['cells', 'singlets', 'Is_Live', 'Is_Singlet', '_temp_live_labels', '_temp_live_mask_contour', '_temp_live_mask_manual', '_temp_singlet_mask_manual', '_temp_singlet_mask_auto']
                    channels = [c for c in numeric_cols if c not in exclude_cols]

                    if not channels:
                         channels = exp.data.columns.tolist() # Fallback

                    # Try to default to standard Scatter channels if available
                    default_x = "FSC 488/10-A" if "FSC 488/10-A" in channels else channels[0]
                    default_y = "SSC 488/10-A" if "SSC 488/10-A" in channels else channels[1] if len(channels) > 1 else channels[0]

                    x_axis = st.selectbox("X Axis", channels, index=channels.index(default_x) if default_x in channels else 0, key="csv_x")
                    y_axis = st.selectbox("Y Axis", channels, index=channels.index(default_y) if default_y in channels else 0, key="csv_y")

                    c_opt1, c_opt2, c_opt3 = st.columns([1, 1, 1])
                    with c_opt1:
                        scale_opt = st.radio("Scale", ["Linear", "Log"], index=1, horizontal=True, key="csv_scale")
                        is_log = (scale_opt == "Log")
                    with c_opt2:
                        drag_tool = st.radio("Interaction", ["Pan", "Box Select"], index=0, horizontal=True, key="csv_interaction_mode")
                        drag_mode = "pan" if drag_tool == "Pan" else "select"
                    with c_opt3:
                        cmap_selected = st.selectbox("Colormap", ["Viridis", "Plasma"], index=0, key="csv_cmap")

                    # --- Helper for Density Plot ---
                    def make_density_plot(data_df, x_curr, y_curr, use_log, cmap_name, plot_title, x_range=None, y_range=None):
                        if len(data_df) == 0:
                            st.warning(f"No events in {plot_title}")
                            return None, None
                            
                        df_viz = data_df.copy()
                        color_c = None
                        try:
                            x_d = np.log10(df_viz[x_curr]) if use_log else df_viz[x_curr]
                            y_d = np.log10(df_viz[y_curr]) if use_log else df_viz[y_curr]
                            mask = np.isfinite(x_d) & np.isfinite(y_d)
                            x_c, y_c = x_d[mask], y_d[mask]
                            
                            if len(x_c) > 0:
                                H, xedges, yedges = np.histogram2d(x_c, y_c, bins=100)
                                x_bins = np.digitize(x_c, xedges) - 1
                                y_bins = np.digitize(y_c, yedges) - 1
                                x_bins = np.clip(x_bins, 0, H.shape[0]-1)
                                y_bins = np.clip(y_bins, 0, H.shape[1]-1)
                                df_viz['Density'] = 0.0
                                df_viz.loc[mask, 'Density'] = H[x_bins, y_bins]
                                color_c = 'Density'
                        except Exception:
                            color_c = None

                        if len(df_viz) > 5000:
                            df_p = df_viz.sample(n=5000, random_state=42)
                        else:
                            df_p = df_viz

                        f = px.scatter(df_p, x=x_curr, y=y_curr, color=color_c,
                                        title=plot_title,
                                        color_continuous_scale=cmap_name,
                                        opacity=0.8,
                                        log_x=use_log, log_y=use_log,
                                        render_mode='svg')
                        
                        if color_c:
                            f.update_traces(marker=dict(size=5))
                            f.update_layout(coloraxis_colorbar=dict(title=dict(text='Count', font=dict(color='black')), len=0.6, yanchor='bottom', y=0.0, tickfont=dict(color='black')))

                        # Apply fixed ranges if provided (ensures comparison scale)
                        layout_args = dict(
                             plot_bgcolor='white', paper_bgcolor='white', height=500,
                             margin=dict(l=40, r=40, t=60, b=40),
                             xaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', color='black'),
                             yaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', color='black'),
                             font=dict(color="black"), legend=dict(font=dict(color="black"))
                        )
                        if x_range: layout_args['xaxis']['range'] = x_range
                        if y_range: layout_args['yaxis']['range'] = y_range

                        f.update_layout(**layout_args)
                        return f, df_p

                    # --- Display Logic ---
                    has_gating_cols = ('cells' in exp.data.columns and 'singlets' in exp.data.columns)
                    is_full_data = False
                    if has_gating_cols:
                        total_n = len(exp.data)
                        # Ensure boolean
                        exp.data['cells'] = exp.data['cells'].fillna(False).astype(bool)
                        exp.data['singlets'] = exp.data['singlets'].fillna(False).astype(bool)
                        gated_n = (exp.data['cells'] & exp.data['singlets']).sum()
                        if gated_n < total_n:
                            is_full_data = True

                    if is_full_data:
                        # Calculate consistent axis ranges
                        x_series = exp.data[x_axis]
                        y_series = exp.data[y_axis]
                        range_x, range_y = None, None

                        try:
                            if is_log:
                                xp = x_series[x_series > 0]
                                yp = y_series[y_series > 0]
                                if not xp.empty:
                                    lx_min, lx_max = np.log10(xp.min()), np.log10(xp.max())
                                    pad_x = (lx_max - lx_min) * 0.05 if lx_max != lx_min else 0.1
                                    range_x = [lx_min - pad_x, lx_max + pad_x]
                                if not yp.empty:
                                    ly_min, ly_max = np.log10(yp.min()), np.log10(yp.max())
                                    pad_y = (ly_max - ly_min) * 0.05 if ly_max != ly_min else 0.1
                                    range_y = [ly_min - pad_y, ly_max + pad_y]
                            else:
                                x_min, x_max = x_series.min(), x_series.max()
                                y_min, y_max = y_series.min(), y_series.max()
                                pad_x = (x_max - x_min) * 0.05 if x_max != x_min else 1.0
                                pad_y = (y_max - y_min) * 0.05 if y_max != y_min else 1.0
                                range_x = [x_min - pad_x, x_max + pad_x]
                                range_y = [y_min - pad_y, y_max + pad_y]
                        except Exception as e:
                            # Fallback to autoscaling if range calc fails
                            pass

                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("#### Full Population")
                            fig1, df1 = make_density_plot(exp.data, x_axis, y_axis, is_log, cmap_selected, "Full Population", range_x, range_y)
                            if fig1: st.plotly_chart(fig1, use_container_width=True)
                        
                        with col2:
                            st.write("#### Gated Population")
                            df_gated = exp.data[exp.data['cells'] & exp.data['singlets']]
                            fig2, df2 = make_density_plot(df_gated, x_axis, y_axis, is_log, cmap_selected, "Gated Population (Live+Singlets)", range_x, range_y)
                            if fig2: st.plotly_chart(fig2, use_container_width=True)

                        if st.button("Save Plots to Canvas", key="btn_save_csv_density_dual"):
                            if df1 is not None:
                                mf1 = create_canvas_figure(df1, x_axis, y_axis, "Full Population", is_log=is_log, color_col='Density' if 'Density' in df1 else None)
                                add_plot_to_canvas(mf1, "Full Population")
                            if df2 is not None:
                                mf2 = create_canvas_figure(df2, x_axis, y_axis, "Gated Population", is_log=is_log, color_col='Density' if 'Density' in df2 else None)
                                add_plot_to_canvas(mf2, "Gated Population")
                            st.success("Plots added to canvas!")
                    else:
                        st.write("#### Population Density (Singlets)")
                        fig, df_p = make_density_plot(exp.data, x_axis, y_axis, is_log, cmap_selected, "Population Density")
                        if fig: st.plotly_chart(fig, use_container_width=True)
                        
                        if st.button("Save Pop Density to Canvas", key="btn_save_csv_density"):
                            mpl_fig = create_canvas_figure(df_p, x_axis, y_axis, "Processed Population Density",
                                                           is_log=is_log, color_col='Density' if 'Density' in df_p else None)
                            add_plot_to_canvas(mpl_fig, "Processed Population Density")
                            st.success("Plot added to canvas!")
                            plt.close(mpl_fig)

            st.divider()
            if st.button("Proceed to Analysis", key="btn_proceed_analysis_csv", type="primary"):
                 st.session_state.current_tab = "Analysis"
                 st.rerun()

            return

        st.write("Configuring Gating...")

        # Load Base Experiment (Clean state)
        if 'base_experiment' not in st.session_state:
            with st.spinner("Loading Experiment Data..."):
                loader = DataLoader()
                st.session_state.base_experiment = loader.load_as_cytoflow_experiment(st.session_state.loaded_files)
                
                # Check for Compensated Channels
                if st.session_state.base_experiment:
                     comp_cols = [c for c in st.session_state.base_experiment.data.columns if c.startswith("comp_")]
                     if comp_cols:
                         st.session_state.has_precomputed_compensation = True
                         st.toast(f"Detected {len(comp_cols)} compensated channels in dataset.", icon="ℹ️")

        # We work with base_experiment for channel selection
        base_exp = st.session_state.base_experiment

        if base_exp:
            # --- Step 1 & 2: Live/Dead Gating ---
            is_live_gate_ready = st.session_state.get('gate_accepted', False)

            with st.expander("Live/Dead Gating Parameters", expanded=True):
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
                # NOTE: We treat *any* detected cell-line column (even a single unique value)
                # as "Per Cell Line" mode so downstream manual gating and singlet gating follow
                # the same (more robust) code path.
                potential_cols = ['Cell Line', 'Cell_Line', 'cell_line', 'CellType', 'cell_type', 'Strain', 'strain']
                for c in potential_cols:
                    if c in base_exp.data.columns:
                        unique_vals = base_exp.data[c].dropna().astype(str).unique()
                        if len(unique_vals) >= 1:
                            cell_line_col = c
                            unique_cell_lines = sorted(unique_vals)
                            break

                is_multi_line_meta = (cell_line_col is not None)

                active_cell_line = None
                working_exp = base_exp

                if is_multi_line_meta:
                    if len(unique_cell_lines) > 1:
                        st.info(f"Detected multiple cell lines in '{cell_line_col}': {', '.join(unique_cell_lines)}")
                    else:
                        st.info(f"Detected cell line in '{cell_line_col}': {', '.join(unique_cell_lines)}")
                    gating_mode = "Per Cell Line"

                    # Selector for current cell line
                    active_cell_line = st.selectbox("Select Cell Line to Gate", unique_cell_lines, key="cell_line_sel_top")

                    # Filter Experiment for this cell line
                    import cytoflow
                    try:
                        subset_mask = base_exp.data[cell_line_col].astype(str) == str(active_cell_line)
                        subset_data = base_exp.data[subset_mask].copy()
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
                    options = ["MLP", "Density", "GMM", "DBSCAN"]
                    if gating_mode == "Mixed Population (Cluster)":
                        options = ["GMM", "DBSCAN"]

                    gating_method = st.selectbox("Gating Method", options, index=0, key=f"method{ukey}")

                    # Context for MLP
                    mlp_cell_type = "colo"
                    if gating_method == "MLP":
                        cell_type_opts = ["colo", "hek", "hug1n", "snu", "b16", "cho"]

                        # Auto-select based on active_cell_line if it matches one of the options
                        default_idx = 0
                        if active_cell_line:
                            normalized_line = str(active_cell_line).lower()
                            # 1. Exact Match
                            if normalized_line in cell_type_opts:
                                default_idx = cell_type_opts.index(normalized_line)
                            else:
                                # 2. Partial/Fuzzy Match (e.g. "hek293" -> "hek")
                                for i, opt in enumerate(cell_type_opts):
                                    if opt in normalized_line:
                                        default_idx = i
                                        break

                        mlp_cell_type = st.selectbox("Cell Line (Model Context)", cell_type_opts, index=default_idx, format_func=lambda x: x.upper(), key=f"ctype{ukey}")

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
                    elif gating_method == "GMM":
                        num_comp = st.number_input("Num Components", min_value=1, max_value=5, value=2, key=f"comp{ukey}")
                        keep_percent = 0.9 # Unused
                        sigma_val = 2.0 # Unused
                        min_samples = 50
                    else: # MLP
                        st.caption("Using Pre-trained MLP Model")
                        num_comp = 2
                        keep_percent = 0.9
                        sigma_val = 2.0
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
                'working_exp_id': id(base_exp), # Ensure we re-run if the subset/file changes
                'model_cell_type': mlp_cell_type
            }

            # Key for storing last params needs to be unique per cell line too,
            # or we just rely on the 'cell_line' field in the dict to trigger change.
            last_gating_params = st.session_state.get('last_gating_params', {})

            # Check if params changed OR if we don't have a result yet
            should_run_gating = (current_gating_params != last_gating_params) or ('gated_experiment' not in st.session_state)

            if should_run_gating:
                # Reset acceptance state so user sees new boundaries
                if 'gate_accepted' in st.session_state:
                    del st.session_state['gate_accepted']
                
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
                                                                    min_samples=min_samples,
                                                                    cell_type=mlp_cell_type)

                        # Update State
                        st.session_state.gated_experiment = gated_experiment
                        st.session_state.last_gating_params = current_gating_params
                    except Exception as e:
                        st.error(f"Gating failed: {e}")

            # If we have a gated experiment, show it (persisting the view)
            if 'gated_experiment' in st.session_state:
                st.divider()
                st.subheader("3. Live Gating Verification")

                exp = st.session_state.gated_experiment
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

                    if gating_mode == "Multiple Cell Lines" and "_temp_live_labels" in df.columns:
                        color_col = "_temp_live_labels"
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
                        fig.update_layout(coloraxis_colorbar=dict(
                            title=dict(text="Count", font=dict(color="black")),
                            tickfont=dict(color="black")
                        ))

                    # Improved Plot Layout (White Background)
                    fig.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        xaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', color='black', tickfont=dict(color='black'), title_font=dict(color='black')),
                        yaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', color='black', tickfont=dict(color='black'), title_font=dict(color='black')),
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
                                                    line=dict(color='red', width=4),
                                                    name=f"Saved Gate ({active_cell_line})"))

                    # --- Auto-Gating Visualization (Centroids & Ellipses) ---
                    # ALWAYS overlay Auto-Gating if available
                    # Optimization: Do not re-compute contour if gate is already accepted (User Request)
                    # For Per-Cell-Line mode, we only skip if the GLOBAL gating is complete.
                    if '_temp_live_labels' in df.columns and not st.session_state.get('gate_accepted'):
                        unique_labels = df['_temp_live_labels'].unique()

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

                            subset = df[df['_temp_live_labels'] == label]
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
                                    subset = df[df['_temp_live_labels'] == lbl]
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
                                        line_color = 'red'
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

                        else:
                            st.warning("No distinct clusters found! All points were classified as noise (-1) or fell below the 1% size threshold. Try increasing 'Epsilon' or decreasing 'Min Samples'.")

                    # --- Validation Plot (Live vs Non-Live) ---
                    # User Request: "separate scatter plot below... where live cells and non-live cells are colored"
                    if '_temp_live_labels' in df.columns:
                        st.write("### Gating Classification")

                        # Create validation dataframe (downsampled if needed)
                        # We use the same plot_df for consistency
                        # We explicitly convert the gate column to string to ensure discrete coloring
                        val_df = plot_df.copy()
                        val_df['_temp_live_labels'] = val_df['_temp_live_labels'].astype(str)

                        # Map to meaningful labels
                        live_lbl = str(st.session_state.get('suggested_live_label', '1'))

                        def get_label_desc(val):
                            return "Live Cells" if str(val) == live_lbl else "Debris / Dead"

                        val_df['Cond_Label'] = val_df['_temp_live_labels'].apply(get_label_desc)

                        # Define discrete color map
                        color_map_val = {
                            "Live Cells": "green",
                            "Debris / Dead": "red"
                        }

                        fig_val = px.scatter(val_df, x=x_channel, y=y_channel,
                                            color='Cond_Label',
                                            title=f"{gating_method} Classification (Points) vs. Computed Gate (Line)",
                                            color_discrete_map=color_map_val,
                                            opacity=0.6,
                                            log_x=is_log, log_y=is_log,
                                            render_mode='svg')

                        # White background for validation plot too
                        fig_val.update_layout(
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            xaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', color='black', tickfont=dict(color='black'), title_font=dict(color='black')),
                            yaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', color='black', tickfont=dict(color='black'), title_font=dict(color='black')),
                            font=dict(color="black"),
                            legend=dict(
                                font=dict(color="black"),
                                title=dict(font=dict(color="black"))
                            )
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

                        if st.session_state.get('developer_mode', False):
                            st.plotly_chart(fig_val, width="stretch")

                            # Add to Canvas button for Classification Plot
                            canvas_title_val = f"{gating_method} Classification Validation"
                            if st.button(f"Add \"{canvas_title_val}\" to Canvas", key=f"add_val_gate_to_canvas{ukey}"):
                                boundaries_to_plot_val = []
                                if boundary_x is not None:
                                    boundaries_to_plot_val.append({'vertices': list(zip(boundary_x, boundary_y)), 'color': 'black', 'width': 2})

                                mpl_fig_val = create_canvas_figure(
                                    val_df, x_channel, y_channel,
                                    canvas_title_val,
                                    is_log=is_log,
                                    color_col='Cond_Label',
                                    color_map=color_map_val,
                                    boundaries=boundaries_to_plot_val
                                )
                                add_plot_to_canvas(mpl_fig_val, canvas_title_val)
                                st.success("Plot added to canvas!")
                                plt.close(mpl_fig_val)


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
                    # Generate a robust key based on params
                    import hashlib
                    param_str = str(sorted(current_gating_params.items()))
                    param_hash = hashlib.md5(param_str.encode()).hexdigest()
                    
                    selection = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points" if drag_mode=="pan" else "box", key=f"live_gating_plot_{param_hash}")

                    if st.session_state.get('developer_mode', False):
                        canvas_title = "Live/Dead Gating"
                        if active_cell_line:
                            canvas_title += f" ({active_cell_line})"
                        if st.button(f"Add \"{canvas_title}\" to Canvas", key=f"add_live_gate_to_canvas{ukey}"):
                            boundaries_to_plot = []
                            if active_cell_line and 'cell_line_gates' in st.session_state and active_cell_line in st.session_state.cell_line_gates:
                                saved_gate = st.session_state.cell_line_gates[active_cell_line]
                                if saved_gate.get('type') == 'polygon':
                                    boundaries_to_plot.append({'vertices': saved_gate['vertices'], 'color': 'black', 'width': 2})
                            if boundary_x is not None:
                                boundaries_to_plot.append({'vertices': list(zip(boundary_x, boundary_y)), 'color': 'black', 'width': 2})
                            if st.session_state.get('manual_vertices'):
                                boundaries_to_plot.append({'vertices': st.session_state.manual_vertices, 'color': 'yellow', 'width': 2})

                            mpl_fig = create_canvas_figure(plot_df, x_channel, y_channel, canvas_title,
                                                        is_log=is_log, color_col=color_col if continuous_color else None,
                                                        boundaries=boundaries_to_plot, cmap=cmap_selected.lower())
                            add_plot_to_canvas(mpl_fig, canvas_title)
                            st.success("Plot added to canvas!")
                            plt.close(mpl_fig)

                    # Active Learning Controls (Developer Mode Only)
                    if st.session_state.get('developer_mode', False):
                        render_uncertainty_controls(exp, x_channel, y_channel)
                        render_labeling_controls(exp, x_channel, y_channel, selection, plot_df=plot_df)
                        render_reset_controls()

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
                        
                        # Check acceptance state
                        is_accepted = False
                        if active_cell_line:
                            is_accepted = (active_cell_line in st.session_state.get('cell_line_gates', {}))
                        else:
                            is_accepted = st.session_state.get('gate_accepted', False)

                        # Layout: Accept | Reject | Add to Canvas
                        # We use 3 columns now
                        col1, col2, col3 = st.columns([1, 1, 1])

                        btn_label = "✅ Accept Auto-Gate"
                        if active_cell_line:
                            btn_label = f"✅ Accept Gate for {active_cell_line}"
                        
                        if is_accepted:
                            btn_label = "Gate Applied (Reset to Change)"

                        if col1.button(btn_label, disabled=is_accepted, key=f"accept_btn_{ukey}"):
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
                                
                                # Calculate stats for this line
                                # working_exp is the subset for this line.
                                # If we apply the gate, how many are left?
                                line_total = len(working_exp.data)
                                line_kept = 0
                                
                                if gate_data.get('type') == 'polygon':
                                    from matplotlib.path import Path
                                    p = Path(gate_data['vertices'])
                                    ch = gate_data['channels']
                                    pts = working_exp.data[ch].values
                                    line_kept = np.sum(p.contains_points(pts))
                                elif gate_data.get('type') == 'label':
                                    lbl = gate_data['label']
                                    # Ensure column exists and match
                                    if '_temp_live_labels' in working_exp.data.columns:
                                        line_kept = np.sum(working_exp.data['_temp_live_labels'].astype(str) == str(lbl))
                                
                                gate_data['stats_kept'] = int(line_kept)
                                gate_data['stats_total'] = int(line_total)

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
                                    
                                    # Calculate aggregation stats for all cell lines
                                    total_kept = 0
                                    total_input = 0
                                    
                                    # We need to sum up kept/total for each line
                                    # Since we don't have the "kept" count stored, we might need a rough estimate or 
                                    # wait for the "Final Assembly" export step to count exactly.
                                    # However, user wants feedback NOW.
                                    # Let's iterate and calculate based on stored gates.
                                    
                                    # Re-calculating exactly is expensive. 
                                    # Let's try to do it efficiently or store it incrementally.
                                    # Better approach: Store stats in cell_line_gates when saving.
                                    
                                    # If not stored, we display generic message.
                                    has_stats = all('stats_kept' in g for g in st.session_state.cell_line_gates.values())
                                    if has_stats:
                                        total_kept = sum(g['stats_kept'] for g in st.session_state.cell_line_gates.values())
                                        total_input = sum(g['stats_total'] for g in st.session_state.cell_line_gates.values())
                                        pct = (total_kept / total_input * 100) if total_input > 0 else 0
                                        st.session_state.live_gate_stats = f"{total_kept} / {total_input} ({pct:.1f}%)"
                                    else:
                                        st.session_state.live_gate_stats = "Multi-Line Gated"

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
                                exp = gater.apply_manual_gate(base_exp, "_temp_live_mask_contour", x_channel, y_channel, verts)

                                # Filter to keep only the cell inside the contour
                                if "_temp_live_mask_contour" in exp.data.columns:
                                     # Filter in-place
                                     exp.data = exp.data[exp.data["_temp_live_mask_contour"]].copy()

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
                                    'channel': '_temp_live_labels'
                                }

                                exp = st.session_state.gated_experiment

                                # Filter to keep only the Live Label population
                                # We modify the experiment in-place for the session
                                filtered_df = exp.data[exp.data['_temp_live_labels'] == label].copy()
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


                        if col2.button("❌ Reject / Manual Gate", key=f"reject_btn_{ukey}", disabled=is_accepted):
                            st.session_state.show_manual_gating = True
                        
                        # Persistent Add to Canvas Button
                        ctitle = "Live/Dead Gating"
                        if active_cell_line: ctitle += f" ({active_cell_line})"
                        
                        if col3.button("Add Plot to Canvas", key=f"add_canvas_persist_{ukey}"):
                            # Re-use canvas logic
                             boundaries_to_plot = []
                             if active_cell_line and 'cell_line_gates' in st.session_state and active_cell_line in st.session_state.cell_line_gates:
                                saved_gate = st.session_state.cell_line_gates[active_cell_line]
                                if saved_gate.get('type') == 'polygon':
                                    boundaries_to_plot.append({'vertices': saved_gate['vertices'], 'color': 'black', 'width': 2})
                             if boundary_x is not None:
                                boundaries_to_plot.append({'vertices': list(zip(boundary_x, boundary_y)), 'color': 'black', 'width': 2})
                             if st.session_state.get('manual_vertices'):
                                boundaries_to_plot.append({'vertices': st.session_state.manual_vertices, 'color': 'yellow', 'width': 2})

                             mpl_fig = create_canvas_figure(plot_df, x_channel, y_channel, ctitle,
                                                        is_log=is_log, color_col=color_col if continuous_color else None,
                                                        boundaries=boundaries_to_plot, cmap=cmap_selected.lower())
                             add_plot_to_canvas(mpl_fig, ctitle)
                             st.success("Plot added to canvas!")
                             plt.close(mpl_fig)

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
                            current_poly_str = f"1000000, 1000000\n100000000, 1000000\n100000000, 100000000\n1000000, 100000000"

                        vertices_text = st.text_area("Polygon Vertices (x, y per line)", current_poly_str, height=150, key="manual_live_verts_input")

                        if st.button("Apply Manual Gate"):
                            # Parse
                            try:
                                vertices = []
                                for line in vertices_text.strip().split('\n'):
                                    p = line.split(',')
                                    vertices.append((float(p[0].strip()), float(p[1].strip())))

                                gater = GatingManager()

                                # Apply manual gate in the same "Per Cell Line" path when cell-line metadata exists
                                # (even if only one line), so Singlet Detection can reliably re-apply the saved gate.
                                total_cells = len(base_exp.data)
                                manual_target_exp = base_exp
                                active_for_log = None

                                if is_multi_line_meta and active_cell_line:
                                    active_for_log = active_cell_line
                                    try:
                                        import cytoflow
                                        subset_mask = base_exp.data[cell_line_col].astype(str) == str(active_cell_line)
                                        subset_data = base_exp.data[subset_mask].copy()
                                        subset_data.reset_index(drop=True, inplace=True)
                                        total_cells = len(subset_data)
                                        
                                        # Create temp experiment for subset
                                        new_metadata = {}
                                        for col in subset_data.columns:
                                            if col in base_exp.metadata:
                                                new_metadata[col] = base_exp.metadata[col]
                                            else:
                                                 if subset_data[col].dtype.name in ['object', 'category']:
                                                     new_metadata[col] = {'type': 'condition', 'class': 'category'}
                                                 else:
                                                     new_metadata[col] = {'type': 'channel'}
                                        manual_target_exp = cytoflow.Experiment(data=subset_data, metadata=new_metadata)
                                    except Exception:
                                        manual_target_exp = base_exp

                                # Apply to chosen experiment (subset or full)
                                manual_exp = gater.apply_manual_gate(manual_target_exp, "_temp_live_mask_manual", x_channel, y_channel, vertices)

                                # Store params for Export
                                if is_multi_line_meta and active_cell_line:
                                    if 'cell_line_gates' not in st.session_state:
                                        st.session_state.cell_line_gates = {}
                                    
                                    # Stats calculation
                                    m_kept = 0
                                    if "_temp_live_mask_manual" in manual_exp.data.columns:
                                        m_kept = manual_exp.data["_temp_live_mask_manual"].sum()
                                    
                                    st.session_state.cell_line_gates[active_cell_line] = {
                                        'type': 'polygon',
                                        'vertices': vertices,
                                        'channels': [x_channel, y_channel],
                                        'stats_kept': int(m_kept),
                                        'stats_total': int(total_cells)
                                    }
                                    st.session_state.live_gate_params = {
                                        'type': 'per_cell_line',
                                        'gates': st.session_state.cell_line_gates,
                                        'col': cell_line_col
                                    }
                                    
                                    # Aggregate all stats
                                    t_kept = sum(g.get('stats_kept', 0) for g in st.session_state.cell_line_gates.values())
                                    t_total = sum(g.get('stats_total', 0) for g in st.session_state.cell_line_gates.values())
                                    if t_total > 0:
                                        st.session_state.live_gate_stats = f"{t_kept} / {t_total} ({(t_kept/t_total)*100:.1f}%)"
                                else:
                                    st.session_state.live_gate_params = {
                                        'type': 'polygon',
                                        'vertices': vertices,
                                        'channels': [x_channel, y_channel]
                                    }

                                # Filter manually to match Auto behavior
                                if "_temp_live_mask_manual" in manual_exp.data.columns:
                                     manual_exp.data = manual_exp.data[manual_exp.data["_temp_live_mask_manual"]].copy()

                                kept_cells = len(manual_exp.data)
                                if kept_cells == 0:
                                    st.warning('Selected gate contains 0 cells. Please select a valid region containing cells.')
                                    st.stop()

                                percent = (kept_cells / total_cells) * 100 if total_cells > 0 else 0
                                st.session_state.live_gate_stats = f"{kept_cells} / {total_cells} ({percent:.1f}%)"

                                st.session_state.gated_experiment = manual_exp
                                st.session_state.manual_vertices = vertices # Save for visualization

                                # Determine Cell Line for logging
                                current_cell_line = active_for_log or "Unknown"
                                if not active_for_log and st.session_state.get('loaded_files'):
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
                                st.session_state.show_manual_gating = False
                                st.rerun()

                            except Exception as e:
                                st.error(f"Invalid format: {e}")

                    # --- Step 3: Singlet Detection ---
                    if st.session_state.get('gate_accepted'):
                        st.divider()

                        # Stats Display (Moved here per user request)
                        if st.session_state.get('live_gate_stats'):
                            st.success(f"Final Live Cells: {st.session_state.live_gate_stats}")

                        st.subheader("4. Singlet Detection")

                        # Automatically proceed to Singlet Detection
                        is_doublet_mode = True # not st.session_state.get('doublets_removed', False)
                        st.session_state.show_doublet_gating = is_doublet_mode

                        # --- Per Cell Line Mode Detection for Singlets ---
                        active_line_singlets = None
                        if is_multi_line_meta:
                            if len(unique_cell_lines) > 1:
                                st.info("Detected multiple cell lines. Process singlets per cell line.")
                            else:
                                st.info("Processing singlets for detected cell line.")

                            # Selector for current cell line (independent from Step 1)
                            active_line_singlets = st.selectbox("Select Cell Line for Singlet Gating", unique_cell_lines, key="singlet_cell_line_sel")

                            # Filter Base Experiment for this cell line
                            subset_mask = base_exp.data[cell_line_col].astype(str) == str(active_line_singlets)
                            subset_data = base_exp.data[subset_mask].copy()
                            # subset_data.reset_index(drop=True, inplace=True) # REMOVED to preserve index alignment

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
                                    # Re-run auto gating to filter
                                    gp = gate['auto_params']

                                    # Need a temporary experiment wrapper for this subset
                                    import cytoflow
                                    new_metadata = {}
                                    for col in subset_data.columns:
                                        if col in base_exp.metadata:
                                            new_metadata[col] = base_exp.metadata[col]
                                        else:
                                            # Minimal fallback
                                            if subset_data[col].dtype.name in ['object', 'category']:
                                                new_metadata[col] = {'type': 'condition', 'class': 'category'}
                                            else:
                                                new_metadata[col] = {'type': 'channel'}

                                    temp_exp = cytoflow.Experiment(data=subset_data, metadata=new_metadata)

                                    gater = GatingManager()
                                    try:
                                        res_exp = gater.auto_gate_live_cells(
                                            temp_exp,
                                            x_channel=gp['x'],
                                            y_channel=gp['y'],
                                            method=gp['method'].lower(),
                                            num_components=gp['comp'],
                                            keep=gp['keep'],
                                            sigma=gp['sigma'],
                                            eps=gp['eps'],
                                            min_samples=gp['min_samples'],
                                            use_outlier_removal=gp['use_outliers'],
                                            cell_type=gp.get('model_cell_type', 'colo')
                                        )
                                        # Now filter
                                        target_label = gate['label']
                                        # Ensure type match (string vs int/other)
                                        if '_temp_live_labels' in res_exp.data.columns:
                                            res_exp.data['_temp_live_labels'] = res_exp.data['_temp_live_labels'].astype(str)
                                            df_gated = res_exp.data[res_exp.data['_temp_live_labels'] == str(target_label)].copy()
                                        else:
                                             df_gated = subset_data

                                    except Exception as e:
                                        st.error(f"Failed to reconstruct gate for {active_line_singlets}: {e}")
                                        df_gated = subset_data
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
                            # Resolved default vs session state for singlet method
                            # --- MANUAL MODE OVERRIDE ---
                            is_manual_active = st.session_state.get('doublet_manual_mode', False)

                            if is_manual_active:
                                st.info("Manual Mode Active")
                                # Define viz_df (Downsampled)
                                subset_n = 5000
                                viz_df = viz_df_full.sample(subset_n, random_state=42) if len(viz_df_full) > subset_n else viz_df_full

                                # Visualization (Scatter)
                                fig_m = go.Figure()
                                if d_is_log:
                                    x_p = np.log10(viz_df[d_x] + 1)
                                    y_p = np.log10(viz_df[d_y] + 1)
                                else:
                                    x_p = viz_df[d_x]
                                    y_p = viz_df[d_y]

                                fig_m.add_trace(go.Scatter(
                                    x=x_p, y=y_p, mode='markers',
                                    marker=dict(size=4, color='#0d0887', opacity=0.6), name='Events'
                                ))

                                # Visualise Applied Gate
                                if "singlet_gate_params" in st.session_state and st.session_state.singlet_gate_params.get("type") == "polygon":
                                    apv = st.session_state.singlet_gate_params["vertices"]
                                    apx_ = [v[0] for v in apv] + [apv[0][0]]
                                    apy_ = [v[1] for v in apv] + [apv[0][1]]
                                    if d_is_log:
                                        apx_ = np.log10(np.array(apx_) + 1)
                                        apy_ = np.log10(np.array(apy_) + 1)

                                    fig_m.add_trace(go.Scatter(x=apx_, y=apy_, mode="lines",
                                                            line=dict(color="lime", width=3, dash="dash"), name="Applied Manual Gate"))

                                # Visualise Pending Selection
                                if st.session_state.get('pending_doublet_verts'):
                                    pv = st.session_state.pending_doublet_verts
                                    # Close loop for viz
                                    pv_plot = pv + [pv[0]]
                                    pvx = [p[0] for p in pv_plot]
                                    pvy = [p[1] for p in pv_plot]

                                    if d_is_log:
                                        pvx = np.log10(np.array(pvx) + 1)
                                        pvy = np.log10(np.array(pvy) + 1)

                                    fig_m.add_trace(go.Scatter(x=pvx, y=pvy, mode='lines+markers',
                                                            line=dict(color='#f0f921', width=3), name="Manual Selection"))

                                fig_m.update_layout(
                                    title="Manual Singlet Gating",
                                    xaxis_title=d_x + (" (Log)" if d_is_log else ""),
                                    yaxis_title=d_y + (" (Log)" if d_is_log else ""),
                                    plot_bgcolor="white", paper_bgcolor="white",
                                    font=dict(color="black"),
                                    xaxis=dict(
                                        color="black",
                                        tickfont=dict(color="black"),
                                        title_font=dict(color="black"),
                                        showgrid=True,
                                        gridcolor="lightgray"
                                    ),
                                    yaxis=dict(
                                        color="black",
                                        tickfont=dict(color="black"),
                                        title_font=dict(color="black"),
                                        showgrid=True,
                                        gridcolor="lightgray"
                                    ),
                                    legend=dict(
                                        title=dict(font=dict(color="black")),
                                        font=dict(color="black")
                                    ),
                                    dragmode="pan"
                                )

                                # Render Plot
                                sel_m = st.plotly_chart(fig_m, width="stretch", on_select="rerun", selection_mode="points", key="singlet_manual_plot")

                                if st.button("Add Plot to Canvas", key="add_canvas_manual_singlet"):
                                    # Create MPL Figure for Canvas (Support Download SVG)
                                    b_list = []
                                    # 1. Applied Gate
                                    if "singlet_gate_params" in st.session_state and st.session_state.singlet_gate_params.get("type") == "polygon":
                                        b_list.append({
                                            "vertices": st.session_state.singlet_gate_params["vertices"],
                                            "color": "lime",
                                            "width": 2
                                        })
                                    # 2. Pending Selection
                                    if st.session_state.get("pending_doublet_verts"):
                                         b_list.append({
                                            "vertices": st.session_state.pending_doublet_verts,
                                            "color": "#f0f921",
                                            "width": 2
                                        })

                                    # Use fake column for purple color
                                    v_df_c = viz_df.copy()
                                    v_df_c["Events_"] = "Events"
                                    # Purple #0d0887

                                    mpl_fig_m = create_canvas_figure(
                                        v_df_c, d_x, d_y, "Manual Singlet Gating",
                                        is_log=d_is_log,
                                        color_col="Events_",
                                        color_map={"Events": "#0d0887"},
                                        boundaries=b_list
                                    )
                                    add_plot_to_canvas(mpl_fig_m, "Manual Singlet Gating")
                                    st.success("Plot added to canvas!")
                                    plt.close(mpl_fig_m)

                                # Selection Logic
                                if sel_m and sel_m.get("selection"):
                                    pts = sel_m["selection"]["points"]
                                    if len(pts) > 0:
                                        sx = np.array([p["x"] for p in pts])
                                        sy = np.array([p["y"] for p in pts])

                                        if d_is_log:
                                            sx = 10**sx - 1
                                            sy = 10**sy - 1

                                        candidates = list(zip(sx, sy))
                                        current_verts = st.session_state.get('pending_doublet_verts', [])
                                        updated_verts = list(current_verts)

                                        changed = False
                                        for c in candidates:
                                            is_close = False
                                            for ex in current_verts:
                                                 if np.isclose(c[0], ex[0], atol=1e-5) and np.isclose(c[1], ex[1], atol=1e-5):
                                                     is_close = True
                                                     break
                                            if not is_close:
                                                updated_verts.append(c)
                                                changed = True

                                        if changed:
                                            if len(updated_verts) >= 3:
                                                arr_u = np.array(updated_verts)
                                                cx, cy = np.mean(arr_u[:,0]), np.mean(arr_u[:,1])
                                                angs = np.arctan2(arr_u[:,1]-cy, arr_u[:,0]-cx)
                                                idxs = np.argsort(angs)
                                                updated_verts = [updated_verts[i] for i in idxs]

                                            st.session_state.pending_doublet_verts = updated_verts
                                            st.session_state.doublet_verts_input = "\n".join([f"{v[0]}, {v[1]}" for v in updated_verts])
                                            st.rerun()

                                # Manual Interface
                                st.info("Manual Mode: Click points to define polygon \nor enter vertices coordinates directly.")
                                current_poly_str_d = ""
                                if st.session_state.get('pending_doublet_verts'):
                                     current_poly_str_d = "\n".join([f"{v[0]}, {v[1]}" for v in st.session_state.pending_doublet_verts])
                                else:
                                     current_poly_str_d = "1000000, 1000000\n100000000, 1000000\n100000000, 100000000\n1000000, 100000000"

                                def _sync_text_to_plot_d():
                                     txt = st.session_state.doublet_verts_input
                                     if txt:
                                         p_verts = []
                                         try:
                                             for line in txt.strip().split("\n"):
                                                 parts = line.split(",")
                                                 if len(parts) >= 2:
                                                     p_verts.append((float(parts[0].strip()), float(parts[1].strip())))
                                             if len(p_verts) > 0:
                                                 st.session_state.pending_doublet_verts = p_verts
                                         except Exception:
                                             pass

                                vertices_text_d = st.text_area("Polygon Vertices", current_poly_str_d, height=150, key="doublet_verts_input", on_change=_sync_text_to_plot_d)

                                col_m1, col_m2, col_m3 = st.columns(3)
                                if col_m1.button("✅ Apply Manual Gate", key=f"confirm_manual_singlets{sukey}"):
                                      # Apply logic
                                      parsed_verts = []
                                      try:
                                          lines = vertices_text_d.strip().split('\n')
                                          for line in lines:
                                              parts = line.split(',')
                                              if len(parts) >= 2:
                                                  parsed_verts.append((float(parts[0].strip()), float(parts[1].strip())))

                                          if len(parsed_verts) < 3:
                                              st.error("Need 3+ points")
                                          else:
                                              # Save params
                                              st.session_state.singlet_gate_params = {
                                                  'type': 'polygon',
                                                  'vertices': parsed_verts,
                                                  'channels': [d_x, d_y]
                                              }

                                              # Apply Gate
                                              from glow.gating.geometry import points_in_polygon

                                              # Linear Check
                                              X_lin_full = df_gated[[d_x, d_y]].values
                                              mask = points_in_polygon(X_lin_full, parsed_verts)

                                              kept = sum(mask)
                                              total = len(mask)
                                              st.session_state.singlet_gate_stats = f"{kept} / {total} ({(kept/total)*100:.1f}%)"

                                              exp.data["_temp_singlet_mask_auto"] = False
                                              exp.data.loc[df_gated.index[mask], "_temp_singlet_mask_auto"] = True
                                              st.session_state.gated_experiment = exp
                                              st.session_state.doublets_removed = True
                                              # Keep enable to see result
                                              st.success(f"Manual Gate Applied! {kept} events.")
                                              st.rerun()

                                      except Exception as e:
                                          st.error(f"Error: {e}")

                                if col_m2.button("Undo Last Point", key="undo_pt"):
                                    curr = st.session_state.get('pending_doublet_verts', [])
                                    if curr:
                                        st.session_state.pending_doublet_verts = curr[:-1]
                                        st.rerun()

                                if col_m3.button("Exit Manual Mode", key="exit_manual_s"):
                                    st.session_state.doublet_manual_mode = False
                                    st.rerun()

                            else:
                                s_key = f"s_method{ukey}"
                                if s_key not in st.session_state:
                                    st.session_state[s_key] = "MLP (Auto-Detection)"
                                singlet_method = st.radio("Singlet Method", ["GMM (Auto-Ellipse)", "MLP (Auto-Detection)"], horizontal=True, key=s_key)
                                if "MLP" in singlet_method:
                                    try:
                                        learner = get_active_learner("singlet")
                                        if learner:
                                            X_lin = df_gated[[d_x, d_y]].values
                                            X_pred = np.log10(X_lin + 1) if np.max(X_lin) > 10 else X_lin
                                            c_type = str(active_line_singlets).lower() if active_line_singlets else "colo"
                                            # Context
                                            ctx_ids = np.zeros(len(X_pred), dtype=int)
                                            ctx_types = np.array([c_type] * len(X_pred))
                                            preds = learner.model.predict(X_pred, dataset_ids=ctx_ids, cell_types=ctx_types)
                                            # Viz
                                            subset_n = 5000
                                            viz_df = viz_df_full.sample(subset_n, random_state=42) if len(viz_df_full) > subset_n else viz_df_full
                                            v_lin = viz_df[[d_x, d_y]].values
                                            v_pred_in = np.log10(v_lin + 1) if np.max(v_lin) > 10 else v_lin
                                            v_ids = np.zeros(len(v_lin), dtype=int)
                                            v_types = np.array([c_type] * len(v_lin))
                                            viz_df["Singlet"] = learner.model.predict(v_pred_in, dataset_ids=v_ids, cell_types=v_types)
                                            # Use categorical column for Plotly to ensure discrete colors
                                            viz_df["Singlet_Cat"] = viz_df["Singlet"].apply(lambda x: "Singlet" if x==1 else "Doublet")

                                            # Plasma Colors: Doublet (0) -> Purple (#0d0887), Singlet (1) -> Yellow (#f0f921)
                                            color_map_singlet = {"Doublet": "#0d0887", "Singlet": "#f0f921"}

                                            fig_mlp = px.scatter(viz_df, x=d_x, y=d_y, color="Singlet_Cat", title="Singlet Detection",
                                                               color_discrete_map=color_map_singlet, opacity=0.6,
                                                               log_x=d_is_log, log_y=d_is_log, render_mode="svg")
                                            fig_mlp.update_layout(
                                                plot_bgcolor="white",
                                                paper_bgcolor="white",
                                                dragmode="pan",
                                                clickmode="event+select",
                                                legend_title_text="Prediction",
                                                font=dict(color="black"),
                                                xaxis=dict(
                                                    color="black",
                                                    tickfont=dict(color="black"),
                                                    title_font=dict(color="black"),
                                                    showgrid=True,
                                                    gridcolor="lightgray"
                                                ),
                                                yaxis=dict(
                                                    color="black",
                                                    tickfont=dict(color="black"),
                                                    title_font=dict(color="black"),
                                                    showgrid=True,
                                                    gridcolor="lightgray"
                                                ),
                                                legend=dict(
                                                    title=dict(text="Prediction", font=dict(color="black")),
                                                    font=dict(color="black")
                                                )
                                            )
                                            unc_trace = get_uncertainty_trace(st.session_state.gated_experiment if "gated_experiment" in st.session_state else base_exp, d_x, d_y)
                                            if unc_trace: fig_mlp.add_trace(unc_trace)
                                            selection = st.plotly_chart(fig_mlp, width="stretch", on_select="rerun", selection_mode="box", key="mlp_singlet_plot")

                                            canvas_title_s = "Singlet Detection"
                                            if active_line_singlets:
                                                canvas_title_s += f" ({active_line_singlets})"

                                            if st.button(f"Add '{canvas_title_s}' to Canvas", key=f"add_singlet_mlp_gate_{sukey}"):
                                                mpl_fig = create_canvas_figure(viz_df, d_x, d_y, canvas_title_s, is_log=d_is_log, color_col="Singlet_Cat", color_map=color_map_singlet)
                                                add_plot_to_canvas(mpl_fig, canvas_title_s)
                                                st.success("Plot added to canvas!")
                                                plt.close(mpl_fig)
                                            if st.session_state.get('developer_mode', False):
                                                render_uncertainty_controls(st.session_state.gated_experiment if "gated_experiment" in st.session_state else base_exp, d_x, d_y, model_type="singlet")
                                                render_labeling_controls(st.session_state.gated_experiment if "gated_experiment" in st.session_state else base_exp, d_x, d_y, selection, plot_df=viz_df, model_type="singlet")
                                                render_reset_controls("singlet")
                                            st.divider()
                                            col_ma, col_mr = st.columns(2)

                                            def _reject_s_callback():
                                                # st.session_state[f"s_method{ukey}"] = "GMM (Auto-Ellipse)"
                                                st.session_state.doublet_manual_mode = True

                                            col_mr.button("❌ Reject / Manual Gate", key=f"reject_mlp_{ukey}", on_click=_reject_s_callback)

                                            if col_ma.button(f"✅ Accept Singlets for {c_type.upper()}", key=f"accept_mlp_{ukey}"):
                                                st.session_state.singlet_gate_params = {"type": "mlp", "diff_channels": [d_x, d_y], "cell_type": c_type}
                                                mask_full = (preds == 1)
                                                kept, total = sum(mask_full), len(preds)
                                                st.session_state.singlet_gate_stats = f"{kept} / {total} ({(kept/total)*100:.1f}%)"

                                                # Label only, do not remove rows
                                                exp.data["_temp_singlet_mask_auto"] = False
                                                exp.data.loc[df_gated.index[mask_full], "_temp_singlet_mask_auto"] = True

                                                st.session_state.gated_experiment = exp
                                                st.session_state.doublets_removed = True
                                                # st.session_state.show_doublet_gating = False
                                                st.success(f"MLP Gate Applied. Labeled {kept} singlets (IsSinglet=True).")
                                                st.rerun()
                                        else:
                                            st.error("Failed to load MLP model.")
                                    except Exception as e:
                                        st.error(f"MLP Error: {e}")
                                else:
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
                                        # Fast density via histogram (Count)
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
                                                colorbar=dict(
                                                    title=dict(text='Density', font=dict(color='black')),
                                                    tickfont=dict(color='black')
                                                )
                                            ),
                                            name='Events',
                                            # Ensure this trace handles selection
                                            selected=dict(marker=dict(color='orange', size=6)), # Visual feedback on selection
                                            unselected=dict(marker=dict(opacity=0.5))
                                        ))

                                        # Auto-Gate Ellipse -> Converted to 5-Point Polygon (Pentagon)
                                        # User Req: "auto selection for singlets should be 5 points connected by lines"
                                        theta_poly = np.linspace(0, 2*np.pi, 6) # 6 points to close the loop (5 sides)
                                        ell_x_poly = d_mean[0] + 2.5 * lambda_d[0] * v_d[0, 0] * np.cos(theta_poly) + 2.5 * lambda_d[1] * v_d[0, 1] * np.sin(theta_poly)
                                        ell_y_poly = d_mean[1] + 2.5 * lambda_d[0] * v_d[1, 0] * np.cos(theta_poly) + 2.5 * lambda_d[1] * v_d[1, 1] * np.sin(theta_poly)

                                        ell_x_plot = np.log10(ell_x_poly + 1) if d_is_log else ell_x_poly
                                        ell_y_plot = np.log10(ell_y_poly + 1) if d_is_log else ell_y_poly

                                        # Draw the Auto-Gate
                                        fig_d.add_trace(go.Scatter(x=ell_x_plot, y=ell_y_plot, mode='lines+markers',
                                                                line=dict(color='black', width=3), # User Request: Black full line
                                                                marker=dict(size=6, color='black'),
                                                                name="Auto Singlets"))

                                        # Applied Manual Gate (Persistence)
                                        if "singlet_gate_params" in st.session_state and st.session_state.singlet_gate_params.get("type") == "polygon":
                                            apv = st.session_state.singlet_gate_params["vertices"]
                                            apx_ = [v[0] for v in apv] + [apv[0][0]]
                                            apy_ = [v[1] for v in apv] + [apv[0][1]]
                                            if d_is_log:
                                                apx_ = np.log10(np.array(apx_) + 1)
                                                apy_ = np.log10(np.array(apy_) + 1)
                                            fig_d.add_trace(go.Scatter(x=apx_, y=apy_, mode="lines+markers",
                                                                    line=dict(color="lime", width=3, dash="dash"), name="Applied Manual Gate"))

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
                                                                        line=dict(color='#f0f921', width=3), name="Manual Selection"))

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
                                            dragmode="pan",
                                            clickmode="event+select",
                                            plot_bgcolor="white",
                                            paper_bgcolor="white",
                                            font=dict(color="black"),
                                            xaxis=dict(color="black", tickfont=dict(color="black"), title_font=dict(color="black"), showgrid=True, gridcolor="lightgray"),
                                            yaxis=dict(color="black", tickfont=dict(color="black"), title_font=dict(color="black"), showgrid=True, gridcolor="lightgray"),
                                            legend=dict(
                                                font=dict(color="black"),
                                                orientation="h",
                                                yanchor="bottom",
                                                y=1.02,
                                                xanchor="right",
                                                x=1
                                            ),
                                            # height=500, # Commented out to match MLP
                                            # xaxis=dict(range=...), # Commented out to match MLP auto-range
                                            uirevision="constant"
                                        )
                                        # fig_d.update_xaxes(showgrid=True, gridcolor='rgba(0,0,0,0.1)', linecolor='black', zeroline=False)
                                        # fig_d.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.1)', linecolor='black', zeroline=False)

                                        # Selection Interaction
                                        # User req for Shift selection is handled by Plotly standard behavior with clickmode

                                        sel_d = st.plotly_chart(fig_d, width="stretch", on_select="rerun", selection_mode="points", key="doublet_chart")

                                        canvas_title_gmm = "Singlet Gating (GMM)"
                                        if active_line_singlets:
                                            canvas_title_gmm += f" ({active_line_singlets})"

                                        if st.button(f"Add '{canvas_title_gmm}' to Canvas", key=f"add_singlet_gmm_gate_{sukey}"):
                                            viz_df_canvas = viz_df.copy()
                                            c_col = None # Force consistent Blue color (User Request)

                                            # Removed density calc here to ensure canvas matches the blue theme seen in UI
                                            theta_poly = np.linspace(0, 2*np.pi, 100)
                                            ex = d_mean[0] + 2.5 * lambda_d[0] * v_d[0, 0] * np.cos(theta_poly) + 2.5 * lambda_d[1] * v_d[0, 1] * np.sin(theta_poly)
                                            ey = d_mean[1] + 2.5 * lambda_d[0] * v_d[1, 0] * np.cos(theta_poly) + 2.5 * lambda_d[1] * v_d[1, 1] * np.sin(theta_poly)

                                            bounds_list = [{'vertices': list(zip(ex, ey)), 'color': 'black', 'width': 2}]
                                            if st.session_state.get('pending_doublet_verts'):
                                                 bounds_list.append({'vertices': st.session_state.pending_doublet_verts, 'color': 'cyan', 'width': 2})

                                            mpl_fig = create_canvas_figure(viz_df_canvas, d_x, d_y, canvas_title_gmm, is_log=d_is_log, color_col=c_col, boundaries=bounds_list)
                                            add_plot_to_canvas(mpl_fig, canvas_title_gmm)
                                            st.success("Plot added to canvas!")
                                            plt.close(mpl_fig)

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
                                            # st.session_state.show_doublet_gating = False
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

                                    except Exception as e:
                                        st.error(f"Error in Singlet Gating: {e}")
                        if st.session_state.get('doublets_removed'):
                            # Stats for Singlets
                            if st.session_state.get('singlet_gate_stats'):
                                st.success(f"Final Singlets: {st.session_state.singlet_gate_stats}")
                            else:
                                st.success("Doublets have been removed from the dataset.")

                            st.divider()
                            st.subheader("5. Final Data Settings")
                            
                            export_only_selected_channels = False # Initialize default

                            if 'base_experiment' in st.session_state and st.session_state.base_experiment:
                                # --- Helpers for Channel Sorting ---
                                COMMON_FLUORS = [
                                    "GFP", "EGFP", "mCherry", "Cherry", "APC", "FITC", "PE", "DAPI", "PerCP", 
                                    "BV", "Brilliant", "Cy", "Cy3", "Cy5", "Cy7", "BFP", "EBFP", "TagBFP", "SB", 
                                    "RFP", "dsRed", "tdTomato", "YFP", "EYFP", "mNeon", "AmCyan", "Pacific Blue", 
                                    "Pacific Orange", "Alexa", "eFluor", "Hoechst", "PI", "7-AAD", "Zombie", "Live/Dead", "DraQ"
                                ]
                                def is_common_fluor(ch_name):
                                    return any(f.upper() in ch_name.upper() for f in COMMON_FLUORS)
                                def sort_channels(channels):
                                    return sorted(list(channels), key=lambda x: (0 if is_common_fluor(x) else 1, x))

                                all_channels = sort_channels(st.session_state.base_experiment.channels)
                                
                                # --- Initialization of Defaults (if not set) ---
                                if 'persistent_selected_channels' not in st.session_state:
                                    defaults = []
                                    common_area = [c for c in all_channels if is_common_fluor(c) and ("-A" in c or "Area" in c)]
                                    if common_area:
                                        defaults = common_area
                                    else:
                                        any_fluor_area = [c for c in all_channels if ("-A" in c or "Area" in c) and not any(x in c.upper() for x in ["FSC", "SSC", "TIME"])]
                                        defaults = any_fluor_area if any_fluor_area else [c for c in all_channels if not any(x in c.upper() for x in ["FSC", "SSC", "TIME"])]
                                    if not defaults: defaults = all_channels
                                    st.session_state.persistent_selected_channels = defaults
                                    # Init checkbox keys
                                    for ch in all_channels:
                                        st.session_state[f"checkbox_ch_{ch}"] = (ch in defaults)

                                # --- UI Controls ---
                                st.markdown("Select fluorescence channels to include in your analysis. Unselected channels will be hidden.")
                                
                                # Identify Mandatory Channels (FSC/SSC)
                                mandatory_channels = [c for c in all_channels if any(x in c.upper() for x in ["FSC", "SSC"])]
                                selectable_channels = [c for c in all_channels if c not in mandatory_channels]
                                
                                b1, b2, _ = st.columns([1, 1, 2])
                                if b1.button("Select All", key="gating_sel_all"):
                                    st.session_state.persistent_selected_channels = list(all_channels)
                                    for ch in selectable_channels: st.session_state[f"checkbox_ch_{ch}"] = True
                                    st.rerun()
                                if b2.button("Clear All", key="gating_clear_all"):
                                    st.session_state.persistent_selected_channels = list(mandatory_channels)
                                    for ch in selectable_channels: st.session_state[f"checkbox_ch_{ch}"] = False
                                    st.rerun()

                                # --- Checkbox Grid ---
                                # Sync first: if persistent exists, ensure checkbox keys match (in case of rerun)
                                current_selected = set(st.session_state.get('persistent_selected_channels', []))
                                for ch in selectable_channels:
                                    k = f"checkbox_ch_{ch}"
                                    if k not in st.session_state: st.session_state[k] = (ch in current_selected)

                                new_selection = list(mandatory_channels)
                                
                                if mandatory_channels:
                                    st.info(f"**Structural channels included automatically:** {', '.join(mandatory_channels)}")

                                with st.expander("Select Fluorescence Channels", expanded=True):
                                    n_cols = 3
                                    grid_cols = st.columns(n_cols)
                                    for i, ch in enumerate(selectable_channels):
                                        with grid_cols[i % n_cols]:
                                            if st.checkbox(ch, key=f"checkbox_ch_{ch}"):
                                                new_selection.append(ch)

                                # Update State
                                st.session_state.persistent_selected_channels = new_selection
                                st.session_state.selected_channels = new_selection
                                export_only_selected_channels = (len(new_selection) < len(all_channels))
                                # Also update filter_channels_active for compatibility
                                st.session_state.filter_channels_active = export_only_selected_channels

                                if export_only_selected_channels:
                                     st.info(f"{len(new_selection)} channels selected. {len(all_channels) - len(new_selection)} channels dropped.")

                            else:
                                st.info("Load data to enable channel selection.")

                            st.divider()
                            with st.expander("6. Export Data (Optional)", expanded=False):
                                st.write("Configure export options below.")
                            
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

                                # User-specified filename
                                custom_export_name = st.text_input("Custom Filename (Optional)", value="", placeholder=f"Default: {experiment_name}_gated (or _full)", help="Enter a custom name for the exported file. Extension will be added automatically.")

                                if st.button("Generate Export File", key="generate_export"):
                                    with st.spinner("Reconstructing dataset for export..."):
                                        # Clone base
                                        full_exp = st.session_state.base_experiment.clone()
                                        bgater = GatingManager()

                                        # 1. Re-Apply Live Gate (if exists)
                                        l_params = st.session_state.get('live_gate_params')
                                        if l_params:
                                            if l_params['type'] == 'polygon':
                                                full_exp = bgater.apply_manual_gate(full_exp, "_temp_live_mask_reconstructed", l_params['channels'][0], l_params['channels'][1], l_params['vertices'])
                                            elif l_params['type'] == 'label':
                                                gp = st.session_state.get('last_gating_params', {})
                                                if gp:
                                                    method_name = gp.get('method', 'density').lower() if isinstance(gp.get('method'), str) else 'density'
                                                    res_exp = bgater.auto_gate_live_cells(full_exp, gp.get('x'), gp.get('y'), method=method_name, num_components=gp.get('comp', 2), keep=gp.get('keep', 0.9), sigma=gp.get('sigma', 2.0), eps=gp.get('eps', 0.05), min_samples=gp.get('min_samples', 50), use_outlier_removal=gp.get('use_outliers', False))
                                                    lbl = l_params['label']
                                                    if '_temp_live_labels' in res_exp.data.columns:
                                                        matches = (res_exp.data['_temp_live_labels'].astype(str) == str(lbl))
                                                        full_exp.data['_temp_live_mask_reconstructed'] = matches
                                                        # Register metadata to prevent Cytoflow crashes
                                                        if '_temp_live_mask_reconstructed' not in full_exp.metadata:
                                                            full_exp.metadata['_temp_live_mask_reconstructed'] = {'type': 'condition'}
                                            elif l_params['type'] == 'per_cell_line':
                                                # Handle Per-Cell-Line Gating Reconstruction
                                                full_exp.data['_temp_live_mask_reconstructed'] = False
                                                if '_temp_live_mask_reconstructed' not in full_exp.metadata:
                                                    full_exp.metadata['_temp_live_mask_reconstructed'] = {'type': 'condition'}

                                                gates = l_params['gates']
                                                col_name = l_params['col']
                                                for line_val, gate_info in gates.items():
                                                    row_mask = (full_exp.data[col_name].astype(str) == str(line_val))
                                                    if not row_mask.any(): continue
                                                    
                                                    subset_idxs = full_exp.data.index[row_mask]
                                                    if gate_info['type'] == 'polygon':
                                                        from matplotlib.path import Path
                                                        v = gate_info['vertices']
                                                        ch = gate_info['channels']
                                                        path = Path(v)
                                                        points = full_exp.data.loc[subset_idxs, ch].values
                                                        is_in = path.contains_points(points)
                                                        full_exp.data.loc[subset_idxs, '_temp_live_mask_reconstructed'] = is_in

                                            # 2. Re-Apply Singlet Gate (if exists)
                                            s_params = st.session_state.get('singlet_gate_params')
                                            if s_params:
                                                if s_params['type'] == 'polygon':
                                                    full_exp = bgater.apply_manual_gate(full_exp, "_temp_singlet_mask_manual", s_params['channels'][0], s_params['channels'][1], s_params['vertices'])

                                            # 3. Final Assembly
                                            df_export = full_exp.data.copy()

                                            # Create "cells" column
                                            if '_temp_live_mask_reconstructed' in df_export.columns:
                                                df_export['cells'] = df_export['_temp_live_mask_reconstructed']
                                            elif '_temp_live_labels' in df_export.columns and l_params and l_params.get('type') == 'label':
                                                lbl = l_params['label']
                                                df_export['cells'] = (df_export['_temp_live_labels'].astype(str) == str(lbl))
                                            else:
                                                # If no live gate, everything is 'cells' OR 'not cells'? Assuming all.
                                                df_export['cells'] = True

                                            # Create "singlets" column
                                            if '_temp_singlet_mask_manual' in df_export.columns:
                                                df_export['singlets'] = df_export['_temp_singlet_mask_manual']
                                            elif s_params and s_params.get("type") == "mlp":
                                                try:
                                                    from glow.active_learning.models import SingletClassifier
                                                    clf = SingletClassifier.load("data/singlet_model_full.pkl")
                                                    X = df_export[s_params["diff_channels"]].values
                                                    if np.max(X) > 10: X = np.log10(X + 1)
                                                    c_type = s_params.get("cell_type", "colo")
                                                    ids = np.zeros(len(X), dtype=int)
                                                    cts = np.array([c_type] * len(X))
                                                    preds = clf.predict(X, dataset_ids=ids, cell_types=cts)
                                                    df_export["singlets"] = (preds == 1)
                                                except Exception as e:
                                                    st.warning(f"MLP Singlet calc failed: {e}")
                                                    df_export["singlets"] = False
                                            elif s_params and s_params.get('type') == 'gmm_mahalanobis':
                                                 try:
                                                    X = df_export[s_params['channels']].values
                                                    d_mean = s_params['mean']
                                                    inv_cov = np.linalg.inv(s_params['cov'])
                                                    threshold = s_params.get('threshold', 2.5)
                                                    diff = X - d_mean
                                                    dist = np.sqrt(np.sum((diff @ inv_cov) * diff, axis=1))
                                                    df_export['singlets'] = (dist <= threshold)
                                                 except Exception:
                                                     df_export['singlets'] = False
                                            elif s_params and s_params.get('type') == 'per_cell_line':
                                                 # Handle Per-Cell-Line Singlet
                                                 df_export['singlets'] = False
                                                 gates = s_params['gates']
                                                 col_name = s_params['col']
                                                 for line_val, gate in gates.items():
                                                     row_mask = (df_export[col_name].astype(str) == str(line_val))
                                                     if not row_mask.any(): continue
                                                     subset_idxs = df_export.index[row_mask]
                                                     if gate['type'] == 'gmm_mahalanobis':
                                                         ch = gate['channels']
                                                         X_s = df_export.loc[subset_idxs, ch].values
                                                         d_m = gate['mean']
                                                         i_c = np.linalg.inv(gate['cov'])
                                                         thr = gate.get('threshold', 2.5)
                                                         df_s = X_s - d_m
                                                         d_sq = np.sum((df_s @ i_c) * df_s, axis=1)
                                                         df_export.loc[subset_idxs, 'singlets'] = (np.sqrt(d_sq) <= thr)
                                                     elif gate['type'] == 'polygon':
                                                         from matplotlib.path import Path
                                                         v = gate['vertices']
                                                         ch = gate['channels']
                                                         path = Path(v)
                                                         points = df_export.loc[subset_idxs, ch].values
                                                         is_in = path.contains_points(points)
                                                         df_export.loc[subset_idxs, 'singlets'] = is_in
                                            else:
                                                # Default False if no singlet gate defined
                                                df_export['singlets'] = False # Or True if skipped?

                                            # Safe-fill NaNs
                                            df_export['cells'] = df_export['cells'].fillna(False).astype(bool)
                                            df_export['singlets'] = df_export['singlets'].fillna(False).astype(bool)

                                            # Enforce Logic: Singlets must be Cells
                                            df_export.loc[~df_export['cells'], 'singlets'] = False

                                            # MODE SELECTION
                                            if export_mode == "Gated Population (Live + Singlets)":
                                                # Filter ROWS
                                                df_export = df_export[df_export['cells'] & df_export['singlets']].copy()
                                                default_suffix = "_gated"
                                            else:
                                                default_suffix = "_full"
                                            
                                            # Determine Filename
                                            if custom_export_name and custom_export_name.strip():
                                                base_name = custom_export_name.strip()
                                                # Strip extension if user typed it
                                                if base_name.lower().endswith(".zip"): base_name = base_name[:-4]
                                                if base_name.lower().endswith(".csv.gz"): base_name = base_name[:-7]
                                                if base_name.lower().endswith(".csv"): base_name = base_name[:-4]
                                                
                                                file_name = f"{base_name}.csv.gz"
                                            else:
                                                file_name = f"{experiment_name}{default_suffix}.csv.gz"

                                            # Drop Unselected Channels
                                            if export_only_selected_channels and st.session_state.get('selected_channels'):
                                                all_ch = list(getattr(st.session_state.base_experiment, 'channels', []))
                                                keep_ch = list(st.session_state.selected_channels)
                                                to_drop = [c for c in all_ch if c not in keep_ch]
                                                cols_to_drop = [c for c in to_drop if c in df_export.columns]
                                                if cols_to_drop:
                                                    df_export.drop(columns=cols_to_drop, inplace=True)
                                                # Defragment frame
                                                df_export = df_export.copy()

                                            # Cleanup temps
                                            temp_cols = [c for c in df_export.columns if str(c).startswith('_temp_')]
                                            if temp_cols: df_export.drop(columns=temp_cols, inplace=True)

                                            # Compress using ZIP (More robust than single GZIP file)
                                            import zipfile
                                            buf = io.BytesIO()
                                            csv_name = file_name.replace('.csv.gz', '.csv')
                                            zip_name = file_name.replace('.csv.gz', '.zip')
                                        
                                            with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                                                zf.writestr(csv_name, df_export.to_csv(index=False))
                                            
                                            st.session_state.final_csv_export = buf.getvalue()
                                            st.session_state.final_csv_name = zip_name
                                        
                                            st.success(f"Export ready: {zip_name} ({len(st.session_state.final_csv_export)/1024:.1f} KB)")
                                    
                                if 'final_csv_export' in st.session_state and st.session_state.final_csv_export:
                                    st.download_button("Download Exported CSV (ZIP)", data=st.session_state.final_csv_export, file_name=st.session_state.final_csv_name, mime="application/zip")
                                    if st.button("Clear Export Cache"):
                                        del st.session_state.final_csv_export
                                        st.rerun()

                            # Compensation Option
                            do_compensation = st.checkbox("Run Compensation before Analysis (Optional)", key="enable_comp_step")

                            if st.button("✅ Finalize Gating & Proceed to Analysis", type="primary", key="proceed_analysis_main"):
                                with st.spinner("Finalizing analysis dataset..."):
                                    # --- Reconstruction Logic for Analysis (Ported from View.py) ---
                                    # Use GatingManager helper or raw modification? 
                                    # Since we need to create 'analysis_experiment', let's use the robust logic.
                                    # Note: We duplicate logic here because GatingManager shouldn't depend on Streamlit session state ideally.
                                    try:
                                        base = st.session_state.base_experiment
                                        full_exp = base.clone()
                                        
                                        # Standardize Metadata
                                        full_exp.data['cells'] = False
                                        full_exp.metadata['cells'] = {'type': 'condition'}
                                        full_exp.data['singlets'] = False
                                        full_exp.metadata['singlets'] = {'type': 'condition'}
                                        
                                        bgater_an = GatingManager()
                                        
                                        # 1. LIVE Gate
                                        l_params = st.session_state.get('live_gate_params')
                                        if l_params:
                                            if l_params['type'] == 'polygon':
                                                full_exp = bgater_an.apply_manual_gate(full_exp, "_temp_live_mask_rec", l_params['channels'][0], l_params['channels'][1], l_params['vertices'])
                                                full_exp.data['cells'] = full_exp.data["_temp_live_mask_rec"].astype(bool)
                                            elif l_params['type'] == 'label':
                                                gp = st.session_state.get('last_gating_params', {})
                                                if gp:
                                                    method_name = gp.get('method', 'density').lower() if isinstance(gp.get('method'), str) else 'density'
                                                    res_exp = bgater_an.auto_gate_live_cells(full_exp, gp.get('x'), gp.get('y'), method=method_name, num_components=gp.get('comp', 2), keep=gp.get('keep', 0.9), sigma=gp.get('sigma', 2.0), eps=gp.get('eps', 0.05), min_samples=gp.get('min_samples', 50), use_outlier_removal=gp.get('use_outliers', False))
                                                    lbl = l_params['label']
                                                    if '_temp_live_labels' in res_exp.data.columns:
                                                        full_exp.data['cells'] = (res_exp.data['_temp_live_labels'].astype(str) == str(lbl))
                                            elif l_params['type'] == 'per_cell_line':
                                                # Per-Cell Line Live Gate Logic
                                                full_exp.data['cells'] = False # Reset
                                                gates = l_params['gates']
                                                col_name = l_params['col']
                                                for line_val, gate_info in gates.items():
                                                    row_mask = (full_exp.data[col_name].astype(str) == str(line_val))
                                                    if not row_mask.any(): continue
                                                    
                                                    subset_idxs = full_exp.data.index[row_mask]
                                                    if gate_info['type'] == 'polygon':
                                                        from matplotlib.path import Path
                                                        v = gate_info['vertices']
                                                        ch = gate_info['channels']
                                                        path = Path(v)
                                                        points = full_exp.data.loc[subset_idxs, ch].values
                                                        is_in = path.contains_points(points)
                                                        full_exp.data.loc[subset_idxs, 'cells'] = is_in
                                        else:
                                            # No gate = everything? or nothing?
                                            # Usually if no gate is applied, we assume everything is live or user skipped.
                                            full_exp.data['cells'] = True # Default to True if skipped? Or False? Let's assume True if user skipped Live Gating (rare)

                                        # 2. SINGLET Gate
                                        s_params = st.session_state.get('singlet_gate_params')
                                        if s_params:
                                            if s_params['type'] == 'polygon':
                                                full_exp = bgater_an.apply_manual_gate(full_exp, "_temp_singlet_mask_rec", s_params['channels'][0], s_params['channels'][1], s_params['vertices'])
                                                full_exp.data['singlets'] = full_exp.data["_temp_singlet_mask_rec"].astype(bool)
                                            elif s_params.get('type') == 'gmm_mahalanobis':
                                                X = full_exp.data[s_params['channels']].values
                                                d_mean = s_params['mean']
                                                inv_cov = np.linalg.inv(s_params['cov'])
                                                threshold = s_params.get('threshold', 2.5)
                                                diff = X - d_mean
                                                dist = np.sqrt(np.sum((diff @ inv_cov) * diff, axis=1))
                                                full_exp.data['singlets'] = (dist <= threshold)
                                            elif s_params.get("type") == "mlp":
                                                try:
                                                    from glow.active_learning.models import SingletClassifier
                                                    clf = SingletClassifier.load("data/singlet_model_full.pkl")
                                                    X = full_exp.data[s_params["diff_channels"]].values
                                                    if np.max(X) > 10: X = np.log10(X + 1)
                                                    c_type = s_params.get("cell_type", "colo")
                                                    ids = np.zeros(len(X), dtype=int)
                                                    cts = np.array([c_type] * len(X))
                                                    preds = clf.predict(X, dataset_ids=ids, cell_types=cts)
                                                    full_exp.data["singlets"] = (preds == 1)
                                                except Exception as e:
                                                    st.warning(f"MLP Singlet calc failed: {e}")
                                                    full_exp.data["singlets"] = False
                                            elif s_params.get('type') == 'per_cell_line':
                                                 # Handle Per-Cell-Line Singlet
                                                 gates = s_params['gates']
                                                 col_name = s_params['col']
                                                 for line_val, gate in gates.items():
                                                     row_mask = (full_exp.data[col_name].astype(str) == str(line_val))
                                                     if not row_mask.any(): continue
                                                     subset_idxs = full_exp.data.index[row_mask]
                                                     if gate['type'] == 'gmm_mahalanobis':
                                                         ch = gate['channels']
                                                         X_s = full_exp.data.loc[subset_idxs, ch].values
                                                         d_m = gate['mean']
                                                         i_c = np.linalg.inv(gate['cov'])
                                                         thr = gate.get('threshold', 2.5)
                                                         df_s = X_s - d_m
                                                         d_sq = np.sum((df_s @ i_c) * df_s, axis=1)
                                                         full_exp.data.loc[subset_idxs, 'singlets'] = (np.sqrt(d_sq) <= thr)
                                                     elif gate['type'] == 'polygon':
                                                         from matplotlib.path import Path
                                                         v = gate['vertices']
                                                         ch = gate['channels']
                                                         path = Path(v)
                                                         points = full_exp.data.loc[subset_idxs, ch].values
                                                         is_in = path.contains_points(points)
                                                         full_exp.data.loc[subset_idxs, 'singlets'] = is_in

                                        # Dependency: singlets must be cells
                                        full_exp.data.loc[~full_exp.data['cells'], 'singlets'] = False

                                        # Cleanup temps
                                        temp_c = [c for c in full_exp.data.columns if str(c).startswith('_temp_')]
                                        if temp_c: full_exp.data.drop(columns=temp_c, inplace=True)

                                        # --- Drop Unselected Channels for Analysis ---
                                        if export_only_selected_channels and st.session_state.get('selected_channels'):
                                            keep_ch = set(st.session_state.selected_channels)
                                            
                                            # If precomputed compensation exists, also keep the corresponding comp_ channels
                                            # even if they weren't explicitly in the 'selected_channels' list (which usually just has raw names)
                                            if st.session_state.get('has_precomputed_compensation', False):
                                                 for raw_c in list(keep_ch):
                                                      comp_c = f"comp_{raw_c}"
                                                      if comp_c in full_exp.data.columns:
                                                           keep_ch.add(comp_c)

                                            if hasattr(base, 'channels'):
                                                all_ch = list(base.channels)
                                                to_drop = [c for c in all_ch if c not in keep_ch]
                                                
                                                # Drop from DataFrame
                                                cols_to_drop = [c for c in to_drop if c in full_exp.data.columns]
                                                if cols_to_drop:
                                                    full_exp.data.drop(columns=cols_to_drop, inplace=True)
                                                
                                                # Drop from Metadata
                                                for c in to_drop:
                                                    if c in full_exp.metadata:
                                                        del full_exp.metadata[c]

                                        st.session_state.analysis_experiment = full_exp
                                        st.session_state.gating_complete = True
                                        
                                        if do_compensation:
                                            st.session_state.run_compensation = True
                                            st.session_state.current_tab = "Compensation"
                                        else:
                                            st.session_state.run_compensation = False
                                            st.session_state.current_tab = "Analysis"

                                        # Optimize Cache: Clear raw files now that we have analysis_experiment
                                        if DataLoader:
                                            DataLoader.clear_upload_cache()
                                            
                                        st.rerun()

                                    except Exception as e:
                                        st.error(f"Error preparing analysis: {e}")

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
