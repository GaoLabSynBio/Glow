import streamlit as st
import pandas as pd
import numpy as np
from typing import Any
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
from glow.analysis.stats import get_summary_statistics
from glow.analysis.positive import get_channel_data_for_samples, calculate_positive_stats_recursive
from glow.canvas.functions import add_plot_to_canvas
from glow.utils import logicle_transform
from glow.data_loading.sample_metadata import label_map_from_data


def sample_label(sample_id):
    """Display label for a stable Sample id (falls back to the id itself)."""
    return st.session_state.get('_sample_labels', {}).get(str(sample_id), str(sample_id))

def update_threshold_value(model_key, widget_key):
    """Callback to sync widget value to model state."""
    if widget_key in st.session_state:
        st.session_state[model_key] = st.session_state[widget_key]

def render_analysis_tab():
    st.header("Analysis & Statistics")

    # -------------------------------------------------------------------------
    # 1. Data Retrieval
    # -------------------------------------------------------------------------
    if 'analysis_experiment' in st.session_state:
        experiment = st.session_state.analysis_experiment
    elif 'gated_experiment' in st.session_state:
        st.warning("Using filtered Experiment. Statistics (Live %) will be relative to the filtered subset (100%).")
        experiment = st.session_state.gated_experiment
    elif 'base_experiment' in st.session_state:
        st.warning("Using base experiment (no custom gating applied). Statistics might be limited.")
        experiment = st.session_state.base_experiment
    else:
        st.error("No experiment loaded. Please go back to Gating tab.")
        return

    # Check for High Resolution Data Scaling
    apply_scaling = False
    scaling_factor = 1.0
    if hasattr(experiment, 'metadata') and experiment.metadata.get('is_high_res', False):
        st.warning("High-resolution data detected. Data logic is scaled by default (1/10000).")
        use_raw = st.toggle("Show Raw Values (Unscaled)", value=False, key="analysis_show_raw_vals")
        apply_scaling = not use_raw
        if apply_scaling:
            scaling_factor = experiment.metadata.get('scaling_factor', 10000.0)

    # --- UNIFY COMPENSATED CHANNEL NAMES ---
    # Requirement: Rename "Comp_{fluor}-A" (or comp_) to "Comp-{fluor}-A"
    # Working on a clone/copy to avoid modifying shared session state used by Gating
    if hasattr(experiment, 'clone'):
        experiment = experiment.clone()

    rename_map = {}
    for col in experiment.data.columns:
        # Check for comp_ or Comp_ prefix (but not Comp-)
        if (col.startswith("comp_") or col.startswith("Comp_")) and not col.startswith("Comp-"):
             parts = col.split('_', 1)
             if len(parts) == 2:
                 suffix = parts[1]
                 new_name = f"Comp-{suffix}"
                 rename_map[col] = new_name

    if rename_map:
        experiment.data = experiment.data.rename(columns=rename_map)
        # Update session state gates if they reference old names
        if 'positive_gates' in st.session_state:
             for g in st.session_state.positive_gates:
                 if g.get('channel') in rename_map:
                     old_name = g['channel']
                     new_name = rename_map[old_name]
                     g['channel'] = new_name
                     # Try to update label if it looks generic
                     if 'label' in g:
                         g['label'] = g['label'].replace(old_name, new_name)

    # -------------------------------------------------------------------------
    # 2. Summary Statistics Table
    # -------------------------------------------------------------------------
    st.subheader("Summary Statistics")

    # Calculate base stats
    stats_df = get_summary_statistics(experiment)

    # --- Row Exclusion for Sub-Gating & Visualization ---
    # Apply "Exclusion" (filter for Live/Singlets if flags present) for the REST of analysis
    # This keeps the data "clean" for positive population definition while keeping original metadata
    if 'cells' in experiment.data.columns and 'singlets' in experiment.data.columns:
        # Create a filtered copy for visualization and sub-gating
        # Summary statistics above were calculated on the full experiment to get correct % flags
        filtered_experiment = experiment.clone() if hasattr(experiment, 'clone') else experiment
        filtered_experiment.data = experiment.data[experiment.data['cells'] & experiment.data['singlets']].copy()
        experiment = filtered_experiment
        st.info(f"Analysis is being performed on the gated (Live & Singlet) population: {len(experiment.data)} events.")

    # Initialize positive gates storage if not present
    if 'positive_gates' not in st.session_state:
        st.session_state.positive_gates = [] # List of dicts

    if 'threshold_gate' not in st.session_state:
        st.session_state.threshold_gate = {}

    # Migration Check (Old to New Format)
    if st.session_state.positive_gates and 'parent' not in st.session_state.positive_gates[0]:
        new_gates = []
        for g in st.session_state.positive_gates:
            if 'parent' not in g:
                g['parent'] = 'Singlets'
            if 'label' not in g:
                g['label'] = f"{g['channel']}+"
            new_gates.append(g)
        st.session_state.positive_gates = new_gates

    # Ensure threshold_gate object is synced with list
    for g in st.session_state.positive_gates:
        if 'channel' in g:
            lbl = g.get('label', f"{g['channel']}+")
        else:
            lbl = g.get('label', "Gate")

        if 'threshold' in g:
            st.session_state.threshold_gate[lbl] = g['threshold']

    # Display labels (Experiment Name [+ rep N]) for the stable Sample ids
    st.session_state['_sample_labels'] = label_map_from_data(experiment.data)

    # Dynamically build full table using Recursive Logic
    full_df = stats_df.copy()

    for gate in st.session_state.positive_gates:
        # Use new recursive calculator
        pos_df = calculate_positive_stats_recursive(experiment, gate, st.session_state.positive_gates)
        if not pos_df.empty:
            full_df = pd.merge(full_df, pos_df, on='Sample', how='left')

    st.dataframe(full_df, width="stretch")

    st.divider()

    # -------------------------------------------------------------------------
    # 3. Positive Population Identification (Hierarchical)
    # -------------------------------------------------------------------------
    st.subheader("Define Positive Populations")

    # List Existing Gates (Read-Only)
    if st.session_state.positive_gates:
        st.write("### Defined Gates")
        for i, g in enumerate(st.session_state.positive_gates):
            # Safe label retrieval
            if 'channel' in g:
                 fallback_lbl = f"{g['channel']}+"
            else:
                 fallback_lbl = "Gate"

            lbl = g.get('label', fallback_lbl)

            with st.expander(f"Gate {i+1}: {lbl} (Parent: {g.get('parent', 'Singlets')})"):

                gate_type = g.get('type', 'binary')
                is_1d = False

                if gate_type == 'quadrant':
                     st.write("**Type:** 2D Quadrant")
                     st.write(f"**Channels:** {g.get('channel_x')} vs {g.get('channel_y')}")
                     q_val = g.get('quadrant')
                     if isinstance(q_val, list):
                         q_val = "/".join(q_val)
                     st.write(f"**Quadrant:** {q_val} (Thresholds: {g.get('threshold_x'):.2f}, {g.get('threshold_y'):.2f})")
                     st.info("Visualization not available for 2D gates in this summary view.")

                elif gate_type == 'polygon':
                     st.write("**Type:** 2D Polygon")
                     st.write(f"**Channels:** {g.get('channel_x')} vs {g.get('channel_y')}")
                     st.write(f"**Vertices:** {len(g.get('vertices', []))} points")
                     st.info("Visualization not available for 2D gates in this summary view.")

                elif gate_type == 'range':
                     st.write(f"**Channel:** {g.get('channel')}")
                     st.write(f"**Range:** {g.get('min'):.2f} - {g.get('max'):.2f}")
                     st.write(f"**Filter Path:** {g.get('parent', 'Singlets')} -> {g['channel']} in [{g.get('min'):.2f}, {g.get('max'):.2f}]")
                     is_1d = True

                else:
                     # Binary / Legacy
                     st.write(f"**Channel:** {g.get('channel')}")
                     op = ">" if g.get('direction', 'positive') == 'positive' else "<="
                     t = g.get('threshold', 0)
                     st.write(f"**Threshold:** {t:.2f} ({g.get('direction', 'positive')})")
                     st.write(f"**Filter Path:** {g.get('parent', 'Singlets')} -> {g['channel']} {op} {t:.2f}")
                     is_1d = True

                # --- Visualization (1D Only) ---
                if is_1d:
                    # Fetch data for this gate context
                    try:
                        # Use SAVED samples if available (to match user's creation view), otherwise default default
                        if 'samples' in g:
                            disp_samples = g['samples']
                        else:
                            # Fallback for legacy gates or missing data
                            if 'Sample' in full_df.columns:
                                avail_samples = full_df['Sample'].unique().tolist()
                            else:
                                avail_samples = experiment.data['Sample'].unique().tolist()
                            # Default to first 1 sample to mimic "creation" view default
                            disp_samples = avail_samples[:1]

                        # Get Data from PARENT population
                        p_gate = g.get('parent', 'Singlets')
                        ch = g['channel']

                        data_map_v = get_channel_data_for_samples(experiment, ch, disp_samples, parent_gate_name=p_gate, defined_gates=st.session_state.positive_gates, apply_scaling=apply_scaling)

                        # Prepare Plot
                        fig_g = go.Figure()

                        # Determine range for axis
                        all_vals_v = []
                        for s in data_map_v:
                            all_vals_v.extend(data_map_v[s])
                        all_vals_v = np.array([x for x in all_vals_v if not np.isnan(x)])

                        if len(all_vals_v) > 0:
                            # Auto-Log detection: if max > 1000 usually log is better for flow
                            # But let's check negative values.
                            use_log_v = (np.mean(all_vals_v) > 100) and (np.min(all_vals_v) > 0)

                            # Calculate Domain
                            if use_log_v:
                                # Filter <= 0
                                pos_v = all_vals_v[all_vals_v > 0]
                                if len(pos_v) > 0:
                                    x_min_v, x_max_v = np.percentile(pos_v, 0.1), np.percentile(pos_v, 99.9)
                                    d_dom = np.logspace(np.log10(x_min_v), np.log10(x_max_v), 200)
                                else:
                                    d_dom = np.linspace(0, 1, 100)
                            else:
                                 x_min_v, x_max_v = np.percentile(all_vals_v, 0.1), np.percentile(all_vals_v, 99.9)
                                 d_dom = np.linspace(x_min_v, x_max_v, 200)

                            # Plot KDEs
                            colors_v = px.colors.qualitative.Plotly
                            for idx, s in enumerate(disp_samples):
                                if s in data_map_v:
                                    v = data_map_v[s]
                                    v = v[~np.isnan(v)]
                                    if len(v) < 10:
                                        continue # Skip empty

                                    try:
                                        if use_log_v:
                                            v_p = v[v > 0]
                                            if len(v_p) < 10:
                                                continue
                                            kde = gaussian_kde(np.log10(v_p))
                                            y_dom = kde(np.log10(d_dom))
                                        else:
                                            kde = gaussian_kde(v)
                                            y_dom = kde(d_dom)

                                        c = colors_v[idx % len(colors_v)]
                                        fig_g.add_trace(go.Scatter(x=d_dom, y=y_dom, mode='lines', line=dict(color=c), name=sample_label(s), fill='tozeroy', opacity=0.3))
                                    except Exception:
                                        pass

                            # Add Threshold/Range Line
                            # Gates are stored in Standard (Scaled) Units.
                            # Adjust gate values only if we are displaying Raw Data (apply_scaling=False).
                            if g.get('type') == 'range':
                                 t_min, t_max = g.get('min'), g.get('max')
                                 if not apply_scaling:
                                     t_min *= scaling_factor
                                     t_max *= scaling_factor
                                 
                                 fig_g.add_vrect(x0=t_min, x1=t_max, fillcolor="green", opacity=0.1, layer="below", line_width=0)
                                 fig_g.add_vline(x=t_min, line_dash="dash", line_color="green")
                                 fig_g.add_vline(x=t_max, line_dash="dash", line_color="green")
                            else:
                                 thr = g['threshold']
                                 if not apply_scaling:
                                     thr *= scaling_factor
                                 
                                 is_pos = (g.get('direction', 'positive') == 'positive')
                                 color_line = "green" if is_pos else "red"
                                 fig_g.add_vline(x=thr, line_dash="dash", line_color=color_line, line_width=2)

                                 # Shading
                                 # Approximate simplifying shading
                                 dom_min = min(d_dom) if len(d_dom)>0 else 0
                                 dom_max = max(d_dom) if len(d_dom)>0 else 1
                                 
                                 if is_pos:
                                     fig_g.add_vrect(x0=thr, x1=dom_max, fillcolor="green", opacity=0.1, layer="below", line_width=0)
                                 else:
                                     fig_g.add_vrect(x0=dom_min, x1=thr, fillcolor="red", opacity=0.1, layer="below", line_width=0)

                            fig_g.update_layout(
                                margin=dict(l=0, r=0, t=20, b=0),
                                height=200,
                                xaxis_type="log" if use_log_v else "linear",
                                showlegend=True, # Show legend to identify samples
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10, color="black")),
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                font=dict(color="black"),
                                xaxis=dict(showgrid=True, gridcolor='lightgrey', color='black', tickfont=dict(color='black')),
                                yaxis=dict(showgrid=False, showticklabels=False, color='black', tickfont=dict(color='black'))
                            )
                            st.plotly_chart(fig_g, width="stretch")
                    except Exception as e:
                        st.error(f"Could not load visualization: {e}")

                if st.button(f"Delete Gate {i+1}", key=f"del_gate_{i}"):
                     removed = st.session_state.positive_gates.pop(i)
                     # Sycn with threshold_gate object
                     if 'threshold_gate' in st.session_state and removed.get('label') in st.session_state.threshold_gate:
                         del st.session_state.threshold_gate[removed['label']]
                     st.rerun()


    # Add New Gate Section
    if 'show_gate_creator' not in st.session_state:
        st.session_state.show_gate_creator = False

    if not st.session_state.show_gate_creator:
        if st.button("Add New Gate", type="primary"):
            st.session_state.show_gate_creator = True
            st.rerun()
    else:
        st.divider()
        st.markdown("#### New Gate Definition")

        col_opts, col_plots = st.columns([1, 3])

        with col_opts:
            # --- PARENT SELECTION ---
            # Options: Singlets + any defined gates
            parent_options = ['Singlets'] + [g['label'] for g in st.session_state.positive_gates]
            parent_gate = st.selectbox("Select Parent Population", parent_options, index=0)

            # --- MODE SELECTION ---
            st.divider()
            st.markdown("#### Analysis Settings")
            analysis_mode = st.selectbox("Analysis Mode", ["1D Histogram", "2D Quadrant", "2D Polygon"], index=0)

            # --- CHANNEL SELECTION ---
            # Identify all numeric channels in the dataset
            all_numeric = [c for c in experiment.data.columns if pd.api.types.is_numeric_dtype(experiment.data[c])]

            # Filter out system flags and internal bookkeeping columns
            ignore_cols = ['cells', 'singlets', 'Is_Live', 'Is_Singlet', 'LiveGate', 'SingletGate', 'Density']

            # Get selected channels from Gating tab
            # Use persistent state to ensure it survives tab switching
            user_selected = st.session_state.get('persistent_selected_channels')

            # --- Compensation Switch ---
            # Check if compensated channels exist
            comp_cols = [c for c in all_numeric if c.startswith("Comp-")]
            has_comp = len(comp_cols) > 0
            use_comp = False

            if has_comp:
                use_comp = st.checkbox("Use Compensated Data", value=True, key="use_comp_analysis")

            # --- Helper for Fluorophore Filtering ---
            COMMON_FLUORS = [
                "GFP", "EGFP", "mCherry", "Cherry", "APC", "FITC", "PE", "DAPI", "PerCP",
                "BV", "Brilliant", "Cy", "Cy3", "Cy5", "Cy7", "BFP", "EBFP", "TagBFP", "SB",
                "RFP", "dsRed", "tdTomato", "YFP", "EYFP", "mNeon", "AmCyan", "Pacific Blue",
                "Pacific Orange", "Alexa", "eFluor", "Hoechst", "PI", "7-AAD", "Zombie", "Live/Dead", "DraQ"
            ]
            def is_fluorophore(ch_name):
                # Must NOT be Scatter or Time
                if any(x in ch_name.upper() for x in ["FSC", "SSC", "TIME"]):
                    return False
                # Must match common list OR contain "-A"/"Area" but passed scatter check
                # Actually, user requested "common fluorophores" filter specifically.
                return any(f.upper() in ch_name.upper() for f in COMMON_FLUORS)

            # Define valid channels: must be in user's selection and not ignored
            if user_selected is not None:
                # Filter 1: In User Selection
                step1 = [c for c in all_numeric if c not in ignore_cols and c in user_selected]

                # --- Apply Compensation Logic ---
                # If use_comp, we replace original channels with "Comp-" versions IF available
                final_selection = []

                if use_comp:
                    for c in step1:
                        comp_name = f"Comp-{c}"
                        if comp_name in all_numeric:
                            final_selection.append(comp_name)
                        else:
                            # User request: if a channel does not have Comp-xxx-A channel, just show that channel
                            final_selection.append(c)
                else:
                    final_selection = step1

                # Filter 2: Is Fluorophore (Analysis Specific Requirement)
                # Apply to the potentially comp-swapped list
                valid_channels = sorted([c for c in final_selection if is_fluorophore(c)])

                # Fallback
                if not valid_channels:
                    valid_channels = sorted([c for c in final_selection if not any(x in c.upper() for x in ["FSC", "SSC", "TIME"])])
            else:
                # Fallback if gating wasn't visited - strict fluor filter on all numeric
                # Also apply comp check here just in case
                source_cols = all_numeric
                if use_comp:
                    comp_set = set(x for x in all_numeric if x.startswith("Comp-"))
                    shadowed = set(x.replace("Comp-", "") for x in comp_set)
                    source_cols = []
                    for c in all_numeric:
                        if c.startswith("Comp-"):
                            source_cols.append(c)
                        elif c in shadowed:
                            continue
                        else:
                            source_cols.append(c)

                valid_channels = sorted([
                    c for c in source_cols
                    if c not in ignore_cols and is_fluorophore(c)
                ])

            # Final safety check: if everything was filtered out, show at least non-scatter
            if not valid_channels:
                # Just show whatever is numeric and not ignored
                valid_channels = sorted([c for c in experiment.data.columns if pd.api.types.is_numeric_dtype(experiment.data[c]) and c not in ignore_cols])

            # --- Zipped CSV / Special Context Handling ---
            if "Experiment_Name" in experiment.data.columns:
                # Requirement: Only selectable channels are fluorophore-A channels and no Experiment_Name
                strict_channels = []
                for c in valid_channels:
                    if c == "Experiment_Name":
                        continue
                    # Only allow channels ending in "-A" (Area) AND are fluorophores (excludes Scatter)
                    if c.endswith("-A") and is_fluorophore(c):
                        strict_channels.append(c)

                if strict_channels:
                    valid_channels = strict_channels
                else:
                    # Fallback: at least remove Experiment_Name
                    valid_channels = [vc for vc in valid_channels if vc != "Experiment_Name"]


        if analysis_mode == "1D Histogram":
            # =========================================================
            # 1D HISTOGRAM / DENSITY LOGIC (Legacy)
            # =========================================================
            with col_opts:
                st.markdown("#### Hist Parameters")
                # Use a key to prevent reset issues, though usually fine
                selected_channel = st.selectbox("Select X-Axis Channel", valid_channels, key="analysis_channel_select")

            if selected_channel:
                # B. Sample Selection for Visualization
                sample_list = full_df['Sample'].unique().tolist()
                # Default to the first sample
                default_samples = [sample_list[0]] if sample_list else []

                with col_opts:
                    selected_samples = st.multiselect("Select Samples to Overlay", sample_list, default=default_samples, format_func=sample_label, key="analysis_sample_select")

                if selected_samples:
                    # Fetch Data, Filtered by Parent gate
                    data_map = get_channel_data_for_samples(experiment, selected_channel, selected_samples, parent_gate_name=parent_gate, defined_gates=st.session_state.positive_gates, apply_scaling=apply_scaling)

                    # determine range for slider
                    all_values = []
                    for s in data_map:
                        # Add check for valid arrays
                        if len(data_map[s]) > 0:
                             all_values.extend(data_map[s])

                    # Robust NaN filtering
                    all_values = np.array([x for x in all_values if not np.isnan(x)])

                    with col_opts:
                        st.markdown("#### Axis & Overlay")
                        use_log = st.checkbox("Log Scale Plot", value=True)
                        use_ridgeline = st.checkbox("Ridgeline Mode", value=True)

                        ridgeline_offset = 0.6
                        if use_ridgeline:
                             ridgeline_offset = st.slider("Offset", 0.1, 2.0, 0.6, 0.1)

                        # Check validity for log
                        if use_log and np.max(all_values) <= 0:
                            st.warning("No positive values for Log scale. Switching to Linear.")
                            use_log = False

                        # We need to determine a common X-axis range for the plot
                        # Based on the aggregation of all selected data
                        if use_log:
                            valid_data_for_range = all_values[all_values > 0]
                            if len(valid_data_for_range) == 0:
                                 # Should be caught above, but safety
                                 plot_min, plot_max = 0, 1
                            else:
                                # Use percentiles to find "High Density Region" and exclude far outliers
                                # Clamp min to 0.01 to avoid extreme scaling from artifacts
                                plot_min = max(0.01, float(np.percentile(valid_data_for_range, 0.1))) # 0.1th percentile
                                plot_max = float(np.percentile(valid_data_for_range, 99.9)) # 99.9th percentile
                        else:
                            valid_data_for_range = all_values
                            # For Linear, user requested min to be 0
                            plot_min = 0.0
                            # Max 99.9th percentile to catch main pop
                            plot_max = float(np.percentile(valid_data_for_range, 99.9))
                            if plot_max <= 0:
                                plot_max = 1.0

                        # Ensure non-singular
                        if plot_min >= plot_max:
                            plot_max = plot_min + 1.0

                        # C. Threshold Selection
                        st.markdown("#### Threshold Selection")

                        # 1. Gate Type Selection
                        gate_type = st.radio("Gate Type", ["Binary (Threshold)", "Range (Min - Max)"], horizontal=True)

                        is_range = (gate_type == "Range (Min - Max)")

                        is_negative = False
                        if not is_range:
                            gate_direction = st.radio("Population Direction", ["Positive (-->)", "Negative (<--)"], horizontal=True)
                            is_negative = (gate_direction == "Negative (<--)")

                        st.info("Tip: Select a point on the plot to automatically set the threshold.")

                        # keys for session state - INCLUDE PARENT NAME to avoid collisions
                        thr_key = f"thresh_val_{selected_channel}_p{parent_gate}"
                        min_key = f"range_min_{selected_channel}_p{parent_gate}"
                        max_key = f"range_max_{selected_channel}_p{parent_gate}"

                        # Initialize logic
                        if thr_key not in st.session_state:
                             st.session_state[thr_key] = (plot_min + plot_max) / 2.0
                        if min_key not in st.session_state:
                             st.session_state[min_key] = plot_min + (plot_max - plot_min) * 0.2
                        if max_key not in st.session_state:
                             st.session_state[max_key] = plot_max - (plot_max - plot_min) * 0.2

                        # Capture selection events BEFORE rendering chart to avoid "select twice" bug
                        # We check the session state key directly
                        plot_key = f"plot_{selected_channel}_{parent_gate}"
                        selection_data = st.session_state.get(plot_key)

                        if selection_data and selection_data.get("selection"):
                            sel = selection_data["selection"]
                            # 1. Box Selection (Range)
                            if sel.get("box"):
                                 box = sel["box"][0]
                                 x_range = box.get("x", [])
                                 if x_range and len(x_range) >= 2:
                                      sel_min, sel_max = sorted(x_range[:2])
                                      if is_range:
                                           if abs(float(sel_min) - st.session_state[min_key]) > 0.0001 or \
                                              abs(float(sel_max) - st.session_state[max_key]) > 0.0001:
                                                st.session_state[min_key] = float(sel_min)
                                                st.session_state[max_key] = float(sel_max)
                                                st.rerun()
                                      else:
                                           # For binary, use box center or start? User likely wants box for range, click for binary.
                                           # But let's support box as threshold set too.
                                           if abs(float(sel_min) - st.session_state[thr_key]) > 0.0001:
                                                st.session_state[thr_key] = float(sel_min)
                                                st.rerun()

                            # 2. Point Click (Binary)
                            if sel.get("points"):
                                 point = sel["points"][0]
                                 sel_x = point.get("x")
                                 if sel_x is not None:
                                      if is_range:
                                           # Move closest bound
                                           dist_min = abs(float(sel_x) - st.session_state[min_key])
                                           dist_max = abs(float(sel_x) - st.session_state[max_key])
                                           if dist_min < dist_max:
                                                if abs(float(sel_x) - st.session_state[min_key]) > 0.0001:
                                                     st.session_state[min_key] = float(sel_x)
                                                     st.rerun()
                                           else:
                                                if abs(float(sel_x) - st.session_state[max_key]) > 0.0001:
                                                     st.session_state[max_key] = float(sel_x)
                                                     st.rerun()
                                      else:
                                           if abs(float(sel_x) - st.session_state[thr_key]) > 0.0001:
                                                st.session_state[thr_key] = float(sel_x)
                                                st.rerun()

                        # Render Inputs
                        if is_range:
                              w_min_key = f"w_{min_key}"
                              # Sync Model -> Widget (if external change happened)
                              if w_min_key not in st.session_state or abs(float(st.session_state.get(w_min_key, 0)) - float(st.session_state[min_key])) > 0.0001:
                                  st.session_state[w_min_key] = float(st.session_state[min_key])

                              curr_min = st.number_input("Min Threshold", key=w_min_key, on_change=update_threshold_value, args=(min_key, w_min_key))

                              w_max_key = f"w_{max_key}"
                              if w_max_key not in st.session_state or abs(float(st.session_state.get(w_max_key, 0)) - float(st.session_state[max_key])) > 0.0001:
                                  st.session_state[w_max_key] = float(st.session_state[max_key])

                              curr_max = st.number_input("Max Threshold", key=w_max_key, on_change=update_threshold_value, args=(max_key, w_max_key))

                              curr_threshold = (curr_min, curr_max)
                        else:
                             # For binary mode, use a single point selection logic
                             w_thr_key = f"w_{thr_key}"
                             if w_thr_key not in st.session_state or abs(float(st.session_state.get(w_thr_key, 0)) - float(st.session_state[thr_key])) > 0.0001:
                                  st.session_state[w_thr_key] = float(st.session_state[thr_key])

                             curr_val = st.number_input("Threshold Value", key=w_thr_key, on_change=update_threshold_value, args=(thr_key, w_thr_key))
                             curr_threshold = curr_val

                        # D. Interactive KDE Plot
                        # Colors
                        if len(selected_samples) > 0:
                             colors = px.colors.sample_colorscale("Viridis", np.linspace(0, 1, len(selected_samples)))
                        else:
                             colors = px.colors.qualitative.Plotly


                        # Generate support points for KDE
                        # If Log: generate in log space then transform back
                        if use_log:
                            # Log filtering ensures plot_min > 0
                            if plot_min <= 0:
                                plot_min = np.min(valid_data_for_range[valid_data_for_range > 0])

                            x_log_min = np.log10(plot_min)
                            x_log_max = np.log10(plot_max)
                            # Add buffer
                            pad = (x_log_max - x_log_min) * 0.1
                            x_log_domain = np.linspace(x_log_min - pad, x_log_max + pad, 200)
                            x_domain = np.power(10, x_log_domain)
                        else:
                            # Linear
                            pad = (plot_max - plot_min) * 0.1
                            # Ensure we don't go below 0 if requested
                            start = max(0, plot_min - pad)
                            x_domain = np.linspace(start, plot_max + pad, 200)

                        fig = go.Figure()

                        # Add an invisible "Click Catcher" trace for binary mode to make the whole plot area clickable
                        # (Plotly point selection only triggers on actual traces)
                        if not is_range:
                            fig.add_trace(go.Scatter(
                                x=x_domain,
                                y=[0] * len(x_domain),
                                mode='markers',
                                marker=dict(size=20, opacity=0),
                                showlegend=False,
                                hoverinfo='none',
                                name="click_catcher"
                            ))

                        # Iterate samples and plot KDE
                        for i, s in enumerate(selected_samples):
                            if s in data_map:
                                vals = data_map[s]

                                vals = vals[~np.isnan(vals)]

                                if len(vals) < 2:
                                    continue

                                # Prepare data for KDE
                                if use_log:
                                    # Filter positive
                                    qt_vals = vals[vals > 0]
                                    if len(qt_vals) < 2:
                                        continue
                                    dataset = np.log10(qt_vals)
                                    # Eval on log domain
                                    try:
                                        kde = gaussian_kde(dataset)
                                        y_vals = kde(np.log10(x_domain))
                                    except Exception:
                                        continue
                                else:
                                    dataset = vals
                                    try:
                                        kde = gaussian_kde(dataset)
                                        y_vals = kde(x_domain)
                                    except Exception:
                                        continue

                                # Normalize Area? Gaussian_kde integrates to 1.
                                # We can plot as is.

                                color = colors[i % len(colors)]

                                if use_ridgeline:
                                    # Normalize y to max 1 for consistent stacking
                                    if np.max(y_vals) > 0:
                                        y_vals = y_vals / np.max(y_vals)

                                    # Apply Offset
                                    base_y = i * ridgeline_offset
                                    y_plotted = y_vals + base_y

                                    # Create filled area (Pseudo-Ridgeline)
                                    # We use 'toself' with a closed polygon to fill relative to the baseline
                                    x_poly = np.concatenate([x_domain, x_domain[::-1]])
                                    y_poly = np.concatenate([y_plotted, np.full_like(y_plotted, base_y)])

                                    fig.add_trace(go.Scatter(
                                        x=x_poly,
                                        y=y_poly,
                                        fill='toself',
                                        mode='lines',
                                        line=dict(color='white', width=0.1),
                                        fillcolor=color,
                                        name=sample_label(s),
                                        opacity=0.5,
                                        hoverinfo='skip' # Simplify hover for filled area
                                    ))
                                    # Add Line on top for clarity
                                    fig.add_trace(go.Scatter(
                                        x=x_domain,
                                        y=y_plotted,
                                        mode='lines',
                                        line=dict(color='black', width=1.5),
                                        showlegend=False,
                                        hoverinfo='skip'
                                    ))
                                else:
                                    fig.add_trace(go.Scatter(
                                        x=x_domain,
                                        y=y_vals,
                                        mode='lines+markers', # Markers needed for click targeting
                                        marker=dict(size=20, color=color, opacity=0), # Invisible markers but hit-testable
                                        line=dict(color=color, width=2),
                                        name=sample_label(s),
                                        fill='tozeroy',
                                        opacity=0.5,
                                        hovertemplate=f"<b>{s}</b><br>Intensity: %{{x}}<br>Density: %{{y}}<extra></extra>"
                                    ))

                        # Update Layout
                        # Range determined by logic above
                        layout_xaxis_range = None # Auto, or use strict
                        if use_log:
                            # strict range from percentiles
                            layout_xaxis_range = [np.log10(plot_min), np.log10(plot_max * 1.5)] # Little extra headroom
                        else:
                            layout_xaxis_range = [0, plot_max * 1.1]

                        # Get Current Value for Line (from State)
                        # We use the state value because we haven't rendered component yet
                        # curr_val_for_line = st.session_state[thr_key]

                        # Add Threshold Line (Visual Feedback)
                        if is_range:
                             rmin = float(st.session_state[min_key])
                             rmax = float(st.session_state[max_key])

                             # Add shaded region
                             fig.add_vrect(x0=rmin, x1=rmax, fillcolor="green", opacity=0.1, layer="below", line_width=0)

                             fig.add_vline(x=rmin, line_width=2, line_dash="dash", line_color="green",
                                          annotation_text=f"Min: {rmin:.1f}", annotation_position="top left")
                             fig.add_vline(x=rmax, line_width=2, line_dash="dash", line_color="green",
                                          annotation_text=f"Max: {rmax:.1f}", annotation_position="top right")
                        else:
                             # Usually Positive = Green/Blue, Negative = Red/Gray.
                             # Let's use Green for Positive, Red for Negative shading.
                             shade_color = "red" if is_negative else "green"

                             curr_val_for_line = float(st.session_state[thr_key])
                             if is_negative:
                                 # Shade from Min to Threshold
                                 # Use plot_min, but ensure it covers the view
                                 range_min = plot_min if plot_min > 0 else (x_domain[0] if len(x_domain)>0 else 0)
                                 if use_log and range_min <= 0:
                                     range_min = 1e-1 # Safety

                                 fig.add_vrect(x0=range_min/10, x1=curr_val_for_line, fillcolor=shade_color, opacity=0.1, layer="below", line_width=0)
                                 annot_pos = "top left"
                             else:
                                 # Shade from Threshold to Max
                                 fig.add_vrect(x0=curr_val_for_line, x1=plot_max*10, fillcolor=shade_color, opacity=0.1, layer="below", line_width=0)
                                 annot_pos = "top right"

                             fig.add_vline(x=curr_val_for_line, line_width=2, line_dash="dash", line_color="red",
                                          annotation_text=f"Threshold: {curr_val_for_line:.1f}", annotation_position=annot_pos)

                        # Dynamic Height and Y-Axis for Ridgeline
                        plot_height = 600
                        y_title = "Density"
                        yaxis_dict: dict[str, Any] = dict(fixedrange=True, showgrid=True, gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', color='black', tickfont=dict(color='black'), title_font=dict(color='black'))

                        if use_ridgeline and len(selected_samples) > 0:
                             total_y_span = 1.0 + (len(selected_samples) - 1) * ridgeline_offset
                             # Increase per-sample height allocation to prevent squeezing
                             plot_height = int(max(400, total_y_span * 200 + 200))

                             y_title = "Samples"
                             yaxis_dict['tickvals'] = [i * ridgeline_offset for i in range(len(selected_samples))]
                             yaxis_dict['ticktext'] = [str(i) for i in range(len(selected_samples))]

                        fig.update_layout(
                            height=plot_height,
                            title=f"Density Plot: {selected_channel} (Subset: {parent_gate})",
                            xaxis_title=selected_channel,
                            yaxis_title=y_title,
                            plot_bgcolor="white",
                            paper_bgcolor="white",
                            # Log axes if requested
                            xaxis_type="log" if use_log else "linear",
                            xaxis_range=layout_xaxis_range,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="black")),
                            hovermode="x unified",
                            clickmode='event+select', # Allow clicking to select points
                            # Dragmode 'select' enables box selection
                            # Fixed range prevents accidental zoom toggling
                            dragmode='select' if is_range else 'zoom',
                            xaxis=dict(fixedrange=True, showgrid=True, gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', color='black', tickfont=dict(color='black'), title_font=dict(color='black')),
                            yaxis=yaxis_dict,
                            font=dict(color="black")
                        )

                        # Capture selection events
                        # on_select="rerun" ensures app updates when a point is selected (clicked)
                        # Enable box selection
                        # Add unique KEY to prevent state leakage
                        with col_plots:
                             st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode=["points", "box"] if is_range else ["points"], key=f"plot_{selected_channel}_{parent_gate}")

                             _, col_add = st.columns([3, 1])
                             if col_add.button("Add This Density Plot to Canvas", key=f"add_analysis_density_{selected_channel}_{parent_gate}"):
                                 # Create Matplotlib Figure for Canvas
                                 import matplotlib.pyplot as plt
                                 mpl_fig, mpl_ax = plt.subplots(figsize=(6, 4))

                                 for i, s in enumerate(selected_samples):
                                     if s in data_map:
                                         vals = data_map[s]
                                         vals = vals[~np.isnan(vals)]
                                         if len(vals) < 2:
                                             continue

                                         try:
                                             if use_log:
                                                 qt_vals = vals[vals > 0]
                                                 if len(qt_vals) < 2:
                                                     continue
                                                 dataset = np.log10(qt_vals)
                                                 kde = gaussian_kde(dataset)
                                                 y_vals = kde(np.log10(x_domain))
                                             else:
                                                 dataset = vals
                                                 kde = gaussian_kde(dataset)
                                                 y_vals = kde(x_domain)

                                             raw_color = colors[i % len(colors)]
                                             # Fix for Plotly 'rgb(r, g, b)' strings which Matplotlib rejects
                                             color = raw_color
                                             if isinstance(raw_color, str) and raw_color.startswith("rgb"):
                                                 try:
                                                     # Extract numbers
                                                     import re
                                                     nums = [float(n) for n in re.findall(r"[-+]?\d*\.\d+|\d+", raw_color)]
                                                     if len(nums) >= 3:
                                                         # Normalize 0-255 to 0-1
                                                         r, g, b = nums[0]/255.0, nums[1]/255.0, nums[2]/255.0
                                                         # Force RGB tuple to ensure no alpha compounding issues with fill_between
                                                         color = (r, g, b)
                                                 except Exception:
                                                     pass

                                             if use_ridgeline:
                                                 # Ridgeline Logic: Normalize and Offset
                                                 y_norm = y_vals
                                                 if np.max(y_norm) > 0:
                                                     y_norm = y_norm / np.max(y_norm)

                                                 base_y = i * ridgeline_offset
                                                 y_shifted = y_norm + base_y

                                                 # Fill comes first (for legend label logic)
                                                 # Use zorder to manage stacking (higher index on top or bottom?)
                                                 # Let's stack naturally
                                                 # Use explicit facecolor and alpha to guarantee visibility
                                                 mpl_ax.fill_between(x_domain, y_shifted, y2=base_y, facecolor=color, alpha=0.6, zorder=10+i, label=sample_label(s))
                                                 # Outline in black/dark on top of fill
                                                 mpl_ax.plot(x_domain, y_shifted, color='black', linewidth=1, zorder=10+i+0.1)

                                             else:
                                                 # Standard Overlay
                                                 mpl_ax.plot(x_domain, y_vals, color=color, label=sample_label(s), linewidth=2, zorder=10)
                                                 try:
                                                     mpl_ax.fill_between(x_domain, y_vals, y2=0, color=color, alpha=0.3, zorder=5)
                                                 except Exception as e:
                                                     st.warning(f"Fill error for {s}: {e}")

                                         except Exception as e:
                                             st.warning(f"Plot error for {s}: {e}")
                                             continue

                                 if use_log:
                                     mpl_ax.set_xscale('log')

                                 if use_ridgeline:
                                     # Remove ticks and labels for cleaner ridgeline
                                     mpl_ax.set_yticks([])
                                     mpl_ax.set_yticklabels([])
                                     mpl_ax.set_ylabel("") # Remove 'Sample' label too

                                     # Adjust y-limit to show top plot
                                     total_height = (len(selected_samples) - 1) * ridgeline_offset + 1.2
                                     mpl_ax.set_ylim(bottom=0, top=total_height)
                                 else:
                                     mpl_ax.set_title(f"Density Plot: {selected_channel} (Subset: {parent_gate})", pad=40)
                                     mpl_ax.set_ylabel("Density")
                                     mpl_ax.set_ylim(bottom=0)

                                 mpl_ax.set_xlabel(selected_channel)
                                 mpl_ax.spines['top'].set_visible(False)
                                 mpl_ax.spines['right'].set_visible(False)

                                 # Add Threshold Line (Background)
                                 if is_range:
                                     rmin = st.session_state[min_key]
                                     rmax = st.session_state[max_key]
                                     mpl_ax.axvline(x=rmin, color='green', linestyle='--', linewidth=2, zorder=3)
                                     mpl_ax.axvline(x=rmax, color='green', linestyle='--', linewidth=2, zorder=3)
                                     mpl_ax.axvspan(rmin, rmax, color='green', alpha=0.1, zorder=1)
                                 else:
                                     mpl_ax.axvline(x=curr_val_for_line, color='red', linestyle='--', linewidth=2, zorder=3)
                                     shade_color = "red" if is_negative else "green"
                                     if is_negative:
                                         mpl_ax.axvspan(x_domain[0], curr_val_for_line, color=shade_color, alpha=0.1, zorder=1)
                                     else:
                                         mpl_ax.axvspan(curr_val_for_line, x_domain[-1], color=shade_color, alpha=0.1, zorder=1)

                                 # Moves legend outside and above plot
                                 mpl_ax.legend(bbox_to_anchor=(0., 1.02), loc='lower left', ncol=1, borderaxespad=0., frameon=False)
                                 mpl_fig.tight_layout()

                                 add_plot_to_canvas(mpl_fig, f"Density Plot: {selected_channel} (Subset: {parent_gate})")
                                 st.success("Plot added to canvas!")
                                 plt.close(mpl_fig)

                        # Finalize Gate
                        with col_plots:
                             st.divider()
                             st.markdown("#### Apply Gate")
                             # Use primary button style for emphasis
                             if st.button("Save & Apply Gate", type="primary"):
                                 # Logic to save the gate
                                 suffix = "+"
                                 if is_range:
                                     suffix = "_rng"
                                 elif is_negative:
                                     suffix = "-"

                                 gate_label = f"{selected_channel}{suffix}"
                                 if parent_gate != "Singlets":
                                     gate_label = f"{parent_gate} -> {selected_channel}{suffix}"

                                 # Determine final gate object structure
                                 new_gate = {
                                     'channel': selected_channel,
                                     'parent': parent_gate,
                                     'label': gate_label,
                                     'type': 'range' if is_range else 'binary',
                                     'direction': 'negative' if is_negative else 'positive',
                                     'samples': selected_samples # Persist the samples used for visualization
                                 }

                                 # Apply Scaling Logic for Storage
                                 # Goal: Always store gates in Dataframe Units (Scaled).
                                 # If apply_scaling is True (Plot is Scaled), input is already Scaled. Do nothing.
                                 # If apply_scaling is False (Plot is Raw), input is Raw. Divide by factor.
                                 final_threshold = curr_threshold
                                 if not apply_scaling and scaling_factor != 1.0:
                                     if isinstance(curr_threshold, (list, tuple, np.ndarray)):
                                         final_threshold = [float(x) / scaling_factor for x in curr_threshold]
                                     else:
                                         final_threshold = float(curr_threshold) / scaling_factor
                                 # If apply_scaling is True, we assume the user defined the gate on the Scaled plot,
                                 # so the value is already in the correct domain for the dataframe.

                                 if is_range and isinstance(final_threshold, (list, tuple, np.ndarray)) and len(final_threshold) >= 2:
                                      new_gate['min'] = final_threshold[0]
                                      new_gate['max'] = final_threshold[1]
                                      # For backward compat or display, we can store threshold as tuple or string representation
                                      new_gate['threshold'] = final_threshold
                                 else:
                                      new_gate['threshold'] = final_threshold

                                 st.session_state.positive_gates.append(new_gate)

                                 # Save to threshold_gate object for plotting usage
                                 st.session_state.threshold_gate[gate_label] = final_threshold

                                 st.session_state.show_gate_creator = False # Hide creator
                                 st.success(f"Added Gate: {new_gate['label']}")
                                 st.rerun()

                             if st.button("Cancel"):
                                 st.session_state.show_gate_creator = False
                                 st.rerun()




        elif analysis_mode == "2D Quadrant":
            # =========================================================
            # 2D QUADRANT LOGIC
            # =========================================================
            with col_opts:
                st.markdown("#### Axis & Samples")
                sel_x = st.selectbox("X Channel", valid_channels, key="quad_x")
                idx_y = 1 if len(valid_channels) > 1 else 0
                sel_y = st.selectbox("Y Channel", valid_channels, index=idx_y, key="quad_y")

                sample_list = full_df['Sample'].unique().tolist()
                def_samp = [sample_list[0]] if sample_list else []
                sel_samps = st.multiselect("Samples", sample_list, default=def_samp, format_func=sample_label, key="quad_samp")

            if sel_x and sel_y and sel_samps:
                # Fetch Data
                d_x = get_channel_data_for_samples(experiment, sel_x, sel_samps, parent_gate, st.session_state.positive_gates, apply_scaling=apply_scaling)
                d_y = get_channel_data_for_samples(experiment, sel_y, sel_samps, parent_gate, st.session_state.positive_gates, apply_scaling=apply_scaling)

                # Combine
                arr_x, arr_y = [], []
                for s in sel_samps:
                    if s in d_x and s in d_y:
                        vx, vy = d_x[s], d_y[s]
                        # Trim to min length if mismatch (unlikely if same dataframe)
                        length = min(len(vx), len(vy))
                        arr_x.extend(vx[:length])
                        arr_y.extend(vy[:length])

                arr_x = np.array(arr_x)
                arr_y = np.array(arr_y)
                valid_mask = (~np.isnan(arr_x)) & (~np.isnan(arr_y))
                arr_x = arr_x[valid_mask]
                arr_y = arr_y[valid_mask]

                if len(arr_x) == 0:
                    st.warning("No data.")
                else:
                    with col_opts:
                         st.markdown("#### Settings")
                         use_log = st.checkbox("Log Scale", value=True, key="quad_log")
                         show_density = st.checkbox("Show Density", value=True, key="quad_density")

                    if use_log:
                        pos_mask = (arr_x > 0) & (arr_y > 0)
                        arr_x_p = arr_x[pos_mask]
                        arr_y_p = arr_y[pos_mask]
                        if len(arr_x_p) == 0:
                            use_log = False
                        else:
                             arr_x, arr_y = arr_x_p, arr_y_p

                    # Init Thresholds
                    qk_x = f"qx_{sel_x}_{sel_y}_{parent_gate}"
                    qk_y = f"qy_{sel_x}_{sel_y}_{parent_gate}"

                    min_x, max_x = np.min(arr_x), np.max(arr_x)
                    min_y, max_y = np.min(arr_y), np.max(arr_y)

                    if qk_x not in st.session_state:
                        st.session_state[qk_x] = (min_x + max_x)/2
                    if qk_y not in st.session_state:
                        st.session_state[qk_y] = (min_y + max_y)/2

                    # Interaction
                    q_plot_key = f"plot_quad_{sel_x}_{sel_y}"
                    if q_plot_key in st.session_state and st.session_state[q_plot_key].get("selection"):
                         sel = st.session_state[q_plot_key]["selection"]
                         if sel.get("points"):
                              pt = sel["points"][0]
                              st.session_state[qk_x] = pt["x"]
                              st.session_state[qk_y] = pt["y"]

                    # Inputs
                    with col_opts:
                         curr_tx = st.number_input(f"{sel_x} Threshold", key=qk_x, format="%.4f")
                         curr_ty = st.number_input(f"{sel_y} Threshold", key=qk_y, format="%.4f")

                    # Scatter Plot
                    # Put calculations and plot in col_plots
                    with col_plots:
                             density_vals = None
                             if show_density and len(arr_x) > 0:
                                 try:
                                     # Histogram-based density (use logicle for negative-safe transform)
                                     dx_calc = logicle_transform(arr_x) if use_log else arr_x
                                     dy_calc = logicle_transform(arr_y) if use_log else arr_y

                                     # Calculate density using percentile range to avoid outlier squeeze
                                     rx = [np.percentile(dx_calc, 0.5), np.percentile(dx_calc, 99.5)]
                                     ry = [np.percentile(dy_calc, 0.5), np.percentile(dy_calc, 99.5)]

                                     # Improved Density Resolution: More bins + Gaussian Smoothing
                                     H, xedges, yedges = np.histogram2d(dx_calc, dy_calc, bins=300, range=[rx, ry])
                                     H = gaussian_filter(H, sigma=2)

                                     # Digitize needs to handle full range, values outside will clip to edge bins
                                     x_bins = np.digitize(dx_calc, xedges) - 1
                                     y_bins = np.digitize(dy_calc, yedges) - 1
                                     x_bins = np.clip(x_bins, 0, H.shape[0]-1)
                                     y_bins = np.clip(y_bins, 0, H.shape[1]-1)
                                     density_vals = H[x_bins, y_bins]
                                 except Exception as e:
                                     print(f"Density Error: {e}")

                             # Downsample for speed
                             if len(arr_x) > 5000:
                                  # Use fixed seed for consistent visualization
                                  rng = np.random.RandomState(42)
                                  indices = rng.choice(len(arr_x), 5000, replace=False)
                                  plot_x, plot_y = arr_x[indices], arr_y[indices]
                                  plot_density = density_vals[indices] if density_vals is not None else None
                             else:
                                  plot_x, plot_y = arr_x, arr_y
                                  plot_density = density_vals

                             # Sort by density for correct z-order (dense on top)
                             if show_density and plot_density is not None:
                                 sort_idx = np.argsort(plot_density)
                                 plot_x = plot_x[sort_idx]
                                 plot_y = plot_y[sort_idx]
                                 plot_density = plot_density[sort_idx]

                                 marker_dict = dict(
                                     size=6,
                                     opacity=0.5,
                                     color=plot_density,
                                     colorscale='Viridis',
                                     showscale=True,
                                     colorbar=dict(title='Density')
                                 )
                             else:
                                 marker_dict = dict(size=6, opacity=0.5, color='slateblue')

                             fig = go.Figure()

                             # Define Quadrant Shapes (Colored Backgrounds)
                             x_span = max_x - min_x
                             y_span = max_y - min_y
                             if use_log:
                                 b_xmin = min_x / 5
                                 b_xmax = max_x * 5
                                 b_ymin = min_y / 5
                                 b_ymax = max_y * 5
                             else:
                                 pad_x = x_span * 0.1 if x_span > 0 else 1.0
                                 pad_y = y_span * 0.1 if y_span > 0 else 1.0
                                 b_xmin = min_x - pad_x
                                 b_xmax = max_x + pad_x
                                 b_ymin = min_y - pad_y
                                 b_ymax = max_y + pad_y

                             # Q1: UL (X-, Y+) - Top Left
                             fig.add_shape(type="rect",
                                 x0=b_xmin, y0=curr_ty, x1=curr_tx, y1=b_ymax,
                                 fillcolor="rgba(255, 99, 71, 0.15)", line_width=0, layer="below"
                             )
                             # Q2: UR (X+, Y+) - Top Right
                             fig.add_shape(type="rect",
                                 x0=curr_tx, y0=curr_ty, x1=b_xmax, y1=b_ymax,
                                 fillcolor="rgba(60, 179, 113, 0.15)", line_width=0, layer="below"
                             )
                             # Q3: LR (X+, Y-) - Bottom Right
                             fig.add_shape(type="rect",
                                 x0=curr_tx, y0=b_ymin, x1=b_xmax, y1=curr_ty,
                                 fillcolor="rgba(100, 149, 237, 0.15)", line_width=0, layer="below"
                             )
                             # Q4: LL (X-, Y-) - Bottom Left
                             fig.add_shape(type="rect",
                                 x0=b_xmin, y0=b_ymin, x1=curr_tx, y1=curr_ty,
                                 fillcolor="rgba(128, 128, 128, 0.15)", line_width=0, layer="below"
                             )

                             fig.add_trace(go.Scattergl(
                                 x=plot_x, y=plot_y, mode='markers',
                                 marker=marker_dict,
                                 name='Events'
                             ))

                             # Add Quadrant Labels
                             fig.add_annotation(x=0.02, y=0.98, xref="paper", yref="paper", text="Q1", showarrow=False, font=dict(size=20, color="black"), xanchor="left", yanchor="top")
                             fig.add_annotation(x=0.98, y=0.98, xref="paper", yref="paper", text="Q2", showarrow=False, font=dict(size=20, color="black"), xanchor="right", yanchor="top")
                             fig.add_annotation(x=0.98, y=0.02, xref="paper", yref="paper", text="Q3", showarrow=False, font=dict(size=20, color="black"), xanchor="right", yanchor="bottom")
                             fig.add_annotation(x=0.02, y=0.02, xref="paper", yref="paper", text="Q4", showarrow=False, font=dict(size=20, color="black"), xanchor="left", yanchor="bottom")

                             # Add Lines (Crosshair)
                             fig.add_vline(x=curr_tx, line_color="black", line_width=1, line_dash="dash")
                             fig.add_hline(y=curr_ty, line_color="black", line_width=1, line_dash="dash")

                             # Calculate plot range based on data
                             if use_log:
                                 # Range in log10 units for Plotly log axes
                                 # Add slight padding (0.1 log units is factor of ~1.25)
                                 pad = 0.2
                                 range_x = [np.log10(min_x) - pad, np.log10(max_x) + pad]
                                 range_y = [np.log10(min_y) - pad, np.log10(max_y) + pad]
                             else:
                                 pad_x = (max_x - min_x) * 0.05
                                 range_x = [min_x - pad_x, max_x + pad_x]
                                 pad_y = (max_y - min_y) * 0.05
                                 range_y = [min_y - pad_y, max_y + pad_y]

                             fig.update_layout(
                                 title=f"Quadrant: {sel_x} vs {sel_y}",
                                 plot_bgcolor="white",
                                 paper_bgcolor="white",
                                 height=800,
                                 xaxis_title=sel_x, yaxis_title=sel_y,
                                 xaxis_type="log" if use_log else "linear",
                                 yaxis_type="log" if use_log else "linear",
                                 xaxis_range=range_x,
                                 yaxis_range=range_y,
                                 dragmode="pan", clickmode="event+select",
                                 font=dict(color="black"),
                                 xaxis=dict(color='black', tickfont=dict(color='black'), title_font=dict(color='black'), showgrid=True, gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey'),
                                 yaxis=dict(color='black', tickfont=dict(color='black'), title_font=dict(color='black'), showgrid=True, gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey')
                             )

                             st.plotly_chart(fig, width="stretch", on_select="rerun", key=q_plot_key)

                    # Add to Canvas & Controls to Options
                    with col_opts:
                        st.markdown("#### Actions")
                        if st.button("Add This Scatter Plot to Canvas", key=f"add_analysis_quad_{sel_x}_{sel_y}_{parent_gate}"):
                            import matplotlib.pyplot as plt
                            mpl_fig, mpl_ax = plt.subplots(figsize=(6, 6))

                            if show_density and density_vals is not None:
                                 # Combine all for density plot to match Plotly view
                                 # We have plot_x and plot_y and plot_density already prepared (downsampled) or full
                                 # But plot_density is downsampled or full from 'arr_x'.
                                 # For consistency, let's use the same 'plot_x', 'plot_y', 'plot_density' used in Plotly
                                 sc = mpl_ax.scatter(plot_x, plot_y, c=plot_density, cmap='viridis', s=10, alpha=0.6, edgecolors="none")
                                 plt.colorbar(sc, ax=mpl_ax, label='Density')
                            else:
                                colors_c = px.colors.qualitative.Plotly
                                for i, s in enumerate(sel_samps):
                                     if s in d_x and s in d_y:
                                          vx, vy = d_x[s], d_y[s]
                                          l_min = min(len(vx), len(vy))
                                          vx = vx[:l_min]
                                          vy = vy[:l_min]
                                          mask = (~np.isnan(vx)) & (~np.isnan(vy))
                                          if use_log:
                                              mask = mask & (vx > 0) & (vy > 0)
                                          vx = vx[mask]
                                          vy = vy[mask]
                                          if len(vx) > 0:
                                               if len(vx) > 2000:
                                                    rng = np.random.RandomState(42)
                                                    idx = rng.choice(len(vx), 2000, replace=False)
                                                    vx = vx[idx]
                                                    vy = vy[idx]
                                               c = colors_c[i%len(colors_c)]
                                               mpl_ax.scatter(vx, vy, s=10, color=c, alpha=0.6, label=sample_label(s), edgecolors="none")
                                mpl_ax.legend()

                            if use_log:
                                mpl_ax.set_xscale("log")
                                mpl_ax.set_yscale("log")
                            mpl_ax.set_title(f"{sel_x} vs {sel_y}")
                            mpl_ax.set_xlabel(sel_x)
                            mpl_ax.set_ylabel(sel_y)
                            mpl_ax.axvline(x=curr_tx, color="black", linestyle="--")
                            mpl_ax.axhline(y=curr_ty, color="black", linestyle="--")

                            mpl_fig.tight_layout()
                            add_plot_to_canvas(mpl_fig, f"Scatter: {sel_x} vs {sel_y}")
                            st.success("Added to Canvas!")
                            plt.close(mpl_fig)

                        st.markdown("#### Save Quadrants")
                        st.write("Select which populations to create:")

                        # Nested columns for quadrant selection inside options
                        if True: # Just to match indent level flow

                         sq1, sq2 = st.columns(2)
                         save_UL = sq1.checkbox(f"Q1: {sel_x}- {sel_y}+", value=True)
                         save_UR = sq2.checkbox(f"Q2: {sel_x}+ {sel_y}+", value=True)
                         sq3, sq4 = st.columns(2)
                         save_LR = sq3.checkbox(f"Q3: {sel_x}+ {sel_y}-", value=True)
                         save_LL = sq4.checkbox(f"Q4: {sel_x}- {sel_y}-", value=True)

                         if st.button("Create Selected Gates"):
                             selected_quads = []
                             suffixes = []
                             if save_UL:
                                 selected_quads.append("-+")
                                 suffixes.append("-+")
                             if save_UR:
                                 selected_quads.append("++")
                                 suffixes.append("++")
                             if save_LR:
                                 selected_quads.append("+-")
                                 suffixes.append("+-")
                             if save_LL:
                                 selected_quads.append("--")
                                 suffixes.append("--")

                             if selected_quads:
                                 # Combined Label
                                 if len(suffixes) == 1:
                                     lbl_suffix = f"{sel_x}{suffixes[0][0]} {sel_y}{suffixes[0][1]}"
                                 else:
                                     lbl_suffix = f"{sel_x}/{sel_y} Combined"

                                 lbl = f"{parent_gate} -> {lbl_suffix}" if parent_gate!='Singlets' else lbl_suffix

                                 final_tx = curr_tx
                                 final_ty = curr_ty
                                 # Standardize to Scaled Units
                                 # If not apply_scaling (Raw), divide by factor.
                                 if not apply_scaling and scaling_factor != 1.0:
                                     final_tx /= scaling_factor
                                     final_ty /= scaling_factor

                                 new_g = {
                                     'type': 'quadrant',
                                     'parent': parent_gate,
                                     'label': lbl,
                                     'channel_x': sel_x, 'channel_y': sel_y,
                                     'threshold_x': final_tx, 'threshold_y': final_ty,
                                     'quadrant': selected_quads,
                                     'samples': sel_samps
                                 }
                                 st.session_state.positive_gates.append(new_g)

                             st.session_state.show_gate_creator = False
                             st.rerun()

                         if st.button("Cancel Quad"):
                              st.session_state.show_gate_creator = False
                              st.rerun()

        elif analysis_mode == "2D Polygon":
            # =========================================================
            # 2D POLYGON LOGIC
            # =========================================================
            with col_opts:
                st.markdown("#### Axis & Samples")
                sel_x = st.selectbox("X Channel", valid_channels, key="poly_x")
                idx_y = 1 if len(valid_channels) > 1 else 0
                sel_y = st.selectbox("Y Channel", valid_channels, index=idx_y, key="poly_y")

                sample_list = full_df['Sample'].unique().tolist()
                def_samp = [sample_list[0]] if sample_list else []
                sel_samps = st.multiselect("Samples", sample_list, default=def_samp, format_func=sample_label, key="poly_samp")

            if sel_x and sel_y and sel_samps:
                # Fetch Data
                d_x = get_channel_data_for_samples(experiment, sel_x, sel_samps, parent_gate, st.session_state.positive_gates, apply_scaling=apply_scaling)
                d_y = get_channel_data_for_samples(experiment, sel_y, sel_samps, parent_gate, st.session_state.positive_gates, apply_scaling=apply_scaling)

                # Combine
                arr_x, arr_y = [], []
                for s in sel_samps:
                    if s in d_x and s in d_y:
                        vx, vy = d_x[s], d_y[s]
                        length = min(len(vx), len(vy))
                        arr_x.extend(vx[:length])
                        arr_y.extend(vy[:length])

                arr_x = np.array(arr_x)
                arr_y = np.array(arr_y)
                valid_mask = (~np.isnan(arr_x)) & (~np.isnan(arr_y))
                arr_x = arr_x[valid_mask]
                arr_y = arr_y[valid_mask]

                if len(arr_x) == 0:
                    st.warning("No data.")
                else:
                    with col_opts:
                        st.markdown("#### Settings")
                        use_log = st.checkbox("Log Scale", value=True, key="poly_log")
                        show_density = st.checkbox("Show Density", value=True, key="poly_density")

                    if use_log:
                        pos_mask = (arr_x > 0) & (arr_y > 0)
                        arr_x_p = arr_x[pos_mask]
                        arr_y_p = arr_y[pos_mask]
                        if len(arr_x_p) == 0:
                            use_log = False
                        else:
                             arr_x, arr_y = arr_x_p, arr_y_p

                    # Manage Vertices
                    poly_key = f"poly_verts_{sel_x}_{sel_y}_{parent_gate}"
                    if poly_key not in st.session_state:
                        st.session_state[poly_key] = []

                    # Graph Interaction
                    plot_key = f"plot_poly_{sel_x}_{sel_y}"

                    # Retrieve Selection
                    if plot_key in st.session_state and st.session_state[plot_key].get("selection"):
                         sel = st.session_state[plot_key]["selection"]
                         if sel.get("points"):
                              pts = sel["points"]
                              current_verts = st.session_state[poly_key]
                              new_pts_list = []
                              for p in pts:
                                   px_v, py_v = p["x"], p["y"]
                                   is_new = True
                                   for existing in current_verts:
                                        if abs(existing[0]-px_v) < 1e-5 and abs(existing[1]-py_v) < 1e-5:
                                             is_new = False
                                             break
                                   if is_new:
                                        new_pts_list.append((px_v, py_v))
                              if new_pts_list:
                                   st.session_state[poly_key].extend(new_pts_list)

                    # Control Buttons
                    with col_opts:
                         st.markdown("#### Vertices")
                         col_b1, col_b2 = st.columns(2)
                         with col_b1:
                              if st.button("Clear Vertices"):
                                   st.session_state[poly_key] = []
                                   st.rerun()
                              if st.button("Remove Last Point"):
                                   if st.session_state[poly_key]:
                                        st.session_state[poly_key].pop()
                                        st.rerun()

                         with col_b2:
                              st.info(f"Points Selected: {len(st.session_state[poly_key])}")

                    # Scatter Plot with Polygon Overlay
                    with col_plots:
                        density_vals = None
                        if show_density and len(arr_x) > 0:
                            try:
                                # Histogram-based density (use logicle for negative-safe transform)
                                dx_calc = logicle_transform(arr_x) if use_log else arr_x
                                dy_calc = logicle_transform(arr_y) if use_log else arr_y

                                # Calculate density using percentile range
                                rx = [np.percentile(dx_calc, 0.5), np.percentile(dx_calc, 99.5)]
                                ry = [np.percentile(dy_calc, 0.5), np.percentile(dy_calc, 99.5)]

                                # Improved Density Resolution: More bins + Gaussian Smoothing
                                H, xedges, yedges = np.histogram2d(dx_calc, dy_calc, bins=300, range=[rx, ry])
                                H = gaussian_filter(H, sigma=2)

                                x_bins = np.digitize(dx_calc, xedges) - 1
                                y_bins = np.digitize(dy_calc, yedges) - 1
                                x_bins = np.clip(x_bins, 0, H.shape[0]-1)
                                y_bins = np.clip(y_bins, 0, H.shape[1]-1)
                                density_vals = H[x_bins, y_bins]
                            except Exception as e:
                                print(f"Density Error: {e}")

                        if len(arr_x) > 5000:
                             rng = np.random.RandomState(42)
                             indices = rng.choice(len(arr_x), 5000, replace=False)
                             plot_x, plot_y = arr_x[indices], arr_y[indices]
                             plot_density = density_vals[indices] if density_vals is not None else None
                        else:
                             plot_x, plot_y = arr_x, arr_y
                             plot_density = density_vals

                        # Sort by density for correct z-order (dense on top)
                        if show_density and plot_density is not None:
                            sort_idx = np.argsort(plot_density)
                            plot_x = plot_x[sort_idx]
                            plot_y = plot_y[sort_idx]
                            plot_density = plot_density[sort_idx]

                            marker_dict = dict(
                                size=6,
                                opacity=0.5,
                                # ... existing code ...
                                color=plot_density,
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(title='Density')
                            )
                        else:
                            marker_dict = dict(size=6, opacity=0.5, color='slateblue')

                        fig = go.Figure()

                        # 1. Data Trace
                        fig.add_trace(go.Scattergl(
                            x=plot_x, y=plot_y, mode='markers',
                            marker=marker_dict,
                            name='Events',
                            hoverinfo='none'
                        ))

                        # 2. Polygon Trace
                        verts = st.session_state[poly_key]
                        if len(verts) > 0:
                             if len(verts) >= 3:
                                  vx = [v[0] for v in verts] + [verts[0][0]]
                                  vy = [v[1] for v in verts] + [verts[0][1]]
                             else:
                                  vx = [v[0] for v in verts]
                                  vy = [v[1] for v in verts]

                             fig.add_trace(go.Scattergl(
                                  x=vx, y=vy, mode='lines+markers',
                                  line=dict(color='orange', width=2),
                                  marker=dict(size=6, color='orange'),
                                  name='Polygon'
                             ))

                        # Calculate plot range based on data for consistent scaling
                        min_x, max_x = np.min(arr_x), np.max(arr_x)
                        min_y, max_y = np.min(arr_y), np.max(arr_y)

                        if use_log:
                            pad = 0.2
                            range_x = [np.log10(min_x) - pad, np.log10(max_x) + pad]
                            range_y = [np.log10(min_y) - pad, np.log10(max_y) + pad]
                        else:
                            pad_x = (max_x - min_x) * 0.05
                            range_x = [min_x - pad_x, max_x + pad_x]
                            pad_y = (max_y - min_y) * 0.05
                            range_y = [min_y - pad_y, max_y + pad_y]

                        fig.update_layout(
                            title=f"Polygon: {sel_x} vs {sel_y} (Click on points to define vertices)",
                            plot_bgcolor="white",
                            paper_bgcolor="white",
                            height=800,
                            xaxis_title=sel_x, yaxis_title=sel_y,
                            xaxis_type="log" if use_log else "linear",
                            yaxis_type="log" if use_log else "linear",
                            xaxis_range=range_x,
                            yaxis_range=range_y,
                            dragmode="pan",
                            clickmode="event+select",
                            font=dict(color="black"),
                            xaxis=dict(color='black', tickfont=dict(color='black'), title_font=dict(color='black'), showgrid=True, gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey'),
                            yaxis=dict(color='black', tickfont=dict(color='black'), title_font=dict(color='black'), showgrid=True, gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey'),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )

                        st.plotly_chart(fig, width="stretch", on_select="rerun", key=plot_key, selection_mode="points")

                    # Add to Canvas Button & Controls
                    with col_opts:
                        st.markdown("#### Actions")
                        if st.button("Add This Scatter Plot to Canvas", key=f"add_analysis_poly_{sel_x}_{sel_y}_{parent_gate}"):
                            import matplotlib.pyplot as plt
                            mpl_fig, mpl_ax = plt.subplots(figsize=(6, 6))

                            if show_density and density_vals is not None:
                                 # Use pre-calculated density points
                                 sc = mpl_ax.scatter(plot_x, plot_y, c=plot_density, cmap='viridis', s=10, alpha=0.6, edgecolors="none")
                                 plt.colorbar(sc, ax=mpl_ax, label='Density')
                            else:
                                colors_c = px.colors.qualitative.Plotly
                                for i, s in enumerate(sel_samps):
                                     if s in d_x and s in d_y:
                                          vx, vy = d_x[s], d_y[s]
                                          l_min = min(len(vx), len(vy))
                                          vx = vx[:l_min]
                                          vy = vy[:l_min]
                                          mask = (~np.isnan(vx)) & (~np.isnan(vy))
                                          if use_log:
                                              mask = mask & (vx > 0) & (vy > 0)
                                          vx = vx[mask]
                                          vy = vy[mask]
                                          if len(vx) > 0:
                                               if len(vx) > 2000:
                                                    rng = np.random.RandomState(42)
                                                    idx = rng.choice(len(vx), 2000, replace=False)
                                                    vx = vx[idx]
                                                    vy = vy[idx]
                                               c = colors_c[i%len(colors_c)]
                                               mpl_ax.scatter(vx, vy, s=10, color=c, alpha=0.6, label=sample_label(s), edgecolors="none")
                                mpl_ax.legend()

                            if use_log:
                                 mpl_ax.set_xscale("log")
                                 mpl_ax.set_yscale("log")

                            mpl_ax.set_title(f"Polygon: {sel_x} vs {sel_y}")
                            mpl_ax.set_xlabel(sel_x)
                            mpl_ax.set_ylabel(sel_y)

                            # Plot Polygon
                            if len(verts) > 0:
                                 if len(verts) >= 3:
                                      vx_poly = [v[0] for v in verts] + [verts[0][0]]
                                      vy_poly = [v[1] for v in verts] + [verts[0][1]]
                                 else:
                                      vx_poly = [v[0] for v in verts]
                                      vy_poly = [v[1] for v in verts]
                                 mpl_ax.plot(vx_poly, vy_poly, color='orange', linewidth=2, marker='o', markersize=4)

                            mpl_fig.tight_layout()
                            add_plot_to_canvas(mpl_fig, f"Polygon Scatter: {sel_x} vs {sel_y}")
                            st.success("Added to Canvas!")
                            plt.close(mpl_fig)

                        st.write("#### Save Polygon Gate")
                        if len(verts) >= 3:
                             if st.button("Create Gate"):
                                  lbl = f"{parent_gate} -> {sel_x}:{sel_y} Poly" if parent_gate!='Singlets' else f"{sel_x}:{sel_y} Poly"

                                  final_verts = verts
                                  # Standardize to Scaled Units
                                  if not apply_scaling and scaling_factor != 1.0:
                                      final_verts = []
                                      for v in verts:
                                          # v is [x, y] list
                                          final_verts.append([v[0] / scaling_factor, v[1] / scaling_factor])

                                  new_g = {
                                      'type': 'polygon',
                                      'parent': parent_gate,
                                      'label': lbl,
                                      'channel_x': sel_x, 'channel_y': sel_y,
                                      'vertices': final_verts,
                                      'samples': sel_samps
                                  }
                                  st.session_state.positive_gates.append(new_g)
                                  del st.session_state[poly_key]
                                  st.session_state.show_gate_creator = False
                                  st.rerun()
                        else:
                             st.warning("Select at least 3 points to form a polygon.")

                        if st.button("Cancel Polygon"):
                             st.session_state.show_gate_creator = False
                             st.rerun()
    # -------------------------------------------------------------------------
    # 4. Navigation
    # -------------------------------------------------------------------------
    st.divider()
    if st.button("Finish Analysis & Proceed to Plotting", type="primary"):
        st.session_state.analysis_complete = True
        st.session_state.current_tab = "Plot"
        st.rerun()
