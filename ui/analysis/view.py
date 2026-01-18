import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import gaussian_kde
from glow.analysis.stats import get_summary_statistics
from glow.analysis.positive import calculate_positive_stats, get_channel_data_for_samples, calculate_positive_stats_recursive
from glow.canvas.functions import add_plot_to_canvas

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
        lbl = g.get('label', f"{g['channel']}+")
        st.session_state.threshold_gate[lbl] = g['threshold']

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
            with st.expander(f"Gate {i+1}: {g.get('label', g['channel']+'+')} (Parent: {g.get('parent', 'Singlets')})"):
                st.write(f"**Channel:** {g['channel']}")

                # Check Type
                if g.get('type') == 'range':
                     st.write(f"**Range:** {g.get('min'):.2f} - {g.get('max'):.2f}")
                     st.write(f"**Filter Path:** {g.get('parent', 'Singlets')} -> {g['channel']} in [{g.get('min'):.2f}, {g.get('max'):.2f}]")
                else:
                     op = ">" if g.get('direction', 'positive') == 'positive' else "<="
                     st.write(f"**Threshold:** {g['threshold']:.2f} ({g.get('direction', 'positive')})")
                     st.write(f"**Filter Path:** {g.get('parent', 'Singlets')} -> {g['channel']} {op} {g['threshold']:.2f}")

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

        # --- PARENT SELECTION ---
        # Options: Singlets + any defined gates
        parent_options = ['Singlets'] + [g['label'] for g in st.session_state.positive_gates]
        parent_gate = st.selectbox("Select Parent Population", parent_options, index=0)

        # --- CHANNEL SELECTION ---
        # Identify all numeric channels in the dataset
        all_numeric = [c for c in experiment.data.columns if pd.api.types.is_numeric_dtype(experiment.data[c])]

        # Filter out system flags and internal bookkeeping columns
        ignore_cols = ['cells', 'singlets', 'Is_Live', 'Is_Singlet', 'LiveGate', 'SingletGate', 'Density']

        # Get selected channels from Gating tab
        # Use persistent state to ensure it survives tab switching
        user_selected = st.session_state.get('persistent_selected_channels')

        # Define valid channels: must be in user's selection and not ignored
        if user_selected is not None:
            valid_channels = sorted([
                c for c in all_numeric
                if c not in ignore_cols and c in user_selected
            ])
        else:
            # Fallback if gating wasn't visited - hide scatter by default
            valid_channels = sorted([
                c for c in all_numeric
                if c not in ignore_cols and not any(x in c.upper() for x in ["FSC", "SSC", "TIME"])
            ])

        # Final safety check: if everything was filtered out (e.g. only FSC/SSC exist), show them.
        if not valid_channels:
            valid_channels = sorted([c for c in all_numeric if c not in ignore_cols])

        # Use columns for layout
        c1, c2 = st.columns([1, 2])

        with c1:
            # Use a key to prevent reset issues, though usually fine
            selected_channel = st.selectbox("Select X-Axis Channel", valid_channels, key="analysis_channel_select")

        if selected_channel:
            # B. Sample Selection for Visualization
            sample_list = full_df['Sample'].unique().tolist()
            # Default to the first sample
            default_samples = [sample_list[0]] if sample_list else []

            with c2:
                selected_samples = st.multiselect("Select Samples to Overlay", sample_list, default=default_samples, key="analysis_sample_select")

            if selected_samples:
                # Fetch Data, Filtered by Parent gate
                data_map = get_channel_data_for_samples(experiment, selected_channel, selected_samples, parent_gate_name=parent_gate, defined_gates=st.session_state.positive_gates)

                # determine range for slider
                all_values = []
                for s in data_map:
                    # Add check for valid arrays
                    if len(data_map[s]) > 0:
                         all_values.extend(data_map[s])

                # Robust NaN filtering
                all_values = np.array([x for x in all_values if not np.isnan(x)])

                if len(all_values) == 0:
                    st.warning("No valid data found for selected samples.")
                else:
                    # Checkbox for log scale
                    use_log = st.checkbox("Log Scale Plot", value=True)

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
                            plot_min = float(np.percentile(valid_data_for_range, 0.1)) # 0.1th percentile
                            plot_max = float(np.percentile(valid_data_for_range, 99.9)) # 99.9th percentile
                    else:
                        valid_data_for_range = all_values
                        # For Linear, user requested min to be 0
                        plot_min = 0.0
                        # Max 99.9th percentile to catch main pop
                        plot_max = float(np.percentile(valid_data_for_range, 99.9))
                        if plot_max <= 0: plot_max = 1.0

                    # Ensure non-singular
                    if plot_min >= plot_max:
                        plot_max = plot_min + 1.0

                    # C. Threshold Selection
                    st.write("### Threshold Selection")

                    # 1. Gate Type Selection
                    col_type, col_dir = st.columns([2, 1])
                    with col_type:
                        gate_type = st.radio("Gate Type", ["Binary (Threshold)", "Range (Min - Max)"], horizontal=True)

                    is_range = (gate_type == "Range (Min - Max)")

                    is_negative = False
                    if not is_range:
                        with col_dir:
                            gate_direction = st.radio("Population", ["Positive (>)", "Negative (<)"], horizontal=True)
                            is_negative = (gate_direction == "Negative (<)")

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

                    # Checkbounds for single threshold
                    if st.session_state[thr_key] < plot_min: st.session_state[thr_key] = plot_min
                    if st.session_state[thr_key] > plot_max: st.session_state[thr_key] = plot_max

                    # Create placeholder for input widget to be rendered AFTER chart interaction logic
                    # input_placeholder = st.empty() # REMOVED: Render directly to avoid flickering

                    # REMOVED: Render Inputs Immediately (using current Session State)
                    # if is_range:
                    #      col_min, col_max = st.columns(2)
                    #      with col_min:
                    #           curr_min = st.number_input("Min Threshold", key=min_key, min_value=float(plot_min), max_value=float(plot_max))
                    #      with col_max:
                    #           curr_max = st.number_input("Max Threshold", key=max_key, min_value=float(plot_min), max_value=float(plot_max))
                    #      curr_threshold = (curr_min, curr_max)
                    # else:
                    #      curr_val = st.number_input("Threshold Value", key=thr_key, min_value=float(plot_min), max_value=float(plot_max))
                    #      curr_threshold = curr_val

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
                         col_min, col_max = st.columns(2)
                         with col_min:
                              # Clip values to ensure they are within the dynamic bounds allowed by the widget
                              # We use plot_max * 10 to allow user to go beyond percentile limits if they type it
                              curr_min = st.number_input("Min Threshold", key=min_key, 
                                                       min_value=float(min(plot_min, st.session_state[min_key])), 
                                                       max_value=float(max(plot_max*2, st.session_state[min_key])))
                         with col_max:
                              curr_max = st.number_input("Max Threshold", key=max_key, 
                                                       min_value=float(min(plot_min, st.session_state[max_key])), 
                                                       max_value=float(max(plot_max*2, st.session_state[max_key])))
                         curr_threshold = (curr_min, curr_max)
                    else:
                         # For binary mode, use a single point selection logic
                         curr_val = st.number_input("Threshold Value", key=thr_key, 
                                                  min_value=float(min(plot_min, st.session_state[thr_key])), 
                                                  max_value=float(max(plot_max*2, st.session_state[thr_key])))
                         curr_threshold = curr_val

                    # D. Interactive KDE Plot
                    fig = go.Figure()

                    # Add an invisible "Click Catcher" trace for binary mode to make the whole plot area clickable
                    # (Plotly point selection only triggers on actual traces)
                    if not is_range:
                        fig.add_trace(go.Scatter(
                            x=x_domain,
                            y=[0] * len(x_domain),
                            mode='markers',
                            marker=dict(size=10, opacity=0),
                            showlegend=False,
                            hoverinfo='none',
                            name="click_catcher"
                        ))

                    # Colors
                    colors = px.colors.qualitative.Plotly


                    # Generate support points for KDE
                    # If Log: generate in log space then transform back
                    if use_log:
                        # Log filtering ensures plot_min > 0
                        if plot_min <= 0: plot_min = np.min(valid_data_for_range[valid_data_for_range > 0])

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
                                if len(qt_vals) < 2: continue
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

                            fig.add_trace(go.Scatter(
                                x=x_domain,
                                y=y_vals,
                                mode='lines+markers', # Markers needed for click targeting
                                marker=dict(size=0.1, color=color, opacity=0), # Invisible markers but hit-testable
                                line=dict(color=color, width=2),
                                name=s,
                                fill='tozeroy',
                                opacity=0.6,
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
                    curr_val_for_line = st.session_state[thr_key]

                    # Add Threshold Line (Visual Feedback)
                    if is_range:
                         rmin = st.session_state[min_key]
                         rmax = st.session_state[max_key]

                         # Add shaded region
                         fig.add_vrect(x0=rmin, x1=rmax, fillcolor="green", opacity=0.1, layer="below", line_width=0)

                         fig.add_vline(x=rmin, line_width=2, line_dash="dash", line_color="green",
                                      annotation_text=f"Min: {rmin:.1f}", annotation_position="top left")
                         fig.add_vline(x=rmax, line_width=2, line_dash="dash", line_color="green",
                                      annotation_text=f"Max: {rmax:.1f}", annotation_position="top right")
                    else:
                         fill_color = "red" if is_negative else "blue" # Use red for neg? Or vice versa. Let's use standard convention.
                         # Usually Positive = Green/Blue, Negative = Red/Gray.
                         # Let's use Green for Positive, Red for Negative shading.
                         shade_color = "red" if is_negative else "green"

                         if is_negative:
                             # Shade from Min to Threshold
                             # Use plot_min, but ensure it covers the view
                             range_min = plot_min if plot_min > 0 else (x_domain[0] if len(x_domain)>0 else 0)
                             if use_log and range_min <= 0: range_min = 1e-1 # Safety

                             fig.add_vrect(x0=range_min/10, x1=curr_val_for_line, fillcolor=shade_color, opacity=0.1, layer="below", line_width=0)
                             annot_pos = "top left"
                         else:
                             # Shade from Threshold to Max
                             fig.add_vrect(x0=curr_val_for_line, x1=plot_max*10, fillcolor=shade_color, opacity=0.1, layer="below", line_width=0)
                             annot_pos = "top right"

                         fig.add_vline(x=curr_val_for_line, line_width=2, line_dash="dash", line_color="red",
                                      annotation_text=f"Threshold: {curr_val_for_line:.1f}", annotation_position=annot_pos)

                    fig.update_layout(
                        title=f"Density Plot: {selected_channel} (Subset: {parent_gate})",
                        xaxis_title=selected_channel,
                        yaxis_title="Density",
                        # Log axes if requested
                        xaxis_type="log" if use_log else "linear",
                        xaxis_range=layout_xaxis_range,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        hovermode="closest",
                        clickmode='event+select', # Allow clicking to select points
                        # Dragmode 'select' enables box selection
                        # Fixed range prevents accidental zoom toggling
                        dragmode='select',
                        xaxis=dict(fixedrange=True),
                        yaxis=dict(fixedrange=True),
                    )

                    # Capture selection events
                    # on_select="rerun" ensures app updates when a point is selected (clicked)
                    # Enable box selection
                    # Add unique KEY to prevent state leakage
                    selection = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode=["points", "box"], key=f"plot_{selected_channel}_{parent_gate}")

                    if st.button("Add This Density Plot to Canvas", key=f"add_analysis_density_{selected_channel}_{parent_gate}"):
                        # Create Matplotlib Figure for Canvas
                        import matplotlib.pyplot as plt
                        mpl_fig, mpl_ax = plt.subplots(figsize=(6, 4))

                        for i, s in enumerate(selected_samples):
                            if s in data_map:
                                vals = data_map[s]
                                vals = vals[~np.isnan(vals)]
                                if len(vals) < 2: continue

                                try:
                                    if use_log:
                                        qt_vals = vals[vals > 0]
                                        if len(qt_vals) < 2: continue
                                        dataset = np.log10(qt_vals)
                                        kde = gaussian_kde(dataset)
                                        y_vals = kde(np.log10(x_domain))
                                    else:
                                        dataset = vals
                                        kde = gaussian_kde(dataset)
                                        y_vals = kde(x_domain)

                                    color = colors[i % len(colors)]
                                    mpl_ax.plot(x_domain, y_vals, color=color, label=s, linewidth=2, zorder=10)

                                    # Isolate fill to catch specific errors
                                    try:
                                        mpl_ax.fill_between(x_domain, y_vals, y2=0, color=color, alpha=0.5, zorder=5)
                                    except Exception as e:
                                        st.warning(f"Fill error for {s}: {e}")
                                except Exception as e:
                                    st.warning(f"Plot error for {s}: {e}")
                                    continue

                        if use_log:
                            mpl_ax.set_xscale('log')

                        mpl_ax.set_title(f"Density Plot: {selected_channel} (Subset: {parent_gate})")
                        mpl_ax.set_xlabel(selected_channel)
                        mpl_ax.set_ylabel("Density")
                        mpl_ax.set_ylim(bottom=0)
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

                        mpl_ax.legend()
                        mpl_fig.tight_layout()

                        add_plot_to_canvas(mpl_fig, f"Density Plot: {selected_channel} (Subset: {parent_gate})")
                        st.success("Plot added to canvas!")
                        plt.close(mpl_fig)

                    # Finalize Gate
                    if st.button("Save & Apply Gate"):
                        # Logic to save the gate
                        suffix = "+"
                        if is_range: suffix = "_rng"
                        elif is_negative: suffix = "-"

                        gate_label = f"{selected_channel}{suffix}"
                        if parent_gate != "Singlets":
                            gate_label = f"{parent_gate} -> {selected_channel}{suffix}"

                        # Determine final gate object structure
                        new_gate = {
                            'channel': selected_channel,
                            'parent': parent_gate,
                            'label': gate_label,
                            'type': 'range' if is_range else 'binary',
                            'direction': 'negative' if is_negative else 'positive'
                        }

                        if is_range:
                             new_gate['min'] = curr_threshold[0]
                             new_gate['max'] = curr_threshold[1]
                             # For backward compat or display, we can store threshold as tuple or string representation
                             new_gate['threshold'] = curr_threshold
                        else:
                             new_gate['threshold'] = curr_threshold

                        st.session_state.positive_gates.append(new_gate)

                        # Save to threshold_gate object for plotting usage
                        st.session_state.threshold_gate[gate_label] = curr_threshold

                        st.session_state.show_gate_creator = False # Hide creator
                        st.success(f"Added Gate: {new_gate['label']}")
                        st.rerun()

                    with ac2:
                        if st.button("Cancel"):
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
