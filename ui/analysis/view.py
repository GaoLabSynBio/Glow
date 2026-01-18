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
        if 'channel' in g:
            lbl = g.get('label', f"{g['channel']}+")
        else:
            lbl = g.get('label', "Gate")

        if 'threshold' in g:
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
                     st.write(f"**Type:** 2D Quadrant")
                     st.write(f"**Channels:** {g.get('channel_x')} vs {g.get('channel_y')}")
                     q_val = g.get('quadrant')
                     if isinstance(q_val, list): q_val = "/".join(q_val)
                     st.write(f"**Quadrant:** {q_val} (Thresholds: {g.get('threshold_x'):.2f}, {g.get('threshold_y'):.2f})")
                     st.info("Visualization not available for 2D gates in this summary view.")

                elif gate_type == 'polygon':
                     st.write(f"**Type:** 2D Polygon")
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

                        data_map_v = get_channel_data_for_samples(experiment, ch, disp_samples, parent_gate_name=p_gate, defined_gates=st.session_state.positive_gates)

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
                                    if len(v) < 10: continue # Skip empty

                                    try:
                                        if use_log_v:
                                            v_p = v[v > 0]
                                            if len(v_p) < 10: continue
                                            kde = gaussian_kde(np.log10(v_p))
                                            y_dom = kde(np.log10(d_dom))
                                        else:
                                            kde = gaussian_kde(v)
                                            y_dom = kde(d_dom)

                                        c = colors_v[idx % len(colors_v)]
                                        fig_g.add_trace(go.Scatter(x=d_dom, y=y_dom, mode='lines', line=dict(color=c), name=s, fill='tozeroy', opacity=0.3))
                                    except:
                                        pass

                            # Add Threshold/Range Line
                            if g.get('type') == 'range':
                                 t_min, t_max = g.get('min'), g.get('max')
                                 fig_g.add_vrect(x0=t_min, x1=t_max, fillcolor="green", opacity=0.1, layer="below", line_width=0)
                                 fig_g.add_vline(x=t_min, line_dash="dash", line_color="green")
                                 fig_g.add_vline(x=t_max, line_dash="dash", line_color="green")
                            else:
                                 thr = g['threshold']
                                 is_pos = (g.get('direction', 'positive') == 'positive')
                                 color_line = "green" if is_pos else "red"
                                 fig_g.add_vline(x=thr, line_dash="dash", line_color=color_line, line_width=2)

                                 # Shading
                                 # Approximate simplifying shading
                                 if is_pos:
                                     fig_g.add_vrect(x0=thr, x1=max(d_dom), fillcolor="green", opacity=0.1, layer="below", line_width=0)
                                 else:
                                     fig_g.add_vrect(x0=min(d_dom), x1=thr, fillcolor="red", opacity=0.1, layer="below", line_width=0)

                            fig_g.update_layout(
                                margin=dict(l=0, r=0, t=20, b=0),
                                height=200,
                                xaxis_type="log" if use_log_v else "linear",
                                showlegend=True, # Show legend to identify samples
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
                                # plot_bgcolor='white', # Removed to support Dark Mode / Theme
                                xaxis=dict(showgrid=True, gridcolor='lightgrey'),
                                yaxis=dict(showgrid=False, showticklabels=False)
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

        # --- PARENT SELECTION ---
        # Options: Singlets + any defined gates
        parent_options = ['Singlets'] + [g['label'] for g in st.session_state.positive_gates]
        parent_gate = st.selectbox("Select Parent Population", parent_options, index=0)

        # --- MODE SELECTION ---
        st.write("### Analysis Mode")
        mode_c1, mode_c2 = st.columns([1, 3])
        with mode_c1:
            analysis_mode = st.radio("Mode", ["1D Histogram", "2D Quadrant", "2D Polygon"], horizontal=True, label_visibility="collapsed")

        is_quadrant = (analysis_mode == "2D Quadrant")

        # --- CHANNEL SELECTION ---
        # Identify all numeric channels
        all_numeric = [c for c in experiment.data.columns if pd.api.types.is_numeric_dtype(experiment.data[c])]

        # Filter out system flags
        ignore_cols = ['cells', 'singlets', 'Is_Live', 'Is_Singlet', 'LiveGate', 'SingletGate', 'Density']

        # Get selected channels from Gating tab
        user_selected = st.session_state.get('persistent_selected_channels')

        # Helper
        COMMON_FLUORS = [
            "GFP", "EGFP", "mCherry", "Cherry", "APC", "FITC", "PE", "DAPI", "PerCP",
            "BV", "Brilliant", "Cy", "Cy3", "Cy5", "Cy7", "BFP", "EBFP", "TagBFP", "SB",
            "RFP", "dsRed", "tdTomato", "YFP", "EYFP", "mNeon", "AmCyan", "Pacific Blue",
            "Pacific Orange", "Alexa", "eFluor", "Hoechst", "PI", "7-AAD", "Zombie", "Live/Dead", "DraQ"
        ]
        def is_fluorophore(ch_name):
            if any(x in ch_name.upper() for x in ["FSC", "SSC", "TIME"]): return False
            return any(f.upper() in ch_name.upper() for f in COMMON_FLUORS)

        # Define valid channels
        if user_selected is not None:
            step1 = [c for c in all_numeric if c not in ignore_cols and c in user_selected]
            valid_channels = sorted([c for c in step1 if is_fluorophore(c)])
            if not valid_channels:
                 valid_channels = sorted([c for c in step1 if not any(x in c.upper() for x in ["FSC", "SSC", "TIME"])])
        else:
            valid_channels = sorted([c for c in all_numeric if c not in ignore_cols and is_fluorophore(c)])

        if not valid_channels:
             valid_channels = sorted([c for c in all_numeric if c not in ignore_cols and not any(x in c.upper() for x in ["FSC", "SSC"])])
        if not valid_channels:
            valid_channels = sorted([c for c in all_numeric if c not in ignore_cols])


        if analysis_mode == "1D Histogram":
            # =========================================================
            # 1D HISTOGRAM / DENSITY LOGIC (Legacy)
            # =========================================================
            c1, c2 = st.columns([1, 2])
            with c1:
                selected_channel = st.selectbox("Select X-Axis Channel", valid_channels, key="analysis_channel_select")

            if selected_channel:
                # Sample Selection
                sample_list = full_df['Sample'].unique().tolist()
                default_samples = [sample_list[0]] if sample_list else []

                with c2:
                    selected_samples = st.multiselect("Select Samples to Overlay", sample_list, default=default_samples, key="analysis_sample_select")

                if selected_samples:
                    data_map = get_channel_data_for_samples(experiment, selected_channel, selected_samples, parent_gate_name=parent_gate, defined_gates=st.session_state.positive_gates)
                    # ... [Paste existing 1D Logic compressed or re-implemented] ...
                    # To keep file size manageable and safe, I will re-implement the core parts of 1D logic here
                    # effectively restoring it but indented under 'if not is_quadrant'

                    all_values = []
                    for s in data_map:
                        if len(data_map[s]) > 0: all_values.extend(data_map[s])
                    all_values = np.array([x for x in all_values if not np.isnan(x)])

                    if len(all_values) == 0:
                        st.warning("No valid data found.")
                    else:
                        use_log = st.checkbox("Log Scale Plot", value=True)
                        if use_log and np.max(all_values) <= 0: use_log = False

                        # Range Logic
                        if use_log:
                            v_p = all_values[all_values > 0]
                            if len(v_p) > 0:
                                plot_min, plot_max = float(np.percentile(v_p, 0.1)), float(np.percentile(v_p, 99.9))
                            else: plot_min, plot_max = 0, 1
                        else:
                            plot_min, plot_max = 0.0, float(np.percentile(all_values, 99.9))
                            if plot_max <= 0: plot_max = 1.0
                        if plot_min >= plot_max: plot_max = plot_min + 1.0

                        st.write("### Threshold Selection")
                        col_type, col_dir = st.columns([2, 1])
                        with col_type:
                            gate_type = st.radio("Gate Type", ["Binary (Threshold)", "Range (Min - Max)"], horizontal=True)
                        is_range = (gate_type == "Range (Min - Max)")
                        is_negative = False
                        if not is_range:
                            with col_dir:
                                gate_direction = st.radio("Population", ["Positive (>)", "Negative (<)"], horizontal=True)
                                is_negative = (gate_direction == "Negative (<)")

                        thr_key, min_key, max_key = f"t_{selected_channel}_{parent_gate}", f"min_{selected_channel}_{parent_gate}", f"max_{selected_channel}_{parent_gate}"
                        if thr_key not in st.session_state: st.session_state[thr_key] = (plot_min + plot_max)/2
                        if min_key not in st.session_state: st.session_state[min_key] = plot_min + (plot_max-plot_min)*0.2
                        if max_key not in st.session_state: st.session_state[max_key] = plot_max - (plot_max-plot_min)*0.2

                        # Plot Interaction (Restored Logic)
                        plot_key = f"plot_1d_{selected_channel}_{parent_gate}"
                        selection_data = st.session_state.get(plot_key)

                        if selection_data and selection_data.get("selection"):
                            sel = selection_data["selection"]
                            # 1. Box Selection (Range)
                            if sel.get("box"):
                                 box = sel["box"][0]
                                 if "x" in box:
                                     x_range = box["x"]
                                     if x_range and len(x_range) >= 2:
                                          sel_min, sel_max = sorted(x_range[:2])
                                          if is_range:
                                               if abs(float(sel_min) - st.session_state[min_key]) > 0.0001 or                                                   abs(float(sel_max) - st.session_state[max_key]) > 0.0001:
                                                    st.session_state[min_key] = float(sel_min)
                                                    st.session_state[max_key] = float(sel_max)
                                                    st.rerun()
                                          else:
                                               if abs(float(sel_min) - st.session_state[thr_key]) > 0.0001:
                                                    st.session_state[thr_key] = float(sel_min)
                                                    st.rerun()

                            # 2. Point Click (Binary)
                            if sel.get("points"):
                                 point = sel["points"][0]
                                 sel_x = point.get("x")
                                 if sel_x is not None:
                                      if is_range:
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

                        # Widgets
                        if is_range:
                             m1, m2 = st.columns(2)
                             with m1: curr_min = st.number_input("Min", key=min_key)
                             with m2: curr_max = st.number_input("Max", key=max_key)
                             curr_threshold = (curr_min, curr_max)
                        else:
                             curr_val = st.number_input("Threshold", key=thr_key)
                             curr_threshold = curr_val

                        # Plot
                        x_dom = np.logspace(np.log10(plot_min if plot_min>0 else 1e-4), np.log10(plot_max), 1000) if use_log else np.linspace(plot_min, plot_max, 1000)
                        fig = go.Figure()
                        colors = px.colors.qualitative.Plotly

                        # Add click catcher
                        fig.add_trace(go.Scatter(x=x_dom, y=[0]*len(x_dom), mode='markers', marker=dict(size=20, opacity=0), hoverinfo='none', showlegend=False, name="click_catcher"))
                        
                        for i, s in enumerate(selected_samples):
                            if s in data_map:
                                v = data_map[s]; v = v[~np.isnan(v)]
                                if len(v) < 5: continue
                                try:
                                    if use_log:
                                         v = v[v>0]
                                         if len(v)<5: continue
                                         kde = gaussian_kde(np.log10(v))
                                         y = kde(np.log10(x_dom))
                                    else:
                                         kde = gaussian_kde(v)
                                         y = kde(x_dom)
                                    c = colors[i%len(colors)]
                                    # Use lines+markers but with empty markers to allow clicking line
                                    fig.add_trace(go.Scatter(x=x_dom, y=y, mode='lines', line=dict(color=c), fill='tozeroy', opacity=0.5, name=s))
                                except: pass

                        # Lines
                        if is_range:
                             fig.add_vline(x=curr_min, line_dash="dash", line_color="green")
                             fig.add_vline(x=curr_max, line_dash="dash", line_color="green")
                             fig.add_vrect(x0=curr_min, x1=curr_max, fillcolor="green", opacity=0.1)
                        else:
                             col = "red" if is_negative else "green"
                             fig.add_vline(x=curr_val, line_dash="dash", line_color=col)
                             if is_negative: fig.add_vrect(x0=plot_min, x1=curr_val, fillcolor=col, opacity=0.1)
                             else: fig.add_vrect(x0=curr_val, x1=plot_max, fillcolor=col, opacity=0.1)
                        
                        fig.update_layout(xaxis_type="log" if use_log else "linear", 
                                        xaxis_range=[np.log10(plot_min), np.log10(plot_max*1.2)] if use_log else [0, plot_max*1.1],
                                        title=f"1D: {selected_channel}", height=300, clickmode='event+select', dragmode='select')
                        fig.update_xaxes(fixedrange=True)
                        fig.update_yaxes(fixedrange=True)

                        st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode=["points", "box"], key=plot_key)
                        # Add to Canvas Button (Restored)
                        if st.button("Add This Density Plot to Canvas", key=f"add_analysis_density_{selected_channel}_{parent_gate}"):
                            import matplotlib.pyplot as plt
                            mpl_fig, mpl_ax = plt.subplots(figsize=(6, 4))
                            x_domain_mpl = x_dom
                            for i, s in enumerate(selected_samples):
                                if s in data_map:
                                    v = data_map[s]; v = v[~np.isnan(v)]
                                    if len(v) < 5: continue
                                    try:
                                        if use_log:
                                            v = v[v>0]
                                            if len(v)<5: continue
                                            kde = gaussian_kde(np.log10(v))
                                            y_vals = kde(np.log10(x_domain_mpl))
                                        else:
                                            kde = gaussian_kde(v)
                                            y_vals = kde(x_domain_mpl)
                                        c = colors[i%len(colors)]
                                        mpl_ax.plot(x_domain_mpl, y_vals, color=c, label=s, linewidth=2, zorder=10)
                                        mpl_ax.fill_between(x_domain_mpl, y_vals, y2=0, color=c, alpha=0.3, zorder=5)
                                    except: pass
                            if use_log: mpl_ax.set_xscale("log")
                            mpl_ax.set_title(f"Density: {selected_channel} ({parent_gate})")
                            mpl_ax.set_xlabel(selected_channel); mpl_ax.set_ylabel("Density")
                            mpl_ax.spines["top"].set_visible(False); mpl_ax.spines["right"].set_visible(False)
                            mpl_ax.set_ylim(bottom=0)
                            if is_range:
                                 mpl_ax.axvline(x=curr_min, color="green", linestyle="--", linewidth=2)
                                 mpl_ax.axvline(x=curr_max, color="green", linestyle="--", linewidth=2)
                                 mpl_ax.axvspan(curr_min, curr_max, color="green", alpha=0.1)
                            else:
                                 col_c = "red" if is_negative else "green"
                                 mpl_ax.axvline(x=curr_val, color=col_c, linestyle="--", linewidth=2)
                                 if is_negative: mpl_ax.axvspan(x_domain_mpl[0], curr_val, color=col_c, alpha=0.1)
                                 else: mpl_ax.axvspan(curr_val, x_domain_mpl[-1], color=col_c, alpha=0.1)
                            mpl_ax.legend()
                            mpl_fig.tight_layout()
                            add_plot_to_canvas(mpl_fig, f"Density: {selected_channel} ({parent_gate})")
                            st.success("Added directly to Canvas!")
                            plt.close(mpl_fig)

                        # Save Buttons
                        sv1, sv2 = st.columns(2)
                        with sv1:
                            if st.button("Save Gate"):
                                suffix = "+"
                                if is_range: suffix = "_rng"
                                elif is_negative: suffix = "-"
                                lbl = f"{parent_gate} -> {selected_channel}{suffix}" if parent_gate != 'Singlets' else f"{selected_channel}{suffix}"

                                new_g = {
                                    'channel': selected_channel, 'parent': parent_gate,
                                    'label': lbl,
                                    'type': 'range' if is_range else 'binary',
                                    'direction': 'negative' if is_negative else 'positive',
                                    'samples': selected_samples
                                }
                                if is_range:
                                     new_g['min'] = curr_min; new_g['max'] = curr_max; new_g['threshold'] = (curr_min, curr_max)
                                else: new_g['threshold'] = curr_val

                                st.session_state.positive_gates.append(new_g)
                                st.session_state.show_gate_creator = False
                                st.rerun()
                        with sv2:
                             if st.button("Cancel"): st.session_state.show_gate_creator = False; st.rerun()

        elif analysis_mode == "2D Quadrant":
            # =========================================================
            # 2D QUADRANT LOGIC
            # =========================================================
            c_x, c_y, c_s = st.columns([1, 1, 2])
            with c_x:
                sel_x = st.selectbox("X Channel", valid_channels, key="quad_x")
            with c_y:
                idx_y = 1 if len(valid_channels) > 1 else 0
                sel_y = st.selectbox("Y Channel", valid_channels, index=idx_y, key="quad_y")

            sample_list = full_df['Sample'].unique().tolist()
            with c_s:
                def_samp = [sample_list[0]] if sample_list else []
                sel_samps = st.multiselect("Samples", sample_list, default=def_samp, key="quad_samp")

            if sel_x and sel_y and sel_samps:
                # Fetch Data
                d_x = get_channel_data_for_samples(experiment, sel_x, sel_samps, parent_gate, st.session_state.positive_gates)
                d_y = get_channel_data_for_samples(experiment, sel_y, sel_samps, parent_gate, st.session_state.positive_gates)

                # Combine
                arr_x, arr_y = [], []
                for s in sel_samps:
                    if s in d_x and s in d_y:
                        vx, vy = d_x[s], d_y[s]
                        # Trim to min length if mismatch (unlikely if same dataframe)
                        l = min(len(vx), len(vy))
                        arr_x.extend(vx[:l])
                        arr_y.extend(vy[:l])

                arr_x = np.array(arr_x); arr_y = np.array(arr_y)
                valid_mask = (~np.isnan(arr_x)) & (~np.isnan(arr_y))
                arr_x = arr_x[valid_mask]; arr_y = arr_y[valid_mask]

                if len(arr_x) == 0:
                    st.warning("No data.")
                else:
                    use_log = st.checkbox("Log Scale", value=True, key="quad_log")
                    if use_log:
                        pos_mask = (arr_x > 0) & (arr_y > 0)
                        arr_x_p = arr_x[pos_mask]; arr_y_p = arr_y[pos_mask]
                        if len(arr_x_p) == 0: use_log = False
                        else:
                             arr_x, arr_y = arr_x_p, arr_y_p

                    # Init Thresholds
                    qk_x = f"qx_{sel_x}_{sel_y}_{parent_gate}"
                    qk_y = f"qy_{sel_x}_{sel_y}_{parent_gate}"

                    min_x, max_x = np.min(arr_x), np.max(arr_x)
                    min_y, max_y = np.min(arr_y), np.max(arr_y)

                    if qk_x not in st.session_state: st.session_state[qk_x] = (min_x + max_x)/2
                    if qk_y not in st.session_state: st.session_state[qk_y] = (min_y + max_y)/2

                    # Interaction
                    q_plot_key = f"plot_quad_{sel_x}_{sel_y}"
                    if q_plot_key in st.session_state and st.session_state[q_plot_key].get("selection"):
                         sel = st.session_state[q_plot_key]["selection"]
                         if sel.get("points"):
                              pt = sel["points"][0]
                              st.session_state[qk_x] = pt["x"]
                              st.session_state[qk_y] = pt["y"]

                    # Inputs
                    col_tx, col_ty = st.columns(2)
                    with col_tx: curr_tx = st.number_input(f"{sel_x} Threshold", key=qk_x, format="%.4f")
                    with col_ty: curr_ty = st.number_input(f"{sel_y} Threshold", key=qk_y, format="%.4f")

                    # Scatter Plot
                    # Downsample for speed
                    if len(arr_x) > 5000:
                         # Use fixed seed for consistent visualization
                         rng = np.random.RandomState(42)
                         indices = rng.choice(len(arr_x), 5000, replace=False)
                         plot_x, plot_y = arr_x[indices], arr_y[indices]
                    else:
                         plot_x, plot_y = arr_x, arr_y

                    fig = go.Figure()

                    # Define Quadrant Shapes (Colored Backgrounds)
                    # We use data bounds extended slightly to cover the view.
                    # Q1: UL (X < Eth, Y > Eth) -> Red/Orange
                    # Q2: UR (X > Eth, Y > Eth) -> Green
                    # Q3: LR (X > Eth, Y < Eth) -> Blue/Purple
                    # Q4: LL (X < Eth, Y < Eth) -> Gray/Blue

                    # Calculate bounds for shapes
                    x_span = max_x - min_x; y_span = max_y - min_y
                    # Heuristic padding
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
                        marker=dict(size=3, color='blue', opacity=0.5),
                        name='Events'
                    ))

                    # Add Lines (Crosshair) - Dashed black for better visibility on colors
                    fig.add_vline(x=curr_tx, line_color="black", line_width=1, line_dash="dash")
                    fig.add_hline(y=curr_ty, line_color="black", line_width=1, line_dash="dash")

                    # Annotations for Quadrants
                    # Q1: UL (X-, Y+), Q2: UR (X+, Y+), Q3: LR (X+, Y-), Q4: LL (X-, Y-)
                    # Using annotations for clarity

                    fig.update_layout(
                        title=f"Quadrant: {sel_x} vs {sel_y}",
                        xaxis_title=sel_x, yaxis_title=sel_y,
                        xaxis_type="log" if use_log else "linear",
                        yaxis_type="log" if use_log else "linear",
                        dragmode="pan", clickmode="event+select"
                    )

                    st.plotly_chart(fig, width="stretch", on_select="rerun", key=q_plot_key)
                    # Add to Canvas Button (New)
                    if st.button("Add This Scatter Plot to Canvas", key=f"add_analysis_quad_{sel_x}_{sel_y}_{parent_gate}"):
                        import matplotlib.pyplot as plt
                        mpl_fig, mpl_ax = plt.subplots(figsize=(6, 6))
                        colors_c = px.colors.qualitative.Plotly
                        for i, s in enumerate(sel_samps):
                             if s in d_x and s in d_y:
                                  vx, vy = d_x[s], d_y[s]
                                  l_min = min(len(vx), len(vy))
                                  vx = vx[:l_min]; vy = vy[:l_min]
                                  mask = (~np.isnan(vx)) & (~np.isnan(vy))
                                  if use_log: mask = mask & (vx > 0) & (vy > 0)
                                  vx = vx[mask]; vy = vy[mask]
                                  if len(vx) > 0:
                                       if len(vx) > 2000:
                                            rng = np.random.RandomState(42)
                                            idx = rng.choice(len(vx), 2000, replace=False)
                                            vx = vx[idx]; vy = vy[idx]
                                       c = colors_c[i%len(colors_c)]
                                       mpl_ax.scatter(vx, vy, s=10, color=c, alpha=0.6, label=s, edgecolors="none")
                        if use_log: mpl_ax.set_xscale("log"); mpl_ax.set_yscale("log")
                        mpl_ax.set_title(f"{sel_x} vs {sel_y}")
                        mpl_ax.set_xlabel(sel_x); mpl_ax.set_ylabel(sel_y)
                        mpl_ax.axvline(x=curr_tx, color="black", linestyle="--")
                        mpl_ax.axhline(y=curr_ty, color="black", linestyle="--")
                        mpl_ax.legend()
                        mpl_fig.tight_layout()
                        add_plot_to_canvas(mpl_fig, f"Scatter: {sel_x} vs {sel_y}")
                        st.success("Added to Canvas!")
                        plt.close(mpl_fig)

                    st.write("#### Save Quadrants")
                    st.write("Select which populations to create:")

                    sq1, sq2, sq3, sq4 = st.columns(4)
                    save_UL = sq1.checkbox(f"Q1: {sel_x}- {sel_y}+", value=True)
                    save_UR = sq2.checkbox(f"Q2: {sel_x}+ {sel_y}+", value=True)
                    save_LR = sq3.checkbox(f"Q3: {sel_x}+ {sel_y}-", value=True)
                    save_LL = sq4.checkbox(f"Q4: {sel_x}- {sel_y}-", value=True) # Usually Q4 is double neg

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

                            new_g = {
                                'type': 'quadrant',
                                'parent': parent_gate,
                                'label': lbl,
                                'channel_x': sel_x, 'channel_y': sel_y,
                                'threshold_x': curr_tx, 'threshold_y': curr_ty,
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
            c_x, c_y, c_s = st.columns([1, 1, 2])
            with c_x:
                sel_x = st.selectbox("X Channel", valid_channels, key="poly_x")
            with c_y:
                idx_y = 1 if len(valid_channels) > 1 else 0
                sel_y = st.selectbox("Y Channel", valid_channels, index=idx_y, key="poly_y")

            sample_list = full_df['Sample'].unique().tolist()
            with c_s:
                def_samp = [sample_list[0]] if sample_list else []
                sel_samps = st.multiselect("Samples", sample_list, default=def_samp, key="poly_samp")

            if sel_x and sel_y and sel_samps:
                # Fetch Data
                d_x = get_channel_data_for_samples(experiment, sel_x, sel_samps, parent_gate, st.session_state.positive_gates)
                d_y = get_channel_data_for_samples(experiment, sel_y, sel_samps, parent_gate, st.session_state.positive_gates)

                # Combine
                arr_x, arr_y = [], []
                for s in sel_samps:
                    if s in d_x and s in d_y:
                        vx, vy = d_x[s], d_y[s]
                        l = min(len(vx), len(vy))
                        arr_x.extend(vx[:l])
                        arr_y.extend(vy[:l])

                arr_x = np.array(arr_x); arr_y = np.array(arr_y)
                valid_mask = (~np.isnan(arr_x)) & (~np.isnan(arr_y))
                arr_x = arr_x[valid_mask]; arr_y = arr_y[valid_mask]

                if len(arr_x) == 0:
                    st.warning("No data.")
                else:
                    use_log = st.checkbox("Log Scale", value=True, key="poly_log")
                    if use_log:
                        pos_mask = (arr_x > 0) & (arr_y > 0)
                        arr_x_p = arr_x[pos_mask]; arr_y_p = arr_y[pos_mask]
                        if len(arr_x_p) == 0: use_log = False
                        else:
                             arr_x, arr_y = arr_x_p, arr_y_p

                    # Manage Vertices
                    poly_key = f"poly_verts_{sel_x}_{sel_y}_{parent_gate}"
                    if poly_key not in st.session_state: st.session_state[poly_key] = []

                    # Graph Interaction
                    plot_key = f"plot_poly_{sel_x}_{sel_y}"

                    # Retrieve Selection
                    if plot_key in st.session_state and st.session_state[plot_key].get("selection"):
                         sel = st.session_state[plot_key]["selection"]
                         if sel.get("points"):
                              pts = sel["points"]
                              # Add new unique points
                              current_verts = st.session_state[poly_key]
                              new_pts_list = []

                              # If user selected multiple points (lasso/box), append all
                              # If user single clicked, append one
                              for p in pts:
                                   px_v, py_v = p["x"], p["y"]
                                   # Check uniqueness (simple de-dupe against existing)
                                   is_new = True
                                   for existing in current_verts:
                                        if abs(existing[0]-px_v) < 1e-5 and abs(existing[1]-py_v) < 1e-5:
                                             is_new = False; break
                                   if is_new:
                                        new_pts_list.append((px_v, py_v))

                              if new_pts_list:
                                   st.session_state[poly_key].extend(new_pts_list)

                    # Manually add points via text input? No, click is preferred.

                    # Control Buttons
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
                    # Downsample logic for Scatter
                    if len(arr_x) > 5000:
                         # Use fixed seed for consistent visualization
                         rng = np.random.RandomState(42)
                         indices = rng.choice(len(arr_x), 5000, replace=False)
                         plot_x, plot_y = arr_x[indices], arr_y[indices]
                    else:
                         plot_x, plot_y = arr_x, arr_y

                    fig = go.Figure()

                    # 1. Data Trace
                    fig.add_trace(go.Scattergl(
                        x=plot_x, y=plot_y, mode='markers',
                        marker=dict(size=3, color='blue', opacity=0.5),
                        name='Events',
                        hoverinfo='none'
                    ))

                    # 2. Polygon Trace
                    verts = st.session_state[poly_key]
                    if len(verts) > 0:
                         # For plotting, close the loop if >= 3
                         if len(verts) >= 3:
                              vx = [v[0] for v in verts] + [verts[0][0]]
                              vy = [v[1] for v in verts] + [verts[0][1]]
                         else:
                              vx = [v[0] for v in verts]
                              vy = [v[1] for v in verts]

                         fig.add_trace(go.Scatter(
                              x=vx, y=vy, mode='lines+markers',
                              line=dict(color='orange', width=2),
                              marker=dict(size=6, color='orange'),
                              name='Polygon'
                         ))

                    fig.update_layout(
                        title=f"Polygon: {sel_x} vs {sel_y} (Click points to define)",
                        xaxis_title=sel_x, yaxis_title=sel_y,
                        xaxis_type="log" if use_log else "linear",
                        yaxis_type="log" if use_log else "linear",
                        dragmode="pan", # pan allows clicking points. 'lasso' or 'select' allows area.
                        clickmode="event+select"
                    )

                    st.plotly_chart(fig, width="stretch", on_select="rerun", key=plot_key, selection_mode="points")
                    st.write("#### Save Polygon Gate")
                    if len(verts) >= 3:
                         if st.button("Create Gate"):
                              lbl = f"{parent_gate} -> {sel_x}:{sel_y} Poly" if parent_gate!='Singlets' else f"{sel_x}:{sel_y} Poly"
                              new_g = {
                                  'type': 'polygon',
                                  'parent': parent_gate,
                                  'label': lbl,
                                  'channel_x': sel_x, 'channel_y': sel_y,
                                  'vertices': verts, # List of (x,y)
                                  'samples': sel_samps
                              }
                              st.session_state.positive_gates.append(new_g)
                              # Clean up
                              del st.session_state[poly_key]
                              st.session_state.show_gate_creator = False
                              st.rerun()
                    else:
                         st.warning("Select at least 3 points to form a polygon.")

                    if st.button("Cancel Polygon"):
                         st.session_state.show_gate_creator = False
                         st.rerun()




    st.divider()
    if st.button("Finish Analysis & Proceed to Plotting", type="primary"):
        st.session_state.analysis_complete = True
        st.session_state.current_tab = "Plot"
        st.rerun()
