import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import importlib
import io
import warnings
from glow.plotting import utils, categorical
from glow.canvas.functions import add_plot_to_canvas

def download_btn(fig, key):
    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="This figure includes Axes that are not compatible with tight_layout")
        fig.savefig(buf, format="svg", bbox_inches="tight")
    buf.seek(0)
    st.download_button(
         label="Download SVG",
         data=buf.getvalue(),
         file_name=f"{key}.svg",
         mime="image/svg+xml",
         key=f"dl_{key}"
    )

def render_bar_ui(filtered_experiment, channels, hue_var, conditions):
    """
    Renders UI and plot logic for Bar Plot.
    """
    # Create split view: Options (Left), Plot (Right)
    col_opts, col_plot = st.columns([1, 3])

    # --- Grouping Settings (Variables to be shared) ---
    selected_groups = None
    final_order = None

    with col_plot:
        # Move Grouping Settings Here (Detailed Options)
        st.markdown("### Grouping Configuration")

        # Force Group By to Experiment_Name/Sample if possible, as requested
        target_group_cols = ["Experiment_Name", "Experiment Name", "Sample", "sample"]
        found_group_col = None
        for t in target_group_cols:
            if t in conditions:
                found_group_col = t
                break

        if found_group_col:
            hue_var = found_group_col
            # Show as info but allow override via checkbox? For now stick to logic
            st.info(f"Grouping by **{hue_var}**")
        else:
            hue_var = st.selectbox("Group By (X-Axis)", conditions)
        
        if hue_var:
            all_groups = sorted(filtered_experiment.data[hue_var].unique())
            
            # Use columns for layout in wider area
            gc1, gc2 = st.columns([3, 1])
            
            with gc1: 
                selected_groups = st.multiselect("Select Groups to Plot", all_groups, default=all_groups)
            with gc2:
                st.write("") # Spacer to align checkbox
                st.write("")
                manual_order = st.checkbox("Manual Group Order", value=False)
            
            final_order = selected_groups

            if manual_order and selected_groups:
                    order_text = st.text_area("Specify Order (comma separated)", value=", ".join(selected_groups), height=68)
                    if order_text:
                        custom_order = [x.strip() for x in order_text.split(",") if x.strip()]
                        final_order = [x for x in custom_order if x in selected_groups]
                        remaining = [x for x in selected_groups if x not in final_order]
                        final_order.extend(remaining)

        st.divider()

    with col_opts:
        # --- Channel & Statistics (Stacked vertically, no columns to squeeze) ---
        st.markdown("#### Plot Settings")
        
        selected_channel = st.selectbox("Output Channel (Y-axis)", channels)
        
        correction_method = st.selectbox(
            "Visual Significance Method",
            ["t-test", "Bonferroni Correction", "FDR (Benjamini-Hochberg)"],
            help="Selects how p-values are adjusted for multiple comparisons."
        )

        # Map friendly name to backend key
        corr_map = {
            "t-test": "t-test",
            "Bonferroni Correction": "bonferroni",
            "FDR (Benjamini-Hochberg)": "fdr_bh"
        }
        selected_correction = corr_map[correction_method]


        # --- Gating Logic ---
        st.markdown("#### Gating (Optional)")

        # NEW: Import Analysis Gates
        analysis_gates_to_apply = []
        if ('positive_gates' in st.session_state and st.session_state.positive_gates) or \
           ('cells' in filtered_experiment.data.columns) or ('singlets' in filtered_experiment.data.columns):

            with st.expander("Import Gates from Analysis Tab"):
                st.info("You can choose to apply gates defined in the Analysis tab to this plot.")

                # 1. Fundamental Gates (if present)
                built_in_options = []
                if 'cells' in filtered_experiment.data.columns:
                    built_in_options.append("Pre-Gate: Live Cells")
                if 'singlets' in filtered_experiment.data.columns:
                    built_in_options.append("Pre-Gate: Singlets")

                # 2. Positive Gates from Analysis
                pos_gate_labels = [g['label'] for g in st.session_state.get('positive_gates', [])]

                gate_options = built_in_options + pos_gate_labels
                selected_analysis_labels = st.multiselect("Select Analysis Gates to Apply", options=gate_options, default=built_in_options)

                # Map selected labels back to gate objects
                def resolve_gate_chain(labels_to_res):
                    to_apply = []
                    seen_labels = set()

                    # Use a queue-like approach to find all parents
                    remaining = list(labels_to_res)
                    while remaining:
                        lbl = remaining.pop(0)
                        if lbl in seen_labels:
                            continue

                        found = False
                        # 1. Check Built-ins
                        if lbl == "Pre-Gate: Live Cells":
                            to_apply.append({'channel': 'cells', 'threshold': 0.5, 'label': lbl})
                            seen_labels.add(lbl)
                            found = True
                        elif lbl == "Pre-Gate: Singlets":
                            to_apply.append({'channel': 'singlets', 'threshold': 0.5, 'label': lbl})
                            seen_labels.add(lbl)
                            # Singlets implies Live Cells
                            if "Pre-Gate: Live Cells" not in seen_labels:
                                remaining.append("Pre-Gate: Live Cells")
                            found = True

                        # 2. Check Positive Gates
                        if not found:
                            for g in st.session_state.get('positive_gates', []):
                                if g['label'] == lbl:
                                    to_apply.append(g)
                                    seen_labels.add(lbl)
                                    # Check for Parent
                                    p = g.get('parent')
                                    if p and p != "Singlets":
                                        if p not in seen_labels:
                                            remaining.append(p)
                                    elif p == "Singlets":
                                        if "Pre-Gate: Singlets" not in seen_labels:
                                            remaining.append("Pre-Gate: Singlets")
                                        found = True
                                        break # Found this label's object
                    return to_apply

                analysis_gates_to_apply = resolve_gate_chain(selected_analysis_labels)

        if 'bar_gates' not in st.session_state:
            st.session_state.bar_gates = []

        with st.expander("Active Local Gates", expanded=True):
            if not st.session_state.bar_gates:
                st.info("No local gates defined for this plot.")
            else:
                for i, gate in enumerate(st.session_state.bar_gates):
                    gc1, gc2 = st.columns([4, 1])
                    with gc1:
                        th = gate['threshold']
                        st.write(f"**Local Gate {i+1}**: {gate['channel']} ({th[0]:.2f} - {th[1]:.2f})")
                    with gc2:
                        if st.button("🗑️", key=f"del_gate_{i}"):
                            st.session_state.bar_gates.pop(i)
                            st.rerun()

        with st.expander("Add New Local Gate"):
            importlib.reload(utils)

            # Suggest a different channel for marker if possible
            default_marker_idx = 0
            if len(channels) > 1 and channels[0] == selected_channel:
                default_marker_idx = 1
            new_gate_channel = st.selectbox("Marker Channel for New Local Gate", channels, index=default_marker_idx)

            # --- Two-Dot Range Selection Linked with Raw Score ---

            # Initialize Temporary State for the "New" gate being configured
            if 'new_gate_range_p' not in st.session_state:
                st.session_state['new_gate_range_p'] = (95.0, 99.9)

            # Define Callbacks
            def on_new_slider_change():
                p_min, p_max = st.session_state['new_gate_range_p']
                try:
                    v_min = utils.get_percentile_threshold(filtered_experiment.data, new_gate_channel, p_min)
                    v_max = utils.get_percentile_threshold(filtered_experiment.data, new_gate_channel, p_max)
                    st.session_state['new_gate_raw_min'] = float(v_min)
                    st.session_state['new_gate_raw_max'] = float(v_max)
                except Exception:
                    pass

            def on_new_raw_change():
                v_min = st.session_state['new_gate_raw_min']
                v_max = st.session_state['new_gate_raw_max']
                try:
                    p_min = utils.get_percentile_of_value(filtered_experiment.data, new_gate_channel, v_min)
                    p_max = utils.get_percentile_of_value(filtered_experiment.data, new_gate_channel, v_max)
                    p_min = max(0.0, min(100.0, p_min))
                    p_max = max(0.0, min(100.0, p_max))
                    if p_min > p_max:
                        p_min, p_max = p_max, p_min
                    st.session_state['new_gate_range_p'] = (p_min, p_max)
                except Exception:
                    pass

            # Handle Channel Change
            if 'last_new_gate_channel' not in st.session_state:
                st.session_state.last_new_gate_channel = new_gate_channel

            if st.session_state.last_new_gate_channel != new_gate_channel:
                 st.session_state.last_new_gate_channel = new_gate_channel
                 on_new_slider_change()

            if 'new_gate_raw_min' not in st.session_state:
                on_new_slider_change()

            st.markdown("**Percentile Range**")
            st.slider("Percentile", 0.0, 100.0, key='new_gate_range_p', on_change=on_new_slider_change, step=0.1)

            st.markdown(f"**Value Range** ({new_gate_channel})")
            rc1, rc2 = st.columns(2)
            with rc1:
                st.number_input("Min", key='new_gate_raw_min', on_change=on_new_raw_change)
            with rc2:
                st.number_input("Max", key='new_gate_raw_max', on_change=on_new_raw_change)

            if st.button("Add This Gate", type="secondary"):
                st.session_state.bar_gates.append({
                    'channel': new_gate_channel,
                    'threshold': (st.session_state['new_gate_raw_min'], st.session_state['new_gate_raw_max'])
                })
                st.success(f"Added gate on {new_gate_channel}")
                st.rerun()


        # --- Comparisons ---
        comparisons = []
        if hue_var and selected_groups:
            st.markdown("#### Comparisons")
            with st.expander("Add Comparisons"):
                if 'bar_comparisons' not in st.session_state:
                    st.session_state.bar_comparisons = []

                # Stacked UI for narrow column
                vis_groups = sorted(selected_groups)
                if len(vis_groups) >= 2:
                    g1 = st.selectbox("Group 1", vis_groups, key="g1")
                    g2 = st.selectbox("Group 2", vis_groups, key="g2")

                    if st.button("Add Pair"):
                        # Avoid duplicates
                        pair = (g1, g2)
                        if pair not in st.session_state.bar_comparisons and g1 != g2:
                            st.session_state.bar_comparisons.append(pair)

                if st.session_state.bar_comparisons:
                    # Sanitize old session state data (remove labels if present)
                    valid_comparisons = []
                    for comp in st.session_state.bar_comparisons:
                        if len(comp) >= 2:
                            valid_comparisons.append(comp[:2])
                    st.session_state.bar_comparisons = valid_comparisons

                    st.markdown("**Selected Comparisons:**")
                    # Show as a table
                    comp_df = pd.DataFrame(st.session_state.bar_comparisons, columns=["Group 1", "Group 2"])
                    st.dataframe(comp_df, hide_index=True, width="stretch")

                    if st.button("Clear Comparisons"):
                        st.session_state.bar_comparisons = []

                comparisons = st.session_state.bar_comparisons

        # --- Style Settings ---
        st.markdown("#### Style Settings")
        
        # Stacked options to avoid squeezing
        show_dist = st.checkbox("Show Distributions", value=True)
        use_bars = st.checkbox("Plot as Bars", value=False, help="Draw bars indicating the mean instead of just lines.")
        log_scale = st.checkbox("Log Scale Y-Axis", value=True)

        font_size = st.number_input("Font Size", min_value=6, max_value=30, value=12)
        step_fraction = st.slider("Bracket Spacing", 0.05, 0.30, 0.15, 0.01, help="Vertical distance between significance brackets.")
        
        error_bar_type = st.selectbox("Error Bar", ["SEM", "SD", "None"], index=0)
        error_bar_val = error_bar_type.lower() if error_bar_type != "None" else None

        # Auto-detect Replicate Column
        replicate_col = None
        all_columns = filtered_experiment.data.columns.tolist()
        # Try to find 'Replicate'
        for c in all_columns:
            if c.lower() == 'replicate':
                replicate_col = c
                break

        # X-Axis Annotations
        # Filter conditions to exclude 'Experiment_Name' or 'Replicate'
        target_names = ["Experiment_Name", "Experiment Name", "Experiment_name", "experiment_name", "Sample", "sample"]
        valid_conditions = [c for c in conditions if c not in target_names and c != replicate_col]
        annotation_cols = st.multiselect("Additional Conditions for X-Axis Labels", valid_conditions)

        # Show event count for feedback
        event_count = len(filtered_experiment.data)
        st.info(f"Using {event_count:,} events for plotting.")


        if st.button("Generate Plot", type="primary", key="btn_bar"):
            st.session_state.bar_generated = True

    with col_plot:
        if st.session_state.get('bar_generated'):
            # reload to get latest backend changes
            importlib.reload(utils)
            importlib.reload(categorical)

            with st.spinner("Generating plots..."):
                try:
                    # Combine local gates and selected analysis gates
                    all_applied_gates = st.session_state.bar_gates + analysis_gates_to_apply

                    # Normalize gates for backend compatibility (e.g. Range gates from Analysis)
                    normalized_gates = []
                    for g in all_applied_gates:
                        ng = g.copy()
                        # Fix Range Gate (Analysis uses min/max, Backend uses threshold tuple)
                        if ng.get('type') == 'range' and 'min' in ng and 'max' in ng:
                             ng['threshold'] = (ng['min'], ng['max'])
                        normalized_gates.append(ng)

                    if normalized_gates:
                        st.write(f"Applying {len(normalized_gates)} filters...")

                    fig = categorical.plot_means_with_significance(
                        filtered_experiment,
                        channel=selected_channel,
                        statistic='mean', # Always mean
                        group_by=hue_var,
                        comparisons=comparisons,
                        hue_order=final_order,
                        show_distribution=show_dist,
                        replicate_col=replicate_col,
                        log_scale=log_scale,
                        annotation_cols=annotation_cols,
                        correction_method=selected_correction,
                        use_bars=use_bars,
                        font_size=font_size,
                        bracket_spacing=step_fraction,
                        gates=normalized_gates,
                        error_bar=error_bar_val if error_bar_val else "ci" # Default if None
                    )
                    st.pyplot(fig)
                    download_btn(fig, "bar_plot")

                    canvas_title = f"Bar Plot: {selected_channel} by {hue_var}" if hue_var else f"Bar Plot: {selected_channel}"
                    if st.button("Add Bar Plot to Canvas", key="add_bar_to_canvas"):
                        if add_plot_to_canvas(fig, canvas_title):
                            st.success("Plot added to canvas!")

                    # Close figure to free memory, unless it was stored as a direct reference in Canvas
                    if not getattr(fig, '_glow_is_direct_ref', False):
                        plt.close(fig)
                except Exception as e:
                    st.error(f"Error: {e}")
