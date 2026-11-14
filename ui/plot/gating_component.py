import streamlit as st
import pandas as pd
from glow.plotting import utils, plots

def render_gating_component(filtered_experiment, channels, key_prefix="plot"):
    """
    Renders the shared Gating UI component (Analysis Gates + Local Gates).
    Returns (local_gates, analysis_gates_resolved)
    """
    st.markdown("#### Gating (Optional)")

    # 1. Import Analysis Gates
    analysis_gates_to_apply = []
    if ('positive_gates' in st.session_state and st.session_state.positive_gates) or \
       ('cells' in filtered_experiment.data.columns) or ('singlets' in filtered_experiment.data.columns):

        with st.expander("Import Gates from Analysis Tab"):
            st.info("You can choose to apply gates defined in the Analysis tab to this plot.")

            # Fundamental Gates
            built_in_options = []
            if 'cells' in filtered_experiment.data.columns:
                built_in_options.append("Pre-Gate: Live Cells")
            if 'singlets' in filtered_experiment.data.columns:
                built_in_options.append("Pre-Gate: Singlets")

            # Positive Gates
            pos_gate_labels = [g['label'] for g in st.session_state.get('positive_gates', [])]

            gate_options = built_in_options + pos_gate_labels
            selected_analysis_labels = st.multiselect(
                "Select Analysis Gates to Apply", 
                options=gate_options, 
                default=built_in_options,
                key=f"{key_prefix}_analysis_gates_select"
            )

            # Map selected labels back to gate objects
            def resolve_gate_chain(labels_to_res):
                to_apply = []
                seen_labels = set()
                remaining = list(labels_to_res)
                while remaining:
                    lbl = remaining.pop(0)
                    if lbl in seen_labels:
                        continue
                    found = False
                    # Built-ins
                    if lbl == "Pre-Gate: Live Cells":
                        to_apply.append({'channel': 'cells', 'threshold': 0.5, 'label': lbl})
                        seen_labels.add(lbl)
                        found = True
                    elif lbl == "Pre-Gate: Singlets":
                        to_apply.append({'channel': 'singlets', 'threshold': 0.5, 'label': lbl})
                        seen_labels.add(lbl)
                        if "Pre-Gate: Live Cells" not in seen_labels:
                            remaining.append("Pre-Gate: Live Cells")
                        found = True
                    # Positive Gates
                    if not found:
                        for g in st.session_state.get('positive_gates', []):
                            if g['label'] == lbl:
                                to_apply.append(g)
                                seen_labels.add(lbl)
                                p = g.get('parent')
                                if p and p != "Singlets":
                                    if p not in seen_labels:
                                        remaining.append(p)
                                elif p == "Singlets":
                                    if "Pre-Gate: Singlets" not in seen_labels:
                                        remaining.append("Pre-Gate: Singlets")
                                    found = True
                                    break
                return to_apply

            analysis_gates_to_apply = resolve_gate_chain(selected_analysis_labels)

    # 2. Local Gates
    gate_key = f"{key_prefix}_local_gates"
    if gate_key not in st.session_state:
        st.session_state[gate_key] = []

    with st.expander("Active Local Gates", expanded=True):
        if not st.session_state[gate_key]:
            st.info("No local gates defined for this plot.")
        else:
            def del_gate_callback(index_to_del, g_key):
                if 0 <= index_to_del < len(st.session_state[g_key]):
                    st.session_state[g_key].pop(index_to_del)
            
            for i, gate in enumerate(st.session_state[gate_key]):
                gc1, gc2 = st.columns([4, 1])
                with gc1:
                    th = gate['threshold']
                    st.write(f"**Local Gate {i+1}**: {gate['channel']} ({th[0]:.2f} - {th[1]:.2f})")
                with gc2:
                    st.button(
                        "🗑️", 
                        key=f"{key_prefix}_del_gate_{i}", 
                        on_click=del_gate_callback, 
                        args=(i, gate_key)
                    )

    with st.expander("Add New Local Gate"):

        # Ensure different new gate channel per plot type if possible to avoid clashes?
        # Actually standard widget keys should be uniqueified by prefix.
        
        # Suggest default
        default_marker_idx = 0
        if len(channels) > 1:
            # Simple heuristic, pick 2nd if available
            default_marker_idx = 1
            
        new_gate_channel = st.selectbox(
            "Marker Channel for New Local Gate", 
            channels, 
            index=default_marker_idx,
            key=f"{key_prefix}_new_gate_channel_select"
        )

        range_p_key = f"{key_prefix}_new_gate_range_p"
        raw_min_key = f"{key_prefix}_new_gate_raw_min"
        raw_max_key = f"{key_prefix}_new_gate_raw_max"

        if range_p_key not in st.session_state:
            st.session_state[range_p_key] = (95.0, 99.9)

        # Callbacks
        def on_new_slider_change():
            p_min, p_max = st.session_state[range_p_key]
            try:
                v_min = utils.get_percentile_threshold(filtered_experiment.data, new_gate_channel, p_min)
                v_max = utils.get_percentile_threshold(filtered_experiment.data, new_gate_channel, p_max)
                st.session_state[raw_min_key] = float(v_min)
                st.session_state[raw_max_key] = float(v_max)
            except Exception:
                pass

        def on_new_raw_change():
            v_min = st.session_state[raw_min_key]
            v_max = st.session_state[raw_max_key]
            try:
                p_min = utils.get_percentile_of_value(filtered_experiment.data, new_gate_channel, v_min)
                p_max = utils.get_percentile_of_value(filtered_experiment.data, new_gate_channel, v_max)
                p_min = max(0.0, min(100.0, p_min))
                p_max = max(0.0, min(100.0, p_max))
                if p_min > p_max: p_min, p_max = p_max, p_min
                st.session_state[range_p_key] = (p_min, p_max)
            except Exception:
                pass

        # Trigger update if raw keys missing
        if raw_min_key not in st.session_state:
            on_new_slider_change()

        st.markdown("**Percentile Range**")
        st.slider(
            "Percentile", 0.0, 100.0, 
            key=range_p_key, 
            on_change=on_new_slider_change, 
            step=0.1
        )

        st.markdown(f"**Value Range** ({new_gate_channel})")
        rc1, rc2 = st.columns(2)
        with rc1:
            st.number_input("Min", key=raw_min_key, on_change=on_new_raw_change)
        with rc2:
            st.number_input("Max", key=raw_max_key, on_change=on_new_raw_change)

        def add_gate_callback(channel, g_key, r_min_key, r_max_key):
             st.session_state[g_key].append({
                'channel': channel,
                'threshold': (st.session_state[r_min_key], st.session_state[r_max_key]),
                'is_data_units': True  # Mark this gate as being in data units already
            })
        
        if st.button(
            "Add This Gate", 
            type="secondary", 
            key=f"{key_prefix}_add_gate_btn",
            on_click=add_gate_callback, 
            args=(new_gate_channel, gate_key, raw_min_key, raw_max_key)
        ):
            st.success(f"Added gate on {new_gate_channel}")

    return st.session_state[gate_key], analysis_gates_to_apply


def apply_gates_to_dataframe(df, gates):
    """
    Applies a list of gates to a DataFrame.
    Data and gates share one unit convention (high-res data is scaled once at load),
    so no unit correction happens here.
    """
    plot_df = df.copy()

    from glow.gating.geometry import points_in_polygon
    
    processed_gates = [g.copy() for g in gates]

    # Filtering Loop
    for gate in processed_gates:
        g_type = gate.get('type')
        
        if g_type == 'quadrant':
             ch_x = gate['channel_x']
             ch_y = gate['channel_y']
             th_x = gate['threshold_x']
             th_y = gate['threshold_y']
             quads = gate.get('quadrant', [])
             if isinstance(quads, str): quads = [quads]
             
             if ch_x in plot_df.columns and ch_y in plot_df.columns:
                 mask = pd.Series(False, index=plot_df.index)
                 sub_df = plot_df[[ch_x, ch_y]]
                 
                 for q in quads:
                     m = None
                     if q == '++':
                         m = (sub_df[ch_x] > th_x) & (sub_df[ch_y] > th_y)
                     elif q == '+-':
                         m = (sub_df[ch_x] > th_x) & (sub_df[ch_y] <= th_y)
                     elif q == '-+':
                         m = (sub_df[ch_x] <= th_x) & (sub_df[ch_y] > th_y)
                     elif q == '--':
                         m = (sub_df[ch_x] <= th_x) & (sub_df[ch_y] <= th_y)
                     
                     if m is not None:
                         mask |= m
                 plot_df = plot_df[mask]

        elif g_type == 'polygon':
             ch_x = gate['channel_x']
             ch_y = gate['channel_y']
             verts = gate['vertices']
             
             if ch_x in plot_df.columns and ch_y in plot_df.columns:
                 pts = plot_df[[ch_x, ch_y]].values
                 mask = points_in_polygon(pts, verts)
                 plot_df = plot_df[mask]

        else:
            ch = gate.get('channel')
            th = gate.get('threshold')
            
            if ch in plot_df.columns:
                if isinstance(th, (tuple, list)) and len(th) == 2:
                    plot_df = plot_df[(plot_df[ch] >= th[0]) & (plot_df[ch] <= th[1])]
                elif gate.get('min') is not None and gate.get('max') is not None:
                     plot_df = plot_df[(plot_df[ch] >= gate['min']) & (plot_df[ch] <= gate['max'])]
                elif th is not None:
                    # Directionality
                    direction = gate.get('direction', 'positive')
                    if direction in ['negative', 'Negative (<)']:
                        plot_df = plot_df[plot_df[ch] < th]
                    else:
                        plot_df = plot_df[plot_df[ch] > th]
                    
    return plot_df, processed_gates
