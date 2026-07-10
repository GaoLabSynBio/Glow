import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import copy
from glow.plotting import utils, categorical
from glow.canvas.functions import add_plot_to_canvas
from common.widgets import download_svg_button as download_btn
from common.errors import show_error
from .gating_component import render_gating_component, apply_gates_to_dataframe

def render_bar_ui(filtered_experiment, channels, hue_var, conditions):
    """
    Renders UI and plot logic for Bar Plot.
    """
    # --- Gating Logic ---
    # Sync old state if exists
    if 'bar_gates' in st.session_state and 'bar_plot_local_gates' not in st.session_state:
            st.session_state['bar_plot_local_gates'] = st.session_state.bar_gates

    local_gates, analysis_gates_to_apply = render_gating_component(filtered_experiment, channels, key_prefix="bar_plot")

    # Sync back for legacy access
    st.session_state.bar_gates = local_gates

    # Create split view: Options (Left), Plot (Right)
    col_opts, col_plot = st.columns([1, 3])

    # --- Grouping Settings (Variables to be shared) ---
    selected_groups = None
    final_order = None

    with col_plot:
        # Move Grouping Settings Here (Detailed Options)
        st.markdown("### Grouping Configuration")

        # Group By is fully user-selectable. We default to the experiment/sample
        # column when present (the most common choice), but the user can group by
        # ANY available condition (Cell Line, Treatment, Dose, timepoint, ...).
        preferred_group_cols = ["Experiment_Name", "Experiment Name", "Sample", "sample"]
        default_group_idx = 0
        for pref in preferred_group_cols:
            if pref in conditions:
                default_group_idx = conditions.index(pref)
                break

        if conditions:
            hue_var = st.selectbox(
                "Group By (X-Axis)", conditions, index=default_group_idx,
                help="Choose which metadata column defines the groups on the X-axis.",
            )
        else:
            st.warning("No grouping conditions are available for this dataset.")
            hue_var = None

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

        # Default to t-test (uncorrected)
        selected_correction = "t-test"

        # --- Comparisons ---
        comparisons = []
        if hue_var and selected_groups:
            st.markdown("#### Comparisons")
            has_replicate = any(str(c).lower() == 'replicate' for c in filtered_experiment.data.columns)
            if not has_replicate:
                st.caption(
                    "⚠️ No **Replicate** column detected, so significance can't be computed "
                    "(each group is treated as n=1). Brackets will show fold change only. "
                    "Add a Replicate column in the metadata to enable statistics."
                )
            with st.expander("Add Comparisons"):
                if 'bar_comparisons' not in st.session_state:
                    st.session_state.bar_comparisons = []

                # Stacked UI for narrow column
                vis_groups = sorted(selected_groups)
                if len(vis_groups) >= 2:
                    st.selectbox("Group 1", vis_groups, key="g1")
                    st.selectbox("Group 2", vis_groups, key="g2")

                    def add_pair_callback():
                        p1 = st.session_state.get('g1')
                        p2 = st.session_state.get('g2')
                        if p1 and p2:
                            new_pair = (p1, p2)
                            if new_pair not in st.session_state.bar_comparisons and p1 != p2:
                                st.session_state.bar_comparisons.append(new_pair)

                    if st.button("Add Pair", on_click=add_pair_callback):
                        pass

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

                    def clear_comps_callback():
                        st.session_state.bar_comparisons = []

                    if st.button("Clear Comparisons", on_click=clear_comps_callback):
                        pass

                comparisons = st.session_state.bar_comparisons

            # --- Correction Logic inside Comparisons Section ---
            st.markdown("###### Significance Correction")

            # Calculate state
            num_comps = len(st.session_state.get('bar_comparisons', []))
            is_correction_disabled = num_comps < 2

            if is_correction_disabled:
                 correction_options = ["t-test (None)"]
            else:
                 correction_options = ["Bonferroni Correction", "FDR (Benjamini-Hochberg)"]

            c_method = st.selectbox(
                "Method",
                correction_options,
                disabled=is_correction_disabled,
                help="Applies multiple testing correction to p-values. Requires at least 2 comparisons.",
                key="multi_corr_method"
            )

            if is_correction_disabled:
                 st.caption("Correction disabled (Define ≥2 comparisons).")
                 selected_correction = "t-test"
            else:
                 if "Bonferroni" in c_method:
                     selected_correction = "bonferroni"
                 elif "FDR" in c_method:
                     selected_correction = "fdr_bh"
                 else:
                     selected_correction = "t-test"

        # --- Style Settings ---
        st.markdown("#### Style Settings")

        # Stacked options to avoid squeezing
        # Keys added to persist state across reruns (e.g. when adding gates)
        show_dist = st.checkbox("Show Distributions", value=True, key="bar_style_show_dist")
        use_bars = st.checkbox("Plot as Bars", value=False, help="Draw bars indicating the mean instead of just lines.", key="bar_style_use_bars")
        log_scale = st.checkbox("Log Scale Y-Axis", value=True, key="bar_style_log_scale")

        font_size = st.number_input("Font Size", min_value=6, max_value=30, value=12, key="bar_style_font_size")
        step_fraction = st.slider("Bracket Spacing", 0.05, 0.30, 0.10, 0.01, help="Vertical distance between significance brackets.", key="bar_style_spacing")

        error_bar_type = st.selectbox("Error Bar", ["SEM", "SD", "None"], index=0, key="bar_style_error_bar")
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


    with col_plot:
        if st.button("Generate Plot", type="primary", key="btn_bar"):
            st.session_state.bar_generated = True

        if st.session_state.get('bar_generated'):
            with st.spinner("Generating plots..."):
                try:
                    # --- Gating Application ---
                    all_applied_gates = local_gates + analysis_gates_to_apply

                    # Apply Gates (Filtering + Scaling)
                    filtered_df, processed_gates = apply_gates_to_dataframe(
                        filtered_experiment.data,
                        all_applied_gates,
                        experiment_metadata=getattr(filtered_experiment, 'metadata', None),
                        show_raw=st.session_state.get('plot_show_raw_vals', False)
                    )

                    if filtered_df.empty:
                        st.warning("No events remain after applying gates. Please adjust your gates.")
                        return

                    if log_scale:
                         if not (filtered_df[selected_channel] > 0).any():
                              st.warning(f"Data has no positive values in {selected_channel}, and therefore cannot be log-scaled. Uncheck Log Scale or adjust gates.")
                              return

                    if processed_gates:
                        st.write(f"Applying {len(processed_gates)} filters...")

                    # Create temp experiment for backend
                    plot_experiment = copy.copy(filtered_experiment)
                    plot_experiment.data = filtered_df

                    fig = categorical.plot_means_with_significance(
                        plot_experiment,
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
                        gates=[], # Gates are already applied
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

                    # --- Export CSV Logic ---
                    st.divider()
                    st.markdown("#### Export Stats")

                    # Calculate applied gate strings using processed gates
                    gate_descriptions = []
                    for gate in processed_gates:
                         ch = gate.get('channel')
                         th = gate.get('threshold')
                         # Check direction? apply_gates_to_dataframe might have stripped it, or not?
                         # It returns original dict with potentially modified threshold
                         if isinstance(th, (tuple, list)) and len(th) == 2:
                             gate_descriptions.append(f"{ch} {th[0]:.1f}-{th[1]:.1f}")
                         else:
                             # Simplified description
                             gate_descriptions.append(f"{ch} > {th:.1f}" if gate.get('direction') == 'positive' else f"{ch} < {th:.1f}")

                    gates_applied_str = "; ".join(gate_descriptions) if gate_descriptions else "None"

                    # 2. Determine Grouping Columns
                    # Priority: Sample, Experiment, Replicate
                    group_candidates = ['Sample', 'Experiment_Name', 'Experiment Name', replicate_col]
                    actual_group_cols = [c for c in group_candidates if c and c in filtered_df.columns]

                    # Ensure hue_var is included if it's grouping the bars
                    if hue_var and hue_var in filtered_df.columns and hue_var not in actual_group_cols:
                        actual_group_cols.append(hue_var)

                    # Remove duplicates while preserving order
                    actual_group_cols = list(dict.fromkeys(actual_group_cols))

                    if actual_group_cols:
                        # compute mean
                        summary_df = filtered_df.groupby(actual_group_cols)[selected_channel].mean().reset_index()
                        summary_df.rename(columns={selected_channel: f"Mean_{selected_channel}"}, inplace=True)

                        # Add metadata
                        summary_df['Gates_Applied'] = gates_applied_str

                        # Reorder Columns: Group keys, then Mean, then Gates
                        final_cols = actual_group_cols + [f"Mean_{selected_channel}", "Gates_Applied"]
                        summary_df = summary_df[final_cols]

                        # Export
                        csv_data = summary_df.to_csv(index=False)
                        st.download_button(
                            label="Download Mean Values (CSV)",
                            data=csv_data,
                            file_name=f"bar_stats_{selected_channel}.csv",
                            mime="text/csv",
                            help="Export the mean values of the plotted bars."
                        )
                    else:
                        st.warning("Could not determine grouping columns for export.")

                except Exception as e:
                    show_error(
                        "Could not generate the bar plot.",
                        cause="the selected channel, grouping, or gates may leave no valid data",
                        fix="try a different Output Channel, widen your gates, or turn off Log Scale if the data has no positive values",
                        exc=e,
                    )
