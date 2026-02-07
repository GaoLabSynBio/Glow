import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import importlib
import copy
import warnings
import plotly.graph_objects as go
import numpy as np
from scipy.stats import gaussian_kde
from glow.plotting import plots
from glow.canvas.functions import add_plot_to_canvas
from .gating_component import render_gating_component, apply_gates_to_dataframe
import io

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

def render_histogram_ui(filtered_experiment, channels, hue_var, conditions, scale):
    """
    Renders UI and plot logic for 1D Histogram.
    """
    # --- Gating (Unified Component) ---
    local_gates, analysis_gates_to_apply = render_gating_component(filtered_experiment, channels, key_prefix="hist_plot")

    # Create split view: Options (Left), Plot (Right)
    col_opts, col_plot = st.columns([1, 3])

    # --- LEFT COLUMN: Options ---
    with col_opts:
        st.markdown("#### Plot Settings")

        selected_channel = st.selectbox("Select Channel", channels)
        scale = st.selectbox("Scale", ["log", "linear", "logicle"], index=0, help="Axis scale transformation.")

        # User requested: 1D plot groupby should be fixed to Experiment Name if available
        target_cols = ["Experiment Name", "Experiment_Name", "Sample", "sample"]
        found_col = None
        for t in target_cols:
            if t in conditions:
                found_col = t
                break

        if found_col:
            hue_var = found_col
            st.info(f"Grouping by: {hue_var}")
        else:
             # Default to "None" for rendering options purposes (will be overridden by "Configuration" column selection)
             hue_var = None

        # Histogram specific settings
        st.markdown("---")
        st.markdown("**Rendering Options**")

        viz_mode = st.radio(
            "Visualization Style",
            ["Ridgeline (Matplotlib)", "Ridgeline (Overlay)", "Standard (Matplotlib)"],
            index=0,
            help="Ridgeline: Overlaid plots. Standard: Stacked separate plots."
        )

        ridgeline_offset = 0.5
        if "Ridgeline" in viz_mode:
            default_offset = 0.8 if viz_mode == "Ridgeline (Matplotlib)" else 0.5
            ridgeline_offset = st.slider("Overlay Offset", 0.0, 2.0, default_offset, 0.1)

        smooth_sigma = st.slider(
            "Smoothing (Matplotlib only)",
            min_value=0.0,
            max_value=10.0,
            value=0.5,
            step=0.5
        )

        normalization_mode = "total"
        if hue_var:
            normalization_mode = st.selectbox(
                "Merged Plot Normalization",
                options=[
                    ("Original (raw counts)", "original"),
                    ("Modal normalization (FlowJo-style)", "modal"),
                    ("Total count normalization", "total"),
                ],
                index=2,
                format_func=lambda x: x[0],
                help="Original: preserve event count differences. Modal: scale each group so peaks match. Total: scale each group so total area sums to 1."
            )[1]

        show_merged = st.checkbox("Show Merged/Main Plot", value=True)
        show_individual = st.checkbox("Show Individual Groups", value=False)
        show_yaxis = st.checkbox("Show Y-Axis", value=False, help="Display frequency/count axis for subplots.")

        legend_columns = None
        legend_font_size = 10
        # Legend configuration moved to col_plot (Configuration section) but variables needed for matplotlib call
        # Initialize defaults here, will be overwritten if hue_var is active

    # --- RIGHT COLUMN: Plot Display ---
    with col_plot:

        # --- Configuration Section (Right above Generate Plot) ---
        st.markdown("#### Configuration")

        # 1. Group By Selection
        # Re-implement Group By logic here
        # If hue_var was pre-determined by defaults (found_col), use it as index
        # Otherwise use defaults
        group_opts = ["None"] + conditions
        default_idx = 0
        if hue_var and hue_var in group_opts:
             default_idx = group_opts.index(hue_var)
        elif len(conditions) > 0:
             default_idx = 1 # Default to first condition if available

        compare_by = st.selectbox(
            "Color/Group By",
            group_opts,
            index=default_idx,
            help="Select attribute for coloring/grouping histograms"
        )
        hue_var = None if compare_by == "None" else compare_by

        # 2. Legend / Stats Configuration
        if hue_var:
             st.markdown("**Legend & Statistics**")
             c_leg1, c_leg2 = st.columns(2)

             with c_leg1:
                 # Metadata Cols (Grouping info)
                 # Default include grouping var + any other non-varying cols (heuristic?)
                 # For simplicity default to just the Group By variable
                 avail_meta = [hue_var] + [c for c in conditions if c != hue_var]
                 legend_meta_cols = st.multiselect(
                     "Metadata Columns",
                     avail_meta,
                     default=[hue_var],
                     help="Columns identifying the sample groups in the legend table."
                 )

             with c_leg2:
                 # Statistics Cols
                 avail_stats = ["Mean", "Median", "Count"]
                 legend_stats_cols = st.multiselect(
                     "Statistics",
                     avail_stats,
                     default=["Mean", "Count"],
                     help="Statistical metrics to display for each group."
                 )

             legend_columns = legend_meta_cols + legend_stats_cols

             # Font Size (Matplotlib only currently, but could apply to HTML table CSS)
             legend_font_size = st.number_input("Legend Font Size", min_value=5, max_value=20, value=10, step=1)

        replicate_col_selection = None
        show_replicates = False
        if viz_mode == "Ridgeline (Matplotlib)":
             show_replicates = st.checkbox("Show Replicates", value=False, help="Overlay individual replicates (e.g. files) on the same ridge.")
             if show_replicates:
                 # Attempt to identify potential replicate columns
                 # Use 'conditions' plus search for "filename" or "Experiment Name" if not present
                 options = list(conditions)

                 # Add other likely columns if they exist in the dataframe columns (checking filtered_experiment.data)
                 # Note: This runs before "Generate Plot", so we access the base data
                 candidates = ["replicate", "Replicate", "Experiment Name", "Experiment_Name", "filename", "File", "file", "Sample", "sample"]
                 existing_cols = set(filtered_experiment.data.columns)

                 for c in candidates:
                     if c in existing_cols and c not in options:
                         options.append(c)

                 if options:
                    # Default selection logic: Prioritize candidates order (e.g. replicate > Experiment Name)
                    default_idx = 0
                    for c in candidates:
                        if c in options:
                            default_idx = options.index(c)
                            break

                    replicate_col_selection = st.selectbox("Replicate Identifier", options, index=default_idx)
                 else:
                    st.warning("No suitable columns found to identify replicates.")

        st.write("") # Spacer

        if st.button("Generate Plot", type="primary", key="btn_hist"):
            st.session_state.hist_generated = True

        if st.session_state.get('hist_generated'):
            # Reload plots module
            from glow.plotting import histogram, plots as main_plots
            importlib.reload(histogram)
            importlib.reload(main_plots)

            with st.spinner("Generating plots..."):
                try:
                    # Apply Gates (Filtering + Scaling)
                    all_applied_gates = local_gates + analysis_gates_to_apply

                    filtered_df, processed_gates = apply_gates_to_dataframe(
                        filtered_experiment.data,
                        all_applied_gates,
                        experiment_metadata=getattr(filtered_experiment, 'metadata', None),
                        show_raw=st.session_state.get('plot_show_raw_vals', False)
                    )

                    if filtered_df.empty:
                        st.warning("No events remain after applying gates. Please adjust your gates.")
                        return

                    if scale == 'log':
                        # Check for positive values
                        if not (filtered_df[selected_channel] > 0).any():
                             st.warning(f"Data has no positive values in {selected_channel}, and therefore cannot be log-scaled. Use Linear or Logicle, or adjust gates.")
                             return

                    if processed_gates:
                        st.write(f"Applying {len(processed_gates)} filters...")

                    # Create temp experiment
                    plot_experiment = copy.copy(filtered_experiment)
                    plot_experiment.data = filtered_df

                    # Merged Plot
                    if show_merged:
                        st.subheader("Merged Plot")

                        if viz_mode == "Ridgeline (Overlay)":
                            try:
                                # --- Plotly Ridgeline Logic ---
                                fig = go.Figure()

                                plot_data = plot_experiment.data
                                groups = sorted(plot_data[hue_var].unique()) if hue_var and hue_var in plot_data.columns else ["All Events"]

                                import seaborn as sns
                                import matplotlib.colors as mc

                                if len(groups) > 1:
                                    # Match Matplotlib Logic: sns.color_palette("bright", n_groups)
                                    # Convert to Hex for Plotly
                                    palette = sns.color_palette("bright", len(groups))
                                    colors = [mc.to_hex(c) for c in palette]
                                else:
                                    # Single color fallback (bright blueish from palette or default)
                                    colors = [mc.to_hex(sns.color_palette("bright", 1)[0])]

                                all_vals = plot_data[selected_channel].values
                                all_vals = all_vals[~np.isnan(all_vals)]
                                use_log = (scale == 'log')

                                # Domain Calculation
                                if use_log:
                                    valid_vals = all_vals[all_vals > 0]
                                    if len(valid_vals) == 0:
                                        x_min, x_max = 1.0, 100
                                    else:
                                        x_min, x_max = np.percentile(valid_vals, 0.1), np.percentile(valid_vals, 99.9)
                                        if x_min <= 0:
                                            x_min = min(valid_vals) # Should not happen due to filter
                                        if x_min < 1.0:
                                            x_min = 1.0 # Enforce min 10^0

                                    # Fallback if range is 0
                                    if x_min >= x_max:
                                        x_max = x_min * 10

                                    x_log_domain = np.linspace(np.log10(x_min), np.log10(x_max), 200)
                                    x_domain = np.power(10, x_log_domain)
                                else:
                                    if len(all_vals) > 0:
                                        x_min, x_max = np.percentile(all_vals, 0.1), np.percentile(all_vals, 99.9)
                                    else:
                                        x_min, x_max = 0, 1

                                    if x_min >= x_max:
                                        x_max = x_min + 1.0
                                    pad = (x_max - x_min) * 0.1
                                    x_domain = np.linspace(x_min - pad, x_max + pad, 200)

                                for i, group in enumerate(groups):
                                    if hue_var and hue_var in plot_data.columns:
                                        g_vals = plot_data[plot_data[hue_var] == group][selected_channel].values
                                    else:
                                        g_vals = all_vals

                                    g_vals = g_vals[~np.isnan(g_vals)]
                                    if len(g_vals) < 2:
                                        continue

                                    try:
                                        if use_log:
                                            g_vals = g_vals[g_vals > 0]
                                            if len(g_vals) < 2:
                                                continue
                                            kde = gaussian_kde(np.log10(g_vals))
                                            y_vals = kde(np.log10(x_domain))
                                        else:
                                            kde = gaussian_kde(g_vals)
                                            y_vals = kde(x_domain)

                                        # Normalize
                                        if np.max(y_vals) > 0:
                                            y_vals = y_vals / np.max(y_vals)

                                        # Base and Offset
                                        base_y = i * ridgeline_offset
                                        y_plotted = y_vals + base_y

                                        # Color
                                        c = colors[i % len(colors)]

                                        # Create filled area (Pseudo-Ridgeline)
                                        # We use 'toself' with a closed polygon to fill relative to the baseline (base_y)
                                        x_poly = np.concatenate([x_domain, x_domain[::-1]])
                                        y_poly = np.concatenate([y_plotted, np.full_like(y_plotted, base_y)])

                                        fig.add_trace(go.Scatter(
                                            x=x_poly,
                                            y=y_poly,
                                            fill='toself',
                                            mode='lines',
                                            line=dict(color='white', width=0.1), # Invisible boundary for fill
                                            fillcolor=c,
                                            name=str(group),
                                            opacity=0.7,
                                            hovertemplate=f"<b>{group}</b><br>x: %{{x}}<br>Density: %{{y}}<extra></extra>"
                                        ))

                                        # Add Line on top for clarity (Black Outline)
                                        fig.add_trace(go.Scatter(
                                            x=x_domain,
                                            y=y_plotted,
                                            mode='lines',
                                            line=dict(color='black', width=1.0),
                                            showlegend=False,
                                            hoverinfo='skip'
                                        ))
                                    except Exception:
                                        continue

                                # Scale Height based on Offset to prevent squeezing
                                # Base height per plot (e.g. 50px) + Extra for stacking
                                # If offset is large, we need much more height.
                                # Total Height ~= (N_Groups * Offset * Scale_Factor) + Base_Per_Plot

                                # Heuristic: Start with 300px base.
                                # Add 30px per group if offset is small.
                                # If offset is large (>0.5), we are stacking high.
                                # Max value of y is roughly 1.0 per plot.
                                # Total Y range is roughly 1.0 + (N-1)*offset.

                                # To keep "visual height" of 1.0 unit constant (say 100 pixels),
                                # Total Pixels = (1.0 + (len(groups)-1)*ridgeline_offset) * 100

                                unit_height_px = 150 # Height for one histogram (normalized 0-1)
                                total_y_units = 1.0 + max(0, (len(groups) - 1)) * ridgeline_offset
                                calc_height = int(total_y_units * unit_height_px)

                                # Clamp sensible limits
                                calc_height = max(400, min(calc_height, 2000))

                                fig.update_layout(
                                    xaxis_type="log" if use_log else "linear",
                                    title=f"Ridgeline: {selected_channel} by {hue_var}" if hue_var else None,
                                    height=calc_height,
                                    template="plotly_white",
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                    margin=dict(l=20, r=20, t=40, b=20)
                                )
                                if not show_yaxis:
                                    fig.update_yaxes(showticklabels=False, title=None)

                                st.plotly_chart(fig, width="stretch")

                                # --- Separate Legend Table for Ridgeline ---
                                if hue_var and len(groups) > 0:
                                     # Calculate stats for the table
                                     table_data = []
                                     for idx, g in enumerate(groups):
                                         # Get subset
                                         sub_df = plot_data[plot_data[hue_var] == g]
                                         row = {}

                                         # Color indicator (can't easily do block in st.dataframe, maybe just use text or st.color_picker disabled?)
                                         # We'll just show the data

                                         # Metadata Cols
                                         # Re-use legend_meta_cols from config if available (need to ensure scope)
                                         # The variables legend_meta_cols and legend_stats_cols were defined in the Configuration block above
                                         # which is in the same scope (function body).

                                         # However, Streamlit reruns might clear locals if not careful, but this is all one run.
                                         # We reconstructed legend_columns = meta + stats

                                         # Let's iterate legend_columns to build the row
                                         if legend_columns:
                                              for col in legend_columns:
                                                   if col == "Mean":
                                                        vals = sub_df[selected_channel]
                                                        # Log scale aware stats? existing logic uses geometric mean if log
                                                        # Let's stick to standard arithmetic mean for simplicity or match matplotlib
                                                        # Matplotlib logic: if log_scale, geometric mean.
                                                        v = vals[vals > 0] if scale == 'log' else vals
                                                        if scale == 'log':
                                                             val = np.exp(np.log(v).mean()) if len(v) > 0 else 0
                                                        else:
                                                             val = v.mean()
                                                        row["Mean"] = f"{val:.2f}"

                                                   elif col == "Median":
                                                        row["Median"] = f"{sub_df[selected_channel].median():.2f}"

                                                   elif col == "Count":
                                                        row["Count"] = len(sub_df)

                                                   elif col == hue_var:
                                                        row[col] = g

                                                   elif col in sub_df.columns:
                                                        us = sub_df[col].unique()
                                                        row[col] = str(us[0]) if len(us) == 1 else "Mixed"
                                                   else:
                                                        row[col] = "-"
                                              table_data.append(row)

                                     if table_data:
                                          st.markdown("##### Statistics")
                                          st.dataframe(pd.DataFrame(table_data), hide_index=True, width="stretch")

                                # Canvas Button
                                canvas_title = f"Ridgeline: {selected_channel} by {hue_var}" if hue_var else f"{selected_channel}"
                                if st.button("Add Merged Plot to Canvas", key="add_merged_ridgeline_to_canvas"):
                                    if add_plot_to_canvas(fig, canvas_title):
                                        st.success("Plot added to canvas!")

                            except Exception as e:
                                st.error(f"Error generating Ridgeline plot: {e}")

                        elif viz_mode == "Ridgeline (Matplotlib)":
                            # New Matplotlib Ridgeline Implementation
                            try:
                                fig, stats = histogram.plot_ridgeline_matplotlib(
                                    plot_experiment,
                                    channel=selected_channel,
                                    hue=hue_var,
                                    scale=scale,
                                    offset=ridgeline_offset,
                                    smooth_sigma=smooth_sigma,
                                    normalization=normalization_mode,
                                    show_yaxis=show_yaxis,
                                    legend_cols=legend_columns,
                                    show_replicates=show_replicates,
                                    replicate_col=replicate_col_selection
                                )
                                with warnings.catch_warnings():
                                    warnings.filterwarnings("ignore", category=UserWarning)
                                    # Explicitly disable container width scaling to respect calculated figsize
                                    st.pyplot(fig, width="content")

                                if stats:
                                    st.markdown("#### Statistics")
                                    st.dataframe(pd.DataFrame(stats))

                                download_btn(fig, "hist_ridge_mpl")

                                canvas_title = f"Ridgeline (Mpl): {selected_channel}"
                                if st.button("Add Merged Plot to Canvas", key="add_mpl_ridge_canvas"):
                                    if add_plot_to_canvas(fig, canvas_title):
                                        st.success("Plot added to canvas!")

                                plt.close(fig)
                            except Exception as e:
                                st.error(f"Error generating Matplotlib Ridgeline: {e}")

                        else:
                            # Standard Matplotlib Logic
                            fig, stats = main_plots.plot_1d_histogram(
                                plot_experiment,
                                channel=selected_channel,
                                hue=hue_var,
                                scale=scale,
                                legend_cols=legend_columns,
                                legend_font_size=legend_font_size,
                                smooth_sigma=smooth_sigma,
                                normalization=normalization_mode,
                                show_yaxis=show_yaxis
                            )
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore", category=UserWarning, message="This figure includes Axes that are not compatible with tight_layout")
                                st.pyplot(fig)

                            if stats:
                                st.markdown("#### Statistics")
                                st.dataframe(pd.DataFrame(stats))

                            download_btn(fig, "hist_merged")

                            canvas_title = f"Histogram: {selected_channel} by {hue_var}" if hue_var else f"Histogram: {selected_channel}"
                            if st.button("Add Merged Plot to Canvas", key="add_merged_hist_to_canvas"):
                                if add_plot_to_canvas(fig, canvas_title):
                                    st.success("Plot added to canvas!")

                            if not getattr(fig, '_glow_is_direct_ref', False):
                                plt.close(fig)

                    # Individual Plots
                    if show_individual and hue_var:
                        st.subheader("Individual Plots")
                        groups = sorted(plot_experiment.data[hue_var].unique())

                        import seaborn as sns
                        palette = sns.color_palette("bright", len(groups))

                        cols = st.columns(2)
                        for i, group in enumerate(groups):
                            sub_exp = plots.filter_experiment(plot_experiment, hue_var, [group])
                            with cols[i % 2]:
                                fig, _ = main_plots.plot_1d_histogram(
                                    sub_exp,
                                    channel=selected_channel,
                                    hue=None,
                                    scale=scale,
                                    color=palette[i],
                                    title=f"Experiment: {group}",
                                    smooth_sigma=smooth_sigma,
                                    normalization=normalization_mode,
                                    show_yaxis=show_yaxis
                                )
                                with warnings.catch_warnings():
                                    warnings.filterwarnings("ignore", category=UserWarning, message="This figure includes Axes that are not compatible with tight_layout")
                                    st.pyplot(fig)

                                download_btn(fig, f"hist_ind_{i}")
                                if st.button(f"Add \"{group}\" Histogram to Canvas", key=f"add_hist_ind_{i}_to_canvas"):
                                    if add_plot_to_canvas(fig, f"Histogram: {selected_channel} ({group})"):
                                        st.success("Plot added to canvas!")

                                if not getattr(fig, '_glow_is_direct_ref', False):
                                    plt.close(fig)
                except Exception as e:
                    st.error(f"Error: {e}")
