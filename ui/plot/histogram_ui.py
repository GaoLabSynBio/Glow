import streamlit as st
import matplotlib.pyplot as plt
import importlib
import copy
import warnings
import plotly.graph_objects as go
import plotly.express as px
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
            compare_by = st.selectbox(
                "Color/Group By",
                ["None"] + conditions,
                index=0 if not conditions else 1,
                help="Select attribute for coloring/grouping histograms"
            )
            hue_var = None if compare_by == "None" else compare_by

        # Histogram specific settings
        st.markdown("---")
        st.markdown("**Rendering Options**")

        viz_mode = st.radio(
            "Visualization Style", 
            ["Ridgeline (Overlay)", "Standard (Matplotlib)"], 
            index=0,
            help="Ridgeline: Interactive overlaid plots (like Analysis tab). Standard: Stacked separate plots."
        )

        ridgeline_offset = 0.5
        if viz_mode == "Ridgeline (Overlay)":
             ridgeline_offset = st.slider("Overlay Offset", 0.0, 2.0, 0.5, 0.1)

        smooth_sigma = st.slider(
            "Smoothing (Standard only)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.2,
            help="Controls smoothing strength of the histogram curve (Matplotlib mode). Higher values = smoother curve."
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
        if hue_var:
             # Legend Column Customization
             st.markdown("**Legend Configuration**")
             extra_meta = [c for c in conditions if c != hue_var]
             avail_cols = [hue_var] + extra_meta + ["Mean", "Median", "Count"]

             # Default: hue_var + Mean
             default_cols = [hue_var, "Mean"]
             legend_columns = st.multiselect("Legend Columns", avail_cols, default=default_cols)
             legend_font_size = st.number_input("Legend Font Size", min_value=5, max_value=20, value=10, step=1)

        st.markdown("---")

    # --- RIGHT COLUMN: Plot Display ---
    with col_plot:
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
                                
                                if len(groups) > 1:
                                    colors = px.colors.sample_colorscale("Viridis", np.linspace(0, 1, len(groups)))
                                else:
                                    colors = ["#636EFA"]

                                all_vals = plot_data[selected_channel].values
                                all_vals = all_vals[~np.isnan(all_vals)]
                                use_log = (scale == 'log')
                                
                                # Domain Calculation
                                if use_log:
                                    valid_vals = all_vals[all_vals > 0]
                                    if len(valid_vals) == 0:
                                        x_min, x_max = 0.1, 100
                                    else:
                                        x_min, x_max = np.percentile(valid_vals, 0.1), np.percentile(valid_vals, 99.9)
                                        if x_min <= 0: x_min = min(valid_vals) # Should not happen due to filter
                                    
                                    # Fallback if range is 0
                                    if x_min >= x_max: x_max = x_min * 10
                                    
                                    x_log_domain = np.linspace(np.log10(x_min), np.log10(x_max), 200)
                                    x_domain = np.power(10, x_log_domain)
                                else:
                                    if len(all_vals) > 0:
                                        x_min, x_max = np.percentile(all_vals, 0.1), np.percentile(all_vals, 99.9)
                                    else:
                                        x_min, x_max = 0, 1
                                    
                                    if x_min >= x_max: x_max = x_min + 1.0
                                    pad = (x_max - x_min) * 0.1
                                    x_domain = np.linspace(x_min - pad, x_max + pad, 200)

                                for i, group in enumerate(groups):
                                    if hue_var and hue_var in plot_data.columns:
                                        g_vals = plot_data[plot_data[hue_var] == group][selected_channel].values
                                    else:
                                        g_vals = all_vals
                                    
                                    g_vals = g_vals[~np.isnan(g_vals)]
                                    if len(g_vals) < 2: continue
                                    
                                    try:
                                        if use_log:
                                            g_vals = g_vals[g_vals > 0]
                                            if len(g_vals) < 2: continue
                                            kde = gaussian_kde(np.log10(g_vals))
                                            y_vals = kde(np.log10(x_domain))
                                        else:
                                            kde = gaussian_kde(g_vals)
                                            y_vals = kde(x_domain)
                                        
                                        # Normalize
                                        if np.max(y_vals) > 0:
                                            y_vals = y_vals / np.max(y_vals)
                                        
                                        # Offset
                                        y_plotted = y_vals + (i * ridgeline_offset)
                                        
                                        # Color
                                        c = colors[i % len(colors)]
                                        
                                        fig.add_trace(go.Scatter(
                                            x=x_domain, y=y_plotted, 
                                            mode='lines', 
                                            line=dict(color=c, width=1), 
                                            name=str(group), 
                                            fill='tozeroy', 
                                            opacity=0.6,
                                            hovertemplate=f"<b>{group}</b><br>x: %{{x}}<br>Density: %{{y}}<extra></extra>"
                                        ))
                                    except Exception:
                                        continue

                                fig.update_layout(
                                    xaxis_type="log" if use_log else "linear",
                                    title=f"Ridgeline: {selected_channel} by {hue_var}" if hue_var else None,
                                    height=max(400, 100 + len(groups) * 20),
                                    template="plotly_white",
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                    margin=dict(l=20, r=20, t=40, b=20)
                                )
                                if not show_yaxis:
                                    fig.update_yaxes(showticklabels=False, title=None)

                                st.plotly_chart(fig, use_container_width=True)

                                # Canvas Button
                                canvas_title = f"Ridgeline: {selected_channel} by {hue_var}" if hue_var else f"{selected_channel}"
                                if st.button("Add Merged Plot to Canvas", key="add_merged_ridgeline_to_canvas"):
                                    if add_plot_to_canvas(fig, canvas_title):
                                        st.success("Plot added to canvas!")

                            except Exception as e:
                                st.error(f"Error generating Ridgeline plot: {e}")

                        else:
                            # Standard Matplotlib Logic
                            fig = main_plots.plot_1d_histogram(
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
                                fig = main_plots.plot_1d_histogram(
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
