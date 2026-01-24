import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .utils import get_point_density
import plotly.express as px

def plot_2d_density_with_running_means(experiment, x_channel, y_channel, facet_col=None, hue_col=None, xscale='log', yscale='log', batch_size=1000, show_dots=True, show_grid=False):
    """
    Plots a row of 2D plots (one per facet).
    Overlays running means:
    - Main lines for the current facet (bold).
    - Ghost lines for other facets (faint, for comparison).
    Adapted from user provided 'annotate_2d_density_with_running_mean'.
    """
    df = experiment.data

    # User requested: no matter what is chosen for color and facet, each experiment name should has its own curve.
    # We use "Experiment Name" as the internal grouping variable for trend lines.
    exp_col = None
    for c in ["Experiment Name", "Experiment_Name", "Sample", "sample"]:
        if c in df.columns:
            exp_col = c
            break

    # 1. Determine Facets (Axes)
    if facet_col and facet_col in df.columns:
        facets = sorted(df[facet_col].unique())
    else:
        facets = ["All Data"]

    n_facets = len(facets)
    # Use constrained_layout or tight_layout
    fig, axs = plt.subplots(1, n_facets, figsize=(5 * n_facets, 5), constrained_layout=True, sharex=True, sharey=True)

    if n_facets == 1:
        axs = [axs]

    # 2. Determine Hues (Lines within each facet)
    # The 'Hues' here determine the COLOR of the lines.
    # If hue_col is provided, we color by that.
    # Otherwise, if we group by experiment, we might want unique colors for experiments or a single color.
    # To satisfy the request: each experiment gets a curve.
    # We will iterate over unique experiments within each facet.
    if hue_col and hue_col in df.columns:
        unique_colors = sorted(df[hue_col].unique())
    else:
        unique_colors = [None]

    # Color Palette per Color Category
    palette = sns.color_palette("bright", len(unique_colors)) if len(unique_colors) > 0 else ['black']
    color_map = {h: palette[i] for i, h in enumerate(unique_colors)}

    # 3. Pre-calculate all trend lines
    all_traces = []

    for facet in facets:
        if facet == "All Data":
            facet_df = df
        else:
            facet_df = df[df[facet_col] == facet]

        if len(facet_df) == 0:
            continue

        # Group by Experiment Name (the requested individual curve)
        if exp_col:
            groups = facet_df.groupby(exp_col, observed=False)
        else:
            groups = [(None, facet_df)]

        for group_name, subset_df in groups:
            if len(subset_df) < (batch_size // 5):
                continue

            # Determine color and label based on hue_col
            if hue_col and hue_col in df.columns:
                # Assuming one hue per experiment in a facet (common in flow)
                # If an experiment has multiple hues, this takes the first one found
                h_val = subset_df[hue_col].iloc[0]
                color = color_map.get(h_val, 'black')
                label = f"{h_val} ({group_name})" if group_name is not None else str(h_val)
            else:
                color = 'black'
                label = str(group_name) if group_name is not None else "Mean"

            # Calculate Rolling Mean
            subset_df = subset_df.sort_values(by=x_channel)

            eff_batch = batch_size
            if len(subset_df) < eff_batch:
                eff_batch = max(5, int(len(subset_df) * 0.1))

            roller = subset_df.rolling(window=eff_batch, min_periods=eff_batch//2, center=True)
            roll_x = roller[x_channel].mean()
            roll_y = roller[y_channel].mean()

            all_traces.append({
                'facet': facet,
                'group': group_name,
                'x': roll_x,
                'y': roll_y,
                'color': color,
                'label': label,
                'hue_val': h_val if hue_col else None
            })

    # 4. Plotting Loop
    for i, ax in enumerate(axs):
        current_facet = facets[i]
        ax.set_title(str(current_facet))

        # A. Background Scatter (Density or Grey)
        if show_dots:
            if current_facet == "All Data":
                bg_data = df
            else:
                bg_data = df[df[facet_col] == current_facet]

            # Sample for performance/visuals if NOT calculating full density?
            # Or sample anyway because density on 1M points is slow to render.
            n_points = len(bg_data)
            if n_points > 10000:
                 # If coloring by density, ideally we estimate density on FULL data, then sample points to plot.
                 # But for speed, sample first.
                 bg_data = bg_data.sample(10000, random_state=42)

            x_vals = bg_data[x_channel].values
            y_vals = bg_data[y_channel].values

            # Density Color Calculation
            z, mask = get_point_density(x_vals, y_vals, log_x=(xscale=='log'), log_y=(yscale=='log'))
            x_vals = x_vals[mask]
            y_vals = y_vals[mask]
            z = z[mask] # 0..1

            # Sort by density so high density dots are on top
            sort_idx = np.argsort(z)

            ax.scatter(x_vals[sort_idx], y_vals[sort_idx], c=z[sort_idx], cmap='viridis_r', s=3, alpha=0.6, rasterized=True, zorder=0)

        if show_grid:
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
        else:
            ax.grid(False)
            ax.set_facecolor('white')

        # B. Plot Traces
        for trace in all_traces:
            # Check if this trace belongs to current facet
            is_current = (trace['facet'] == current_facet)

            if is_current:
                # Main Line: Bold, opaque
                ax.plot(trace['x'], trace['y'], color=trace['color'], linewidth=3, alpha=1.0, label=trace['label'], zorder=10)

            # Removed Ghost Lines (compare=False behavior for individual plots)


        # Add Legend only to first plot or make it unique?
        # If hues are consistent, one legend is enough.
        # User requested to remove legend on separate plots
        # if i == 0 and len(all_traces) > 0:
        #     # Filter handles to avoid duplicates and ghost lines
        #     handles, labels = ax.get_legend_handles_labels()
        #     by_label = dict(zip(labels, handles))
        #     if by_label:
        #         ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize='small')

        # Axis scales/labels
        if xscale == 'log':
            ax.set_xscale('log')
        if yscale == 'log':
            ax.set_yscale('log')
        ax.set_xlabel(x_channel)


    # 5. Create Separate Summary Figure
    fig_summary, ax_sum = plt.subplots(figsize=(6, 6))
    ax_sum.set_title("Merged Comparison")

    if show_dots:
         # For summary, maybe just sample randomly from everything?
        if len(df) > 5000:
            sum_bg = df.sample(5000, random_state=42)
        else:
            sum_bg = df

        # Density on pooled data? Or just grey?
        # "merged is with itself" -> usually trends.
        # But if we want scatter, maybe just grey is safer to avoid confusion with density of specific groups.
        # Or calculate global density. Let's do global density.
        sx = sum_bg[x_channel].values
        sy = sum_bg[y_channel].values
        sz, smask = get_point_density(sx, sy, log_x=(xscale=='log'), log_y=(yscale=='log'))
        sx = sx[smask]
        sy = sy[smask]
        sz = sz[smask]
        s_idx = np.argsort(sz)

        ax_sum.scatter(sx[s_idx], sy[s_idx], c=sz[s_idx], cmap='viridis_r', s=2, alpha=0.5, rasterized=True, zorder=0)

    if show_grid:
        ax_sum.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    else:
        ax_sum.grid(False)
        ax_sum.set_facecolor('white')

    for trace in all_traces:
        # Plot all traces on summary
        # For the legend, we still want to group by Hue if labels are messy
        ax_sum.plot(trace['x'], trace['y'], color=trace['color'], linewidth=2.5, alpha=0.8, label=trace['label'], zorder=10)


    # De-duplicate legend to show unique Hues or group names
    handles, labels = ax_sum.get_legend_handles_labels()
    # If we have many lines but few hues, we can simplify legend to show just hues
    if hue_col:
        # Map back to hue label only for a cleaner legend
        clean_labels = []
        clean_handles = []
        seen = set()
        for h, lab in zip(handles, labels):
             # Extract the hue part from "Hue (Experiment)"
             base_l = lab.split(" (")[0]
             if base_l not in seen:
                  clean_labels.append(base_l)
                  clean_handles.append(h)
                  seen.add(base_l)
        ax_sum.legend(clean_handles, clean_labels, loc='best', fontsize='small', title=hue_col)
    else:
        by_label = dict(zip(labels, handles))
        if by_label:
            ax_sum.legend(by_label.values(), by_label.keys(), loc='best', fontsize='small')

    if xscale == 'log':
        ax_sum.set_xscale('log')
    if yscale == 'log':
        ax_sum.set_yscale('log')
    ax_sum.set_xlabel(x_channel)
    ax_sum.set_ylabel(y_channel)


    return fig, fig_summary

def plot_2d_scatter(experiment, x_channel, y_channel, hue=None, facet=None, xscale='log', yscale='log', show_trend=False, show_dots=True, show_grid=False, title=None):
    """
    Wrapper decision maker for 2D plots.
    """
    # If show_trend is requested, use the fancy specialized plot
    if show_trend:
        # We Map 'facet' arg to the 'facet_col' and 'hue' to 'hue_col'
        return plot_2d_density_with_running_means(
            experiment, x_channel, y_channel,
            facet_col=facet,
            hue_col=hue,
            xscale=xscale,
            yscale=yscale,
            batch_size=1000,
            show_dots=show_dots,
            show_grid=show_grid
        )

    # Otherwise standard scatter
    df = experiment.data
    fig, ax = plt.subplots(figsize=(8, 6))

    # Log scale
    log_x = (xscale == 'log')
    log_y = (yscale == 'log')

    sns.scatterplot(
        data=df, x=x_channel, y=y_channel, hue=hue,
        s=10, alpha=0.6, ax=ax
    )

    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')

    if title:
        ax.set_title(title)

    return fig


def plot_2d_scatter_plotly(experiment, x_channel, y_channel, hue=None, facet=None, scale='linear', show_grid=False, title=None):
    """
    Plotly version of 2D Scatter.
    """
    df = experiment.data.copy()

    # Ensure numerical consistency if Scale is log
    log_x = (scale == 'log')
    log_y = (scale == 'log')

    # Plotly handles log scaling in update_layout

    # Construct title
    full_title = title if title else f"{x_channel} vs {y_channel}"

    # Basic Scatter
    # We use plotly express for easy facet/hue handling

    # Subsample if too large for interactivity? Plotly WebGL can handle ~100k reasonably well.
    # Let's cap at 50k for safety for now or let it ride.
    if len(df) > 50000:
        df = df.sample(50000, random_state=42)
        full_title += " (Subsampled 50k)"

    fig = px.scatter(
        df,
        x=x_channel,
        y=y_channel,
        color=hue if hue in df.columns else None,
        facet_col=facet if facet in df.columns else None,
        facet_col_wrap=3 if facet else 0,
        title=full_title,
        opacity=0.6,
        render_mode='webgl' # Faster rendering
    )

    # Update Axes
    fig.update_xaxes(showgrid=show_grid)
    fig.update_yaxes(showgrid=show_grid)

    if log_x:
        fig.update_xaxes(type="log")
    if log_y:
        fig.update_yaxes(type="log")

    # Layout Tweaks
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig
