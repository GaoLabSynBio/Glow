import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.colors as mc
import colorsys
from .utils import get_point_density
import plotly.express as px

def calc_log_limits(data_series):
    """
    Calculates log axis limits:
    Min: Closest lower 10^x (minimal 10^0)
    Max: 10^5 higher than min
    """
    # Filter valid positive values
    valid = data_series[data_series > 0]
    if len(valid) == 0:
        return None

    min_val = valid.min()
    # floor log10
    base_exponent = np.floor(np.log10(min_val))

    # "minimal 10^0"
    final_exponent = max(0, base_exponent)

    low = 10**final_exponent
    high = 10**(final_exponent + 5)

    return low, high

def plot_2d_density_with_running_means(experiment, x_channel, y_channel, facet_col=None, hue_col=None, xscale='log', yscale='log', batch_size=1000, show_dots=True, show_dots_summary=None, show_grid=False, merged_figsize=(5, 5)):
    """
    Plots a row of 2D plots (one per facet).
    Overlays running means:
    - Main lines for the current facet (bold).
    - Ghost lines for other facets (faint, for comparison).
    Adapted from user provided 'annotate_2d_density_with_running_mean'.
    """
    # Heuristic for legend scaling based on summary figure size (fixed 5x5 currently)
    # If we want it to be "dynamic", we assume 5x5 is the "standard" size.
    # Legend is separate.
    legend_font_size = 14
    legend_marker_scale = 1.5

    if show_dots_summary is None:
        show_dots_summary = show_dots

    df = experiment.data

    # Calc custom limits if log
    x_lims = calc_log_limits(df[x_channel]) if xscale == 'log' else None
    y_lims = calc_log_limits(df[y_channel]) if yscale == 'log' else None

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

    max_cols = 5
    n_cols = min(n_facets, max_cols)
    n_rows = (n_facets + n_cols - 1) // n_cols

    # Use constrained_layout or tight_layout
    # Defaut to 300px (3 inches) for individual plots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), constrained_layout=True, sharex=True, sharey=True)

    if n_rows * n_cols > 1:
        axs = axs.flatten()
    else:
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

    # Color Palette per Color Category: Distinct Hues, Excluding Greens
    n_needed = len(unique_colors) if len(unique_colors) > 0 else 1

    # Generate a large pool to pick from (HUSL for perceptual uniformity)
    n_pool = max(50, n_needed * 5)
    pool_palette = sns.color_palette("husl", n_pool)

    valid_colors = []
    for c in pool_palette:
        h, b, s = colorsys.rgb_to_hls(*c)
        valid_colors.append(c)

    # Select indices maximizing distance
    safe_palette = []
    if len(valid_colors) > 0:
        if len(valid_colors) >= n_needed:
             # Use endpoint=False to avoid picking the first and last colors which are often similar (Red vs Red-Purple)
             # given that the valid pool wraps around in hue space (Start is Red, End is Red-wrap)
             indices = np.linspace(0, len(valid_colors), n_needed, endpoint=False, dtype=int)
             selection = [valid_colors[i] for i in indices]
        else:
             # Cycle if needed
             selection = valid_colors * (n_needed // len(valid_colors) + 1)
             selection = selection[:n_needed]
    else:
        selection = sns.color_palette("bright", n_needed)

    # Apply brightness/saturation tweaks
    for c in selection:
         h, b, s = colorsys.rgb_to_hls(*c)
         # Slightly brighten and ensure high saturation as requested
         new_l = min(0.85, max(0.5, b * 1.1))
         # Increase saturation slightly but cap at 1.0
         new_s = min(1.0, max(0.6, s * 1.2))
         safe_palette.append(colorsys.hls_to_rgb(h, new_l, new_s))

    color_map = {h: safe_palette[i] for i, h in enumerate(unique_colors)}

    # Detect Replicate Column
    rep_col = None
    for c in df.columns:
        if c.lower() == 'replicate':
            rep_col = c
            break

    # 3. Pre-calculate Facet Traces (For Separate Plots - Split by Replicates)
    facet_traces = []

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

        for group_name, exp_subset in groups:
            # Handle Replicates: Split exp_subset into replicates if available
            replicate_subgroups = []
            if rep_col:
                # Group by replicate
                for rep_val, rep_df in exp_subset.groupby(rep_col, observed=False):
                    replicate_subgroups.append((rep_val, rep_df))
            else:
                # Treat as single group
                replicate_subgroups.append(("All", exp_subset))

            # Determine base color and label based on hue_col
            if hue_col and hue_col in df.columns:
                h_val = exp_subset[hue_col].iloc[0]
                base_color = color_map.get(h_val, 'black')
                base_label = f"{h_val} ({group_name})" if group_name is not None else str(h_val)
            else:
                h_val = None
                base_color = 'black'
                base_label = str(group_name) if group_name is not None else "Mean" # Use group name if no hue

            # Iterate through replicates to create traces
            n_reps = len(replicate_subgroups)
            for i, (rep_name, subset_df) in enumerate(replicate_subgroups):

                if len(subset_df) < (batch_size // 5):
                    continue

                # Calculate Color Variation (Lightness Shift)
                final_color = base_color
                label = base_label

                if n_reps > 1:
                    label = f"{base_label} (Rep {rep_name})"
                    try:
                        c_rgb = mc.to_rgb(base_color)
                        c_hls = colorsys.rgb_to_hls(*c_rgb)
                        # Spread lightness centered around original L
                        # Shift range +/- 0.15 depending on number of reps
                        if n_reps > 1:
                            # Map i to range [-1, 1]
                            rel_pos = (i / (n_reps - 1)) * 2 - 1  if n_reps > 1 else 0
                            # Shift lightness: e.g. -0.15 to +0.15
                            l_shift = rel_pos * 0.15
                            new_l = max(0.2, min(0.9, c_hls[1] + l_shift))
                            final_color = colorsys.hls_to_rgb(c_hls[0], new_l, c_hls[2])
                    except Exception:
                        pass # Keep base color if conversion fails

                # Calculate Rolling Mean
                subset_df = subset_df.sort_values(by=x_channel)

                eff_batch = batch_size
                if len(subset_df) < eff_batch:
                    eff_batch = max(5, int(len(subset_df) * 0.1))

                roller = subset_df.rolling(window=eff_batch, min_periods=eff_batch//2, center=True)
                roll_x = roller[x_channel].mean()
                roll_y = roller[y_channel].mean()

                facet_traces.append({
                    'facet': facet,
                    'group': group_name,
                    'x': roll_x,
                    'y': roll_y,
                    'color': final_color,
                    'label': label,
                    'hue_val': h_val
                })

    # 4. Pre-calculate Summary Traces (For Merged Plot - Pooled Replicates)
    summary_traces = []

    summary_group_col = hue_col if hue_col else exp_col
    # If no hue and no experiment column found? Rare, but default to whole DF
    if not summary_group_col:
        sum_groups = [None]
    else:
        # Sort to ensure consistent color mapping
        sum_groups = sorted(df[summary_group_col].unique())

    for g_val in sum_groups:
        if g_val is None:
            g_df = df
        else:
            g_df = df[df[summary_group_col] == g_val]

        if len(g_df) < (batch_size // 5):
            continue

        if hue_col:
            base_color = color_map.get(g_val, 'black')
        else:
            # When grouping by experiment (no Hue), assign distinct colors cyclically from safe palette
            if safe_palette and sum_groups:
                 base_color = safe_palette[sum_groups.index(g_val) % len(safe_palette)]
            else:
                 base_color = 'black'

        # Calculate Rolling Mean on POOLED data (all replicates)
        g_df_sorted = g_df.sort_values(by=x_channel)

        eff_batch = batch_size
        if len(g_df_sorted) < eff_batch:
            eff_batch = max(5, int(len(g_df_sorted) * 0.1))

        roller = g_df_sorted.rolling(window=eff_batch, min_periods=eff_batch//2, center=True)
        roll_x = roller[x_channel].mean()
        roll_y = roller[y_channel].mean()

        summary_traces.append({
            'x': roll_x,
            'y': roll_y,
            'color': base_color,
            'label': str(g_val)
        })


    # 4. Plotting Loop
    for i, ax in enumerate(axs):
        if i >= n_facets:
            ax.set_visible(False)
            continue

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
            # z is already filtered to match mask by get_point_density, so we don't apply mask to it again.
            # z = z[mask] <- This was the bug

            # Sort by density so high density dots are on top
            sort_idx = np.argsort(z)

            ax.scatter(x_vals[sort_idx], y_vals[sort_idx], c=z[sort_idx], cmap='viridis_r', s=3, alpha=0.6, rasterized=True, zorder=0)

        if show_grid:
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
        else:
            ax.grid(False)
            ax.set_facecolor('white')

        # B. Plot Traces (Facet specific)
        for trace in facet_traces:
            # Check if this trace belongs to current facet
            is_current = (trace['facet'] == current_facet)

            if is_current:
                # Main Line: Bold, opaque
                ax.plot(trace['x'], trace['y'], color=trace['color'], linewidth=3, alpha=1.0, label=trace['label'], zorder=10)

        # Axis scales/labels
        if xscale == 'log':
            ax.set_xscale('log')
            if x_lims:
                ax.set_xlim(x_lims)
        if yscale == 'log':
            ax.set_yscale('log')
            if y_lims:
                ax.set_ylim(y_lims)

        # Only show xlabel on bottom-most plot in column (handle ragged grid)
        if i + n_cols >= n_facets:
            ax.set_xlabel(x_channel)
            ax.tick_params(labelbottom=True)

        # Only show ylabel on left-most plot in row
        if i % n_cols == 0:
            ax.set_ylabel(y_channel)


    # 5. Create Separate Summary Figure
    # Default to 500px (5 inches) for merged plot
    fig_summary, ax_sum = plt.subplots(figsize=merged_figsize)
    ax_sum.set_title("Merged")

    if show_dots_summary:
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

    for trace in summary_traces:
        # Plot pooled/summary traces on summary
        # For the legend, we still want to group by Hue if labels are messy
        ax_sum.plot(trace['x'], trace['y'], color=trace['color'], linewidth=3.5, alpha=1.0, label=trace['label'], zorder=10)


    # De-duplicate legend to show unique Hues or group names
    handles, labels = ax_sum.get_legend_handles_labels()

    legend_args = None
    legend_kwargs = {}

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
        if clean_handles:
             legend_args = (clean_handles, clean_labels)
             legend_kwargs = {'title': hue_col, 'loc': 'center left', 'fontsize': legend_font_size, 'markerscale': legend_marker_scale}
    else:
        by_label = dict(zip(labels, handles))
        if by_label:
            legend_args = (by_label.values(), by_label.keys())
            legend_kwargs = {'loc': 'center left', 'fontsize': legend_font_size, 'markerscale': legend_marker_scale}

    # Separate Legend Figure
    fig_legend = None
    if legend_args:
        # Create a separate figure for the legend
        # Scale figure size based on font/marker scale?
        # Default font size is ~10pt. If user sets 5, box should shrink.
        scale_factor = (legend_font_size / 10.0) * legend_marker_scale

        n_items = len(legend_args[0])
        fig_h = max(1.0, n_items * 0.2 * scale_factor)
        fig_w = 1.5 * scale_factor

        fig_legend = plt.figure(figsize=(fig_w, fig_h))
        fig_legend.patch.set_alpha(0.0)

        # Enforce fontsize
        legend_kwargs['fontsize'] = legend_font_size
        if 'title_fontsize' not in legend_kwargs:
             legend_kwargs['title_fontsize'] = legend_font_size

        # Use fig.legend to place it centered or filling the figure
        # We put it at center of figure
        fig_legend.legend(*legend_args, **legend_kwargs)

    if xscale == 'log':
        ax_sum.set_xscale('log')
        if x_lims:
            ax_sum.set_xlim(x_lims)
    if yscale == 'log':
        ax_sum.set_yscale('log')
        if y_lims:
            ax_sum.set_ylim(y_lims)
    ax_sum.set_xlabel(x_channel)
    ax_sum.set_ylabel(y_channel)

    # Force 1:1 Aspect Ratio on the Axes Box
    ax_sum.set_box_aspect(1)

    return fig, fig_summary, fig_legend

def plot_2d_scatter(experiment, x_channel, y_channel, hue=None, facet=None, xscale='log', yscale='log', show_trend=False, show_dots=True, show_dots_summary=None, show_grid=False, title=None, figsize=(4, 4), merged_figsize=(5, 5)):
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
            show_dots_summary=show_dots_summary,
            show_grid=show_grid,
            merged_figsize=merged_figsize
        )

    # Otherwise standard scatter
    df = experiment.data
    fig, ax = plt.subplots(figsize=figsize)

    # Log scale
    log_x = (xscale == 'log')
    log_y = (yscale == 'log')

    sns.scatterplot(
        data=df, x=x_channel, y=y_channel, hue=hue,
        s=10, alpha=0.6, ax=ax
    )

    if log_x:
        ax.set_xscale('log')
        x_lims = calc_log_limits(df[x_channel])
        if x_lims:
            ax.set_xlim(x_lims)
    if log_y:
        ax.set_yscale('log')
        y_lims = calc_log_limits(df[y_channel])
        if y_lims:
            ax.set_ylim(y_lims)

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
