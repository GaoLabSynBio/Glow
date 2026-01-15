import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import gridspec

def _gaussian_kernel1d(sigma: float, radius: int | None = None) -> np.ndarray:
    if sigma is None or sigma <= 0:
        return np.array([1.0])
    if radius is None:
        radius = int(np.ceil(3 * float(sigma)))
    x = np.arange(-radius, radius + 1)
    k = np.exp(-(x ** 2) / (2 * float(sigma) ** 2))
    k /= k.sum()
    return k


def _smooth_1d(y: np.ndarray, sigma: float) -> np.ndarray:
    k = _gaussian_kernel1d(sigma)
    if k.size == 1:
        return y
    return np.convolve(y, k, mode="same")


def _histogram_curve(values: np.ndarray, bins: int, log_scale: bool) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if log_scale:
        values = values[values > 0]
    if values.size == 0:
        return np.array([]), np.array([])

    if log_scale:
        vmin = values.min()
        vmax = values.max()
        if vmin <= 0:
            vmin = np.min(values[values > 0]) if np.any(values > 0) else 1e-3
        edges = np.logspace(np.log10(vmin), np.log10(vmax), int(bins) + 1)
    else:
        edges = np.linspace(values.min(), values.max(), int(bins) + 1)

    counts, edges = np.histogram(values, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts.astype(float)


def plot_1d_histogram(
    experiment,
    channel,
    hue=None,
    scale='log',
    bins=100,
    color=None,
    legend_cols=None,
    legend_font_size=10,
    title=None,
    smooth_sigma: float = 2.0,
    normalization: str = "total",
):
    """
    Generates a 1D Ridge Plot (stacked histograms) if hue is provided.
    Includes a statistics table matching the visual style of the reference.
    """
    df = experiment.data
    log_scale = (scale == 'log')

    norm = (normalization or "").lower().strip()
    if norm in {"none", "original", "raw"}:
        norm = "original"
    elif norm in {"modal", "mode"}:
        norm = "modal"
    elif norm in {"total", "count", "total_count", "area", "density"}:
        norm = "total"
    else:
        # Backward-safe default that matches KDE-style behavior
        norm = "total"

    # If no hue, simple histogram (Individual Plots)
    if not hue or hue not in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))

        c = color if color is not None else 'gray'

        x, y = _histogram_curve(df[channel].values, bins=bins, log_scale=log_scale)
        if x.size > 0:
            if norm == "total":
                s = y.sum()
                y = y / s if s > 0 else y
            elif norm == "modal":
                m = y.max()
                y = y / m if m > 0 else y

            y_s = _smooth_1d(y, smooth_sigma)
            ax.fill_between(x, y_s, step="mid", alpha=0.5, color=c)
            ax.plot(x, y_s, color=c, linewidth=2)

        if log_scale:
            ax.set_xscale('log')

        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"Histogram of {channel}")
        return fig

    # Ridge Plot Logic
    groups = sorted(df[hue].unique())
    n_groups = len(groups)

    if n_groups == 0:
        return plt.figure()

    # Determine columns to show in legend table
    if legend_cols is None:
        legend_cols = [hue, "Mean"]

    # If legend_cols is empty, user might want to hide table, but we will show at least colors?
    # Or just return plot without table?
    show_table = len(legend_cols) > 0

    # Calculate stats for the table
    table_data = []

    for g in groups:
        sub = df[df[hue] == g]
        row_vals = []

        # First col is always Color Block (added later during table construction) hence strict loop over requested cols
        for col_key in legend_cols:
            if col_key == "Mean":
                vals = sub[channel]
                if log_scale:
                     v = vals[vals > 0]
                     mean_val = np.exp(np.log(v).mean()) if len(v) > 0 else 0
                else:
                    mean_val = vals.mean()
                row_vals.append(f"{mean_val:.2f}")

            elif col_key == "Median":
                 median_val = sub[channel].median()
                 row_vals.append(f"{median_val:.2f}")

            elif col_key == "Count":
                 row_vals.append(str(len(sub)))

            elif col_key == hue:
                row_vals.append(str(g))

            elif col_key in df.columns:
                # Try to get unique value
                u_vals = sub[col_key].unique()
                if len(u_vals) == 1:
                    row_vals.append(str(u_vals[0]))
                else:
                    row_vals.append("Mixed")
            else:
                row_vals.append("-")

        table_data.append(row_vals)

    # Layout
    plt.rcParams.update({'font.size': 10})

    # Responsive height
    if show_table:
        fig = plt.figure(figsize=(10, max(4, n_groups * 0.8)))
        # GridSpec: 4 parts plot, 1 part table (or wider if more cols)
        gs = gridspec.GridSpec(n_groups, 2, width_ratios=[3, 1 + 0.2*len(legend_cols)], wspace=0.1, hspace=0)
    else:
        fig = plt.figure(figsize=(8, max(4, n_groups * 0.8)))
        gs = gridspec.GridSpec(n_groups, 1, hspace=0)

    palette = sns.color_palette("bright", n_groups)

    # Determine x limits
    if log_scale:
        valid_data = df[df[channel] > 0][channel]
        if len(valid_data) > 0:
            x_min = 10**(np.floor(np.log10(valid_data.min())))
            x_max = 10**(np.ceil(np.log10(valid_data.max())))
        else:
            x_min, x_max = 0.1, 1000
    else:
        x_min = df[channel].min()
        x_max = df[channel].max()

    axes = []
    y_max_global = 0.0
    cached_curves: list[tuple[np.ndarray, np.ndarray]] = []

    # Precompute curves so we can enforce consistent y limits across rows
    for group in groups:
        subset = df[df[hue] == group]
        x, y = _histogram_curve(subset[channel].values, bins=bins, log_scale=log_scale)
        if x.size == 0:
            cached_curves.append((x, y))
            continue

        if norm == "total":
            s = y.sum()
            y = y / s if s > 0 else y
        elif norm == "modal":
            m = y.max()
            y = y / m if m > 0 else y

        y = _smooth_1d(y, smooth_sigma)
        cached_curves.append((x, y))
        if y.size > 0:
            y_max_global = max(y_max_global, float(np.nanmax(y)))

    for i, group in enumerate(groups):
        # Plot Axis (Left Column)
        if show_table:
            ax = fig.add_subplot(gs[i, 0])
        else:
            ax = fig.add_subplot(gs[i])

        axes.append(ax)

        color = palette[i]

        x, y = cached_curves[i]
        if x.size > 0:
            ax.fill_between(x, y, step="mid", alpha=0.5, color=color)
            ax.plot(x, y, color=color, linewidth=1.5)

        # Aesthetics
        ax.set_ylabel("")
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.grid(False)
        ax.set_facecolor('white')

        # X Axis
        ax.set_xlim(x_min, x_max)
        if y_max_global > 0:
            ax.set_ylim(0, y_max_global * 1.05)
        if i < n_groups - 1:
            ax.set_xlabel("")
            ax.set_xticklabels([])
            ax.spines['bottom'].set_visible(False)
            ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
        else:
            ax.set_xlabel(channel)
            ax.spines['bottom'].set_visible(True)

        if log_scale:
            ax.set_xscale('log')

    # Table (Right Column)
    if show_table:
        ax_table = fig.add_subplot(gs[:, 1])
        ax_table.axis('off')

        # Legend Table Header
        col_labels = [""] + legend_cols

        # Use a Unicode block for the color box: ■ (U+25A0)
        # Prepend color block char to each row
        table_content = [["■"] + row for row in table_data]

        # Dynamic column widths
        n_cols = len(col_labels)
        # First col (color) small, others equal
        widths = [0.05] + [0.95/(n_cols-1)] * (n_cols-1)

        tbl = ax_table.table(
            cellText=table_content,
            colLabels=col_labels,
            loc='center',
            cellLoc='left',
            colWidths=widths,
            edges='horizontal'
        )

        tbl.auto_set_font_size(False)
        tbl.set_fontsize(legend_font_size)
        tbl.scale(1, 1.5)

        # Style stats
        for i in range(n_groups):
            # Row index i+1 (header is 0)

            # 1. Color Block (Column 0)
            cell_color = tbl[(i+1, 0)]
            # Set TEXT color to the palette color
            cell_color.set_text_props(color=palette[i], weight='bold', size=14)
            # Clear background/borders for cleaner look
            cell_color.set_facecolor('white')
            cell_color.set_edgecolor('white')

            # 2. Style other cells?
            for j in range(1, n_cols):
                 cell = tbl[(i+1, j)]
                 if j == 1: # Usually the ID column
                     cell.set_text_props(weight='bold')

    return fig
