import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import gridspec
from matplotlib.ticker import ScalarFormatter, NullFormatter, MaxNLocator, LogLocator, LogFormatterMathtext, FuncFormatter

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
        valid_vals = values[values > 0]
        if len(valid_vals) == 0:
             return np.array([]), np.array([])

        # Robust min/max: Clamp min to 0.01 to avoid extreme scaling from artifacts/noise
        vmin = max(valid_vals.min(), 0.01)
        vmax = valid_vals.max()

        if vmin >= vmax:
             vmax = vmin * 10

        edges = np.logspace(np.log10(vmin), np.log10(vmax), int(bins) + 1)
    else:
        edges = np.linspace(values.min(), values.max(), int(bins) + 1)

    counts, edges = np.histogram(values, bins=edges)
    if log_scale:
        # For log-spaced edges, the geometric mean is the correct "visual center" on a log axis
        # centers = sqrt(edge[i] * edge[i+1])
        centers = np.sqrt(edges[:-1] * edges[1:])
    else:
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
    show_yaxis: bool = False,
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
        if log_scale:
            ax.set_xscale('log')

        x, y = _histogram_curve(df[channel].values, bins=bins, log_scale=log_scale)
        if x.size > 0:
            if norm == "total":
                s = y.sum()
                y = y / s if s > 0 else y
            elif norm == "modal":
                m = y.max()
                y = y / m if m > 0 else y

            y_s = _smooth_1d(y, smooth_sigma)
            # Use ax.fill() with explicit polygon vertices for robust rendering
            # This handles log scale and figure resizing better than fill_between
            poly_x = np.concatenate([[x[0]], x, [x[-1]]])
            poly_y = np.concatenate([[0], y_s, [0]])
            ax.fill(poly_x, poly_y, alpha=0.5, color=c, zorder=1)
            ax.plot(x, y_s, color=c, linewidth=2, zorder=2)
            ax.set_ylim(bottom=0)

        if not show_yaxis:
            ax.set_yticks([])
            ax.set_ylabel("")

        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"Histogram of {channel}")
        return fig, None

    # Ridge Plot Logic
    groups = sorted(df[hue].unique())
    n_groups = len(groups)

    if n_groups == 0:
        return plt.figure(), None

    # Determine columns to return in stats data
    if legend_cols is None:
        legend_cols = [hue, "Mean"]

    # Calculate stats for the table
    # We will return this as a list of dicts suitable for pd.DataFrame construction
    stats_data = []

    for g in groups:
        sub = df[df[hue] == g]
        row_dict = {}

        for col_key in legend_cols:
            if col_key == "Mean":
                vals = sub[channel]
                if log_scale:
                     v = vals[vals > 0]
                     mean_val = np.exp(np.log(v).mean()) if len(v) > 0 else 0
                else:
                    mean_val = vals.mean()
                row_dict["Mean"] = f"{mean_val:.2f}"

            elif col_key == "Median":
                 median_val = sub[channel].median()
                 row_dict["Median"] = f"{median_val:.2f}"

            elif col_key == "Count":
                 row_dict["Count"] = str(len(sub))

            elif col_key == hue:
                row_dict[hue] = str(g)

            elif col_key in df.columns:
                # Try to get unique value
                u_vals = sub[col_key].unique()
                if len(u_vals) == 1:
                    row_dict[col_key] = str(u_vals[0])
                else:
                    row_dict[col_key] = "Mixed"
            else:
                row_dict[col_key] = "-"
        
        # Ensure Hue column is present for mapping/coloring in UI if needed, even if not in legend_cols
        if hue not in row_dict:
            row_dict[hue] = str(g)
            
        stats_data.append(row_dict)

    # Layout
    plt.rcParams.update({'font.size': 10})

    # If y-axis is shown, reduce overlap to prevent tick collisions
    h_space = -0.1 if not show_yaxis else 0.05

    # Responsive height
    fig = plt.figure(figsize=(8, max(4, n_groups * 0.8)))
    gs = gridspec.GridSpec(n_groups, 1, hspace=h_space)

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
        ax = fig.add_subplot(gs[i])
        axes.append(ax)
        ax.set_zorder(n_groups - i)

        color = palette[i]

        x, y = cached_curves[i]
        if x.size > 0:
            # Fill with alpha, outline solid with same color
            ax.fill_between(x, y, color=color, alpha=0.6, zorder=2)
            # Thicker outline (3.0 -> 3.5 to be safe/bold)
            ax.plot(x, y, color=color, linewidth=3.5, zorder=3)

        # Aesthetics
        ax.set_ylabel("")
        if show_yaxis:
            ax.spines['left'].set_visible(True)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='upper'))
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            ax.yaxis.get_offset_text().set_visible(False)
            ax.tick_params(axis='y', which='both', left=True, labelleft=True, labelsize=8)
        else:
            ax.set_yticks([])
            ax.yaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.spines['left'].set_visible(False)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)
        ax.patch.set_alpha(0)

        # X Axis Limits
        ax.set_xlim(x_min, x_max)
        if y_max_global > 0:
            ax.set_ylim(0, y_max_global * 1.3)

        # Log Scale Configuration
        if log_scale:
            ax.set_xscale('log')
            # Explicitly set locators
            # Major ticks: Powers of 10
            ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
            # Minor ticks: 2-9 multiples. Use list explicitly
            ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=[2, 3, 4, 5, 6, 7, 8, 9], numticks=100))
            
            # Formatter: Explicit FuncFormatter for safe x10^z labeling
            def log_10_fmt(x, pos):
                # Avoid log(0)
                if x <= 0: return ""
                log_val = np.log10(x)
                if np.isclose(log_val, np.round(log_val)):
                    return f"$10^{{{int(np.round(log_val))}}}$"
                return ""
            
            ax.xaxis.set_major_formatter(FuncFormatter(log_10_fmt))
            # Force null formatter on minor ticks to ensure no labels appear
            ax.xaxis.set_minor_formatter(NullFormatter())

        # Subplot Specific Visibility
        if i < n_groups - 1:
            ax.set_xlabel("")
            # Hide everything for upper plots
            ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            ax.spines['bottom'].set_visible(False)
            # Faint guide line
            ax.axhline(0, color='gray', linewidth=0.5, alpha=0.3)
        else:
            # Bottom Plot: Show Axis and Ticks
            ax.set_xlabel(channel)
            
            # Major Ticks: Visible, Labeled
            ax.tick_params(axis='x', which='major', bottom=True, direction='out', length=6, width=1.2, colors='black', labelbottom=True, pad=5)
            # Minor Ticks: Visible, Unlabeled. Explicitly force labelbottom=False again
            ax.tick_params(axis='x', which='minor', bottom=True, direction='out', length=3, width=0.8, colors='black', labelbottom=False)
            
            ax.spines['bottom'].set_visible(True)
            ax.spines['bottom'].set_linewidth(1.0)
            ax.spines['bottom'].set_color('black')
            
            # Solid baseline
            ax.axhline(0, color='black', linewidth=1.0)
            
            # Solid baseline
            ax.axhline(0, color='black', linewidth=1.0)
    
    plt.tight_layout()
    return fig, stats_data


def plot_ridgeline_matplotlib(
    experiment,
    channel,
    hue=None,
    scale='log',
    bins=100,
    color=None,
    offset=0.5,
    smooth_sigma: float = 2.0,
    normalization: str = "total",
    show_yaxis: bool = False,
    legend_cols=None,
):
    """
    Generates a Single-Axis Ridgeline Plot (Matplotlib) with adjustable offset.
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
        norm = "total"

    groups = sorted(df[hue].unique()) if hue and hue in df.columns else [None]
    n_groups = len(groups)
    
    # If no hue, treat as single group
    if not hue or hue not in df.columns:
        groups = [None]
        n_groups = 1

    # Determine columns to return in stats data
    if legend_cols is None:
        legend_cols = [hue, "Mean"] if hue else ["Mean"]

    # Calculate stats for the table
    stats_data = []

    for g in groups:
        if g is None:
            sub = df
        else:
            sub = df[df[hue] == g]

        row_dict = {}

        for col_key in legend_cols:
            if col_key == "Mean":
                vals = sub[channel]
                if log_scale:
                     v = vals[vals > 0]
                     mean_val = np.exp(np.log(v).mean()) if len(v) > 0 else 0
                else:
                    mean_val = vals.mean()
                row_dict["Mean"] = f"{mean_val:.2f}"

            elif col_key == "Median":
                 median_val = sub[channel].median()
                 row_dict["Median"] = f"{median_val:.2f}"

            elif col_key == "Count":
                 row_dict["Count"] = str(len(sub))

            elif col_key == hue:
                row_dict[hue] = str(g)

            elif col_key in df.columns:
                u_vals = sub[col_key].unique()
                if len(u_vals) == 1:
                    row_dict[col_key] = str(u_vals[0])
                else:
                    row_dict[col_key] = "Mixed"
            else:
                row_dict[col_key] = "-"

        if hue and hue not in row_dict:
            row_dict[hue] = str(g)

        stats_data.append(row_dict)
    
    # Layout - Single Axis
    plt.rcParams.update({'font.size': 10})
    
    # Height scales with number of groups AND offset to maintain consistent per-group visual height.
    # Total vertical span in units of "one histogram height" is approx 1 + (N-1)*offset.
    # We allocate more inches per unit height + overhead.
    effective_units = 1.0 + max(0, n_groups - 1) * offset
    
    # 2.0 inches base overhead (Axes labels, Tick labels, Title space)
    # 0.8 inches per effective histogram unit height
    # This ensures that if offset increases, total height increases proportionally.
    calc_height = 2.0 + (effective_units * 0.8) 
    
    # Ensure min height of 3 inches
    fig, ax = plt.subplots(figsize=(8, max(3, calc_height)))
    
    # Colors
    if n_groups > 1:
        palette = sns.color_palette("bright", n_groups)
    else:
        palette = [color if color else 'gray']

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

    cached_curves = []
    y_max_global = 0.0

    # 1. Compute Curves and Global Max
    for group in groups:
        if group is not None:
             subset = df[df[hue] == group]
        else:
             subset = df # No hue

        x, y = _histogram_curve(subset[channel].values, bins=bins, log_scale=log_scale)
        if x.size == 0:
            cached_curves.append((np.array([]), np.array([])))
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

    if y_max_global <= 0:
        return fig, []

    # 2. Determine Offsets
    offset_step = y_max_global * offset
    
    # 3. Plot Loops (Single Axis)
    # Iterate Backwards (Top to Bottom) - Painter's Algorithm
    yticks = []
    yticklabels = []

    for i in range(n_groups):
        group = groups[i]
        x, y = cached_curves[i]
        c = palette[i] if n_groups > 1 else palette[0]
        
        # Base Y position
        # First sample at top, last at bottom
        base_y = (n_groups - 1 - i) * offset_step
        yticks.append(base_y)
        label_text = str(group) if group is not None else "All"
        yticklabels.append(label_text)
        
        if x.size > 0:
            # Shift Y
            y_shifted = y + base_y
            
            # zorder: Lower Y (Front) -> Higher Z
            z = (n_groups - i) * 2 
            
            # Fill
            ax.fill_between(x, base_y, y_shifted, color=c, alpha=0.6, zorder=z)
            # Outline
            ax.plot(x, y_shifted, color=c, linewidth=2.0, zorder=z+1)

        # Baseline
        ax.axhline(base_y, color='gray', linewidth=0.5, alpha=0.3, zorder=-1)

    # 4. Axis Configuration
    
    # Y Axis: Show Group Labels
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.tick_params(axis='y', left=False) # Hide ticks
    
    # Clear Spines
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(False)
    
    # X Axis Setup
    ax.set_xlim(x_min, x_max)
    # Limit calculation: (N-1)*offset + 1.2*height
    top_limit = ((n_groups-1) * offset_step) + y_max_global * 1.2
    ax.set_ylim(0, top_limit if top_limit > 0 else 1)

    if log_scale:
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=[2, 3, 4, 5, 6, 7, 8, 9], numticks=100))
        
        def log_10_fmt(x, pos):
            if x <= 0: return ""
            log_val = np.log10(x)
            if np.isclose(log_val, np.round(log_val)):
                return f"$10^{{{int(np.round(log_val))}}}$"
            return ""
        
        ax.xaxis.set_major_formatter(FuncFormatter(log_10_fmt))
        ax.xaxis.set_minor_formatter(NullFormatter())
    
    ax.set_xlabel(channel)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.0)
    
    # Ticks style
    ax.tick_params(axis='x', which='major', bottom=True, direction='out', length=6, width=1.2, colors='black', labelbottom=True, pad=5)
    ax.tick_params(axis='x', which='minor', bottom=True, direction='out', length=3, width=0.8, colors='black', labelbottom=False)
    
    # Add vertical line at left boundary
    ax.axvline(x_min, color='black', linewidth=1.0, zorder=1000)
    
    plt.tight_layout()
    return fig, stats_data

