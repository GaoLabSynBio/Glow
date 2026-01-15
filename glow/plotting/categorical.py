import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import gmean
from .utils import calculate_significance, adjust_pvalues, get_significance_label

def plot_means_with_significance(experiment, channel, group_by, comparisons=[], statistic='mean',
                               hue_order=None, show_distribution=True, replicate_col=None,
                               log_scale=False, annotation_cols=[], correction_method='t-test',
                               use_bars=False, font_size=12, bracket_spacing=0.15,
                               gates=None, gate_channel=None, gate_threshold=None):
    """
    Violin/Strip plot with mean lines, replicate dots, and significance brackets.
    correction_method: 't-test', 'bonferroni', 'fdr'
    gates: List of dicts [{'channel': str, 'threshold': (low, high) or float}]
    """
    df = experiment.data

    # 1. Prepare Data (Gating)
    plot_df = df.copy()

    # Handle multiple gates
    if gates:
        print(f"DEBUG: Applying {len(gates)} gates to Bar Plot")
        for i, gate in enumerate(gates):
            ch = gate.get('channel')
            th = gate.get('threshold')
            direction = gate.get('direction', 'positive') # Default to positive

            if ch in plot_df.columns and th is not None:
                before = len(plot_df)
                if isinstance(th, (tuple, list)) and len(th) == 2:
                    plot_df = plot_df[(plot_df[ch] >= th[0]) & (plot_df[ch] <= th[1])]
                else:
                    if direction == 'negative' or direction == 'Negative (<)':
                        plot_df = plot_df[plot_df[ch] < th]
                    else:
                        plot_df = plot_df[plot_df[ch] > th]
                print(f"DEBUG: Gate {i} ({ch}) reduced points from {before} to {len(plot_df)}")

    # Backward compatibility for single gate
    if gate_channel and gate_threshold is not None:
        if isinstance(gate_threshold, (tuple, list)) and len(gate_threshold) == 2:
            plot_df = plot_df[(plot_df[gate_channel] >= gate_threshold[0]) & (plot_df[gate_channel] <= gate_threshold[1])]
        else:
            plot_df = plot_df[plot_df[gate_channel] > gate_threshold]

    # Combined Filter Check
    if len(plot_df) == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, f"No events remaining\nafter applying gates",
                ha='center', va='center', color='red', fontsize=14)
        ax.set_axis_off()
        return fig

    if group_by not in df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Invalid Group Column", ha='center')
        return fig

    unique_groups = hue_order if hue_order else sorted(df[group_by].unique())
    # Filter groups to only those present in data
    unique_groups = [g for g in unique_groups if g in df[group_by].values]

    # Adjust width based on whether we are showing distributions
    width_per_group = 1.5 if show_distribution else 0.8
    fig, ax = plt.subplots(figsize=(max(5, len(unique_groups)*width_per_group), 6 if show_distribution else 5))

    # If log_scale and data is large, we should log transform the data for plotting
    # so that the KDE (violin shape) is calculated on log-space (Gaussian-like)
    # rather than linear space (skewed).
    # We will then label the Y-axis with 10^x formats.

    transformed_channel = channel
    if log_scale:
        transformed_channel = f"log_{channel}"
        # Filter positive values
        plot_df = plot_df[plot_df[channel] > 0].copy()
        plot_df[transformed_channel] = np.log10(plot_df[channel])
        # We will plot using this transformed column

    # Ensure replicate col is categorical if used for hue
    if replicate_col and replicate_col in plot_df.columns:
        plot_df[replicate_col] = plot_df[replicate_col].astype(str)

    # 2. Distribution Plot (Violin or Strip)
    # Using 'jitter' to spread points
    if show_distribution:
        # Step-like histogram violin (mimic) or smooth violin?
        # User requested "look more like this" (image has stepped look usually, or at least hollow)
        # We use violinplot with 'white' face and 'gray' edge.
        # bw_adjust=1 (default) gives standard smoothing.
        # width=0.8 to separate groups

        # Use hue=replicate_col to split violins by replicate
        # dodge=False to overlap them at the same position
        # Use a Greys palette for replicates so they are distinguishable
        # Use a nice palette for replicates if applicable (e.g. pastel)
        # Avoid Greys which was for the hollow style
        n_reps = len(plot_df[replicate_col].unique()) if replicate_col else 1
        # User requests Hollow Violins with Grey outlines (no fill colors)
        # We use a Greys palette for the outlines
        n_reps = len(plot_df[replicate_col].unique()) if replicate_col else 1
        # Select grey variants.
        violin_palette = sns.color_palette("Greys", n_colors=n_reps + 2)[1:-1]

        v = sns.violinplot(
            data=plot_df, x=group_by, y=transformed_channel if log_scale else channel,
            hue=replicate_col if replicate_col else None,
            order=unique_groups,
            dodge=False, # Overlap violins
            ax=ax,
            palette=violin_palette if replicate_col else None,
            inner=None, linewidth=1.5, saturation=1, cut=0, width=0.8,
            bw_adjust=0.8, legend=False
        )

        # Set facecolor to none to make it hollow
        import matplotlib
        for collection in ax.collections:
            if isinstance(collection, matplotlib.collections.PolyCollection):
                original_face = collection.get_facecolor()
                collection.set_edgecolor(original_face)
                collection.set_facecolor('none')
                collection.set_alpha(1.0) # Opaque lines


    # 3. Replicate Dots & Mean Lines
    # Map groups to x-coordinates
    group_map = {g: i for i, g in enumerate(unique_groups)}
    palette = sns.color_palette("bright", len(unique_groups))

    # 3. Replicate Dots & Mean Lines
    # Map groups to x-coordinates
    group_map = {g: i for i, g in enumerate(unique_groups)}
    palette = sns.color_palette("bright", len(unique_groups))

    # We maintain two types of means:
    # 1. rep_means_linear: List of linear means of each replicate (for stats & FC)
    # 2. plot_mean_val: The value to plot the green line at (Log space if log_scale)

    group_stats_info = {} # Store linear replicate means for each group

    # Process each group to calculate stats
    for i, grp in enumerate(unique_groups):
        sub = plot_df[plot_df[group_by] == grp]
        if len(sub) == 0:
            group_stats_info[grp] = {'linear_reps': np.array([]), 'linear_grand_mean': 0}
            continue

        linear_reps = np.array([])
        linear_grand_mean = 0

        # Calculate Linear Replicate Means (Always Linear for Stats/FC)
        if replicate_col and replicate_col in sub.columns:
            # Group by replicate -> Calculate Linear Mean of each
            linear_rep_means = sub.groupby(replicate_col)[channel].mean()
            # Remove NaNs
            linear_reps = linear_rep_means.dropna().values
            if len(linear_reps) > 0:
                 linear_grand_mean = np.mean(linear_reps)
            else:
                 # No replicates valid? fallback to all events linear mean
                 linear_grand_mean = sub[channel].mean()
                 linear_reps = sub[channel].values # Treat events as replicates for T-test?? No, this breaks N.
                 # If no replicates, we have 1 replicate = mean
                 linear_reps = np.array([linear_grand_mean])
        else:
             # No replicate column
             linear_grand_mean = sub[channel].mean()
             linear_reps = np.array([linear_grand_mean]) # 1 sample

        group_stats_info[grp] = {'linear_reps': linear_reps, 'linear_grand_mean': linear_grand_mean}

        # 3.2 Drawing Dots (Replicates)
        # We need to render the dots at their correct Y position
        # If log_scale is True, we plot Log10(LinearMeanOfReplicate)
        # If log_scale is False, we plot LinearMeanOfReplicate

        if replicate_col and replicate_col in sub.columns:
             # We need a dataframe for stripplot
             # Re-access groupby to keep replicate IDs
             lrm = sub.groupby(replicate_col)[channel].mean()
             dot_data = []
             for rid, val in lrm.items():
                 y_pos = np.log10(val) if log_scale and val > 0 else val
                 dot_data.append({group_by: grp, replicate_col: str(rid), 'y_val': y_pos})

             if dot_data:
                 d_df = pd.DataFrame(dot_data)
                 sns.stripplot(
                     data=d_df, x=group_by, y='y_val', hue=replicate_col,
                     jitter=False, dodge=True, ax=ax, legend=False,
                     palette=['black']*len(d_df), s=5, edgecolor='white', linewidth=0.5
                 )

        # 3.3 Drawing Mean Line (or Bar)
        # We plot the specific Y value corresponding to the "Mean of Means"
        # If log_scale, plot log10(linear_grand_mean)
        plot_mean_val = np.log10(linear_grand_mean) if log_scale and linear_grand_mean > 0 else linear_grand_mean

        x_idx = group_map[grp]

        if use_bars and not show_distribution:
            # Draw Bar
            # Use 'muted' green or user requested style
            bar_color = '#74C476' # Muted Green
            ax.bar(x_idx, plot_mean_val, width=0.6, color=bar_color, edgecolor='black', linewidth=1.5, zorder=0, alpha=0.9) # zorder=0 to be behind dots

            # Optionally keep the green mean line on top? Or remove it?
            pass
        else:
            # Standard Line
            ax.hlines(y=plot_mean_val, xmin=x_idx-0.3, xmax=x_idx+0.3, colors='green', linewidth=2, zorder=15, alpha=0.8)


    # 4. Significance Brackets
    if comparisons:
        # Determine strict Y-limit based on what is visible
        if show_distribution:
            # Full distribution visible, use data max
            if log_scale:
                 max_data = plot_df[transformed_channel].max()
            else:
                 max_data = plot_df[channel].max()
        else:
            # Only dots/means visible. Find max of means.
            # We can use the max of 'group_stats_info' values (plot_mean_val or y_pos of dots)
            all_ymax = []
            for grp, info in group_stats_info.items():
                # Add mean line position
                val = info['linear_grand_mean']
                y_val = np.log10(val) if log_scale and val > 0 else val
                all_ymax.append(y_val)

                # Add dots positions
                for r in info['linear_reps']:
                    y_r = np.log10(r) if log_scale and r > 0 else r
                    all_ymax.append(y_r)

            if all_ymax:
                max_data = max(all_ymax)
            else:
                max_data = 0

        current_y = max_data

        # 1. Collect all raw p-values first for adjustment
        comparison_results = []
        for i, comp_tuple in enumerate(comparisons):
            if len(comp_tuple) == 2:
                g1, g2 = comp_tuple
                custom_label = None
            elif len(comp_tuple) == 3:
                 g1, g2, custom_label = comp_tuple
            else:
                 continue

            if g1 not in group_stats_info or g2 not in group_stats_info: continue
            if g1 == g2: continue

            # Use stored Linear Replicate Means for Statistics
            info1 = group_stats_info[g1]
            info2 = group_stats_info[g2]

            d1 = info1['linear_reps']
            d2 = info2['linear_reps']

            # Calculate Fold Change of Grand Means
            m1 = info1['linear_grand_mean']
            m2 = info2['linear_grand_mean']
            fc = m2 / m1 if m1 != 0 else 0

            # Use custom t-test on Replicate Means
            # If len(d1) < 2, ttest returns NaN or error usually
            p = 1.0
            if len(d1) > 1 and len(d2) > 1:
                try:
                    from scipy import stats
                    _, p = stats.ttest_ind(d1, d2, equal_var=False)
                except:
                    p = 1.0

            comparison_results.append({
                'g1': g1, 'g2': g2, 'p': p, 'fc': fc, 'custom': custom_label,
                'x1': group_map[g1], 'x2': group_map[g2]
            })

        # 2. Adjust p-values
        raw_p = [r['p'] for r in comparison_results]
        adjusted_p = adjust_pvalues(raw_p, method=correction_method)

        # 3. Draw Brackets
        for i, res in enumerate(comparison_results):
            p_adj = adjusted_p[i]
            fc = res['fc']
            sig_label = get_significance_label(p_adj)

            # Use custom label if provided, else construct standard one
            label = res['custom']
            if not label or label == '*':
                 # User Request: Just Fold Change, append stars if significant. No 'ns' text.
                 suffix = sig_label if sig_label != "ns" else ""
                 label = f"{fc:.1f}x{suffix}"

            # Draw
            x1, x2 = res['x1'], res['x2']

            # User Request: "More far away from each other" -> Increase step size
            step_fraction = bracket_spacing

            # Calculate range for step
            if show_distribution:
                 y_min_val = plot_df[transformed_channel].min() if log_scale else plot_df[channel].min()
                 y_range = max_data - y_min_val
            else:
                 # Estimate range from data points visible
                 # If all_ymax was calculated earlier, we can reuse
                 y_range = 1 # fallback
                 if 'all_ymax' in locals() and all_ymax:
                     y_min = min(all_ymax)
                     y_range = max_data - y_min

            step = y_range * step_fraction
            if step == 0: step = 0.5

            current_y += step
            h = step * 0.2

            ax.plot([x1, x1, x2, x2], [current_y, current_y+h, current_y+h, current_y], lw=1, c='black')
            ax.text((x1+x2)/2, current_y + h, label, ha='center', va='bottom', fontsize=font_size)

    # Final Axis Formatting
    ax.set_xlabel("")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=font_size)

    if log_scale:
        from matplotlib.ticker import FuncFormatter
        # We plotted in Log10 space. Y values are 2, 3, 4, 5...
        # We want labels 10^2, 10^3...
        # Or nicely formatted 100, 1000...

        def log_formatter(x, pos):
            return f"$10^{{{int(x)}}}$"

        ax.yaxis.set_major_formatter(FuncFormatter(log_formatter))
        ax.set_ylabel(f"Log10 {channel}", fontsize=font_size)
    else:
        ax.set_ylabel(channel, fontsize=font_size)

    # Update x-axis labels with annotations if requested
    if annotation_cols:
        # Prepare table data: Columns are Groups, Row 1 is Group Name, Subsequent rows are Annotations
        # Row labels
        row_labels = [group_by] + annotation_cols

        # Build cell text (list of lists, rows x cols)
        cell_text = []

        # 1. Group Name Row
        cell_text.append([str(g) for g in unique_groups])

        # 2. Annotation Rows
        for col in annotation_cols:
            row_vals = []
            for grp in unique_groups:
                subset = plot_df[plot_df[group_by] == grp]
                val = ""
                if not subset.empty:
                    # Representative value (first row)
                    val = str(subset.iloc[0][col])
                    # If mixed values exists, maybe indicate? For now assume 1:1 mapping
                row_vals.append(val)
            cell_text.append(row_vals)

        # Create Table
        # Turn off x tick labels
        ax.set_xticklabels([])
        ax.tick_params(axis='x', length=0)

        # Add table at bottom
        the_table = ax.table(cellText=cell_text, rowLabels=row_labels, loc='bottom', cellLoc='center', edges='open')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(font_size)
        the_table.scale(1, 1.5) # Add some padding height

        # Remove table borders (cells)
        for key, cell in the_table.get_celld().items():
            cell.set_linewidth(0)

    # Ensure no grid lines on the plot
    ax.grid(False)

    # Text Overlap Prevention
    # Use layout engine to spacing (only if compatible, otherwise tight_layout with padding)
    try:
        fig.tight_layout()
    except:
        pass

    return fig
