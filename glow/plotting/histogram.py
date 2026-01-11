import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import gridspec

def plot_1d_histogram(experiment, channel, hue=None, scale='log', bins=100, color=None, legend_cols=None, legend_font_size=10, title=None):
    """
    Generates a 1D Ridge Plot (stacked histograms) if hue is provided.
    Includes a statistics table matching the visual style of the reference.
    """
    df = experiment.data
    log_scale = (scale == 'log')

    # If no hue, simple histogram (Individual Plots)
    if not hue or hue not in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        c = color if color is not None else 'gray'
        
        # Use histplot with KDE and step element for "bin wise with curve" style
        sns.histplot(
            data=df, x=channel, log_scale=log_scale, bins=bins,
            kde=True, 
            element="step", 
            fill=True, 
            color=c, 
            alpha=0.5,
            line_kws={'linewidth': 2},
            ax=ax
        )
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
    
    for i, group in enumerate(groups):
        # Plot Axis (Left Column)
        if show_table:
            ax = fig.add_subplot(gs[i, 0])
        else:
            ax = fig.add_subplot(gs[i])
            
        axes.append(ax)
        
        subset = df[df[hue] == group]
        color = palette[i]
        
        # Plot
        # Use KDE for smooth appearance as requested ("like the plot provided")
        sns.kdeplot(
            data=subset, x=channel, 
            log_scale=log_scale, 
            fill=True, 
            color=color, alpha=0.5, linewidth=1.5,
            edgecolor=color,
            ax=ax
        )
        
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
        if i < n_groups - 1:
            ax.set_xlabel("")
            ax.set_xticklabels([])
            ax.spines['bottom'].set_visible(False)
            ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
        else:
            ax.set_xlabel(channel)
            ax.spines['bottom'].set_visible(True)

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

    return fig
