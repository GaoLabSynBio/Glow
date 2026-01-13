import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde

def plot_1d_histogram(experiment, channel, hue=None, scale='log', bins=100, color=None, legend_cols=None, legend_font_size=10, title=None):
    """
    Generates a 1D Ridge Plot (stacked histograms) using Plotly.
    Each group is separated into its own subplot (row).
    """
    df = experiment.data
    log_scale = (scale == 'log')

    # Define groups
    if hue and hue in df.columns:
        groups = sorted(df[hue].unique())
    else:
        # Single group wrapper - Use a simple Plotly Figure (No Subplots)
        groups = [channel]
        n_groups = 1
        
        fig = go.Figure()
        
        data_vals = df[channel].values
        if log_scale:
            data_for_kde = data_vals[data_vals > 0]
            if len(data_for_kde) < 2: 
                return fig # Empty
            data_log = np.log10(data_for_kde)
            x_grid = np.linspace(data_log.min(), data_log.max(), bins)
            x_plot = 10**x_grid
            eval_grid = x_grid
        else:
            x_plot = np.linspace(data_vals.min(), data_vals.max(), bins)
            data_log = data_vals
            eval_grid = x_plot

        kde = gaussian_kde(data_log)
        density = kde(eval_grid)
        
        # Normalize
        density = density / density.max() if density.max() > 0 else density

        c = color if color else 'rgb(31, 119, 180)'
        # Fix for tuple color from seaborn palette
        if isinstance(c, tuple):
            if len(c) == 3:
                c = f"rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})"
            elif len(c) == 4:
                c = f"rgba({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)}, {c[3]})"

        fig.add_trace(go.Scatter(
            x=x_plot, y=density, mode='lines', 
            fill='tozeroy', name=channel,
            line=dict(color=c, width=2)
        ))
        
        fig.update_layout(
            title=title if title else f"Histogram of {channel}",
            xaxis_title=channel + (" (Log)" if log_scale else ""),
            yaxis_title="Relative Density",
            showlegend=True,
            plot_bgcolor='white', paper_bgcolor='white',
            xaxis=dict(type="log" if log_scale else "linear", showgrid=True, gridcolor='lightgrey', color='black'),
            yaxis=dict(showticklabels=False, showgrid=False),
            font=dict(color='black'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(color='black'),
                bgcolor="rgba(255,255,255,0.6)"
            )
        )
        return fig

    # --- Merged / Ridge Plot (Multi-group) ---
    n_groups = len(groups)
    
    # Create Subplots
    fig = make_subplots(
        rows=n_groups, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.01, # Tight stacking
        print_grid=False
    )

    # Color Palette - Use a distinct set of colors
    colors = px.colors.qualitative.Plotly
    if len(groups) > len(colors):
        colors = colors * (len(groups) // len(colors) + 1)

    # Determine Global Range for X-axis comparison across all subplots
    if log_scale:
        valid_data = df[df[channel] > 0][channel]
        if len(valid_data) > 0:
            x_min_log = np.log10(valid_data.min())
            x_max_log = np.log10(valid_data.max())
        else:
            x_min_log, x_max_log = 0, 4
        x_grid = np.linspace(x_min_log, x_max_log, bins)
        x_plot = 10**x_grid
    else:
        x_min = df[channel].min()
        x_max = df[channel].max()
        x_plot = np.linspace(x_min, x_max, bins)

    for i, g in enumerate(groups):
        subset = df[df[hue] == g]
        label = str(g)
        c = colors[i]

        if len(subset) < 2:
            # Add dummy trace to maintain row index if empty
            fig.add_trace(go.Scatter(x=[x_plot[0]], y=[0], showlegend=False), row=i+1, col=1)
            continue

        data_vals = subset[channel].values
        
        try:
            if log_scale:
                data_for_kde = data_vals[data_vals > 0]
                if len(data_for_kde) < 2: continue
                data_for_kde = np.log10(data_for_kde)
                eval_grid = x_grid
            else:
                data_for_kde = data_vals
                eval_grid = x_plot

            kde = gaussian_kde(data_for_kde)
            density = kde(eval_grid)
            
            # Normalize density for consistent visual height (0 to 1) in ridge plot
            if density.max() > 0:
                density = density / density.max()
            
            row_idx = i + 1
            
            # Use rgba for transparency in the filled area
            # Plotly handles hex->rgba or rgb->rgba if we specify opacity in fillcolor
            
            # Fix for tuple color from seaborn palette
            if isinstance(c, tuple):
                if len(c) == 3:
                    c = f"rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})"
                elif len(c) == 4:
                    c = f"rgba({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)}, {c[3]})"
            
            # Use rgba for transparency in the filled area
            fill_color = c
            if isinstance(c, str):
                if c.startswith('#'):
                    r = int(c[1:3], 16)
                    g_val = int(c[3:5], 16)
                    b = int(c[5:7], 16)
                    fill_color = f"rgba({r}, {g_val}, {b}, 0.5)"
                elif c.startswith('rgb('):
                    fill_color = c.replace('rgb(', 'rgba(').replace(')', ', 0.5)')

            fig.add_trace(go.Scatter(
                x=x_plot, 
                y=density, 
                mode='lines',
                name=label,
                fill='tozeroy',
                line=dict(width=2, color=c),
                fillcolor=fill_color,
                showlegend=True
            ), row=row_idx, col=1)
            
            # Performance/Aesthetics: Hide Y axis for this subplot
            fig.update_yaxes(
                showticklabels=False, 
                showgrid=False, 
                zeroline=False, 
                range=[0, 1.1], # Keep normalized height consistent
                row=row_idx, col=1
            )
            
            # Set X-axis type for this subplot
            fig.update_xaxes(
                type="log" if log_scale else "linear",
                showgrid=True,
                gridcolor='rgba(200,200,200,0.2)',
                row=row_idx, col=1
            )

        except Exception as e:
            print(f"KDE failed for {label}: {e}")

    # Final Layout Cleanup
    fig.update_layout(
        title=title if title else f"Density Plot of {channel}",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color='black', size=12),
            title=dict(text=f"{hue}: " if hue else "", font=dict(color='black')),
            bgcolor="rgba(255,255,255,0.6)"
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'),
        height=max(400, n_groups * 100),
        margin=dict(t=100, b=60, l=60, r=60) # Balanced margins
    )
    
    # Hide labels for all except bottom X axis
    if n_groups > 1:
        for i in range(1, n_groups):
            fig.update_xaxes(showticklabels=False, title_text="", row=i, col=1)
        # Label bottom
        fig.update_xaxes(title_text=channel + (" (Log)" if log_scale else ""), row=n_groups, col=1, color='black')

    return fig

