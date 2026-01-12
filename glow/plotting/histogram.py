import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde

def plot_1d_histogram(experiment, channel, hue=None, scale='log', bins=100, color=None, legend_cols=None, legend_font_size=10, title=None):
    """
    Generates a 1D Density Plot (Histogram-like) using Plotly.
    Supports hue grouping and log/linear scales.
    
    Args:
        experiment: Cytoflow experiment
        channel: Channel name
        hue: Grouping column
        scale: 'log' or 'linear'
        bins: Number of points for KDE evaluation
        color: Specific color (if no hue)
        legend_cols: Ignored in Plotly version (interactive legend used instead)
        legend_font_size: Ignored
        title: Plot title
    """
    df = experiment.data
    log_scale = (scale == 'log')

    fig = go.Figure()

    # Define groups
    if hue and hue in df.columns:
        groups = sorted(df[hue].unique())
    else:
        # Single group wrapper
        groups = ["All Events"]
        # Use a dummy column if needed or just handle logic below
        
    # Color Palette
    if hue:
        colors = px.colors.qualitative.Plotly * (len(groups) // 10 + 1)
    else:
        colors = [color if color else 'rgb(31, 119, 180)']

    # Determine Range for Grid
    if log_scale:
        valid_data = df[df[channel] > 0][channel]
        if len(valid_data) > 0:
            x_min = np.log10(valid_data.min())
            x_max = np.log10(valid_data.max())
        else:
            x_min, x_max = -1, 4 # Fallback
    else:
        x_min = df[channel].min()
        x_max = df[channel].max()

    # Evaluation Grid
    x_grid = np.linspace(x_min, x_max, bins)
    x_plot = 10**x_grid if log_scale else x_grid

    for i, g in enumerate(groups):
        if hue:
            subset = df[df[hue] == g]
            label = str(g)
            c = colors[i]
        else:
            subset = df
            label = channel
            c = colors[0]

        if len(subset) < 2:
            continue

        data = subset[channel].values
        
        try:
            if log_scale:
                # Filter positive and log transform for KDE
                data = data[data > 0]
                if len(data) < 2: continue
                data_kde = np.log10(data)
            else:
                data_kde = data

            kde = gaussian_kde(data_kde)
            density = kde(x_grid if log_scale else x_plot)
            
            # Normalize density to count-like or max 1 for visualization? 
            # Seaborn 'density' is area=1. Let's stick to density.
            
            fig.add_trace(go.Scatter(
                x=x_plot, 
                y=density, 
                mode='lines',
                name=label,
                fill='tozeroy',
                line=dict(width=2, color=c),
                # fillcolor=c # Transparent fill usually handled by defaults or opacity
            ))
            
        except Exception as e:
            # Singular matrix or other KDE error
            print(f"KDE failed for {label}: {e}")

    # Layout
    fig.update_layout(
        title=title if title else f"Density Plot of {channel}",
        xaxis_title=channel + (" (Log)" if log_scale else ""),
        yaxis_title="Density",
        xaxis=dict(
            type="log" if log_scale else "linear",
            showgrid=True,
            gridcolor='lightgrey'
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor='lightgrey'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'),
        legend=dict(title=hue if hue else None)
    )

    return fig

