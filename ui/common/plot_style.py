"""FlowJo-like styling for the 2-D gating scatter plots.

FlowJo's dot plots have a solid black frame around a square plot area, no grid lines,
short outward tick marks with minor ticks between the decade labels, and a white
background. ``flowjo_layout`` applies that to a Plotly figure; ``flowjo_axes``
applies the same look to a matplotlib ``Axes`` used for the canvas/SVG export.

The square plot area is achieved by fixing the figure's pixel size (plot area ``size``
plus fixed, non-auto-expanding margins) and rendering with ``st.plotly_chart(fig,
width="content")`` so Streamlit does not stretch it to the column width.
"""
from __future__ import annotations

# Plot-area size in pixels for the main gating plots and for the smaller side-by-side
# density previews.
MAIN_SIZE = 560
PREVIEW_SIZE = 380

# Fixed margins: the right margin holds the colour bar, the top margin the title and
# the horizontal legend. ``autoexpand=False`` keeps Plotly from growing them, which
# would otherwise shrink the plot area and break the 1:1 box.
_MARGIN = dict(l=70, r=110, t=80, b=60, autoexpand=False)

_TICK_COLOR = "black"


def _axis_style(log: bool | None) -> dict:
    style = dict(
        showgrid=False,
        zeroline=False,
        showline=True,
        mirror=True,            # draw the line on both sides → closed box
        linecolor=_TICK_COLOR,
        linewidth=1.2,
        ticks="outside",
        ticklen=4,
        tickwidth=1,
        tickcolor=_TICK_COLOR,
        minor=dict(ticks="outside", ticklen=2, tickwidth=0.8, tickcolor=_TICK_COLOR, showgrid=False),
        color=_TICK_COLOR,
        tickfont=dict(color=_TICK_COLOR),
        title_font=dict(color=_TICK_COLOR),
    )
    if log is not None:
        style["type"] = "log" if log else "linear"
    return style


def _decade_ticks(fig):
    """Log axes: label decades only (10², 10³ …) and leave the 2–9 minor ticks unlabelled,
    as FlowJo does, instead of Plotly's default "2 3 4 5 …" minor labels."""
    for axis in (fig.layout.xaxis, fig.layout.yaxis):
        if axis.type == "log":
            axis.update(dtick=1, exponentformat="power", showexponent="all",
                        minor=dict(dtick="D1", ticks="outside", showgrid=False))


def flowjo_layout(fig, *, x_log=None, y_log=None, size=MAIN_SIZE, title=None,
                  x_title=None, y_title=None, legend_top=True):
    """Apply the FlowJo look to a Plotly figure in place and return it.

    ``x_log``/``y_log``: True/False to set the axis type, None to keep what the figure
    already has (e.g. set by ``px.scatter(log_x=...)``).
    ``size``: pixel size of the square plot area.
    """
    fig.update_xaxes(**_axis_style(x_log))
    fig.update_yaxes(**_axis_style(y_log))
    _decade_ticks(fig)

    layout = dict(
        width=size + _MARGIN["l"] + _MARGIN["r"],
        height=size + _MARGIN["t"] + _MARGIN["b"],
        autosize=False,
        margin=_MARGIN,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color=_TICK_COLOR),
    )
    if title is not None:
        layout["title"] = dict(text=title, x=0.0, xanchor="left", y=0.98, yanchor="top",
                               font=dict(color=_TICK_COLOR, size=15))
    if x_title is not None:
        layout["xaxis_title"] = x_title
    if y_title is not None:
        layout["yaxis_title"] = y_title
    if legend_top:
        layout["legend"] = dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                                font=dict(color=_TICK_COLOR),
                                title=dict(font=dict(color=_TICK_COLOR)))
    else:
        layout["legend"] = dict(font=dict(color=_TICK_COLOR), title=dict(font=dict(color=_TICK_COLOR)))
    fig.update_layout(**layout)

    # Colour bar (if any) lives in the right margin, scaled to the plot height.
    fig.update_layout(coloraxis_colorbar=dict(len=1.0, thickness=14, x=1.02, xanchor="left",
                                              tickfont=dict(color=_TICK_COLOR),
                                              title=dict(font=dict(color=_TICK_COLOR))))
    for tr in fig.data:
        marker = getattr(tr, "marker", None)
        if marker is not None and getattr(marker, "colorbar", None) is not None and marker.showscale:
            marker.colorbar.update(len=1.0, thickness=14, x=1.02, xanchor="left")
    return fig


HIST_WIDTH = 700        # plot-area width of the Analysis gate-creator histogram
HIST_ASPECT = 2.5       # width : height
HIST_PREVIEW_WIDTH = 450  # per-gate mini histograms in the gate list
_HIST_MARGIN = dict(l=65, r=25, t=75, b=55, autoexpand=False)


def flowjo_hist_layout(fig, *, width=HIST_WIDTH, aspect=HIST_ASPECT, height=None,
                       x_log=None, title=None, x_title=None, y_title=None, legend_top=True):
    """FlowJo-like 1-D histogram / density layout: a wide box (``aspect`` : 1), solid frame,
    no grid, minor ticks, labels only at major ticks. ``height`` overrides the aspect ratio
    (used by the ridgeline mode, whose height grows with the number of samples)."""
    plot_h = int(height) if height else int(round(width / aspect))
    fig.update_xaxes(**_axis_style(x_log))
    fig.update_yaxes(**_axis_style(None))
    fig.update_yaxes(minor=dict(ticks=""))   # density / sample axis: major ticks only
    _decade_ticks(fig)
    layout = dict(
        width=width + _HIST_MARGIN["l"] + _HIST_MARGIN["r"],
        height=plot_h + _HIST_MARGIN["t"] + _HIST_MARGIN["b"],
        autosize=False,
        margin=_HIST_MARGIN,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color=_TICK_COLOR),
    )
    if title is not None:
        layout["title"] = dict(text=title, x=0.0, xanchor="left", y=0.98, yanchor="top",
                               font=dict(color=_TICK_COLOR, size=15))
    if x_title is not None:
        layout["xaxis_title"] = x_title
    if y_title is not None:
        layout["yaxis_title"] = y_title
    if legend_top:
        layout["legend"] = dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                                font=dict(color=_TICK_COLOR), title=dict(font=dict(color=_TICK_COLOR)))
    fig.update_layout(**layout)
    return fig


def add_threshold_line(fig, x, *, log=False, **vline_kwargs):
    """``fig.add_vline`` that keeps its annotation on the line when the x axis is log.

    Plotly draws the line shape in data units but places the annotation in axis units,
    which for a log axis means log10(value); without this fix the label lands at 10**x.
    """
    import math
    fig.add_vline(x=x, **vline_kwargs)
    if log and vline_kwargs.get("annotation_text") and x > 0 and fig.layout.annotations:
        fig.layout.annotations[-1].x = math.log10(x)
    return fig


def flowjo_hist_axes(ax, *, log=False, aspect=HIST_ASPECT, title=None):
    """matplotlib counterpart of ``flowjo_hist_layout`` (box aspect 1 : ``aspect``)."""
    flowjo_axes(ax, log=False, title=title)
    if log:
        ax.set_xscale("log")
    ax.tick_params(axis="y", which="minor", left=False)   # major ticks only on the y axis
    ax.set_box_aspect(1.0 / aspect)
    return ax


def flowjo_axes(ax, *, log=False, title=None):
    """Apply the FlowJo look to a matplotlib Axes (square box, frame, no grid, minor ticks)."""
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(_TICK_COLOR)
        spine.set_linewidth(1.2)
    ax.minorticks_on()
    ax.tick_params(which="major", direction="out", length=4, width=1, colors=_TICK_COLOR,
                   top=False, right=False)
    ax.tick_params(which="minor", direction="out", length=2, width=0.8, colors=_TICK_COLOR,
                   top=False, right=False)
    ax.set_box_aspect(1)
    if title is not None:
        ax.set_title(title, loc="left")
    return ax
