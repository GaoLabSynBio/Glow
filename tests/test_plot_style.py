import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from ui.common.plot_style import (
    MAIN_SIZE, HIST_WIDTH, HIST_ASPECT,
    flowjo_layout, flowjo_hist_layout, flowjo_axes, flowjo_hist_axes, add_threshold_line,
)


def _df(n=200):
    rng = np.random.default_rng(0)
    return pd.DataFrame({"x": rng.lognormal(7, 0.5, n), "y": rng.lognormal(7, 0.5, n), "d": rng.random(n)})


def _plot_area(layout):
    m = layout.margin
    return layout.width - m.l - m.r, layout.height - m.t - m.b


def test_flowjo_layout_is_square_box_without_grid():
    fig = px.scatter(_df(), x="x", y="y", color="d", log_x=True, log_y=True)
    flowjo_layout(fig, size=MAIN_SIZE, title="t")
    w, h = _plot_area(fig.layout)
    assert w == h == MAIN_SIZE
    assert fig.layout.margin.autoexpand is False
    for ax in (fig.layout.xaxis, fig.layout.yaxis):
        assert ax.showgrid is False and ax.showline and ax.mirror is True
        assert ax.ticks == "outside" and ax.minor.ticks == "outside"
        assert ax.type == "log" and ax.dtick == 1 and ax.exponentformat == "power"
        assert ax.minor.dtick == "D1"


def test_flowjo_layout_keeps_existing_axis_type_and_sets_requested_one():
    fig = go.Figure(go.Scatter(x=[1, 2], y=[1, 2]))
    fig.update_xaxes(type="log")
    flowjo_layout(fig)
    assert fig.layout.xaxis.type == "log" and fig.layout.yaxis.type is None
    flowjo_layout(fig, x_log=False, y_log=True)
    assert fig.layout.xaxis.type == "linear" and fig.layout.yaxis.type == "log"


def test_hist_layout_aspect_ratio_and_height_override():
    fig = go.Figure(go.Scatter(x=[1, 10, 100], y=[0, 1, 0]))
    flowjo_hist_layout(fig, width=HIST_WIDTH, x_log=True)
    w, h = _plot_area(fig.layout)
    assert w == HIST_WIDTH and abs(w / h - HIST_ASPECT) < 0.02
    assert fig.layout.xaxis.showgrid is False and fig.layout.yaxis.showgrid is False
    assert fig.layout.xaxis.type == "log" and fig.layout.xaxis.dtick == 1
    flowjo_hist_layout(fig, width=HIST_WIDTH, height=900)
    assert _plot_area(fig.layout)[1] == 900


def test_add_threshold_line_moves_annotation_to_log10_on_log_axis():
    fig = go.Figure(go.Scatter(x=[1, 10, 100], y=[0, 1, 0]))
    add_threshold_line(fig, 150.0, log=True, line_dash="dash", annotation_text="thr")
    assert fig.layout.shapes[-1].x0 == 150.0          # line is drawn in data units
    assert math.isclose(fig.layout.annotations[-1].x, math.log10(150.0))
    fig2 = go.Figure(go.Scatter(x=[1, 10, 100], y=[0, 1, 0]))
    add_threshold_line(fig2, 150.0, log=False, annotation_text="thr")
    assert fig2.layout.annotations[-1].x == 150.0     # linear axis untouched
    fig3 = go.Figure(go.Scatter(x=[1, 10, 100], y=[0, 1, 0]))
    add_threshold_line(fig3, 150.0, log=True)        # no annotation → nothing to fix
    assert len(fig3.layout.annotations) == 0


def test_matplotlib_helpers():
    fig, ax = plt.subplots()
    ax.scatter([1, 10], [1, 10])
    flowjo_axes(ax, log=True, title="t")
    assert ax.get_xscale() == "log" and ax.get_box_aspect() == 1
    assert all(s.get_visible() for s in ax.spines.values())
    assert not any(l.get_visible() for l in ax.get_xgridlines())
    fig2, ax2 = plt.subplots()
    ax2.plot([1, 10, 100], [0, 1, 0])
    flowjo_hist_axes(ax2, log=True)
    assert ax2.get_xscale() == "log" and ax2.get_yscale() == "linear"
    assert math.isclose(ax2.get_box_aspect(), 1 / HIST_ASPECT)
    plt.close("all")
