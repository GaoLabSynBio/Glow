"""Canvas 'Export all' ZIP: naming, order, and per-plot files with settings applied."""
import io
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from glow.canvas.functions import build_canvas_zip, safe_filename  # noqa: E402


def _names(zip_bytes):
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        return zf.namelist()


def test_safe_filename():
    assert safe_filename("Pairwise (kde): FITC-A, PE-A") == "Pairwise_kde_FITC-A_PE-A"
    assert safe_filename("///") == "plot"
    assert len(safe_filename("x" * 200)) == 60


def test_build_canvas_zip_orders_and_names_files():
    data = build_canvas_zip(
        [("B plot", {"png": b"1", "svg": b"2"}), ("A plot", {"csv": b"3"})],
        extra_files={"all.pdf": b"4"},
    )
    assert _names(data) == ["01_B_plot.png", "01_B_plot.svg", "02_A_plot.csv", "all.pdf"]


def test_build_canvas_zip_same_titles_do_not_collide():
    data = build_canvas_zip([("Same", {"png": b"1"}), ("Same", {"png": b"2"})])
    assert _names(data) == ["01_Same.png", "02_Same.png"]


def test_export_all_zip_from_canvas_entries():
    from canvas.view import _export_all_zip

    fig, ax = plt.subplots()
    ax.scatter([1, 2, 3], [3, 1, 2], label="cells")
    entry = {"type": "matplotlib", "title": "Scatter", "fig": fig,
             "settings": {"width_scale": 2.0, "dot_scale": 2.0}}
    names = _names(_export_all_zip([entry]))
    assert names == ["01_Scatter.png", "01_Scatter.svg", "01_Scatter.pdf", "canvas_all_plots.pdf"]
    # Export works on a copy: the canvas figure itself is not resized by exporting.
    assert not hasattr(fig, "_glow_orig_size")
    plt.close(fig)
