import streamlit as st
import io
import json
import pickle
import threading
import numpy as np
import warnings
from glow.canvas.functions import (
    get_canvas_plots, remove_plot_from_canvas, move_plot_in_canvas,
    build_canvas_zip, safe_filename,
)
import matplotlib.text as mpl_text
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.collections import PolyCollection, PathCollection
import matplotlib.colors as mcolors


def _color_to_hex(color):
    """Convert any matplotlib color to hex string."""
    try:
        if isinstance(color, str):
            if color.startswith('#'):
                return color
            return mcolors.to_hex(color)
        elif isinstance(color, (tuple, list, np.ndarray)):
            # RGBA or RGB tuple
            if len(color) >= 3:
                return mcolors.to_hex(color[:3])
        return "#808080"  # Fallback gray
    except Exception:
        return "#808080"


def _has_density_scatter(fig):
    """
    Check if a figure has density-colored scatter points (PathCollections with array data).
    """
    for ax in fig.axes:
        for coll in ax.collections:
            if isinstance(coll, PathCollection):
                arr = coll.get_array()
                if arr is not None and len(arr) > 1:
                    return True
    return False


def _extract_categorical_scatter(fig):
    """
    Extract categorical scatter collections from a matplotlib figure.
    
    These are PathCollections with:
    - No array data (single color per collection)
    - A label (from legend)
    
    Returns list of dicts: [{"name": str, "original_color": color, "collection": PathCollection}]
    """
    categories = []
    
    for ax in fig.axes:
        # Get legend handles and labels if available
        legend = ax.get_legend()
        legend_map = {}
        if legend:
            try:
                handles = legend.legend_handles
                texts = legend.get_texts()
                for handle, text in zip(handles, texts):
                    legend_map[id(handle)] = text.get_text()
            except AttributeError:
                try:
                    handles = legend.legendHandles
                    texts = legend.get_texts()
                    for handle, text in zip(handles, texts):
                        legend_map[id(handle)] = text.get_text()
                except Exception:
                    pass
        
        for coll in ax.collections:
            if isinstance(coll, PathCollection):
                arr = coll.get_array()
                # Skip density plots (have array data)
                if arr is not None and len(arr) > 1:
                    continue
                
                # Get label - try legend first, then label property
                label = legend_map.get(id(coll), None)
                if not label:
                    label = coll.get_label()
                
                # Skip if no meaningful label or internal matplotlib label
                if not label or label.startswith('_'):
                    continue
                
                # Get the color
                facecolors = coll.get_facecolor()
                if len(facecolors) > 0:
                    color = facecolors[0]  # First color (should be uniform for categorical)
                    
                    categories.append({
                        "name": label,
                        "original_color": color,
                        "collection": coll
                    })
    
    return categories


def _has_categorical_scatter(fig):
    """Check if a figure has categorical scatter plots (multiple PathCollections with labels)."""
    return len(_extract_categorical_scatter(fig)) > 0


def _apply_mpl_category_colors(fig, category_colors):
    """
    Apply custom colors to categorical scatter collections in a matplotlib figure.
    Also updates the legend to match.
    
    Args:
        fig: matplotlib Figure
        category_colors: dict mapping category name to hex color
    """
    if not category_colors:
        return
    
    categories = _extract_categorical_scatter(fig)
    
    for cat_info in categories:
        cat_name = cat_info["name"]
        if cat_name in category_colors:
            new_color = category_colors[cat_name]
            collection = cat_info["collection"]
            try:
                # Set the facecolor for the scatter collection
                collection.set_facecolor(new_color)
                # Also update edgecolor if it was originally the same
                collection.set_edgecolor(new_color)
            except Exception:
                pass
    
    # Update legend colors to match
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend:
            try:
                # Get legend handles and texts
                try:
                    handles = legend.legend_handles
                except AttributeError:
                    handles = legend.legendHandles
                texts = legend.get_texts()
                
                for handle, text in zip(handles, texts):
                    label = text.get_text()
                    if label in category_colors:
                        new_color = category_colors[label]
                        # Update the legend handle color
                        if hasattr(handle, 'set_facecolor'):
                            handle.set_facecolor(new_color)
                        if hasattr(handle, 'set_edgecolor'):
                            handle.set_edgecolor(new_color)
                        if hasattr(handle, 'set_color'):
                            handle.set_color(new_color)
                        # For PathCollection legend handles
                        if hasattr(handle, 'set_facecolors'):
                            handle.set_facecolors(new_color)
                        if hasattr(handle, 'set_edgecolors'):
                            handle.set_edgecolors(new_color)
            except Exception:
                pass


def _is_closed_polygon_line(line):
    """
    Check if a Line2D is a closed polygon (gate boundary, not a trend line).
    Closed polygons have their first and last points at the same location.
    """
    xdata = line.get_xdata()
    ydata = line.get_ydata()
    if len(xdata) < 3:
        return False
    # Check if first and last points are the same (closed polygon)
    if np.allclose([xdata[0], ydata[0]], [xdata[-1], ydata[-1]], rtol=0.01):
        return True
    return False


def _has_colorable_lines(fig):
    """
    Check if a figure has Line2D objects that can be colored (not axis/grid lines or gate boundaries).
    """
    for ax in fig.axes:
        lines = [obj for obj in ax.get_children() if isinstance(obj, Line2D)]
        for line in lines:
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            if len(xdata) > 2 and len(ydata) > 2:
                # Skip horizontal/vertical lines (axis lines)
                if np.allclose(ydata, ydata[0]) or np.allclose(xdata, xdata[0]):
                    continue
                # Skip closed polygons (gate boundaries)
                if _is_closed_polygon_line(line):
                    continue
                return True
    return False


def _is_pure_density_plot(fig):
    """
    Check if a figure is ONLY a density plot (has density scatter but NO colorable lines).
    These should only have colormap picker, not line color picker.
    """
    return _has_density_scatter(fig) and not _has_colorable_lines(fig)


def _extract_colorable_samples(fig, lines_only=False):
    """
    Extract samples that can have their colors changed from a matplotlib figure.
    
    Returns list of dicts: [{"name": str, "original_color": color, "artists": [artist, ...]}]
    
    Handles:
    - Histogram ridge plots: Line2D + corresponding PolyCollection fill
    - Bar plots: Rectangle patches grouped by x-position/label
    - Facet plots with running mean lines: Line2D objects (when lines_only=True)
    
    Args:
        fig: matplotlib Figure
        lines_only: If True, only extract Line2D objects (for facet plots with density scatter)
    
    Does NOT handle:
    - Pure density scatter plots (PathCollection with array data, no lines)
    """
    # Skip pure density plots (no lines to color)
    if _is_pure_density_plot(fig):
        return []
    
    samples = []
    
    for ax in fig.axes:
        # ============================================================
        # Method 1: Extract from Line2D objects (histogram curves)
        # ============================================================
        lines = [obj for obj in ax.get_children() if isinstance(obj, Line2D)]
        # Filter out axis lines, grid lines, gate boundaries, etc.
        data_lines = []
        for line in lines:
            # Skip lines that are likely axis/grid lines
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            if len(xdata) > 2 and len(ydata) > 2:  # Data lines have many points
                # Skip horizontal/vertical lines (axis lines)
                if np.allclose(ydata, ydata[0]) or np.allclose(xdata, xdata[0]):
                    continue
                # Skip closed polygons (gate boundaries)
                if _is_closed_polygon_line(line):
                    continue
                data_lines.append(line)
        
        # For histogram ridge plots, each line has a corresponding fill
        # Group lines with their fills by matching colors
        # For facet plots (lines_only=True), skip fill matching
        poly_collections = []
        if not lines_only:
            poly_collections = [obj for obj in ax.collections if isinstance(obj, PolyCollection)]
        
        for idx, line in enumerate(data_lines):
            line_color = line.get_color()
            
            # Try to get a label for this sample
            label = line.get_label()
            if not label or label.startswith('_'):
                # Try to get label from legend
                legend = ax.get_legend()
                if legend:
                    try:
                        for handle, text in zip(legend.legend_handles, legend.get_texts()):
                            if handle == line:
                                label = text.get_text()
                                break
                    except AttributeError:
                        # Older matplotlib versions use legendHandles
                        try:
                            for handle, text in zip(legend.legendHandles, legend.get_texts()):
                                if handle == line:
                                    label = text.get_text()
                                    break
                        except Exception:
                            pass
                
                # If still no label, use index or axis title
                if not label or label.startswith('_'):
                    # Check if this is a ridge plot (multiple axes with one line each)
                    if len(data_lines) == 1 and len(fig.axes) > 1:
                        # This is likely a ridge plot subplot
                        ax_idx = list(fig.axes).index(ax)
                        label = f"Sample {ax_idx + 1}"
                    else:
                        label = f"Line {idx + 1}"
            
            # Find matching fill (PolyCollection with same/similar color)
            # Skip for lines_only mode (facet plots)
            matching_fills = []
            if not lines_only:
                for poly in poly_collections:
                    try:
                        face_colors = poly.get_facecolor()
                        if len(face_colors) > 0:
                            poly_color = face_colors[0][:3]  # RGB
                            line_rgb = mcolors.to_rgb(line_color)
                            # Check if colors match (with tolerance for alpha differences)
                            if np.allclose(poly_color, line_rgb, atol=0.1):
                                matching_fills.append(poly)
                    except Exception:
                        pass
            
            # Check if this sample already exists (same label)
            existing = next((s for s in samples if s["name"] == label), None)
            if existing:
                existing["artists"].append(line)
                existing["artists"].extend(matching_fills)
            else:
                samples.append({
                    "name": label,
                    "original_color": line_color,
                    "artists": [line] + matching_fills
                })
        
        # ============================================================
        # Method 2: Extract from Rectangle patches (bar plots)
        # ============================================================
        rectangles = [obj for obj in ax.get_children() if isinstance(obj, Rectangle)]
        # Filter out axis background rectangle
        data_rects = [r for r in rectangles if r.get_width() > 0 and r.get_height() > 0 
                      and r.get_facecolor()[3] > 0]  # Has visible fill
        
        if data_rects and not data_lines:  # Only process if no lines (avoid double-processing)
            # Group rectangles by their x-position (bars at same x are same group)
            # Or by color if they're grouped that way
            x_positions = {}
            for rect in data_rects:
                x_center = rect.get_x() + rect.get_width() / 2
                # Round to group nearby bars
                x_key = round(x_center, 1)
                if x_key not in x_positions:
                    x_positions[x_key] = []
                x_positions[x_key].append(rect)
            
            # Try to get labels from x-axis tick labels
            try:
                tick_labels = [t.get_text() for t in ax.get_xticklabels()]
                tick_positions = ax.get_xticks()
            except Exception:
                tick_labels = []
                tick_positions = []
            
            for idx, (x_key, rects) in enumerate(sorted(x_positions.items())):
                # Find closest tick label
                label = f"Group {idx + 1}"
                if tick_labels and tick_positions is not None and len(tick_positions) > 0:
                    try:
                        closest_idx = np.argmin(np.abs(np.array(tick_positions) - x_key))
                        if closest_idx < len(tick_labels) and tick_labels[closest_idx]:
                            label = tick_labels[closest_idx]
                    except Exception:
                        pass
                
                # Get color from first rectangle
                rect_color = rects[0].get_facecolor()
                
                samples.append({
                    "name": label,
                    "original_color": rect_color,
                    "artists": rects
                })
    
    return samples


def _apply_sample_colors(fig, sample_colors):
    """
    Apply custom colors to samples in a figure.
    
    Args:
        fig: matplotlib Figure
        sample_colors: dict mapping sample name to hex color
    """
    if not sample_colors:
        return
    
    # Detect plot type to use appropriate extraction mode
    has_density = _has_density_scatter(fig)
    has_lines = _has_colorable_lines(fig)
    
    if has_density and has_lines:
        # Facet plot: only color lines
        colorable_samples = _extract_colorable_samples(fig, lines_only=True)
    elif not has_density:
        # Pure histogram or bar plot
        colorable_samples = _extract_colorable_samples(fig, lines_only=False)
    else:
        # Pure density plot - no sample colors to apply
        return
    
    for sample_info in colorable_samples:
        sample_name = sample_info["name"]
        if sample_name in sample_colors:
            new_color = sample_colors[sample_name]
            
            for artist in sample_info["artists"]:
                try:
                    if isinstance(artist, Line2D):
                        artist.set_color(new_color)
                    elif isinstance(artist, Rectangle):
                        artist.set_facecolor(new_color)
                    elif isinstance(artist, PolyCollection):
                        # For fills, set both face and edge color
                        artist.set_facecolor(mcolors.to_rgba(new_color, alpha=0.6))
                        artist.set_edgecolor(new_color)
                except Exception:
                    pass


# ============================================================
# Plotly Category Color Support
# ============================================================

def _extract_plotly_categories(fig):
    """
    Extract categorical traces from a Plotly figure that can have their colors changed.
    
    Returns list of dicts: [{"name": str, "original_color": hex_color, "trace_idx": int}]
    
    Handles scatter traces with discrete colors (not colorscale/continuous).
    """
    categories = []
    
    if not hasattr(fig, 'data'):
        return categories
    
    for idx, trace in enumerate(fig.data):
        # Check if this is a scatter trace with a discrete color
        trace_type = trace.get('type', 'scatter')
        if trace_type not in ['scatter', 'scattergl']:
            continue
        
        # Get the trace name (category name)
        name = trace.get('name', '')
        if not name or name == 'Events':  # Skip generic names
            continue
        
        # Check if this trace has a single color (not colorscale)
        marker = trace.get('marker', {})
        if marker:
            color = marker.get('color')
            colorscale = marker.get('colorscale')
            
            # If colorscale is set and color is an array, this is a density plot
            if colorscale and isinstance(color, (list, tuple, np.ndarray)):
                continue
            
            # If color is a single value (string), this is a categorical trace
            if isinstance(color, str):
                categories.append({
                    "name": name,
                    "original_color": color,
                    "trace_idx": idx
                })
    
    return categories


def _has_plotly_categories(fig):
    """Check if a Plotly figure has categorical traces."""
    return len(_extract_plotly_categories(fig)) > 0


def _apply_plotly_category_colors(fig, category_colors):
    """
    Apply custom colors to categorical traces in a Plotly figure.
    
    Args:
        fig: Plotly Figure
        category_colors: dict mapping category name to hex color
    """
    if not category_colors:
        return
    
    categories = _extract_plotly_categories(fig)
    
    for cat_info in categories:
        cat_name = cat_info["name"]
        if cat_name in category_colors:
            new_color = category_colors[cat_name]
            trace_idx = cat_info["trace_idx"]
            fig.data[trace_idx].marker.color = new_color


# ============================================================
# Canvas settings, rendering and export
# ============================================================

_TIGHT_LAYOUT_WARNING = "This figure includes Axes that are not compatible with tight_layout"
_FONT_FAMILIES = ["default", "sans-serif", "serif", "monospace"]
_CMAPS = ["viridis", "plasma", "inferno", "magma", "cividis", "Greys", "Purples", "Blues", "Greens",
          "Oranges", "Reds", "YlOrBr", "YlOrRd", "OrRd", "PuRd", "RdPu", "BuPu", "GnBu", "PuBu",
          "YlGnBu", "PuBuGn", "BuGn", "YlGn", "jet", "rainbow", "Spectral"]
_DEFAULT_SETTINGS = {"width_scale": 1.0, "height_scale": 1.0, "font_scale": 1.0, "dot_scale": 1.0,
                     "font_family": "default", "main_cmap": "viridis"}
_TYPE_LABELS = {"plotly": "Interactive plot (Plotly)", "asperatio": "Asperatio table"}
_MIME = {"png": "image/png", "svg": "image/svg+xml", "pdf": "application/pdf",
         "html": "text/html", "csv": "text/csv"}

# Figures are mutated when settings are applied; "Export all" runs on Streamlit's download
# thread, so figure access from both sides goes through this lock.
_FIG_LOCK = threading.Lock()


def _settings(plot_data):
    """The plot's settings dict, with defaults filled in and legacy keys dropped."""
    settings = plot_data.setdefault("settings", {})
    for key, value in _DEFAULT_SETTINGS.items():
        settings.setdefault(key, value)
    for legacy in ("override_color_en", "override_line_color_en"):
        settings.pop(legacy, None)
    return settings


def _mpl_bytes(fig, fmt, dpi=None):
    buf = io.BytesIO()
    kwargs = {"format": fmt, "bbox_inches": "tight"}
    if dpi is not None:
        kwargs["dpi"] = dpi
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=_TIGHT_LAYOUT_WARNING)
        fig.savefig(buf, **kwargs)
    return buf.getvalue()


def _has_mappable(plot_data):
    """True if the plot maps values to a colormap (density scatter / colorscale)."""
    if plot_data.get("type") == "matplotlib":
        for ax in plot_data["fig"].axes:
            for c in ax.collections:
                if hasattr(c, "get_cmap") and hasattr(c, "get_array"):
                    arr = c.get_array()
                    if arr is not None and len(arr) > 1:
                        return True
    elif plot_data.get("type") == "plotly":
        for t in getattr(plot_data["data"], "data", []):
            if "marker" in t and ("color" in t.marker or "colorscale" in t.marker):
                if "colorscale" in t.marker or (
                        isinstance(t.marker.color, (list, tuple, np.ndarray)) and len(t.marker.color) > 1):
                    return True
    return False


# ---------- controls ----------

def _color_picker_rows(plot_id, prefix, items, store):
    """One name + colour-picker row per distinct name; changed colours are written to `store`.

    Colours are applied by name, so repeated names (e.g. "Group 1" on every pairwise subplot)
    share one picker.
    """
    seen = set()
    for info in items:
        name = info["name"]
        if name in seen:
            continue
        seen.add(name)
        current = store.get(name, _color_to_hex(info["original_color"]))
        col_name, col_picker = st.columns([2, 1])
        col_name.text(name)
        new_color = col_picker.color_picker("Color", value=current, key=f"{prefix}_{plot_id}_{name}",
                                            label_visibility="collapsed")
        if new_color != current:
            store[name] = new_color


def _render_color_controls(plot_id, plot_data):
    """Per-sample / per-category colour pickers, depending on what the figure contains."""
    ptype = plot_data.get("type")
    title, items = None, []
    if ptype == "matplotlib":
        f = plot_data["fig"]
        if _has_categorical_scatter(f):  # e.g. Singlet/Doublet scatter
            title, prefix, store_key = "Category Colors", "mplcat", "mpl_category_colors"
            items = _extract_categorical_scatter(f)
        else:
            prefix, store_key = "sc", "sample_colors"
            has_density = _has_density_scatter(f)
            if has_density and _has_colorable_lines(f):  # facet plot: trend lines only
                title, items = "Trend Line Colors", _extract_colorable_samples(f, lines_only=True)
            elif not has_density:  # histogram or bar plot
                title, items = "Sample Colors", _extract_colorable_samples(f, lines_only=False)
    elif ptype == "plotly":
        title, prefix, store_key = "Category Colors", "pcat", "category_colors"
        items = _extract_plotly_categories(plot_data["data"])
    if not items:
        return
    st.markdown(f"**{title}**")
    _color_picker_rows(plot_id, prefix, items, plot_data["settings"].setdefault(store_key, {}))


def _render_customize_controls(plot_id, plot_data):
    settings = _settings(plot_data)
    st.markdown("**Customize**")
    c_w, c_h = st.columns(2)
    settings["width_scale"] = c_w.slider("Width Scale", 0.5, 3.0, float(settings["width_scale"]), 0.1,
                                         key=f"ws_{plot_id}")
    settings["height_scale"] = c_h.slider("Height Scale", 0.5, 3.0, float(settings["height_scale"]), 0.1,
                                          key=f"hs_{plot_id}")
    settings["font_scale"] = st.slider("Font/Zoom Scale", 0.5, 3.0, float(settings["font_scale"]), 0.1,
                                       key=f"fs_{plot_id}")
    settings["dot_scale"] = st.slider("Dot Scale", 0.1, 5.0, float(settings["dot_scale"]), 0.1,
                                      key=f"ds_{plot_id}")
    family = settings["font_family"] if settings["font_family"] in _FONT_FAMILIES else "default"
    settings["font_family"] = st.selectbox("Font Family", _FONT_FAMILIES,
                                           index=_FONT_FAMILIES.index(family), key=f"ff_{plot_id}")
    if _has_mappable(plot_data):
        cmap = settings["main_cmap"] if settings["main_cmap"] in _CMAPS else "viridis"
        settings["main_cmap"] = st.selectbox("Select Colormap", _CMAPS, index=_CMAPS.index(cmap),
                                             key=f"mc_{plot_id}")
    else:
        settings["main_cmap"] = "viridis"
    _render_color_controls(plot_id, plot_data)


# ---------- applying settings ----------

def _apply_mpl_settings(fig, settings):
    """Resize, restyle and recolour a matplotlib figure in place according to `settings`."""
    if not hasattr(fig, "_glow_orig_size"):
        fig._glow_orig_size = fig.get_size_inches()
    w_orig, h_orig = fig._glow_orig_size
    font_scale = settings["font_scale"]
    # Font scaling: shrink the figure in inches (and raise the dpi when rendering) so text grows
    # relative to the data area.
    fig.set_size_inches(w_orig * settings["width_scale"] / font_scale,
                        h_orig * settings["height_scale"] / font_scale)

    if settings["font_family"] != "default":
        for txt in fig.findobj(match=mpl_text.Text):
            txt.set_fontfamily(settings["font_family"])

    for ax in fig.axes:
        for coll in ax.collections:
            if isinstance(coll, PolyCollection):  # keep ridge/fill colours as they are
                continue
            if hasattr(coll, "get_sizes") and hasattr(coll, "set_sizes"):
                if not hasattr(coll, "_glow_orig_sizes"):
                    sizes = coll.get_sizes()
                    coll._glow_orig_sizes = sizes.copy() if len(sizes) > 0 else [10]
                coll.set_sizes([s * settings["dot_scale"] for s in coll._glow_orig_sizes])
            if hasattr(coll, "set_cmap") and hasattr(coll, "get_array"):
                arr = coll.get_array()
                if arr is not None and len(arr) > 1:
                    coll.set_cmap(settings["main_cmap"])

    if settings.get("sample_colors"):
        _apply_sample_colors(fig, settings["sample_colors"])
    if settings.get("mpl_category_colors"):
        _apply_mpl_category_colors(fig, settings["mpl_category_colors"])
    # Redraw so fills/polygons on log axes follow the resize.
    fig.canvas.draw()


def _apply_plotly_settings(fig, settings):
    font = {"size": 12 * settings["font_scale"]}
    if settings["font_family"] != "default":
        font["family"] = settings["font_family"]
    fig.update_layout(font=font)
    if not _has_plotly_categories(fig):  # colorscale only for continuous colouring
        fig.update_traces(marker=dict(colorscale=settings["main_cmap"]))
    if settings.get("category_colors"):
        _apply_plotly_category_colors(fig, settings["category_colors"])
    if settings["dot_scale"] != 1.0:
        fig.update_traces(marker=dict(size=8 * settings["dot_scale"]), selector=dict(type="scatter"))
    # Width follows the container, so the aspect ratio is expressed through the height.
    fig.update_layout(height=int(450 * settings["height_scale"] / settings["width_scale"]), width=None)


# ---------- export ----------

def _export_files(plot_data):
    """{extension: bytes} for one canvas entry, with its settings applied. Caller holds _FIG_LOCK."""
    ptype = plot_data.get("type")
    settings = _settings(plot_data)
    files = {}
    if ptype == "matplotlib":
        fig = plot_data["fig"]
        _apply_mpl_settings(fig, settings)
        files["png"] = _mpl_bytes(fig, "png", dpi=300)
        files["svg"] = _mpl_bytes(fig, "svg")
        files["pdf"] = _mpl_bytes(fig, "pdf")
    elif ptype == "plotly":
        fig = plot_data["data"]
        _apply_plotly_settings(fig, settings)
        for fmt, scale in (("png", 2), ("pdf", 1)):
            try:
                files[fmt] = fig.to_image(format=fmt, scale=scale)  # needs kaleido
            except Exception:
                pass
        if not files:
            files["html"] = fig.to_html(include_plotlyjs="cdn").encode()
    elif ptype == "asperatio" and hasattr(plot_data.get("data"), "to_csv"):
        files["csv"] = plot_data["data"].to_csv(index=False).encode()
    return files


def _detached_copy(plot_data):
    """Copy of a canvas entry with its own figure object, so export can't disturb the page."""
    copy = dict(plot_data)
    copy["settings"] = dict(plot_data.get("settings", {}))
    key = "fig" if plot_data.get("type") == "matplotlib" else "data"
    if key in plot_data:
        try:
            copy[key] = pickle.loads(pickle.dumps(plot_data[key]))
        except Exception:
            pass  # unpicklable figure: export the original (still under the lock)
    return copy


def _export_all_zip(entries):
    """ZIP of every canvas plot (canvas order) plus one multi-page PDF of the matplotlib plots."""
    with _FIG_LOCK:
        exported, pdf_figs = [], []
        for plot_data in entries:
            plot_data = _detached_copy(plot_data)
            files = _export_files(plot_data)
            if files:
                exported.append((plot_data.get("title", "plot"), files))
            if plot_data.get("type") == "matplotlib":
                pdf_figs.append(plot_data["fig"])
        extra = {}
        if pdf_figs:
            buf = io.BytesIO()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, message=_TIGHT_LAYOUT_WARNING)
                with PdfPages(buf) as pdf:
                    for fig in pdf_figs:
                        pdf.savefig(fig, bbox_inches="tight")
            extra["canvas_all_plots.pdf"] = buf.getvalue()
    return build_canvas_zip(exported, extra)


# ---------- layout ----------

def _thumbnail(plot_data):
    """Small PNG of a matplotlib plot, cached until its settings change; None for other types."""
    if plot_data.get("type") != "matplotlib":
        return None
    key = json.dumps(plot_data.get("settings", {}), sort_keys=True, default=str)
    cached = plot_data.get("_thumb")
    if cached and cached[0] == key:
        return cached[1]
    try:
        with _FIG_LOCK:
            png = _mpl_bytes(plot_data["fig"], "png", dpi=30)
    except Exception:
        return None
    plot_data["_thumb"] = (key, png)
    return png


def _selected_plot_id(plots):
    selected = st.session_state.get("canvas_selected")
    if selected not in plots:
        selected = next(iter(plots))
        st.session_state["canvas_selected"] = selected
    return selected


def _render_plot_list(plots, selected_id):
    """Left pane: one thumbnail card per plot; clicking its title selects it."""
    for i, (plot_id, plot_data) in enumerate(plots.items(), start=1):
        with st.container(border=True):
            thumb = _thumbnail(plot_data)
            if thumb is not None:
                st.image(thumb, width="stretch")
            else:
                st.caption(_TYPE_LABELS.get(plot_data.get("type"), "Plot"))
            marker = "▸ " if plot_id == selected_id else ""
            if st.button(f"{marker}{i}. {plot_data.get('title', plot_id)}", key=f"sel_{plot_id}",
                         width="stretch", disabled=plot_id == selected_id):
                st.session_state["canvas_selected"] = plot_id
                st.rerun()


def _render_selected_header(plot_id, title, position, count):
    """Title of the selected plot with Move up / Move down / Delete (confirmed)."""
    c_title, c_up, c_down, c_del = st.columns([4, 1.2, 1.2, 1], vertical_alignment="center")
    c_title.markdown(f"#### {title}")
    if c_up.button("↑ Move up", key=f"up_{plot_id}", disabled=position == 0, width="stretch"):
        move_plot_in_canvas(plot_id, "up")
        st.rerun()
    if c_down.button("↓ Move down", key=f"down_{plot_id}", disabled=position == count - 1, width="stretch"):
        move_plot_in_canvas(plot_id, "down")
        st.rerun()
    with c_del.popover("Delete", width="stretch"):
        st.write(f"Remove **{title}** from the canvas? This cannot be undone.")
        if st.button("Yes, delete", key=f"rm_{plot_id}", type="primary"):
            remove_plot_from_canvas(plot_id)
            st.session_state.pop("canvas_selected", None)
            st.rerun()


def _render_download_row(plot_id, title, files):
    if not files:
        st.caption("No download available for this plot type.")
        return
    stem = safe_filename(title)
    for col, (ext, data) in zip(st.columns(len(files)), files.items()):
        col.download_button(f"Download {ext.upper()}", data, f"{stem}.{ext}",
                            _MIME.get(ext, "application/octet-stream"),
                            key=f"dl_{ext}_{plot_id}", width="stretch")


def _render_plot_body(plot_id, plot_data):
    ptype = plot_data.get("type")
    if ptype == "matplotlib":
        with _FIG_LOCK:
            files = _export_files(plot_data)
            preview = _mpl_bytes(plot_data["fig"], "png", dpi=100 * plot_data["settings"]["font_scale"])
        st.image(preview, width="stretch")
    elif ptype == "plotly":
        with _FIG_LOCK:
            files = _export_files(plot_data)
        st.plotly_chart(plot_data["data"], width="stretch", key=f"canvas_chart_{plot_id}")
    elif ptype == "asperatio":
        st.dataframe(plot_data["data"])
        files = _export_files(plot_data)
    else:
        st.warning(f"Unsupported plot type: {ptype}")
        return
    _render_download_row(plot_id, plot_data.get("title", "plot"), files)


def _render_selected_plot(plot_id, plot_data):
    """Right pane: the selected plot large, its customize controls beside it."""
    plot_col, ctrl_col = st.columns([3, 1.3])
    with ctrl_col:  # controls first so this rerun renders with the new settings
        _render_customize_controls(plot_id, plot_data)
    with plot_col:
        try:
            _render_plot_body(plot_id, plot_data)
        except Exception as e:
            st.error(f"Error rendering plot: {e}")


def render_canvas_tab():
    st.header("Plot Canvas")

    plots = get_canvas_plots()
    if not plots:
        st.info("No plots added to canvas yet. Use 'Add to Canvas' buttons from other tabs.")
        return

    selected_id = _selected_plot_id(plots)
    order = list(plots)
    entries = [plots[pid] for pid in order]

    c_info, c_export = st.columns([3, 1], vertical_alignment="center")
    c_info.caption(f"{len(order)} plot(s). Select one on the left to customize it; "
                   "Export all downloads every plot (PNG/SVG/PDF) and one multi-page PDF as a ZIP.")
    c_export.download_button("Export all (ZIP)", data=lambda: _export_all_zip(entries),
                             file_name="glow_canvas.zip", mime="application/zip",
                             key="canvas_export_all", on_click="ignore", width="stretch")

    list_col, main_col = st.columns([1, 3])
    # Fill the right pane first so thumbnails reflect settings changed in this rerun.
    with main_col:
        _render_selected_header(selected_id, plots[selected_id].get("title", selected_id),
                                order.index(selected_id), len(order))
        _render_selected_plot(selected_id, plots[selected_id])
    with list_col:
        _render_plot_list(plots, selected_id)

