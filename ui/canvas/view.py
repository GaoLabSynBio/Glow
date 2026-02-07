import streamlit as st
import io
import numpy as np
import warnings
from glow.canvas.functions import get_canvas_plots, remove_plot_from_canvas, move_plot_in_canvas
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

def render_canvas_tab():
    st.header("Plot Canvas")

    plots = get_canvas_plots()
    if not plots:
        st.info("No plots added to canvas yet. Use 'Add to Canvas' buttons from other tabs.")
        return

    # Global Canvas Layout Settings
    with st.expander("Canvas Layout Settings", expanded=False):
        n_cols = st.slider("Number of Columns", 1, 4, 2, key="canvas_n_cols")

    # Creates columns for grid layout
    cols = st.columns(n_cols)

    # Iterate over plots
    # keys are now UUIDs, values are dicts containing 'title'
    for i, (plot_id, plot_data) in enumerate(plots.items()):
        col = cols[i % n_cols]

        # Extract Display Title
        title = plot_data.get("title", plot_id)

        with col:
            st.markdown(f"### {title}")

            if "settings" not in plot_data:
                plot_data["settings"] = {"font_scale": 1.0, "dot_scale": 1.0}
            settings = plot_data["settings"]

            # Control Toolbar
            # Use unique plot_id for keys to avoid collisions if titles were (hypothetically) same
            # although usually unique per run, now definitely unique.
            c1, c2, c3, c4 = st.columns([1, 1, 2, 1])
            with c1:
                if st.button("⬆", key=f"up_{plot_id}", help="Move Backward"):
                    move_plot_in_canvas(plot_id, "up")
                    st.rerun()
            with c2:
                if st.button("⬇", key=f"down_{plot_id}", help="Move Forward"):
                    move_plot_in_canvas(plot_id, "down")
                    st.rerun()

            with st.expander("Customize"):
                # Aspect Ratio Controls
                c_ar1, c_ar2 = st.columns(2)
                with c_ar1:
                    width_scale = st.slider("Width Scale", 0.5, 3.0, settings.get("width_scale", 1.0), 0.1, key=f"ws_{plot_id}")
                with c_ar2:
                    height_scale = st.slider("Height Scale", 0.5, 3.0, settings.get("height_scale", 1.0), 0.1, key=f"hs_{plot_id}")

                # Font & Markers
                font_scale = st.slider("Font/Zoom Scale", 0.5, 3.0, settings.get("font_scale", 1.0), 0.1, key=f"fs_{plot_id}")
                dot_scale = st.slider("Dot Scale", 0.1, 5.0, settings.get("dot_scale", 1.0), 0.1, key=f"ds_{plot_id}")

                # Introspection to determine controls
                has_mappable = False

                # We need to peek at the object to know what controls to show
                if plot_data.get("type") == "matplotlib":
                    f = plot_data["fig"]
                    for ax in f.axes:
                         for c in ax.collections:
                             # Check for ScalarMappable characteristics
                             # It must have a colormap and an array of values mapping to it
                             if hasattr(c, "get_cmap") and hasattr(c, "get_array"):
                                 arr = c.get_array()
                                 if arr is not None and len(arr) > 1: # Must have data to map
                                     has_mappable = True
                                     break
                         if has_mappable:
                             break

                elif plot_data.get("type") == "plotly":
                    f = plot_data["data"]
                    # Check traces
                    if hasattr(f, "data"):
                        for t in f.data:
                            if "marker" in t and ("color" in t.marker or "colorscale" in t.marker):
                                # check if color is array or colorscale present
                                if "colorscale" in t.marker or (isinstance(t.marker.color, (list, tuple, np.ndarray)) and len(t.marker.color) > 1):
                                     has_mappable = True
                                     break

                # Font Family
                font_families = ["default", "sans-serif", "serif", "monospace"]
                font_family = st.selectbox("Font Family", font_families, index=font_families.index(settings.get("font_family", "default")), key=f"ff_{plot_id}")

                # Colormap Override (If Mappable)
                # Ensure override is always active if detected
                main_cmap = "viridis"
                if has_mappable:
                    # User requested default to "viridis" (which is coincidentally detecting as default anyway)
                    # And allow changing.

                    cmaps = ["viridis", "plasma", "inferno", "magma", "cividis", "Greys", "Purples", "Blues", "Greens", "Oranges", "Reds", "YlOrBr", "YlOrRd", "OrRd", "PuRd", "RdPu", "BuPu", "GnBu", "PuBu", "YlGnBu", "PuBuGn", "BuGn", "YlGn", "jet", "rainbow", "Spectral"]
                    current_cmap = settings.get("main_cmap", "viridis")
                    if current_cmap not in cmaps:
                        current_cmap = "viridis"

                    main_cmap = st.selectbox("Select Colormap", cmaps, index=cmaps.index(current_cmap), key=f"mc_{plot_id}")

                # Save settings
                plot_data["settings"]["width_scale"] = width_scale
                plot_data["settings"]["height_scale"] = height_scale
                plot_data["settings"]["font_scale"] = font_scale
                plot_data["settings"]["dot_scale"] = dot_scale
                plot_data["settings"]["font_family"] = font_family
                plot_data["settings"]["main_cmap"] = main_cmap
                # Force cleanup of old keys if present
                if "override_color_en" in plot_data["settings"]:
                    del plot_data["settings"]["override_color_en"]
                if "override_line_color_en" in plot_data["settings"]:
                    del plot_data["settings"]["override_line_color_en"]

                # ============================================================
                # Sample Color Customization (for histograms, bar plots, and 
                # trend lines in facet plots)
                # ============================================================
                if plot_data.get("type") == "matplotlib":
                    f = plot_data["fig"]
                    has_density = _has_density_scatter(f)
                    has_lines = _has_colorable_lines(f)
                    
                    # For facet plots (density + lines), extract lines only
                    # For pure histograms/bar plots, extract everything
                    if has_density and has_lines:
                        # Facet plot: show line colors only
                        colorable_samples = _extract_colorable_samples(f, lines_only=True)
                    elif not has_density:
                        # Pure histogram or bar plot
                        colorable_samples = _extract_colorable_samples(f, lines_only=False)
                    else:
                        # Pure density plot - no sample colors
                        colorable_samples = []
                    
                    if colorable_samples:
                        st.markdown("---")
                        if has_density and has_lines:
                            st.markdown("**Trend Line Colors**")
                        else:
                            st.markdown("**Sample Colors**")
                        
                        # Initialize sample_colors in settings if not present
                        if "sample_colors" not in plot_data["settings"]:
                            plot_data["settings"]["sample_colors"] = {}
                        
                        sample_colors = plot_data["settings"]["sample_colors"]
                        
                        for sample_info in colorable_samples:
                            sample_name = sample_info["name"]
                            original_color = sample_info["original_color"]
                            
                            # Get current color (either customized or original)
                            current_hex = sample_colors.get(sample_name, _color_to_hex(original_color))
                            
                            col_name, col_picker = st.columns([2, 1])
                            with col_name:
                                st.text(sample_name)
                            with col_picker:
                                new_color = st.color_picker(
                                    "Color",
                                    value=current_hex,
                                    key=f"sc_{plot_id}_{sample_name}",
                                    label_visibility="collapsed"
                                )
                                if new_color != current_hex:
                                    sample_colors[sample_name] = new_color

                # ============================================================
                # Plotly Category Color Customization (for categorical scatter plots)
                # ============================================================
                if plot_data.get("type") == "plotly":
                    f = plot_data["data"]
                    plotly_categories = _extract_plotly_categories(f)
                    
                    if plotly_categories:
                        st.markdown("---")
                        st.markdown("**Category Colors**")
                        
                        # Initialize category_colors in settings if not present
                        if "category_colors" not in plot_data["settings"]:
                            plot_data["settings"]["category_colors"] = {}
                        
                        category_colors = plot_data["settings"]["category_colors"]
                        
                        for cat_info in plotly_categories:
                            cat_name = cat_info["name"]
                            original_color = cat_info["original_color"]
                            
                            # Get current color (either customized or original)
                            current_hex = category_colors.get(cat_name, _color_to_hex(original_color))
                            
                            col_name, col_picker = st.columns([2, 1])
                            with col_name:
                                st.text(cat_name)
                            with col_picker:
                                new_color = st.color_picker(
                                    "Color",
                                    value=current_hex,
                                    key=f"pcat_{plot_id}_{cat_name}",
                                    label_visibility="collapsed"
                                )
                                if new_color != current_hex:
                                    category_colors[cat_name] = new_color

            with c4:
                if st.button("🗑", key=f"rm_{plot_id}", help="Remove Plot"):
                    remove_plot_from_canvas(plot_id)
                    st.rerun()

            # Render logic
            ptype = plot_data.get("type")
            try:
                if ptype == "matplotlib":
                    fig = plot_data["fig"]

                    # Logic to simulate "Font Scaling" by resizing the figure in inches
                    # while maintaining pixel density.
                    if not hasattr(fig, "_glow_orig_size"):
                        fig._glow_orig_size = fig.get_size_inches()

                    w_orig, h_orig = fig._glow_orig_size

                    # Apply Aspect Ratio & Font Scale
                    # To effectively scale font UP, we shrink the figure size (inches) so that relative text size is larger
                    # To stretch width, we multiply width inches
                    target_w = (w_orig * width_scale) / font_scale
                    target_h = (h_orig * height_scale) / font_scale

                    fig.set_size_inches(target_w, target_h)

                    # Update Font Family
                    # Changing global params often tricky, let's iterate text objects
                    if font_family != "default":
                        # Use introspection or a more robust find method if matplotlib.text is not directly exposed
                        # usually available as matplotlib.text.Text
                        import matplotlib.text as mpl_text
                        for txt in fig.findobj(match=mpl_text.Text):
                             txt.set_fontfamily(font_family)

                    # Apply Dot Scaling & Color Override for Matplotlib
                    # Iterate through axes and collections
                    for ax in fig.axes:
                        # Update Axis Labels/Title Colors if needed? Usually we just color data.

                        for collection in ax.collections:
                            # Skip PolyCollections (like fill_between/ridgelines) to preserve their specific facecolors
                            # Importing here to assume matplotlib is available if we have a fig
                            from matplotlib.collections import PolyCollection
                            if isinstance(collection, PolyCollection):
                                continue

                            # Only apply to PathCollections (Scatter) which have sizes
                            if hasattr(collection, "get_sizes") and hasattr(collection, "set_sizes"):
                                # Check if we stored original sizes
                                if not hasattr(collection, "_glow_orig_sizes"):
                                    sizes = collection.get_sizes()
                                    if len(sizes) > 0:
                                        collection._glow_orig_sizes = sizes.copy()
                                    else:
                                        collection._glow_orig_sizes = [10] # Default s=10

                                # Apply scale
                                if hasattr(collection, "_glow_orig_sizes"):
                                    new_sizes = [s * dot_scale for s in collection._glow_orig_sizes]
                                    collection.set_sizes(new_sizes)

                            # Color Override (Colormap) - Always Active if detected logic applies
                            # We re-check the mappable condition here to be safe and avoid breaking non-mappables
                            if hasattr(collection, "set_cmap") and hasattr(collection, "get_array"):
                                 arr = collection.get_array()
                                 if arr is not None and len(arr) > 1:
                                     collection.set_cmap(main_cmap)
                        # Automatic "Enabling" without user action might be intrusive if they already liked their plot.
                        # Showing the option is the "Enablement".

                    # Apply custom sample colors if any
                    sample_colors = settings.get("sample_colors", {})
                    if sample_colors:
                        _apply_sample_colors(fig, sample_colors)

                    # Force redraw after all modifications to ensure artists are updated
                    # This is critical for proper rendering of polygons/fills on log scale after resizing
                    fig.canvas.draw()

                    # Render to PNG buffer
                    buf = io.BytesIO()
                    # Increase DPI with scale to maintain sharpness/resolution
                    with warnings.catch_warnings():
                         warnings.filterwarnings("ignore", category=UserWarning, message="This figure includes Axes that are not compatible with tight_layout")
                         fig.savefig(buf, format="png", bbox_inches="tight", dpi=100 * font_scale)

                    st.image(buf, width="stretch")

                    # Reset buffer for downloading SVG (optional, or reuse fig)
                    # Stack buttons vertically
                    buf_svg = io.BytesIO()
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=UserWarning, message="This figure includes Axes that are not compatible with tight_layout")
                            fig.savefig(buf_svg, format="svg", bbox_inches="tight")
                        st.download_button("Download SVG", buf_svg.getvalue(), f"{title}.svg", "image/svg+xml", key=f"dl_svg_{plot_id}")
                    except Exception:
                        pass

                    buf_png = io.BytesIO()
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=UserWarning, message="This figure includes Axes that are not compatible with tight_layout")
                            fig.savefig(buf_png, format="png", bbox_inches="tight", dpi=300)
                        st.download_button("Download PNG", buf_png.getvalue(), f"{title}.png", "image/png", key=f"dl_png_{plot_id}")
                    except Exception:
                        pass

                    buf_pdf = io.BytesIO()
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=UserWarning, message="This figure includes Axes that are not compatible with tight_layout")
                            fig.savefig(buf_pdf, format="pdf", bbox_inches="tight")
                        st.download_button("Download PDF", buf_pdf.getvalue(), f"{title}.pdf", "application/pdf", key=f"dl_pdf_{plot_id}")
                    except Exception:
                        pass

                elif ptype == "plotly":
                    fig = plot_data["data"]

                    # Update Font Size & Family
                    update_font = dict(size=12 * font_scale)
                    if font_family != "default":
                        update_font["family"] = font_family

                    fig.update_layout(font=update_font)

                    # Check if this has categorical traces (don't apply colorscale to those)
                    has_categories = _has_plotly_categories(fig)
                    
                    # Color Override - Only for density/continuous plots, not categorical
                    if not has_categories:
                        # Detection logic was: has colorscale or color array.
                        # Update colorscale on traces that have continuous data
                        fig.update_traces(marker=dict(colorscale=main_cmap))

                    # Apply custom category colors if any
                    category_colors = settings.get("category_colors", {})
                    if category_colors:
                        _apply_plotly_category_colors(fig, category_colors)

                    # Update Markers logic for plotly
                    if dot_scale != 1.0:
                        # We don't have easy original size tracking here like mpl object reference
                        # We just apply a fixed size modifier or assume default=8
                        # Simplification: Set size explicitly
                        fig.update_traces(marker=dict(size=8 * dot_scale), selector=dict(type='scatter'))

                    # Usage of Aspect Ratio
                    # We always use container width to ensure fitting

                    layout_updates = {}

                    # To adjust aspect ratio while fitting width:
                    # Width is fixed (Container Width).
                    # If WidthScale increases (Wants wider), we must DECREASE HEIGHT.
                    # If HeightScale increases (Wants taller), we INCREASE HEIGHT.

                    base_h = 450 # Default Plotly height standard

                    # New Height = Base * HeightScale / WidthScale
                    new_height = base_h * (height_scale / width_scale)
                    layout_updates["height"] = int(new_height)
                    layout_updates["width"] = None # Let container handle it

                    fig.update_layout(**layout_updates)

                    st.plotly_chart(fig, width="stretch")

                    # Download Buttons for Plotly
                    # Stack vertically
                    try:
                        # Requires kaleido
                        img_bytes = fig.to_image(format="png", scale=2)
                        st.download_button("Download PNG", img_bytes, f"{title}.png", "image/png", key=f"dl_p_png_{plot_id}")
                    except Exception:
                        pass
                        # st.caption("PNG export needs 'kaleido'")

                    try:
                        img_bytes = fig.to_image(format="pdf")
                        st.download_button("Download PDF", img_bytes, f"{title}.pdf", "application/pdf", key=f"dl_p_pdf_{plot_id}")
                    except Exception:
                        pass
                        # st.caption("PDF export needs 'kaleido'")

                elif ptype == "asperatio":
                    # Custom rendering for asperatio data
                    asp_data = plot_data["data"]
                    st.write("**Asperatio Plot**")
                    st.dataframe(asp_data)

                else:
                    st.warning(f"Unsupported plot type: {ptype}")

            except Exception as e:
                st.error(f"Error rendering plot: {e}")

            st.divider()
