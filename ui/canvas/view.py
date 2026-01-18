import streamlit as st
import io
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from glow.canvas.functions import get_canvas_plots, remove_plot_from_canvas, move_plot_in_canvas

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
                         if has_mappable: break

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
                    if current_cmap not in cmaps: current_cmap = "viridis"
                    
                    main_cmap = st.selectbox("Select Colormap", cmaps, index=cmaps.index(current_cmap), key=f"mc_{plot_id}")
                
                # Save settings
                plot_data["settings"]["width_scale"] = width_scale
                plot_data["settings"]["height_scale"] = height_scale
                plot_data["settings"]["font_scale"] = font_scale
                plot_data["settings"]["dot_scale"] = dot_scale
                plot_data["settings"]["font_family"] = font_family
                plot_data["settings"]["main_cmap"] = main_cmap
                # Force cleanup of old keys if present
                if "override_color_en" in plot_data["settings"]: del plot_data["settings"]["override_color_en"]
                if "override_line_color_en" in plot_data["settings"]: del plot_data["settings"]["override_line_color_en"]

            with c4:
                if st.button("🗑", key=f"rm_{plot_id}", help="Remove Plot"):
                    remove_plot_from_canvas(plot_id)
                    st.rerun()

            # Render logic
            ptype = plot_data.get("type")
            try:
                if ptype == "matplotlib":
                    fig = plot_data["fig"]
                    import matplotlib

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
                        for txt in fig.findobj(match=matplotlib.text.Text):
                             txt.set_fontfamily(font_family)

                    # Apply Dot Scaling & Color Override for Matplotlib
                    # Iterate through axes and collections
                    for ax in fig.axes:
                        # Update Axis Labels/Title Colors if needed? Usually we just color data.
                        
                        for collection in ax.collections:
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

                    # Render to PNG buffer
                    buf = io.BytesIO()
                    # Increase DPI with scale to maintain sharpness/resolution
                    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100 * font_scale)

                    st.image(buf, use_container_width=True)

                    # Reset buffer for downloading SVG (optional, or reuse fig)
                    # Stack buttons vertically
                    buf_svg = io.BytesIO()
                    try:
                        fig.savefig(buf_svg, format="svg", bbox_inches="tight")
                        st.download_button("Download SVG", buf_svg.getvalue(), f"{title}.svg", "image/svg+xml", key=f"dl_svg_{plot_id}")
                    except:
                        pass
                
                    buf_png = io.BytesIO()
                    try:
                        fig.savefig(buf_png, format="png", bbox_inches="tight", dpi=300)
                        st.download_button("Download PNG", buf_png.getvalue(), f"{title}.png", "image/png", key=f"dl_png_{plot_id}")
                    except:
                        pass

                    buf_pdf = io.BytesIO()
                    try:
                        fig.savefig(buf_pdf, format="pdf", bbox_inches="tight")
                        st.download_button("Download PDF", buf_pdf.getvalue(), f"{title}.pdf", "application/pdf", key=f"dl_pdf_{plot_id}")
                    except:
                        pass

                elif ptype == "plotly":
                    fig = plot_data["data"]

                    # Update Font Size & Family
                    update_font = dict(size=12 * font_scale)
                    if font_family != "default":
                        update_font["family"] = font_family
                    
                    fig.update_layout(font=update_font)

                    # Color Override - Always active for mappables
                    # Detection logic was: has colorscale or color array.
                    # We blindly update trace marker colorscale if it exists or we can infer it should?
                    # Safer: Update colorscale on ALL traces that theoretically support it. 
                    # If they don't have scalar data, colorscale is ignored by plotly usually.
                    fig.update_traces(marker=dict(colorscale=main_cmap))

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
                    except Exception as e:
                        pass
                        # st.caption("PNG export needs 'kaleido'")

                    try:
                        img_bytes = fig.to_image(format="pdf")
                        st.download_button("Download PDF", img_bytes, f"{title}.pdf", "application/pdf", key=f"dl_p_pdf_{plot_id}")
                    except Exception as e:
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
