import streamlit as st
import io
import matplotlib.pyplot as plt
import plotly.graph_objects as go
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
    for i, (title, plot_data) in enumerate(plots.items()):
        col = cols[i % n_cols]

        with col:
            st.markdown(f"### {title}")

            if "settings" not in plot_data:
                plot_data["settings"] = {"font_scale": 1.0, "dot_scale": 1.0}
            settings = plot_data["settings"]

            # Control Toolbar
            c1, c2, c3, c4 = st.columns([1, 1, 2, 1])
            with c1:
                if st.button("⬆", key=f"up_{title}", help="Move Backward"):
                    move_plot_in_canvas(title, "up")
                    st.rerun()
            with c2:
                if st.button("⬇", key=f"down_{title}", help="Move Forward"):
                    move_plot_in_canvas(title, "down")
                    st.rerun()

            with st.expander("Customize"):
                # Aspect Ratio Controls
                c_ar1, c_ar2 = st.columns(2)
                with c_ar1:
                    width_scale = st.slider("Width Scale", 0.5, 3.0, settings.get("width_scale", 1.0), 0.1, key=f"ws_{title}")
                with c_ar2:
                    height_scale = st.slider("Height Scale", 0.5, 3.0, settings.get("height_scale", 1.0), 0.1, key=f"hs_{title}")

                # Font & Markers
                font_scale = st.slider("Font/Zoom Scale", 0.5, 3.0, settings.get("font_scale", 1.0), 0.1, key=f"fs_{title}")
                dot_scale = st.slider("Dot Scale", 0.1, 5.0, settings.get("dot_scale", 1.0), 0.1, key=f"ds_{title}")

                # Font Family
                font_families = ["default", "sans-serif", "serif", "monospace"]
                font_family = st.selectbox("Font Family", font_families, index=font_families.index(settings.get("font_family", "default")), key=f"ff_{title}")

                # Color Override
                override_color_en = st.checkbox("Override Color", value=settings.get("override_color_en", False), key=f"oce_{title}")
                main_color = st.color_picker("Main Color", value=settings.get("main_color", "#000000"), key=f"mc_{title}")

                # Save settings
                plot_data["settings"]["width_scale"] = width_scale
                plot_data["settings"]["height_scale"] = height_scale
                plot_data["settings"]["font_scale"] = font_scale
                plot_data["settings"]["dot_scale"] = dot_scale
                plot_data["settings"]["font_family"] = font_family
                plot_data["settings"]["override_color_en"] = override_color_en
                plot_data["settings"]["main_color"] = main_color

            with c4:
                if st.button("🗑", key=f"rm_{title}", help="Remove Plot"):
                    remove_plot_from_canvas(title)
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

                            # Color Override
                            if override_color_en:
                                # Save original color if needed? For now straightforward override
                                collection.set_color(main_color)
                                collection.set_edgecolor(main_color)
                        
                        # Lines (Plots)
                        if override_color_en:
                            for line in ax.lines:
                                line.set_color(main_color)

                    # Render to PNG buffer
                    buf = io.BytesIO()
                    # Increase DPI with scale to maintain sharpness/resolution
                    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100 * font_scale)

                    st.image(buf, width="stretch")

                    # Reset buffer for downloading SVG (optional, or reuse fig)
                    # Stack buttons vertically
                    buf_svg = io.BytesIO()
                    try:
                        fig.savefig(buf_svg, format="svg", bbox_inches="tight")
                        st.download_button("Download SVG", buf_svg.getvalue(), f"{title}.svg", "image/svg+xml", key=f"dl_svg_{title}")
                    except:
                        pass
                
                    buf_png = io.BytesIO()
                    try:
                        fig.savefig(buf_png, format="png", bbox_inches="tight", dpi=300)
                        st.download_button("Download PNG", buf_png.getvalue(), f"{title}.png", "image/png", key=f"dl_png_{title}")
                    except:
                        pass

                    buf_pdf = io.BytesIO()
                    try:
                        fig.savefig(buf_pdf, format="pdf", bbox_inches="tight")
                        st.download_button("Download PDF", buf_pdf.getvalue(), f"{title}.pdf", "application/pdf", key=f"dl_pdf_{title}")
                    except:
                        pass

                elif ptype == "plotly":
                    fig = plot_data["data"]

                    # Update Font Size & Family
                    update_font = dict(size=12 * font_scale)
                    if font_family != "default":
                        update_font["family"] = font_family
                    
                    fig.update_layout(font=update_font)

                    # Color Override
                    if override_color_en:
                        fig.update_traces(marker=dict(color=main_color), line=dict(color=main_color))
                        # Note: This overrides everything to one color. 

                    # Update Markers logic for plotly
                    if dot_scale != 1.0:
                        # We don't have easy original size tracking here like mpl object reference
                        # We just apply a fixed size modifier or assume default=8
                        # Simplification: Set size explicitly
                        fig.update_traces(marker=dict(size=8 * dot_scale), selector=dict(type='scatter'))

                    # Usage of Aspect Ratio
                    # If specific scaling is requested, we use graph_objects specific layout width/height
                    # Use container width typically fills column.
                    # To "Stretch", we might need to fix the height while width fills?
                    # Or set both if we want precise control.
                    
                    use_container_width = True
                    layout_updates = {}
                    
                    if width_scale != 1.0 or height_scale != 1.0:
                        use_container_width = False # Take manual control
                        # Base size approximation
                        base_w = 600
                        base_h = 400
                        layout_updates["width"] = base_w * width_scale
                        layout_updates["height"] = base_h * height_scale
                    else:
                        # Reset if default
                        layout_updates["width"] = None
                        layout_updates["height"] = None
                        
                    fig.update_layout(**layout_updates)

                    st.plotly_chart(fig, use_container_width=use_container_width)

                    # Download Buttons for Plotly
                    # Stack vertically
                    try:
                        # Requires kaleido
                        img_bytes = fig.to_image(format="png", scale=2)
                        st.download_button("Download PNG", img_bytes, f"{title}.png", "image/png", key=f"dl_p_png_{title}")
                    except Exception as e:
                        pass
                        # st.caption("PNG export needs 'kaleido'")

                    try:
                        img_bytes = fig.to_image(format="pdf")
                        st.download_button("Download PDF", img_bytes, f"{title}.pdf", "application/pdf", key=f"dl_p_pdf_{title}")
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
