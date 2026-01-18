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
                font_scale = st.slider("Font/Zoom Scale", 0.5, 3.0, settings.get("font_scale", 1.0), 0.1, key=f"fs_{title}")
                # Dot scale only valid for Plotly currently
                dot_scale = st.slider("Dot Scale", 0.1, 5.0, settings.get("dot_scale", 1.0), 0.1, key=f"ds_{title}")

                plot_data["settings"]["font_scale"] = font_scale
                plot_data["settings"]["dot_scale"] = dot_scale

            with c4:
                if st.button("🗑", key=f"rm_{title}", help="Remove Plot"):
                    remove_plot_from_canvas(title)
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

                    w, h = fig._glow_orig_size
                    # Adjust size inversely to scale (Smaller inches = Relatively larger text elements)
                    fig.set_size_inches(w / font_scale, h / font_scale)

                    # Apply Dot Scaling for Matplotlib
                    # Iterate through axes and collections to scale markers
                    for ax in fig.axes:
                        for collection in ax.collections:
                            # Only apply to PathCollections (Scatter) which have sizes
                            if hasattr(collection, "get_sizes") and hasattr(collection, "set_sizes"):
                                # Check if we stored original sizes
                                if not hasattr(collection, "_glow_orig_sizes"):
                                    sizes = collection.get_sizes()
                                    if len(sizes) > 0:
                                        collection._glow_orig_sizes = sizes.copy()
                                    else:
                                        # Fallback if sizes not set explicitly
                                        collection._glow_orig_sizes = [10] # Default s=10

                                # Apply scale
                                if hasattr(collection, "_glow_orig_sizes"):
                                    new_sizes = [s * dot_scale for s in collection._glow_orig_sizes]
                                    collection.set_sizes(new_sizes)

                    # Render to PNG buffer
                    buf = io.BytesIO()
                    # Increase DPI with scale to maintain sharpness/resolution
                    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100 * font_scale)

                    st.image(buf, width="stretch")

                    # Reset buffer for downloading SVG (optional, or reuse fig)
                    buf_svg = io.BytesIO()
                    try:
                        fig.savefig(buf_svg, format="svg", bbox_inches="tight")
                        st.download_button("Download SVG", buf_svg.getvalue(), f"{title}.svg", "image/svg+xml", key=f"dl_svg_{title}")
                    except:
                        pass

                elif ptype == "plotly":
                    fig = plot_data["data"]

                    # Update Font
                    fig.update_layout(font=dict(size=12 * font_scale))

                    # Update Markers
                    if dot_scale != 1.0:
                        fig.update_traces(marker=dict(size=8 * dot_scale), selector=dict(type='scatter'))

                    st.plotly_chart(fig, width="stretch")

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
