import streamlit as st
from bokeh.plotting import figure
import numpy as np
from PIL import Image
import io

# Import from shiny new streamlit-bokeh
try:
    from streamlit_bokeh import streamlit_bokeh
except ImportError:
    st.error("Please install streamlit-bokeh via pip install streamlit-bokeh")
    def streamlit_bokeh(*args, **kwargs):
        pass

from glow.canvas.functions import get_canvas_plots, remove_plot_from_canvas

def render_canvas_tab():
    st.header("Plot Canvas")

    plots = get_canvas_plots()
    if not plots:
        st.info("No plots added to canvas yet. Use 'Add to Canvas' buttons from other tabs.")
        return

    # Iterate over copy of items to allow modification (deletion) during loop
    for title, plot_data in list(plots.items()):
        st.subheader(title)

        try:
            if isinstance(plot_data, dict) and plot_data.get("type") == "png":
                # Display PNG via Bokeh
                image_data = plot_data["data"]
                image = Image.open(io.BytesIO(image_data)).convert("RGBA")

                # Convert to 2D uint32 array for Bokeh (fixes BidiComponent 3D array error)
                img_rgba = np.array(image) # (H, W, 4) uint8
                img_view = img_rgba.view(dtype=np.uint32).reshape(img_rgba.shape[:2]) # (H, W) uint32
                img_array = np.flipud(img_view)
                h, w = img_array.shape

                # Create Bokeh Figure
                # We use scale_width to attempt responsiveness
                p = figure(x_range=(0, w), y_range=(0, h),
                           title=None,
                           tools="pan,wheel_zoom,reset,save",
                           sizing_mode="scale_width")

                # Maintain aspect ratio
                p.aspect_ratio = w / h if h > 0 else 1.0

                p.image_rgba(image=[img_array], x=0, y=0, dw=w, dh=h)
                p.axis.visible = False
                p.grid.visible = False
                p.min_border = 0
                p.outline_line_color = None

                # Using the installed streamlit-bokeh
                streamlit_bokeh(p)

                # SVG Download Button
                if "svg_data" in plot_data:
                    st.download_button(
                        label="Download SVG",
                        data=plot_data["svg_data"],
                        file_name=f"{title}.svg",
                        mime="image/svg+xml",
                        key=f"dl_svg_{title}"
                    )

            elif isinstance(plot_data, dict) and plot_data.get("type") == "plotly":
                st.plotly_chart(plot_data["data"], use_container_width=True)

            else:
                st.warning(f"Plot '{title}' is in an unsupported format. Please remove and re-add.")

        except Exception as e:
            st.error(f"Error rendering plot '{title}': {e}")

        if st.button(f"Remove: {title}", key=f"remove_canvas_{title}"):
            remove_plot_from_canvas(title)
            st.rerun()

        st.divider()
