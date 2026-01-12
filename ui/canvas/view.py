import streamlit as st
import plotly.graph_objects as go
import plotly.io
from glow.canvas.functions import get_canvas_plots, remove_plot_from_canvas, update_canvas_plot

def sanitize_color(color, default):
    if not color or not str(color).startswith("#"):
        # Map common named colors to hex if Plotly uses them
        mapping = {"white": "#ffffff", "black": "#000000", "red": "#ff0000", "blue": "#0000ff", "green": "#008000"}
        return mapping.get(str(color).lower(), default)
    return color

def render_canvas_tab():
    st.header("Plot Canvas")
    
    plots = get_canvas_plots()
    
    if not plots:
        st.info("No plots added to canvas yet. Add them from the Gating or Plot tabs.")
        return

    for title, plot_json in list(plots.items()):
        st.write(f"### {title}")
        
        # Load figure
        fig = go.Figure(plotly.io.from_json(plot_json))
        
        # Customization options
        with st.expander(f"Edit & Annotate: {title}"):
            # 1. Layout & Colors
            st.markdown("**Layout & Colors**")
            c1, c2, c3 = st.columns(3)
            new_title = c1.text_input("Main Title", value=fig.layout.title.text or title, key=f"title_{title}")
            
            # Sanitize color strings (Streamlit requires hex)
            current_bg = sanitize_color(fig.layout.plot_bgcolor, "#ffffff")
            current_font = sanitize_color(fig.layout.font.color, "#000000")
            
            bg_color = c2.color_picker("Background", value=current_bg, key=f"bg_{title}")
            font_color = c3.color_picker("Font Color", value=current_font, key=f"fcolor_{title}")
            
            c4, c5, _ = st.columns(3)
            font_size = c4.number_input("Font Size", 8, 40, value=int(fig.layout.font.size or 12), key=f"fsize_{title}")
            trace_color = c5.color_picker("Data Color (All Traces)", key=f"tcolor_{title}")

            st.markdown("**Dimensions (px)**")
            d1, d2, _ = st.columns(3)
            # Default to 0 (Auto)
            current_w = fig.layout.width or 0
            current_h = fig.layout.height or 0
            
            p_width = d1.number_input("Width (0=Auto)", 0, 2000, value=int(current_w), key=f"w_{title}")
            p_height = d2.number_input("Height (0=Auto)", 0, 2000, value=int(current_h), key=f"h_{title}")

            # 2. Axes
            st.markdown("**Axes Labels**")
            ax1, ax2 = st.columns(2)
            x_label = ax1.text_input("X Axis Label", value=fig.layout.xaxis.title.text or "", key=f"xtitle_{title}")
            y_label = ax2.text_input("Y Axis Label", value=fig.layout.yaxis.title.text or "", key=f"ytitle_{title}")

            # 3. Text Annotations
            st.markdown("**Add Text Annotation**")
            ann_col1, ann_col2, ann_col3, ann_col4 = st.columns([2, 1, 1, 1])
            ann_text = ann_col1.text_input("Text Content", key=f"ann_txt_{title}")
            ann_x = ann_col2.number_input("X (0-1)", 0.0, 1.0, 0.5, 0.1, key=f"ann_x_{title}")
            ann_y = ann_col3.number_input("Y (0-1)", 0.0, 1.0, 0.5, 0.1, key=f"ann_y_{title}")
            
            if ann_col4.button("Add Text", key=f"add_ann_{title}"):
                if ann_text:
                    fig.add_annotation(
                        text=ann_text, x=ann_x, y=ann_y, 
                        xref="paper", yref="paper", 
                        showarrow=False, font=dict(color=font_color, size=font_size)
                    )
                    update_canvas_plot(title, fig)
                    st.rerun()

            # Action Buttons
            st.divider()
            b_col1, b_col2, b_col3 = st.columns(3)
            if b_col1.button("Apply Style Changes", key=f"apply_{title}", type="primary"):
                fig.update_layout(
                    title_text=new_title,
                    plot_bgcolor=bg_color,
                    paper_bgcolor=bg_color,
                    font=dict(color=font_color, size=font_size),
                    xaxis_title=x_label,
                    yaxis_title=y_label,
                    width=p_width if p_width > 0 else None,
                    height=p_height if p_height > 0 else None
                )
                update_canvas_plot(title, fig)
                st.rerun()
                
            if b_col2.button("Apply Data Color", key=f"apply_c_{title}"):
                fig.update_traces(marker_color=trace_color, line_color=trace_color)
                update_canvas_plot(title, fig)
                st.rerun()
                
            if b_col3.button("Clear Annotations", key=f"clear_ann_{title}"):
                fig.layout.annotations = []
                update_canvas_plot(title, fig)
                st.rerun()

        # Determine if we use container width or fixed width
        use_container = (fig.layout.width is None or fig.layout.width <= 0)
        st.plotly_chart(fig, use_container_width=use_container)
        
        if st.button(f"Remove {title}", key=f"remove_{title}"):
            remove_plot_from_canvas(title)
            st.rerun()
        st.divider()
