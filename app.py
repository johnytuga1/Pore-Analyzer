import streamlit as st

# PATCH compatibilidade Streamlit / streamlit-drawable-canvas
try:
    import streamlit.elements.image as st_image
    from streamlit.elements.lib import image_utils
    from types import SimpleNamespace

    def image_to_url_fixed(
        image,
        width=None,
        clamp=True,
        channels="RGB",
        output_format="PNG",
        image_id=None,
        *args,
        **kwargs
    ):
        layout_config = SimpleNamespace(width=width)

        return image_utils.image_to_url(
            image,
            layout_config,
            clamp,
            channels,
            output_format,
            image_id
        )

    st_image.image_to_url = image_to_url_fixed

except Exception as e:
    st.warning(f"Patch not applied: {e}")

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from io import BytesIO


st.set_page_config(layout="wide")

st.title("Pore Analyzer V3.1 - SEM Images")

uploaded_file = st.file_uploader(
    "Upload SEM image",
    type=["png", "jpg", "jpeg", "tif", "tiff"]
)

scale_real_um = st.sidebar.number_input("Scale value (µm)", value=100.0)
scale_bar_pixels = st.sidebar.number_input("Scale bar length (pixels)", value=235.0)

um_per_pixel = scale_real_um / scale_bar_pixels

min_area_px = st.sidebar.number_input("Min pore area px", value=300)
max_area_px = st.sidebar.number_input("Max pore area px", value=50000)
threshold_offset = st.sidebar.slider("Threshold offset", -50, 100, 8)

drawing_mode = st.sidebar.selectbox(
    "Drawing mode",
    ["transform", "rect"],
    index=0
)

st.sidebar.markdown("---")
fit_to_screen = st.sidebar.checkbox("Fit image to screen", value=True)
display_width_setting = st.sidebar.slider(
    "Image display width (px)",
    min_value=600,
    max_value=1400,
    value=1000,
    step=50,
    disabled=fit_to_screen
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    image_width = int(image.size[0])
    image_height = int(image.size[1])

    # ---------------------------------------------------------
    # DISPLAY SCALE
    # The image/canvas can be shown smaller than the original.
    # All measurements are converted back to original-image pixels.
    # ---------------------------------------------------------
    if fit_to_screen:
        target_display_width = min(1000, image_width)
    else:
        target_display_width = min(int(display_width_setting), image_width)

    display_scale = target_display_width / image_width
    display_width = int(round(image_width * display_scale))
    display_height = int(round(image_height * display_scale))

    display_image = image.resize(
        (display_width, display_height),
        Image.Resampling.LANCZOS
    )

    img_np = np.array(image)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    crop_limit = int(h * 0.90)
    gray_crop = gray[:crop_limit, :]

    # ---------------------------------------------------------
    # PORE DETECTION V3
    # Detect dark depressions relative to their local surroundings.
    # Black-hat morphology highlights dark regions smaller than the
    # surrounding neighbourhood and is more suitable for textured SEM.
    # ---------------------------------------------------------
    blur = cv2.GaussianBlur(gray_crop, (5, 5), 0)

    # Local neighbourhood used to estimate the surrounding surface.
    local_kernel_size = st.sidebar.slider(
        "Pore neighbourhood size",
        min_value=15,
        max_value=101,
        value=41,
        step=2
    )
    local_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (local_kernel_size, local_kernel_size)
    )

    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, local_kernel)

    # Existing Threshold offset is used as sensitivity.
    # Higher offset -> lower threshold -> more candidate pores.
    sensitivity = int(np.clip(threshold_offset + 50, 1, 100))
    percentile = float(np.clip(100 - sensitivity * 0.45, 50, 99.5))
    detection_level = np.percentile(blackhat, percentile)

    _, th = cv2.threshold(
        blackhat,
        detection_level,
        255,
        cv2.THRESH_BINARY
    )

    kernel_small = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((5, 5), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel_small, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    # Preview so detection can be tuned before editing rectangles.
    with st.sidebar.expander("Detection Mask", expanded=False):
        st.image(
            th,
            caption="White regions = pore candidates",
            use_container_width=True
        )

    contours, _ = cv2.findContours(
        th,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    initial_rects = []

    for cnt in contours:
        area_px = cv2.contourArea(cnt)

        if area_px < min_area_px or area_px > max_area_px:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        initial_rects.append({
            "type": "rect",
            "left": int(round(x * display_scale)),
            "top": int(round(y * display_scale)),
            "width": int(round(bw * display_scale)),
            "height": int(round(bh * display_scale)),
            "fill": "rgba(255, 165, 0, 0.05)",
            "stroke": "orange",
            "strokeWidth": 2,
        })

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Edit rectangles")

        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.05)",
            stroke_width=2,
            stroke_color="orange",
            background_image=display_image,
            update_streamlit=True,
            height=display_height,
            width=display_width,
            drawing_mode=drawing_mode,
            initial_drawing={
                "version": "4.4.0",
                "objects": initial_rects
            },
            key="canvas",
        )

    rects = []

    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]

        particle_id = 1

        for obj in objects:
            if obj["type"] == "rect":

                # Canvas coordinates are display pixels.
                # Convert them back to original-image pixels before measuring.
                x_display = float(obj["left"])
                y_display = float(obj["top"])
                bw_display = float(obj["width"]) * float(obj.get("scaleX", 1))
                bh_display = float(obj["height"]) * float(obj.get("scaleY", 1))

                x = x_display / display_scale
                y = y_display / display_scale
                bw = bw_display / display_scale
                bh = bh_display / display_scale

                width_um = bw * um_per_pixel
                height_um = bh * um_per_pixel
                area_um2 = width_um * height_um

                rects.append({
                    "Particle_ID": particle_id,
                    "X_px": round(x, 2),
                    "Y_px": round(y, 2),
                    "Width_px": round(bw, 2),
                    "Height_px": round(bh, 2),
                    "Width_um": round(width_um, 2),
                    "Height_um": round(height_um, 2),
                    "Rectangle_Area_um2": round(area_um2, 2)
                })

                particle_id += 1

    df = pd.DataFrame(rects)

    with col2:
        st.subheader("Measurements")
        st.dataframe(df, use_container_width=True)

        excel_buffer = BytesIO()
        df.to_excel(excel_buffer, index=False)
        excel_buffer.seek(0)

        st.download_button(
            label="Download Excel",
            data=excel_buffer,
            file_name="pores_measurements.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )