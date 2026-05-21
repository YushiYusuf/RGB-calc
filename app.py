import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="RGB-based PDL Response Estimator",
    layout="wide",
)

st.title("RGB-based PDL Response Estimator for Capillary Malformation")
st.caption("Prototype app: upload one clinical photograph, select lesion and control ROIs, and calculate RGB, CAS, CAR, and EEV.")

st.warning(
    "This is a research prototype. The result depends strongly on lighting, camera settings, exposure, white balance, and ROI selection. "
    "Do not use this as a standalone clinical decision tool."
)

# -----------------------------
# Helper functions
# -----------------------------
def calculate_mean_rgb(image_array: np.ndarray, roi: tuple[int, int, int, int]) -> dict:
    """Calculate mean RGB values within ROI.

    roi = (x, y, w, h)
    """
    x, y, w, h = roi
    roi_pixels = image_array[y:y+h, x:x+w, :3]

    if roi_pixels.size == 0:
        return {"R": np.nan, "G": np.nan, "B": np.nan}

    mean_rgb = roi_pixels.reshape(-1, 3).mean(axis=0)
    return {
        "R": float(mean_rgb[0]),
        "G": float(mean_rgb[1]),
        "B": float(mean_rgb[2]),
    }


def calculate_cas(rgb: dict) -> dict:
    """Calculate Color Assessment Score for R, G, and B.

    CAS(channel) = channel / sqrt(R^2 + G^2 + B^2)
    """
    r, g, b = rgb["R"], rgb["G"], rgb["B"]
    denominator = np.sqrt(r**2 + g**2 + b**2)

    if denominator == 0 or np.isnan(denominator):
        return {"CAS_R": np.nan, "CAS_G": np.nan, "CAS_B": np.nan}

    return {
        "CAS_R": float(r / denominator),
        "CAS_G": float(g / denominator),
        "CAS_B": float(b / denominator),
    }


def calculate_indices(lesion_rgb: dict, control_rgb: dict) -> dict:
    lesion_cas = calculate_cas(lesion_rgb)
    control_cas = calculate_cas(control_rgb)

    car_r = lesion_cas["CAS_R"] / control_cas["CAS_R"]
    car_g = lesion_cas["CAS_G"] / control_cas["CAS_G"]
    car_b = lesion_cas["CAS_B"] / control_cas["CAS_B"]
    eev = car_r / car_g

    return {
        **{f"Lesion_{k}": v for k, v in lesion_rgb.items()},
        **{f"Control_{k}": v for k, v in control_rgb.items()},
        **lesion_cas,
        **{f"Control_{k}": v for k, v in control_cas.items()},
        "CAR_R": float(car_r),
        "CAR_G": float(car_g),
        "CAR_B": float(car_b),
        "EEV": float(eev),
    }


def draw_roi_boxes(image: Image.Image, lesion_roi: tuple[int, int, int, int], control_roi: tuple[int, int, int, int]) -> Image.Image:
    preview = image.copy()
    draw = ImageDraw.Draw(preview)

    lx, ly, lw, lh = lesion_roi
    cx, cy, cw, ch = control_roi

    draw.rectangle([lx, ly, lx + lw, ly + lh], outline="red", width=4)
    draw.text((lx, max(0, ly - 20)), "Lesion", fill="red")

    draw.rectangle([cx, cy, cx + cw, cy + ch], outline="blue", width=4)
    draw.text((cx, max(0, cy - 20)), "Control", fill="blue")

    return preview


def clamp_roi(x: int, y: int, w: int, h: int, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    x = max(0, min(x, image_width - 1))
    y = max(0, min(y, image_height - 1))
    w = max(1, min(w, image_width - x))
    h = max(1, min(h, image_height - y))
    return x, y, w, h

# -----------------------------
# Sidebar settings
# -----------------------------
st.sidebar.header("Settings")
cutoff = st.sidebar.number_input(
    "EEV cut-off",
    min_value=0.0,
    max_value=5.0,
    value=1.133,
    step=0.001,
    format="%.3f",
)

roi_size = st.sidebar.number_input(
    "Default ROI size, pixels",
    min_value=10,
    max_value=500,
    value=101,
    step=1,
)

st.sidebar.markdown(
    "Formula:  \
    CAS = channel / sqrt(R² + G² + B²)  \
    CAR = lesion CAS / control CAS  \
    EEV = CAR(R) / CAR(G)"
)

# -----------------------------
# Image upload
# -----------------------------
uploaded_file = st.file_uploader("Upload a clinical photograph", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("Upload an image to begin.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
image_array = np.array(image)
image_width, image_height = image.size

st.subheader("1. Uploaded image")
st.write(f"Image size: {image_width} × {image_height} pixels")

# -----------------------------
# ROI selection
# -----------------------------
st.subheader("2. Select ROIs")
st.write("Enter the top-left coordinates and size of each ROI. A click-and-drag ROI tool can be added later, but this numeric version is stable and easy to validate.")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Lesion ROI")
    lesion_x = st.number_input("Lesion x", min_value=0, max_value=max(0, image_width - 1), value=min(50, max(0, image_width - 1)), step=1)
    lesion_y = st.number_input("Lesion y", min_value=0, max_value=max(0, image_height - 1), value=min(50, max(0, image_height - 1)), step=1)
    lesion_w = st.number_input("Lesion width", min_value=1, max_value=image_width, value=min(int(roi_size), image_width), step=1)
    lesion_h = st.number_input("Lesion height", min_value=1, max_value=image_height, value=min(int(roi_size), image_height), step=1)

with col2:
    st.markdown("### Control ROI")
    control_x = st.number_input("Control x", min_value=0, max_value=max(0, image_width - 1), value=min(200, max(0, image_width - 1)), step=1)
    control_y = st.number_input("Control y", min_value=0, max_value=max(0, image_height - 1), value=min(50, max(0, image_height - 1)), step=1)
    control_w = st.number_input("Control width", min_value=1, max_value=image_width, value=min(int(roi_size), image_width), step=1)
    control_h = st.number_input("Control height", min_value=1, max_value=image_height, value=min(int(roi_size), image_height), step=1)

lesion_roi = clamp_roi(int(lesion_x), int(lesion_y), int(lesion_w), int(lesion_h), image_width, image_height)
control_roi = clamp_roi(int(control_x), int(control_y), int(control_w), int(control_h), image_width, image_height)

preview = draw_roi_boxes(image, lesion_roi, control_roi)
st.image(preview, caption="ROI preview: red = lesion, blue = control", use_container_width=True)

# -----------------------------
# Calculation
# -----------------------------
st.subheader("3. RGB and index calculation")

lesion_rgb = calculate_mean_rgb(image_array, lesion_roi)
control_rgb = calculate_mean_rgb(image_array, control_roi)
results = calculate_indices(lesion_rgb, control_rgb)

rgb_table = pd.DataFrame(
    [
        {"ROI": "Lesion", **lesion_rgb},
        {"ROI": "Control", **control_rgb},
    ]
)

cas_table = pd.DataFrame(
    [
        {
            "ROI": "Lesion",
            "CAS_R": results["CAS_R"],
            "CAS_G": results["CAS_G"],
            "CAS_B": results["CAS_B"],
        },
        {
            "ROI": "Control",
            "CAS_R": results["Control_CAS_R"],
            "CAS_G": results["Control_CAS_G"],
            "CAS_B": results["Control_CAS_B"],
        },
    ]
)

index_table = pd.DataFrame(
    [
        {
            "CAR_R": results["CAR_R"],
            "CAR_G": results["CAR_G"],
            "CAR_B": results["CAR_B"],
            "EEV": results["EEV"],
            "Cut-off": cutoff,
        }
    ]
)

left, middle, right = st.columns(3)
with left:
    st.markdown("### Mean RGB")
    st.dataframe(rgb_table, use_container_width=True)

with middle:
    st.markdown("### CAS")
    st.dataframe(cas_table, use_container_width=True)

with right:
    st.markdown("### CAR and EEV")
    st.dataframe(index_table, use_container_width=True)

# -----------------------------
# Interpretation
# -----------------------------
st.subheader("4. Interpretation")

eev = results["EEV"]

if eev >= cutoff:
    st.success(f"EEV = {eev:.3f}. This is above the cut-off of {cutoff:.3f}. PDL response may be favourable.")
else:
    st.error(f"EEV = {eev:.3f}. This is below the cut-off of {cutoff:.3f}. PDL response may be less favourable.")

st.caption(
    "Interpretation is based on the reported threshold in the poster. External validation and standardised photography are required before clinical use."
)

# -----------------------------
# Export
# -----------------------------
st.subheader("5. Export")

export_row = {
    "file_name": uploaded_file.name,
    "image_width": image_width,
    "image_height": image_height,
    "lesion_x": lesion_roi[0],
    "lesion_y": lesion_roi[1],
    "lesion_w": lesion_roi[2],
    "lesion_h": lesion_roi[3],
    "control_x": control_roi[0],
    "control_y": control_roi[1],
    "control_w": control_roi[2],
    "control_h": control_roi[3],
    **results,
    "cutoff": cutoff,
    "prediction": "PDL likely effective" if eev >= cutoff else "PDL less likely effective",
}

export_df = pd.DataFrame([export_row])
st.dataframe(export_df, use_container_width=True)

csv = export_df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="Download result as CSV",
    data=csv,
    file_name="rgb_pdl_result.csv",
    mime="text/csv",
)
