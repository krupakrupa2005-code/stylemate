import streamlit as st
import pandas as pd
import numpy as np
import joblib
import cv2
from ultralytics import YOLO
import math
from pathlib import Path

st.set_page_config(page_title="StyleMate", page_icon="👗", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATHS = [BASE_DIR / "stylemate_model.pkl", BASE_DIR / "dataset" / "stylemate_model.pkl"]

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    for path in MODEL_PATHS:

        if path.exists():

            return joblib.load(path), path

    return None, None


model, model_path = load_model()


if model is None:

    st.error(
        "⚠️ StyleMate model was not found. "
        "Make sure stylemate_model.pkl is inside the project folder "
        "or the dataset folder."
    )

    st.stop()


# ============================================================
# GET MODEL CATEGORIES
# ============================================================

categorical_features = [
    "Body_Shape",
    "Colour_Palette",
    "Garment_Type",
    "Garment_Colour",
    "Sleeve",
    "Neckline",
    "Pattern",
    "Fabric",
    "Occasion"
]


def get_model_categories(feature_name):

    """
    Extract categories directly from the OneHotEncoder
    inside the saved StyleMate pipeline.
    """

    try:

        preprocessor = model.named_steps["preprocessor"]

        encoder = preprocessor.named_transformers_["cat"]

        category_index = categorical_features.index(feature_name)

        categories = encoder.categories_[category_index]

        return list(categories)

    except Exception:

        return []


# ============================================================
# YOLO COMPUTER VISION MODELS
# ============================================================

@st.cache_resource
def load_pose_model():
    path = BASE_DIR / "yolo11n-pose.pt"
    if path.exists():
        return YOLO(str(path))
    return YOLO("yolo11n-pose.pt")


@st.cache_resource
def load_seg_model():
    path = BASE_DIR / "yolo11n-seg.pt"
    if path.exists():
        return YOLO(str(path))
    return YOLO("yolo11n-seg.pt")


try:
    pose_model = load_pose_model()
    seg_model = load_seg_model()
except Exception as e:
    st.error("YOLO could not be loaded. Install it using `pip install ultralytics`.")
    st.exception(e)
    st.stop()


# ============================================================
# BODY SHAPE CLASSIFICATION
# ============================================================

def determine_body_shape(
    shoulder_width,
    waist_width,
    hip_width
):

    """
    Rule-based classification using estimated
    shoulder, waist and hip proportions.

    These thresholds are intentionally simple
    and are used as an interpretable CV layer.
    """

    if waist_width <= 0:

        return "Rectangle"

    shoulder_hip_ratio = shoulder_width / hip_width

    waist_shoulder_ratio = waist_width / shoulder_width

    waist_hip_ratio = waist_width / hip_width


    # Hourglass
    if (
        0.90 <= shoulder_hip_ratio <= 1.10
        and waist_shoulder_ratio <= 0.78
        and waist_hip_ratio <= 0.78
    ):

        return "Hourglass"


    # Pear
    if (
        hip_width > shoulder_width * 1.05
        and waist_hip_ratio < 0.82
    ):

        return "Pear"


    # Inverted Triangle
    if (
        shoulder_width > hip_width * 1.05
        and waist_shoulder_ratio < 0.82
    ):

        return "Inverted Triangle"


    # Apple
    if (
        waist_width >= shoulder_width * 0.90
        and waist_width >= hip_width * 0.90
    ):

        return "Apple"


    # Rectangle
    if (
        0.90 <= shoulder_hip_ratio <= 1.10
        and waist_shoulder_ratio > 0.78
    ):

        return "Rectangle"


    # Fallback
    if hip_width > shoulder_width:

        return "Pear"

    elif shoulder_width > hip_width:

        return "Inverted Triangle"

    return "Rectangle"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def point_to_pixel(landmark, width, height):

    x = int(landmark.x * width)

    y = int(landmark.y * height)

    return x, y


def euclidean_distance(p1, p2):

    return float(
        np.sqrt(
            (p1[0] - p2[0]) ** 2
            +
            (p1[1] - p2[1]) ** 2
        )
    )


def largest_contiguous_width(row):

    """
    Finds the largest continuous foreground section
    in a segmentation-mask row.
    """

    indices = np.where(row)[0]

    if len(indices) == 0:

        return 0

    splits = np.split(
        indices,
        np.where(np.diff(indices) > 1)[0] + 1
    )

    if not splits:

        return 0

    largest = max(
        splits,
        key=len
    )

    return int(
        largest[-1] - largest[0] + 1
    )


def estimate_width_from_mask(
    mask,
    y_center,
    x_min=None,
    x_max=None,
    radius=5
):

    height, width = mask.shape

    y_center = int(
        np.clip(
            y_center,
            0,
            height - 1
        )
    )

    widths = []

    for y in range(
        max(0, y_center - radius),
        min(height, y_center + radius + 1)
    ):

        row = mask[y] > 0.5

        if x_min is not None:

            row[:max(0, int(x_min))] = False

        if x_max is not None:

            row[min(width, int(x_max) + 1):] = False

        width_value = largest_contiguous_width(row)

        if width_value > 0:

            widths.append(width_value)

    if not widths:

        return None

    return float(
        np.median(widths)
    )


# ============================================================
# COMPUTER VISION BODY ANALYSIS
# ============================================================

def analyze_body_image(image, height_cm):
    """YOLO Pose + Segmentation body-proportion analysis."""

    pose_results = pose_model(image, verbose=False)

    if not pose_results or pose_results[0].keypoints is None:
        return None, None, "No body was detected."

    result = pose_results[0]

    if result.keypoints.xy is None or len(result.keypoints.xy) == 0:
        return None, None, "No body landmarks were detected."

    # First detected person.
    keypoints = result.keypoints.xy[0].cpu().numpy()

    if keypoints.shape[0] < 17:
        return None, None, "Body landmarks are incomplete."

    # YOLO COCO pose indices:
    # 0 nose, 5/6 shoulders, 11/12 hips, 15/16 ankles.
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    left_ankle = keypoints[15]
    right_ankle = keypoints[16]
    nose = keypoints[0]

    shoulder_center = (left_shoulder + right_shoulder) / 2
    hip_center = (left_hip + right_hip) / 2
    ankle_center = (left_ankle + right_ankle) / 2

    def distance(p1, p2):
        return math.sqrt(
            (p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2
        )

    shoulder_width_px = distance(left_shoulder, right_shoulder)
    hip_width_px = distance(left_hip, right_hip)
    torso_length_px = distance(shoulder_center, hip_center)
    leg_length_px = distance(hip_center, ankle_center)

    if shoulder_width_px <= 0 or hip_width_px <= 0:
        return None, None, "Could not estimate shoulder or hip width."

    # Estimate waist location between shoulders and hips.
    waist_y = shoulder_center[1] + 0.55 * (
        hip_center[1] - shoulder_center[1]
    )

    # Use segmentation for the waist silhouette.
    waist_width_px = None
    seg_results = seg_model(image, verbose=False)

    if seg_results and seg_results[0].masks is not None:
        masks = seg_results[0].masks.data

        if len(masks) > 0:
            mask = masks[0].cpu().numpy()
            image_h, image_w = image.shape[:2]

            mask = cv2.resize(
                mask,
                (image_w, image_h),
                interpolation=cv2.INTER_NEAREST
            )

            hip_left_x = min(left_hip[0], right_hip[0])
            hip_right_x = max(left_hip[0], right_hip[0])
            margin = hip_width_px * 0.45

            x_min = max(0, int(hip_left_x - margin))
            x_max = min(image_w - 1, int(hip_right_x + margin))

            widths = []

            for y in range(
                max(0, int(waist_y) - 6),
                min(image_h, int(waist_y) + 7)
            ):
                row = mask[y] > 0.5
                row[:x_min] = False
                row[x_max + 1:] = False
                indices = np.where(row)[0]

                if len(indices):
                    widths.append(indices[-1] - indices[0] + 1)

            if widths:
                waist_width_px = float(np.median(widths))

    if waist_width_px is None:
        waist_width_px = (
            0.65 * shoulder_width_px +
            0.35 * hip_width_px
        )

    # Approximate scale using user-provided height.
    top_y = min(nose[1], left_shoulder[1], right_shoulder[1])
    bottom_y = max(left_ankle[1], right_ankle[1])
    body_height_px = bottom_y - top_y

    if body_height_px <= 0:
        return None, None, "Could not estimate body height."

    pixels_per_cm = body_height_px / float(height_cm)

    shoulder_cm = shoulder_width_px / pixels_per_cm
    waist_cm = waist_width_px / pixels_per_cm
    hip_cm = hip_width_px / pixels_per_cm

    hip_waist_ratio = hip_cm / waist_cm if waist_cm > 0 else 0
    shoulder_hip_ratio = shoulder_cm / hip_cm if hip_cm > 0 else 0
    shoulder_waist_ratio = shoulder_cm / waist_cm if waist_cm > 0 else 0
    torso_leg_ratio = torso_length_px / leg_length_px if leg_length_px > 0 else 0

    body_shape = determine_body_shape(
        shoulder_width_px,
        waist_width_px,
        hip_width_px
    )

    # Annotated image.
    annotated = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)

    for x, y in keypoints:
        cv2.circle(annotated, (int(x), int(y)), 4, (255, 255, 255), -1)

    cv2.line(
        annotated,
        tuple(left_shoulder.astype(int)),
        tuple(right_shoulder.astype(int)),
        (255, 80, 200),
        4
    )

    cv2.line(
        annotated,
        tuple(left_hip.astype(int)),
        tuple(right_hip.astype(int)),
        (80, 180, 255),
        4
    )

    waist_y_int = int(waist_y)

    cv2.line(
        annotated,
        (
            max(0, int(shoulder_center[0] - shoulder_width_px * 0.55)),
            waist_y_int
        ),
        (
            min(
                image.shape[1] - 1,
                int(shoulder_center[0] + shoulder_width_px * 0.55)
            ),
            waist_y_int
        ),
        (100, 255, 150),
        3
    )

    analysis = {
        "shoulder_cm": round(shoulder_cm, 1),
        "waist_cm": round(waist_cm, 1),
        "hip_cm": round(hip_cm, 1),
        "hip_waist_ratio": round(hip_waist_ratio, 2),
        "shoulder_hip_ratio": round(shoulder_hip_ratio, 2),
        "shoulder_waist_ratio": round(shoulder_waist_ratio, 2),
        "torso_length_px": round(torso_length_px, 1),
        "leg_length_px": round(leg_length_px, 1),
        "torso_leg_ratio": round(torso_leg_ratio, 2),
        "body_shape": body_shape
    }

    return analysis, annotated, None


# ============================================================
# STYLEMATE USER INTERFACE
# ============================================================

st.title("👗 StyleMate")
st.subheader("AI-Based Personalized Fashion Recommendation System")
st.write("Upload a full-body photo to analyse body proportions, then select your fashion preferences to get Top-3 style recommendations.")

st.divider()

# ------------------------------------------------------------
# PHOTO INPUT
# ------------------------------------------------------------

st.header("🧍 AI Body Proportion Analysis")

left, right = st.columns([2, 1])

with left:
    uploaded_file = st.file_uploader(
        "Upload a clear, front-facing full-body photograph",
        type=["jpg", "jpeg", "png"]
    )

with right:
    height_cm = st.number_input(
        "Your height (cm)",
        min_value=120.0,
        max_value=220.0,
        value=165.0,
        step=1.0
    )

cv_analysis = None

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Could not read the uploaded image.")
    else:
        with st.spinner("🔍 Analysing body proportions with YOLO..."):
            cv_analysis, annotated_image, error_message = analyze_body_image(image, height_cm)

        if error_message:
            st.error(error_message)
            st.info("Use a front-facing image with the complete body visible, including both ankles.")
        else:
            image_col, result_col = st.columns(2)

            with image_col:
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)

            with result_col:
                st.image(annotated_image, caption="YOLO Body Analysis", use_container_width=True)

            st.success("✅ Body landmarks detected successfully")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Shoulder", f"{cv_analysis['shoulder_cm']:.1f} cm")
            m2.metric("Waist", f"{cv_analysis['waist_cm']:.1f} cm")
            m3.metric("Hip", f"{cv_analysis['hip_cm']:.1f} cm")
            m4.metric("Hip / Waist", f"{cv_analysis['hip_waist_ratio']:.2f}")

            st.write(
                f"**Shoulder/Hip:** {cv_analysis['shoulder_hip_ratio']:.2f}  |  "
                f"**Shoulder/Waist:** {cv_analysis['shoulder_waist_ratio']:.2f}  |  "
                f"**Torso/Leg:** {cv_analysis['torso_leg_ratio']:.2f}"
            )

st.divider()

# ------------------------------------------------------------
# FASHION PROFILE
# ------------------------------------------------------------

st.header("👤 Your Fashion Profile")

body_shape_options = get_model_categories("Body_Shape")
if not body_shape_options:
    body_shape_options = ["Hourglass", "Pear", "Rectangle", "Apple", "Inverted Triangle"]

if cv_analysis:
    detected_shape = cv_analysis["body_shape"]
    default_index = body_shape_options.index(detected_shape) if detected_shape in body_shape_options else 0
    selected_body_shape = st.selectbox(
        "Body Shape — confirm or change the computer-vision estimate",
        body_shape_options,
        index=default_index
    )
    st.caption(f"Computer Vision estimate: {detected_shape}")
else:
    selected_body_shape = st.selectbox("Body Shape", body_shape_options)

if cv_analysis:
    shoulder_value = float(cv_analysis["shoulder_cm"])
    waist_value = float(cv_analysis["waist_cm"])
    hip_value = float(cv_analysis["hip_cm"])
    ratio_value = float(cv_analysis["hip_waist_ratio"])
else:
    c1, c2, c3 = st.columns(3)
    with c1:
        shoulder_value = st.number_input("Shoulder (cm)", 20.0, 70.0, 38.0, 0.5)
    with c2:
        waist_value = st.number_input("Waist (cm)", 20.0, 80.0, 30.0, 0.5)
    with c3:
        hip_value = st.number_input("Hip (cm)", 20.0, 90.0, 40.0, 0.5)
    ratio_value = hip_value / waist_value if waist_value > 0 else 0

# Get categories from the trained model.
def options(feature, fallback):
    values = get_model_categories(feature)
    return values if values else fallback

colour_palette_options = options("Colour_Palette", ["Warm", "Cool", "Neutral"])
garment_type_options = options("Garment_Type", ["Dress", "Top", "Trousers", "Skirt", "Outfit"])
garment_colour_options = options("Garment_Colour", ["Black", "White", "Blue", "Red", "Green"])
sleeve_options = options("Sleeve", ["Sleeveless", "Short", "3/4", "Long"])
neckline_options = options("Neckline", ["V-Neck", "Round", "Boat-Neck", "Square"])
pattern_options = options("Pattern", ["Plain", "Floral", "Striped", "Checked"])
fabric_options = options("Fabric", ["Cotton", "Rayon", "Silk", "Denim", "Polyester"])
occasion_options = options("Occasion", ["Casual", "Party", "Formal", "Office"])

p1, p2 = st.columns(2)
with p1:
    colour_palette = st.selectbox("Colour Palette", colour_palette_options)
with p2:
    occasion = st.selectbox("Occasion", occasion_options)

st.subheader("👚 Garment Preferences")

g1, g2, g3 = st.columns(3)
with g1:
    garment_type = st.selectbox("Garment Type", garment_type_options)
with g2:
    garment_colour = st.selectbox("Garment Colour", garment_colour_options)
with g3:
    sleeve = st.selectbox("Sleeve", sleeve_options)

g4, g5, g6 = st.columns(3)
with g4:
    neckline = st.selectbox("Neckline", neckline_options)
with g5:
    pattern = st.selectbox("Pattern", pattern_options)
with g6:
    fabric = st.selectbox("Fabric", fabric_options)

st.divider()

# ------------------------------------------------------------
# RECOMMENDATIONS
# ------------------------------------------------------------

if st.button("✨ Generate My Style Recommendations", type="primary", use_container_width=True):
    input_data = pd.DataFrame([{
        "Body_Shape": selected_body_shape,
        "Shoulder_cm": shoulder_value,
        "Waist_cm": waist_value,
        "Hip_cm": hip_value,
        "Hip_Waist_Ratio": ratio_value,
        "Colour_Palette": colour_palette,
        "Garment_Type": garment_type,
        "Garment_Colour": garment_colour,
        "Sleeve": sleeve,
        "Neckline": neckline,
        "Pattern": pattern,
        "Fabric": fabric,
        "Occasion": occasion
    }])

    try:
        probabilities = model.predict_proba(input_data)[0]
        classes = model.classes_
        top_indices = np.argsort(probabilities)[-3:][::-1]

        st.header("✨ Your Top 3 Style Recommendations")

        for rank, index in enumerate(top_indices, start=1):
            style_name = classes[index]
            probability = probabilities[index]
            st.write(f"### #{rank}  {style_name}")
            st.progress(float(probability), text=f"Model confidence: {probability:.1%}")

        st.subheader("🔎 Analysis Summary")
        st.info(
            f"Body Shape: {selected_body_shape}\n\n"
            f"Hip/Waist Ratio: {ratio_value:.2f}\n\n"
            f"Occasion: {occasion}\n\n"
            f"Garment: {garment_type}\n\n"
            "The selected profile and garment attributes were passed to the StyleMate Random Forest model."
        )

    except Exception as e:
        st.error("Something went wrong while generating the recommendation.")
        st.exception(e)

st.caption("StyleMate • Computer Vision + Machine Learning")
