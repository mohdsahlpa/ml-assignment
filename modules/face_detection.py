"""Application 1: Face Detection Application using OpenCV Haar Cascades."""

import io
import time
from typing import Any

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from utils.sample_generator import generate_sample_portrait


@st.cache_resource
def load_cascades() -> dict[str, cv2.CascadeClassifier]:
    """Loads OpenCV Haar Cascade classifiers."""
    base_path = cv2.data.haarcascades
    return {
        "face": cv2.CascadeClassifier(base_path + "haarcascade_frontalface_default.xml"),
        "face_alt": cv2.CascadeClassifier(base_path + "haarcascade_frontalface_alt2.xml"),
        "eye": cv2.CascadeClassifier(base_path + "haarcascade_eye.xml"),
        "smile": cv2.CascadeClassifier(base_path + "haarcascade_smile.xml"),
    }


def detect_faces_and_features(
    rgb_image: np.ndarray,
    scale_factor: float = 1.1,
    min_neighbors: int = 4,
    min_size: int = 35,
    detect_eyes: bool = True,
    detect_smiles: bool = False,
) -> tuple[np.ndarray, list[dict[str, Any]], float]:
    """Detects faces, eyes, and smiles on an RGB image."""
    start_time = time.perf_counter()
    cascades = load_cascades()

    annotated = rgb_image.copy()
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    faces = cascades["face"].detectMultiScale(
        gray_eq,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_size, min_size),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    if len(faces) == 0:
        faces = cascades["face_alt"].detectMultiScale(
            gray_eq,
            scaleFactor=scale_factor,
            minNeighbors=max(2, min_neighbors - 1),
            minSize=(min_size, min_size),
        )

    detections: list[dict[str, Any]] = []

    for idx, (x, y, w, h) in enumerate(faces, start=1):
        face_roi_gray = gray_eq[y : y + h, x : x + w]
        face_crop_rgb = rgb_image[y : y + h, x : x + w].copy()

        eyes_list: list[tuple[int, int, int, int]] = []
        smiles_list: list[tuple[int, int, int, int]] = []

        if detect_eyes and face_roi_gray.size > 0:
            upper_half = face_roi_gray[0 : int(h * 0.65), :]
            raw_eyes = cascades["eye"].detectMultiScale(
                upper_half,
                scaleFactor=1.1,
                minNeighbors=max(2, min_neighbors),
                minSize=(max(12, w // 10), max(12, h // 10)),
            )
            for ex, ey, ew, eh in raw_eyes:
                eyes_list.append((int(x + ex), int(y + ey), int(ew), int(eh)))

        if detect_smiles and face_roi_gray.size > 0:
            lower_half_y = int(h * 0.50)
            lower_half = face_roi_gray[lower_half_y:h, :]
            raw_smiles = cascades["smile"].detectMultiScale(
                lower_half,
                scaleFactor=1.3,
                minNeighbors=max(8, min_neighbors * 3),
                minSize=(max(20, w // 4), max(10, h // 8)),
            )
            for sx, sy, sw, sh in raw_smiles:
                smiles_list.append((int(x + sx), int(y + lower_half_y + sy), int(sw), int(sh)))

        # Face bounding box (Green)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(
            annotated,
            f"Face {idx}",
            (x, max(15, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 0),
            2,
        )

        # Eyes (Blue)
        for ex, ey, ew, eh in eyes_list:
            cv2.rectangle(annotated, (ex, ey), (ex + ew, ey + eh), (0, 120, 255), 2)

        # Smile (Red)
        for sx, sy, sw, sh in smiles_list:
            cv2.rectangle(annotated, (sx, sy), (sx + sw, sy + sh), (220, 50, 50), 2)

        detections.append(
            {
                "id": idx,
                "bbox": (int(x), int(y), int(w), int(h)),
                "eyes_count": len(eyes_list),
                "smile_detected": len(smiles_list) > 0,
                "crop": face_crop_rgb,
            }
        )

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
    return annotated, detections, elapsed_ms


def render_face_detection_app() -> None:
    """Renders the Face Detection interface."""
    st.title("1. Face Detection Application")
    st.write("Detect faces in images using OpenCV Haar Cascades via camera input, file upload, or sample images.")

    st.subheader("Detection Settings")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        scale_factor = st.slider("Scale Factor", 1.05, 1.40, 1.10, 0.05)
    with c2:
        min_neighbors = st.slider("Min Neighbors", 1, 10, 4, 1)
    with c3:
        min_size = st.slider("Min Face Size (px)", 20, 120, 35, 5)
    with c4:
        detect_eyes = st.checkbox("Detect Eyes", value=True)
        detect_smiles = st.checkbox("Detect Smile", value=False)

    st.divider()

    input_mode = st.radio(
        "Input Source:",
        options=["Upload Image", "Camera Input", "Sample Image"],
        horizontal=True,
        key="face_input_channel",
    )

    rgb_input = None

    if input_mode == "Camera Input":
        cam_file = st.camera_input("Take a photo")
        if cam_file is not None:
            rgb_input = np.array(Image.open(cam_file).convert("RGB"))

    elif input_mode == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file is not None:
            rgb_input = np.array(Image.open(uploaded_file).convert("RGB"))

    else:
        preset = st.selectbox("Sample Scene:", ["Single Portrait", "Two Faces"])
        preset_key = "group" if preset == "Two Faces" else "single"
        rgb_input = generate_sample_portrait(preset=preset_key)

    if rgb_input is None:
        st.info("Please upload an image or take a photo to run face detection.")
        return

    annotated_rgb, detections, elapsed_ms = detect_faces_and_features(
        rgb_image=rgb_input,
        scale_factor=scale_factor,
        min_neighbors=min_neighbors,
        min_size=min_size,
        detect_eyes=detect_eyes,
        detect_smiles=detect_smiles,
    )

    st.write(f"**Faces Detected:** {len(detections)} | **Processing Time:** {elapsed_ms} ms")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(rgb_input, use_container_width=True)

    with col2:
        st.subheader("Detection Result")
        st.image(annotated_rgb, use_container_width=True)

        buf = io.BytesIO()
        Image.fromarray(annotated_rgb).save(buf, format="PNG")
        st.download_button(
            label="Download Processed Image",
            data=buf.getvalue(),
            file_name="detected_faces.png",
            mime="image/png",
        )

    if detections:
        st.subheader("Detected Face Regions")
        cols = st.columns(min(4, len(detections)))
        for idx, det in enumerate(detections):
            x, y, w, h = det["bbox"]
            with cols[idx % len(cols)]:
                st.image(det["crop"], caption=f"Face {det['id']} ({w}x{h}px)", use_container_width=True)
                st.caption(f"Position: x={x}, y={y} | Eyes: {det['eyes_count']}")
