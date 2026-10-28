"""Instance 1: Mobile-First Face Detection Application using OpenCV Haar Cascades."""

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
    """Loads and caches OpenCV Haar Cascade classifiers."""
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
    detect_smiles: bool = True,
    privacy_mode: str = "None",
) -> tuple[np.ndarray, list[dict[str, Any]], float]:
    """Runs multi-cascade face, eye, and smile detection on an RGB image.

    Returns:
        annotated_rgb: Annotated RGB image array
        detections: List of metadata dicts per detected face
        elapsed_ms: Detection execution time in milliseconds
    """
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
    img_h, img_w = rgb_image.shape[:2]
    total_pixels = max(1, img_h * img_w)

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

        if privacy_mode == "Gaussian Blur":
            ksize = max(15, (w // 3) | 1)
            annotated[y : y + h, x : x + w] = cv2.GaussianBlur(
                annotated[y : y + h, x : x + w], (ksize, ksize), 30
            )
        elif privacy_mode == "Pixelate Mosaic":
            roi = annotated[y : y + h, x : x + w]
            small = cv2.resize(roi, (max(4, w // 12), max(4, h // 12)), interpolation=cv2.INTER_LINEAR)
            annotated[y : y + h, x : x + w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        # Draw high-contrast Signal Amber tactical bounding box (RGB: 245, 158, 11)
        box_color = (245, 158, 11)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)

        # Corner brackets for mobile HUD aesthetic
        corner_len = max(10, min(w, h) // 5)
        for cx, cy, dx, dy in [
            (x, y, 1, 1),
            (x + w, y, -1, 1),
            (x, y + h, 1, -1),
            (x + w, y + h, -1, -1),
        ]:
            cv2.line(annotated, (cx, cy), (cx + dx * corner_len, cy), (251, 191, 36), 4)
            cv2.line(annotated, (cx, cy), (cx, cy + dy * corner_len), (251, 191, 36), 4)

        # Label banner
        label = f"FACE #{idx} [{w}x{h}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        banner_y1 = max(0, y - th - 10)
        cv2.rectangle(annotated, (x, banner_y1), (x + tw + 12, y), box_color, -1)
        cv2.putText(
            annotated,
            label,
            (x + 6, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (15, 20, 25),
            2,
            cv2.LINE_AA,
        )

        # Draw Eyes in Emerald (RGB: 16, 185, 129)
        for ex, ey, ew, eh in eyes_list:
            center_pt = (ex + ew // 2, ey + eh // 2)
            radius = max(4, (ew + eh) // 4)
            cv2.circle(annotated, center_pt, radius, (16, 185, 129), 2)

        # Draw Smiles in Crimson Coral (RGB: 239, 68, 68)
        for sx, sy, sw, sh in smiles_list:
            cv2.rectangle(annotated, (sx, sy), (sx + sw, sy + sh), (239, 68, 68), 2)

        area_pct = round((w * h / total_pixels) * 100, 2)
        detections.append(
            {
                "id": idx,
                "bbox": (int(x), int(y), int(w), int(h)),
                "eyes_count": len(eyes_list),
                "smile_detected": len(smiles_list) > 0,
                "area_pct": area_pct,
                "crop": face_crop_rgb,
            }
        )

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
    return annotated, detections, elapsed_ms


def _convert_rgb_to_png_bytes(rgb_img: np.ndarray) -> bytes:
    """Encodes an RGB numpy array into PNG bytes for download."""
    pil_img = Image.fromarray(rgb_img)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def render_face_detection_app() -> None:
    """Renders the Face Detection Mobile Application interface."""
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-tag">INSTANCE 01 // COMPUTER VISION MOBILEINTERFACE</div>
            <h1 class="hero-title">TACTICAL FACE DETECTION STUDIO</h1>
            <p class="hero-subtitle">
                Mobile-ready biometric region detector powered by OpenCV Haar Feature-based Cascade Classifiers.
                Capture directly from your mobile camera, upload a photo, or inspect calibrated test portraits.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("⚙️ DETECTION ENGINE PARAMETERS & PRIVACY CONTROLS", expanded=True):
        c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.4])
        with c1:
            scale_factor = st.slider(
                "Scale Factor",
                min_value=1.05,
                max_value=1.40,
                value=1.10,
                step=0.05,
                help="Controls how much the image size is reduced at each image scale pyramid step.",
            )
        with c2:
            min_neighbors = st.slider(
                "Min Neighbors",
                min_value=1,
                max_value=10,
                value=4,
                step=1,
                help="Higher values result in fewer false positives but require stronger facial contrast.",
            )
        with c3:
            min_size = st.slider(
                "Min Face Size (px)",
                min_value=20,
                max_value=120,
                value=35,
                step=5,
                help="Minimum bounding box dimension in pixels.",
            )
        with c4:
            privacy_mode = st.selectbox(
                "Privacy Filter Mode",
                options=["None", "Gaussian Blur", "Pixelate Mosaic"],
                index=0,
                help="Optionally anonymize detected faces before exporting.",
            )
            fcol1, fcol2 = st.columns(2)
            with fcol1:
                detect_eyes = st.checkbox("Eye Tracking", value=True)
            with fcol2:
                detect_smiles = st.checkbox("Smile Sensor", value=True)

    input_mode = st.radio(
        "SELECT INPUT CHANNEL",
        options=[
            "🖼️ Built-in Sample Portraits",
            "📱 Mobile Camera Snapshot",
            "📂 Upload Photo from Device",
        ],
        horizontal=True,
    )

    rgb_input = None
    source_label = ""

    if input_mode == "📱 Mobile Camera Snapshot":
        st.info(
            "📱 **Mobile Ready:** Tap **Take Photo** below to capture a frame from your phone's front or rear camera."
        )
        cam_file = st.camera_input("Capture Face Snapshot")
        if cam_file is not None:
            pil_image = Image.open(cam_file).convert("RGB")
            rgb_input = np.array(pil_image)
            source_label = "Live Camera Capture"

    elif input_mode == "📂 Upload Photo from Device":
        uploaded_file = st.file_uploader(
            "Choose a portrait or group photo (JPG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
        )
        if uploaded_file is not None:
            pil_image = Image.open(uploaded_file).convert("RGB")
            rgb_input = np.array(pil_image)
            source_label = f"Uploaded File ({uploaded_file.name})"

    else:
        preset_col, info_col = st.columns([1, 2])
        with preset_col:
            preset_choice = st.selectbox(
                "Select Test Scene",
                options=["Single Portrait (Smiling)", "Dual Group Portrait"],
            )
        with info_col:
            st.caption(
                "Procedurally shaded synthetic portraits calibrated for Haar luminance contrast (dark eye sockets, bright nasal bridge, ocular highlights). Switch to **Mobile Camera** or **Upload Photo** anytime to test real photos."
            )
        preset_key = "group" if "Dual" in preset_choice else "single"
        rgb_input = generate_sample_portrait(preset=preset_key)
        source_label = f"Calibrated Sample ({preset_choice})"

    if rgb_input is None:
        st.warning("Waiting for image input. Capture a photo or upload an image above to begin analysis.")
        return

    annotated_rgb, detections, elapsed_ms = detect_faces_and_features(
        rgb_image=rgb_input,
        scale_factor=scale_factor,
        min_neighbors=min_neighbors,
        min_size=min_size,
        detect_eyes=detect_eyes,
        detect_smiles=detect_smiles,
        privacy_mode=privacy_mode,
    )

    total_eyes = sum(d["eyes_count"] for d in detections)
    total_smiles = sum(1 for d in detections if d["smile_detected"])
    img_h, img_w = rgb_input.shape[:2]

    # Telemetry Metrics Bar
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("FACES DETECTED", f"{len(detections)}")
    m2.metric("OCULAR REGIONS", f"{total_eyes}")
    m3.metric("SMILE SIGNATURES", f"{total_smiles}")
    m4.metric("INFERENCE LATENCY", f"{elapsed_ms} ms")

    st.markdown("---")

    # Side-by-side Viewport (stacks cleanly on mobile viewports)
    col_orig, col_proc = st.columns(2)
    with col_orig:
        st.markdown(f"#### 01 // RAW INPUT FRAME (`{img_w}×{img_h}px`)")
        st.image(rgb_input, caption=source_label, use_container_width=True)

    with col_proc:
        st.markdown("#### 02 // ANNOTATED BIOMETRIC HUD")
        st.image(
            annotated_rgb,
            caption=f"Amber: Face | Emerald: Eyes | Coral: Smile ({elapsed_ms} ms)",
            use_container_width=True,
        )

        png_bytes = _convert_rgb_to_png_bytes(annotated_rgb)
        st.download_button(
            label="⬇️ DOWNLOAD ANNOTATED IMAGE (PNG)",
            data=png_bytes,
            file_name="face_detection_output.png",
            mime="image/png",
            use_container_width=True,
        )

    # Cropped Face Inspector & Region Telemetry
    st.markdown("---")
    st.markdown("#### 03 // EXTRACTED FACE ROIs & TELEMETRY LOG")
    if not detections:
        st.warning(
            "No faces detected with current thresholds. Try lowering **Min Neighbors** (e.g. to 2 or 3) or reducing **Scale Factor** (e.g. to 1.05)."
        )
    else:
        crop_cols = st.columns(min(4, len(detections)))
        for idx, det in enumerate(detections):
            col = crop_cols[idx % len(crop_cols)]
            x, y, w, h = det["bbox"]
            with col:
                st.image(det["crop"], caption=f"Face #{det['id']} ({w}×{h}px)", use_container_width=True)
                st.markdown(
                    f"""
                    - **Coords:** `x={x}, y={y}`
                    - **Frame Coverage:** `{det['area_pct']}%`
                    - **Eyes Found:** `{det['eyes_count']}`
                    - **Expression:** `{'Smiling 😊' if det['smile_detected'] else 'Neutral 😐'}`
                    """
                )
