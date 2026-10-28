"""Sample portrait generator and styling helpers for the Streamlit Dual-App Suite."""


import cv2
import numpy as np


def _draw_shaded_realistic_face(
    canvas: np.ndarray,
    center: tuple[int, int],
    scale: float = 1.0,
    smiling: bool = True,
) -> None:
    """Draws a realistic Haar-cascade-compatible shaded human face onto a BGR canvas.

    Haar cascades rely on regional luminance contrasts:
    - Darker eye sockets relative to forehead and nose bridge
    - Brighter nose bridge and cheekbones
    - Distinct pupil/iris contrast and mouth shadow line
    """
    cx, cy = center
    head_w = int(95 * scale)
    head_h = int(125 * scale)

    # Hair / crown background shadow
    cv2.ellipse(
        canvas,
        (cx, cy - int(18 * scale)),
        (int(head_w * 1.06), int(head_h * 1.04)),
        0,
        0,
        360,
        (35, 40, 48),
        -1,
    )

    # Base skin tone oval with vertical luminance gradient
    for dy in range(-head_h, head_h):
        ratio = (dy + head_h) / max(1, (2 * head_h))
        half_x = int(head_w * np.sqrt(max(0.0, 1.0 - (dy / head_h) ** 2)))
        if half_x <= 0:
            continue
        # Forehead & cheeks slightly brighter, jaw slightly shaded
        shade = 1.0 - 0.18 * abs(ratio - 0.42)
        b = int(np.clip(158 * shade, 90, 235))
        g = int(np.clip(186 * shade, 110, 240))
        r = int(np.clip(218 * shade, 130, 250))
        cv2.line(canvas, (cx - half_x, cy + dy), (cx + half_x, cy + dy), (b, g, r), 1)

    # Brighter forehead & cheek highlight zones (crucial for Haar contrast)
    cv2.ellipse(
        canvas,
        (cx, cy - int(42 * scale)),
        (int(58 * scale), int(24 * scale)),
        0,
        0,
        360,
        (175, 205, 235),
        -1,
    )
    for side in (-1, 1):
        cv2.ellipse(
            canvas,
            (cx + side * int(42 * scale), cy + int(12 * scale)),
            (int(26 * scale), int(18 * scale)),
            0,
            0,
            360,
            (172, 202, 232),
            -1,
        )

    # Darker eye socket regions (Haar horizontal feature: dark eyes vs bright bridge/cheeks)
    eye_y = cy - int(14 * scale)
    eye_dx = int(34 * scale)
    for side in (-1, 1):
        ex = cx + side * eye_dx
        # Socket shadow
        cv2.ellipse(
            canvas,
            (ex, eye_y),
            (int(22 * scale), int(13 * scale)),
            0,
            0,
            360,
            (95, 112, 135),
            -1,
        )
        # Eyebrow arch
        cv2.ellipse(
            canvas,
            (ex, eye_y - int(14 * scale)),
            (int(22 * scale), int(5 * scale)),
            0,
            180,
            360,
            (40, 46, 56),
            max(2, int(4 * scale)),
        )
        # Sclera (white of eye)
        cv2.ellipse(
            canvas,
            (ex, eye_y),
            (int(14 * scale), int(7 * scale)),
            0,
            0,
            360,
            (225, 230, 235),
            -1,
        )
        # Iris & Pupil
        cv2.circle(canvas, (ex, eye_y), max(3, int(6 * scale)), (52, 62, 78), -1)
        cv2.circle(canvas, (ex, eye_y), max(2, int(3 * scale)), (18, 20, 24), -1)
        # Specular highlight
        cv2.circle(
            canvas,
            (ex - max(1, int(2 * scale)), eye_y - max(1, int(2 * scale))),
            max(1, int(1.5 * scale)),
            (250, 250, 255),
            -1,
        )

    # Bright nose bridge between eyes (Haar vertical feature: dark-bright-dark)
    cv2.ellipse(
        canvas,
        (cx, cy + int(4 * scale)),
        (int(10 * scale), int(28 * scale)),
        0,
        0,
        360,
        (185, 212, 242),
        -1,
    )
    # Nostril shadow at nose base
    cv2.ellipse(
        canvas,
        (cx, cy + int(30 * scale)),
        (int(16 * scale), int(6 * scale)),
        0,
        0,
        360,
        (110, 128, 152),
        -1,
    )

    # Mouth region
    mouth_y = cy + int(56 * scale)
    if smiling:
        # Smile shadow & teeth contrast
        cv2.ellipse(
            canvas,
            (cx, mouth_y - int(4 * scale)),
            (int(30 * scale), int(14 * scale)),
            0,
            10,
            170,
            (70, 75, 115),
            -1,
        )
        cv2.ellipse(
            canvas,
            (cx, mouth_y - int(4 * scale)),
            (int(24 * scale), int(8 * scale)),
            0,
            15,
            165,
            (235, 240, 245),
            -1,
        )
        cv2.ellipse(
            canvas,
            (cx, mouth_y - int(4 * scale)),
            (int(30 * scale), int(14 * scale)),
            0,
            10,
            170,
            (65, 70, 110),
            max(2, int(3 * scale)),
        )
    else:
        cv2.ellipse(
            canvas,
            (cx, mouth_y),
            (int(24 * scale), int(6 * scale)),
            0,
            0,
            360,
            (85, 95, 130),
            -1,
        )
        cv2.line(
            canvas,
            (cx - int(24 * scale), mouth_y),
            (cx + int(24 * scale), mouth_y),
            (45, 50, 70),
            max(2, int(3 * scale)),
        )

    # Soft Gaussian blend so gradients look natural to Haar filters
    roi_y1 = max(0, cy - int(head_h * 1.15))
    roi_y2 = min(canvas.shape[0], cy + int(head_h * 1.15))
    roi_x1 = max(0, cx - int(head_w * 1.15))
    roi_x2 = min(canvas.shape[1], cx + int(head_w * 1.15))
    if roi_y2 > roi_y1 and roi_x2 > roi_x1:
        canvas[roi_y1:roi_y2, roi_x1:roi_x2] = cv2.GaussianBlur(
            canvas[roi_y1:roi_y2, roi_x1:roi_x2], (5, 5), 1.2
        )


def generate_sample_portrait(preset: str = "single") -> np.ndarray:
    """Returns an RGB numpy array of a sample portrait scene for testing face detection."""
    height, width = 480, 640
    # Warm studio slate background
    canvas = np.full((height, width, 3), (38, 44, 52), dtype=np.uint8)

    if preset == "group":
        _draw_shaded_realistic_face(canvas, (175, 245), scale=0.88, smiling=True)
        _draw_shaded_realistic_face(canvas, (455, 240), scale=0.92, smiling=False)
    else:
        _draw_shaded_realistic_face(canvas, (320, 240), scale=1.15, smiling=True)

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
