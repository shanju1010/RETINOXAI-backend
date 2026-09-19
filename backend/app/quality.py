import cv2
import numpy as np


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def assess_quality(image_bgr: np.ndarray) -> dict:
    """
    Prototype fundus-image quality gate.

    Checks:
    - Sharpness / focus
    - Brightness / illumination
    - Contrast

    This is a prototype heuristic and is not a clinically
    validated image-gradability model.
    """

    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image")

    # Convert to grayscale
    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    # Reduce computation while keeping enough information
    gray = cv2.resize(
        gray,
        (512, 512),
        interpolation=cv2.INTER_AREA,
    )

    # ---------------------------------------------------------
    # 1. Sharpness
    # ---------------------------------------------------------

    sharpness = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )

    # ---------------------------------------------------------
    # 2. Brightness
    # ---------------------------------------------------------

    brightness = float(
        gray.mean()
    )

    # ---------------------------------------------------------
    # 3. Contrast
    # ---------------------------------------------------------

    contrast = float(
        gray.std()
    )

    # ---------------------------------------------------------
    # Normalize individual quality measures
    # ---------------------------------------------------------

    sharpness_score = _clamp01(
        sharpness / 250.0
    )

    brightness_score = _clamp01(
        1.0
        - abs(brightness - 115.0) / 115.0
    )

    contrast_score = _clamp01(
        contrast / 65.0
    )

    # ---------------------------------------------------------
    # Overall quality score
    # ---------------------------------------------------------

    score = (
        0.45 * sharpness_score
        + 0.30 * brightness_score
        + 0.25 * contrast_score
    )

    # ---------------------------------------------------------
    # Prototype acceptance rules
    #
    # Slightly more tolerant than the previous 0.55 cutoff,
    # but retain a hard sharpness floor to reject obviously
    # blurry images.
    # ---------------------------------------------------------

    acceptable = bool(
        score > 0.45
        and sharpness >= 50
        and brightness >= 35
        and brightness <= 215
        and contrast >= 15
    )

    # ---------------------------------------------------------
    # Reasons
    # ---------------------------------------------------------

    reasons = []

    if sharpness < 50:
        reasons.append(
            "Image may be blurry or out of focus"
        )

    if brightness < 35:
        reasons.append(
            "Image may be too dark"
        )

    elif brightness > 215:
        reasons.append(
            "Image may be too bright"
        )

    if contrast < 15:
        reasons.append(
            "Image may have low contrast"
        )

    if score <= 0.45 and not reasons:
        reasons.append(
            "Overall image quality is below the prototype threshold"
        )

    # ---------------------------------------------------------
    # Human-readable status
    # ---------------------------------------------------------

    status = (
        "acceptable"
        if acceptable
        else "retake"
    )

    return {
        "acceptable": acceptable,
        "status": status,
        "quality_score": round(
            float(score),
            3,
        ),
        "sharpness": round(
            sharpness,
            2,
        ),
        "brightness": round(
            brightness,
            2,
        ),
        "contrast": round(
            contrast,
            2,
        ),
        "reasons": reasons,
    }


def enhance_fundus(
    image_bgr: np.ndarray,
) -> np.ndarray:
    """
    Lightweight CLAHE enhancement for prototype inference.
    """

    lab = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2LAB,
    )

    l, a, b = cv2.split(
        lab
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    l = clahe.apply(
        l
    )

    enhanced = cv2.merge(
        (l, a, b)
    )

    return cv2.cvtColor(
        enhanced,
        cv2.COLOR_LAB2BGR,
    )