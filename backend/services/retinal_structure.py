from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class OpticDiscResult:
    detected: bool
    center_x: int
    center_y: int
    radius: int
    confidence: float


@dataclass
class FoveaResult:
    estimated: bool
    center_x: int
    center_y: int
    confidence: float
    method: str


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    """Ensure image is a 3-channel BGR uint8 image."""
    if image is None:
        raise ValueError("Image could not be read.")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    return image


def _retina_mask(image: np.ndarray) -> np.ndarray:
    """
    Approximate the retinal field of view.
    This is a prototype mask, not a clinical field-of-view detector.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Fundus is generally much brighter than the black border.
    mask = (gray > 8).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (21, 21),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    # Keep the largest connected component.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(
            labels == largest_label,
            255,
            0,
        ).astype(np.uint8)

    return mask


def segment_vessels(image: np.ndarray) -> np.ndarray:
    """
    Prototype retinal vessel segmentation using:
    green channel -> CLAHE -> multi-scale blackhat ->
    threshold -> morphology.

    Output:
        Binary vessel mask, uint8 values {0, 255}.
    """
    image = _ensure_bgr(image)

    green = image[:, :, 1]

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced = clahe.apply(green)

    vessel_response = np.zeros_like(enhanced, dtype=np.float32)

    # Multi-scale blackhat helps reveal dark tubular structures.
    for kernel_size in (9, 15, 21):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )

        blackhat = cv2.morphologyEx(
            enhanced,
            cv2.MORPH_BLACKHAT,
            kernel,
        )

        vessel_response = np.maximum(
            vessel_response,
            blackhat.astype(np.float32),
        )

    vessel_response = cv2.normalize(
        vessel_response,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)

    # Adaptive threshold for vessel candidates.
    vessel_mask = cv2.adaptiveThreshold(
        vessel_response,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        -1,
    )

    # Remove isolated noise.
    small_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    vessel_mask = cv2.morphologyEx(
        vessel_mask,
        cv2.MORPH_OPEN,
        small_kernel,
    )

    vessel_mask = cv2.morphologyEx(
        vessel_mask,
        cv2.MORPH_CLOSE,
        small_kernel,
    )

    # Keep only pixels inside the retinal field.
    retina = _retina_mask(image)
    vessel_mask = cv2.bitwise_and(
        vessel_mask,
        retina,
    )

    # Remove very small connected components.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        vessel_mask,
        connectivity=8,
    )

    cleaned = np.zeros_like(vessel_mask)

    min_area = max(4, int(image.shape[0] * image.shape[1] * 0.000003))

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area >= min_area:
            cleaned[labels == label] = 255

    return cleaned


def detect_optic_disc(image: np.ndarray) -> OpticDiscResult:
    """
    Prototype optic-disc localization based on bright retinal structures.

    The result is explicitly treated as an estimate.
    """
    image = _ensure_bgr(image)

    green = image[:, :, 1]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    retina = _retina_mask(image)

    blurred = cv2.GaussianBlur(
        green,
        (0, 0),
        7,
    )

    valid_pixels = blurred[retina > 0]

    if valid_pixels.size == 0:
        h, w = gray.shape
        return OpticDiscResult(
            detected=False,
            center_x=w // 2,
            center_y=h // 2,
            radius=max(10, min(h, w) // 20),
            confidence=0.0,
        )

    # High-intensity candidate regions.
    threshold = float(np.percentile(valid_pixels, 99.2))

    bright = np.zeros_like(blurred)
    bright[(blurred >= threshold) & (retina > 0)] = 255

    # Join nearby bright pixels.
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (25, 25),
    )

    bright = cv2.morphologyEx(
        bright,
        cv2.MORPH_CLOSE,
        close_kernel,
    )

    bright = cv2.morphologyEx(
        bright,
        cv2.MORPH_OPEN,
        close_kernel,
    )

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bright,
        connectivity=8,
    )

    h, w = gray.shape

    best_score = -1.0
    best_candidate: tuple[int, int, int, float] | None = None

    for label in range(1, num_labels):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]

        if area <= 0:
            continue

        cx, cy = centroids[label]

        # Ignore tiny highlights.
        area_ratio = area / float(h * w)

        if area_ratio < 0.0005:
            continue

        if area_ratio > 0.08:
            continue

        # Favor reasonably compact structures.
        aspect_ratio = width / max(height, 1)

        if aspect_ratio > 3.0 or aspect_ratio < 0.33:
            continue

        mean_intensity = float(np.mean(
            blurred[labels == label]
        ))

        compactness_penalty = 1.0 / (
            1.0 + abs(np.log(max(aspect_ratio, 1e-6)))
        )

        score = (
            mean_intensity
            * np.sqrt(area)
            * compactness_penalty
        )

        if score > best_score:
            radius = int(
                max(
                    8,
                    np.sqrt(area / np.pi),
                )
            )

            best_score = score
            best_candidate = (
                int(round(cx)),
                int(round(cy)),
                radius,
                mean_intensity,
            )

    if best_candidate is None:
        # Fallback: brightest point inside retina.
        masked = np.where(
            retina > 0,
            blurred,
            0,
        )

        _, max_value, _, max_location = cv2.minMaxLoc(
            masked
        )

        cx, cy = max_location
        radius = max(10, min(h, w) // 25)

        return OpticDiscResult(
            detected=False,
            center_x=int(cx),
            center_y=int(cy),
            radius=radius,
            confidence=0.35,
        )

    cx, cy, radius, intensity = best_candidate

    confidence = float(
        np.clip(
            (intensity - 120.0) / 100.0,
            0.35,
            0.95,
        )
    )

    return OpticDiscResult(
        detected=True,
        center_x=cx,
        center_y=cy,
        radius=radius,
        confidence=confidence,
    )


def estimate_fovea(
    image: np.ndarray,
    optic_disc: OpticDiscResult,
) -> FoveaResult:
    """
    Calibrated prototype fovea localization.

    The optic-disc-to-fovea displacement was calibrated from the
    IDRiD localization training annotations.

    This is an estimation method for the prototype and is not
    a clinically validated fovea detector.
    """
    image = _ensure_bgr(image)

    h, w = image.shape[:2]

    if not optic_disc.detected:
        return FoveaResult(
            estimated=False,
            center_x=w // 2,
            center_y=h // 2,
            confidence=0.0,
            method="calibrated geometric estimate",
        )

    # ---------------------------------------------------------
    # IDRiD-calibrated normalized optic-disc -> fovea offsets
    # ---------------------------------------------------------

    RIGHT_DX = -0.30189
    RIGHT_DY = 0.03757

    LEFT_DX = 0.30667
    LEFT_DY = 0.06426

    # Determine retinal side using optic-disc position.
    if optic_disc.center_x >= w / 2:
        dx = RIGHT_DX
        dy = RIGHT_DY
        eye_side = "right"
    else:
        dx = LEFT_DX
        dy = LEFT_DY
        eye_side = "left"

    # Convert normalized displacement back to pixels.
    estimated_x = int(
        round(
            optic_disc.center_x + dx * w
        )
    )

    estimated_y = int(
        round(
            optic_disc.center_y + dy * h
        )
    )

    # Keep the estimate inside the image.
    estimated_x = int(
        np.clip(
            estimated_x,
            0,
            w - 1,
        )
    )

    estimated_y = int(
        np.clip(
            estimated_y,
            0,
            h - 1,
        )
    )

    return FoveaResult(
        estimated=True,
        center_x=estimated_x,
        center_y=estimated_y,
        confidence=0.55,
        method=(
            f"IDRiD-calibrated geometric estimate "
            f"({eye_side} eye)"
        ),
    )

def analyze_retinal_structures(
    image: np.ndarray,
) -> dict[str, Any]:
    """
    Run the complete Phase-1 retinal structure analysis.
    """
    image = _ensure_bgr(image)

    vessel_mask = segment_vessels(image)

    optic_disc = detect_optic_disc(image)

    fovea = estimate_fovea(
        image,
        optic_disc,
    )

    vessel_pixels = int(
        np.count_nonzero(vessel_mask)
    )

    retina_mask = _retina_mask(image)

    retina_pixels = int(
        np.count_nonzero(retina_mask)
    )

    vessel_density = 0.0

    if retina_pixels > 0:
        vessel_density = (
            vessel_pixels / retina_pixels
        )

    return {
        "vessels": {
            "analyzed": True,
            "candidate_pixels": vessel_pixels,
            "density": round(
                float(vessel_density),
                4,
            ),
        },
        "optic_disc": asdict(optic_disc),
        "fovea": asdict(fovea),
        "vessel_mask": vessel_mask,
        "retina_mask": retina_mask,
    }


def annotate_retinal_structures(
    image: np.ndarray,
    result: dict[str, Any],
) -> np.ndarray:
    """
    Create a human-readable annotated retinal image.
    """
    image = _ensure_bgr(image).copy()

    vessel_mask = result["vessel_mask"]

    # Vessel overlay
    vessel_overlay = np.zeros_like(image)
    vessel_overlay[:, :, 1] = vessel_mask

    image = cv2.addWeighted(
        image,
        0.78,
        vessel_overlay,
        0.32,
        0,
    )

    optic_disc = result["optic_disc"]

    if optic_disc["detected"]:
        center = (
            optic_disc["center_x"],
            optic_disc["center_y"],
        )

        radius = optic_disc["radius"]

        cv2.circle(
            image,
            center,
            radius,
            (0, 0, 255),
            3,
        )

        cv2.circle(
            image,
            center,
            4,
            (0, 0, 255),
            -1,
        )

        cv2.putText(
            image,
            "Optic disc",
            (
                center[0] + radius + 8,
                center[1],
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    fovea = result["fovea"]

    if fovea["estimated"]:
        center = (
            fovea["center_x"],
            fovea["center_y"],
        )

        cv2.drawMarker(
            image,
            center,
            (255, 0, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=24,
            thickness=3,
        )

        cv2.putText(
            image,
            "Estimated fovea",
            (
                center[0] + 10,
                center[1] - 10,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    # Header
    cv2.rectangle(
        image,
        (0, 0),
        (image.shape[1], 55),
        (20, 20, 20),
        -1,
    )

    cv2.putText(
        image,
        "RETINOXAI - Retinal Structure Analysis",
        (18, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return image


def save_structure_outputs(
    image: np.ndarray,
    result: dict[str, Any],
    output_dir: str | Path,
    prefix: str = "retinal_structure",
) -> dict[str, str]:
    """
    Save vessel mask and annotated image.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    vessel_path = output_dir / f"{prefix}_vessels.png"
    annotated_path = output_dir / f"{prefix}_annotated.png"

    cv2.imwrite(
        str(vessel_path),
        result["vessel_mask"],
    )

    annotated = annotate_retinal_structures(
        image,
        result,
    )

    cv2.imwrite(
        str(annotated_path),
        annotated,
    )

    return {
        "vessel_mask": str(vessel_path),
        "annotated_image": str(annotated_path),
    }


def make_api_result(
    result: dict[str, Any],
) -> dict[str, Any]:
    """
    Remove NumPy/OpenCV objects and create JSON-safe output.
    """
    optic_disc = result["optic_disc"]
    fovea = result["fovea"]
    vessels = result["vessels"]

    return {
        "vessels": vessels,
        "optic_disc": optic_disc,
        "fovea": fovea,
    }