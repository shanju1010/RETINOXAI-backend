from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def retina_mask(image: np.ndarray) -> np.ndarray:
    """Approximate retinal field-of-view mask."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mask = np.where(gray > 10, 255, 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (31, 31),
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

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    if n > 1:
        largest = 1 + np.argmax(
            stats[1:, cv2.CC_STAT_AREA]
        )
        mask = np.where(
            labels == largest,
            255,
            0,
        ).astype(np.uint8)

    return mask


def remove_small_components(
    mask: np.ndarray,
    min_area: int,
    max_area: int | None = None,
) -> np.ndarray:

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    cleaned = np.zeros_like(mask)

    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_area:
            continue

        if max_area is not None and area > max_area:
            continue

        cleaned[labels == i] = 255

    return cleaned


def detect_microaneurysms(
    image: np.ndarray,
    roi: np.ndarray,
) -> np.ndarray:
    """
    Prototype microaneurysm candidate detection.
    Uses small dark structures in the green channel.
    """
    green = image[:, :, 1]

    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(16, 16),
    )

    enhanced = clahe.apply(green)

    blackhat = cv2.morphologyEx(
        enhanced,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (9, 9),
        ),
    )

    threshold = np.percentile(
        blackhat[roi > 0],
        99.2,
    )

    mask = np.where(
        (blackhat >= threshold) & (roi > 0),
        255,
        0,
    ).astype(np.uint8)

    mask = remove_small_components(
        mask,
        min_area=8,
        max_area=250,
    )

    return mask


def detect_hard_exudates(
    image: np.ndarray,
    roi: np.ndarray,
) -> np.ndarray:
    """
    Prototype hard-exudate candidate detection.
    Targets bright yellow/white retinal regions.
    """
    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB,
    )

    L, A, B = cv2.split(lab)

    # Bright pixels with yellow/neutral characteristics
    threshold_L = np.percentile(
        L[roi > 0],
        96,
    )

    mask = (
        (L >= threshold_L)
        & (B >= np.percentile(B[roi > 0], 65))
        & (roi > 0)
    )

    mask = (
        mask.astype(np.uint8)
        * 255
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5),
        ),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (11, 11),
        ),
    )

    return remove_small_components(
        mask,
        min_area=30,
        max_area=30000,
    )


def detect_hemorrhages(
    image: np.ndarray,
    roi: np.ndarray,
) -> np.ndarray:
    """
    Prototype hemorrhage candidate detection.
    Targets dark reddish retinal structures.
    """
    b, g, r = cv2.split(image)

    red_darkness = (
        g.astype(np.int16)
        + b.astype(np.int16)
        - 2 * r.astype(np.int16)
    )

    red_darkness = cv2.normalize(
        red_darkness,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)

    threshold = np.percentile(
        red_darkness[roi > 0],
        94,
    )

    mask = (
        (red_darkness >= threshold)
        & (roi > 0)
    ).astype(np.uint8) * 255

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5),
        ),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (9, 9),
        ),
    )

    return remove_small_components(
        mask,
        min_area=40,
        max_area=12000,
    )


def detect_soft_exudates(
    image: np.ndarray,
    roi: np.ndarray,
) -> np.ndarray:
    """
    Prototype soft-exudate candidate detection.
    Uses bright low-frequency regions.
    """
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    blurred = cv2.GaussianBlur(
        gray,
        (0, 0),
        8,
    )

    threshold = np.percentile(
        blurred[roi > 0],
        97,
    )

    mask = (
        (blurred >= threshold)
        & (roi > 0)
    ).astype(np.uint8) * 255

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (9, 9),
        ),
    )

    return remove_small_components(
        mask,
        min_area=100,
        max_area=25000,
    )


def count_components(
    mask: np.ndarray,
) -> int:
    n, _, _, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    return max(
        0,
        n - 1,
    )


def analyze_lesions(
    image: np.ndarray,
) -> dict[str, Any]:

    if image is None:
        raise ValueError(
            "Image is None."
        )

    roi = retina_mask(image)

    ma = detect_microaneurysms(
        image,
        roi,
    )

    ex = detect_hard_exudates(
        image,
        roi,
    )

    he = detect_hemorrhages(
        image,
        roi,
    )

    se = detect_soft_exudates(
        image,
        roi,
    )

    return {
        "microaneurysm": {
            "detected": bool(
                np.count_nonzero(ma)
            ),
            "candidate_count": count_components(ma),
            "mask": ma,
        },
        "hard_exudate": {
            "detected": bool(
                np.count_nonzero(ex)
            ),
            "candidate_count": count_components(ex),
            "mask": ex,
        },
        "haemorrhage": {
            "detected": bool(
                np.count_nonzero(he)
            ),
            "candidate_count": count_components(he),
            "mask": he,
        },
        "soft_exudate": {
            "detected": bool(
                np.count_nonzero(se)
            ),
            "candidate_count": count_components(se),
            "mask": se,
        },
        "roi": roi,
    }


def annotate_lesions(
    image: np.ndarray,
    result: dict[str, Any],
) -> np.ndarray:

    output = image.copy()

    # BGR colors
    overlays = [
        (
            result["microaneurysm"]["mask"],
            (0, 0, 255),
            "MA",
        ),
        (
            result["hard_exudate"]["mask"],
            (0, 255, 255),
            "EX",
        ),
        (
            result["haemorrhage"]["mask"],
            (255, 0, 255),
            "HE",
        ),
        (
            result["soft_exudate"]["mask"],
            (255, 255, 0),
            "SE",
        ),
    ]

    for mask, color, _ in overlays:

        colored = np.zeros_like(output)
        colored[:, :] = color

        output = np.where(
            mask[:, :, None] > 0,
            cv2.addWeighted(
                output,
                0.45,
                colored,
                0.55,
                0,
            ),
            output,
        )

    # Header
    cv2.rectangle(
        output,
        (0, 0),
        (output.shape[1], 80),
        (20, 20, 20),
        -1,
    )

    cv2.putText(
        output,
        "RETINOXAI - Lesion Candidate Analysis",
        (25, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    text = (
        f"MA: {result['microaneurysm']['candidate_count']}   "
        f"EX: {result['hard_exudate']['candidate_count']}   "
        f"HE: {result['haemorrhage']['candidate_count']}   "
        f"SE: {result['soft_exudate']['candidate_count']}"
    )

    cv2.putText(
        output,
        text,
        (25, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output