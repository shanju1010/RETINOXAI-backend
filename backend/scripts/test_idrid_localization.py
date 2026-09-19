from pathlib import Path

import cv2
import pandas as pd

from backend.services.retinal_structure import (
    detect_optic_disc,
    estimate_fovea,
)


ROOT = Path(
    r"C:\retinoxai-prototype\data\IDRiD\C. Localization"
)

IMAGE_DIR = (
    ROOT
    / "1. Original Images"
    / "a. Training Set"
)

OD_CSV = (
    ROOT
    / "2. Groundtruths"
    / "1. Optic Disc Center Location"
    / "a. IDRiD_OD_Center_Training Set_Markups.csv"
)

FOVEA_CSV = (
    ROOT
    / "2. Groundtruths"
    / "2. Fovea Center Location"
    / "IDRiD_Fovea_Center_Training Set_Markups.csv"
)

OUTPUT_DIR = Path(
    r"C:\retinoxai-prototype\backend\outputs"
)


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load ground-truth CSVs
    od_df = pd.read_csv(
        OD_CSV,
        usecols=[
            "Image No",
            "X- Coordinate",
            "Y - Coordinate",
        ],
    )

    fovea_df = pd.read_csv(
        FOVEA_CSV,
        usecols=[
            "Image No",
            "X- Coordinate",
            "Y - Coordinate",
        ],
    )

    # Use first training image for quick verification
    image_id = str(
        od_df.iloc[0]["Image No"]
    ).strip()

    od_row = od_df[
        od_df["Image No"].astype(str).str.strip()
        == image_id
    ].iloc[0]

    fovea_row = fovea_df[
        fovea_df["Image No"].astype(str).str.strip()
        == image_id
    ].iloc[0]

    image_path = IMAGE_DIR / f"{image_id}.jpg"

    print("\n==========================================")
    print("RETINOXAI - IDRiD LOCALIZATION TEST")
    print("==========================================")

    print(f"\nImage: {image_id}")
    print(f"Path : {image_path}")

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image: {image_path}"
        )

    print(
        f"Image size: "
        f"{image.shape[1]} x {image.shape[0]}"
    )

    # -------------------------------------------------
    # Ground truth
    # -------------------------------------------------

    gt_od_x = float(
        od_row["X- Coordinate"]
    )
    gt_od_y = float(
        od_row["Y - Coordinate"]
    )

    gt_fovea_x = float(
        fovea_row["X- Coordinate"]
    )
    gt_fovea_y = float(
        fovea_row["Y - Coordinate"]
    )

    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------

    optic_disc = detect_optic_disc(
        image
    )

    fovea = estimate_fovea(
        image,
        optic_disc,
    )

    pred_od_x = optic_disc.center_x
    pred_od_y = optic_disc.center_y

    pred_fovea_x = fovea.center_x
    pred_fovea_y = fovea.center_y

    # -------------------------------------------------
    # Errors
    # -------------------------------------------------

    od_error = (
        (
            pred_od_x - gt_od_x
        ) ** 2
        +
        (
            pred_od_y - gt_od_y
        ) ** 2
    ) ** 0.5

    fovea_error = (
        (
            pred_fovea_x - gt_fovea_x
        ) ** 2
        +
        (
            pred_fovea_y - gt_fovea_y
        ) ** 2
    ) ** 0.5

    # -------------------------------------------------
    # Print results
    # -------------------------------------------------

    print("\nOPTIC DISC")
    print("----------")

    print(
        f"Ground truth : "
        f"({gt_od_x:.0f}, {gt_od_y:.0f})"
    )

    print(
        f"Predicted    : "
        f"({pred_od_x}, {pred_od_y})"
    )

    print(
        f"Localization error: "
        f"{od_error:.2f} pixels"
    )

    print(
        f"Detection confidence: "
        f"{optic_disc.confidence:.2f}"
    )

    print("\nFOVEA")
    print("-----")

    print(
        f"Ground truth : "
        f"({gt_fovea_x:.0f}, {gt_fovea_y:.0f})"
    )

    print(
        f"Predicted    : "
        f"({pred_fovea_x}, {pred_fovea_y})"
    )

    print(
        f"Localization error: "
        f"{fovea_error:.2f} pixels"
    )

    print(
        f"Estimate confidence: "
        f"{fovea.confidence:.2f}"
    )

    # -------------------------------------------------
    # Visualization
    # -------------------------------------------------

    annotated = image.copy()

    # Ground truth optic disc = yellow
    cv2.circle(
        annotated,
        (
            int(gt_od_x),
            int(gt_od_y),
        ),
        55,
        (0, 255, 255),
        5,
    )

    cv2.putText(
        annotated,
        "GT Optic Disc",
        (
            int(gt_od_x) + 60,
            int(gt_od_y),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        3,
        cv2.LINE_AA,
    )

    # Predicted optic disc = red
    cv2.circle(
        annotated,
        (
            pred_od_x,
            pred_od_y,
        ),
        optic_disc.radius,
        (0, 0, 255),
        5,
    )

    cv2.circle(
        annotated,
        (
            pred_od_x,
            pred_od_y,
        ),
        8,
        (0, 0, 255),
        -1,
    )

    cv2.putText(
        annotated,
        "Predicted OD",
        (
            pred_od_x + 20,
            pred_od_y - 20,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        3,
        cv2.LINE_AA,
    )

    # Ground truth fovea = cyan
    cv2.drawMarker(
        annotated,
        (
            int(gt_fovea_x),
            int(gt_fovea_y),
        ),
        (255, 255, 0),
        cv2.MARKER_CROSS,
        45,
        5,
    )

    cv2.putText(
        annotated,
        "GT Fovea",
        (
            int(gt_fovea_x) + 20,
            int(gt_fovea_y) + 35,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 0),
        3,
        cv2.LINE_AA,
    )

    # Predicted fovea = blue
    cv2.drawMarker(
        annotated,
        (
            pred_fovea_x,
            pred_fovea_y,
        ),
        (255, 0, 0),
        cv2.MARKER_CROSS,
        45,
        5,
    )

    cv2.putText(
        annotated,
        "Predicted Fovea",
        (
            pred_fovea_x + 20,
            pred_fovea_y - 20,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 0, 0),
        3,
        cv2.LINE_AA,
    )

    # Header
    cv2.rectangle(
        annotated,
        (0, 0),
        (image.shape[1], 90),
        (20, 20, 20),
        -1,
    )

    cv2.putText(
        annotated,
        "RETINOXAI - IDRiD Localization",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )

    cv2.putText(
        annotated,
        f"OD Error: {od_error:.1f}px | "
        f"Fovea Error: {fovea_error:.1f}px",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    output_path = (
        OUTPUT_DIR
        / f"{image_id}_localization.png"
    )

    cv2.imwrite(
        str(output_path),
        annotated,
    )

    print("\nGenerated:")
    print(output_path)

    print(
        "\n=========================================="
    )


if __name__ == "__main__":
    main()