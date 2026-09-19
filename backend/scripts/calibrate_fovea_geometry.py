from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from backend.services.retinal_structure import detect_optic_disc


ROOT = Path(
    r"C:\retinoxai-prototype\data\IDRiD\C. Localization"
)

TRAIN_IMAGE_DIR = (
    ROOT
    / "1. Original Images"
    / "a. Training Set"
)

TEST_IMAGE_DIR = (
    ROOT
    / "1. Original Images"
    / "b. Testing Set"
)

OD_TRAIN_CSV = (
    ROOT
    / "2. Groundtruths"
    / "1. Optic Disc Center Location"
    / "a. IDRiD_OD_Center_Training Set_Markups.csv"
)

OD_TEST_CSV = (
    ROOT
    / "2. Groundtruths"
    / "1. Optic Disc Center Location"
    / "b. IDRiD_OD_Center_Testing Set_Markups.csv"
)

FOVEA_TRAIN_CSV = (
    ROOT
    / "2. Groundtruths"
    / "2. Fovea Center Location"
    / "IDRiD_Fovea_Center_Training Set_Markups.csv"
)

FOVEA_TEST_CSV = (
    ROOT
    / "2. Groundtruths"
    / "2. Fovea Center Location"
    / "IDRiD_Fovea_Center_Testing Set_Markups.csv"
)


def load_points(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=[
            "Image No",
            "X- Coordinate",
            "Y - Coordinate",
        ],
    ).copy()

    df["Image No"] = (
        df["Image No"]
        .astype(str)
        .str.strip()
    )

    df["X- Coordinate"] = pd.to_numeric(
        df["X- Coordinate"],
        errors="coerce",
    )

    df["Y - Coordinate"] = pd.to_numeric(
        df["Y - Coordinate"],
        errors="coerce",
    )

    return df.dropna()


def calculate_geometry(
    od_df: pd.DataFrame,
    fovea_df: pd.DataFrame,
) -> dict[str, tuple[float, float]]:

    merged = od_df.merge(
        fovea_df,
        on="Image No",
        suffixes=("_od", "_fovea"),
    )

    right_side = []
    left_side = []

    for _, row in merged.iterrows():

        # Localization image dimensions
        width = 4288.0
        height = 2848.0

        od_x = float(row["X- Coordinate_od"])
        od_y = float(row["Y - Coordinate_od"])

        fovea_x = float(row["X- Coordinate_fovea"])
        fovea_y = float(row["Y - Coordinate_fovea"])

        dx = (fovea_x - od_x) / width
        dy = (fovea_y - od_y) / height

        if od_x >= width / 2:
            right_side.append((dx, dy))
        else:
            left_side.append((dx, dy))

    right = np.array(right_side)
    left = np.array(left_side)

    result = {}

    if len(right) > 0:
        result["right"] = (
            float(np.median(right[:, 0])),
            float(np.median(right[:, 1])),
        )

    if len(left) > 0:
        result["left"] = (
            float(np.median(left[:, 0])),
            float(np.median(left[:, 1])),
        )

    return result


def estimate_from_geometry(
    od_x: float,
    od_y: float,
    width: int,
    height: int,
    geometry: dict[str, tuple[float, float]],
) -> tuple[int, int]:

    if od_x >= width / 2:
        dx, dy = geometry["right"]
    else:
        dx, dy = geometry["left"]

    x = int(
        round(
            od_x + dx * width
        )
    )

    y = int(
        round(
            od_y + dy * height
        )
    )

    x = int(
        np.clip(
            x,
            0,
            width - 1,
        )
    )

    y = int(
        np.clip(
            y,
            0,
            height - 1,
        )
    )

    return x, y


def main() -> None:

    print(
        "\n=========================================="
    )
    print(
        "RETINOXAI - FOVEA GEOMETRY CALIBRATION"
    )
    print(
        "=========================================="
    )

    # -------------------------------------------------
    # Load training annotations
    # -------------------------------------------------

    od_train = load_points(
        OD_TRAIN_CSV
    )

    fovea_train = load_points(
        FOVEA_TRAIN_CSV
    )

    geometry = calculate_geometry(
        od_train,
        fovea_train,
    )

    print("\nLEARNED TRAINING GEOMETRY")
    print("-------------------------")

    for side, values in geometry.items():
        dx, dy = values

        print(
            f"{side}: "
            f"dx={dx:.5f}, "
            f"dy={dy:.5f}"
        )

    # -------------------------------------------------
    # Test set
    # -------------------------------------------------

    od_test = load_points(
        OD_TEST_CSV
    )

    fovea_test = load_points(
        FOVEA_TEST_CSV
    )

    merged = od_test.merge(
        fovea_test,
        on="Image No",
        suffixes=("_od", "_fovea"),
    )

    od_errors = []
    fovea_errors = []

    print(
        "\nTESTING ON IDRiD TEST SET"
    )
    print(
        "-------------------------"
    )

    for _, row in merged.iterrows():

        image_id = str(
            row["Image No"]
        ).strip()

        # Testing images are IDRiD_001.jpg etc.
        image_path = (
            TEST_IMAGE_DIR
            / f"{image_id}.jpg"
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        height, width = image.shape[:2]

        # Ground truth OD
        gt_od_x = float(
            row["X- Coordinate_od"]
        )

        gt_od_y = float(
            row["Y - Coordinate_od"]
        )

        # Ground truth fovea
        gt_fovea_x = float(
            row["X- Coordinate_fovea"]
        )

        gt_fovea_y = float(
            row["Y - Coordinate_fovea"]
        )

        # Current optic disc detector
        pred_od = detect_optic_disc(
            image
        )

        pred_od_x = pred_od.center_x
        pred_od_y = pred_od.center_y

        od_error = float(
            np.hypot(
                pred_od_x - gt_od_x,
                pred_od_y - gt_od_y,
            )
        )

        # New geometry-based fovea
        pred_fovea_x, pred_fovea_y = (
            estimate_from_geometry(
                pred_od_x,
                pred_od_y,
                width,
                height,
                geometry,
            )
        )

        fovea_error = float(
            np.hypot(
                pred_fovea_x - gt_fovea_x,
                pred_fovea_y - gt_fovea_y,
            )
        )

        od_errors.append(
            od_error
        )

        fovea_errors.append(
            fovea_error
        )

    # -------------------------------------------------
    # Results
    # -------------------------------------------------

    od_errors = np.asarray(
        od_errors
    )

    fovea_errors = np.asarray(
        fovea_errors
    )

    print(
        f"\nImages evaluated: "
        f"{len(fovea_errors)}"
    )

    print("\nOPTIC DISC")
    print("----------")

    print(
        f"Mean error   : "
        f"{od_errors.mean():.2f} px"
    )

    print(
        f"Median error : "
        f"{np.median(od_errors):.2f} px"
    )

    print("\nFOVEA")
    print("-----")

    print(
        f"Mean error   : "
        f"{fovea_errors.mean():.2f} px"
    )

    print(
        f"Median error : "
        f"{np.median(fovea_errors):.2f} px"
    )

    print(
        "\n=========================================="
    )


if __name__ == "__main__":
    main()