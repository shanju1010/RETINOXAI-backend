from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    f1_score,
)

from backend.app.model import (
    CLASS_NAMES,
    load_checkpoint,
    prepare_pil,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\retinoxai-prototype"
)

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "IDRiD"
    / "B. Disease Grading"
)

IMAGE_DIR = (
    DATA_ROOT
    / "1. Original Images"
    / "b. Testing Set"
)

LABEL_CSV = (
    DATA_ROOT
    / "2. Groundtruths"
    / "b. IDRiD_Disease Grading_Testing Labels.csv"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "backend"
    / "models"
    / "best.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "backend"
    / "outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HELPERS
# ============================================================

def calculate_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, float, int, int, int, int]:

    # Grade >= 2 = Referable DR
    true_ref = y_true >= 2
    pred_ref = y_pred >= 2

    tn = int(
        np.sum(
            (~true_ref) & (~pred_ref)
        )
    )

    fp = int(
        np.sum(
            (~true_ref) & pred_ref
        )
    )

    fn = int(
        np.sum(
            true_ref & (~pred_ref)
        )
    )

    tp = int(
        np.sum(
            true_ref & pred_ref
        )
    )

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return (
        sensitivity,
        specificity,
        tp,
        tn,
        fp,
        fn,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=============================================="
    )
    print(
        "RETINOXAI - IDRiD EXTERNAL DR EVALUATION"
    )
    print(
        "=============================================="
    )

    print(
        f"Device     : {DEVICE}"
    )

    print(
        f"Checkpoint : {CHECKPOINT}"
    )

    print(
        f"Images     : {IMAGE_DIR}"
    )

    print(
        f"Labels     : {LABEL_CSV}"
    )

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT}"
        )

    if not LABEL_CSV.exists():
        raise FileNotFoundError(
            f"Label CSV not found:\n{LABEL_CSV}"
        )

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image directory not found:\n{IMAGE_DIR}"
        )

    # --------------------------------------------------------
    # Load labels
    # --------------------------------------------------------

    df = pd.read_csv(
        LABEL_CSV,
        usecols=[
            "Image name",
            "Retinopathy grade",
            "Risk of macular edema ",
        ],
    )

    df["Image name"] = (
        df["Image name"]
        .astype(str)
        .str.strip()
    )

    df["Retinopathy grade"] = pd.to_numeric(
        df["Retinopathy grade"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "Image name",
            "Retinopathy grade",
        ]
    ).copy()

    df["Retinopathy grade"] = (
        df["Retinopathy grade"]
        .astype(int)
    )

    print(
        f"\nImages in label file: {len(df)}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_checkpoint(
        str(CHECKPOINT),
        DEVICE,
    )

    model.eval()

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    y_true: list[int] = []
    y_pred: list[int] = []
    confidences: list[float] = []
    image_names: list[str] = []

    print(
        "\nRunning inference..."
    )

    for idx, row in df.iterrows():

        image_name = str(
            row["Image name"]
        ).strip()

        true_grade = int(
            row["Retinopathy grade"]
        )

        image_path = (
            IMAGE_DIR
            / f"{image_name}.jpg"
        )

        if not image_path.exists():
            print(
                f"[WARNING] Missing image: "
                f"{image_path}"
            )
            continue

        try:
            image = Image.open(
                image_path
            ).convert("RGB")

            tensor = prepare_pil(
                image
            ).to(DEVICE)

            with torch.no_grad():

                logits = model(
                    tensor
                )

                probs = torch.softmax(
                    logits,
                    dim=1,
                )[0]

            pred_grade = int(
                torch.argmax(probs).item()
            )

            confidence = float(
                probs[pred_grade].item()
            )

            image_names.append(
                image_name
            )

            y_true.append(
                true_grade
            )

            y_pred.append(
                pred_grade
            )

            confidences.append(
                confidence
            )

        except Exception as exc:

            print(
                f"[WARNING] Failed on "
                f"{image_name}: {exc}"
            )

    if not y_true:
        raise RuntimeError(
            "No images were successfully evaluated."
        )

    y_true_np = np.asarray(
        y_true,
        dtype=int,
    )

    y_pred_np = np.asarray(
        y_pred,
        dtype=int,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true_np,
        y_pred_np,
    )

    macro_f1 = f1_score(
        y_true_np,
        y_pred_np,
        average="macro",
        zero_division=0,
    )

    qwk = cohen_kappa_score(
        y_true_np,
        y_pred_np,
        weights="quadratic",
    )

    sensitivity, specificity, tp, tn, fp, fn = (
        calculate_binary_metrics(
            y_true_np,
            y_pred_np,
        )
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true_np,
        y_pred_np,
        labels=[0, 1, 2, 3, 4],
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        "\n=============================================="
    )

    print(
        "IDRiD EXTERNAL EVALUATION RESULTS"
    )

    print(
        "=============================================="
    )

    print(
        f"Images evaluated : {len(y_true_np)}"
    )

    print(
        f"Accuracy         : {accuracy:.4f}"
    )

    print(
        f"Accuracy (%)     : {accuracy * 100:.2f}%"
    )

    print(
        f"Macro-F1         : {macro_f1:.4f}"
    )

    print(
        f"QWK              : {qwk:.4f}"
    )

    print(
        "\nREFERABLE DR (Grade >= 2)"
    )

    print(
        "---------------------------"
    )

    print(
        f"Sensitivity      : "
        f"{sensitivity:.4f} "
        f"({sensitivity * 100:.2f}%)"
    )

    print(
        f"Specificity      : "
        f"{specificity:.4f} "
        f"({specificity * 100:.2f}%)"
    )

    print(
        f"TP               : {tp}"
    )

    print(
        f"TN               : {tn}"
    )

    print(
        f"FP               : {fp}"
    )

    print(
        f"FN               : {fn}"
    )

    print(
        "\nCONFUSION MATRIX"
    )

    print(
        "----------------"
    )

    print(
        cm
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print(
        "\nCLASSIFICATION REPORT"
    )

    print(
        "---------------------"
    )

    report = classification_report(
        y_true_np,
        y_pred_np,
        labels=[0, 1, 2, 3, 4],
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    print(
        report
    )

    # --------------------------------------------------------
    # Mean confidence
    # --------------------------------------------------------

    mean_confidence = float(
        np.mean(confidences)
    )

    print(
        f"Mean prediction confidence: "
        f"{mean_confidence * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Save prediction CSV
    # --------------------------------------------------------

    predictions = pd.DataFrame(
        {
            "image_name": image_names,
            "true_grade": y_true_np,
            "predicted_grade": y_pred_np,
            "confidence": confidences,
        }
    )

    predictions[
        "correct"
    ] = (
        predictions["true_grade"]
        == predictions["predicted_grade"]
    )

    predictions[
        "true_referable"
    ] = (
        predictions["true_grade"]
        >= 2
    )

    predictions[
        "predicted_referable"
    ] = (
        predictions["predicted_grade"]
        >= 2
    )

    prediction_path = (
        OUTPUT_DIR
        / "idrid_grading_predictions.csv"
    )

    predictions.to_csv(
        prediction_path,
        index=False,
    )

    # --------------------------------------------------------
    # Save metrics text file
    # --------------------------------------------------------

    metrics_path = (
        OUTPUT_DIR
        / "idrid_external_evaluation.txt"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "RETINOXAI - IDRiD EXTERNAL EVALUATION\n"
        )

        f.write(
            "====================================\n\n"
        )

        f.write(
            f"Images evaluated: {len(y_true_np)}\n"
        )

        f.write(
            f"Accuracy: {accuracy:.6f}\n"
        )

        f.write(
            f"Macro-F1: {macro_f1:.6f}\n"
        )

        f.write(
            f"QWK: {qwk:.6f}\n"
        )

        f.write(
            "\nReferable DR (Grade >= 2)\n"
        )

        f.write(
            f"Sensitivity: {sensitivity:.6f}\n"
        )

        f.write(
            f"Specificity: {specificity:.6f}\n"
        )

        f.write(
            f"TP: {tp}\n"
        )

        f.write(
            f"TN: {tn}\n"
        )

        f.write(
            f"FP: {fp}\n"
        )

        f.write(
            f"FN: {fn}\n"
        )

        f.write(
            "\nConfusion Matrix:\n"
        )

        f.write(
            np.array2string(cm)
        )

        f.write(
            "\n\nClassification Report:\n"
        )

        f.write(
            report
        )

    print(
        "\nSaved:"
    )

    print(
        prediction_path
    )

    print(
        metrics_path
    )

    print(
        "\n=============================================="
    )


if __name__ == "__main__":
    main()