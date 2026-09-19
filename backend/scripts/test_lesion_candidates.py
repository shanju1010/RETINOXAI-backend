from pathlib import Path

import cv2

from backend.services.lesion_candidates import (
    analyze_lesions,
    annotate_lesions,
)


IMAGE_PATH = Path(
    r"C:\retinoxai-prototype\data\IDRiD"
    r"\A. Segmentation"
    r"\1. Original Images"
    r"\a. Training Set"
    r"\IDRiD_01.jpg"
)

OUTPUT_DIR = Path(
    r"C:\retinoxai-prototype\backend\outputs"
)


def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = cv2.imread(
        str(IMAGE_PATH)
    )

    if image is None:
        raise RuntimeError(
            f"Could not read {IMAGE_PATH}"
        )

    result = analyze_lesions(
        image
    )

    print(
        "\n======================================"
    )
    print(
        "RETINOXAI - LESION CANDIDATE ANALYSIS"
    )
    print(
        "======================================"
    )

    print(
        "\nMicroaneurysms (MA)"
    )

    print(
        f"Detected: "
        f"{result['microaneurysm']['detected']}"
    )

    print(
        f"Candidates: "
        f"{result['microaneurysm']['candidate_count']}"
    )

    print(
        "\nHard Exudates (EX)"
    )

    print(
        f"Detected: "
        f"{result['hard_exudate']['detected']}"
    )

    print(
        f"Candidates: "
        f"{result['hard_exudate']['candidate_count']}"
    )

    print(
        "\nHaemorrhages (HE)"
    )

    print(
        f"Detected: "
        f"{result['haemorrhage']['detected']}"
    )

    print(
        f"Candidates: "
        f"{result['haemorrhage']['candidate_count']}"
    )

    print(
        "\nSoft Exudates (SE)"
    )

    print(
        f"Detected: "
        f"{result['soft_exudate']['detected']}"
    )

    print(
        f"Candidates: "
        f"{result['soft_exudate']['candidate_count']}"
    )

    annotated = annotate_lesions(
        image,
        result,
    )

    output_path = (
        OUTPUT_DIR
        / "IDRiD_01_lesion_candidates.png"
    )

    cv2.imwrite(
        str(output_path),
        annotated,
    )

    print(
        "\nGenerated:"
    )

    print(
        output_path
    )

    print(
        "\n======================================"
    )


if __name__ == "__main__":
    main()