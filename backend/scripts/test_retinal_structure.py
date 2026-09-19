from pathlib import Path
import cv2

from backend.services.retinal_structure import (
    analyze_retinal_structures,
    save_structure_outputs,
    make_api_result,
)


IMAGE_PATH = Path(
    r"C:\retinoxai-prototype\data\train_images\04efb1a284cc.png"
)

OUTPUT_DIR = Path(
    r"C:\retinoxai-prototype\backend\outputs"
)


def main() -> None:
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH}"
        )

    image = cv2.imread(
        str(IMAGE_PATH)
    )

    if image is None:
        raise RuntimeError(
            "OpenCV could not read the image."
        )

    result = analyze_retinal_structures(
        image
    )

    files = save_structure_outputs(
        image,
        result,
        OUTPUT_DIR,
        prefix="04efb1a284cc",
    )

    print("\n====================================")
    print("RETINOXAI RETINAL STRUCTURE ANALYSIS")
    print("====================================")

    print("\nVessels")
    print(
        f"Analyzed: "
        f"{result['vessels']['analyzed']}"
    )
    print(
        f"Candidate pixel ratio: "
        f"{result['vessels']['density']}"
    )

    print("\nOptic Disc")
    print(
        f"Detected: "
        f"{result['optic_disc']['detected']}"
    )
    print(
        f"Center: "
        f"({result['optic_disc']['center_x']}, "
        f"{result['optic_disc']['center_y']})"
    )
    print(
        f"Confidence: "
        f"{result['optic_disc']['confidence']:.2f}"
    )

    print("\nFovea")
    print(
        f"Estimated: "
        f"{result['fovea']['estimated']}"
    )
    print(
        f"Center: "
        f"({result['fovea']['center_x']}, "
        f"{result['fovea']['center_y']})"
    )
    print(
        f"Confidence: "
        f"{result['fovea']['confidence']:.2f}"
    )
    print(
        f"Method: "
        f"{result['fovea']['method']}"
    )

    print("\nGenerated files:")
    print(files["vessel_mask"])
    print(files["annotated_image"])

    print("\nJSON response:")
    print(
        make_api_result(result)
    )


if __name__ == "__main__":
    main()