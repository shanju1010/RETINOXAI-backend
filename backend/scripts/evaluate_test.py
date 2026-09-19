import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)


IMAGE_SIZE = 224
NUM_CLASSES = 5
MODEL_NAME = "efficientnet_b0"

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR",
]


class AptosDataset(Dataset):
    def __init__(self, frame, image_dir):
        self.frame = frame.reset_index(drop=True)
        self.image_dir = Path(image_dir)

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _find_image(self, stem):
        for ext in [".png", ".jpg", ".jpeg"]:
            path = self.image_dir / f"{stem}{ext}"
            if path.exists():
                return path

        raise FileNotFoundError(
            f"Image not found: {stem}"
        )

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]

        image_path = self._find_image(
            str(row["id_code"])
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        image = self.transform(image)

        label = int(row["diagnosis"])

        return image, label


def load_model(checkpoint_path, device):
    model = timm.create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=NUM_CLASSES,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test-csv",
        required=True,
    )

    parser.add_argument(
        "--image-dir",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--output-dir",
        default="backend/evaluation/test",
    )

    args = parser.parse_args()

    test_csv = Path(args.test_csv)
    image_dir = Path(args.image_dir)
    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(test_csv)

    required = {"id_code", "diagnosis"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    print("TEST SET")
    print("========")
    print("Test images:", len(df))

    print("\nDistribution:")
    print(
        df["diagnosis"]
        .value_counts()
        .sort_index()
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nDevice:", device)

    dataset = AptosDataset(
        df,
        image_dir,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = load_model(
        checkpoint,
        device,
    )

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(
                device,
                non_blocking=True,
            )

            logits = model(images)

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            y_true.extend(
                labels.numpy().tolist()
            )

            y_pred.extend(
                predictions.cpu().numpy().tolist()
            )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    qwk = cohen_kappa_score(
        y_true,
        y_pred,
        weights="quadratic",
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
    )

    print("\n==============================")
    print("RETINOXAI INDEPENDENT TEST")
    print("==============================")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"QWK      : {qwk:.4f}")
    print(f"Macro-F1 : {macro_f1:.4f}")

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(cm)

    np.save(
        output_dir / "y_true.npy",
        y_true,
    )

    np.save(
        output_dir / "y_pred.npy",
        y_pred,
    )

    np.savetxt(
        output_dir / "confusion_matrix.csv",
        cm,
        delimiter=",",
        fmt="%d",
    )

    print("\nSaved test results to:")
    print(output_dir)


if __name__ == "__main__":
    main()