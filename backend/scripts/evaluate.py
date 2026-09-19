import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


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
    def __init__(self, frame: pd.DataFrame, image_dir: Path):
        self.frame = frame.reset_index(drop=True)
        self.image_dir = image_dir

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _find_image(self, stem: str) -> Path:
        for ext in [".png", ".jpg", ".jpeg"]:
            path = self.image_dir / f"{stem}{ext}"
            if path.exists():
                return path

        raise FileNotFoundError(
            f"Image not found for id_code={stem}"
        )

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]

        image_path = self._find_image(str(row["id_code"]))
        image = Image.open(image_path).convert("RGB")

        image = self.transform(image)
        label = int(row["diagnosis"])

        return image, label


def load_model(checkpoint_path: Path, device: torch.device):
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


def evaluate(model, loader, device):
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            logits = model(images)
            predictions = torch.argmax(logits, dim=1)

            y_true.extend(labels.numpy().tolist())
            y_pred.extend(predictions.cpu().numpy().tolist())

    return np.array(y_true), np.array(y_pred)


def save_confusion_matrix(cm, output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 7))

    image = ax.imshow(cm)

    ax.set(
        xticks=np.arange(NUM_CLASSES),
        yticks=np.arange(NUM_CLASSES),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted Label",
        ylabel="True Label",
        title="RETINOXAI Validation Confusion Matrix",
    )

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        required=True,
        help="Path to train.csv",
    )

    parser.add_argument(
        "--image-dir",
        required=True,
        help="Directory containing fundus images",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Model checkpoint",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--output-dir",
        default="backend/evaluation",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)
    image_dir = Path(args.image_dir)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Same split used during training.
    frame = pd.read_csv(csv_path)

    required_columns = {"id_code", "diagnosis"}
    missing = required_columns - set(frame.columns)

    if missing:
        raise ValueError(
            f"CSV missing columns: {sorted(missing)}"
        )

    _, val_df = train_test_split(
        frame,
        test_size=0.2,
        random_state=42,
        stratify=frame["diagnosis"],
    )

    print("Validation images:", len(val_df))

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    dataset = AptosDataset(
        val_df,
        image_dir,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = load_model(
        checkpoint_path,
        device,
    )

    y_true, y_pred = evaluate(
        model,
        loader,
        device,
    )

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
    print("RETINOXAI VALIDATION RESULTS")
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

    cm_path = output_dir / "validation_confusion_matrix.png"

    save_confusion_matrix(
        cm,
        cm_path,
    )

    np.save(
        output_dir / "y_true.npy",
        y_true,
    )

    np.save(
        output_dir / "y_pred.npy",
        y_pred,
    )

    print("\nSaved:")
    print(cm_path)


if __name__ == "__main__":
    main()