import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import cohen_kappa_score, f1_score


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
    def __init__(
        self,
        frame: pd.DataFrame,
        image_dir: Path,
        train: bool,
    ):
        self.frame = frame.reset_index(drop=True)
        self.image_dir = image_dir

        if train:
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(12),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
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
            candidate = self.image_dir / f"{stem}{ext}"

            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"Image not found for id_code={stem}"
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


def evaluate(model, loader, device):
    model.eval()

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

    return float(qwk), float(macro_f1)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-csv",
        required=True,
    )

    parser.add_argument(
        "--val-csv",
        required=True,
    )

    parser.add_argument(
        "--image-dir",
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--out",
        default="backend/models/best_balanced.pt",
    )

    args = parser.parse_args()

    train_df = pd.read_csv(
        args.train_csv
    )

    val_df = pd.read_csv(
        args.val_csv
    )

    required_columns = {
        "id_code",
        "diagnosis",
    }

    for name, df in [
        ("train", train_df),
        ("validation", val_df),
    ]:
        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"{name} CSV missing columns: "
                f"{sorted(missing)}"
            )

    print("Train images:", len(train_df))
    print("Validation images:", len(val_df))

    print("\nTrain distribution:")
    print(
        train_df["diagnosis"]
        .value_counts()
        .sort_index()
    )

    print("\nValidation distribution:")
    print(
        val_df["diagnosis"]
        .value_counts()
        .sort_index()
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nDevice:", device)

    # -----------------------------
    # Datasets
    # -----------------------------

    train_dataset = AptosDataset(
        train_df,
        Path(args.image_dir),
        train=True,
    )

    val_dataset = AptosDataset(
        val_df,
        Path(args.image_dir),
        train=False,
    )

    # -----------------------------
    # WeightedRandomSampler
    # -----------------------------

    class_counts = (
        train_df["diagnosis"]
        .value_counts()
        .sort_index()
    )

    class_weights = (
        1.0
        / class_counts.to_numpy(dtype=np.float64)
    )

    sample_weights = train_df["diagnosis"].map(
        lambda label: class_weights[int(label)]
    ).to_numpy()

    sample_weights = torch.as_tensor(
        sample_weights,
        dtype=torch.double,
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print("\nBalanced sampling enabled:")
    print("WeightedRandomSampler = ON")
    print("Replacement           = True")

    # -----------------------------
    # Model
    # -----------------------------

    model = timm.create_model(
        MODEL_NAME,
        pretrained=True,
        num_classes=NUM_CLASSES,
    )

    model = model.to(device)

    # Standard loss for this experiment.
    # We are balancing through the sampler.
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=2,
        factor=0.5,
    )

    best_qwk = -1.0

    output_path = Path(args.out)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------
    # Training
    # -----------------------------

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()

        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(images)

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * images.size(0)
            )

        qwk, macro_f1 = evaluate(
            model,
            val_loader,
            device,
        )

        scheduler.step(qwk)

        average_loss = (
            running_loss
            / len(train_dataset)
        )

        print(
            f"Epoch {epoch:02d} | "
            f"loss={average_loss:.4f} | "
            f"QWK={qwk:.4f} | "
            f"macro-F1={macro_f1:.4f}"
        )

        if qwk > best_qwk:
            best_qwk = qwk

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model": MODEL_NAME,
                    "num_classes": NUM_CLASSES,
                    "best_qwk": best_qwk,
                    "training_method": "WeightedRandomSampler",
                },
                output_path,
            )

            print(
                "Saved:",
                output_path,
            )

    print(
        "\nBest validation QWK:",
        best_qwk,
    )


if __name__ == "__main__":
    main()