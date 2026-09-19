import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGE_SIZE = 224
NUM_CLASSES = 5
MODEL_NAME = "efficientnet_b0"


class AptosDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_dir: Path, train: bool):
        self.frame = frame.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
            transforms.RandomRotation(12) if train else transforms.Lambda(lambda x: x),
            transforms.ColorJitter(brightness=0.15, contrast=0.15) if train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _find_image(self, stem: str) -> Path:
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = self.image_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Image not found for id={stem}")

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        image = Image.open(self._find_image(str(row["id_code"]))).convert("RGB")
        return self.transform(image), int(row["diagnosis"])


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(1).cpu().numpy()
            y_pred.extend(pred.tolist())
            y_true.extend(y.numpy().tolist())
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return float(qwk), float(macro_f1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to APTOS train.csv")
    ap.add_argument("--image-dir", required=True, help="Directory containing fundus images")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="models/best.pt")
    args = ap.parse_args()

    frame = pd.read_csv(args.csv)
    needed = {"id_code", "diagnosis"}
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    train_df, val_df = train_test_split(
        frame,
        test_size=0.2,
        random_state=42,
        stratify=frame["diagnosis"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds = AptosDataset(train_df, Path(args.image_dir), train=True)
    val_ds = AptosDataset(val_df, Path(args.image_dir), train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES).to(device)

    counts = np.bincount(train_df["diagnosis"].to_numpy(), minlength=NUM_CLASSES)
    weights = counts.sum() / np.maximum(counts, 1)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    best_qwk = -1.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)

        qwk, macro_f1 = evaluate(model, val_loader, device)
        scheduler.step(qwk)
        avg_loss = running / len(train_ds)
        print(f"Epoch {epoch:02d} | loss={avg_loss:.4f} | QWK={qwk:.4f} | macro-F1={macro_f1:.4f}")

        if qwk > best_qwk:
            best_qwk = qwk
            torch.save({
                "model_state_dict": model.state_dict(),
                "model": MODEL_NAME,
                "num_classes": NUM_CLASSES,
                "best_qwk": best_qwk,
            }, out_path)
            print("Saved:", out_path)

    print("Best validation QWK:", best_qwk)


if __name__ == "__main__":
    main()
