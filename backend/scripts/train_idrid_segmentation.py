from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(
    r"C:\retinoxai-prototype\data\IDRiD\A. Segmentation"
)

IMAGE_DIR = (
    ROOT
    / "1. Original Images"
    / "a. Training Set"
)

MASK_ROOT = (
    ROOT
    / "2. All Segmentation Groundtruths"
    / "a. Training Set"
)

OUTPUT_DIR = Path(
    r"C:\retinoxai-prototype\backend\models"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

IMAGE_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 15
LEARNING_RATE = 1e-3
VAL_RATIO = 0.20
SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = [
    "microaneurysm",
    "hard_exudate",
    "haemorrhage",
    "soft_exudate",
    "optic_disc",
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# FILE PATHS
# ============================================================

MASK_DIRS = {
    "microaneurysm":
        MASK_ROOT / "1. Microaneurysms",

    "haemorrhage":
        MASK_ROOT / "2. Haemorrhages",

    "hard_exudate":
        MASK_ROOT / "3. Hard Exudates",

    "soft_exudate":
        MASK_ROOT / "4. Soft Exudates",

    "optic_disc":
        MASK_ROOT / "5. Optic Disc",
}


# ============================================================
# DATASET
# ============================================================

class IDRiDSegmentationDataset(Dataset):

    def __init__(
        self,
        image_ids: list[str],
        augment: bool = False,
    ) -> None:

        self.image_ids = image_ids
        self.augment = augment
    def __len__(self) -> int:
        return len(self.image_ids)
    
    @staticmethod
    def resize_and_pad(
        image: np.ndarray,
        size: int,
        is_mask: bool = False,
    ) -> np.ndarray:

        h, w = image.shape[:2]

        scale = min(
            size / h,
            size / w,
        )

        new_w = max(
            1,
            int(round(w * scale)),
        )

        new_h = max(
            1,
            int(round(h * scale)),
        )

        interpolation = (
            cv2.INTER_NEAREST
            if is_mask
            else cv2.INTER_AREA
        )

        resized = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=interpolation,
        )

        if image.ndim == 3:
            canvas = np.zeros(
                (size, size, image.shape[2]),
                dtype=image.dtype,
            )
        else:
            canvas = np.zeros(
                (size, size),
                dtype=image.dtype,
            )

        top = (size - new_h) // 2
        left = (size - new_w) // 2

        canvas[
            top:top + new_h,
            left:left + new_w
        ] = resized

        return canvas

    @staticmethod
    def load_mask(
        path: Path,
        size: int,
    ) -> np.ndarray:

        if not path.exists():
            return np.zeros(
                (size, size),
                dtype=np.uint8,
            )

        mask = cv2.imread(
            str(path),
            cv2.IMREAD_GRAYSCALE,
        )

        if mask is None:
            return np.zeros(
                (size, size),
                dtype=np.uint8,
            )

        mask = (
            mask > 0
        ).astype(np.uint8)

        mask = (
            IDRiDSegmentationDataset
            .resize_and_pad(
                mask,
                size,
                is_mask=True,
            )
        )

        return mask

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        image_id = self.image_ids[index]

        image_path = (
            IMAGE_DIR
            / f"{image_id}.jpg"
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            raise RuntimeError(
                f"Could not read: {image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        original_h, original_w = image.shape[:2]

        image = self.resize_and_pad(
            image,
            IMAGE_SIZE,
            is_mask=False,
        )

        masks = []

        # --------------------------------------------
        # MA
        # --------------------------------------------

        ma_path = (
            MASK_DIRS["microaneurysm"]
            / f"{image_id}_MA.tif"
        )

        masks.append(
            self.load_mask(
                ma_path,
                IMAGE_SIZE,
            )
        )

        # --------------------------------------------
        # EX
        # --------------------------------------------

        ex_path = (
            MASK_DIRS["hard_exudate"]
            / f"{image_id}_EX.tif"
        )

        masks.append(
            self.load_mask(
                ex_path,
                IMAGE_SIZE,
            )
        )

        # --------------------------------------------
        # HE
        # --------------------------------------------

        he_path = (
            MASK_DIRS["haemorrhage"]
            / f"{image_id}_HE.tif"
        )

        masks.append(
            self.load_mask(
                he_path,
                IMAGE_SIZE,
            )
        )

        # --------------------------------------------
        # SE
        # --------------------------------------------

        se_path = (
            MASK_DIRS["soft_exudate"]
            / f"{image_id}_SE.tif"
        )

        masks.append(
            self.load_mask(
                se_path,
                IMAGE_SIZE,
            )
        )

        # --------------------------------------------
        # OPTIC DISC
        # --------------------------------------------

        od_path = (
            MASK_DIRS["optic_disc"]
            / f"{image_id}_OD.tif"
        )

        masks.append(
            self.load_mask(
                od_path,
                IMAGE_SIZE,
            )
        )

        masks = np.stack(
            masks,
            axis=0,
        ).astype(np.float32)

        # --------------------------------------------
        # Data augmentation
        # --------------------------------------------

        if self.augment:

            if random.random() < 0.5:

                image = np.fliplr(
                    image
                ).copy()

                masks = np.fliplr(
                    masks
                ).copy()

            if random.random() < 0.5:

                image = np.flipud(
                    image
                ).copy()

                masks = np.flipud(
                    masks
                ).copy()

        image = (
            image.astype(np.float32)
            / 255.0
        )

        image = torch.from_numpy(
            image.transpose(2, 0, 1)
        ).float()

        masks = torch.from_numpy(
            masks
        ).float()

        return image, masks


# ============================================================
# U-NET
# ============================================================

class DoubleConv(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:

        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=1,
            ),
            nn.BatchNorm2d(
                out_channels
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1,
            ),
            nn.BatchNorm2d(
                out_channels
            ),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.block(x)


class UNet(nn.Module):

    def __init__(
        self,
        out_channels: int = 5,
    ) -> None:

        super().__init__()

        self.enc1 = DoubleConv(
            3,
            32,
        )

        self.enc2 = DoubleConv(
            32,
            64,
        )

        self.enc3 = DoubleConv(
            64,
            128,
        )

        self.enc4 = DoubleConv(
            128,
            256,
        )

        self.pool = nn.MaxPool2d(
            2
        )

        self.bottleneck = DoubleConv(
            256,
            512,
        )

        self.up4 = nn.ConvTranspose2d(
            512,
            256,
            2,
            stride=2,
        )

        self.dec4 = DoubleConv(
            512,
            256,
        )

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            2,
            stride=2,
        )

        self.dec3 = DoubleConv(
            256,
            128,
        )

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            2,
            stride=2,
        )

        self.dec2 = DoubleConv(
            128,
            64,
        )

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            2,
            stride=2,
        )

        self.dec1 = DoubleConv(
            64,
            32,
        )

        self.out = nn.Conv2d(
            32,
            out_channels,
            1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        b = self.bottleneck(
            self.pool(e4)
        )

        d4 = self.up4(b)

        d4 = torch.cat(
            [d4, e4],
            dim=1,
        )

        d4 = self.dec4(d4)

        d3 = self.up3(d4)

        d3 = torch.cat(
            [d3, e3],
            dim=1,
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1,
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1,
        )

        d1 = self.dec1(d1)

        return self.out(d1)


# ============================================================
# LOSS
# ============================================================

def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:

    probs = torch.sigmoid(
        logits
    )

    smooth = 1.0

    dims = (
        0,
        2,
        3,
    )

    intersection = (
        probs * targets
    ).sum(dims)

    denominator = (
        probs + targets
    ).sum(dims)

    dice = (
        2.0 * intersection
        + smooth
    ) / (
        denominator
        + smooth
    )

    return (
        1.0 - dice
    ).mean()


def combined_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:

    bce = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
    )

    dice = dice_loss(
        logits,
        targets,
    )

    return (
        0.5 * bce
        + 0.5 * dice
    )


# ============================================================
# METRICS
# ============================================================

@torch.no_grad()
def mean_dice(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[float, list[float]]:

    probs = torch.sigmoid(
        logits
    )

    preds = (
        probs >= 0.5
    ).float()

    class_scores = []

    for c in range(
        targets.shape[1]
    ):

        pred = preds[:, c]
        target = targets[:, c]

        intersection = (
            pred * target
        ).sum()

        denominator = (
            pred.sum()
            + target.sum()
        )

        # For an empty target and empty prediction,
        # treat the result as perfect for that sample.
        if denominator.item() == 0:
            score = 1.0
        else:
            score = (
                (
                    2.0 * intersection
                    + 1e-6
                )
                /
                (
                    denominator
                    + 1e-6
                )
            ).item()

        class_scores.append(
            float(score)
        )

    return (
        float(np.mean(class_scores)),
        class_scores,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=========================================="
    )
    print(
        "RETINOXAI - IDRiD LESION SEGMENTATION"
    )
    print(
        "=========================================="
    )

    print(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Find available training images
    # --------------------------------------------------------

    image_ids = []

    for path in sorted(
        IMAGE_DIR.glob("*.jpg")
    ):

        image_ids.append(
            path.stem
        )

    if len(image_ids) == 0:
        raise RuntimeError(
            "No IDRiD training images found."
        )

    print(
        f"Images found: {len(image_ids)}"
    )

    # --------------------------------------------------------
    # Split train / validation
    # --------------------------------------------------------

    rng = random.Random(
        SEED
    )

    shuffled = image_ids.copy()

    rng.shuffle(
        shuffled
    )

    val_count = max(
        1,
        int(
            len(shuffled)
            * VAL_RATIO
        ),
    )

    val_ids = shuffled[
        :val_count
    ]

    train_ids = shuffled[
        val_count:
    ]

    print(
        f"Training images  : {len(train_ids)}"
    )

    print(
        f"Validation images: {len(val_ids)}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = IDRiDSegmentationDataset(
        train_ids,
        augment=True,
    )

    val_dataset = IDRiDSegmentationDataset(
        val_ids,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = UNet(
        out_channels=5
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    scaler = (
        torch.amp.GradScaler("cuda")
        if torch.cuda.is_available()
        else None
    )

    best_dice = -1.0

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        train_loss = 0.0

        train_bar = tqdm(
            train_loader,
            desc=(
                f"Epoch "
                f"{epoch}/{EPOCHS} "
                "Train"
            ),
        )

        for images, masks in train_bar:

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            masks = masks.to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            if scaler is not None:

                with torch.amp.autocast(
                    "cuda"
                ):

                    logits = model(
                        images
                    )

                    loss = combined_loss(
                        logits,
                        masks,
                    )

                scaler.scale(
                    loss
                ).backward()

                scaler.step(
                    optimizer
                )

                scaler.update()

            else:

                logits = model(
                    images
                )

                loss = combined_loss(
                    logits,
                    masks,
                )

                loss.backward()

                optimizer.step()

            train_loss += (
                loss.item()
                * images.size(0)
            )

        train_loss /= len(
            train_dataset
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        val_loss = 0.0

        all_scores = []

        with torch.no_grad():

            for images, masks in val_loader:

                images = images.to(
                    DEVICE,
                    non_blocking=True,
                )

                masks = masks.to(
                    DEVICE,
                    non_blocking=True,
                )

                logits = model(
                    images
                )

                loss = combined_loss(
                    logits,
                    masks,
                )

                val_loss += (
                    loss.item()
                    * images.size(0)
                )

                batch_mean, batch_scores = (
                    mean_dice(
                        logits,
                        masks,
                    )
                )

                all_scores.append(
                    batch_scores
                )

        val_loss /= len(
            val_dataset
        )

        class_scores = np.mean(
            np.asarray(
                all_scores
            ),
            axis=0,
        )

        val_mean_dice = float(
            np.mean(
                class_scores
            )
        )

        scheduler.step(
            val_mean_dice
        )

        print(
            f"\nEpoch {epoch}/{EPOCHS}"
        )

        print(
            f"Train Loss : "
            f"{train_loss:.4f}"
        )

        print(
            f"Val Loss   : "
            f"{val_loss:.4f}"
        )

        print(
            f"Mean Dice  : "
            f"{val_mean_dice:.4f}"
        )

        for name, score in zip(
            CLASS_NAMES,
            class_scores,
        ):
            print(
                f"  {name:18s}: "
                f"{score:.4f}"
            )

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if val_mean_dice > best_dice:

            best_dice = (
                val_mean_dice
            )

            checkpoint_path = (
                OUTPUT_DIR
                / "idrid_segmentation_best.pt"
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "image_size":
                        IMAGE_SIZE,

                    "class_names":
                        CLASS_NAMES,

                    "best_val_dice":
                        best_dice,
                },
                checkpoint_path,
            )

            print(
                "\nSaved best checkpoint:"
            )

            print(
                checkpoint_path
            )

    print(
        "\n=========================================="
    )

    print(
        "IDRiD SEGMENTATION TRAINING COMPLETE"
    )

    print(
        f"Best validation Mean Dice: "
        f"{best_dice:.4f}"
    )

    print(
        "Checkpoint:"
    )

    print(
        OUTPUT_DIR
        / "idrid_segmentation_best.pt"
    )

    print(
        "=========================================="
    )


if __name__ == "__main__":
    main()