from pathlib import Path

import timm
import torch
from PIL import Image
from torchvision import transforms

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
MODEL_NAME = "efficientnet_b0"
NUM_CLASSES = 5
IMAGE_SIZE = 224


def build_model(pretrained: bool = False):
    return timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=NUM_CLASSES)


def load_checkpoint(checkpoint_path: str, device: torch.device):
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")

    model = build_model(pretrained=False)
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def prepare_pil(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    return INFER_TRANSFORM(image).unsqueeze(0)
