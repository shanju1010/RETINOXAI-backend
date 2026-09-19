import numpy as np
import torch
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def generate_gradcam(
    model,
    input_tensor: torch.Tensor,
    original_rgb: Image.Image,
):
    """Create a Grad-CAM heatmap for the predicted class."""

    # EfficientNet-B0 final convolution feature layer
    target_layer = model.conv_head

    # Get the predicted class
    with torch.no_grad():
        logits = model(input_tensor)
        predicted_class = int(torch.argmax(logits, dim=1).item())

    # Use the official Grad-CAM classification target
    targets = [ClassifierOutputTarget(predicted_class)]

    # Generate CAM
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets,
        )[0]

    # Resize original image to CAM size
    resized = original_rgb.resize(
        (input_tensor.shape[-1], input_tensor.shape[-2])
    )

    rgb_float = np.asarray(resized).astype(np.float32) / 255.0

    # Overlay heatmap
    overlay = show_cam_on_image(
        rgb_float,
        grayscale_cam,
        use_rgb=True,
    )

    return predicted_class, Image.fromarray(overlay)