"""Image preprocessing for inference.

Functions to preprocess raw user drawings into 28x28 normalized tensors
suitable for model inference.
"""

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

def preprocess_drawing(image: np.ndarray | Image.Image, is_emnist: bool = False) -> torch.Tensor:
    """Preprocess a drawing (black background, white stroke or vice-versa) into a model-ready tensor.

    Args:
        image: Input image (numpy array or PIL Image).
        is_emnist: Whether to apply EMNIST-specific transpose (rotation/flip).

    Returns:
        Tensor of shape (1, 1, 28, 28).
    """
    if isinstance(image, Image.Image):
        # Convert PIL to numpy array
        if image.mode == 'RGBA':
            # Check if image actually has transparent pixels
            alpha = np.array(image)[:, :, 3]
            if np.min(alpha) < 255:
                # It has transparent areas (like Gradio sketchpad), use alpha as the drawing
                img = alpha
            else:
                # Opaque image (like our Flask canvas), convert to grayscale
                img = np.array(image.convert("L"))
        else:
            img = np.array(image.convert("L"))
    else:
        # Assume it's an OpenCV/Numpy image
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                alpha = image[:, :, 3]
                if np.min(alpha) < 255:
                    img = alpha
                else:
                    img = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            elif image.shape[2] == 3:
                # RGB/BGR image
                img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                img = image[:, :, 0]
        else:
            img = image.copy()

    # We expect a black background with white digit/character.
    # If the background is mostly white, invert it.
    if np.mean(img) > 127:
        img = cv2.bitwise_not(img)

    # Find the bounding box of the non-zero (white) pixels
    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        
        # Crop to the bounding box
        cropped = img[y:y+h, x:x+w]
        
        # Pad to make it square, keeping the drawing centered
        size = max(w, h)
        
        # Add some padding (similar to MNIST where digits don't touch the edge)
        pad = int(size * 0.2)
        size += pad * 2
        
        # Create an empty black square image
        square = np.zeros((size, size), dtype=np.uint8)
        
        # Calculate offsets to paste the cropped image in the center
        x_offset = (size - w) // 2
        y_offset = (size - h) // 2
        
        square[y_offset:y_offset+h, x_offset:x_offset+w] = cropped
        img = square

    # Resize to 28x28
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # Convert back to PIL for transforms
    pil_img = Image.fromarray(img)
    
    if is_emnist:
        # EMNIST requires transpose
        pil_img = pil_img.transpose(Image.TRANSPOSE)

    # Transform to tensor and normalize (MNIST/EMNIST stats)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    tensor = transform(pil_img)
    
    # Add batch dimension (1, 1, 28, 28)
    return tensor.unsqueeze(0)
