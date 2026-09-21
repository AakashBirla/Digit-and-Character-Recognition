#!/usr/bin/env python3
"""Stage 5: Real-time drawing and recognition interface.

Provides a Gradio web interface for users to draw digits and characters
and get real-time predictions from the trained models.

Usage:
    python app/app.py
"""

import logging
import sys
from pathlib import Path

import gradio as gr
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.cnn import CNN
from models.mlp import MLP
from src.utils.preprocessing import preprocess_drawing
from src.utils.seed import get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Paths to models
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
MODELS = {
    "CNN - MNIST (Digits 0-9)": {
        "path": CHECKPOINT_DIR / "mnist_cnn_best.pth",
        "dataset": "mnist",
        "architecture": "cnn",
    },
    "CNN - EMNIST (Characters A-Z, a-z, 0-9)": {
        "path": CHECKPOINT_DIR / "emnist_cnn_best.pth",
        "dataset": "emnist",
        "architecture": "cnn",
    },
    "MLP - MNIST (Digits 0-9)": {
        "path": CHECKPOINT_DIR / "mnist_mlp_best.pth",
        "dataset": "mnist",
        "architecture": "mlp",
    },
}

# Global dictionary to cache loaded models
loaded_models = {}
device = get_device()


def load_model(model_name: str) -> tuple[torch.nn.Module, dict, bool]:
    """Load a model or get it from cache."""
    if model_name in loaded_models:
        return loaded_models[model_name]
        
    config = MODELS[model_name]
    checkpoint_path = config["path"]
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_mapping = checkpoint["class_mapping"]
    
    # Convert keys to int
    class_mapping = {int(k): str(v) for k, v in class_mapping.items()}
    num_classes = len(class_mapping)
    
    if config["architecture"] == "cnn":
        model = CNN(num_classes=num_classes)
    else:
        model = MLP(num_classes=num_classes)
        
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    is_emnist = config["dataset"] == "emnist"
    
    loaded_models[model_name] = (model, class_mapping, is_emnist)
    return loaded_models[model_name]


def predict(image, model_name):
    """Predict the character drawn on the image canvas."""
    if image is None:
        return "Please draw something first!"
    
    # Gradio sketchpad returns a dict with 'composite' key containing the RGBA image
    if isinstance(image, dict):
        img_array = image["composite"]
    else:
        img_array = image
        
    # Check if empty drawing (all transparent or white)
    import numpy as np
    if np.sum(img_array) == 0 or np.all(img_array == 255):
        return "Canvas is empty!"
        
    try:
        model, class_mapping, is_emnist = load_model(model_name)
    except FileNotFoundError as e:
        return f"Error: Model file not found. Have you trained {model_name}?"
        
    try:
        # Preprocess the drawing
        tensor = preprocess_drawing(img_array, is_emnist=is_emnist)
        tensor = tensor.to(device)
        
        # Predict
        with torch.no_grad():
            output = model(tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            
        # Get top 3 predictions
        top_prob, top_class = torch.topk(probabilities, 3)
        
        top_prob = top_prob.cpu().numpy()
        top_class = top_class.cpu().numpy()
        
        result = ""
        for i in range(3):
            char = class_mapping[top_class[i]]
            prob = top_prob[i] * 100
            result += f"{i+1}. Predicted Character: '{char}' (Confidence: {prob:.2f}%)\n"
            
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return f"Error processing image: {e}"


def create_app():
    """Create and configure the Gradio app."""
    
    with gr.Blocks(title="Handwritten Digit & Character Recognition") as app:
        gr.Markdown("# Handwritten Digit & Character Recognition")
        gr.Markdown(
            "Draw a digit (0-9) or character (A-Z) in the box below, "
            "select a model, and click **Predict**."
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                # Sketchpad for drawing
                canvas = gr.Sketchpad(
                    label="Draw Here",
                    type="numpy",
                    crop_size=(280, 280),
                    layers=False,
                    brush=gr.Brush(colors=["#FFFFFF"], thickness=20)
                )
                
                model_selector = gr.Dropdown(
                    choices=list(MODELS.keys()),
                    value="CNN - EMNIST (Characters A-Z, a-z, 0-9)",
                    label="Select Model"
                )
                
                with gr.Row():
                    clear_btn = gr.Button("Clear")
                    predict_btn = gr.Button("Predict", variant="primary")
                    
            with gr.Column(scale=1):
                output_text = gr.Textbox(
                    label="Top 3 Predictions",
                    lines=5,
                    interactive=False
                )
                
        # Handle button clicks
        predict_btn.click(
            fn=predict,
            inputs=[canvas, model_selector],
            outputs=output_text
        )
        
        # We need a small JS script to actually clear the sketchpad widget in Gradio
        clear_btn.click(
            fn=lambda: None,
            inputs=None,
            outputs=canvas
        )
        
    return app


if __name__ == "__main__":
    logger.info("Starting up Real-time Drawing App...")
    
    # Pre-load the default model to fail fast if it doesn't exist
    try:
        logger.info("Pre-loading default model...")
        load_model("CNN - EMNIST (Characters A-Z, a-z, 0-9)")
    except Exception as e:
        logger.warning(f"Could not pre-load model: {e}")
        
    app = create_app()
    app.launch(share=False, server_name="127.0.0.1", server_port=7860)
