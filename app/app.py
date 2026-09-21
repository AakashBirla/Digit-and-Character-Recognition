#!/usr/bin/env python3
"""Stage 5: Real-time drawing and recognition interface (Flask).

Provides a Flask web interface for users to draw digits and characters
and get real-time predictions from the trained models.

Usage:
    python app/app.py
"""

import base64
import io
import logging
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request
import torch
from PIL import Image

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

# Initialize Flask app
app = Flask(__name__)

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
        
    config = MODELS.get(model_name)
    if not config:
        raise ValueError(f"Unknown model: {model_name}")
        
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


@app.route("/")
def index():
    """Render the main drawing page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction requests from the frontend."""
    data = request.json
    if not data or "image" not in data or "model" not in data:
        return jsonify({"error": "Invalid request parameters"}), 400
        
    image_b64 = data["image"]
    model_name = data["model"]
    
    # Decode base64 image
    try:
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        logger.error(f"Image decode error: {e}")
        return jsonify({"error": "Failed to decode image"}), 400

    # Load model
    try:
        model, class_mapping, is_emnist = load_model(model_name)
    except Exception as e:
        logger.error(f"Model load error: {e}")
        return jsonify({"error": str(e)}), 500

    # Preprocess and predict
    try:
        # Preprocess the drawing
        tensor = preprocess_drawing(image, is_emnist=is_emnist)
        logger.info(f"Processed tensor shape: {tensor.shape}, non-zero pixels: {torch.count_nonzero(tensor).item()}")
        
        # Check if the tensor is entirely empty
        if torch.count_nonzero(tensor).item() == 0 or torch.allclose(tensor, torch.min(tensor)):
            return jsonify({"error": "Canvas is empty!"}), 400
            
        tensor = tensor.to(device)
        
        # Predict
        with torch.no_grad():
            output = model(tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            
        # Get top 3 predictions
        top_prob, top_class = torch.topk(probabilities, 3)
        
        top_prob = top_prob.cpu().numpy()
        top_class = top_class.cpu().numpy()
        
        predictions = []
        for i in range(3):
            char = class_mapping[top_class[i]]
            prob = float(top_prob[i] * 100)
            predictions.append({
                "char": char,
                "prob": prob
            })
            
        return jsonify({"predictions": predictions})
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": f"Error processing image: {e}"}), 500


if __name__ == "__main__":
    logger.info("Starting up Real-time Drawing App (Flask)...")
    
    # Pre-load the default model to fail fast if it doesn't exist
    try:
        logger.info("Pre-loading default model...")
        load_model("CNN - EMNIST (Characters A-Z, a-z, 0-9)")
    except Exception as e:
        logger.warning(f"Could not pre-load model: {e}")
        
    # Start Flask server
    app.run(host="127.0.0.1", port=5000, debug=True)
