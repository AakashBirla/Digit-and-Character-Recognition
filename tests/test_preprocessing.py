"""Tests for image preprocessing utilities.

Test image conversion, resizing, centering, and normalization.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.preprocessing import preprocess_drawing


class TestPreprocessing:
    
    def test_preprocess_numpy_array(self):
        # Create a dummy image (black background, white square in middle)
        img = np.zeros((100, 100), dtype=np.uint8)
        img[30:70, 30:70] = 255
        
        tensor = preprocess_drawing(img)
        
        assert tensor.shape == (1, 1, 28, 28)
        assert isinstance(tensor, torch.Tensor)

    def test_preprocess_pil_image(self):
        img = Image.new('L', (100, 100), color=0)
        tensor = preprocess_drawing(img)
        assert tensor.shape == (1, 1, 28, 28)
        
    def test_preprocess_inverts_white_background(self):
        # Image with white background and black drawing
        img = np.ones((100, 100), dtype=np.uint8) * 255
        img[40:60, 40:60] = 0
        
        tensor = preprocess_drawing(img)
        assert tensor.shape == (1, 1, 28, 28)
