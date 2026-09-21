# Digit and Character Recognition System

A complete handwritten digit and character recognition system built from scratch using **PyTorch**. The project demonstrates the full machine learning pipeline from data loading through model training to real-time inference.

## Features

- ✅ MNIST digit dataset loading and visualization
- ✅ EMNIST character dataset preparation and documentation
- ✅ Configurable data pipeline with DataLoaders
- ✅ Image normalization and preprocessing
- ✅ Dataset analysis and visualization utilities
- ✅ Reproducible experiments with seed management
- ✅ Automatic GPU/MPS/CPU device detection
- 🔲 MLP baseline classifier (Stage 2)
- 🔲 CNN classifier (Stage 3)
- 🔲 Training pipeline with metrics (Stage 4)
- 🔲 Real-time drawing interface (Stage 5)

## Project Structure

```text
digit-character-recognition/
│
├── data/                        # Downloaded datasets (gitignored)
│   └── README.md                # Dataset documentation
│
├── models/                      # Neural network architectures
│   └── __init__.py
│
├── src/                         # Source code
│   ├── data/                    # Data loading modules
│   │   ├── mnist_loader.py      # MNIST dataset pipeline
│   │   └── emnist_loader.py     # EMNIST dataset pipeline
│   │
│   ├── training/                # Training modules (Stage 2+)
│   ├── evaluation/              # Evaluation metrics (Stage 4+)
│   ├── inference/               # Prediction pipeline (Stage 5)
│   │
│   └── utils/                   # Utility modules
│       ├── visualization.py     # Plotting and visualization
│       ├── preprocessing.py     # Image preprocessing
│       └── seed.py              # Reproducibility utilities
│
├── app/                         # Drawing application (Stage 5)
├── tests/                       # Test suite
├── checkpoints/                 # Model checkpoints (Stage 2+)
├── results/
│   ├── figures/                 # Generated plots
│   └── metrics/                 # Experiment results
│
├── explore_data.py              # Stage 1: Data exploration script
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── LICENSE                      # MIT License
└── README.md                    # This file
```

## Installation

```bash
# Clone the repository
git clone https://github.com/AakashBirla/Digit-and-Character-Recognition.git
cd Digit-and-Character-Recognition

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Datasets

### MNIST (Handwritten Digits)

| Property | Value |
|---|---|
| Images | 70,000 (60,000 train + 10,000 test) |
| Image size | 28×28 grayscale |
| Classes | 10 (digits 0–9) |
| Normalization | mean=0.1307, std=0.3081 |

### EMNIST Balanced (Handwritten Characters)

| Property | Value |
|---|---|
| Split | Balanced |
| Classes | 47 |
| Class breakdown | 10 digits + 26 uppercase + 11 lowercase |
| Image size | 28×28 grayscale |
| Orientation | Transposed (auto-corrected in loader) |

**EMNIST Balanced class mapping:**
- Classes 0–9: Digits `0`–`9`
- Classes 10–35: Uppercase `A`–`Z`
- Classes 36–46: Visually distinct lowercase: `a, b, d, e, f, g, h, n, q, r, t`

**Key differences from MNIST:**
- EMNIST images are transposed and require orientation correction
- EMNIST Balanced has 47 classes vs MNIST's 10
- The Balanced split merges visually ambiguous lowercase letters with their uppercase counterparts

## Stage 1: Data Pipeline

### How to Run

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the data exploration script
python explore_data.py

# Run tests
python -m pytest tests/ -v
```

### What it does

1. Downloads the MNIST dataset automatically
2. Loads training and test sets with normalization
3. Creates DataLoaders with configurable batch sizes
4. Prints dataset statistics:
   - Image shape
   - Number of training/test samples
   - Number of classes
   - Class distribution
5. Generates visualization plots saved to `results/figures/`:
   - Sample images grid
   - Class distribution bar chart
   - Batch samples grid
   - Pixel intensity distribution histogram

### Results

Plots are saved to `results/figures/`:
- `mnist_sample_images.png` — Grid of sample digit images
- `mnist_class_distribution.png` — Class distribution bar chart
- `mnist_batch_samples.png` — Sample batch from DataLoader
- `mnist_pixel_distribution.png` — Pixel intensity histogram

## Project Evolution

| Stage | Description | Status |
|---|---|---|
| **Stage 1** | MNIST/EMNIST data pipeline and visualization | ✅ Complete |
| **Stage 2** | MLP baseline classifier | 🔲 Pending |
| **Stage 3** | CNN digit and character classifier | 🔲 Pending |
| **Stage 4** | Training pipeline and evaluation metrics | 🔲 Pending |
| **Stage 5** | Real-time drawing recognition app | 🔲 Pending |

## Technology Stack

- **Python** 3.12+
- **PyTorch** 2.14+ (with CUDA 13.0 support)
- **Torchvision** 0.29+
- **NumPy**, **Matplotlib**, **Pandas**
- **OpenCV** (headless)
- **Pillow**, **scikit-learn**, **tqdm**

## Future Improvements

- Data augmentation (rotation, scaling, elastic deformation)
- Better CNN architectures (ResNet, EfficientNet)
- Batch normalization and dropout
- Learning-rate scheduling
- Transfer learning
- Transformer-based OCR
- Word/sentence recognition
- ONNX export for production
- TensorRT inference optimization
- Web deployment (Flask/FastAPI + React)

## License

MIT License — see [LICENSE](LICENSE) for details.
