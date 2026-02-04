---
title: GI-NET Ki-67 Grade Prediction
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# GI-NET Ki-67 Grade Prediction from H&E

A web application for predicting Ki-67 proliferation grade (G1 vs G2+G3) from H&E histopathology images of gastrointestinal neuroendocrine tumors (GI-NETs).

## Overview

This application uses an Attention-Based Multiple Instance Learning (ABMIL) model ensemble to predict Ki-67 proliferation grade directly from H&E stained histopathology images, without requiring Ki-67 immunohistochemistry.

### Supported Grades

- **G1**: Ki-67 <3% (low proliferation)
- **G2+G3**: Ki-67 ≥3% (intermediate/high proliferation)

### Model Performance

| Metric | Value |
|--------|-------|
| Overall Accuracy | 94.9% |
| G1 Sensitivity | 97% |
| G2+G3 Sensitivity | 93% |
| Cohen's Kappa | 0.90 |

## Usage

Simply upload an H&E histopathology image (TIFF, PNG, or JPG) and click "Predict Grade". The application will:

1. Process the image (tiling for large images)
2. Extract features using H-optimus-0 foundation model
3. Predict grade using the ABMIL ensemble
4. Display results with confidence scores and clinical interpretation

## Technical Details

### Architecture

- **Feature Extractor**: H-optimus-0 (1536-dimensional features)
- **Classifier**: Attention-Based Multiple Instance Learning (ABMIL)
- **Ensemble**: 5-fold cross-validation ensemble

### Image Processing

- **Small images** (≤1024×1024): Processed as single tile
- **Large images**: Divided into 1024×1024 tiles with quality control
  - Minimum 30% tissue content required per tile
  - Maximum 500 tiles per image
  - All tiles resized to 224×224 for feature extraction

## Local Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended) or CPU

### Setup

1. Clone the repository:

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/gi-net-ki67-prediction
cd gi-net-ki67-prediction
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
python app.py
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_DIR` | Path to model weights directory | `./models/ABMIL_binary` |
| `MODEL_REPO` | HF Hub repo ID for weight download | (none) |
| `DEVICE` | Compute device (`cuda` or `cpu`) | `cuda` |

## Project Structure

```
.
├── app.py                 # Gradio web interface
├── model.py               # ABMIL model architecture
├── feature_extractor.py   # H-optimus-0 wrapper
├── predictor.py           # GINETPredictor class
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── models/
    └── ABMIL_binary/
        ├── model_fold0.pt
        ├── model_fold1.pt
        ├── model_fold2.pt
        ├── model_fold3.pt
        └── model_fold4.pt
```

## Disclaimer

This tool is for **research and clinical decision support only**. It is not intended for primary diagnosis. Final grading decisions should be made by a qualified pathologist, ideally with Ki-67 immunohistochemistry when clinically indicated.

## License

MIT License
