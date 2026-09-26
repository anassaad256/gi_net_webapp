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

A web application for predicting Ki-67 proliferation grade (G1 vs G2+G3) from H&E histopathology tiles of gastrointestinal neuroendocrine tumors (GI-NETs).

> **⚠️ For research and testing use only. Not for clinical use.** This tool has not been validated or approved for diagnosis, grading, or any patient-care decision.

## Overview

This application uses an Attention-Based Multiple Instance Learning (ABMIL) model ensemble to predict Ki-67 proliferation grade from a set of H&E tiles from one case. As in the published study, all tiles from a case are aggregated into a single case-level prediction.

### Supported Grades

- **G1**: Ki-67 <3% (low proliferation)
- **G2+G3**: Ki-67 ≥3% (intermediate/high proliferation)

### Model Performance

Held-out test set, n=44 cases, single institution, case level:

| Metric | Value |
|--------|-------|
| Balanced Accuracy | 94.9% |
| G1 Sensitivity | 97% |
| G2+G3 Sensitivity | 93% |
| Cohen's Kappa | 0.90 |

## Usage

Upload tiles from **one case** (multiple PNG/JPG/TIFF files, or a single .zip) and click "Predict Case Grade".

- Tiles should be 1024×1024 px at 40× magnification, as in training; larger images are cut into 1024×1024 tiles.
- Ideally upload 100 to 500 tiles sampled across the tumor. Fewer than 50 tiles triggers a reliability warning; a single tile is not a meaningful input (tile-level accuracy in the study was about 69%).

The application will:

1. Pool all tiles, reject tiles with <30% tissue, and randomly sample down to 500 tiles (fixed seed)
2. Extract features for each tile using the H-optimus-0 foundation model
3. Score the whole bag with the 5-fold ABMIL ensemble
4. Show the case-level prediction, the most-attended tiles, and attention for every tile

## Technical Details

### Architecture

- **Feature Extractor**: H-optimus-0 (1536-dimensional features)
- **Classifier**: Attention-Based Multiple Instance Learning (ABMIL)
- **Ensemble**: 5-fold cross-validation ensemble

### Image Processing

- **Tiles** (≤1024×1024): each uploaded image is one tile
- **Large images**: divided into non-overlapping 1024×1024 tiles
- **Quality control**: minimum 30% tissue content per tile
- **Maximum 500 tiles per case** (random sample if more), matching training
- All tiles resized to 224×224 for feature extraction
- **Attention**: averaged across the 5 fold models and shown per tile

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

This tool is for **research and testing use only. It is not for clinical use.** It is not a medical device and has not been validated or approved for diagnosis, grading, or treatment decisions. It was developed on a single-institution dataset. Ki-67 grading must be performed by a qualified pathologist using standard methods, including Ki-67 immunohistochemistry.

## License

MIT License
