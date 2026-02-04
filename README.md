# GI-NET Ki-67 Grade Prediction Web Application

A web application for predicting Ki-67 proliferation grade (G1 vs G2+G3) from H&E histopathology images of gastrointestinal neuroendocrine tumors (GI-NETs).

## Overview

This application uses an Attention-Based Multiple Instance Learning (ABMIL) model ensemble to predict Ki-67 proliferation grade directly from H&E stained histopathology images, without requiring Ki-67 immunohistochemistry.

### Supported Grades

- **G1**: Ki-67 <3% (low proliferation)
- **G2+G3**: Ki-67 ≥3% (intermediate/high proliferation)

### Model Performance

- Overall Accuracy: 94.9%
- G1 Sensitivity: 97%
- G2+G3 Sensitivity: 93%
- Cohen's Kappa: 0.90

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended) or CPU

### Setup

1. Clone or download this repository:

```bash
cd gi_net_webapp
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy model weights to the `models/ABMIL_binary/` directory:

```
models/
└── ABMIL_binary/
    ├── model_fold0.pt
    ├── model_fold1.pt
    ├── model_fold2.pt
    ├── model_fold3.pt
    └── model_fold4.pt
```

## Usage

### Running the Web Application

```bash
python app.py
```

The application will be available at:
- Local: `http://localhost:7860`
- Public: A shareable link will be displayed in the terminal (if `share=True`)

### Environment Variables

- `MODEL_DIR`: Path to model weights directory (default: `./models/ABMIL_binary`)
- `DEVICE`: Device to use for inference - `cuda` or `cpu` (default: `cuda`)

Example:
```bash
MODEL_DIR=/path/to/models DEVICE=cpu python app.py
```

## Project Structure

```
gi_net_webapp/
├── app.py                 # Main Gradio web application
├── model.py               # ABMIL model architecture
├── feature_extractor.py   # H-optimus-0 feature extraction
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

## Technical Details

### Feature Extraction

The application uses [H-optimus-0](https://huggingface.co/bioptimus/H-optimus-0), a vision transformer foundation model trained on histopathology images, to extract 1536-dimensional feature vectors from image tiles.

### ABMIL Architecture

The classifier uses Attention-Based Multiple Instance Learning:

1. **Encoder**: Projects 1536-dim features to 256-dim hidden space with LayerNorm and GELU activation
2. **Attention**: Computes attention weights for each tile using a two-layer network with Tanh activation
3. **Aggregation**: Weighted sum of tile features based on attention scores
4. **Classifier**: Two-layer MLP with LayerNorm and GELU for final classification

### Image Processing Pipeline

- **Small images** (≤1024×1024): Treated as a single tile
- **Large images**: Divided into 1024×1024 tiles with quality control:
  - Tiles must contain at least 30% tissue (non-background content)
  - Maximum 500 tiles per image
  - All tiles resized to 224×224 for feature extraction

### Ensemble Prediction

The final prediction is an average of probabilities from 5 models trained using 5-fold cross-validation.

## Input Requirements

- **Formats**: TIFF, PNG, JPG
- **Content**: H&E stained histopathology images
- **Size**: Any resolution (automatically processed)

## Output

- **Predicted grade**: G1 or G2+G3
- **Confidence percentage**: Model certainty in the prediction
- **Probability for each class**: Detailed probability breakdown
- **Clinical interpretation**: Recommendations based on confidence level

## Notes

- First run will download H-optimus-0 (~600MB) automatically
- GPU is recommended for faster inference
- Model weights are approximately 10MB total for all 5 folds

## Disclaimer

This AI prediction tool is intended for research and clinical decision support only. It is not intended for primary diagnosis. Final grading decisions should be made by a qualified pathologist, ideally with Ki-67 immunohistochemistry when clinically indicated.

## License

[Add your license here]
