"""
GI-NET Ki-67 grade predictor.

This module provides the main prediction interface that combines feature extraction
and the ABMIL ensemble for predicting Ki-67 proliferation grade from H&E images.
"""

import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from model import ABMIL
from feature_extractor import FeatureExtractor


def process_image(
    image: np.ndarray | Image.Image,
    feature_extractor: FeatureExtractor,
    tile_size: int = 1024,
    max_tiles: int = 500,
) -> np.ndarray:
    """
    Process uploaded image into features.

    For small images (<= 1024x1024): Treat as single tile
    For large images: Extract tiles, apply QC, extract features

    Args:
        image: Input image as numpy array or PIL Image
        feature_extractor: Feature extractor instance
        tile_size: Size of tiles to extract from large images
        max_tiles: Maximum number of tiles to process

    Returns:
        Feature array of shape (n_tiles, 1536)
    """
    # Convert PIL Image to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))

    h, w = image.shape[:2]

    # Small image - treat as single tile
    if h <= tile_size and w <= tile_size:
        if HAS_CV2:
            resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        else:
            pil_img = Image.fromarray(image)
            pil_img = pil_img.resize((224, 224), Image.Resampling.LANCZOS)
            resized = np.array(pil_img)
        features = feature_extractor.extract(resized)
        return features.reshape(1, -1)  # (1, 1536)

    # Large image - extract tiles
    tiles = []
    step = tile_size

    for y in range(0, h - tile_size + 1, step):
        for x in range(0, w - tile_size + 1, step):
            tile = image[y : y + tile_size, x : x + tile_size]

            # Quality control: check tissue content
            if HAS_CV2:
                gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
            else:
                gray = np.mean(tile, axis=2).astype(np.uint8)

            tissue_mask = (gray < 220) & (gray > 30)
            tissue_ratio = tissue_mask.sum() / tissue_mask.size

            if tissue_ratio >= 0.3:  # At least 30% tissue
                if HAS_CV2:
                    resized_tile = cv2.resize(
                        tile, (224, 224), interpolation=cv2.INTER_AREA
                    )
                else:
                    pil_tile = Image.fromarray(tile)
                    pil_tile = pil_tile.resize((224, 224), Image.Resampling.LANCZOS)
                    resized_tile = np.array(pil_tile)
                tiles.append(resized_tile)

    # Fallback: use whole image resized if no tiles pass QC
    if not tiles:
        if HAS_CV2:
            resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        else:
            pil_img = Image.fromarray(image)
            pil_img = pil_img.resize((224, 224), Image.Resampling.LANCZOS)
            resized = np.array(pil_img)
        tiles = [resized]

    # Limit number of tiles
    if len(tiles) > max_tiles:
        indices = np.random.choice(len(tiles), max_tiles, replace=False)
        tiles = [tiles[i] for i in indices]

    # Extract features from all tiles
    features = feature_extractor.extract_batch(tiles)
    return features  # (n_tiles, 1536)


class GINETPredictor:
    """
    GI-NET Ki-67 grade predictor using ensemble of ABMIL models.

    This class loads an ensemble of 5 ABMIL models (one per cross-validation fold)
    and combines their predictions for robust grade classification.

    Args:
        model_dir: Directory containing model weight files (model_fold0.pt through model_fold4.pt)
        device: Device to run models on ("cuda" or "cpu")
    """

    def __init__(self, model_dir: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Loading models on {self.device}...")

        # Load ensemble of 5 fold models
        self.models = []
        for i in range(5):
            model_path = os.path.join(model_dir, f"model_fold{i}.pt")
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model file not found: {model_path}. "
                    f"Please ensure all 5 model files (model_fold0.pt through model_fold4.pt) "
                    f"are present in {model_dir}"
                )

            model = ABMIL(feature_dim=1536, hidden_dim=256, n_classes=2, dropout=0.4)
            model.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True)
            )
            model.to(self.device).eval()
            self.models.append(model)
            print(f"  Loaded model fold {i}")

        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(device=str(self.device))
        print("All models loaded successfully.")

    @torch.no_grad()
    def predict(self, image: np.ndarray | Image.Image) -> dict[str, Any]:
        """
        Predict Ki-67 grade from image.

        Args:
            image: Input H&E histopathology image

        Returns:
            Dictionary containing:
                - prediction: Grade prediction ("G1" or "G2+G3")
                - confidence: Confidence score (0-1)
                - prob_g1: Probability of G1
                - prob_g2g3: Probability of G2+G3
                - n_tiles: Number of tiles analyzed
                - interpretation: Clinical interpretation text
        """
        # Extract features from image
        features = process_image(image, self.feature_extractor)
        n_tiles = len(features)

        # Prepare input tensor
        x = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        mask = torch.ones(1, n_tiles).to(self.device)

        # Ensemble prediction - average probabilities from all models
        all_probs = []
        for model in self.models:
            logits, _ = model(x, mask)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            all_probs.append(probs)

        # Average probabilities across ensemble
        avg_probs = np.mean(all_probs, axis=0)
        pred_class = int(np.argmax(avg_probs))
        confidence = float(avg_probs[pred_class])

        # Generate clinical interpretation
        interpretation = self._get_interpretation(pred_class, avg_probs, n_tiles)

        return {
            "prediction": "G1" if pred_class == 0 else "G2+G3",
            "confidence": confidence,
            "prob_g1": float(avg_probs[0]),
            "prob_g2g3": float(avg_probs[1]),
            "n_tiles": n_tiles,
            "interpretation": interpretation,
        }

    def _get_interpretation(
        self, pred_class: int, probs: np.ndarray, n_tiles: int
    ) -> str:
        """
        Generate clinical interpretation based on prediction.

        Args:
            pred_class: Predicted class (0=G1, 1=G2+G3)
            probs: Class probabilities
            n_tiles: Number of tiles analyzed

        Returns:
            Formatted interpretation string with clinical recommendations
        """
        prob_g1 = probs[0] * 100
        prob_g2g3 = probs[1] * 100

        if pred_class == 0:  # G1
            if prob_g1 >= 90:
                confidence_level = "high"
                recommendation = (
                    "Based on H&E morphology, this case strongly suggests G1 (Ki-67 <3%). "
                    "Consider whether Ki-67 IHC is necessary."
                )
            elif prob_g1 >= 70:
                confidence_level = "moderate"
                recommendation = (
                    "H&E features are consistent with G1, but Ki-67 IHC may be warranted "
                    "for confirmation."
                )
            else:
                confidence_level = "low"
                recommendation = (
                    "Prediction is G1 but with low confidence. Ki-67 IHC is recommended."
                )
        else:  # G2+G3
            if prob_g2g3 >= 90:
                confidence_level = "high"
                recommendation = (
                    "H&E morphology strongly suggests elevated proliferation (G2 or G3). "
                    "Ki-67 IHC is recommended to determine precise grade."
                )
            elif prob_g2g3 >= 70:
                confidence_level = "moderate"
                recommendation = (
                    "Features suggest possible G2/G3. Ki-67 IHC is recommended for accurate grading."
                )
            else:
                confidence_level = "low"
                recommendation = (
                    "Prediction is G2+G3 but with low confidence. Ki-67 IHC is essential."
                )

        grade_text = "G1 (Low Grade)" if pred_class == 0 else "G2+G3 (Intermediate/High Grade)"

        interpretation = f"""
## Prediction: {grade_text}

**Confidence Level:** {confidence_level.upper()} ({probs[pred_class]*100:.1f}%)

**Probabilities:**
- G1 (Ki-67 <3%): {prob_g1:.1f}%
- G2+G3 (Ki-67 ≥3%): {prob_g2g3:.1f}%

**Analysis:** {n_tiles} tissue region(s) analyzed.

**Clinical Recommendation:**
{recommendation}

---

**Model Performance (Validation Data):**
- Overall Accuracy: 94.9%
- G1 Sensitivity: 97%
- G2+G3 Sensitivity: 93%

**Disclaimer:** This AI prediction is intended as a clinical decision support tool only. Final grading decisions should be made by a qualified pathologist, ideally with Ki-67 IHC when clinically indicated.
"""
        return interpretation
