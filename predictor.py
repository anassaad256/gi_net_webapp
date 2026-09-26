"""
GI-NET Ki-67 grade predictor.

This module provides the main prediction interface that combines feature extraction
and the ABMIL ensemble for predicting Ki-67 proliferation grade from H&E images.
"""

import os
from collections.abc import Iterable
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


def download_weights_from_hub(repo_id: str, local_dir: str) -> None:
    """
    Download model weights from Hugging Face Hub.

    Args:
        repo_id: Hugging Face Hub repository ID (e.g., "username/gi-net-weights")
        local_dir: Local directory to save weights
    """
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to download weights. "
            "Install with: pip install huggingface_hub"
        )

    os.makedirs(local_dir, exist_ok=True)

    # List files in the repo and download model weights
    try:
        files = list_repo_files(repo_id)
        model_files = [f for f in files if f.startswith("model_fold") and f.endswith(".pt")]

        if not model_files:
            raise FileNotFoundError(f"No model files found in {repo_id}")

        print(f"Downloading {len(model_files)} model files from {repo_id}...")
        for filename in model_files:
            print(f"  Downloading {filename}...")
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
            )
        print("Download complete.")

    except Exception as e:
        raise RuntimeError(f"Failed to download weights from {repo_id}: {e}")


TILE_SIZE = 1024
MODEL_INPUT_SIZE = 224
MAX_TILES = 500
MIN_TISSUE_RATIO = 0.3
MIN_RECOMMENDED_TILES = 50

RESEARCH_USE_NOTICE = (
    "**⚠️ For research and testing use only. Not for clinical use.** "
    "This tool has not been validated or approved for diagnosis or patient care."
)


def _to_rgb_array(image: np.ndarray | Image.Image) -> np.ndarray:
    """Convert a PIL Image or array to an RGB uint8 numpy array."""
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    if image.ndim == 2:
        return np.stack([image] * 3, axis=-1)
    return image[..., :3]


def _resize(image: np.ndarray, size: int) -> np.ndarray:
    """Resize an RGB array to size x size."""
    if HAS_CV2:
        return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    pil_img = Image.fromarray(image).resize((size, size), Image.Resampling.LANCZOS)
    return np.array(pil_img)


def tissue_ratio(tile: np.ndarray) -> float:
    """Fraction of pixels that look like tissue (same QC rule as training)."""
    if HAS_CV2:
        gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
    else:
        gray = np.mean(tile, axis=2).astype(np.uint8)
    tissue_mask = (gray < 220) & (gray > 30)
    return float(tissue_mask.sum() / tissue_mask.size)


def extract_tiles(
    images: Iterable[tuple[str, np.ndarray | Image.Image]],
    tile_size: int = TILE_SIZE,
    max_tiles: int = MAX_TILES,
    min_tissue_ratio: float = MIN_TISSUE_RATIO,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Turn a set of uploaded images from one case into a bag of tiles.

    Images no larger than tile_size are treated as one tile each (the expected
    input: 1024x1024 tiles at 40x). Larger images are cut into non-overlapping
    tile_size tiles. Every tile goes through the same tissue QC used in training,
    and the bag is randomly sampled down to max_tiles, as in training.

    Args:
        images: (name, image) pairs belonging to one case
        tile_size: Tile size used to cut large images
        max_tiles: Maximum number of tiles kept per case
        min_tissue_ratio: Minimum tissue fraction for a tile to pass QC
        seed: Random seed for reproducible sampling

    Returns:
        tiles: List of dicts with "name" and "image" (224x224 RGB array)
        stats: Counts of candidate, rejected and sampled-out tiles
    """
    # Tiles are QC'd and downsized as they are cut, so only one full-size
    # image is held in memory at a time when `images` is a generator.
    n_candidates = 0
    tiles = []
    for name, image in images:
        image = _to_rgb_array(image)
        h, w = image.shape[:2]

        if h <= tile_size and w <= tile_size:
            candidates = [(name, image)]
        else:
            candidates = [
                (f"{name} @ ({x}, {y})", image[y : y + tile_size, x : x + tile_size])
                for y in range(0, h - tile_size + 1, tile_size)
                for x in range(0, w - tile_size + 1, tile_size)
            ]

        for tile_name, tile in candidates:
            n_candidates += 1
            if tissue_ratio(tile) >= min_tissue_ratio:
                tiles.append({"name": tile_name, "image": _resize(tile, MODEL_INPUT_SIZE)})

    n_passed_qc = len(tiles)
    if len(tiles) > max_tiles:
        rng = np.random.default_rng(seed)
        keep = sorted(rng.choice(len(tiles), max_tiles, replace=False))
        tiles = [tiles[i] for i in keep]

    stats = {
        "n_candidates": n_candidates,
        "n_rejected_qc": n_candidates - n_passed_qc,
        "n_sampled_out": n_passed_qc - len(tiles),
        "n_used": len(tiles),
    }
    return tiles, stats


class GINETPredictor:
    """
    GI-NET Ki-67 grade predictor using ensemble of ABMIL models.

    This class loads an ensemble of 5 ABMIL models (one per cross-validation fold)
    and combines their predictions for robust grade classification.

    Args:
        model_dir: Directory containing model weight files (model_fold0.pt through model_fold4.pt)
        model_repo: Optional Hugging Face Hub repository ID to download weights from
        device: Device to run models on ("cuda" or "cpu")
    """

    def __init__(
        self,
        model_dir: str,
        model_repo: str | None = None,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Download weights from HF Hub if specified and not already present
        if model_repo:
            first_model = os.path.join(model_dir, "model_fold0.pt")
            if not os.path.exists(first_model):
                print(f"Downloading weights from Hugging Face Hub: {model_repo}")
                download_weights_from_hub(model_repo, model_dir)

        # Load ensemble of 5 fold models
        print("Loading ABMIL ensemble models...")
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
        print("Loading H-optimus-0 feature extractor...")
        self.feature_extractor = FeatureExtractor(device=str(self.device))
        print("All models loaded successfully!")

    @torch.no_grad()
    def predict_case(
        self, images: Iterable[tuple[str, np.ndarray | Image.Image]]
    ) -> dict[str, Any]:
        """
        Predict the Ki-67 grade of one case from a set of H&E tiles.

        All tiles are pooled into a single bag and scored together by the ABMIL
        ensemble, matching how the model was trained and validated (case level,
        up to 500 tiles per case).

        Args:
            images: (name, image) pairs from one case. Each image is a
                1024x1024 tile at 40x, or a larger region that gets tiled.

        Returns:
            Dictionary containing:
                - prediction: Grade prediction ("G1" or "G2+G3")
                - confidence: Confidence score (0-1)
                - prob_g1: Probability of G1
                - prob_g2g3: Probability of G2+G3
                - n_tiles: Number of tiles analyzed
                - tile_stats: Counts from tiling and QC (see extract_tiles)
                - tiles: Per-tile dicts with "name", "image" and "attention",
                  sorted by attention (highest first)
                - interpretation: Clinical interpretation text
        """
        tiles, tile_stats = extract_tiles(images)
        if not tiles:
            raise ValueError(
                f"None of the {tile_stats['n_candidates']} tile(s) passed tissue QC "
                f"(at least {MIN_TISSUE_RATIO:.0%} tissue). Upload tiles that are "
                "mostly tumor rather than background or glass."
            )
        n_tiles = len(tiles)

        features = self.feature_extractor.extract_batch([t["image"] for t in tiles])
        x = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        mask = torch.ones(1, n_tiles).to(self.device)

        # Ensemble prediction - average probabilities and attention over folds
        all_probs = []
        all_attn = []
        for model in self.models:
            logits, attn = model(x, mask)
            all_probs.append(F.softmax(logits, dim=1).cpu().numpy()[0])
            all_attn.append(attn.cpu().numpy()[0])

        avg_probs = np.mean(all_probs, axis=0)
        avg_attn = np.mean(all_attn, axis=0)
        pred_class = int(np.argmax(avg_probs))
        confidence = float(avg_probs[pred_class])

        for tile, weight in zip(tiles, avg_attn):
            tile["attention"] = float(weight)
        tiles.sort(key=lambda t: t["attention"], reverse=True)

        interpretation = self._get_interpretation(pred_class, avg_probs, n_tiles)

        return {
            "prediction": "G1" if pred_class == 0 else "G2+G3",
            "confidence": confidence,
            "prob_g1": float(avg_probs[0]),
            "prob_g2g3": float(avg_probs[1]),
            "n_tiles": n_tiles,
            "tile_stats": tile_stats,
            "tiles": tiles,
            "interpretation": interpretation,
        }

    def predict(self, image: np.ndarray | Image.Image) -> dict[str, Any]:
        """Predict from a single image; see predict_case for the multi-tile path."""
        return self.predict_case([("image", image)])

    def _get_interpretation(
        self, pred_class: int, probs: np.ndarray, n_tiles: int
    ) -> str:
        """
        Generate a research-use interpretation of the prediction.

        Args:
            pred_class: Predicted class (0=G1, 1=G2+G3)
            probs: Class probabilities
            n_tiles: Number of tiles analyzed

        Returns:
            Formatted interpretation string
        """
        prob_g1 = probs[0] * 100
        prob_g2g3 = probs[1] * 100
        confidence = probs[pred_class] * 100

        if confidence >= 90:
            confidence_level = "high"
        elif confidence >= 70:
            confidence_level = "moderate"
        else:
            confidence_level = "low"

        if pred_class == 0:  # G1
            summary = "The model's output favors G1 (Ki-67 <3%) for this set of tiles."
        else:  # G2+G3
            summary = "The model's output favors G2+G3 (Ki-67 ≥3%) for this set of tiles."
        if confidence_level == "low":
            summary += " The two classes are close, so treat this output as indeterminate."

        tile_warning = ""
        if n_tiles < MIN_RECOMMENDED_TILES:
            tile_warning = (
                f"\n**Caution:** only {n_tiles} tile(s) were analyzed. The model was "
                f"validated on whole cases (up to {MAX_TILES} tiles each); predictions "
                f"from fewer than {MIN_RECOMMENDED_TILES} tiles are much less reliable.\n"
            )

        grade_text = "G1 (Low Grade)" if pred_class == 0 else "G2+G3 (Intermediate/High Grade)"

        interpretation = f"""
{RESEARCH_USE_NOTICE}

### Model output: {grade_text}

**Confidence Level:** {confidence_level.upper()} ({confidence:.1f}%)

**Probabilities:**
- G1 (Ki-67 <3%): {prob_g1:.1f}%
- G2+G3 (Ki-67 ≥3%): {prob_g2g3:.1f}%

**Analysis:** {n_tiles} tile(s) aggregated with attention-based MIL.
{tile_warning}
**Summary:** {summary}

---

**Model Performance (held-out test set, n=44 cases, single institution, case level):**
- Balanced Accuracy: 94.9%
- G1 Sensitivity: 97%
- G2+G3 Sensitivity: 93%

These results come from a single-institution research dataset and have not been
validated for clinical use. Do not use this output to grade, diagnose, or guide
the treatment of any patient.
"""
        return interpretation
