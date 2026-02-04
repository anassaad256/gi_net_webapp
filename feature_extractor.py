"""
Feature extraction using H-optimus-0 foundation model.

This module provides a wrapper around the H-optimus-0 model from Bioptimus
for extracting features from histopathology image tiles.
"""

import numpy as np
import torch
import timm
from PIL import Image

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class FeatureExtractor:
    """
    Feature extractor using H-optimus-0 foundation model.

    H-optimus-0 is a vision transformer trained on histopathology images
    that produces 1536-dimensional feature vectors.

    Args:
        device: Device to run the model on ("cuda" or "cpu")
    """

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Loading H-optimus-0 on {self.device}...")

        # Load pre-trained H-optimus-0 model
        self.model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True)
        self.model = self.model.to(self.device).eval()

        # Normalization values specific to H-optimus-0
        self.mean = (
            torch.tensor([0.707223, 0.578729, 0.703617])
            .view(1, 3, 1, 1)
            .to(self.device)
        )
        self.std = (
            torch.tensor([0.211883, 0.230117, 0.177517])
            .view(1, 3, 1, 1)
            .to(self.device)
        )

        print("H-optimus-0 loaded successfully.")

    def preprocess(self, image: np.ndarray | Image.Image) -> torch.Tensor:
        """
        Preprocess image for H-optimus-0.

        Args:
            image: Input image as numpy array (H, W, 3) or PIL Image

        Returns:
            Preprocessed tensor of shape (1, 3, 224, 224)
        """
        # Convert PIL Image to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))

        # Resize to 224x224 if needed
        if image.shape[0] != 224 or image.shape[1] != 224:
            if HAS_CV2:
                image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            else:
                pil_img = Image.fromarray(image)
                pil_img = pil_img.resize((224, 224), Image.Resampling.LANCZOS)
                image = np.array(pil_img)

        # Convert to tensor: (H, W, C) -> (1, C, H, W), scale to [0, 1]
        tensor = (
            torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        )
        tensor = tensor.to(self.device)

        # Normalize with H-optimus-0 specific values
        tensor = (tensor - self.mean) / self.std

        return tensor

    @torch.no_grad()
    def extract(self, image: np.ndarray | Image.Image) -> np.ndarray:
        """
        Extract features from a single image.

        Args:
            image: Input image as numpy array or PIL Image

        Returns:
            Feature vector of shape (1536,)
        """
        tensor = self.preprocess(image)
        features = self.model(tensor)
        return features.cpu().numpy().squeeze()

    @torch.no_grad()
    def extract_batch(
        self, images: list[np.ndarray | Image.Image], batch_size: int = 8
    ) -> np.ndarray:
        """
        Extract features from multiple images.

        Args:
            images: List of input images
            batch_size: Number of images to process at once

        Returns:
            Feature array of shape (n_images, 1536)
        """
        all_features = []

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            tensors = torch.cat([self.preprocess(img) for img in batch], dim=0)
            features = self.model(tensors)
            all_features.append(features.cpu().numpy())

        return np.concatenate(all_features, axis=0)
