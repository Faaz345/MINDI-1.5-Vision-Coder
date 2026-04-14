"""
MINDI 1.5 Vision-Coder — Vision Encoder

Uses CLIP ViT-L/14 to encode UI screenshots into embeddings
that the coding model can understand and critique.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class VisionEncoder(nn.Module):
    """CLIP-based vision encoder for UI screenshot understanding."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        projection_dim: int = 768,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir or Path("./checkpoints/vision")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load CLIP model and processor
        self.clip: CLIPModel = CLIPModel.from_pretrained(
            model_name, cache_dir=str(self.cache_dir)
        )
        self.processor: CLIPProcessor = CLIPProcessor.from_pretrained(
            model_name, cache_dir=str(self.cache_dir)
        )

        # Freeze CLIP backbone — we only train the projection layer
        for param in self.clip.parameters():
            param.requires_grad = False

        # Trainable projection: CLIP hidden → LLM embedding space
        clip_hidden_size: int = self.clip.config.vision_config.hidden_size  # 1024
        self.projection = nn.Sequential(
            nn.Linear(clip_hidden_size, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )

        self.to(self.device)

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode a PIL image into a projected embedding tensor."""
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            vision_outputs = self.clip.vision_model(**inputs)
            # Use [CLS] token embedding
            cls_embedding = vision_outputs.last_hidden_state[:, 0, :]

        # Project into LLM embedding space (this part IS trainable)
        projected = self.projection(cls_embedding)
        return projected

    def encode_screenshot(self, screenshot_path: Path) -> torch.Tensor:
        """Load a screenshot from disk and encode it."""
        if not screenshot_path.exists():
            raise FileNotFoundError(f"Screenshot not found: {screenshot_path}")

        image = Image.open(screenshot_path).convert("RGB")
        return self.encode_image(image)

    def save_projection(self, save_dir: Optional[Path] = None) -> Path:
        """Save only the trainable projection weights."""
        save_path = save_dir or self.cache_dir / "projection"
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.projection.state_dict(), save_path / "projection.pt")
        return save_path

    def load_projection(self, load_dir: Path) -> None:
        """Load projection weights from disk."""
        weights_path = load_dir / "projection.pt"
        if not weights_path.exists():
            raise FileNotFoundError(f"Projection weights not found: {weights_path}")
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.projection.load_state_dict(state_dict)
