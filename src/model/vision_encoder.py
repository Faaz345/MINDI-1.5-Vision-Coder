"""
MINDI 1.5 Vision-Coder — Vision Encoder

Uses CLIP ViT-L/14 (frozen) to encode UI screenshots into 256 visual
tokens projected from 1024 → 4096 to match the Qwen hidden dimension.
Output shape: (batch, 256, 4096).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModel


class VisionEncoder(nn.Module):
    """
    CLIP ViT-L/14 vision encoder for MINDI 1.5.

    Extracts ALL 256 patch tokens (excludes CLS) from CLIP and
    projects them from 1024 → 4096 to match Qwen2.5 hidden_size.
    The CLIP backbone is frozen; only the projection layer trains.
    """

    NUM_PATCHES: int = 256  # ViT-L/14: 16×16 patches from 224×224

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        llm_hidden_size: int = 4096,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        torch_dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Initialize the vision encoder.

        Args:
            model_name: HuggingFace CLIP vision model identifier.
            llm_hidden_size: Target projection dimension (must match LLM hidden_size).
            device: Target device ('cuda', 'cpu', or None for auto).
            cache_dir: Local directory for model weight cache.
            torch_dtype: Data type for CLIP weights (projection always float32).
        """
        super().__init__()
        self.model_name = model_name
        self.llm_hidden_size = llm_hidden_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./checkpoints/vision")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load CLIP vision model (no text tower) and image processor
        print(f"[VisionEncoder] Loading {model_name} ...")
        self.clip = CLIPVisionModel.from_pretrained(
            model_name,
            cache_dir=str(self.cache_dir),
            torch_dtype=torch_dtype,
        )
        self.image_processor = CLIPImageProcessor.from_pretrained(
            model_name,
            cache_dir=str(self.cache_dir),
        )

        # Freeze entire CLIP backbone
        for param in self.clip.parameters():
            param.requires_grad = False
        self.clip.eval()

        # Trainable projection: CLIP hidden (1024) → LLM hidden (4096)
        clip_hidden_size: int = self.clip.config.hidden_size  # 1024
        self.projection = nn.Linear(clip_hidden_size, self.llm_hidden_size)

        self.to(self.device)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[VisionEncoder] Loaded — {clip_hidden_size} → {self.llm_hidden_size}")
        print(f"  Trainable: {trainable:,}  |  Total: {total:,}")

    def encode_image(self, image: Optional[Image.Image]) -> Optional[torch.Tensor]:
        """
        Encode a single PIL image into projected patch token embeddings.

        Args:
            image: A PIL Image (RGB), or None.

        Returns:
            Tensor of shape (1, 256, 4096) or None if input is None.
        """
        if image is None:
            return None

        inputs = self.image_processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=self.device, dtype=self.clip.dtype)

        with torch.no_grad():
            vision_outputs = self.clip(pixel_values=pixel_values)
            # last_hidden_state: (batch, 257, 1024) — 1 CLS + 256 patches
            patch_tokens = vision_outputs.last_hidden_state[:, 1:, :]  # (1, 256, 1024)

        # Project into LLM embedding space (trainable)
        projected = self.projection(patch_tokens.float())  # (1, 256, 4096)
        return projected

    def encode_batch(self, images: list[Optional[Image.Image]]) -> list[Optional[torch.Tensor]]:
        """
        Encode a batch of images. None entries pass through as None.

        Args:
            images: List of PIL Images or Nones.

        Returns:
            List of tensors (1, 256, 4096) or Nones matching input order.
        """
        results: list[Optional[torch.Tensor]] = [None] * len(images)
        valid_indices = [i for i, img in enumerate(images) if img is not None]

        if not valid_indices:
            return results

        valid_images = [images[i] for i in valid_indices]
        inputs = self.image_processor(images=valid_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=self.device, dtype=self.clip.dtype)

        with torch.no_grad():
            vision_outputs = self.clip(pixel_values=pixel_values)
            patch_tokens = vision_outputs.last_hidden_state[:, 1:, :]  # (N, 256, 1024)

        projected = self.projection(patch_tokens.float())  # (N, 256, 4096)

        for batch_idx, orig_idx in enumerate(valid_indices):
            results[orig_idx] = projected[batch_idx].unsqueeze(0)  # (1, 256, 4096)

        return results

    def encode_screenshot(self, screenshot_path: Path) -> Optional[torch.Tensor]:
        """
        Load a screenshot from disk and encode it.

        Args:
            screenshot_path: Path to image file.

        Returns:
            Tensor of shape (1, 256, 4096).
        """
        path = Path(screenshot_path)
        if not path.exists():
            raise FileNotFoundError(f"Screenshot not found: {path}")
        image = Image.open(path).convert("RGB")
        return self.encode_image(image)

    def save_projection(self, save_dir: Optional[Path] = None) -> Path:
        """
        Save only the trainable projection weights.

        Args:
            save_dir: Directory to save to. Defaults to cache_dir/projection.

        Returns:
            Path where weights were saved.
        """
        save_path = Path(save_dir) if save_dir else self.cache_dir / "projection"
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.projection.state_dict(), save_path / "projection.pt")
        print(f"[VisionEncoder] Projection saved to {save_path}")
        return save_path

    def load_projection(self, load_dir: Path) -> None:
        """
        Load projection weights from disk.

        Args:
            load_dir: Directory containing projection.pt.
        """
        weights_path = Path(load_dir) / "projection.pt"
        if not weights_path.exists():
            raise FileNotFoundError(f"Projection weights not found: {weights_path}")
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.projection.load_state_dict(state_dict)
        print(f"[VisionEncoder] Projection loaded from {load_dir}")

    def get_num_visual_tokens(self) -> int:
        """Return the number of visual tokens produced per image (256)."""
        return self.NUM_PATCHES


# ── Test block ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  MINDI 1.5 — Vision Encoder Test")
    print("=" * 60)
    print()

    # 1. Initialize encoder
    encoder = VisionEncoder(
        model_name="openai/clip-vit-large-patch14",
        llm_hidden_size=4096,
    )

    # 2. Create a dummy image (224×224 RGB)
    dummy_image = Image.new("RGB", (224, 224), color=(128, 128, 128))

    # 3. Encode single image
    print("\n  Encoding single image ...")
    output = encoder.encode_image(dummy_image)
    assert output is not None
    print(f"  Output shape: {output.shape}")
    assert output.shape == (1, 256, 4096), f"Expected (1, 256, 4096), got {output.shape}"

    # 4. Encode None → should return None
    none_output = encoder.encode_image(None)
    assert none_output is None, "Expected None for None input"
    print("  None input → None output ✓")

    # 5. Encode batch (mixed with None)
    print("\n  Encoding batch [image, None, image] ...")
    batch_results = encoder.encode_batch([dummy_image, None, dummy_image])
    assert batch_results[0] is not None and batch_results[0].shape == (1, 256, 4096)
    assert batch_results[1] is None
    assert batch_results[2] is not None and batch_results[2].shape == (1, 256, 4096)
    print(f"  Batch results: [{batch_results[0].shape}, None, {batch_results[2].shape}]")

    # 6. Check trainable params (only projection should train)
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in encoder.parameters() if not p.requires_grad)
    print(f"\n  Trainable: {trainable:,}")
    print(f"  Frozen:    {frozen:,}")
    assert trainable == 1024 * 4096 + 4096, f"Unexpected trainable count: {trainable}"
    assert frozen > trainable, "CLIP backbone should be frozen"

    # 7. Save and reload projection
    print("\n  Testing save/load projection ...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        save_path = encoder.save_projection(Path(tmp))
        old_weight = encoder.projection.weight.clone()
        # Perturb weights
        encoder.projection.weight.data.fill_(0.0)
        assert not torch.equal(encoder.projection.weight, old_weight)
        # Reload
        encoder.load_projection(Path(tmp))
        assert torch.equal(encoder.projection.weight, old_weight), "Weights not restored!"
    print("  Save/load round-trip ✓")

    print("\n  ✓ All vision encoder tests passed!")
    print("=" * 60)
