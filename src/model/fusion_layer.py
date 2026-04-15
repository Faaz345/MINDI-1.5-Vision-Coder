"""
MINDI 1.5 Vision-Coder — Vision-Language Fusion Layer

Prepends projected visual tokens (256 × 4096) to text token embeddings
and extends the attention mask accordingly.  Uses Linear + LayerNorm
for the visual projection gate.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class VisionLanguageFusion(nn.Module):
    """
    Fuses visual and text embeddings by prepending visual tokens.

    Pipeline:
        1. visual_tokens (batch, 256, 4096) → Linear → LayerNorm
        2. Prepend to text_embeds (batch, seq_len, 4096)
        3. Extend attention_mask to cover the extra 256 visual positions

    All trainable parameters live in the gate projection + LayerNorm.
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        num_visual_tokens: int = 256,
    ) -> None:
        """
        Initialize the fusion layer.

        Args:
            hidden_size: Dimension of both visual and text embeddings (must match).
            num_visual_tokens: Number of visual tokens prepended (default 256).
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_visual_tokens = num_visual_tokens

        # Gate projection: Linear + LayerNorm to align visual features
        self.gate_proj = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        text_embeds: torch.Tensor,
        visual_tokens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Fuse visual tokens into text embeddings.

        Args:
            text_embeds: Text token embeddings (batch, seq_len, hidden_size).
            visual_tokens: Projected visual tokens (batch, 256, hidden_size), or None
                           for text-only inputs.
            attention_mask: Text attention mask (batch, seq_len), or None.

        Returns:
            fused_embeds: (batch, 256 + seq_len, hidden_size) if visual, else unchanged.
            fused_mask: Extended attention mask, or None if input mask was None.
        """
        # Text-only path — no vision tokens to fuse
        if visual_tokens is None:
            return text_embeds, attention_mask

        batch_size = text_embeds.shape[0]
        v_batch = visual_tokens.shape[0]

        # Handle batch size mismatch (single image broadcast to batch)
        if v_batch == 1 and batch_size > 1:
            visual_tokens = visual_tokens.expand(batch_size, -1, -1)

        # Gate projection + LayerNorm
        gated_visual = self.gate_proj(visual_tokens)   # (batch, 256, hidden_size)
        gated_visual = self.layer_norm(gated_visual)    # (batch, 256, hidden_size)

        # Prepend visual tokens to text embeddings
        fused_embeds = torch.cat([gated_visual, text_embeds], dim=1)

        # Extend attention mask
        fused_mask = self._extend_attention_mask(attention_mask, batch_size, text_embeds.device)

        return fused_embeds, fused_mask

    def _extend_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        Extend attention mask to include visual token positions (all attended).

        Args:
            attention_mask: Original text mask (batch, seq_len) or None.
            batch_size: Current batch size.
            device: Target device.

        Returns:
            Extended mask (batch, 256 + seq_len) or None.
        """
        if attention_mask is None:
            return None

        # Visual tokens are always fully attended
        visual_mask = torch.ones(
            batch_size,
            self.num_visual_tokens,
            dtype=attention_mask.dtype,
            device=device,
        )
        return torch.cat([visual_mask, attention_mask], dim=1)

    def get_trainable_params(self) -> dict:
        """
        Count trainable parameters in the fusion layer.

        Returns:
            Dictionary with 'trainable', 'total', and 'trainable_pct'.
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        pct = 100.0 * trainable / total if total > 0 else 0.0
        return {
            "trainable": trainable,
            "total": total,
            "trainable_pct": round(pct, 4),
        }

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_visual_tokens={self.num_visual_tokens}"
        )


# ── Test block ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  MINDI 1.5 — Fusion Layer Test")
    print("=" * 60)
    print()

    BATCH = 2
    SEQ_LEN = 128
    HIDDEN = 4096
    N_VIS = 256

    fusion = VisionLanguageFusion(hidden_size=HIDDEN, num_visual_tokens=N_VIS)
    print(f"  Fusion layer:\n  {fusion}\n")

    # ── Test 1: Vision + Text fusion ─────────────────────────────
    print("  Test 1: Vision + Text fusion")
    text_embeds = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    visual_tokens = torch.randn(BATCH, N_VIS, HIDDEN)
    attention_mask = torch.ones(BATCH, SEQ_LEN, dtype=torch.long)

    fused_embeds, fused_mask = fusion(text_embeds, visual_tokens, attention_mask)

    expected_seq = N_VIS + SEQ_LEN  # 256 + 128 = 384
    assert fused_embeds.shape == (BATCH, expected_seq, HIDDEN), \
        f"Expected ({BATCH}, {expected_seq}, {HIDDEN}), got {fused_embeds.shape}"
    assert fused_mask is not None and fused_mask.shape == (BATCH, expected_seq), \
        f"Expected mask ({BATCH}, {expected_seq}), got {fused_mask.shape}"
    print(f"    fused_embeds: {fused_embeds.shape} ✓")
    print(f"    fused_mask:   {fused_mask.shape} ✓")

    # ── Test 2: Text-only (no vision) ────────────────────────────
    print("\n  Test 2: Text-only (no vision)")
    text_only, mask_only = fusion(text_embeds, None, attention_mask)
    assert text_only.shape == (BATCH, SEQ_LEN, HIDDEN)
    assert mask_only is not None and mask_only.shape == (BATCH, SEQ_LEN)
    print(f"    text_only:  {text_only.shape} ✓")
    print(f"    mask_only:  {mask_only.shape} ✓")

    # ── Test 3: No attention mask ────────────────────────────────
    print("\n  Test 3: Vision fusion without attention mask")
    fused_no_mask, none_mask = fusion(text_embeds, visual_tokens, None)
    assert fused_no_mask.shape == (BATCH, expected_seq, HIDDEN)
    assert none_mask is None
    print(f"    fused_embeds: {fused_no_mask.shape} ✓")
    print(f"    fused_mask:   None ✓")

    # ── Test 4: Single-image broadcast ───────────────────────────
    print("\n  Test 4: Single-image broadcast to batch")
    single_visual = torch.randn(1, N_VIS, HIDDEN)
    fused_bc, mask_bc = fusion(text_embeds, single_visual, attention_mask)
    assert fused_bc.shape == (BATCH, expected_seq, HIDDEN)
    print(f"    fused_embeds: {fused_bc.shape} ✓ (broadcast 1 → {BATCH})")

    # ── Test 5: Trainable params ─────────────────────────────────
    print("\n  Test 5: Parameter counts")
    info = fusion.get_trainable_params()
    # gate_proj: 4096*4096 + 4096 = 16,781,312
    # layer_norm: 4096 + 4096 = 8,192
    expected_params = HIDDEN * HIDDEN + HIDDEN + HIDDEN + HIDDEN  # Linear(w+b) + LN(w+b)
    assert info["trainable"] == expected_params, \
        f"Expected {expected_params}, got {info['trainable']}"
    print(f"    Trainable: {info['trainable']:,}")
    print(f"    Total:     {info['total']:,}")
    print(f"    Pct:       {info['trainable_pct']}%")

    # ── Test 6: Gradient flow ────────────────────────────────────
    print("\n  Test 6: Gradient flow through fusion")
    fusion.zero_grad()
    fused_embeds, _ = fusion(text_embeds, visual_tokens, attention_mask)
    loss = fused_embeds.sum()
    loss.backward()
    assert fusion.gate_proj.weight.grad is not None, "No gradient on gate_proj!"
    assert fusion.layer_norm.weight.grad is not None, "No gradient on layer_norm!"
    print("    gate_proj gradient:  ✓")
    print("    layer_norm gradient: ✓")

    print("\n  ✓ All fusion layer tests passed!")
    print("=" * 60)
