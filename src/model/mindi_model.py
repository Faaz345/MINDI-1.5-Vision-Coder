"""
MINDI 1.5 Vision-Coder — Complete Model

Combines MINDIArchitecture (Qwen2.5-Coder + LoRA), VisionEncoder (CLIP ViT-L/14),
and VisionLanguageFusion into a single MINDI15 class with forward(), generate(),
parse_output(), save(), and load() methods.

Uses the MINDI custom tokenizer (data/tokenizer/mindi_tokenizer/) with 22 special
tokens for agentic code generation capabilities.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from src.model.architecture import MINDIArchitecture
from src.model.fusion_layer import VisionLanguageFusion
from src.model.vision_encoder import VisionEncoder

# ── MINDI special token pairs ────────────────────────────────────────
MINDI_SECTION_TOKENS: dict[str, tuple[str, str]] = {
    "thinking":  ("<|think_start|>",    "<|think_end|>"),
    "file":      ("<|file_start|>",     "<|file_end|>"),
    "code":      ("<|code_start|>",     "<|code_end|>"),
    "critique":  ("<|critique_start|>", "<|critique_end|>"),
    "suggest":   ("<|suggest_start|>",  "<|suggest_end|>"),
    "search":    ("<|search_start|>",   "<|search_end|>"),
    "error":     ("<|error_start|>",    "<|error_end|>"),
    "fix":       ("<|fix_start|>",      "<|fix_end|>"),
}

# Project root (resolved relative to this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TOKENIZER_PATH = PROJECT_ROOT / "data" / "tokenizer" / "mindi_tokenizer"


class MINDI15(nn.Module):
    """
    MINDI 1.5 Vision-Coder — complete multimodal coding model.

    Components:
        - architecture: Qwen2.5-Coder-7B-Instruct + LoRA
        - vision_encoder: CLIP ViT-L/14 (frozen) → 256 tokens × 4096
        - fusion: Linear + LayerNorm prepend fusion
        - tokenizer: MINDI custom tokenizer with 22 special tokens
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        clip_model: str = "openai/clip-vit-large-patch14",
        hidden_size: int = 4096,
        num_visual_tokens: int = 256,
        tokenizer_path: Optional[Path] = None,
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[Path] = None,
    ) -> None:
        """
        Initialize MINDI 1.5 with all components.

        Args:
            model_name: HuggingFace base LLM identifier.
            clip_model: HuggingFace CLIP vision model identifier.
            hidden_size: LLM hidden dimension (must match Qwen config).
            num_visual_tokens: Number of visual tokens from CLIP (256).
            tokenizer_path: Path to MINDI custom tokenizer directory.
            device: Target device ('cuda', 'cpu', or None for auto).
            torch_dtype: Data type for model weights.
            cache_dir: Base directory for model weight caches.
        """
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size
        self.num_visual_tokens = num_visual_tokens
        self.torch_dtype = torch_dtype

        cache_base = Path(cache_dir) if cache_dir else PROJECT_ROOT / "checkpoints"

        print("=" * 60)
        print("  MINDI 1.5 Vision-Coder — Initializing")
        print("=" * 60)

        # 1. Load MINDI custom tokenizer (NOT the base Qwen tokenizer)
        tok_path = Path(tokenizer_path) if tokenizer_path else DEFAULT_TOKENIZER_PATH
        print(f"\n[MINDI15] Loading MINDI tokenizer from {tok_path} ...")
        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            str(tok_path),
            trust_remote_code=True,
        )
        print(f"  Vocab size: {len(self.tokenizer)}")

        # 2. LLM backbone with LoRA
        self.architecture = MINDIArchitecture(
            model_name=model_name,
            device=self.device,
            cache_dir=cache_base / "base",
            torch_dtype=torch_dtype,
        )

        # Resize embeddings to match MINDI tokenizer (includes 22 special tokens)
        self.architecture.resize_embeddings(len(self.tokenizer))

        # Apply LoRA
        self.architecture.apply_lora()

        # 3. Vision encoder (frozen CLIP + trainable projection)
        self.vision_encoder = VisionEncoder(
            model_name=clip_model,
            llm_hidden_size=hidden_size,
            device=self.device,
            cache_dir=cache_base / "vision",
        )

        # 4. Fusion layer
        self.fusion = VisionLanguageFusion(
            hidden_size=hidden_size,
            num_visual_tokens=num_visual_tokens,
        )
        self.fusion.to(self.device)

        # Cache special token IDs
        self._special_ids: dict[str, int] = {}
        for section, (start_tok, end_tok) in MINDI_SECTION_TOKENS.items():
            sid = self.tokenizer.convert_tokens_to_ids(start_tok)
            eid = self.tokenizer.convert_tokens_to_ids(end_tok)
            self._special_ids[f"{section}_start"] = sid
            self._special_ids[f"{section}_end"] = eid

        self._print_summary()

    def _print_summary(self) -> None:
        """Print initialization summary."""
        llm_info = self.architecture.get_trainable_params()
        vis_info = {
            "trainable": sum(p.numel() for p in self.vision_encoder.parameters() if p.requires_grad),
            "total": sum(p.numel() for p in self.vision_encoder.parameters()),
        }
        fus_info = self.fusion.get_trainable_params()

        total_trainable = llm_info["trainable"] + vis_info["trainable"] + fus_info["trainable"]
        total_all = llm_info["total"] + vis_info["total"] + fus_info["total"]

        print()
        print("=" * 60)
        print("  MINDI 1.5 — Initialization Complete")
        print("=" * 60)
        print(f"  LLM trainable (LoRA):   {llm_info['trainable']:>14,}")
        print(f"  Vision trainable:       {vis_info['trainable']:>14,}")
        print(f"  Fusion trainable:       {fus_info['trainable']:>14,}")
        print(f"  ─────────────────────────────────────")
        print(f"  Total trainable:        {total_trainable:>14,}")
        print(f"  Total params:           {total_all:>14,}")
        print(f"  Tokenizer vocab:        {len(self.tokenizer):>14,}")
        print("=" * 60)
        print()

    # ── Forward pass ──────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        image: Optional[Image.Image] = None,
    ) -> dict:
        """
        Forward pass with optional vision input.

        Args:
            input_ids: Token IDs (batch, seq_len).
            attention_mask: Attention mask (batch, seq_len).
            labels: Target token IDs for loss computation (batch, seq_len).
            image: Optional PIL image for multimodal input.

        Returns:
            Dict with 'loss', 'logits', and optionally 'visual_tokens'.
        """
        model = self.architecture.get_model()

        # Get text embeddings from the LLM's embedding layer
        text_embeds = model.get_input_embeddings()(input_ids)

        # Encode vision if image provided
        visual_tokens = None
        if image is not None:
            visual_tokens = self.vision_encoder.encode_image(image)

        # Fuse vision + text
        fused_embeds, fused_mask = self.fusion(text_embeds, visual_tokens, attention_mask)

        # Extend labels if vision tokens were prepended
        if visual_tokens is not None and labels is not None:
            batch_size = labels.shape[0]
            # -100 = ignore index for cross-entropy on visual positions
            visual_labels = torch.full(
                (batch_size, self.num_visual_tokens),
                fill_value=-100,
                dtype=labels.dtype,
                device=labels.device,
            )
            labels = torch.cat([visual_labels, labels], dim=1)

        # Forward through LLM with embeddings (bypass tokenization)
        outputs = model(
            inputs_embeds=fused_embeds,
            attention_mask=fused_mask,
            labels=labels,
        )

        result = {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }
        if visual_tokens is not None:
            result["visual_tokens"] = visual_tokens

        return result

    # ── Generation ────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        repetition_penalty: float = 1.1,
    ) -> str:
        """
        Generate text from a prompt, optionally conditioned on an image.

        Uses the MINDI custom tokenizer (with special tokens) for both
        encoding the prompt and decoding the output.

        Args:
            prompt: Input text prompt.
            image: Optional PIL image for multimodal generation.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling threshold.
            do_sample: Whether to sample (False = greedy).
            repetition_penalty: Penalty for repeated tokens.

        Returns:
            Generated text string (decoded with MINDI tokenizer).
        """
        model = self.architecture.get_model()
        model.eval()

        # Tokenize with MINDI tokenizer
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # If image provided, build fused embeddings
        if image is not None:
            text_embeds = model.get_input_embeddings()(input_ids)
            visual_tokens = self.vision_encoder.encode_image(image)
            fused_embeds, fused_mask = self.fusion(text_embeds, visual_tokens, attention_mask)

            output_ids = model.generate(
                inputs_embeds=fused_embeds,
                attention_mask=fused_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        else:
            # Text-only generation (direct input_ids)
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        generated_ids = output_ids[:, input_ids.shape[1]:]
        text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        return text.strip()

    # ── Output parsing ────────────────────────────────────────────

    @staticmethod
    def parse_output(text: str) -> dict[str, list[str]]:
        """
        Parse generated text and extract ALL MINDI special-token sections.

        Extracts content between each pair of special tokens:
            <|think_start|> ... <|think_end|>     → "thinking"
            <|file_start|> ... <|file_end|>       → "file"
            <|code_start|> ... <|code_end|>       → "code"
            <|critique_start|> ... <|critique_end|> → "critique"
            <|suggest_start|> ... <|suggest_end|>   → "suggest"
            <|search_start|> ... <|search_end|>     → "search"
            <|error_start|> ... <|error_end|>       → "error"
            <|fix_start|> ... <|fix_end|>           → "fix"

        Each section may appear multiple times; all occurrences are captured.

        Args:
            text: Raw generated text potentially containing special tokens.

        Returns:
            Dict mapping section name → list of extracted content strings.
            Empty list if section not found. Also includes "raw" with full text.
        """
        result: dict[str, list[str]] = {"raw": [text]}

        for section, (start_tok, end_tok) in MINDI_SECTION_TOKENS.items():
            # Escape the pipe characters for regex
            pattern = re.escape(start_tok) + r"(.*?)" + re.escape(end_tok)
            matches = re.findall(pattern, text, flags=re.DOTALL)
            result[section] = [m.strip() for m in matches]

        return result

    # ── Phase control (for 3-phase training) ──────────────────────

    def set_trainable_components(
        self,
        lora: bool = False,
        vision_projection: bool = False,
        fusion: bool = False,
    ) -> dict[str, int]:
        """
        Enable/disable training for specific components.

        Used by the trainer to implement 3-phase training:
            Phase 1: lora=True,  vision_projection=False, fusion=False
            Phase 2: lora=False, vision_projection=True,  fusion=True
            Phase 3: lora=True,  vision_projection=True,  fusion=True

        Args:
            lora: Whether LoRA adapter parameters should be trainable.
            vision_projection: Whether the vision projection layer should train.
            fusion: Whether the fusion layer should be trainable.

        Returns:
            Dict with trainable param counts per component.
        """
        counts = {}

        # LoRA parameters
        peft_model = self.architecture.peft_model
        if peft_model is not None:
            for name, param in peft_model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = lora
        counts["lora"] = sum(
            p.numel() for n, p in (peft_model or self.architecture.model).named_parameters()
            if "lora_" in n and p.requires_grad
        )

        # Vision projection
        for param in self.vision_encoder.projection.parameters():
            param.requires_grad = vision_projection
        counts["vision_projection"] = sum(
            p.numel() for p in self.vision_encoder.projection.parameters() if p.requires_grad
        )

        # Fusion layer
        for param in self.fusion.parameters():
            param.requires_grad = fusion
        counts["fusion"] = sum(
            p.numel() for p in self.fusion.parameters() if p.requires_grad
        )

        counts["total_trainable"] = counts["lora"] + counts["vision_projection"] + counts["fusion"]

        print(f"[MINDI15] Trainable: LoRA={counts['lora']:,} | "
              f"VisionProj={counts['vision_projection']:,} | "
              f"Fusion={counts['fusion']:,} | "
              f"Total={counts['total_trainable']:,}")

        return counts

    # ── Save / Load ───────────────────────────────────────────────

    def save(self, save_dir: Optional[Path] = None) -> Path:
        """
        Save all trainable weights (LoRA + vision projection + fusion).

        Args:
            save_dir: Root directory for saving. Defaults to checkpoints/mindi15.

        Returns:
            Path to save directory.
        """
        save_path = Path(save_dir) if save_dir else PROJECT_ROOT / "checkpoints" / "mindi15"
        save_path.mkdir(parents=True, exist_ok=True)

        # LoRA adapter
        self.architecture.save_lora(save_path / "lora")

        # Vision projection
        self.vision_encoder.save_projection(save_path / "vision")

        # Fusion layer
        fusion_path = save_path / "fusion"
        fusion_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.fusion.state_dict(), fusion_path / "fusion.pt")

        print(f"[MINDI15] All weights saved to {save_path}")
        return save_path

    def load(self, load_dir: Path) -> None:
        """
        Load all trainable weights (LoRA + vision projection + fusion).

        Args:
            load_dir: Root directory containing saved weights.
        """
        load_path = Path(load_dir)
        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")

        # LoRA adapter
        lora_path = load_path / "lora"
        if lora_path.exists():
            self.architecture.load_lora(lora_path)

        # Vision projection
        vision_path = load_path / "vision"
        if vision_path.exists():
            self.vision_encoder.load_projection(vision_path)

        # Fusion layer
        fusion_file = load_path / "fusion" / "fusion.pt"
        if fusion_file.exists():
            state_dict = torch.load(fusion_file, map_location=self.device, weights_only=True)
            self.fusion.load_state_dict(state_dict)
            print(f"[MINDI15] Fusion loaded from {fusion_file.parent}")

        print(f"[MINDI15] All weights loaded from {load_path}")

    # ── Utilities ─────────────────────────────────────────────────

    def get_all_trainable_params(self) -> dict:
        """Get combined trainable parameter counts across all components."""
        llm = self.architecture.get_trainable_params()
        vis_trainable = sum(
            p.numel() for p in self.vision_encoder.parameters() if p.requires_grad
        )
        fus = self.fusion.get_trainable_params()

        total_trainable = llm["trainable"] + vis_trainable + fus["trainable"]
        total_all = llm["total"] + sum(p.numel() for p in self.vision_encoder.parameters()) + fus["total"]

        return {
            "llm_trainable": llm["trainable"],
            "llm_total": llm["total"],
            "vision_trainable": vis_trainable,
            "fusion_trainable": fus["trainable"],
            "total_trainable": total_trainable,
            "total_params": total_all,
            "trainable_pct": round(100.0 * total_trainable / total_all, 4) if total_all > 0 else 0.0,
        }

    def print_info(self) -> None:
        """Print complete model information."""
        self.architecture.print_model_info()
        info = self.get_all_trainable_params()
        print("  MINDI 1.5 Combined Trainable Parameters:")
        print(f"    LLM (LoRA):       {info['llm_trainable']:>14,}")
        print(f"    Vision proj:      {info['vision_trainable']:>14,}")
        print(f"    Fusion:           {info['fusion_trainable']:>14,}")
        print(f"    Total trainable:  {info['total_trainable']:>14,}")
        print(f"    Total params:     {info['total_params']:>14,}")
        print(f"    Trainable %:      {info['trainable_pct']:>13.2f}%")
        print()


# ── Test block ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  MINDI 1.5 — Complete Model Test")
    print("=" * 60)
    print()

    # ── Test 1: parse_output (no GPU needed) ─────────────────────
    print("  Test 1: parse_output()")
    sample_output = (
        "<|think_start|>The user wants a Python function.<|think_end|>"
        "<|file_start|>main.py<|file_end|>"
        "<|code_start|>def hello():\n    print('Hello MINDI!')<|code_end|>"
        "<|critique_start|>Missing type hints and docstring.<|critique_end|>"
        "<|suggest_start|>Add return type annotation.<|suggest_end|>"
        "<|search_start|>python type hints best practices<|search_end|>"
        "<|error_start|>NameError: name 'x' is not defined<|error_end|>"
        "<|fix_start|>Add x = 0 before the loop.<|fix_end|>"
        "<|think_start|>Let me also add error handling.<|think_end|>"
    )

    parsed = MINDI15.parse_output(sample_output)

    assert len(parsed["thinking"]) == 2, f"Expected 2 thinking sections, got {len(parsed['thinking'])}"
    assert parsed["thinking"][0] == "The user wants a Python function."
    assert parsed["thinking"][1] == "Let me also add error handling."
    assert parsed["file"] == ["main.py"]
    assert parsed["code"] == ["def hello():\n    print('Hello MINDI!')"]
    assert parsed["critique"] == ["Missing type hints and docstring."]
    assert parsed["suggest"] == ["Add return type annotation."]
    assert parsed["search"] == ["python type hints best practices"]
    assert parsed["error"] == ["NameError: name 'x' is not defined"]
    assert parsed["fix"] == ["Add x = 0 before the loop."]
    assert "raw" in parsed
    print("    All 8 section types extracted correctly ✓")
    print(f"    Sections found: {[k for k, v in parsed.items() if k != 'raw' and v]}")

    # ── Test 2: parse_output with missing sections ───────────────
    print("\n  Test 2: parse_output() with partial output")
    partial = "<|code_start|>print('hi')<|code_end|>"
    parsed2 = MINDI15.parse_output(partial)
    assert parsed2["code"] == ["print('hi')"]
    assert parsed2["thinking"] == []
    assert parsed2["file"] == []
    assert parsed2["fix"] == []
    print("    Missing sections return empty lists ✓")

    # ── Test 3: parse_output with empty input ────────────────────
    print("\n  Test 3: parse_output() with empty string")
    parsed3 = MINDI15.parse_output("")
    assert all(v == [] for k, v in parsed3.items() if k != "raw")
    print("    Empty input returns all empty lists ✓")

    # ── Test 4: Verify MINDI_SECTION_TOKENS covers all 8 ────────
    print("\n  Test 4: Token coverage")
    expected_sections = {"thinking", "file", "code", "critique", "suggest", "search", "error", "fix"}
    assert set(MINDI_SECTION_TOKENS.keys()) == expected_sections
    print(f"    All 8 sections defined: {sorted(expected_sections)} ✓")

    # ── GPU-dependent tests (skip if no CUDA) ────────────────────
    if torch.cuda.is_available():
        print("\n  Test 5: Full model initialization (GPU)")
        model = MINDI15()
        model.print_info()

        # Test set_trainable_components (Phase 1)
        print("\n  Test 6: Phase 1 — LoRA only")
        counts = model.set_trainable_components(lora=True, vision_projection=False, fusion=False)
        assert counts["lora"] > 0
        assert counts["vision_projection"] == 0
        assert counts["fusion"] == 0

        # Test set_trainable_components (Phase 2)
        print("\n  Test 7: Phase 2 — Vision bridge only")
        counts = model.set_trainable_components(lora=False, vision_projection=True, fusion=True)
        assert counts["lora"] == 0
        assert counts["vision_projection"] > 0
        assert counts["fusion"] > 0

        # Test set_trainable_components (Phase 3)
        print("\n  Test 8: Phase 3 — All trainable")
        counts = model.set_trainable_components(lora=True, vision_projection=True, fusion=True)
        assert counts["lora"] > 0
        assert counts["vision_projection"] > 0
        assert counts["fusion"] > 0

        # Test forward (text only)
        print("\n  Test 9: Forward pass (text only)")
        tokens = model.tokenizer("Hello MINDI!", return_tensors="pt")
        input_ids = tokens["input_ids"].to(model.device)
        attn_mask = tokens["attention_mask"].to(model.device)
        result = model(input_ids=input_ids, attention_mask=attn_mask, labels=input_ids)
        assert result["loss"] is not None
        print(f"    Loss: {result['loss'].item():.4f}")
        print(f"    Logits: {result['logits'].shape}")

        # Test forward (with image)
        print("\n  Test 10: Forward pass (with dummy image)")
        dummy_img = Image.new("RGB", (224, 224), color=(100, 150, 200))
        result_v = model(input_ids=input_ids, attention_mask=attn_mask, labels=input_ids, image=dummy_img)
        assert result_v["loss"] is not None
        assert "visual_tokens" in result_v
        print(f"    Loss: {result_v['loss'].item():.4f}")
        print(f"    Visual tokens: {result_v['visual_tokens'].shape}")

        # Test generate (text only)
        print("\n  Test 11: Generate (text only, short)")
        output = model.generate("Write a hello world in Python:", max_new_tokens=50)
        print(f"    Output: {output[:100]}...")

        print("\n  Test 12: Save/load round-trip")
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            model.save(Path(tmp))
            # Verify files exist
            assert (Path(tmp) / "lora").exists()
            assert (Path(tmp) / "vision" / "projection.pt").exists()
            assert (Path(tmp) / "fusion" / "fusion.pt").exists()
            print("    Save ✓")
    else:
        print("\n  [SKIP] GPU tests (no CUDA available)")
        print("  Tests 5-12 require GPU with ~20GB VRAM")

    print("\n  ✓ All MINDI 1.5 model tests passed!")
    print("=" * 60)
