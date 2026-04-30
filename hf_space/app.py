"""
MINDI 1.5 Vision-Coder — HuggingFace Space (ZeroGPU)

Uses ZeroGPU for free A100 access (40GB VRAM).
Full bf16 model — NO quantization.

IMPORTANT: All .to("cuda") calls MUST be inside @spaces.GPU decorated functions.
ZeroGPU only provides GPU access inside those functions.
"""

import os
import re
import gc
import json
import torch
import spaces
import gradio as gr
from pathlib import Path
from PIL import Image
from huggingface_hub import snapshot_download

# ── Global model reference ──────────────────────────────
MODEL = None
TOKENIZER = None
IS_LOADED = False

# ── Special token definitions ───────────────────────────
SECTION_TOKENS = {
    "thinking":  ("<|think_start|>",    "<|think_end|>"),
    "file":      ("<|file_start|>",     "<|file_end|>"),
    "code":      ("<|code_start|>",     "<|code_end|>"),
    "critique":  ("<|critique_start|>", "<|critique_end|>"),
    "suggest":   ("<|suggest_start|>",  "<|suggest_end|>"),
    "search":    ("<|search_start|>",   "<|search_end|>"),
    "error":     ("<|error_start|>",    "<|error_end|>"),
    "fix":       ("<|fix_start|>",      "<|fix_end|>"),
}


def parse_output(text: str) -> dict:
    result = {}
    for section, (start_tok, end_tok) in SECTION_TOKENS.items():
        pattern = re.escape(start_tok) + r"(.*?)" + re.escape(end_tok)
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            result[section] = [m.strip() for m in matches]
    return result


def clean_output(text: str) -> str:
    text = text.replace("<|im_end|>", "").replace("<|im_start|>", "")
    text = re.sub(r"^(system|user|assistant)\n?", "", text).strip()
    return text


def download_checkpoint():
    """Download checkpoint (CPU-safe, no CUDA needed)."""
    ckpt_dir = Path("/tmp/mindi_ckpt")
    if not ckpt_dir.exists():
        print("[MINDI] Downloading checkpoint from HuggingFace...")
        snapshot_download(
            "Mindigenous/MINDI-1.5-Vision-Coder",
            local_dir=str(ckpt_dir),
            allow_patterns=[
                "checkpoints/phase3_final/**",
                "data/tokenizer/**",
            ],
        )
        print("[MINDI] Download complete")
    return ckpt_dir


def load_tokenizer(ckpt_dir):
    """Load tokenizer (CPU-safe)."""
    global TOKENIZER
    if TOKENIZER is not None:
        return

    from transformers import AutoTokenizer
    tok_path = ckpt_dir / "data" / "tokenizer" / "mindi_tokenizer"
    if not tok_path.exists():
        tok_path = "Qwen/Qwen2.5-Coder-7B-Instruct"
    TOKENIZER = AutoTokenizer.from_pretrained(str(tok_path), trust_remote_code=True)
    print(f"[MINDI] Tokenizer loaded: {len(TOKENIZER)} tokens")


def load_model_to_gpu(ckpt_dir):
    """Load model TO GPU — MUST be called inside @spaces.GPU function."""
    global MODEL, IS_LOADED

    if IS_LOADED:
        return

    from transformers import (
        AutoModelForCausalLM,
        CLIPVisionModel, CLIPImageProcessor,
    )
    from peft import PeftModel
    import torch.nn as nn

    print("[MINDI] Loading full bf16 model to GPU...")

    # Base LLM
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.resize_token_embeddings(len(TOKENIZER))
    print("[MINDI] Base model loaded (bf16)")

    # LoRA
    lora_path = ckpt_dir / "checkpoints" / "phase3_final" / "lora"
    if lora_path.exists():
        base_model = PeftModel.from_pretrained(base_model, str(lora_path))
        print("[MINDI] LoRA loaded")

    # CLIP
    clip_model = CLIPVisionModel.from_pretrained(
        "openai/clip-vit-large-patch14",
        torch_dtype=torch.bfloat16,
    ).to("cuda").eval()
    clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    print("[MINDI] CLIP loaded")

    # Vision projection
    class VisionProjection(nn.Module):
        def __init__(self, clip_dim=1024, llm_dim=3584):
            super().__init__()
            self.projection = nn.Linear(clip_dim, llm_dim)
            self.layer_norm = nn.LayerNorm(llm_dim)
        def forward(self, x):
            return self.layer_norm(self.projection(x))

    vision_proj = VisionProjection().to("cuda").to(torch.bfloat16)
    vision_ckpt = ckpt_dir / "checkpoints" / "phase3_final" / "vision"
    if vision_ckpt.exists():
        for f in vision_ckpt.iterdir():
            if f.suffix in (".pt", ".bin", ".safetensors"):
                state = torch.load(f, map_location="cuda", weights_only=True)
                vision_proj.load_state_dict(state, strict=False)
                print("[MINDI] Vision projection loaded")
                break

    # Fusion
    class SimpleFusion(nn.Module):
        def __init__(self, hidden_size=3584):
            super().__init__()
            self.visual_gate = nn.Linear(hidden_size, hidden_size)
            self.text_gate_param = nn.Parameter(torch.zeros(1))
            self.layer_norm = nn.LayerNorm(hidden_size)
        def forward(self, text_embeds, visual_embeds):
            gated_visual = torch.sigmoid(self.visual_gate(visual_embeds)) * visual_embeds
            combined = torch.cat([gated_visual, text_embeds], dim=1)
            return self.layer_norm(combined)

    fusion = SimpleFusion().to("cuda").to(torch.bfloat16)
    fusion_file = ckpt_dir / "checkpoints" / "phase3_final" / "fusion" / "fusion.pt"
    if fusion_file.exists():
        state = torch.load(fusion_file, map_location="cuda", weights_only=True)
        fusion.load_state_dict(state, strict=False)
        print("[MINDI] Fusion loaded")

    MODEL = {
        "llm": base_model,
        "clip": clip_model,
        "clip_processor": clip_processor,
        "vision_proj": vision_proj,
        "fusion": fusion,
    }
    IS_LOADED = True

    vram = torch.cuda.memory_allocated() / 1e9
    print(f"[MINDI] Ready! VRAM: {vram:.1f} GB")


# Pre-download checkpoint and tokenizer at startup (CPU-safe)
_ckpt_dir = download_checkpoint()
load_tokenizer(_ckpt_dir)


@spaces.GPU(duration=60)
def generate(prompt: str, image: Image.Image = None,
             temperature: float = 0.7, max_tokens: int = 2048) -> str:
    """Generate with full bf16 model. GPU allocated by ZeroGPU."""

    # Load model INSIDE the GPU function
    load_model_to_gpu(_ckpt_dir)

    system_msg = (
        "You are MINDI 1.5 Vision-Coder, an expert AI coding assistant. "
        "When asked for code, respond with complete, working implementations. "
        "Be concise and provide actual code, not descriptions."
    )
    formatted = (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = TOKENIZER(formatted, return_tensors="pt").to("cuda")

    # Vision path
    if image is not None and MODEL["clip"] is not None:
        try:
            pixel_values = MODEL["clip_processor"](
                images=image, return_tensors="pt"
            ).pixel_values.to("cuda", dtype=torch.bfloat16)

            with torch.no_grad():
                clip_out = MODEL["clip"](pixel_values).last_hidden_state
                visual_tokens = MODEL["vision_proj"](clip_out)
                text_embeds = MODEL["llm"].get_input_embeddings()(inputs["input_ids"])
                fused = MODEL["fusion"](text_embeds, visual_tokens)

                outputs = MODEL["llm"].generate(
                    inputs_embeds=fused,
                    attention_mask=torch.ones(fused.shape[:2], device="cuda"),
                    max_new_tokens=int(max_tokens),
                    temperature=max(float(temperature), 0.01),
                    do_sample=float(temperature) > 0,
                    pad_token_id=TOKENIZER.pad_token_id or TOKENIZER.eos_token_id,
                )
            return clean_output(TOKENIZER.decode(outputs[0], skip_special_tokens=False))
        except Exception as e:
            print(f"[WARN] Vision failed: {e}")

    # Text-only path
    with torch.no_grad():
        outputs = MODEL["llm"].generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            temperature=max(float(temperature), 0.01),
            do_sample=float(temperature) > 0,
            pad_token_id=TOKENIZER.pad_token_id or TOKENIZER.eos_token_id,
        )
    generated = outputs[:, inputs["input_ids"].shape[1]:]
    return clean_output(TOKENIZER.decode(generated[0], skip_special_tokens=False))


# ── Gradio endpoint ─────────────────────────────────────

def chat_fn(message: str, image: Image.Image = None,
            temperature: float = 0.7, max_tokens: int = 2048) -> str:
    """Exposed via Gradio API — wraps generate."""
    try:
        response = generate(message, image, temperature, max_tokens)
        sections = parse_output(response)
        return json.dumps({"response": response, "sections": sections})
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower() or "gpu" in error_msg.lower():
            return json.dumps({"response": f"⚠️ GPU quota exceeded. Please try again later or reduce Max Tokens. Error: {error_msg}", "sections": {"error": [error_msg]}})
        return json.dumps({"response": f"Error during generation: {error_msg}", "sections": {"error": [error_msg]}})


# ── Gradio App ──────────────────────────────────────────

with gr.Blocks(title="MINDI 1.5 API", theme=gr.themes.Soft(
    primary_hue="purple", secondary_hue="blue",
)) as demo:
    gr.Markdown("# 🧠 MINDI 1.5 Vision-Coder API\nFull bf16 on ZeroGPU A100 · No quantization")

    with gr.Row():
        with gr.Column(scale=3):
            prompt = gr.Textbox(label="Prompt", placeholder="Write code...", lines=4)
            image = gr.Image(label="Image (optional)", type="pil")
        with gr.Column(scale=1):
            temperature = gr.Slider(0, 2, value=0.7, step=0.1, label="Temperature")
            max_tokens = gr.Slider(128, 4096, value=2048, step=128, label="Max Tokens")
            submit_btn = gr.Button("Generate", variant="primary")

    output = gr.Textbox(label="Response (JSON)", lines=20)

    submit_btn.click(
        fn=chat_fn,
        inputs=[prompt, image, temperature, max_tokens],
        outputs=output,
    )

demo.queue()
demo.launch()
