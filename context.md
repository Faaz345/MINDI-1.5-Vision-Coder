# MINDI 1.5 Vision-Coder — Complete Project Context

> **Last updated:** April 16, 2026
> **Purpose:** This file contains ALL context needed to continue development with any AI assistant. 
> It covers architecture decisions, errors encountered, fixes applied, training state, and exact next steps.

---

## 1. PROJECT OVERVIEW

**MINDI 1.5 Vision-Coder** is a multimodal AI model that generates frontend code (HTML/CSS/JS, Next.js, Tailwind) from UI screenshots and text prompts. It combines:

- **Qwen/Qwen2.5-Coder-7B-Instruct** — 7.62B param base LLM (Apache 2.0)
- **CLIP ViT-L/14** — Frozen vision encoder for UI screenshot understanding
- **LoRA adapters** — Efficient fine-tuning (r=64, alpha=128)
- **Vision-Language Fusion** — Prepend visual tokens to text embeddings
- **22 MINDI Special Tokens** — Structured agentic reasoning (think, code, critique, fix, etc.)
- **3-Phase Training Strategy** — Progressive training on MI300X 192GB

**Repos:**
- **GitHub:** `https://github.com/Faaz345/MINDI-1.5-Vision-Coder.git` (branch: `master`)
- **HuggingFace Model:** `Mindigenous/MINDI-1.5-Vision-Coder` (private, push as `master:main`)
- **HuggingFace Dataset:** `Mindigenous/MINDI-1.5-training-data` (private)
- **HF Token:** Set as `HF_TOKEN` environment variable (stored separately, not in repo)

---

## 2. DIRECTORY STRUCTURE

```
MINDI-1.5-Vision-Coder/
├── src/
│   ├── model/
│   │   ├── architecture.py       # Qwen2.5-Coder + LoRA wrapper (NOT nn.Module)
│   │   ├── mindi_model.py        # MINDI15 main class (nn.Module)
│   │   ├── vision_encoder.py     # CLIP ViT-L/14 (frozen) + trainable projection
│   │   ├── fusion_layer.py       # VisionLanguageFusion with text_gate
│   │   └── __init__.py
│   ├── training/
│   │   ├── mindi_trainer.py      # MINDITrainer: 3-phase loop, streaming data
│   │   ├── data_pipeline.py      # Data processing pipeline
│   │   └── __init__.py
│   ├── agents/                   # Agentic pipeline (orchestrator, error fixer, UI critic)
│   ├── inference/                # Generation pipeline
│   ├── evaluation/               # Evaluation framework
│   ├── search/                   # Tavily search agent
│   ├── sandbox/                  # E2B/Docker code execution
│   ├── tokenizer/                # MINDI tokenizer wrapper
│   └── utils/                    # Config & env loaders
├── scripts/
│   ├── train.py                  # Master training launcher (--dry_run, --phase, --resume)
│   ├── download_websight.py      # Download WebSight v0.2 from HF
│   ├── upload_websight_images.py # Upload images to HF in batches (10K/dir limit)
│   ├── gpu_diagnostic.py         # 6-stage GPU test for MI300X
│   └── ... (data processing scripts)
├── configs/
│   ├── training_config.yaml      # Training hyperparameters
│   ├── model_config.yaml         # Model architecture config
│   ├── data_config.yaml          # Data sources and processing
│   └── search_config.yaml        # Tavily search settings
├── data/
│   ├── processed/                # Text training data (train.jsonl, val.jsonl, test.jsonl)
│   ├── websight/                 # Vision data (52,500 images in subdirs + JSONL)
│   │   ├── train.jsonl           # 50,000 vision-code pairs
│   │   ├── val.jsonl             # 2,500 vision-code pairs
│   │   └── images/
│   │       ├── 00/               # ws_0000000.jpg - ws_0009999.jpg (10K each)
│   │       ├── 01/
│   │       ├── 02/
│   │       ├── 03/
│   │       ├── 04/
│   │       └── 05/               # ws_0050000.jpg - ws_0052499.jpg (2,500)
│   ├── tokenizer/
│   │   ├── mindi_tokenizer/      # Custom tokenizer (vocab 151,685)
│   │   └── base_tokenizer/       # Original Qwen tokenizer
│   └── raw/                      # Raw downloaded data sources
├── api/                          # FastAPI endpoints
├── checkpoints/                  # Model checkpoints
├── logs/                         # Training logs
├── requirements.txt              # Full requirements
├── requirements-training.txt     # Lean MI300X Docker requirements
├── setup_mi300x.sh               # MI300X Docker setup script
├── .gitattributes                # LFS tracking for large tokenizer files
└── .gitignore
```

---

## 3. ARCHITECTURE DETAILS

### 3.1 Model Components

| Component | Class | File | Params | Trainable |
|-----------|-------|------|--------|-----------|
| Base LLM | `MINDIArchitecture` | `architecture.py` | 7.62B | No (frozen) |
| LoRA | via PEFT | `architecture.py` | 161.5M | Yes |
| CLIP Vision | `VisionEncoder` | `vision_encoder.py` | 304M | 4.2M (projection only) |
| Fusion | `VisionLanguageFusion` | `fusion_layer.py` | 16.8M | Yes |
| **Total** | `MINDI15` | `mindi_model.py` | **8.1B** | **182.5M (2.25%)** |

### 3.2 CRITICAL Architecture Notes

1. **`MINDIArchitecture` is NOT an `nn.Module`** — it's a plain Python wrapper class. The actual trainable PeftModel is accessed via `self.architecture.get_model()` and registered as `self.llm` in `MINDI15.__init__()`.

2. **`self.llm = self.architecture.get_model()`** — This line in `mindi_model.py` registers the PeftModel as a proper submodule so `model.parameters()` can find LoRA params. Without this, the optimizer gets zero trainable parameters.

3. **Vision encoder uses `float32` projection** — CLIP backbone is frozen, only `self.projection` (Linear 1024→4096) trains. The projection operates in float32 for stability even though the rest is bf16.

4. **Fusion layer has `text_gate`** — A learnable scalar parameter (init=0) that creates a residual path for text-only inputs. This ensures gradients flow to the fusion layer during Phase 2 even when processing text-only batches (which have no vision tokens and would otherwise be pure passthrough with no gradient).

### 3.3 Forward Pass Flow

```
Image → CLIP (frozen) → 256 patches (1024) → projection (4096) → visual_tokens
Text → tokenizer → input_ids → LLM embedding layer → text_embeds

With image:   fusion = [gated_visual_tokens; text_embeds]  (prepend)
Without image: fusion = text_embeds + sigmoid(text_gate) * (transformed - text_embeds)

fusion → LLM layers (with LoRA) → logits → loss (cross-entropy, labels=-100 for padding)
```

### 3.4 LoRA Configuration

```python
LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
```

### 3.5 MINDI Special Tokens (22 total, 11 pairs)

```
<|think_start|> / <|think_end|>         — Internal reasoning
<|code_start|> / <|code_end|>           — Generated code blocks
<|file_start|> / <|file_end|>           — File references
<|critique_start|> / <|critique_end|>   — Self-critique
<|suggest_start|> / <|suggest_end|>     — Suggestions
<|search_start|> / <|search_end|>       — Search context
<|error_start|> / <|error_end|>         — Error messages
<|fix_start|> / <|fix_end|>             — Fix attempts
<|vision_start|> / <|vision_end|>       — Vision input markers
<|sandbox_start|> / <|sandbox_end|>     — Sandbox execution
<|context_start|> / <|context_end|>     — Context block
```

---

## 4. TRAINING PIPELINE

### 4.1 Three-Phase Training Strategy

| Phase | Name | Steps | LR | Batch | Components | Data | Purpose |
|-------|------|-------|-----|-------|-----------|------|---------|
| 1 | `phase1_lora` | 5,000 | 2e-4 | 16 | LoRA only | Text-only code | Teach coding patterns |
| 2 | `phase2_vision_bridge` | 2,500 | 1e-5 | 8 | Vision+Fusion | WebSight images | Align visual tokens |
| 3 | `phase3_all` | 2,500 | 5e-5 | 12 | All trainable | Mixed text+vision | Joint fine-tuning |

**Total: 10,000 steps**

### 4.2 Training Data

**Text data (Phase 1 + Phase 3):**
- `data/processed/train.jsonl` — 1,304,486 examples, 4.18 GB
- `data/processed/val.jsonl` — 72,471 examples
- Sources: CodeAlpaca, CodeFeedback, EvolCode, MagicCoder, StarCoder (5 langs), Synthetic Next.js

**Vision data (Phase 2 + Phase 3):**
- `data/websight/train.jsonl` — 50,000 image+code pairs, 114 MB JSONL
- `data/websight/val.jsonl` — 2,500 image+code pairs, 5.7 MB JSONL
- `data/websight/images/` — 52,500 JPG screenshots in 6 subdirectories (11.6 GB)
- Source: HuggingFaceM4/WebSight v0.2 (UI screenshot → HTML/CSS pairs)

**WebSight JSONL format:**
```json
{
  "id": "websight_0000001",
  "type": "vision_code",
  "source": "websight_v0.2",
  "image_path": "data/websight/images/00/ws_0000001.jpg",
  "messages": [
    {"role": "system", "content": "You are MINDI 1.5 Vision-Coder..."},
    {"role": "user", "content": "<|vision_start|><|vision_end|>\nGenerate the HTML/CSS code for this UI screenshot."},
    {"role": "assistant", "content": "<|think_start|>...<|think_end|>\n<|code_start|>\n...HTML/CSS...\n<|code_end|>"}
  ],
  "metadata": {"dataset": "websight", "version": "v0.2"}
}
```

**IMPORTANT:** Images are organized in subdirectories of ≤10,000 files each because HuggingFace has a 10K files/directory limit. The JSONL `image_path` fields reference the subdirectory structure (e.g., `data/websight/images/00/ws_0000001.jpg`).

### 4.3 Data Loading

- **`StreamingJSONLDataset`** (in `mindi_trainer.py`) — Streams from disk line-by-line, tokenizes on-the-fly
- **Shuffle buffer** of 10,000 examples (reservoir-style)
- **Image loading** via `_load_image()` — loads PIL images from relative paths
- **Custom collate function** — stacks tensors, keeps images as a list
- **Phase routing** — Phase 1 uses text data, Phase 2 uses WebSight, Phase 3 uses text (with inline images if present)

### 4.4 Key Training Features

- **bf16 precision** — Required for MI300X stability (NOT fp16)
- **Gradient checkpointing** — Enabled even with 192GB VRAM
- **torch.compile()** — Optional, works on ROCm
- **Cosine LR with warmup** — Per-phase schedules
- **Gradient accumulation** — Configurable per phase (default: 4)
- **Emergency checkpoint** — Saved on Ctrl+C
- **Crash checkpoint** — Saved on unhandled exceptions

---

## 5. TRAINING HISTORY & RESULTS

### 5.1 Phase 1 Dry Run — SUCCESS ✅

**Date:** April 15, 2026 (on DigitalOcean MI300X)
**Command:** `python3 scripts/train.py --dry_run --no_wandb`
**Result:** Loss dropped from 1.94 → 0.85 in 10 steps, completed in 12.1 minutes
**VRAM usage:** ~14.3 GB

### 5.2 Phase 2 — First Attempt FAILED ❌

**Error:** `element 0 of tensors does not require grad and does not have a grad_fn`
**Root cause:** Phase 2 trains vision+fusion with LoRA frozen. Text-only data means fusion is pure passthrough (no gradient path). The fusion layer was getting zero gradients because without vision tokens, the text-only path was `return text_embeds, attention_mask` — a pure passthrough with no learnable operation.
**Fix:** Added `text_gate` learnable residual parameter to `VisionLanguageFusion`. Text-only path changed to: `text_embeds + sigmoid(text_gate) * (transformed - text_embeds)`. Also built the WebSight vision data pipeline to provide actual image+code pairs for Phase 2.

### 5.3 Full 3-Phase Dry Run — NOT YET COMPLETED

The MI300X GPU kept hanging/wedging (see Section 6). Phase 2 and 3 with the new WebSight data pipeline have NOT been tested yet.

---

## 6. ERRORS & FIXES — COMPLETE HISTORY

### 6.1 GPU Hang #1 — HSA_OVERRIDE_GFX_VERSION

**Symptom:** GPU completely unresponsive. `torch.cuda.get_device_name(0)` returns blank, any CUDA operation hangs.
**Root cause:** `HSA_OVERRIDE_GFX_VERSION=11.0.0` was set in the Docker container. This conflicts with ROCm 7.0's native MI300X/gfx942 support.
**Fix:** Do NOT set `HSA_OVERRIDE_GFX_VERSION`. ROCm 7.0 natively supports gfx942. Remove it from all scripts/env.
**Commit:** `4a33f96 Remove HSA_OVERRIDE_GFX_VERSION`

### 6.2 No Trainable Parameters in Optimizer

**Symptom:** `RuntimeError: No trainable parameters in phase 'phase1_lora'`
**Root cause:** `MINDIArchitecture` is a plain Python class (not `nn.Module`). When `MINDI15` calls `model.parameters()`, it doesn't find the LoRA parameters because the PeftModel isn't registered as a submodule.
**Fix:** Added `self.llm = self.architecture.get_model()` in `MINDI15.__init__()` to register the PeftModel as a proper nn.Module submodule. Updated `forward()` and `generate()` to use `self.llm` instead of `self.architecture.get_model()`.
**Commit:** `cdc806e Fix: register LLM as nn.Module submodule so optimizer finds LoRA params`

### 6.3 extra_special_tokens Format Error

**Symptom:** `TypeError` when loading tokenizer — transformers 4.55 expects `extra_special_tokens` as a dict, not a list.
**Fix:** Changed `data/tokenizer/mindi_tokenizer/tokenizer_config.json`: converted `extra_special_tokens` from list format to `{"token_name": {"content": "..."}}` dict format.
**Commit:** `02eef51 Fix extra_special_tokens: list to dict for transformers 4.55`

### 6.4 Phase 2 Gradient Flow Crash

**Symptom:** `element 0 of tensors does not require grad and does not have a grad_fn` during Phase 2
**Root cause:** Text-only data → no vision tokens → fusion is pure passthrough → no gradient path to fusion parameters.
**Fix:** (1) Added `text_gate` learnable residual gate in `VisionLanguageFusion` for text-only gradient flow. (2) Built WebSight vision data pipeline with actual image+code pairs.
**Commit:** `4e9835e Fix Phase 2: fusion layer processes text-only via learnable residual gate`

### 6.5 Git LFS Issues

**Symptom:** `tokenizer.json` files >10MB causing push failures to HuggingFace.
**Fix:** Configured `.gitattributes` for LFS tracking. Ran `git lfs migrate import` to rewrite history. Force-pushed to both GitHub and HF.
**Commit:** `161c946 Track large tokenizer files with Git LFS`

### 6.6 HuggingFace Auth for MI300X Clone

**Symptom:** `git clone` from HF failed with auth error in Docker container.
**Fix:** Use token as both username and password: `https://hf_TOKEN:hf_TOKEN@huggingface.co/Mindigenous/MINDI-1.5-Vision-Coder.git`
Also needed: `apt-get install -y git-lfs && git lfs install`

### 6.7 GPU Hang #2 — Driver Wedge After Heavy I/O

**Symptom:** After interrupted HF upload + training attempt, GPU shows 100% utilization with 0% VRAM in `rocm-smi`. Even `torch.randn(device='cuda')` hangs. Docker restart insufficient.
**Kernel log:** `amdgpu: GPU reset begin!` → `device wedged, but recovered through reset` → But GPU% stays at 100%.
**Fix:** 
1. `docker stop rocm`
2. `echo 1 > /sys/bus/pci/devices/0000:83:00.0/reset` (PCI address from `lspci | grep AMD`)
3. If GPU% still 100%: `modprobe -r amdgpu && modprobe amdgpu`
4. Verify `rocm-smi` shows GPU% = 0% before restarting Docker
**Status:** Droplet was deleted. Will need to handle this on fresh droplet if it recurs.

### 6.8 HuggingFace Upload Limits

**Symptom:** `413 Payload Too Large` (25K files/commit) and `400 Bad Request` (10K files/directory)
**Fix:** Reorganized 52,500 images into 6 subdirectories of ≤10K files (`00/` through `05/`). Upload in separate commits per subdirectory. Updated JSONL `image_path` fields to include subdirectory.
**Script:** `scripts/upload_websight_images.py`

---

## 7. MI300X DEPLOYMENT

### 7.1 Infrastructure

- **Provider:** DigitalOcean GPU Droplet
- **GPU:** AMD Instinct MI300X (192GB HBM3 VRAM)
- **Cost:** $1.99/hr
- **Docker container:** Named `rocm`, accessed via `docker exec -it rocm /bin/bash`
- **ROCm/HIP:** 7.0.51831-a3e329ad8
- **PyTorch:** 2.9.0.dev20250821+rocm7.0.0
- **Python:** 3.10

### 7.2 Critical Environment Variables

```bash
export HF_TOKEN=<your-hf-token>     # Get from HF settings page
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_ROCM_ARCH=gfx942
# DO NOT SET: HSA_OVERRIDE_GFX_VERSION (causes GPU hang on ROCm 7.0)
```

### 7.3 Fresh Droplet Setup Procedure

```bash
# 1. SSH into droplet
ssh root@<DROPLET_IP>

# 2. Start Docker
docker start rocm
docker exec -it rocm /bin/bash

# 3. Set environment (inside Docker)
export HF_TOKEN=<your-hf-token>     # Get from HF settings page
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_ROCM_ARCH=gfx942

# 4. Quick GPU test
python3 -c "import torch; print('GPU:', torch.cuda.get_device_name(0)); x=torch.randn(100,device='cuda'); print('OK:', x.sum().item())"

# 5. Install git-lfs
apt-get update && apt-get install -y git-lfs
git lfs install

# 6. Clone code repo
cd /workspace
git clone https://$HF_TOKEN:$HF_TOKEN@huggingface.co/Mindigenous/MINDI-1.5-Vision-Coder.git
cd MINDI-1.5-Vision-Coder

# 7. Install requirements
pip install -r requirements-training.txt

# 8. Download training data from HF dataset repo
python3 -c "
from huggingface_hub import snapshot_download
import os
# HF_TOKEN must be set in environment
snapshot_download(
    repo_id='Mindigenous/MINDI-1.5-training-data',
    repo_type='dataset',
    local_dir='data',
    token=os.environ['HF_TOKEN'],
)
print('Data download complete!')
"

# 9. Verify data
ls -la data/processed/
ls -la data/websight/
ls data/websight/images/ | head

# 10. Run GPU diagnostic
python3 scripts/gpu_diagnostic.py

# 11. Dry run
python3 scripts/train.py --dry_run --no_wandb

# 12. Full training
python3 scripts/train.py --no_wandb
```

### 7.4 GPU Hang Recovery (if it happens again)

```bash
# From HOST (not inside Docker):
docker stop rocm
echo 1 > /sys/bus/pci/devices/0000:83:00.0/reset   # PCI address may differ
rocm-smi  # Verify GPU% = 0%
# If still 100%:
modprobe -r amdgpu && modprobe amdgpu
rocm-smi  # Should show 0% now
docker start rocm
```

---

## 8. HF DATASET REPO STRUCTURE

**Repo:** `Mindigenous/MINDI-1.5-training-data` (private, type: dataset)

```
├── .gitattributes
├── README.md
├── processed/
│   ├── train.jsonl              # 1.3M text examples
│   ├── val.jsonl
│   ├── test.jsonl
│   ├── filter_report.json
│   ├── mindi_filtered.jsonl
│   └── split_meta.json
├── raw/                         # Original data sources (11 files)
├── tokenizer/
│   ├── base_tokenizer/
│   └── mindi_tokenizer/
└── websight/
    ├── train.jsonl              # 50K vision-code JSONL
    ├── val.jsonl                # 2.5K vision-code JSONL
    └── images/
        ├── 00/                  # 10,000 JPGs
        ├── 01/                  # 10,000 JPGs
        ├── 02/                  # 10,000 JPGs
        ├── 03/                  # 10,000 JPGs
        ├── 04/                  # 10,000 JPGs (uploading as of April 16)
        └── 05/                  # 2,500 JPGs  (uploading as of April 16)
```

**NOTE:** As of April 16, 2026, subdirectories 00-03 are uploaded. 04 and 05 are being uploaded via `scripts/upload_websight_images.py`. If upload was interrupted, re-run the script — it skips already-uploaded subdirs.

---

## 9. GIT HISTORY (CHRONOLOGICAL)

```
553fbf7 feat: initial project scaffold for MINDI 1.5 Vision-Coder
11e0d89 Day 1 Complete: Tokenizer setup — 22 MINDI special tokens (vocab 151,685)
59c6c97 Day 2 COMPLETE: 1.48M examples processed, 6GB dataset, WebSight done
2ff5c54 Day 3 COMPLETE: Full model architecture (7 files)
1c36b28 Fix train.py: mem -> memory on line 225
f04f58b Fix setup_mi300x.sh step 2 + add project context summary
35fd5fc Fix setup_mi300x.sh for Docker container on MI300X droplet
5fb9ec3 Add GPU diagnostic script, fix architecture loading with sync
161c946 Track large tokenizer files with Git LFS
4a33f96 Remove HSA_OVERRIDE_GFX_VERSION - ROCm 7.0 native MI300X support
24b5fb1 Add requirements-training.txt for MI300X Docker
02eef51 Fix extra_special_tokens: list to dict for transformers 4.55
cdc806e Fix: register LLM as nn.Module submodule so optimizer finds LoRA params
4e9835e Fix Phase 2: fusion layer text_gate for gradient flow
672896a Add WebSight vision data pipeline: download, image-aware loader, phase routing
```

---

## 10. WHAT WORKS (VERIFIED) ✅

1. **Tokenizer** — 151,685 vocab with 22 MINDI special tokens, loads correctly
2. **Model initialization** — MINDI15 loads all 4 components, 182.5M trainable params
3. **GPU diagnostic** — All 6 tests pass (bf16 matmul, 1GB alloc, CPU→CUDA transfer, forward pass)
4. **Phase 1 dry run** — Loss 1.94 → 0.85 in 10 steps ✅
5. **WebSight download** — 52,500 images (11.6 GB) downloaded and organized
6. **Data format** — JSONL with image_path references, streaming dataset works
7. **Git LFS** — Large tokenizer files tracked correctly
8. **Code pushed** — All code on GitHub master + HF model repo main

---

## 11. WHAT REMAINS (TODO) ❌

1. **Complete WebSight upload to HF** — Subdirs 04 and 05 still uploading (re-run `scripts/upload_websight_images.py` if interrupted)
2. **Full 3-phase dry run** — Phase 2 (WebSight) and Phase 3 (mixed) NOT yet tested with the vision pipeline
3. **Full production training** — 10,000 steps total (Phase 1: 5K, Phase 2: 2.5K, Phase 3: 2.5K)
4. **Inference testing** — Generate code from screenshots after training
5. **Commit `upload_websight_images.py` and `context.md`** — These new files need to be pushed

---

## 12. KNOWN ISSUES & GOTCHAS

### DO NOT:
- Set `HSA_OVERRIDE_GFX_VERSION=11.0.0` — kills GPU on ROCm 7.0
- Use `fp16` on MI300X — use `bf16` for stability
- Try to upload >10K files to a single HF directory — split into subdirs
- Try to commit >25K files in a single HF commit — batch commits
- Use the global Python (base env) on Windows — use venv (global torch DLL is broken)

### WATCH OUT FOR:
- GPU hanging after heavy I/O — check `rocm-smi` shows 0% GPU before training
- Data paths — WebSight images use **relative paths** from project root in JSONL
- `MINDIArchitecture` is NOT `nn.Module` — always use `self.llm` inside MINDI15
- The `text_gate` in fusion starts at 0 (sigmoid=0.5) — this is intentional
- On MI300X, Docker container named `rocm` — always `docker exec -it rocm /bin/bash`

---

## 13. COMMANDS REFERENCE

### Local (Windows, PowerShell, in venv):
```powershell
# Activate venv
& ".\venv\Scripts\Activate.ps1"

# Download WebSight
$env:HF_TOKEN="<your-hf-token>"
python scripts/download_websight.py --num_train 50000 --num_val 2500

# Upload WebSight images to HF (handles subdirs, retry, skip)
python scripts/upload_websight_images.py

# Push code to GitHub + HF
git push origin master
git push hf master:main
```

### MI300X (Linux, Docker, inside container):
```bash
# Dry run (10 steps per phase)
python3 scripts/train.py --dry_run --no_wandb

# Full training
python3 scripts/train.py --no_wandb

# Single phase
python3 scripts/train.py --phase 1 --no_wandb
python3 scripts/train.py --phase 2 --no_wandb
python3 scripts/train.py --phase 3 --no_wandb

# Resume from checkpoint
python3 scripts/train.py --resume checkpoints/training/phase1_lora_step5000 --no_wandb

# GPU diagnostic
python3 scripts/gpu_diagnostic.py
```

---

## 14. NEXT SESSION CHECKLIST

When continuing with a new AI assistant:

1. **Open this directory** in your IDE
2. **Read this file first** to get full context
3. **Check WebSight upload status:**
   ```powershell
   python -c "import os; from huggingface_hub import HfApi; api=HfApi(token=os.environ['HF_TOKEN']); files=[f for f in api.list_repo_files('Mindigenous/MINDI-1.5-training-data', repo_type='dataset') if 'websight/images' in f]; print(f'{len(files)} images in HF repo')"
   ```
4. If <52,500: re-run `python scripts/upload_websight_images.py`
5. **Push any uncommitted files:**
   ```bash
   git add scripts/upload_websight_images.py context.md
   git commit -m "Add WebSight batch uploader and project context"
   git push origin master
   git push hf master:main
   ```
6. **Spin up fresh MI300X droplet** on DigitalOcean
7. **Follow Section 7.3** for setup procedure
8. **Run dry run first** to verify all 3 phases work
9. **Then full training** — `python3 scripts/train.py --no_wandb`

---

## 15. DATA FILE LOCATIONS ON HF DATASET REPO

When cloning data on MI300X using `snapshot_download`, files will land at:

| HF Repo Path | Local Path (relative to project root) |
|---|---|
| `processed/train.jsonl` | `data/processed/train.jsonl` |
| `processed/val.jsonl` | `data/processed/val.jsonl` |
| `websight/train.jsonl` | `data/websight/train.jsonl` |
| `websight/val.jsonl` | `data/websight/val.jsonl` |
| `websight/images/00/*.jpg` | `data/websight/images/00/*.jpg` |
| `tokenizer/mindi_tokenizer/*` | `data/tokenizer/mindi_tokenizer/*` |

The `snapshot_download(local_dir='data')` call places everything correctly because the HF repo structure mirrors the local `data/` directory.

---

*This context file was created on April 16, 2026 during Claude Opus 4.6 session to ensure project continuity.*
