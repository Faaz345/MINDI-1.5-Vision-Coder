# MINDI 1.5 Model Checkpoint Status Report

**Date:** 2026-05-09
**Investigated by:** Cascade (AI assistant)
**Status:** 🔴 CRITICAL — Model weights are MISSING

---

## Summary

Your codex was **correct**. The MINDI 1.5 model checkpoint files are **not present** on HuggingFace. The repo contains only source code, configuration files, and tokenizers — no trained model weights.

## What IS on HuggingFace (confirmed via API)

### Model Repo: `Mindigenous/MINDI-1.5-Vision-Coder`
- **Total files:** 107
- **Contents:** Source code (`src/`, `api/`, `scripts/`), configs (`configs/`), tokenizer (`data/tokenizer/`), frontend (`frontend/`), HF Space (`hf_space/`), documentation
- **Checkpoint files:** **ZERO**
- **No `.safetensors`, `.pt`, `.bin`, or any model weights**

### Dataset Repo: `Mindigenous/MINDI-1.5-training-data`
- **Total files:** 52,532
- **Contents:** Training data (`raw/`, `processed/`), tokenizer copies
- **Checkpoint files:** **ZERO**

## Where the checkpoints SHOULD be

The HF Space (`hf_space/app.py`) expects to download:
```
checkpoints/phase3_final/
  ├── lora/          (LoRA adapter weights)
  ├── vision/        (Vision projection weights)
  └── fusion/        (Fusion layer weights)
data/tokenizer/
  └── mindi_tokenizer/
```

But these paths do **not exist** in the repo.

## Where checkpoints MIGHT still exist

| Location | Status | Likelihood |
|----------|--------|------------|
| **HuggingFace model repo** | ❌ Missing | Confirmed empty |
| **This Windows machine** | ❌ Missing | Confirmed empty (searched for `.safetensors`, `.pt`, `.bin`) |
| **AMD GPU Droplet** (`165.245.141.245`) | ❓ **UNKNOWN** | **Best hope** — you showed a screenshot of this droplet |
| **Modal volume (`mindi-data`)** | ❓ Likely wiped | Upload script exists but checkpoints never made it to HF |

## How the loss likely happened

1. Training completed on AMD MI300X droplet (or Modal)
2. Checkpoints were saved locally on the training machine
3. `modal_upload_now.py` was designed to upload from Modal's volume to HF
4. Either:
   - The upload **never ran successfully**
   - The upload **failed silently** (no error handling in the script)
   - The Modal volume was **deleted** after the run
   - The checkpoints were **overwritten** by a subsequent operation

## Recovery Options

### Option 1: Check the AMD Droplet (RECOMMENDED FIRST STEP)

SSH into your droplet and run the search script:

```bash
# On your local machine, copy and run this script on the droplet:
ssh root@165.245.141.245 'bash -s' < scripts/check_droplet_for_ckpts.sh
```

Or manually:
```bash
ssh root@165.245.141.245
find /mnt /workspace /root -name "*.safetensors" -o -name "*.pt" -o -name "*.bin" 2>/dev/null
du -sh /mnt/mindi /workspace 2>/dev/null
```

**If found:** Use `scripts/restore_checkpoints_from_droplet.py` to upload them to HF.

### Option 2: Retrain from scratch

If droplet has no checkpoints, you must retrain:

1. **Clone dataset repo** (data is safe on HF):
   ```bash
   git clone https://huggingface.co/datasets/Mindigenous/MINDI-1.5-training-data
   ```

2. **Use your MI300X droplet** (shown in screenshot, $1.99/hr):
   - Already has vLLM 0.17.1 on Ubuntu 24.04
   - 192GB VRAM (sufficient for Qwen2.5-Coder-7B + LoRA)

3. **Run training pipeline**:
   ```bash
   # On the droplet
   git clone https://huggingface.co/Mindigenous/MINDI-1.5-Vision-Coder
   cd MINDI-1.5-Vision-Coder
   bash setup_mi300x.sh
   python scripts/train.py --config configs/training_config.yaml
   ```

4. **Upload checkpoints IMMEDIATELY after training**:
   ```bash
   python scripts/upload_everything_to_hf.py
   # Also specifically upload checkpoints:
   huggingface-cli upload Mindigenous/MINDI-1.5-Vision-Coder checkpoints/ checkpoints/
   ```

### Option 3: Use base model (temporary fallback)

If you need the chat UI working NOW without retraining:

Modify `hf_space/app.py` to load the **base Qwen2.5-Coder-7B** instead of your fine-tuned checkpoint:

```python
# In hf_space/app.py, change download_checkpoint() to:
def download_checkpoint():
    from transformers import AutoModelForCausalLM
    # Load base model directly from HuggingFace
    return AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
```

**Trade-off:** Loses your LoRA fine-tune, vision encoder fusion, and agentic behavior. But the chat UI will work.

## Prevention (for next time)

1. **Upload checkpoints IMMEDIATELY after training finishes**
2. **Verify uploads** — always run a download test after uploading
3. **Keep multiple backups**:
   - HuggingFace (primary)
   - Local external drive
   - Cloud storage (S3, GCS, etc.)
4. **Use `hf_space/app.py`'s expected paths** (`checkpoints/phase3_final/`) when uploading, or update the HF Space to match your actual checkpoint names

## Files created for recovery

| File | Purpose |
|------|---------|
| `scripts/check_droplet_for_ckpts.sh` | SSH script to search your AMD droplet for checkpoints |
| `scripts/restore_checkpoints_from_droplet.py` | Downloads found checkpoints from droplet and uploads to HF |

## Next Action Required

**YOU must run the droplet check.** I cannot SSH into your droplet without your credentials.

Run this NOW:
```bash
ssh root@165.245.141.245
find / -name "*.safetensors" 2>/dev/null | head -20
```

If you find checkpoint files, run the recovery script. If not, you need to retrain.
