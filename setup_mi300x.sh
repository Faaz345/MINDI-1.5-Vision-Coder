#!/bin/bash
# ============================================================
# MINDI 1.5 Vision-Coder — MI300X Setup Script
# One command to set up everything on DigitalOcean AMD MI300X
# ============================================================
set -e

echo "============================================================"
echo "  MINDI 1.5 Vision-Coder — MI300X Setup"
echo "  MINDIGENOUS.AI"
echo "============================================================"
echo ""

# ── Check HF_TOKEN ─────────────────────────────────────────────
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: Set HF_TOKEN environment variable first!"
    echo "  export HF_TOKEN=hf_your_token_here"
    exit 1
fi

# ── Step 1: Install ROCm PyTorch ───────────────────────────────
echo "[1/7] Installing ROCm PyTorch (ROCm 6.0) ..."
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.0

# ── Step 2: Get the full project from HF ──────────────────────
echo ""
echo "[2/7] Getting MINDI 1.5 from HuggingFace ..."
if [ -f "requirements.txt" ]; then
    echo "  Already in repo — pulling latest ..."
    git pull
else
    git clone https://Mindigenous:${HF_TOKEN}@huggingface.co/Mindigenous/MINDI-1.5-Vision-Coder
    cd MINDI-1.5-Vision-Coder
fi

# ── Step 3: Install Python requirements ────────────────────────
echo ""
echo "[3/7] Installing Python requirements ..."
pip install -r requirements.txt

# Additional training dependencies
pip install wandb huggingface_hub accelerate

# ── Step 4: Download training data from HF ─────────────────────
echo ""
echo "[4/7] Downloading training dataset ..."
python -c "
from huggingface_hub import snapshot_download
import os

snapshot_download(
    repo_id='Mindigenous/MINDI-1.5-training-data',
    repo_type='dataset',
    local_dir='data/',
    token=os.environ['HF_TOKEN']
)
print('Dataset downloaded!')
"

# Verify data files exist
echo "  Checking data files ..."
if [ ! -f "data/processed/train.jsonl" ]; then
    echo "  ERROR: train.jsonl not found!"
    exit 1
fi
if [ ! -f "data/processed/val.jsonl" ]; then
    echo "  ERROR: val.jsonl not found!"
    exit 1
fi
TRAIN_SIZE=$(du -sh data/processed/train.jsonl | cut -f1)
VAL_SIZE=$(du -sh data/processed/val.jsonl | cut -f1)
echo "  train.jsonl: ${TRAIN_SIZE}"
echo "  val.jsonl:   ${VAL_SIZE}"

# ── Step 5: Set environment variables ──────────────────────────
echo ""
echo "[5/7] Setting environment variables ..."

# ROCm / PyTorch settings
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_ROCM_ARCH="gfx942"
export HIP_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="mindi-1.5-vision-coder"

# Create .env file
cat > .env << EOF
HF_TOKEN=${HF_TOKEN}
HSA_OVERRIDE_GFX_VERSION=11.0.0
PYTORCH_ROCM_ARCH=gfx942
HIP_VISIBLE_DEVICES=0
TOKENIZERS_PARALLELISM=false
WANDB_PROJECT=mindi-1.5-vision-coder
EOF
echo "  .env file created"

# ── Step 6: Verify GPU detected ───────────────────────────────
echo ""
echo "[6/7] Verifying GPU ..."
python -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU name:        {torch.cuda.get_device_name(0)}')
    vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    print(f'  VRAM:            {vram:.1f} GB')
    print(f'  ROCm backend:    {torch.version.hip is not None}')
else:
    print('  WARNING: No GPU detected!')
    exit(1)
"

# Quick bf16 test
python -c "
import torch
x = torch.randn(100, 100, dtype=torch.bfloat16, device='cuda')
y = torch.matmul(x, x.T)
print(f'  bf16 matmul test: PASSED (shape={y.shape})')
"

# ── Step 7: Create output directories ─────────────────────────
echo ""
echo "[7/7] Creating output directories ..."
mkdir -p checkpoints/training
mkdir -p checkpoints/best
mkdir -p logs/training

# ── Done ───────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  MINDI 1.5 Vision-Coder — MI300X Ready!"
echo ""
echo "  Project:  $(pwd)"
echo "  Data:     ${TRAIN_SIZE} train / ${VAL_SIZE} val"
echo "  GPU:      $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'N/A')"
echo ""
echo "  Ready to train!"
echo "  Run:  python scripts/train.py --phase 1"
echo ""
echo "  Or dry run first:"
echo "  Run:  python scripts/train.py --dry_run --no_wandb"
echo "============================================================"
