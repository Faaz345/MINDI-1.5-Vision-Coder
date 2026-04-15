#!/bin/bash
# ============================================================
# MINDI 1.5 Vision-Coder — MI300X Setup Script
# Run INSIDE the Docker container on DigitalOcean AMD MI300X
#
# On the host first:
#   docker exec -it rocm /bin/bash
#   export HF_TOKEN=hf_your_token_here
#   bash setup_mi300x.sh        (if already cloned)
#   OR wget + bash               (if fresh)
# ============================================================
set -e

echo "============================================================"
echo "  MINDI 1.5 Vision-Coder — MI300X Setup"
echo "  MINDIGENOUS.AI"
echo "  (Docker container environment)"
echo "============================================================"
echo ""

# ── Check HF_TOKEN ─────────────────────────────────────────────
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: Set HF_TOKEN environment variable first!"
    echo "  export HF_TOKEN=hf_your_token_here"
    exit 1
fi

# ── Step 1: Verify PyTorch + ROCm (already in Docker image) ───
echo "[1/7] Verifying PyTorch + ROCm (pre-installed in Docker) ..."
python3 -c "
import torch
v = torch.__version__
hip = torch.version.hip or 'None'
print(f'  PyTorch: {v}')
print(f'  ROCm/HIP: {hip}')
assert torch.cuda.is_available(), 'No GPU detected!'
print(f'  GPU: {torch.cuda.get_device_name(0)}')
vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
print(f'  VRAM: {vram:.0f} GB')
print('  [OK] PyTorch + ROCm verified')
"

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
python3 -c "
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

# ROCm / PyTorch settings for MI300X
# NOTE: Do NOT set HSA_OVERRIDE_GFX_VERSION — ROCm 7.0 has native gfx942 support
export PYTORCH_ROCM_ARCH="gfx942"
export HIP_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="mindi-1.5-vision-coder"

# Create .env file for the project
cat > .env << EOF
HF_TOKEN=${HF_TOKEN}
PYTORCH_ROCM_ARCH=gfx942
HIP_VISIBLE_DEVICES=0
TOKENIZERS_PARALLELISM=false
WANDB_PROJECT=mindi-1.5-vision-coder
EOF

# Also add to bashrc so env persists across docker exec sessions
grep -q "HSA_OVERRIDE_GFX_VERSION" ~/.bashrc 2>/dev/null || cat >> ~/.bashrc << 'ENVEOF'

# MINDI 1.5 MI300X environment
export PYTORCH_ROCM_ARCH=gfx942
export HIP_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=mindi-1.5-vision-coder
ENVEOF
echo "  .env file created + bashrc updated"

# ── Step 6: GPU stress test ────────────────────────────────────
echo ""
echo "[6/7] Running GPU verification + bf16 test ..."
python3 -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU name:        {torch.cuda.get_device_name(0)}')
    vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    print(f'  VRAM:            {vram:.0f} GB')
    print(f'  ROCm backend:    {torch.version.hip is not None}')
    # bf16 matmul test
    x = torch.randn(1000, 1000, dtype=torch.bfloat16, device='cuda')
    y = torch.matmul(x, x.T)
    print(f'  bf16 matmul:     PASSED (shape={y.shape})')
    # Memory allocation test
    big = torch.zeros(1024, 1024, 1024, dtype=torch.bfloat16, device='cuda')  # ~2GB
    print(f'  2GB alloc test:  PASSED')
    del big
    torch.cuda.empty_cache()
else:
    print('  ERROR: No GPU detected!')
    exit(1)
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
