#!/usr/bin/env python3
"""Quick GPU diagnostic for MI300X before full training."""
import sys
import torch

print("=" * 50)
print("  MI300X GPU Diagnostic")
print("=" * 50)

# Step 1: Basic GPU info
print("\n[1] GPU Info:")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("  FATAL: No GPU!")
    sys.exit(1)

print(f"  GPU: {torch.cuda.get_device_name(0)}")
props = torch.cuda.get_device_properties(0)
vram_gb = props.total_memory / (1024**3)
print(f"  VRAM: {vram_gb:.0f} GB")
print(f"  ROCm: {torch.version.hip}")

# Step 2: Small tensor test
print("\n[2] Small tensor test:")
x = torch.randn(10, 10, device='cuda', dtype=torch.bfloat16)
y = x @ x.T
print(f"  bf16 matmul: OK (shape={y.shape})")
del x, y
torch.cuda.empty_cache()

# Step 3: Larger allocation
print("\n[3] Large allocation test (1GB):")
big = torch.zeros(256, 1024, 1024, dtype=torch.bfloat16, device='cuda')
print(f"  1GB alloc: OK")
del big
torch.cuda.empty_cache()

# Step 4: Try loading model with from_pretrained on CPU
print("\n[4] Loading Qwen2.5-Coder-7B to CPU ...")
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map=None,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
param_count = sum(p.numel() for p in model.parameters())
print(f"  Loaded: {param_count / 1e9:.2f}B params on CPU")

# Step 5: Move layer by layer to test
print("\n[5] Moving model to CUDA (layer by layer) ...")
try:
    model = model.to('cuda')
    print(f"  Model on CUDA: OK")
    used = torch.cuda.memory_allocated() / (1024**3)
    print(f"  VRAM used: {used:.1f} GB")
except Exception as e:
    print(f"  FAILED: {e}")
    print("  Trying half() first ...")
    model = model.half().to('cuda')
    used = torch.cuda.memory_allocated() / (1024**3)
    print(f"  VRAM used: {used:.1f} GB")

# Step 6: Quick forward pass
print("\n[6] Forward pass test ...")
input_ids = torch.tensor([[1, 2, 3, 4, 5]], device='cuda')
with torch.no_grad():
    out = model(input_ids)
print(f"  Forward: OK (logits shape={out.logits.shape})")

print("\n" + "=" * 50)
print("  ALL TESTS PASSED!")
print("=" * 50)
