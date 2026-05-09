#!/usr/bin/env python3
"""
Check and recover MINDI 1.5 checkpoints from Modal.com persistent volume.

Usage:
    1. pip install modal
    2. modal token new  (authenticate if needed)
    3. python -m modal run scripts/check_modal_volume.py
"""

import modal
import os
from pathlib import Path

VOLUME_PATH = "/mnt/mindi"

app = modal.App("mindi-check-volume")
volume = modal.Volume.from_name("mindi-data", create_if_missing=False)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface_hub",
)


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=1800,
)
def inspect_volume():
    """List all checkpoint files on the Modal volume."""
    work_dir = Path(VOLUME_PATH) / "workspace"
    ckpt_dir = work_dir / "checkpoints"

    print("=" * 60)
    print("  MINDI 1.5 — Modal Volume Inspection")
    print("=" * 60)
    print(f"\nVolume path: {VOLUME_PATH}")
    print(f"Workspace:   {work_dir}")
    print(f"Checkpoints: {ckpt_dir}")
    print()

    # Check workspace exists
    if not work_dir.exists():
        print("[ERROR] Workspace directory NOT FOUND!")
        print("        The volume may be empty or corrupted.")
        return {"error": "workspace not found"}

    # List workspace contents
    print("[INFO] Workspace contents:")
    for item in sorted(work_dir.iterdir()):
        if item.is_dir():
            size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
            print(f"  [DIR]  {item.name:30s} {size / 1e6:10.1f} MB")
        else:
            print(f"  [FILE] {item.name:30s} {item.stat().st_size / 1e6:10.1f} MB")
    print()

    # Check checkpoints
    if not ckpt_dir.exists():
        print("[ERROR] No checkpoints directory found!")
        return {"error": "no checkpoints dir"}

    print("[INFO] Checkpoint directories found:")
    checkpoints = []
    for d in sorted(ckpt_dir.rglob("*")):
        if d.is_dir() and any(d.iterdir()):
            files = list(d.rglob("*"))
            file_count = sum(1 for f in files if f.is_file())
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            print(f"  {d.relative_to(ckpt_dir)}")
            print(f"    Files: {file_count}, Size: {total_size / 1e6:.1f} MB")

            # List key files
            for f in sorted(d.rglob("*")):
                if f.is_file() and f.suffix in ('.safetensors', '.pt', '.bin', '.pth', '.json', '.yaml'):
                    print(f"      {f.name:50s} {f.stat().st_size / 1e6:8.1f} MB")
            print()

            checkpoints.append({
                "path": str(d.relative_to(ckpt_dir)),
                "file_count": file_count,
                "size_mb": total_size / 1e6,
            })

    # Summary
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    if checkpoints:
        total_mb = sum(c["size_mb"] for c in checkpoints)
        print(f"  Found {len(checkpoints)} checkpoint directories")
        print(f"  Total size: {total_mb:.1f} MB ({total_mb / 1024:.1f} GB)")
        for c in checkpoints:
            print(f"    - {c['path']}: {c['file_count']} files, {c['size_mb']:.1f} MB")
    else:
        print("  [WARNING] No checkpoint directories with files found!")

    return {
        "workspace_exists": work_dir.exists(),
        "checkpoints_found": len(checkpoints),
        "checkpoint_details": checkpoints,
    }


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=3600,
    secrets=[modal.Secret.from_dict({
        "HF_TOKEN": os.environ.get("HF_TOKEN", "")
    })] if os.environ.get("HF_TOKEN") else [],
)
def upload_to_hf(target_checkpoint: str = None):
    """Upload checkpoints from Modal volume to HuggingFace."""
    from huggingface_hub import HfApi

    work_dir = Path(VOLUME_PATH) / "workspace"
    ckpt_dir = work_dir / "checkpoints"
    token = os.environ.get("HF_TOKEN", "")

    HF_REPO = "Mindigenous/MINDI-1.5-Vision-Coder"

    print("=" * 60)
    print("  MINDI 1.5 — Upload Checkpoints to HuggingFace")
    print("=" * 60)
    print(f"\nHF Repo: {HF_REPO}")
    print(f"Token:   {'set' if token else 'NOT SET — will fail!'}")
    print()

    if not token:
        print("[ERROR] HF_TOKEN not set! Pass it as an environment variable:")
        print("        HF_TOKEN=hf_... python -m modal run scripts/check_modal_volume.py::upload_to_hf")
        return {"error": "no token"}

    api = HfApi(token=token)

    if target_checkpoint:
        # Upload specific checkpoint
        src = ckpt_dir / target_checkpoint
        if not src.exists():
            print(f"[ERROR] Checkpoint not found: {src}")
            return {"error": "checkpoint not found"}

        dirs = [src]
    else:
        # Upload all checkpoint directories
        dirs = [d for d in ckpt_dir.rglob("*") if d.is_dir() and any(d.iterdir())]
        # Filter to actual checkpoint dirs (have model files)
        dirs = [
            d for d in dirs
            if any(f.suffix in ('.safetensors', '.pt', '.bin') for f in d.rglob("*") if f.is_file())
        ]

    print(f"[INFO] Uploading {len(dirs)} checkpoint(s)...")
    uploaded = []

    for d in dirs:
        rel_path = d.relative_to(ckpt_dir)
        hf_path_prefix = f"checkpoints/{rel_path}"

        print(f"\n[UPLOAD] {rel_path} -> {hf_path_prefix}/")
        files = [f for f in d.rglob("*") if f.is_file()]

        for f in sorted(files):
            rel_file = f.relative_to(d)
            hf_path = f"{hf_path_prefix}/{rel_file}"
            size_mb = f.stat().st_size / 1e6

            print(f"  {rel_file} ({size_mb:.1f} MB) ... ", end="", flush=True)
            try:
                api.upload_file(
                    path_or_fileobj=str(f),
                    path_in_repo=hf_path,
                    repo_id=HF_REPO,
                    repo_type="model",
                )
                print("OK")
            except Exception as e:
                print(f"FAIL: {e}")

        uploaded.append(str(rel_path))
        print(f"  [DONE] {rel_path}")

    print("\n" + "=" * 60)
    print("  UPLOAD COMPLETE")
    print("=" * 60)
    print(f"  Uploaded {len(uploaded)} checkpoint(s):")
    for u in uploaded:
        print(f"    - {u}")

    return {"uploaded": uploaded}


@app.local_entrypoint()
def main():
    print("=" * 60)
    print("  MINDI 1.5 — Modal Volume Check & Recovery")
    print("=" * 60)
    print()
    print("Step 1: Inspecting volume contents...")
    result = inspect_volume.remote()
    print(f"\nResult: {result}")

    if result.get("checkpoints_found", 0) > 0:
        print("\n" + "=" * 60)
        print("CHECKPOINTS FOUND!")
        print("=" * 60)
        print("\nTo upload to HuggingFace, run:")
        print("  HF_TOKEN=hf_your_token_here modal run scripts/check_modal_volume.py::upload_to_hf")
        print("\nOr upload a specific checkpoint:")
        print("  HF_TOKEN=hf_... modal run scripts/check_modal_volume.py::upload_to_hf --target-checkpoint training/phase3_final")
    else:
        print("\n[WARNING] No checkpoints found on Modal volume.")
        print("          Check the AMD GPU droplet instead.")
