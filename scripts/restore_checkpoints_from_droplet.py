#!/usr/bin/env python3
"""
Restore MINDI 1.5 checkpoints from AMD GPU droplet -> HuggingFace.

This script SSHes into your AMD droplet, finds checkpoint files, downloads them,
and uploads them to the HuggingFace model repo.

Usage:
    1. Ensure you have droplet SSH access (key-based auth preferred)
    2. Set HUGGINGFACE_TOKEN in .env
    3. python scripts/restore_checkpoints_from_droplet.py

Requires: paramiko, huggingface_hub, python-dotenv
    pip install paramiko huggingface_hub python-dotenv
"""

import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

# ── Config ──────────────────────────────────────────────────────────────
DROPLET_IP = "165.245.141.245"      # From your AMD droplet screenshot
DROPLET_USER = "root"               # Change if you use a different user
DROPLET_KEY = None                  # Path to SSH key, or None for password
HF_REPO = "Mindigenous/MINDI-1.5-Vision-Coder"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

# ── Paths to search on droplet ───────────────────────────────────────────
SEARCH_PATHS = [
    "/mnt/mindi",
    "/workspace",
    "/root",
    "/opt/mindi",
    "/home/*/workspace",
    "/home/*/mindi",
]

CKPT_EXTENSIONS = (".safetensors", ".pt", ".bin", ".pth", ".ckpt")


def load_token() -> str:
    load_dotenv(ENV_FILE)
    token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        print("ERROR: No HUGGINGFACE_TOKEN or HF_TOKEN in .env")
        sys.exit(1)
    return token


def ssh_find_checkpoints():
    """SSH into droplet and search for checkpoint files."""
    try:
        import paramiko
    except ImportError:
        print("ERROR: paramiko not installed. Run: pip install paramiko")
        sys.exit(1)

    print(f"[SSH] Connecting to {DROPLET_USER}@{DROPLET_IP} ...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        if DROPLET_KEY and Path(DROPLET_KEY).exists():
            client.connect(DROPLET_IP, username=DROPLET_USER, key_filename=DROPLET_KEY)
        else:
            # Will prompt for password if key not provided
            client.connect(DROPLET_IP, username=DROPLET_USER)
    except Exception as e:
        print(f"ERROR: SSH connection failed: {e}")
        sys.exit(1)

    print("[SSH] Connected. Searching for checkpoint files ...")

    # Search for checkpoint directories and files
    find_cmd = "find " + " ".join(SEARCH_PATHS) + f" -type f \( {' -o '.join(f'-name *{ext}' for ext in CKPT_EXTENSIONS)} \) 2>/dev/null | head -50"
    stdin, stdout, stderr = client.exec_command(find_cmd)
    files = stdout.read().decode().strip().split("\n")
    files = [f.strip() for f in files if f.strip()]

    if not files:
        print("[SSH] NO checkpoint files found on droplet!")
        print("[SSH] Searched:", SEARCH_PATHS)
        client.close()
        return None

    print(f"[SSH] Found {len(files)} potential checkpoint files:")
    for f in files:
        print(f"  {f}")

    # Also get sizes
    size_cmd = "; ".join([f"du -sh {f}" for f in files[:10]])
    stdin, stdout, stderr = client.exec_command(size_cmd)
    sizes = stdout.read().decode().strip()
    print("[SSH] File sizes:")
    print(sizes)

    client.close()
    return files


def download_and_upload(files):
    """Download checkpoint files from droplet and upload to HF."""
    if not files:
        return

    token = load_token()
    api = HfApi(token=token)

    print(f"\n[HF] Uploading to {HF_REPO} ...")

    try:
        import paramiko
    except ImportError:
        print("ERROR: paramiko not installed")
        return

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    if DROPLET_KEY and Path(DROPLET_KEY).exists():
        client.connect(DROPLET_IP, username=DROPLET_USER, key_filename=DROPLET_KEY)
    else:
        client.connect(DROPLET_IP, username=DROPLET_USER)

    sftp = client.open_sftp()

    with tempfile.TemporaryDirectory() as tmpdir:
        for remote_path in files:
            remote = Path(remote_path)
            local_path = Path(tmpdir) / remote.name

            print(f"  Downloading {remote.name} ... ", end="", flush=True)
            try:
                sftp.get(str(remote), str(local_path))
                size_mb = local_path.stat().st_size / (1024 * 1024)
                print(f"done ({size_mb:.1f} MB)")
            except Exception as e:
                print(f"FAILED: {e}")
                continue

            # Upload to HF
            # Organize into checkpoints/ folder
            hf_path = f"checkpoints/recovered/{remote.name}"
            print(f"    Uploading to HF as {hf_path} ... ", end="", flush=True)
            try:
                api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=hf_path,
                    repo_id=HF_REPO,
                    repo_type="model",
                )
                print("done")
            except Exception as e:
                print(f"FAILED: {e}")

    sftp.close()
    client.close()
    print("\n[DONE] Checkpoint recovery complete!")


def main():
    print("=" * 60)
    print("  MINDI 1.5 — Checkpoint Recovery Tool")
    print("=" * 60)
    print()
    print(f"Droplet: {DROPLET_IP}")
    print(f"HF Repo: {HF_REPO}")
    print()

    files = ssh_find_checkpoints()
    if files is None:
        print("\n[FAIL] No checkpoints found on droplet.")
        print("Your model weights are likely lost. Options:")
        print("  1. Check if you have backups elsewhere (S3, Google Drive, etc.)")
        print("  2. Retrain from scratch using the training data in the dataset repo")
        print("  3. Use the base Qwen2.5-Coder-7B without your LoRA fine-tune")
        sys.exit(1)

    download_and_upload(files)


if __name__ == "__main__":
    main()
