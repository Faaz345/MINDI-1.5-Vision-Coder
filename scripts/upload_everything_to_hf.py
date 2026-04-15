#!/usr/bin/env python3
"""
Upload ENTIRE MINDI 1.5 Vision-Coder project to HuggingFace.

REPO 1 (model):   Mindigenous/MINDI-1.5-Vision-Coder
REPO 2 (dataset):  Mindigenous/MINDI-1.5-training-data

Both private.  On MI300X we will clone these repos directly.
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

# ── Repo names ─────────────────────────────────────────────────────────
MODEL_REPO  = "Mindigenous/MINDI-1.5-Vision-Coder"
DATASET_REPO = "Mindigenous/MINDI-1.5-training-data"

# ── Model card (written to repo as README.md) ─────────────────────────
MODEL_CARD = """\
---
license: apache-2.0
language:
- en
tags:
- code-generation
- nextjs
- react
- typescript
- vision
- multimodal
- mindi
- mindigenous
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
---

# MINDI 1.5 Vision-Coder

Built by MINDIGENOUS.AI

## Model Description
MINDI 1.5 is an agentic AI coding model
that sees its own output and critiques it.

## Key Features
- Generates Next.js 14 + Tailwind + TypeScript
- Sees screenshots via CLIP ViT-L/14
- Critiques its own UI/UX output
- Searches internet for latest packages
- Tests code in sandbox environment
- Self-fixes errors automatically

## Training
- Base: Qwen/Qwen2.5-Coder-7B-Instruct
- Method: LoRA fine-tuning
- Hardware: AMD MI300X 192GB VRAM
- Dataset: 1,449,428 examples
- Tokens: 859,694,776
- Status: Training in progress

## Built By
Faaz - MINDIGENOUS.AI
Mumbai, India
April 2026
"""

# ── Dataset card ───────────────────────────────────────────────────────
DATASET_CARD = """\
---
license: apache-2.0
language:
- en
tags:
- code-generation
- nextjs
- react
- typescript
- vision
- multimodal
- mindi
- mindigenous
size_categories:
- 1M<n<10M
---

# MINDI 1.5 Training Data

Training dataset for **MINDI 1.5 Vision-Coder** by MINDIGENOUS.AI

## Dataset Statistics
| Metric | Value |
|--------|-------|
| Total examples | 1,449,428 |
| Total tokens | 859,694,776 |
| Avg tokens/example | 593 |
| Avg quality score | 6.49 |
| Sources | 9 |

## Splits
| Split | Examples | Percentage |
|-------|----------|------------|
| Train | 1,304,486 | 90.0% |
| Validation | 72,471 | 5.0% |
| Test | 72,471 | 5.0% |

## Sources
| Source | Examples | Kept % |
|--------|----------|--------|
| starcoderdata | 569,350 | 94.9% |
| websight | 250,987 | 99.99% |
| evol_code | 155,998 | 99.7% |
| codefeedback | 149,865 | 99.9% |
| magicoder | 149,987 | 99.99% |
| synthetic_nextjs | 90,000 | 100% (protected) |
| codealpaca | 59,241 | 98.8% |
| search_examples | 15,000 | 100% (protected) |
| sandbox_examples | 9,000 | 100% (protected) |

## Type Distribution
| Type | Examples |
|------|----------|
| code_generation | 1,183,441 |
| vision_code | 250,987 |
| search | 15,000 |

## Language Distribution
| Language | Examples |
|----------|----------|
| unknown | 490,305 |
| typescript | 375,859 |
| javascript | 298,497 |
| python | 211,842 |
| html | 36,371 |
| java | 32,458 |
| rust | 3,709 |
| go | 387 |

## Format
Each example is a JSON object with:
- `conversations`: list of `{"role": ..., "content": ...}` turns
- `source`: dataset origin
- `type`: code_generation / vision_code / search
- `language`: programming language
- `quality_score`: heuristic quality (0-10+)
- `token_count`: number of tokens

## Quality Filtering
- Protected sources (sandbox, search, synthetic_nextjs) bypass aggressive filters
- MINDI special token bonuses boost agentic examples
- Dedup via SHA-256 content hashing
- Rejection reasons: too_many_tokens (30,637), boilerplate (1,373), duplicate (59)

## Built By
Faaz - MINDIGENOUS.AI
Mumbai, India — April 2026
"""


# ────────────────────────────────────────────────────────────────────────
def load_token() -> str:
    """Load HF token from .env."""
    load_dotenv(ENV_FILE)
    token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        print("ERROR: No HUGGINGFACE_TOKEN or HF_TOKEN found in .env")
        sys.exit(1)
    return token


def ensure_repo(api: HfApi, repo_id: str, repo_type: str, token: str):
    """Create repo if it doesn't exist."""
    try:
        create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=True,
            token=token,
            exist_ok=True,
        )
        print(f"  Repo ready: {repo_id} ({repo_type})")
    except Exception as e:
        print(f"  Repo create/check: {e}")


def upload_folder(api: HfApi, local: Path, remote: str, repo_id: str,
                  repo_type: str, token: str):
    """Upload a local folder to HF repo."""
    if not local.exists():
        print(f"  SKIP (not found): {local}")
        return
    label = str(local.relative_to(PROJECT_ROOT))
    print(f"  Uploading {label}/ to {repo_type} repo ... ", end="", flush=True)
    t0 = time.time()
    api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(local),
        path_in_repo=remote,
        token=token,
        ignore_patterns=["__pycache__", "*.pyc", ".git"],
    )
    print(f"done ({time.time() - t0:.1f}s)")


def upload_file(api: HfApi, local: Path, remote: str, repo_id: str,
                repo_type: str, token: str):
    """Upload a single file to HF repo."""
    if not local.exists():
        print(f"  SKIP (not found): {local.name}")
        return
    size_mb = local.stat().st_size / (1024 * 1024)
    label = str(local.relative_to(PROJECT_ROOT))
    print(f"  Uploading {label} ({size_mb:.1f} MB) to {repo_type} repo ... ",
          end="", flush=True)
    t0 = time.time()
    api.upload_file(
        repo_id=repo_id,
        repo_type=repo_type,
        path_or_fileobj=str(local),
        path_in_repo=remote,
        token=token,
    )
    print(f"done ({time.time() - t0:.1f}s)")


def upload_readme(api: HfApi, content: str, repo_id: str,
                  repo_type: str, token: str):
    """Upload a README.md string to a repo."""
    print(f"  Uploading README.md to {repo_type} repo ... ", end="", flush=True)
    api.upload_file(
        repo_id=repo_id,
        repo_type=repo_type,
        path_or_fileobj=content.encode("utf-8"),
        path_in_repo="README.md",
        token=token,
    )
    print("done")


# ────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  MINDI 1.5 — Upload Everything to HuggingFace")
    print("=" * 60)
    print()

    token = load_token()
    api = HfApi()

    # ── Create repos ───────────────────────────────────────────────
    print("[1/4] Creating repositories ...")
    ensure_repo(api, MODEL_REPO, "model", token)
    ensure_repo(api, DATASET_REPO, "dataset", token)
    print()

    # ── REPO 1: Model (code + configs) ─────────────────────────────
    print("[2/4] Uploading to MODEL repo:", MODEL_REPO)
    print("-" * 50)

    # Folders
    model_folders = [
        (PROJECT_ROOT / "src",            "src"),
        (PROJECT_ROOT / "scripts",        "scripts"),
        (PROJECT_ROOT / "configs",        "configs"),
        (PROJECT_ROOT / "data" / "tokenizer", "data/tokenizer"),
        (PROJECT_ROOT / "tests",          "tests"),
        (PROJECT_ROOT / "api",            "api"),
    ]
    for local, remote in model_folders:
        upload_folder(api, local, remote, MODEL_REPO, "model", token)

    # Single files
    model_files = [
        (PROJECT_ROOT / "requirements.txt",   "requirements.txt"),
        (PROJECT_ROOT / "setup.py",           "setup.py"),
        (PROJECT_ROOT / "activate_mindi.bat", "activate_mindi.bat"),
        (PROJECT_ROOT / ".env.example",       ".env.example"),
    ]
    for local, remote in model_files:
        upload_file(api, local, remote, MODEL_REPO, "model", token)

    # setup_mi300x.sh
    mi300x_sh = PROJECT_ROOT / "setup_mi300x.sh"
    if mi300x_sh.exists():
        upload_file(api, mi300x_sh, "setup_mi300x.sh", MODEL_REPO, "model", token)

    # Model card replaces README.md
    upload_readme(api, MODEL_CARD, MODEL_REPO, "model", token)
    print()

    # ── REPO 2: Dataset ────────────────────────────────────────────
    print("[3/4] Uploading to DATASET repo:", DATASET_REPO)
    print("-" * 50)

    processed = PROJECT_ROOT / "data" / "processed"
    dataset_files = [
        (processed / "train.jsonl",        "processed/train.jsonl"),
        (processed / "val.jsonl",          "processed/val.jsonl"),
        (processed / "test.jsonl",         "processed/test.jsonl"),
        (processed / "mindi_filtered.jsonl", "processed/mindi_filtered.jsonl"),
        (processed / "filter_report.json", "processed/filter_report.json"),
        (processed / "split_meta.json",    "processed/split_meta.json"),
    ]
    for local, remote in dataset_files:
        upload_file(api, local, remote, DATASET_REPO, "dataset", token)

    # Raw data folder
    upload_folder(
        api, PROJECT_ROOT / "data" / "raw", "raw",
        DATASET_REPO, "dataset", token,
    )

    # Tokenizer copy in dataset repo
    upload_folder(
        api, PROJECT_ROOT / "data" / "tokenizer", "tokenizer",
        DATASET_REPO, "dataset", token,
    )

    # Dataset card
    upload_readme(api, DATASET_CARD, DATASET_REPO, "dataset", token)
    print()

    # ── Done ───────────────────────────────────────────────────────
    print("[4/4] Upload complete!")
    print()
    print("╔══════════════════════════════════════╗")
    print("║ UPLOAD COMPLETE!                     ║")
    print("║                                      ║")
    print("║ Model repo:                          ║")
    print("║ huggingface.co/Mindigenous/           ║")
    print("║ MINDI-1.5-Vision-Coder               ║")
    print("║                                      ║")
    print("║ Dataset repo:                        ║")
    print("║ huggingface.co/datasets/             ║")
    print("║ Mindigenous/MINDI-1.5-training-data  ║")
    print("║                                      ║")
    print("║ On MI300X just run:                  ║")
    print("║ git clone https://huggingface.co/    ║")
    print("║ Mindigenous/MINDI-1.5-Vision-Coder   ║")
    print("║                                      ║")
    print("║ Ready to train! 🚀                   ║")
    print("╚══════════════════════════════════════╝")


if __name__ == "__main__":
    main()
