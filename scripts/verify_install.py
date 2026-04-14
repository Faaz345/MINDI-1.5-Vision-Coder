"""
MINDI 1.5 Vision-Coder — Installation Verification Script

Checks that every required package is importable and reports
versions + GPU status. Run after Phase 3 setup.
"""

from __future__ import annotations

import sys
from importlib.metadata import version as pkg_version


def check(package_name: str, import_name: str | None = None) -> bool:
    """Try to import a package and report status."""
    mod = import_name or package_name
    try:
        __import__(mod)
        v = pkg_version(package_name)
        print(f"  \u2705 {package_name} {v}")
        return True
    except Exception as e:
        print(f"  \u274c {package_name} — FAILED — {e}")
        return False


def check_cuda() -> bool:
    """Verify PyTorch CUDA availability."""
    try:
        import torch
        v = torch.__version__
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
            print(f"  \u2705 torch {v} — CUDA available — {gpu} ({vram} GB)")
            return True
        else:
            print(f"  \u26a0\ufe0f torch {v} — NO CUDA (CPU only)")
            return False
    except Exception as e:
        print(f"  \u274c torch — FAILED — {e}")
        return False


def main() -> None:
    print("=" * 60)
    print("  MINDI 1.5 Vision-Coder — Package Verification")
    print("=" * 60)
    print(f"\n  Python: {sys.version}")
    print(f"  Executable: {sys.executable}\n")

    results: list[bool] = []

    print("[PyTorch + CUDA]")
    results.append(check_cuda())
    check("torchvision")
    check("torchaudio")

    print("\n[Group A — Core Transformers]")
    for pkg in ["transformers", "datasets", "tokenizers", "accelerate", "peft", "huggingface-hub"]:
        imp = pkg.replace("-", "_")
        results.append(check(pkg, imp))

    print("\n[Group B — Vision]")
    results.append(check("pillow", "PIL"))
    results.append(check("opencv-python", "cv2"))
    results.append(check("open-clip-torch", "open_clip"))

    print("\n[Group C — Search]")
    for pkg, imp in [("tavily-python", "tavily"), ("duckduckgo-search", "duckduckgo_search"),
                     ("beautifulsoup4", "bs4"), ("playwright", "playwright"),
                     ("requests", "requests"), ("httpx", "httpx"), ("lxml", "lxml")]:
        results.append(check(pkg, imp))

    print("\n[Group D — Sandbox]")
    results.append(check("e2b"))
    results.append(check("docker"))

    print("\n[Group E — Web Framework]")
    for pkg, imp in [("fastapi", "fastapi"), ("uvicorn", "uvicorn"), ("websockets", "websockets"),
                      ("python-multipart", "multipart"), ("python-jose", "jose"), ("passlib", "passlib")]:
        results.append(check(pkg, imp))

    print("\n[Group F — Training Utilities]")
    for pkg, imp in [("wandb", "wandb"), ("bitsandbytes", "bitsandbytes"), ("scipy", "scipy"),
                      ("scikit-learn", "sklearn"), ("einops", "einops")]:
        results.append(check(pkg, imp))

    print("\n[Group G — Vector Store / RAG]")
    results.append(check("faiss-cpu", "faiss"))
    results.append(check("sentence-transformers", "sentence_transformers"))

    print("\n[Group H — Utilities]")
    for pkg, imp in [("rich", "rich"), ("tqdm", "tqdm"), ("python-dotenv", "dotenv"),
                     ("pyyaml", "yaml"), ("numpy", "numpy"), ("pandas", "pandas"),
                     ("matplotlib", "matplotlib")]:
        results.append(check(pkg, imp))

    print("\n[Group I — Code Quality]")
    for pkg in ["black", "isort", "mypy"]:
        results.append(check(pkg))

    # Summary
    passed = sum(results)
    total = len(results)
    print("\n" + "=" * 60)
    if passed == total:
        print(f"  \u2705 ALL {total} PACKAGES VERIFIED — READY TO BUILD!")
    else:
        print(f"  \u26a0\ufe0f {passed}/{total} passed — {total - passed} need fixing")
    print("=" * 60)


if __name__ == "__main__":
    main()
