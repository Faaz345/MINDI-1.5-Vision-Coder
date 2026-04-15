#!/usr/bin/env python3
"""
MINDI 1.5 Vision-Coder — Download WebSight v0.2 Subset

Downloads UI screenshot + HTML/CSS code pairs from HuggingFaceM4/WebSight.
Saves images to data/websight/images/ and creates data/websight/train.jsonl
and data/websight/val.jsonl with the MINDI training format.

Usage:
    python3 scripts/download_websight.py --num_train 50000 --num_val 2500
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Download WebSight dataset subset")
    parser.add_argument("--num_train", type=int, default=50000,
                        help="Number of training examples (default: 50000)")
    parser.add_argument("--num_val", type=int, default=2500,
                        help="Number of validation examples (default: 2500)")
    parser.add_argument("--output_dir", type=str, default="data/websight",
                        help="Output directory")
    parser.add_argument("--version", type=str, default="v0.2",
                        help="WebSight version (v0.1 or v0.2)")
    args = parser.parse_args()

    total = args.num_train + args.num_val
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  MINDI 1.5 — WebSight Dataset Download")
    print("=" * 60)
    print(f"  Version:  {args.version}")
    print(f"  Train:    {args.num_train:,}")
    print(f"  Val:      {args.num_val:,}")
    print(f"  Output:   {output_dir}")
    print()

    # Load dataset with streaming to avoid downloading everything
    print("[1/3] Loading WebSight dataset (streaming) ...")
    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceM4/WebSight",
        args.version,
        split="train",
        streaming=True,
        token=os.environ.get("HF_TOKEN"),
    )

    # Process examples
    print(f"[2/3] Downloading {total:,} examples ...")
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    train_f = open(train_path, "w", encoding="utf-8")
    val_f = open(val_path, "w", encoding="utf-8")

    count = 0
    for i, example in enumerate(ds):
        if i >= total:
            break

        # Extract image and code
        image = example.get("image")
        code = example.get("text", "")

        if image is None or not code.strip():
            continue

        # Save image
        img_filename = f"ws_{i:07d}.jpg"
        img_path = images_dir / img_filename
        image.save(str(img_path), "JPEG", quality=85)

        # Create MINDI-format training example
        entry = {
            "id": f"websight_{i:07d}",
            "type": "vision_code",
            "source": "websight_v0.2",
            "image_path": f"data/websight/images/{img_filename}",
            "messages": [
                {
                    "role": "system",
                    "content": "You are MINDI 1.5 Vision-Coder, a specialized AI for understanding UI screenshots and generating accurate HTML/CSS code."
                },
                {
                    "role": "user",
                    "content": "<|vision_start|><|vision_end|>\nGenerate the HTML/CSS code for this UI screenshot."
                },
                {
                    "role": "assistant",
                    "content": f"<|think_start|>I'll analyze the UI layout and generate the corresponding code.<|think_end|>\n<|code_start|>\n{code.strip()}\n<|code_end|>"
                }
            ],
            "metadata": {
                "dataset": "websight",
                "version": args.version,
            }
        }

        # Split: first num_train → train, rest → val
        if count < args.num_train:
            train_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        else:
            val_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        count += 1
        if count % 1000 == 0:
            print(f"  {count:,}/{total:,} downloaded ...")

    train_f.close()
    val_f.close()

    # Stats
    train_count = min(count, args.num_train)
    val_count = max(0, count - args.num_train)

    print(f"\n[3/3] Done!")
    print(f"  Train: {train_count:,} examples → {train_path}")
    print(f"  Val:   {val_count:,} examples → {val_path}")
    print(f"  Images: {images_dir}")
    print(f"  Disk:  ", end="")
    os.system(f"du -sh {output_dir}")


if __name__ == "__main__":
    main()
