#!/usr/bin/env python3
"""
MINDI 1.5 Vision-Coder — Train / Validation / Test Split

Splits mindi_filtered.jsonl into:
  - train.jsonl      (90%)
  - val.jsonl         (5%)
  - test.jsonl        (5%)

Stratified by source to ensure proportional representation.
Deterministic with a fixed random seed.

Usage:
    python scripts/split_data.py                        # Default 90/5/5
    python scripts/split_data.py --train 0.85 --val 0.10 --test 0.05
    python scripts/split_data.py --seed 42
    python scripts/split_data.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "mindi_filtered.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
VAL_FILE = OUTPUT_DIR / "val.jsonl"
TEST_FILE = OUTPUT_DIR / "test.jsonl"


def run_split(
    train_ratio: float = 0.90,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
    dry_run: bool = False,
) -> None:
    """Split filtered data into train/val/test with stratification by source."""

    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        print(f"ERROR: Ratios must sum to 1.0, got {total_ratio:.3f}")
        sys.exit(1)

    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        print("  Run quality_filter.py first to generate mindi_filtered.jsonl")
        sys.exit(1)

    print(f"Loading examples from {INPUT_FILE.name} ...")
    start = time.time()

    # Group lines by source for stratified splitting
    source_lines: dict[str, list[str]] = {}
    total = 0
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                example = json.loads(line)
                source = example.get("source", "unknown")
            except json.JSONDecodeError:
                source = "unknown"
            source_lines.setdefault(source, []).append(line)

    load_time = time.time() - start
    print(f"  Loaded {total:,} examples in {load_time:.1f}s")
    print(f"  Sources: {len(source_lines)}")
    print()

    # Split settings
    print(f"Split ratios: train={train_ratio:.0%} / val={val_ratio:.0%} / test={test_ratio:.0%}")
    print(f"Random seed:  {seed}")
    print(f"Dry run:      {dry_run}")
    print()

    rng = random.Random(seed)

    train_lines: list[str] = []
    val_lines: list[str] = []
    test_lines: list[str] = []

    source_stats: dict[str, dict[str, int]] = {}

    for source in sorted(source_lines.keys()):
        lines = source_lines[source]
        rng.shuffle(lines)

        n = len(lines)
        n_val = max(1, round(n * val_ratio)) if n >= 3 else 0
        n_test = max(1, round(n * test_ratio)) if n >= 3 else 0
        n_train = n - n_val - n_test

        # Edge case: if too few examples, put all in train
        if n < 3:
            n_train = n
            n_val = 0
            n_test = 0

        train_lines.extend(lines[:n_train])
        val_lines.extend(lines[n_train:n_train + n_val])
        test_lines.extend(lines[n_train + n_val:])

        source_stats[source] = {
            "total": n,
            "train": n_train,
            "val": n_val,
            "test": n_test,
        }

    # Shuffle final lists (so sources are interleaved)
    rng.shuffle(train_lines)
    rng.shuffle(val_lines)
    rng.shuffle(test_lines)

    # ── Summary ───────────────────────────────────────────────────
    print("=" * 60)
    print("  SPLIT SUMMARY")
    print("=" * 60)
    print(f"  Total:       {total:>10,}")
    print(f"  Train:       {len(train_lines):>10,} ({len(train_lines)/total*100:.1f}%)")
    print(f"  Validation:  {len(val_lines):>10,} ({len(val_lines)/total*100:.1f}%)")
    print(f"  Test:        {len(test_lines):>10,} ({len(test_lines)/total*100:.1f}%)")
    print()

    print("  Per-source breakdown:")
    print(f"    {'Source':<25s} {'Total':>8s} {'Train':>8s} {'Val':>8s} {'Test':>8s}")
    print(f"    {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for source in sorted(source_stats.keys()):
        s = source_stats[source]
        print(f"    {source:<25s} {s['total']:>8,} {s['train']:>8,} {s['val']:>8,} {s['test']:>8,}")
    print()

    if not dry_run:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        print("Writing files ...")
        for path, lines, name in [
            (TRAIN_FILE, train_lines, "train"),
            (VAL_FILE, val_lines, "val"),
            (TEST_FILE, test_lines, "test"),
        ]:
            with open(path, "w", encoding="utf-8") as f:
                for line in lines:
                    f.write(line + "\n")
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  {name:<12s} → {path.name:<20s} ({len(lines):>10,} examples, {size_mb:>8.1f} MB)")

        # Save split metadata
        meta = {
            "total": total,
            "train_count": len(train_lines),
            "val_count": len(val_lines),
            "test_count": len(test_lines),
            "train_pct": round(len(train_lines) / total * 100, 2),
            "val_pct": round(len(val_lines) / total * 100, 2),
            "test_pct": round(len(test_lines) / total * 100, 2),
            "seed": seed,
            "source_breakdown": source_stats,
        }
        meta_path = OUTPUT_DIR / "split_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"  Metadata    → {meta_path.name}")

    elapsed = time.time() - start
    print(f"\n  Done in {elapsed:.1f}s")
    print("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MINDI Data Splitter — stratified train/val/test split",
    )
    parser.add_argument("--train", type=float, default=0.90, help="Train ratio (default: 0.90)")
    parser.add_argument("--val", type=float, default=0.05, help="Validation ratio (default: 0.05)")
    parser.add_argument("--test", type=float, default=0.05, help="Test ratio (default: 0.05)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--dry-run", action="store_true", help="Preview split without writing files")

    args = parser.parse_args()
    run_split(
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
