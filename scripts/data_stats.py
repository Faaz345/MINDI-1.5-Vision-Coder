#!/usr/bin/env python3
"""
MINDI 1.5 Vision-Coder — Dataset Statistics Report

Generates comprehensive statistics for the final train/val/test splits:
  - Total counts and sizes
  - Token distribution (min, max, mean, median, p95, p99)
  - Quality score distribution
  - Source breakdown
  - Type breakdown
  - Language breakdown
  - Special token usage

Usage:
    python scripts/data_stats.py                  # Full report
    python scripts/data_stats.py --split train    # Stats for train only
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SPLIT_FILES = {
    "train": PROCESSED_DIR / "train.jsonl",
    "val": PROCESSED_DIR / "val.jsonl",
    "test": PROCESSED_DIR / "test.jsonl",
}

REPORT_FILE = PROCESSED_DIR / "dataset_stats.json"

# ── Special tokens to check ──────────────────────────────────────────

SPECIAL_TOKENS = [
    "<|think_start|>", "<|think_end|>",
    "<|code_start|>", "<|code_end|>",
    "<|critique_start|>", "<|critique_end|>",
    "<|suggest_start|>", "<|suggest_end|>",
    "<|file_start|>", "<|file_end|>",
    "<|search_start|>", "<|search_end|>",
    "<|sandbox_start|>", "<|sandbox_end|>",
    "<|vision_start|>", "<|vision_end|>",
    "<|error_start|>", "<|error_end|>",
    "<|fix_start|>", "<|fix_end|>",
]


def percentile(sorted_data: list[int | float], p: float) -> float:
    """Calculate the p-th percentile from sorted data."""
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return float(sorted_data[f])
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def compute_stats(file_path: Path, split_name: str) -> dict:
    """Compute statistics for a single split file."""

    if not file_path.exists():
        return {"error": f"File not found: {file_path}"}

    tokens_list: list[int] = []
    quality_list: list[float] = []
    source_counts: Counter = Counter()
    type_counts: Counter = Counter()
    lang_counts: Counter = Counter()
    framework_counts: Counter = Counter()
    has_vision_count = 0
    special_token_counts: Counter = Counter()
    msg_count_dist: Counter = Counter()  # number of messages per example
    total_chars = 0

    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue

            count += 1
            meta = ex.get("metadata", {})

            # Token count
            tokens = meta.get("tokens", 0)
            tokens_list.append(tokens)

            # Quality score
            quality = meta.get("quality_score", 0.0)
            quality_list.append(quality)

            # Source, type, language, framework
            source_counts[ex.get("source", "unknown")] += 1
            type_counts[ex.get("type", "unknown")] += 1
            lang_counts[meta.get("language", "unknown")] += 1
            framework_counts[meta.get("framework", "none")] += 1

            # Vision
            if meta.get("has_vision", False):
                has_vision_count += 1

            # Messages
            messages = ex.get("messages", [])
            msg_count_dist[len(messages)] += 1

            # Special tokens in assistant content
            for msg in messages:
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    total_chars += len(content)
                    for tok in SPECIAL_TOKENS:
                        if tok in content:
                            special_token_counts[tok] += 1

    # Sort for percentile computation
    tokens_sorted = sorted(tokens_list)
    quality_sorted = sorted(quality_list)

    file_size_mb = file_path.stat().st_size / (1024 * 1024)

    stats = {
        "split": split_name,
        "file": file_path.name,
        "file_size_mb": round(file_size_mb, 1),
        "count": count,
        "total_tokens": sum(tokens_list),
        "total_chars_assistant": total_chars,
        "has_vision": has_vision_count,
        "tokens": {
            "min": min(tokens_sorted) if tokens_sorted else 0,
            "max": max(tokens_sorted) if tokens_sorted else 0,
            "mean": round(statistics.mean(tokens_list), 1) if tokens_list else 0,
            "median": round(statistics.median(tokens_list), 1) if tokens_list else 0,
            "stdev": round(statistics.stdev(tokens_list), 1) if len(tokens_list) > 1 else 0,
            "p5": round(percentile(tokens_sorted, 5), 1),
            "p25": round(percentile(tokens_sorted, 25), 1),
            "p75": round(percentile(tokens_sorted, 75), 1),
            "p95": round(percentile(tokens_sorted, 95), 1),
            "p99": round(percentile(tokens_sorted, 99), 1),
        },
        "quality_score": {
            "min": round(min(quality_sorted), 2) if quality_sorted else 0,
            "max": round(max(quality_sorted), 2) if quality_sorted else 0,
            "mean": round(statistics.mean(quality_list), 2) if quality_list else 0,
            "median": round(statistics.median(quality_list), 2) if quality_list else 0,
        },
        "source_distribution": dict(source_counts.most_common()),
        "type_distribution": dict(type_counts.most_common()),
        "language_distribution": dict(lang_counts.most_common(30)),
        "framework_distribution": dict(framework_counts.most_common(15)),
        "messages_per_example": dict(sorted(msg_count_dist.items())),
        "special_token_usage": dict(special_token_counts.most_common()),
    }

    return stats


def print_stats(stats: dict) -> None:
    """Pretty-print statistics for a split."""
    if "error" in stats:
        print(f"  ERROR: {stats['error']}")
        return

    print(f"  Split: {stats['split']}")
    print(f"  File:  {stats['file']} ({stats['file_size_mb']:.1f} MB)")
    print(f"  Count: {stats['count']:,}")
    print(f"  Total tokens: {stats['total_tokens']:,}")
    print(f"  Vision examples: {stats['has_vision']:,}")
    print()

    t = stats["tokens"]
    print(f"  Token distribution:")
    print(f"    Min:    {t['min']:>8,}    P5:     {t['p5']:>8,.0f}")
    print(f"    P25:    {t['p25']:>8,.0f}    Median: {t['median']:>8,.0f}")
    print(f"    Mean:   {t['mean']:>8,.0f}    P75:    {t['p75']:>8,.0f}")
    print(f"    P95:    {t['p95']:>8,.0f}    P99:    {t['p99']:>8,.0f}")
    print(f"    Max:    {t['max']:>8,}    Stdev:  {t['stdev']:>8,.0f}")
    print()

    q = stats["quality_score"]
    print(f"  Quality score: min={q['min']:.1f}  mean={q['mean']:.1f}  median={q['median']:.1f}  max={q['max']:.1f}")
    print()

    print(f"  Source distribution:")
    for src, cnt in stats["source_distribution"].items():
        pct = cnt / stats["count"] * 100
        print(f"    {src:<25s} {cnt:>10,} ({pct:5.1f}%)")
    print()

    print(f"  Type distribution:")
    for t_name, cnt in list(stats["type_distribution"].items())[:10]:
        pct = cnt / stats["count"] * 100
        print(f"    {t_name:<25s} {cnt:>10,} ({pct:5.1f}%)")
    print()

    print(f"  Language distribution (top 15):")
    for lang, cnt in list(stats["language_distribution"].items())[:15]:
        pct = cnt / stats["count"] * 100
        print(f"    {lang:<25s} {cnt:>10,} ({pct:5.1f}%)")
    print()

    if stats["special_token_usage"]:
        print(f"  Special token usage (examples containing token):")
        for tok, cnt in stats["special_token_usage"].items():
            pct = cnt / stats["count"] * 100
            print(f"    {tok:<25s} {cnt:>10,} ({pct:5.1f}%)")
        print()


def run_stats(split: str | None = None) -> None:
    """Generate and display statistics."""
    start = time.time()

    if split:
        files = {split: SPLIT_FILES.get(split)}
        if files[split] is None:
            print(f"ERROR: Unknown split '{split}'. Choose from: {list(SPLIT_FILES.keys())}")
            sys.exit(1)
    else:
        files = SPLIT_FILES

    all_stats = {}

    for name, path in files.items():
        print("=" * 60)
        print(f"  Computing stats for: {name}")
        print("=" * 60)
        stats = compute_stats(path, name)
        all_stats[name] = stats
        print_stats(stats)

    # Save JSON report
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)
    print(f"Full report saved to: {REPORT_FILE.name}")

    elapsed = time.time() - start
    print(f"Stats generated in {elapsed:.1f}s")


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MINDI Dataset Statistics — comprehensive split analysis",
    )
    parser.add_argument("--split", type=str, choices=["train", "val", "test"],
                        help="Compute stats for a single split only")

    args = parser.parse_args()
    run_stats(split=args.split)


if __name__ == "__main__":
    main()
