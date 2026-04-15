#!/usr/bin/env python3
"""
MINDI 1.5 Vision-Coder — Quality Filter Pipeline

Filters mindi_all.jsonl to remove low-quality examples:
  1. Token length filter   — drop if <50 tokens or >4096 tokens
  2. Duplicate detection   — SHA-256 hash of assistant content
  3. JSON structure check  — valid schema with required fields
  4. Special token check   — assistant must have code_start/code_end pair
  5. Quality score filter  — keep only quality_score >= 5.0
  6. Content heuristics    — drop empty/trivial/boilerplate responses

Usage:
    python scripts/quality_filter.py                  # Full run
    python scripts/quality_filter.py --dry-run        # Preview only
    python scripts/quality_filter.py --min-tokens 100 # Custom min tokens
    python scripts/quality_filter.py --max-tokens 8192 # Custom max tokens
    python scripts/quality_filter.py --min-quality 7.0 # Stricter quality
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "mindi_all.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "mindi_filtered.jsonl"
REJECT_FILE = PROJECT_ROOT / "data" / "processed" / "mindi_rejected.jsonl"
REPORT_FILE = PROJECT_ROOT / "data" / "processed" / "filter_report.json"

# ── Required schema fields ────────────────────────────────────────────

REQUIRED_FIELDS = {"id", "type", "source", "messages", "metadata"}
REQUIRED_METADATA = {"language", "tokens"}
VALID_ROLES = {"system", "user", "assistant"}

# ── Protected sources (hand-crafted gold data — lighter filtering) ─────

PROTECTED_SOURCES = {"sandbox_examples", "search_examples", "synthetic_nextjs"}

# ── MINDI agentic token scoring bonuses ───────────────────────────────
#   Examples with these tokens teach the model to be an *agent*.
#   Each occurrence adds to the quality_score before the threshold.

MINDI_TOKEN_BONUSES = {
    "<|think_start|>": 2.0,
    "<|search_start|>": 3.0,
    "<|error_start|>": 3.0,
    "<|sandbox_start|>": 3.0,
    "<|critique_start|>": 2.0,
    "<|suggest_start|>": 1.0,
}

# ── Special token pairs that assistant messages should contain ─────────

CODE_TOKEN_PAIRS = [
    ("<|code_start|>", "<|code_end|>"),
]

# At least one of these pairs should be present in assistant content
OPTIONAL_TOKEN_PAIRS = [
    ("<|think_start|>", "<|think_end|>"),
    ("<|critique_start|>", "<|critique_end|>"),
    ("<|suggest_start|>", "<|suggest_end|>"),
    ("<|file_start|>", "<|file_end|>"),
    ("<|search_start|>", "<|search_end|>"),
    ("<|sandbox_start|>", "<|sandbox_end|>"),
    ("<|error_start|>", "<|error_end|>"),
    ("<|fix_start|>", "<|fix_end|>"),
]

# ── Rejection reasons ─────────────────────────────────────────────────

class Reason:
    INVALID_JSON = "invalid_json"
    MISSING_FIELDS = "missing_fields"
    MISSING_METADATA = "missing_metadata"
    NO_MESSAGES = "no_messages"
    BAD_ROLES = "bad_message_roles"
    NO_ASSISTANT = "no_assistant_message"
    EMPTY_ASSISTANT = "empty_assistant_content"
    TOO_SHORT = "too_few_tokens"
    TOO_LONG = "too_many_tokens"
    DUPLICATE = "duplicate_content"
    LOW_QUALITY = "low_quality_score"
    NO_CODE_TOKENS = "missing_code_tokens"
    BOILERPLATE = "boilerplate_content"
    UNMATCHED_TOKENS = "unmatched_special_tokens"


# ── Filter functions ──────────────────────────────────────────────────

def validate_schema(example: dict) -> str | None:
    """Check required fields and structure. Returns rejection reason or None."""
    # Top-level fields
    missing = REQUIRED_FIELDS - set(example.keys())
    if missing:
        return Reason.MISSING_FIELDS

    # Metadata fields
    meta = example.get("metadata", {})
    if not isinstance(meta, dict):
        return Reason.MISSING_METADATA
    missing_meta = REQUIRED_METADATA - set(meta.keys())
    if missing_meta:
        return Reason.MISSING_METADATA

    # Messages array
    messages = example.get("messages", [])
    if not isinstance(messages, list) or len(messages) == 0:
        return Reason.NO_MESSAGES

    # Role validation
    for msg in messages:
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            return Reason.BAD_ROLES
        if msg["role"] not in VALID_ROLES:
            return Reason.BAD_ROLES

    return None


def get_assistant_content(example: dict) -> str:
    """Extract concatenated assistant message content."""
    parts = []
    for msg in example.get("messages", []):
        if msg.get("role") == "assistant":
            parts.append(msg.get("content", ""))
    return "\n".join(parts)


def check_assistant_exists(example: dict) -> str | None:
    """Must have at least one assistant message with non-empty content."""
    content = get_assistant_content(example)
    if not content:
        return Reason.NO_ASSISTANT
    if len(content.strip()) < 10:
        return Reason.EMPTY_ASSISTANT
    return None


def check_token_length(example: dict, min_tokens: int, max_tokens: int) -> str | None:
    """Filter by token count stored in metadata."""
    tokens = example.get("metadata", {}).get("tokens", 0)
    if tokens < min_tokens:
        return Reason.TOO_SHORT
    if tokens > max_tokens:
        return Reason.TOO_LONG
    return None


def compute_mindi_bonus(example: dict) -> float:
    """Compute bonus score for MINDI agentic special tokens."""
    content = get_assistant_content(example)
    bonus = 0.0
    for token, value in MINDI_TOKEN_BONUSES.items():
        if token in content:
            bonus += value
    return bonus


def check_quality_score(example: dict, min_quality: float) -> str | None:
    """Filter by quality_score + MINDI token bonus."""
    score = example.get("metadata", {}).get("quality_score", 0.0)
    score += compute_mindi_bonus(example)
    if score < min_quality:
        return Reason.LOW_QUALITY
    return None


def check_code_tokens(example: dict) -> str | None:
    """Assistant content must contain code_start/code_end pair."""
    content = get_assistant_content(example)

    for start_tok, end_tok in CODE_TOKEN_PAIRS:
        if start_tok in content and end_tok in content:
            # Check ordering: start before end
            if content.index(start_tok) < content.rindex(end_tok):
                return None  # OK

    return Reason.NO_CODE_TOKENS


def check_unmatched_tokens(example: dict) -> str | None:
    """Ensure all special token pairs are properly matched (start count == end count)."""
    content = get_assistant_content(example)
    all_pairs = CODE_TOKEN_PAIRS + OPTIONAL_TOKEN_PAIRS

    for start_tok, end_tok in all_pairs:
        start_count = content.count(start_tok)
        end_count = content.count(end_tok)
        if start_count != end_count:
            return Reason.UNMATCHED_TOKENS

    return None


def check_boilerplate(example: dict) -> str | None:
    """Detect boilerplate/placeholder assistant responses."""
    content = get_assistant_content(example)
    content_lower = content.lower().strip()

    # Very short code blocks (just placeholder)
    code_markers = ("<|code_start|>", "<|code_end|>")
    if code_markers[0] in content and code_markers[1] in content:
        start_idx = content.index(code_markers[0]) + len(code_markers[0])
        end_idx = content.index(code_markers[1])
        code_body = content[start_idx:end_idx].strip()
        if len(code_body) < 5:
            return Reason.BOILERPLATE

    # Repetitive content (same char repeated)
    stripped = content_lower.replace(" ", "").replace("\n", "")
    if len(stripped) > 20:
        unique_chars = len(set(stripped))
        if unique_chars < 5:
            return Reason.BOILERPLATE

    return None


def content_hash(example: dict) -> str:
    """SHA-256 hash of assistant content for deduplication."""
    content = get_assistant_content(example)
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()


# ── Main pipeline ─────────────────────────────────────────────────────

def run_filter(
    dry_run: bool = False,
    min_tokens: int = 50,
    max_tokens: int = 4096,
    min_quality: float = 5.0,
) -> None:
    """Run the full quality filter pipeline."""

    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        sys.exit(1)

    # Count input lines
    print(f"Counting input examples from {INPUT_FILE.name} ...")
    total_input = sum(1 for _ in open(INPUT_FILE, "r", encoding="utf-8"))
    print(f"  Total input: {total_input:,} examples")
    print()

    # Filter settings
    print("Filter settings:")
    print(f"  Min tokens:   {min_tokens}")
    print(f"  Max tokens:   {max_tokens}")
    print(f"  Min quality:  {min_quality}")
    print(f"  Dry run:      {dry_run}")
    print()

    # Stats tracking
    kept = 0
    rejected = 0
    reject_reasons: Counter = Counter()
    source_kept: Counter = Counter()
    source_rejected: Counter = Counter()
    seen_hashes: set[str] = set()
    token_sum = 0
    quality_sum = 0.0

    # Type distribution
    type_counts: Counter = Counter()

    # Language distribution
    lang_counts: Counter = Counter()

    start_time = time.time()

    out_f = None
    rej_f = None
    if not dry_run:
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        out_f = open(OUTPUT_FILE, "w", encoding="utf-8")
        rej_f = open(REJECT_FILE, "w", encoding="utf-8")

    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                # Parse JSON
                try:
                    example = json.loads(line)
                except json.JSONDecodeError:
                    reject_reasons[Reason.INVALID_JSON] += 1
                    rejected += 1
                    if rej_f:
                        rej_f.write(line + "\n")
                    continue

                source = example.get("source", "unknown")
                is_protected = source in PROTECTED_SOURCES

                # Run filter chain (order matters: cheapest first)
                # Protected sources: schema + assistant + token length + unmatched only
                # Regular sources: full chain + dedup
                if is_protected:
                    rejection = (
                        validate_schema(example)
                        or check_assistant_exists(example)
                        or check_token_length(example, min_tokens, max_tokens)
                        or check_unmatched_tokens(example)
                    )
                else:
                    rejection = (
                        validate_schema(example)
                        or check_assistant_exists(example)
                        or check_token_length(example, min_tokens, max_tokens)
                        or check_quality_score(example, min_quality)
                        or check_code_tokens(example)
                        or check_unmatched_tokens(example)
                        or check_boilerplate(example)
                    )

                if rejection is None and not is_protected:
                    # Dedup check (skip for protected sources)
                    h = content_hash(example)
                    if h in seen_hashes:
                        rejection = Reason.DUPLICATE

                if rejection is not None:
                    reject_reasons[rejection] += 1
                    source_rejected[source] += 1
                    rejected += 1
                    if rej_f:
                        rej_f.write(line + "\n")
                    continue

                # Passed all filters
                if not is_protected:
                    seen_hashes.add(h)
                kept += 1
                source_kept[source] += 1
                token_sum += example.get("metadata", {}).get("tokens", 0)
                quality_sum += example.get("metadata", {}).get("quality_score", 0.0)
                type_counts[example.get("type", "unknown")] += 1
                lang_counts[example.get("metadata", {}).get("language", "unknown")] += 1

                if out_f:
                    out_f.write(line + "\n")

                # Progress
                if line_num % 50000 == 0:
                    elapsed = time.time() - start_time
                    rate = line_num / elapsed if elapsed > 0 else 0
                    pct = (line_num / total_input) * 100
                    print(f"  [{pct:5.1f}%] Processed {line_num:>10,} | Kept {kept:>10,} | Rejected {rejected:>10,} | {rate:,.0f} ex/s")

    finally:
        if out_f:
            out_f.close()
        if rej_f:
            rej_f.close()

    elapsed = time.time() - start_time

    # ── Summary report ────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  QUALITY FILTER REPORT")
    print("=" * 60)
    print(f"  Input:       {total_input:>10,} examples")
    print(f"  Kept:        {kept:>10,} examples ({kept/total_input*100:.1f}%)")
    print(f"  Rejected:    {rejected:>10,} examples ({rejected/total_input*100:.1f}%)")
    print(f"  Time:        {elapsed:>10.1f} seconds")
    print(f"  Rate:        {total_input/elapsed:>10,.0f} examples/sec")
    print()

    if kept > 0:
        print(f"  Avg tokens:  {token_sum/kept:>10.0f}")
        print(f"  Avg quality: {quality_sum/kept:>10.2f}")
        print(f"  Total tokens:{token_sum:>10,}")
        print()

    # Rejection breakdown
    print("  Rejection breakdown:")
    for reason, count in reject_reasons.most_common():
        pct = count / total_input * 100
        print(f"    {reason:<30s} {count:>10,} ({pct:.1f}%)")
    print()

    # Source breakdown
    print("  Source breakdown (kept / total):")
    all_sources = sorted(set(list(source_kept.keys()) + list(source_rejected.keys())))
    for src in all_sources:
        k = source_kept.get(src, 0)
        total = k + source_rejected.get(src, 0)
        pct = k / total * 100 if total > 0 else 0
        print(f"    {src:<25s} {k:>8,} / {total:>8,} ({pct:.1f}%)")
    print()

    # Type distribution
    print("  Type distribution (kept):")
    for t, c in type_counts.most_common(10):
        print(f"    {t:<25s} {c:>8,}")
    print()

    # Language distribution (top 15)
    print("  Language distribution (kept, top 15):")
    for lang, c in lang_counts.most_common(15):
        print(f"    {lang:<25s} {c:>8,}")
    print()

    if not dry_run:
        print(f"  Output:  {OUTPUT_FILE}")
        print(f"  Rejects: {REJECT_FILE}")

        # Save machine-readable report
        report = {
            "input_count": total_input,
            "kept_count": kept,
            "rejected_count": rejected,
            "kept_pct": round(kept / total_input * 100, 2),
            "avg_tokens": round(token_sum / kept, 1) if kept > 0 else 0,
            "avg_quality": round(quality_sum / kept, 3) if kept > 0 else 0,
            "total_tokens": token_sum,
            "elapsed_seconds": round(elapsed, 1),
            "filter_settings": {
                "min_tokens": min_tokens,
                "max_tokens": max_tokens,
                "min_quality": min_quality,
            },
            "rejection_breakdown": dict(reject_reasons.most_common()),
            "source_kept": dict(source_kept),
            "source_rejected": dict(source_rejected),
            "type_distribution": dict(type_counts.most_common()),
            "language_distribution": dict(lang_counts.most_common(30)),
        }
        with open(REPORT_FILE, "w", encoding="utf-8") as rf:
            json.dump(report, rf, indent=2)
        print(f"  Report:  {REPORT_FILE}")

    print("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MINDI Quality Filter — remove low-quality training examples",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview counts without writing output")
    parser.add_argument("--min-tokens", type=int, default=50, help="Minimum token count (default: 50)")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum token count (default: 4096)")
    parser.add_argument("--min-quality", type=float, default=5.0, help="Minimum quality_score (default: 5.0)")

    args = parser.parse_args()
    run_filter(
        dry_run=args.dry_run,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        min_quality=args.min_quality,
    )


if __name__ == "__main__":
    main()
