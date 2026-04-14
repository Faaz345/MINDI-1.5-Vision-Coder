"""
MINDI 1.5 Vision-Coder — Step 4: Add MINDI Special Tokens

Loads the base Qwen2.5-Coder tokenizer, adds 22 MINDI-specific
special tokens, saves the updated tokenizer, and reports vocab changes.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


MINDI_SPECIAL_TOKENS = [
    "<|mindi_start|>",
    "<|mindi_end|>",
    "<|code_start|>",
    "<|code_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|critique_start|>",
    "<|critique_end|>",
    "<|suggest_start|>",
    "<|suggest_end|>",
    "<|think_start|>",
    "<|think_end|>",
    "<|file_start|>",
    "<|file_end|>",
    "<|search_start|>",
    "<|search_end|>",
    "<|sandbox_start|>",
    "<|sandbox_end|>",
    "<|error_start|>",
    "<|error_end|>",
    "<|fix_start|>",
    "<|fix_end|>",
]


def main():
    from transformers import AutoTokenizer

    base_dir = PROJECT_ROOT / "data" / "tokenizer" / "base_tokenizer"
    save_dir = PROJECT_ROOT / "data" / "tokenizer" / "mindi_tokenizer"

    print(f"\n{'='*60}")
    print(f"  Step 4: Adding MINDI Special Tokens")
    print(f"{'='*60}")

    # Load base tokenizer
    print(f"\n  Loading base tokenizer from: {base_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(base_dir), trust_remote_code=True)
    original_vocab_size = len(tokenizer)
    print(f"  ✅ Base vocab size: {original_vocab_size:,}")

    # Add special tokens
    print(f"\n  Adding {len(MINDI_SPECIAL_TOKENS)} MINDI special tokens...")
    num_added = tokenizer.add_special_tokens({
        "additional_special_tokens": MINDI_SPECIAL_TOKENS
    })
    new_vocab_size = len(tokenizer)
    print(f"  ✅ Tokens added: {num_added}")
    print(f"  ✅ New vocab size: {new_vocab_size:,}")
    print(f"  ✅ Delta: +{new_vocab_size - original_vocab_size}")

    # Save updated tokenizer
    save_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(save_dir))
    print(f"\n  ✅ Saved MINDI tokenizer to: {save_dir}")

    # Show token ID mapping
    print(f"\n{'='*60}")
    print(f"  Special Token ID Mapping")
    print(f"{'='*60}")
    for token in MINDI_SPECIAL_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"    {token:<25} → ID {token_id}")

    # Verify round-trip for each special token
    print(f"\n{'='*60}")
    print(f"  Round-trip Verification")
    print(f"{'='*60}")
    all_pass = True
    for token in MINDI_SPECIAL_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        decoded = tokenizer.decode([token_id])
        match = decoded == token
        if not match:
            all_pass = False
        status = "✅" if match else "❌"
        print(f"    {status} {token} → {token_id} → \"{decoded}\"")

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Original vocab size:  {original_vocab_size:,}")
    print(f"  New vocab size:       {new_vocab_size:,}")
    print(f"  Special tokens added: {num_added}")
    if all_pass:
        print(f"  Round-trip test:      ✅ ALL {len(MINDI_SPECIAL_TOKENS)} PASSED")
    else:
        print(f"  Round-trip test:      ❌ SOME FAILED")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
