"""
MINDI 1.5 Vision-Coder — Step 3: Download Tokenizer & Test

Downloads ONLY the tokenizer (not model weights) from Qwen/Qwen2.5-Coder-7B-Instruct,
saves it locally, and runs encoding/decoding tests on 8 code strings.
"""

import os
import sys
from pathlib import Path

# Ensure project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


def main():
    from transformers import AutoTokenizer

    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
    save_dir = PROJECT_ROOT / "data" / "tokenizer" / "base_tokenizer"
    hf_token = os.environ.get("HUGGINGFACE_TOKEN", "")

    # ── Download tokenizer ──
    print(f"\n{'='*60}")
    print(f"  Downloading tokenizer: {model_name}")
    print(f"  Save to: {save_dir}")
    print(f"{'='*60}\n")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token if hf_token else None,
        trust_remote_code=True,
    )

    # Save locally
    save_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(save_dir))
    print(f"  ✅ Tokenizer saved to {save_dir}")
    print(f"  ✅ Vocab size: {tokenizer.vocab_size:,}")
    print(f"  ✅ Model max length: {tokenizer.model_max_length:,}")

    # ── List saved files ──
    print(f"\n  Saved files:")
    for f in sorted(save_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name} ({size_kb:.1f} KB)")

    # ── Run tokenizer tests ──
    test_strings = [
        "Build me a Next.js dashboard",
        "import React from 'react'",
        "className='flex items-center gap-4'",
        "'use client'",
        "const [state, setState] = useState(null)",
        "export default function Page() {",
        "npm install framer-motion",
        "async function getData() {",
    ]

    print(f"\n{'='*60}")
    print(f"  Tokenizer Tests — 8 Code Strings")
    print(f"{'='*60}")

    all_pass = True
    for i, text in enumerate(test_strings, 1):
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids)
        match = decoded == text
        if not match:
            all_pass = False

        print(f"\n  Test {i}: \"{text}\"")
        print(f"    Token count: {len(ids)}")
        print(f"    Token IDs:   {ids}")
        print(f"    Decoded:     \"{decoded}\"")
        print(f"    Match:       {'✅ PERFECT' if match else '❌ MISMATCH'}")

    print(f"\n{'='*60}")
    if all_pass:
        print(f"  ✅ ALL 8 TESTS PASSED — Perfect reconstruction!")
    else:
        print(f"  ⚠️  Some tests had reconstruction differences (whitespace normalization is normal)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
