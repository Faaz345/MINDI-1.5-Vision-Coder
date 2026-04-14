"""
MINDI 1.5 Vision-Coder — Step 6: Smoke-test MindiTokenizer wrapper & generate test report.
"""

import sys
import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tokenizer.tokenizer import MindiTokenizer, MINDI_SPECIAL_TOKENS

print("=" * 70)
print("STEP 6: SAVE EVERYTHING — WRAPPER SMOKE TEST + REPORT")
print("=" * 70)

# ── 1. Load via wrapper class ────────────────────────────────────────
print("\n1️⃣  Loading MindiTokenizer wrapper...")
tok = MindiTokenizer()
print(f"   ✅ Loaded from: {tok.tokenizer_path}")
print(f"   Vocab size: {tok.get_vocab_size():,}")

# ── 2. Test encode / decode ──────────────────────────────────────────
print("\n2️⃣  encode() / decode()...")
text = "export default function Hero() { return <h1>Hello</h1>; }"
ids = tok.encode(text)
decoded = tok.decode(ids)
assert decoded.strip() == text.strip(), f"Round-trip failed: {decoded!r}"
print(f"   ✅ Round-trip OK — {len(ids)} tokens")

# ── 3. Test encode_with_special_tokens ───────────────────────────────
print("\n3️⃣  encode_with_special_tokens()...")
special_text = "<|code_start|>\nconsole.log('hi');\n<|code_end|>"
ids2 = tok.encode_with_special_tokens(special_text)
decoded2 = tok.decode(ids2)
assert decoded2.strip() == special_text.strip(), f"Special round-trip failed"
code_start_id = tok.get_special_token_id("code_start")
code_end_id = tok.get_special_token_id("code_end")
assert code_start_id in ids2, "code_start token not found"
assert code_end_id in ids2, "code_end token not found"
print(f"   ✅ Special tokens preserved — {len(ids2)} tokens")

# ── 4. Test encode_conversation ──────────────────────────────────────
print("\n4️⃣  encode_conversation()...")
messages = [
    {"role": "system", "content": "You are MINDI 1.5 Vision-Coder."},
    {"role": "user", "content": "Build a navbar."},
    {"role": "assistant", "content": "<|think_start|>\nPlanning navbar...\n<|think_end|>\n\n<|code_start|>\nexport default function Navbar() { return <nav>Nav</nav>; }\n<|code_end|>"},
]
conv_ids = tok.encode_conversation(messages, wrap_mindi=True)
conv_decoded = tok.decode(conv_ids)
assert "<|mindi_start|>" in conv_decoded, "mindi_start missing"
assert "<|mindi_end|>" in conv_decoded, "mindi_end missing"
assert "<|im_start|>" in conv_decoded, "im_start missing"
assert "<|think_start|>" in conv_decoded, "think_start missing"
assert "<|code_start|>" in conv_decoded, "code_start missing"
print(f"   ✅ Conversation encoded — {len(conv_ids)} tokens, mindi/im/think/code all present")

# ── 5. Test get_special_token_ids ────────────────────────────────────
print("\n5️⃣  get_special_token_ids()...")
all_ids = tok.get_special_token_ids()
assert len(all_ids) == 22, f"Expected 22, got {len(all_ids)}"
for name, tid in all_ids.items():
    assert isinstance(tid, int) and tid > 0, f"Bad ID for {name}: {tid}"
print(f"   ✅ 22 special token IDs returned, all valid integers")

# ── 6. Test get_vocab_size ───────────────────────────────────────────
print("\n6️⃣  get_vocab_size()...")
vs = tok.get_vocab_size()
assert vs == 151685, f"Expected 151685, got {vs}"
print(f"   ✅ Vocab size: {vs:,}")

# ── Generate test report ─────────────────────────────────────────────
print("\n" + "─" * 70)
print("📄 Generating test report...")

report_lines = [
    "=" * 70,
    "MINDI 1.5 VISION-CODER — TOKENIZER TEST REPORT",
    f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "=" * 70,
    "",
    "BASE MODEL: Qwen/Qwen2.5-Coder-7B-Instruct",
    f"VOCAB SIZE: {vs:,}",
    f"SPECIAL TOKENS: {len(all_ids)} (22 MINDI tokens)",
    f"TOKENIZER PATH: data/tokenizer/mindi_tokenizer/",
    "",
    "─" * 70,
    "SPECIAL TOKEN REGISTRY",
    "─" * 70,
]

for name, tid in sorted(all_ids.items(), key=lambda x: x[1]):
    token_str = MINDI_SPECIAL_TOKENS[name]
    report_lines.append(f"  {token_str:<25} → ID {tid}")

report_lines += [
    "",
    "─" * 70,
    "WRAPPER CLASS API TESTS",
    "─" * 70,
    "  ✅ encode()                    — round-trip plain text",
    "  ✅ decode()                    — reconstructs original text",
    "  ✅ encode_with_special_tokens() — preserves special tokens as single IDs",
    "  ✅ encode_conversation()        — formats system/user/assistant with im_start/end + mindi wrapper",
    "  ✅ get_vocab_size()            — returns 151,685",
    "  ✅ get_special_token_ids()     — returns all 22 MINDI token IDs",
    "  ✅ get_special_token_id(name)  — individual token lookup",
    "",
    "─" * 70,
    "CONVERSATION FORMAT TEST (from Step 5)",
    "─" * 70,
    "  Total tokens:       971",
    "  Round-trip:         PERFECT MATCH",
    "  Special tokens:     22/22 preserved as single tokens",
    "  Qwen chat tokens:   im_start ×3, im_end ×3",
    "  Context usage:      971 / 32,768 = 3.0%",
    "",
    "─" * 70,
    "FILES SAVED",
    "─" * 70,
    "  data/tokenizer/base_tokenizer/     — Original Qwen tokenizer (3 files)",
    "  data/tokenizer/mindi_tokenizer/    — MINDI tokenizer with 22 special tokens",
    "  src/tokenizer/tokenizer.py         — MindiTokenizer wrapper class",
    "  logs/tokenizer_test.txt            — This report",
    "  scripts/download_tokenizer.py      — Tokenizer download script",
    "  scripts/add_special_tokens.py      — Special token addition script",
    "  scripts/test_mindi_format.py       — Conversation format test script",
    "",
    "=" * 70,
    "STATUS: ALL TESTS PASSED ✅",
    "=" * 70,
]

report_text = "\n".join(report_lines)

logs_dir = PROJECT_ROOT / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
report_path = logs_dir / "tokenizer_test.txt"
report_path.write_text(report_text, encoding="utf-8")
print(f"   ✅ Saved to: {report_path}")

# ── Final summary ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("✅ STEP 6 COMPLETE: Everything saved!")
print("   • MindiTokenizer wrapper class — 6/6 API methods tested")
print("   • Test report — logs/tokenizer_test.txt")
print(f"   • Tokenizer files — data/tokenizer/mindi_tokenizer/")
print("=" * 70)
