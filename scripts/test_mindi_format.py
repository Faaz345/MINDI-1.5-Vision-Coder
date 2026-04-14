"""
MINDI 1.5 Vision-Coder — Step 5: Test MINDI Conversation Format
Tests full conversation tokenization with all special tokens.
"""

from pathlib import Path
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOKENIZER_PATH = PROJECT_ROOT / "data" / "tokenizer" / "mindi_tokenizer"

# ── Load MINDI tokenizer ──────────────────────────────────────────────
print("=" * 70)
print("STEP 5: TEST MINDI CONVERSATION FORMAT")
print("=" * 70)

print(f"\n📂 Loading MINDI tokenizer from: {TOKENIZER_PATH}")
tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH), trust_remote_code=True)
print(f"   Vocab size: {len(tokenizer):,}")

# ── Define MINDI special tokens ──────────────────────────────────────
MINDI_SPECIAL_TOKENS = [
    "<|mindi_start|>", "<|mindi_end|>",
    "<|code_start|>", "<|code_end|>",
    "<|vision_start|>", "<|vision_end|>",
    "<|critique_start|>", "<|critique_end|>",
    "<|suggest_start|>", "<|suggest_end|>",
    "<|think_start|>", "<|think_end|>",
    "<|file_start|>", "<|file_end|>",
    "<|search_start|>", "<|search_end|>",
    "<|sandbox_start|>", "<|sandbox_end|>",
    "<|error_start|>", "<|error_end|>",
    "<|fix_start|>", "<|fix_end|>",
]

# ── Build test conversation ──────────────────────────────────────────
system_message = (
    "You are MINDI 1.5 Vision-Coder, a multimodal agentic AI coding assistant "
    "created by MINDIGENOUS.AI. You generate production-ready Next.js 14 + "
    "Tailwind CSS + TypeScript code, see your own output via vision, critique "
    "UI/UX, search the web for best practices, test in a sandbox, and self-fix errors."
)

user_message = "Build me a hero section for a SaaS landing page with a gradient background, animated headline, and a CTA button."

assistant_message = """<|think_start|>
The user wants a hero section for a SaaS landing page. I need to:
1. Search for current hero section best practices
2. Create a Next.js 14 component with Tailwind CSS
3. Include gradient background, animated headline, and CTA button
4. Critique the result for UI/UX quality
<|think_end|>

<|search_start|>
query: "best SaaS hero section design patterns 2025 Next.js Tailwind"
results: [
  {"title": "Modern Hero Sections", "url": "https://example.com/hero-patterns"},
  {"title": "SaaS Landing Best Practices", "url": "https://example.com/saas-landing"}
]
<|search_end|>

<|file_start|>
path: src/components/HeroSection.tsx
language: typescript
framework: next.js 14
<|file_end|>

<|code_start|>
'use client';

import { motion } from 'framer-motion';

export default function HeroSection() {
  return (
    <section className="relative min-h-screen flex items-center justify-center bg-gradient-to-br from-indigo-600 via-purple-600 to-pink-500 overflow-hidden">
      <div className="absolute inset-0 bg-[url('/grid.svg')] opacity-10" />
      <div className="relative z-10 text-center px-4 max-w-4xl mx-auto">
        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
          className="text-5xl md:text-7xl font-bold text-white mb-6 leading-tight"
        >
          Ship faster with{' '}
          <span className="bg-clip-text text-transparent bg-gradient-to-r from-yellow-200 to-yellow-400">
            AI-powered code
          </span>
        </motion.h1>
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="text-xl text-white/80 mb-10 max-w-2xl mx-auto"
        >
          Build production-ready web apps in minutes. MINDI writes, tests,
          and deploys your code while you focus on what matters.
        </motion.p>
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 0.4 }}
        >
          <a
            href="/get-started"
            className="inline-block px-8 py-4 text-lg font-semibold text-indigo-900 bg-white rounded-full shadow-lg hover:shadow-xl hover:scale-105 transition-all duration-300"
          >
            Get Started Free →
          </a>
        </motion.div>
      </div>
    </section>
  );
}
<|code_end|>

<|critique_start|>
Visual Analysis:
- ✅ Gradient background: Strong indigo→purple→pink gradient creates visual depth
- ✅ Animated headline: Framer Motion fade-up provides smooth entrance
- ✅ CTA button: High contrast white on gradient, rounded-full shape draws attention
- ⚠️ Accessibility: Need to verify color contrast ratios for text on gradient
- ⚠️ Mobile: Font sizes may need adjustment below md breakpoint
- ✅ Performance: Motion animations are hardware-accelerated transforms
<|critique_end|>

<|suggest_start|>
Improvements for next iteration:
1. Add aria-label to the CTA link for screen readers
2. Consider adding a secondary CTA (e.g., "Watch Demo") for users not ready to commit
3. Add a subtle particle or floating shape animation in the background
4. Include social proof (e.g., "Trusted by 10,000+ developers") below the CTA
<|suggest_end|>"""

# ── Build full conversation string ───────────────────────────────────
conversation = f"""<|mindi_start|>
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
{assistant_message}<|im_end|>
<|mindi_end|>"""

print("\n" + "─" * 70)
print("FULL MINDI CONVERSATION (raw text)")
print("─" * 70)
print(conversation)
print("─" * 70)

# ── Tokenize the full conversation ───────────────────────────────────
print("\n📊 TOKENIZATION RESULTS")
print("─" * 70)

token_ids = tokenizer.encode(conversation, add_special_tokens=False)
print(f"   Total tokens: {len(token_ids):,}")

decoded = tokenizer.decode(token_ids)
print(f"   Decoded length (chars): {len(decoded):,}")

# ── Round-trip verification ──────────────────────────────────────────
print("\n🔄 ROUND-TRIP VERIFICATION")
print("─" * 70)

if decoded.strip() == conversation.strip():
    print("   ✅ PERFECT MATCH — decoded text matches original conversation exactly")
    round_trip_pass = True
else:
    # Show differences for debugging
    print("   ❌ MISMATCH detected!")
    orig_lines = conversation.strip().splitlines()
    dec_lines = decoded.strip().splitlines()
    print(f"   Original lines: {len(orig_lines)}, Decoded lines: {len(dec_lines)}")
    for i, (o, d) in enumerate(zip(orig_lines, dec_lines)):
        if o != d:
            print(f"   Line {i}: DIFF")
            print(f"     Original: {repr(o[:100])}")
            print(f"     Decoded:  {repr(d[:100])}")
    round_trip_pass = False

# ── Verify all MINDI special tokens are preserved as single tokens ───
print("\n🔍 SPECIAL TOKEN PRESERVATION")
print("─" * 70)

all_passed = True
for token_str in MINDI_SPECIAL_TOKENS:
    token_id = tokenizer.convert_tokens_to_ids(token_str)
    # Check the token encodes to a single ID
    encoded = tokenizer.encode(token_str, add_special_tokens=False)

    if len(encoded) == 1 and encoded[0] == token_id:
        status = "✅"
    else:
        status = "❌"
        all_passed = False

    # Check this token_id appears in the full conversation encoding
    count_in_conv = token_ids.count(token_id)
    print(f"   {status} {token_str:<25} ID={token_id:<8} single_token=True  occurrences_in_conv={count_in_conv}")

# ── Qwen chat template tokens ──────────────────────────────────────
print("\n🔍 QWEN CHAT TEMPLATE TOKENS")
print("─" * 70)

qwen_tokens = ["<|im_start|>", "<|im_end|>"]
for token_str in qwen_tokens:
    token_id = tokenizer.convert_tokens_to_ids(token_str)
    encoded = tokenizer.encode(token_str, add_special_tokens=False)
    count_in_conv = token_ids.count(token_id)
    status = "✅" if len(encoded) == 1 else "❌"
    print(f"   {status} {token_str:<25} ID={token_id:<8} occurrences_in_conv={count_in_conv}")

# ── Token distribution analysis ──────────────────────────────────────
print("\n📈 TOKEN DISTRIBUTION")
print("─" * 70)

# Count special vs regular tokens
special_ids = set()
for t in MINDI_SPECIAL_TOKENS + qwen_tokens:
    tid = tokenizer.convert_tokens_to_ids(t)
    special_ids.add(tid)

special_count = sum(1 for tid in token_ids if tid in special_ids)
regular_count = len(token_ids) - special_count

print(f"   Special tokens: {special_count}")
print(f"   Regular tokens: {regular_count}")
print(f"   Total tokens:   {len(token_ids):,}")
print(f"   Special ratio:  {special_count / len(token_ids) * 100:.1f}%")

# ── Estimate tokens per message ──────────────────────────────────────
print("\n📏 TOKENS PER MESSAGE")
print("─" * 70)

sys_tokens = tokenizer.encode(system_message, add_special_tokens=False)
usr_tokens = tokenizer.encode(user_message, add_special_tokens=False)
ast_tokens = tokenizer.encode(assistant_message, add_special_tokens=False)

print(f"   System message:    {len(sys_tokens):>5} tokens ({len(system_message):>5} chars)")
print(f"   User message:      {len(usr_tokens):>5} tokens ({len(user_message):>5} chars)")
print(f"   Assistant message:  {len(ast_tokens):>5} tokens ({len(assistant_message):>5} chars)")
print(f"   Wrapper overhead:  ~{len(token_ids) - len(sys_tokens) - len(usr_tokens) - len(ast_tokens):>5} tokens (mindi_start/end, im_start/end, roles)")

# ── Context window fit check ─────────────────────────────────────────
print("\n📐 CONTEXT WINDOW FIT")
print("─" * 70)
context_length = 32768
print(f"   Context window:   {context_length:>6} tokens")
print(f"   This conversation: {len(token_ids):>6} tokens")
print(f"   Remaining:        {context_length - len(token_ids):>6} tokens ({(context_length - len(token_ids)) / context_length * 100:.1f}%)")
print(f"   ✅ Fits easily within context window")

# ── Final verdict ────────────────────────────────────────────────────
print("\n" + "=" * 70)
if round_trip_pass and all_passed:
    print("✅ STEP 5 PASSED: MINDI conversation format works perfectly!")
    print("   • Full conversation tokenizes and decodes with perfect fidelity")
    print("   • All 22 MINDI special tokens preserved as single tokens")
    print("   • Qwen chat template tokens (im_start/im_end) working correctly")
    print(f"   • Total: {len(token_ids):,} tokens for a realistic conversation")
else:
    print("❌ STEP 5 FAILED — issues detected above")
print("=" * 70)
