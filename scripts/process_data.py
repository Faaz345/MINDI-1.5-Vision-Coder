"""
MINDI 1.5 Vision-Coder — Day 2 Step 2: MINDI Format Converter

Converts ALL raw datasets (JSONL) into unified MINDI training format.

Each output example:
{
  "id": "mindi_000001",
  "type": "code_generation",
  "source": "websight",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user",   "content": "..."},
    {"role": "assistant", "content": "<|think_start|>...<|think_end|>..."}
  ],
  "metadata": {
    "language": "typescript",
    "framework": "nextjs",
    "has_vision": false,
    "tokens": 1024,
    "quality_score": 8.5
  }
}

Usage:
    python scripts/process_data.py                     # Process all
    python scripts/process_data.py --source codealpaca # Process one
    python scripts/process_data.py --dry-run           # Preview only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

# ── Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
LOGS_DIR = PROJECT_ROOT / "logs"
TOKENIZER_PATH = PROJECT_ROOT / "data" / "tokenizer" / "mindi_tokenizer"

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(console=console, rich_tracebacks=True, show_path=False),
        logging.FileHandler(LOGS_DIR / "process_data.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("mindi.process")

# ── System prompt ─────────────────────────────────────────────────────
MINDI_SYSTEM_PROMPT = (
    "You are MINDI 1.5 Vision-Coder, an AI built by MINDIGENOUS.AI. "
    "You are an expert in Next.js 14, React, TypeScript, Tailwind CSS, "
    "and UI/UX design. You see your own output and critique it to make "
    "it better for the user."
)

# ── Tokenizer (lazy loaded) ──────────────────────────────────────────
_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH), trust_remote_code=True)
        log.info(f"Loaded tokenizer (vocab={len(_tokenizer):,})")
    return _tokenizer


def count_tokens(text: str) -> int:
    tok = get_tokenizer()
    return len(tok.encode(text, add_special_tokens=False))


# ── Language detection ────────────────────────────────────────────────
def detect_language(code: str, filename: str = "") -> str:
    """Detect programming language from code content or filename."""
    ext_map = {
        ".py": "python", ".js": "javascript", ".jsx": "javascript",
        ".ts": "typescript", ".tsx": "typescript", ".html": "html",
        ".css": "css", ".json": "json", ".md": "markdown",
        ".rs": "rust", ".go": "go", ".java": "java", ".cpp": "cpp",
        ".c": "c", ".rb": "ruby", ".php": "php", ".swift": "swift",
        ".kt": "kotlin", ".sql": "sql", ".sh": "bash",
    }
    if filename:
        ext = Path(filename).suffix.lower()
        if ext in ext_map:
            return ext_map[ext]

    # Heuristic detection from content
    if "import React" in code or "from 'react'" in code or "jsx" in code.lower():
        return "typescript" if ": " in code and ("interface " in code or "type " in code) else "javascript"
    if "def " in code and "import " in code and ":" in code:
        return "python"
    if "func " in code and "package " in code:
        return "go"
    if "fn " in code and "let mut" in code:
        return "rust"
    if "public class" in code or "public static void" in code:
        return "java"
    if "<!DOCTYPE" in code or "<html" in code:
        return "html"
    if "function " in code or "const " in code or "=>" in code:
        return "javascript"
    return "unknown"


def detect_framework(code: str) -> str:
    """Detect framework from code content."""
    if "'use client'" in code or "next/" in code or "Next" in code:
        return "nextjs"
    if "import React" in code or "from 'react'" in code:
        return "react"
    if "express" in code.lower():
        return "express"
    if "from flask" in code or "Flask(" in code:
        return "flask"
    if "from django" in code:
        return "django"
    if "import vue" in code.lower() or "defineComponent" in code:
        return "vue"
    return "none"


# ── Quality scoring ──────────────────────────────────────────────────
def score_quality(code: str, language: str) -> float:
    """Score code quality on a 1-10 scale using heuristics."""
    score = 5.0

    # Length bonus (not too short, not just boilerplate)
    lines = code.strip().splitlines()
    if len(lines) >= 10:
        score += 0.5
    if len(lines) >= 30:
        score += 0.5
    if len(lines) < 3:
        score -= 2.0

    # Has comments/docstrings
    if "//" in code or "/*" in code or '"""' in code or "'''" in code or "#" in code:
        score += 0.5

    # Has type annotations (TypeScript/Python)
    if language in ("typescript", "python"):
        if ":" in code and ("interface " in code or "type " in code or "-> " in code):
            score += 0.5

    # Has proper imports
    if "import " in code or "from " in code or "require(" in code:
        score += 0.3

    # Has error handling
    if "try" in code or "catch" in code or "except" in code:
        score += 0.3

    # Has exports (module structure)
    if "export " in code or "module.exports" in code:
        score += 0.3

    # Penalize very short or empty
    if len(code.strip()) < 50:
        score -= 1.0

    # Penalize obvious low quality
    if code.count("TODO") > 3 or code.count("FIXME") > 3:
        score -= 0.5
    if "console.log" in code and code.count("console.log") > 5:
        score -= 0.3

    # Has proper function/class structure
    if "function " in code or "class " in code or "def " in code or "const " in code:
        score += 0.3

    # Tailwind/CSS usage
    if "className" in code or "tailwind" in code.lower():
        score += 0.3

    return max(1.0, min(10.0, round(score, 1)))


# ── Converter: wrap code in MINDI format ─────────────────────────────
def wrap_mindi_assistant(
    code: str,
    language: str = "typescript",
    filename: str = "",
    thinking: str = "",
    critique: str = "",
    suggestions: str = "",
) -> str:
    """Wrap code in MINDI special token format."""
    parts = []

    # Thinking block
    if thinking:
        parts.append(f"<|think_start|>\n{thinking}\n<|think_end|>")

    # File metadata
    if filename:
        framework = detect_framework(code)
        parts.append(f"<|file_start|>\npath: {filename}\nlanguage: {language}\nframework: {framework}\n<|file_end|>")

    # Code block
    parts.append(f"<|code_start|>\n{code.strip()}\n<|code_end|>")

    # Critique
    if critique:
        parts.append(f"<|critique_start|>\n{critique}\n<|critique_end|>")

    # Suggestions
    if suggestions:
        parts.append(f"<|suggest_start|>\n{suggestions}\n<|suggest_end|>")

    return "\n\n".join(parts)


def generate_thinking(user_request: str, language: str) -> str:
    """Generate a basic thinking block from the user request."""
    verbs = ["analyze", "implement", "create", "design", "build"]
    verb = random.choice(verbs)
    return (
        f"The user wants me to {verb} something. Let me break this down:\n"
        f"1. Understand the requirements from the request\n"
        f"2. Choose the right approach for {language}\n"
        f"3. Write clean, production-ready code\n"
        f"4. Review for best practices and accessibility"
    )


def generate_critique(language: str, code: str) -> str:
    """Generate a basic code critique."""
    items = [
        "✅ Code structure: Well-organized with clear separation of concerns",
        "✅ Naming: Descriptive variable and function names",
    ]
    if language in ("typescript", "javascript"):
        items.append("✅ Modern syntax: Uses ES6+ features appropriately")
    if "className" in code:
        items.append("✅ Styling: Tailwind CSS classes used correctly")
    items.append("⚠️ Consider adding error handling for edge cases")
    items.append("⚠️ Could benefit from unit tests")
    return "Code Review:\n" + "\n".join(f"- {item}" for item in items)


def generate_suggestions() -> str:
    """Generate improvement suggestions."""
    pool = [
        "Add comprehensive error handling with try/catch",
        "Implement loading and error states for better UX",
        "Add TypeScript strict mode compliance",
        "Write unit tests with Jest and Testing Library",
        "Add JSDoc comments for public API",
        "Consider extracting reusable hooks",
        "Add proper aria attributes for accessibility",
        "Implement responsive design breakpoints",
        "Add performance optimization with useMemo/useCallback",
        "Consider adding Storybook stories for documentation",
    ]
    selected = random.sample(pool, min(4, len(pool)))
    return "Suggested improvements:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(selected))


# ── Source-specific converters ────────────────────────────────────────

def convert_codealpaca(raw: dict, idx: int) -> Optional[dict]:
    """Convert CodeAlpaca example to MINDI format."""
    instruction = raw.get("instruction", "").strip()
    inp = raw.get("input", "").strip()
    output = raw.get("output", "").strip()

    if not instruction or not output:
        return None

    user_content = f"{instruction}\n{inp}".strip() if inp else instruction
    language = detect_language(output)
    quality = score_quality(output, language)

    assistant_content = wrap_mindi_assistant(
        code=output,
        language=language,
        thinking=generate_thinking(instruction, language),
        critique=generate_critique(language, output),
        suggestions=generate_suggestions(),
    )

    tokens = count_tokens(assistant_content)

    return {
        "id": f"mindi_{idx:06d}",
        "type": "code_generation",
        "source": "codealpaca",
        "messages": [
            {"role": "system", "content": MINDI_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "language": language,
            "framework": detect_framework(output),
            "has_vision": False,
            "tokens": tokens,
            "quality_score": quality,
        },
    }


def convert_codefeedback(raw: dict, idx: int) -> Optional[dict]:
    """Convert CodeFeedback example to MINDI format."""
    query = raw.get("query", "").strip()
    answer = raw.get("answer", "").strip()

    if not query or not answer:
        return None

    # Extract code blocks from answer if present
    code_blocks = re.findall(r"```[\w]*\n(.*?)```", answer, re.DOTALL)
    code = "\n\n".join(code_blocks) if code_blocks else answer

    language = detect_language(code)
    quality = score_quality(code, language)

    assistant_content = wrap_mindi_assistant(
        code=code,
        language=language,
        thinking=generate_thinking(query, language),
        critique=generate_critique(language, code),
        suggestions=generate_suggestions(),
    )

    tokens = count_tokens(assistant_content)

    return {
        "id": f"mindi_{idx:06d}",
        "type": "code_generation",
        "source": "codefeedback",
        "messages": [
            {"role": "system", "content": MINDI_SYSTEM_PROMPT},
            {"role": "user", "content": query},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "language": language,
            "framework": detect_framework(code),
            "has_vision": False,
            "tokens": tokens,
            "quality_score": quality,
        },
    }


def convert_starcoderdata(raw: dict, idx: int) -> Optional[dict]:
    """Convert StarCoder raw code to MINDI instruction format."""
    content = raw.get("content", "").strip()
    if not content or len(content) < 50:
        return None

    # Extract metadata
    max_lines = raw.get("max_line_length", 0)
    avg_line = raw.get("avg_line_length", 0)

    language = detect_language(content)
    quality = score_quality(content, language)

    # Create a synthetic user request from the code
    # Extract first comment or function/class name as context
    first_lines = content[:500]
    if "def " in first_lines:
        match = re.search(r"def (\w+)", first_lines)
        func_name = match.group(1) if match else "function"
        user_request = f"Write a {language} function called `{func_name}` with proper implementation"
    elif "class " in first_lines:
        match = re.search(r"class (\w+)", first_lines)
        class_name = match.group(1) if match else "Class"
        user_request = f"Create a {language} class called `{class_name}` with full implementation"
    elif "function " in first_lines or "const " in first_lines:
        match = re.search(r"(?:function|const)\s+(\w+)", first_lines)
        name = match.group(1) if match else "component"
        user_request = f"Implement `{name}` in {language} with clean, modern code"
    elif "export " in first_lines:
        match = re.search(r"export\s+(?:default\s+)?(?:function|class|const)\s+(\w+)", first_lines)
        name = match.group(1) if match else "module"
        user_request = f"Build an exported {language} module `{name}`"
    else:
        user_request = f"Write this {language} code with best practices"

    # Detect filename from content hints
    filename = ""
    if language == "python":
        filename = "main.py"
    elif language == "typescript":
        filename = "index.tsx"
    elif language == "javascript":
        filename = "index.js"

    assistant_content = wrap_mindi_assistant(
        code=content,
        language=language,
        filename=filename,
        thinking=generate_thinking(user_request, language),
        critique=generate_critique(language, content),
        suggestions=generate_suggestions(),
    )

    tokens = count_tokens(assistant_content)

    return {
        "id": f"mindi_{idx:06d}",
        "type": "code_generation",
        "source": "starcoderdata",
        "messages": [
            {"role": "system", "content": MINDI_SYSTEM_PROMPT},
            {"role": "user", "content": user_request},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "language": language,
            "framework": detect_framework(content),
            "has_vision": False,
            "tokens": tokens,
            "quality_score": quality,
        },
    }


def convert_websight(raw: dict, idx: int) -> Optional[dict]:
    """Convert WebSight HTML+screenshot to MINDI format."""
    html = raw.get("text", "").strip()
    if not html:
        return None

    # WebSight has HTML — we keep it as-is (conversion to JSX is a training objective)
    language = "html"
    quality = score_quality(html, language)
    has_image = "image" in raw or "screenshot" in raw

    user_request = "Convert this webpage design into a modern Next.js 14 component with Tailwind CSS"

    thinking = (
        "The user wants me to convert a web design to Next.js. I need to:\n"
        "1. Analyze the HTML structure and visual layout\n"
        "2. Convert HTML elements to React JSX syntax\n"
        "3. Replace CSS classes with Tailwind CSS utilities\n"
        "4. Add TypeScript types and proper component structure\n"
        "5. Ensure responsive design and accessibility"
    )

    assistant_content = wrap_mindi_assistant(
        code=html,
        language="typescript",
        filename="src/components/ConvertedPage.tsx",
        thinking=thinking,
        critique=generate_critique("typescript", html),
        suggestions=generate_suggestions(),
    )

    tokens = count_tokens(assistant_content)

    return {
        "id": f"mindi_{idx:06d}",
        "type": "vision_code",
        "source": "websight",
        "messages": [
            {"role": "system", "content": MINDI_SYSTEM_PROMPT},
            {"role": "user", "content": user_request},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "language": "typescript",
            "framework": "nextjs",
            "has_vision": has_image,
            "tokens": tokens,
            "quality_score": quality,
        },
    }


def convert_synthetic(raw: dict, idx: int) -> Optional[dict]:
    """Convert synthetic data (already in near-MINDI format) to final format."""
    user_content = raw.get("user", "").strip()
    assistant_content = raw.get("assistant", "").strip()
    source = raw.get("source", "synthetic")

    if not user_content or not assistant_content:
        return None

    tokens = count_tokens(assistant_content)
    language = raw.get("language", "typescript")

    return {
        "id": f"mindi_{idx:06d}",
        "type": "code_generation" if "search" not in source else "search",
        "source": source,
        "messages": [
            {"role": "system", "content": MINDI_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "language": language,
            "framework": raw.get("framework", "nextjs"),
            "has_vision": False,
            "tokens": tokens,
            "quality_score": score_quality(assistant_content, language),
        },
    }


def convert_evol_code(raw: dict, idx: int) -> Optional[dict]:
    """Convert EvolInstruct-Code example to MINDI format."""
    instruction = raw.get("instruction", "").strip()
    output = raw.get("output", "").strip()

    if not instruction or not output:
        return None

    code_blocks = re.findall(r"```[\w]*\n(.*?)```", output, re.DOTALL)
    code = "\n\n".join(code_blocks) if code_blocks else output

    language = detect_language(code)
    quality = score_quality(code, language)

    assistant_content = wrap_mindi_assistant(
        code=code,
        language=language,
        thinking=generate_thinking(instruction, language),
        critique=generate_critique(language, code),
        suggestions=generate_suggestions(),
    )

    tokens = count_tokens(assistant_content)

    return {
        "id": f"mindi_{idx:06d}",
        "type": "code_generation",
        "source": "evol_code",
        "messages": [
            {"role": "system", "content": MINDI_SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "language": language,
            "framework": detect_framework(code),
            "has_vision": False,
            "tokens": tokens,
            "quality_score": quality,
        },
    }


def convert_magicoder(raw: dict, idx: int) -> Optional[dict]:
    """Convert Magicoder example to MINDI format."""
    # Magicoder uses problem/solution or instruction/response
    instruction = (raw.get("instruction", "") or raw.get("problem", "")).strip()
    output = (raw.get("response", "") or raw.get("solution", "")).strip()

    if not instruction or not output:
        return None

    code_blocks = re.findall(r"```[\w]*\n(.*?)```", output, re.DOTALL)
    code = "\n\n".join(code_blocks) if code_blocks else output

    language = detect_language(code)
    quality = score_quality(code, language)

    assistant_content = wrap_mindi_assistant(
        code=code,
        language=language,
        thinking=generate_thinking(instruction, language),
        critique=generate_critique(language, code),
        suggestions=generate_suggestions(),
    )

    tokens = count_tokens(assistant_content)

    return {
        "id": f"mindi_{idx:06d}",
        "type": "code_generation",
        "source": "magicoder",
        "messages": [
            {"role": "system", "content": MINDI_SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "language": language,
            "framework": detect_framework(code),
            "has_vision": False,
            "tokens": tokens,
            "quality_score": quality,
        },
    }


# ── Source registry ───────────────────────────────────────────────────
SOURCE_CONVERTERS = {
    "codealpaca": ("codealpaca.jsonl", convert_codealpaca),
    "codefeedback": ("codefeedback.jsonl", convert_codefeedback),
    "starcoder_python": ("starcoder_python.jsonl", convert_starcoderdata),
    "starcoder_javascript": ("starcoder_javascript.jsonl", convert_starcoderdata),
    "starcoder_typescript": ("starcoder_typescript.jsonl", convert_starcoderdata),
    "starcoder_css": ("starcoder_css.jsonl", convert_starcoderdata),
    "starcoder_html": ("starcoder_html.jsonl", convert_starcoderdata),
    "evol_code": ("evol_code.jsonl", convert_evol_code),
    "magicoder": ("magicoder.jsonl", convert_magicoder),
    "websight": ("websight.jsonl", convert_websight),
    "synthetic_nextjs": ("synthetic_nextjs.jsonl", convert_synthetic),
    "search_examples": ("search_examples.jsonl", convert_synthetic),
    "sandbox_examples": ("sandbox_examples.jsonl", convert_synthetic),
}

OUTPUT_FILE = DATA_PROCESSED / "mindi_all.jsonl"


# ── Main processing pipeline ─────────────────────────────────────────
def process_source(
    source_name: str,
    global_idx: int,
    progress: Progress,
    dry_run: bool = False,
) -> tuple[int, int, int]:
    """Process one source, return (converted, skipped, global_idx)."""
    if source_name not in SOURCE_CONVERTERS:
        log.error(f"Unknown source: {source_name}")
        return 0, 0, global_idx

    filename, converter = SOURCE_CONVERTERS[source_name]
    input_path = DATA_RAW / filename

    if not input_path.exists():
        log.warning(f"⏭️  Skipping {source_name}: {input_path} not found (download first)")
        return 0, 0, global_idx

    # Count lines for progress
    total_lines = sum(1 for _ in open(input_path, encoding="utf-8"))
    task = progress.add_task(f"[cyan]{source_name}", total=total_lines)

    converted = 0
    skipped = 0
    output_handle = None

    if not dry_run:
        # Append mode so we can process sources incrementally
        output_handle = open(OUTPUT_FILE, "a", encoding="utf-8")

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    progress.update(task, advance=1)
                    continue

                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    progress.update(task, advance=1)
                    continue

                result = converter(raw, global_idx)

                if result is None:
                    skipped += 1
                else:
                    if not dry_run and output_handle:
                        output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    converted += 1
                    global_idx += 1

                progress.update(task, advance=1)

                # Flush periodically
                if not dry_run and output_handle and converted % 5000 == 0:
                    output_handle.flush()

    finally:
        if output_handle:
            output_handle.close()

    log.info(f"{'[DRY RUN] ' if dry_run else ''}✅ {source_name}: {converted:,} converted, {skipped:,} skipped")
    return converted, skipped, global_idx


def run_processing(
    source: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """Run the full processing pipeline."""
    console.print(Panel.fit(
        "[bold cyan]MINDI 1.5 Vision-Coder — MINDI Format Converter[/]\n"
        "[dim]Day 2 Step 2: Convert raw datasets to MINDI training format[/]",
        border_style="cyan",
    ))

    # Determine sources to process
    if source:
        sources = [source]
    else:
        sources = list(SOURCE_CONVERTERS.keys())

    # Show available files
    available_table = Table(title="📁 Raw Data Files")
    available_table.add_column("Source", style="cyan")
    available_table.add_column("File")
    available_table.add_column("Exists")
    available_table.add_column("Size")

    for src in sources:
        fname, _ = SOURCE_CONVERTERS[src]
        fpath = DATA_RAW / fname
        exists = fpath.exists()
        size = f"{fpath.stat().st_size / (1024*1024):.1f} MB" if exists else "—"
        available_table.add_row(src, fname, "✅" if exists else "❌", size)

    console.print(available_table)

    # Count existing examples in output file to resume from correct ID
    existing_count = 0
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            existing_count = sum(1 for _ in f)
        log.info(f"📄 Existing mindi_all.jsonl has {existing_count:,} examples — appending new data")

    # Process each source
    total_converted = 0
    total_skipped = 0
    global_idx = existing_count

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=2,
    ) as progress:
        for src in sources:
            converted, skipped, global_idx = process_source(
                src, global_idx, progress, dry_run=dry_run
            )
            total_converted += converted
            total_skipped += skipped

    # Summary
    console.print()
    summary = Table(title="📊 Processing Summary")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", justify="right", style="green")

    summary.add_row("Previously existing", f"{existing_count:,}")
    summary.add_row("Newly converted", f"{total_converted:,}")
    summary.add_row("Total skipped", f"{total_skipped:,}")
    grand_total = existing_count + total_converted
    summary.add_row("[bold]Grand total[/]", f"[bold]{grand_total:,}[/]")
    summary.add_row("Global ID range", f"mindi_000000 → mindi_{global_idx - 1:06d}")

    if not dry_run and OUTPUT_FILE.exists():
        size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
        summary.add_row("Output file", str(OUTPUT_FILE.relative_to(PROJECT_ROOT)))
        summary.add_row("Output size", f"{size_mb:.1f} MB")

    console.print(summary)

    if grand_total >= 500_000:
        console.print("\n[bold green]🎉 TARGET REACHED: 500K+ examples in MINDI format![/]")
    elif grand_total > 0:
        remaining = 500_000 - grand_total
        console.print(f"\n[yellow]⏳ {grand_total:,} total examples ({remaining:,} more needed for 500K target)[/]")
    else:
        console.print("\n[yellow]⚠️  No examples converted — download raw data first (scripts/download_datasets.py)[/]")


# ── CLI ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="MINDI Format Converter")
    parser.add_argument("--source", type=str, help="Process a specific source only")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing output")
    args = parser.parse_args()

    run_processing(source=args.source, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
