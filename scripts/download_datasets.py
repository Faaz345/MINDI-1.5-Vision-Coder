"""
MINDI 1.5 Vision-Coder — Day 2 Step 1: Dataset Download Pipeline

Downloads 7 datasets (500K+ examples total) with:
- Rich progress bars
- Network retry with exponential backoff
- Checkpoint/resume support
- Disk space estimation
- Logging to logs/download.log
- Running total of examples

Usage:
    python scripts/download_datasets.py                    # Download all
    python scripts/download_datasets.py --dataset websight # Download one
    python scripts/download_datasets.py --stage 1          # Stage 1 only (small/fast)
    python scripts/download_datasets.py --stage 2          # Stage 2 (starcoder)
    python scripts/download_datasets.py --stage 3          # Stage 3 (websight)
    python scripts/download_datasets.py --synthetic        # Synthetic only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass, field
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

# ── Project paths ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
LOGS_DIR = PROJECT_ROOT / "logs"
CHECKPOINT_FILE = DATA_RAW / ".download_checkpoint.json"

DATA_RAW.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────
console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(console=console, rich_tracebacks=True, show_path=False),
        logging.FileHandler(LOGS_DIR / "download.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("mindi.download")


# ── Checkpoint manager ────────────────────────────────────────────────
class CheckpointManager:
    """Tracks which datasets are complete so downloads can resume."""

    def __init__(self, path: Path = CHECKPOINT_FILE) -> None:
        self.path = path
        self.data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {"completed": {}, "in_progress": {}}

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def is_complete(self, name: str) -> bool:
        return name in self.data["completed"]

    def mark_complete(self, name: str, count: int, size_mb: float) -> None:
        self.data["completed"][name] = {
            "count": count,
            "size_mb": round(size_mb, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.data["in_progress"].pop(name, None)
        self.save()

    def mark_in_progress(self, name: str, count: int) -> None:
        self.data["in_progress"][name] = {"count": count}
        self.save()

    def get_resume_count(self, name: str) -> int:
        return self.data.get("in_progress", {}).get(name, {}).get("count", 0)

    def get_total_examples(self) -> int:
        return sum(v["count"] for v in self.data["completed"].values())


# ── Dataset definitions ───────────────────────────────────────────────
@dataclass
class DatasetConfig:
    name: str
    hf_name: str
    hf_subset: Optional[str]
    hf_split: str
    target_count: int
    output_file: str
    stage: int
    est_size_gb: float
    description: str
    languages: list[str] = field(default_factory=list)
    is_synthetic: bool = False


DATASETS: list[DatasetConfig] = [
    # Stage 1 — Small/fast (5-10 min)
    DatasetConfig(
        name="codealpaca",
        hf_name="sahil2801/CodeAlpaca-20k",
        hf_subset=None,
        hf_split="train",
        target_count=20_000,
        output_file="codealpaca.jsonl",
        stage=1,
        est_size_gb=0.05,
        description="Code instruction-following pairs",
    ),
    DatasetConfig(
        name="codefeedback",
        hf_name="m-a-p/CodeFeedback-Filtered-Instruction",
        hf_subset=None,
        hf_split="train",
        target_count=50_000,
        output_file="codefeedback.jsonl",
        stage=1,
        est_size_gb=0.3,
        description="Code with human feedback",
    ),
    # Stage 2 — Medium (1-2 hours)
    DatasetConfig(
        name="starcoder_python",
        hf_name="bigcode/starcoderdata",
        hf_subset="python",
        hf_split="train",
        target_count=100_000,
        output_file="starcoderdata.jsonl",
        stage=2,
        est_size_gb=2.0,
        description="StarCoder Python code",
        languages=["python"],
    ),
    DatasetConfig(
        name="starcoder_javascript",
        hf_name="bigcode/starcoderdata",
        hf_subset="javascript",
        hf_split="train",
        target_count=100_000,
        output_file="starcoderdata.jsonl",  # appends to same file
        stage=2,
        est_size_gb=2.0,
        description="StarCoder JavaScript code",
        languages=["javascript"],
    ),
    DatasetConfig(
        name="starcoder_typescript",
        hf_name="bigcode/starcoderdata",
        hf_subset="typescript",
        hf_split="train",
        target_count=50_000,
        output_file="starcoderdata.jsonl",  # appends to same file
        stage=2,
        est_size_gb=1.0,
        description="StarCoder TypeScript code",
        languages=["typescript"],
    ),
    # Stage 3 — Large (overnight)
    DatasetConfig(
        name="websight",
        hf_name="HuggingFaceM4/WebSight",
        hf_subset="v0.2",
        hf_split="train",
        target_count=200_000,
        output_file="websight.jsonl",
        stage=3,
        est_size_gb=8.0,
        description="Screenshots + HTML code pairs",
    ),
    # Synthetic — No download needed
    DatasetConfig(
        name="synthetic_nextjs",
        hf_name="",
        hf_subset=None,
        hf_split="",
        target_count=30_000,
        output_file="synthetic_nextjs.jsonl",
        stage=0,
        est_size_gb=0.2,
        description="Synthetic Next.js components with MINDI format",
        is_synthetic=True,
    ),
    DatasetConfig(
        name="search_examples",
        hf_name="",
        hf_subset=None,
        hf_split="",
        target_count=5_000,
        output_file="search_examples.jsonl",
        stage=0,
        est_size_gb=0.03,
        description="MINDI search usage examples",
        is_synthetic=True,
    ),
    DatasetConfig(
        name="sandbox_examples",
        hf_name="",
        hf_subset=None,
        hf_split="",
        target_count=3_000,
        output_file="sandbox_examples.jsonl",
        stage=0,
        est_size_gb=0.02,
        description="MINDI sandbox error-fix examples",
        is_synthetic=True,
    ),
]


# ── Retry helper ──────────────────────────────────────────────────────
def retry_with_backoff(fn, max_retries: int = 5, base_delay: float = 2.0):
    """Call fn() with exponential backoff on failure."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            log.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
            time.sleep(delay)


# ── HuggingFace download ─────────────────────────────────────────────
def download_hf_dataset(
    config: DatasetConfig,
    checkpoint: CheckpointManager,
    progress: Progress,
) -> int:
    """Download a HuggingFace dataset with streaming and save as JSONL."""
    from datasets import load_dataset

    output_path = DATA_RAW / config.output_file
    resume_count = checkpoint.get_resume_count(config.name)

    # For starcoder subsets that share an output file, use append mode
    # but only if this specific subset hasn't been completed
    is_append = config.output_file == "starcoderdata.jsonl" and output_path.exists()
    mode = "a" if is_append else "w"
    if not is_append and resume_count == 0:
        mode = "w"
    elif resume_count > 0:
        mode = "a"
        log.info(f"Resuming {config.name} from example {resume_count:,}")

    task = progress.add_task(
        f"[cyan]{config.name}",
        total=config.target_count,
        completed=resume_count,
    )

    log.info(f"Loading {config.hf_name} (subset={config.hf_subset}, split={config.hf_split}) streaming=True")

    def _load():
        kwargs = {
            "path": config.hf_name,
            "split": config.hf_split,
            "streaming": True,
            "trust_remote_code": True,
        }
        if config.hf_subset:
            kwargs["name"] = config.hf_subset
        return load_dataset(**kwargs)

    ds = retry_with_backoff(_load)

    count = 0
    skipped = 0
    with open(output_path, mode, encoding="utf-8") as f:
        for example in ds:
            if count < resume_count:
                count += 1
                continue

            # Write raw example as JSONL
            try:
                line = json.dumps(example, ensure_ascii=False, default=str)
                f.write(line + "\n")
            except (TypeError, ValueError) as e:
                skipped += 1
                continue

            count += 1
            progress.update(task, completed=count)

            # Periodic checkpoint every 5000 examples
            if count % 5000 == 0:
                checkpoint.mark_in_progress(config.name, count)
                f.flush()

            if count >= config.target_count:
                break

    size_mb = output_path.stat().st_size / (1024 * 1024)
    log.info(f"✅ {config.name}: {count:,} examples, {size_mb:.1f} MB (skipped {skipped})")
    progress.update(task, completed=count)
    return count


# ── Synthetic generators ──────────────────────────────────────────────

# Component templates for synthetic Next.js data
COMPONENT_TYPES = [
    "Navbar", "Hero", "Footer", "Sidebar", "Card", "Modal", "Dropdown",
    "Accordion", "Tabs", "Carousel", "Pagination", "Breadcrumb", "Alert",
    "Toast", "Badge", "Avatar", "Tooltip", "Popover", "Progress", "Spinner",
    "Skeleton", "Table", "Form", "Input", "Select", "Checkbox", "Radio",
    "Switch", "Slider", "DatePicker", "FileUpload", "SearchBar", "CommandPalette",
    "DataTable", "Chart", "Calendar", "Timeline", "Stepper", "Rating",
    "PricingCard", "TestimonialCard", "FeatureGrid", "StatsSection",
    "CTASection", "Newsletter", "LoginForm", "SignupForm", "ProfileCard",
    "DashboardLayout", "SettingsPanel", "NotificationList", "ChatBubble",
]

TAILWIND_COLORS = [
    "slate", "gray", "zinc", "neutral", "stone", "red", "orange", "amber",
    "yellow", "lime", "green", "emerald", "teal", "cyan", "sky", "blue",
    "indigo", "violet", "purple", "fuchsia", "pink", "rose",
]

DESIGN_PATTERNS = [
    "responsive grid layout", "flexbox centering", "gradient background",
    "glassmorphism effect", "dark mode support", "animated entrance",
    "hover transitions", "skeleton loading state", "error boundary",
    "lazy loading", "infinite scroll", "drag and drop", "keyboard navigation",
    "focus management", "scroll animations", "parallax effect",
]

USER_REQUESTS = [
    "Build me a {component} component with {pattern}",
    "Create a modern {component} using Tailwind CSS with {color} theme",
    "I need a {component} that supports dark mode and is fully accessible",
    "Design a {component} with smooth animations and {pattern}",
    "Make a responsive {component} component for a SaaS dashboard",
    "Build a {component} with TypeScript and proper prop types",
    "Create a reusable {component} with {pattern} for a landing page",
    "I want a {component} that looks like the latest {color} design trend",
    "Generate a production-ready {component} with {pattern}",
    "Build a {component} component with Framer Motion animations",
]

CRITIQUE_TEMPLATES = [
    "Visual Analysis:\n- ✅ Layout: Clean {pattern} implementation\n- ✅ Typography: Proper hierarchy with {color} accent colors\n- ⚠️ Accessibility: Consider adding aria-labels to interactive elements\n- ✅ Responsiveness: Works across breakpoints",
    "Design Review:\n- ✅ Color scheme: {color} palette creates good visual harmony\n- ✅ Spacing: Consistent padding and margins\n- ⚠️ Touch targets: Buttons should be at least 44px for mobile\n- ✅ Visual hierarchy: Clear flow from header to content",
    "UI/UX Assessment:\n- ✅ {pattern}: Well implemented with smooth transitions\n- ✅ Contrast: Text is readable against background\n- ⚠️ Loading state: Consider adding skeleton screens\n- ✅ Component structure: Clean separation of concerns",
]

SUGGEST_TEMPLATES = [
    "Improvements for next iteration:\n1. Add aria-label attributes for screen readers\n2. Implement keyboard navigation (Tab, Enter, Escape)\n3. Add loading skeleton state\n4. Consider adding subtle micro-interactions on hover",
    "Suggestions:\n1. Add error boundary wrapper for production safety\n2. Implement responsive breakpoints for sm/md/lg/xl\n3. Add unit tests with @testing-library/react\n4. Consider extracting reusable hooks for state logic",
    "Next steps:\n1. Add dark mode toggle using next-themes\n2. Optimize images with next/image component\n3. Add Storybook stories for documentation\n4. Implement proper TypeScript discriminated unions for variants",
]


def _generate_code_block(component: str, color: str) -> str:
    """Generate a realistic Next.js component code block."""
    props_name = f"{component}Props"
    variants = ["default", "primary", "secondary", "outline", "ghost"]
    variant = random.choice(variants)

    code = f"""'use client';

import {{ useState }} from 'react';
import {{ cn }} from '@/lib/utils';

interface {props_name} {{
  variant?: '{variant}' | 'default';
  className?: string;
  children?: React.ReactNode;
}}

export default function {component}({{ variant = 'default', className, children }}: {props_name}) {{
  const [isActive, setIsActive] = useState(false);

  return (
    <div
      className={{cn(
        'rounded-lg border p-4 transition-all duration-200',
        variant === '{variant}' && 'bg-{color}-50 border-{color}-200 text-{color}-900',
        variant === 'default' && 'bg-white border-gray-200 text-gray-900',
        isActive && 'ring-2 ring-{color}-500 shadow-lg',
        className
      )}}
      onClick={{() => setIsActive(!isActive)}}
      role="button"
      tabIndex={{0}}
      onKeyDown={{(e) => e.key === 'Enter' && setIsActive(!isActive)}}
    >
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">{component}</h3>
        <span className="text-sm text-{color}-600">{{variant}}</span>
      </div>
      <div className="mt-2 text-sm text-gray-600">
        {{children}}
      </div>
    </div>
  );
}}"""
    return code


def generate_synthetic_nextjs(count: int, progress: Progress) -> Generator[dict, None, None]:
    """Generate synthetic Next.js training examples in MINDI format."""
    task = progress.add_task("[magenta]synthetic_nextjs", total=count)

    for i in range(count):
        component = random.choice(COMPONENT_TYPES)
        color = random.choice(TAILWIND_COLORS)
        pattern = random.choice(DESIGN_PATTERNS)

        request_template = random.choice(USER_REQUESTS)
        user_request = request_template.format(
            component=component, color=color, pattern=pattern
        )

        code = _generate_code_block(component, color)
        filename = f"src/components/{component}.tsx"

        thinking = (
            f"The user wants a {component} component. I need to:\n"
            f"1. Create a TypeScript component with proper prop types\n"
            f"2. Use Tailwind CSS with {color} color scheme\n"
            f"3. Implement {pattern}\n"
            f"4. Ensure accessibility with ARIA attributes\n"
            f"5. Add keyboard navigation support"
        )

        critique = random.choice(CRITIQUE_TEMPLATES).format(
            pattern=pattern, color=color
        )
        suggestions = random.choice(SUGGEST_TEMPLATES)

        assistant_content = (
            f"<|think_start|>\n{thinking}\n<|think_end|>\n\n"
            f"<|file_start|>\npath: {filename}\nlanguage: typescript\nframework: next.js 14\n<|file_end|>\n\n"
            f"<|code_start|>\n{code}\n<|code_end|>\n\n"
            f"<|critique_start|>\n{critique}\n<|critique_end|>\n\n"
            f"<|suggest_start|>\n{suggestions}\n<|suggest_end|>"
        )

        yield {
            "id": f"synthetic_{i:06d}",
            "source": "synthetic_nextjs",
            "user": user_request,
            "assistant": assistant_content,
            "component": component,
            "language": "typescript",
            "framework": "nextjs",
        }

        progress.update(task, completed=i + 1)


def generate_search_examples(count: int, progress: Progress) -> Generator[dict, None, None]:
    """Generate synthetic search usage examples."""
    task = progress.add_task("[yellow]search_examples", total=count)

    search_scenarios = [
        ("How to implement dark mode in Next.js 14?", "next.js 14 dark mode implementation next-themes"),
        ("Best practices for React form validation", "react form validation zod react-hook-form 2025"),
        ("How to set up authentication in Next.js?", "next.js 14 authentication NextAuth.js credentials"),
        ("Tailwind CSS animation examples", "tailwind css animation keyframes framer-motion"),
        ("How to optimize images in Next.js?", "next.js image optimization next/image blur placeholder"),
        ("React server components best practices", "react server components RSC data fetching patterns"),
        ("How to deploy Next.js to Vercel?", "next.js 14 vercel deployment environment variables"),
        ("TypeScript utility types for React", "typescript react utility types ComponentProps PropsWithChildren"),
        ("How to use Zustand for state management?", "zustand state management react next.js middleware"),
        ("CSS Grid vs Flexbox for layouts", "css grid flexbox responsive layout patterns 2025"),
        ("How to implement infinite scroll?", "react infinite scroll intersection observer tanstack query"),
        ("Next.js API routes best practices", "next.js 14 route handlers API validation zod"),
        ("How to add SEO to Next.js?", "next.js 14 metadata SEO generateMetadata open graph"),
        ("React testing best practices", "react testing library jest vitest component testing"),
        ("How to use Prisma with Next.js?", "prisma next.js 14 database postgresql schema"),
    ]

    packages_db = [
        ("framer-motion", "Production-ready motion library for React", "npm i framer-motion"),
        ("next-themes", "Dark mode for Next.js apps", "npm i next-themes"),
        ("zustand", "Small, fast state management", "npm i zustand"),
        ("@tanstack/react-query", "Powerful data synchronization", "npm i @tanstack/react-query"),
        ("react-hook-form", "Performant forms with validation", "npm i react-hook-form"),
        ("zod", "TypeScript-first schema validation", "npm i zod"),
        ("tailwind-merge", "Merge Tailwind classes without conflicts", "npm i tailwind-merge"),
        ("clsx", "Tiny utility for constructing className strings", "npm i clsx"),
        ("lucide-react", "Beautiful SVG icons for React", "npm i lucide-react"),
        ("@radix-ui/react-dialog", "Accessible dialog component", "npm i @radix-ui/react-dialog"),
    ]

    for i in range(count):
        scenario = search_scenarios[i % len(search_scenarios)]
        pkg = packages_db[i % len(packages_db)]
        user_q = scenario[0]
        search_query = scenario[1]

        assistant_content = (
            f"<|think_start|>\nThe user is asking about {user_q.lower().rstrip('?')}. "
            f"Let me search for the latest best practices.\n<|think_end|>\n\n"
            f"<|search_start|>\nquery: \"{search_query}\"\n"
            f"results: [\n"
            f"  {{\"title\": \"Official Documentation\", \"url\": \"https://docs.example.com\", \"snippet\": \"Comprehensive guide...\"}},\n"
            f"  {{\"title\": \"Best Practices 2025\", \"url\": \"https://blog.example.com\", \"snippet\": \"Updated approach...\"}}\n"
            f"]\n<|search_end|>\n\n"
            f"Based on my research, here's the recommended approach:\n\n"
            f"First, install the required package:\n```bash\n{pkg[2]}\n```\n\n"
            f"**{pkg[0]}** — {pkg[1]}\n\n"
            f"<|code_start|>\n"
            f"// Example usage of {pkg[0]}\n"
            f"import {{ /* relevant imports */ }} from '{pkg[0]}';\n\n"
            f"export default function Example() {{\n"
            f"  // Implementation based on search results\n"
            f"  return <div>Example using {pkg[0]}</div>;\n"
            f"}}\n"
            f"<|code_end|>"
        )

        yield {
            "id": f"search_{i:06d}",
            "source": "search_examples",
            "user": user_q,
            "assistant": assistant_content,
            "search_query": search_query,
        }

        progress.update(task, completed=i + 1)


def generate_sandbox_examples(count: int, progress: Progress) -> Generator[dict, None, None]:
    """Generate synthetic sandbox error-fix examples."""
    task = progress.add_task("[red]sandbox_examples", total=count)

    error_scenarios = [
        {
            "error": "TypeError: Cannot read properties of undefined (reading 'map')",
            "cause": "Data array is undefined on initial render before API response",
            "fix": "Add optional chaining and fallback: data?.items?.map(...) ?? []",
            "file": "src/components/DataList.tsx",
        },
        {
            "error": "Error: Hydration failed because the initial UI does not match what was rendered on the server",
            "cause": "Using browser-only APIs (window, localStorage) during server render",
            "fix": "Wrap in useEffect or use dynamic import with ssr: false",
            "file": "src/components/ThemeProvider.tsx",
        },
        {
            "error": "Module not found: Can't resolve '@/components/ui/button'",
            "cause": "Path alias not configured in tsconfig.json",
            "fix": "Add paths mapping in tsconfig.json: '@/*': ['./src/*']",
            "file": "tsconfig.json",
        },
        {
            "error": "Warning: Each child in a list should have a unique 'key' prop",
            "cause": "Missing key prop in .map() iteration",
            "fix": "Add key={item.id} to the mapped JSX element",
            "file": "src/components/ItemList.tsx",
        },
        {
            "error": "TypeError: fetch failed - ECONNREFUSED",
            "cause": "API endpoint is unreachable or CORS is not configured",
            "fix": "Use Next.js API route as proxy, add CORS headers",
            "file": "src/app/api/proxy/route.ts",
        },
        {
            "error": "Error: Invalid hook call. Hooks can only be called inside of the body of a function component",
            "cause": "Calling useState inside a conditional or nested function",
            "fix": "Move hook call to the top level of the component function",
            "file": "src/hooks/useAuth.ts",
        },
        {
            "error": "Build error: Type 'string | undefined' is not assignable to type 'string'",
            "cause": "Environment variable might be undefined at build time",
            "fix": "Add non-null assertion or provide default value with ?? ''",
            "file": "src/lib/config.ts",
        },
        {
            "error": "Warning: validateDOMNesting(...): <div> cannot appear as a descendant of <p>",
            "cause": "Invalid HTML nesting - block element inside inline element",
            "fix": "Change outer <p> to <div> or inner <div> to <span>",
            "file": "src/components/Card.tsx",
        },
        {
            "error": "Error: NEXT_REDIRECT in API route",
            "cause": "Using redirect() in a try/catch block catches the redirect error",
            "fix": "Move redirect() call outside of try/catch, or re-throw NEXT_REDIRECT",
            "file": "src/app/api/auth/route.ts",
        },
        {
            "error": "Unhandled Runtime Error: Maximum update depth exceeded",
            "cause": "useEffect dependency causes infinite re-render loop",
            "fix": "Memoize the dependency with useMemo or useCallback",
            "file": "src/hooks/useData.ts",
        },
    ]

    for i in range(count):
        scenario = error_scenarios[i % len(error_scenarios)]

        assistant_content = (
            f"<|think_start|>\n"
            f"I see a build error. Let me analyze:\n"
            f"Error: {scenario['error']}\n"
            f"Root cause: {scenario['cause']}\n"
            f"I need to fix this in {scenario['file']}\n"
            f"<|think_end|>\n\n"
            f"<|error_start|>\n"
            f"File: {scenario['file']}\n"
            f"Error: {scenario['error']}\n"
            f"<|error_end|>\n\n"
            f"<|sandbox_start|>\n"
            f"Running: npm run build\n"
            f"Status: FAILED\n"
            f"Exit code: 1\n"
            f"<|sandbox_end|>\n\n"
            f"<|fix_start|>\n"
            f"Root cause: {scenario['cause']}\n"
            f"Solution: {scenario['fix']}\n"
            f"<|fix_end|>\n\n"
            f"<|file_start|>\npath: {scenario['file']}\nlanguage: typescript\n<|file_end|>\n\n"
            f"<|code_start|>\n"
            f"// Fixed version of {scenario['file']}\n"
            f"// Applied fix: {scenario['fix']}\n"
            f"export default function Fixed() {{\n"
            f"  // Corrected implementation\n"
            f"  return <div>Fixed component</div>;\n"
            f"}}\n"
            f"<|code_end|>\n\n"
            f"<|sandbox_start|>\n"
            f"Running: npm run build\n"
            f"Status: SUCCESS\n"
            f"Exit code: 0\n"
            f"<|sandbox_end|>"
        )

        yield {
            "id": f"sandbox_{i:06d}",
            "source": "sandbox_examples",
            "user": f"I'm getting this error: {scenario['error']}",
            "assistant": assistant_content,
            "error_type": scenario["error"][:50],
        }

        progress.update(task, completed=i + 1)


def write_synthetic(
    config: DatasetConfig,
    checkpoint: CheckpointManager,
    progress: Progress,
) -> int:
    """Generate and write synthetic data."""
    output_path = DATA_RAW / config.output_file

    generators = {
        "synthetic_nextjs": generate_synthetic_nextjs,
        "search_examples": generate_search_examples,
        "sandbox_examples": generate_sandbox_examples,
    }

    gen_fn = generators[config.name]
    count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for example in gen_fn(config.target_count, progress):
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1

    size_mb = output_path.stat().st_size / (1024 * 1024)
    log.info(f"✅ {config.name}: {count:,} examples, {size_mb:.1f} MB")
    return count


# ── Disk space check ──────────────────────────────────────────────────
def check_disk_space(datasets: list[DatasetConfig]) -> bool:
    """Verify enough disk space for planned downloads."""
    import shutil

    total_est_gb = sum(d.est_size_gb for d in datasets)
    usage = shutil.disk_usage(str(DATA_RAW))
    free_gb = usage.free / (1024 ** 3)

    table = Table(title="💾 Disk Space Estimate")
    table.add_column("Item", style="cyan")
    table.add_column("Size", justify="right", style="green")

    for d in datasets:
        table.add_row(d.name, f"{d.est_size_gb:.2f} GB")

    table.add_row("─" * 20, "─" * 10, style="dim")
    table.add_row("Total estimated", f"{total_est_gb:.2f} GB", style="bold")
    table.add_row("Available", f"{free_gb:.1f} GB", style="bold green")
    table.add_row(
        "After download",
        f"~{free_gb - total_est_gb:.1f} GB",
        style="bold yellow" if free_gb - total_est_gb > 50 else "bold red",
    )

    console.print(table)

    if total_est_gb > free_gb * 0.8:
        log.error(f"Not enough disk space! Need {total_est_gb:.1f} GB, have {free_gb:.1f} GB")
        return False

    return True


# ── Main pipeline ─────────────────────────────────────────────────────
def run_pipeline(
    stage: Optional[int] = None,
    dataset_name: Optional[str] = None,
    synthetic_only: bool = False,
) -> None:
    """Run the download pipeline."""
    console.print(Panel.fit(
        "[bold cyan]MINDI 1.5 Vision-Coder — Dataset Download Pipeline[/]\n"
        "[dim]Day 2 Step 1: Download 500K+ training examples[/]",
        border_style="cyan",
    ))

    checkpoint = CheckpointManager()

    # Filter datasets based on args
    if dataset_name:
        targets = [d for d in DATASETS if d.name == dataset_name]
        if not targets:
            log.error(f"Unknown dataset: {dataset_name}. Available: {[d.name for d in DATASETS]}")
            return
    elif synthetic_only:
        targets = [d for d in DATASETS if d.is_synthetic]
    elif stage is not None:
        targets = [d for d in DATASETS if d.stage == stage or (stage == 0 and d.is_synthetic)]
    else:
        targets = DATASETS

    # Show plan
    plan_table = Table(title="📋 Download Plan")
    plan_table.add_column("Dataset", style="cyan")
    plan_table.add_column("Examples", justify="right")
    plan_table.add_column("Est. Size", justify="right")
    plan_table.add_column("Stage")
    plan_table.add_column("Status")

    for d in targets:
        status = "✅ Done" if checkpoint.is_complete(d.name) else "⏳ Pending"
        stage_label = f"Stage {d.stage}" if d.stage > 0 else "Synthetic"
        plan_table.add_row(
            d.name,
            f"{d.target_count:,}",
            f"{d.est_size_gb:.2f} GB",
            stage_label,
            status,
        )

    console.print(plan_table)

    # Check disk space
    pending = [d for d in targets if not checkpoint.is_complete(d.name)]
    if not pending:
        console.print("\n[bold green]✅ All requested datasets already downloaded![/]")
        _print_summary(checkpoint)
        return

    if not check_disk_space(pending):
        return

    # Download with progress
    console.print()
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
        for config in pending:
            if checkpoint.is_complete(config.name):
                log.info(f"Skipping {config.name} (already complete)")
                continue

            log.info(f"\n{'─' * 50}")
            log.info(f"Starting: {config.name} — {config.description}")

            try:
                if config.is_synthetic:
                    count = write_synthetic(config, checkpoint, progress)
                else:
                    count = download_hf_dataset(config, checkpoint, progress)

                size_mb = (DATA_RAW / config.output_file).stat().st_size / (1024 * 1024)
                checkpoint.mark_complete(config.name, count, size_mb)

            except KeyboardInterrupt:
                log.warning(f"\n⚠️  Interrupted during {config.name}. Progress saved — rerun to resume.")
                return
            except Exception as e:
                log.error(f"❌ Failed {config.name}: {e}")
                log.error(traceback.format_exc())
                continue

    _print_summary(checkpoint)


def _print_summary(checkpoint: CheckpointManager) -> None:
    """Print final download summary."""
    console.print()
    summary = Table(title="📊 Download Summary")
    summary.add_column("Dataset", style="cyan")
    summary.add_column("Examples", justify="right")
    summary.add_column("Size", justify="right")
    summary.add_column("Time")

    total_count = 0
    total_mb = 0
    for name, info in checkpoint.data["completed"].items():
        summary.add_row(
            name,
            f"{info['count']:,}",
            f"{info['size_mb']:.1f} MB",
            info.get("timestamp", ""),
        )
        total_count += info["count"]
        total_mb += info["size_mb"]

    summary.add_row("─" * 20, "─" * 10, "─" * 10, "─" * 15, style="dim")
    summary.add_row(
        "[bold]TOTAL[/]",
        f"[bold]{total_count:,}[/]",
        f"[bold]{total_mb:.1f} MB[/]",
        "",
        style="bold green",
    )

    console.print(summary)

    if total_count >= 500_000:
        console.print("\n[bold green]🎉 TARGET REACHED: 500K+ examples downloaded![/]")
    else:
        remaining = 500_000 - total_count
        console.print(f"\n[yellow]⏳ {remaining:,} more examples needed to reach 500K target[/]")


# ── CLI ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="MINDI Dataset Download Pipeline")
    parser.add_argument("--dataset", type=str, help="Download a specific dataset by name")
    parser.add_argument("--stage", type=int, choices=[0, 1, 2, 3], help="Download a specific stage")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic data only")
    args = parser.parse_args()

    run_pipeline(
        stage=args.stage,
        dataset_name=args.dataset,
        synthetic_only=args.synthetic,
    )


if __name__ == "__main__":
    main()
