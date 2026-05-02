# MINDI 1.5 Vision-Coder — Complete Project Context

> **Last updated:** May 2, 2026 (Session 5)
> **Purpose:** This file contains ALL context needed to continue development with any AI assistant.
> It covers architecture decisions, errors encountered, fixes applied, training state, frontend state, and exact next steps.

---

## 1. PROJECT OVERVIEW

**MINDI 1.5 Vision-Coder** is a multimodal AI model that generates frontend code (HTML/CSS/JS, Next.js, Tailwind) from UI screenshots and text prompts. It combines:

- **Qwen/Qwen2.5-Coder-7B-Instruct** — 7.62B param base LLM (Apache 2.0)
- **CLIP ViT-L/14** — Frozen vision encoder for UI screenshot understanding
- **LoRA adapters** — Efficient fine-tuning (r=64, alpha=128)
- **Vision-Language Fusion** — Prepend visual tokens to text embeddings
- **22 MINDI Special Tokens** — Structured agentic reasoning (think, code, critique, fix, etc.)
- **3-Phase Training Strategy** — Progressive training on MI300X 192GB

**Repos:**
- **GitHub:** `https://github.com/Faaz345/MINDI-1.5-Vision-Coder.git` (branch: `master`)
- **HuggingFace Model:** `Mindigenous/MINDI-1.5-Vision-Coder` (private, push as `master:main`)
- **HuggingFace Dataset:** `Mindigenous/MINDI-1.5-training-data` (private)
- **HuggingFace Space:** `Mindigenous/mindi-chat` — live Gradio 5.x Space (ZeroGPU)
- **HF Token:** Set as `HF_TOKEN` environment variable (stored separately, not in repo)

---

## 2. TRAINING STATUS — COMPLETE ✅

All 3 phases of MINDI 1.5 Vision-Coder training are COMPLETE:

| Phase | Steps | Status | Platform |
|-------|-------|--------|----------|
| Phase 1 (LoRA) | 5,000 | ✅ Complete | DigitalOcean MI300X |
| Phase 2 (Vision Bridge) | 2,500 | ✅ Complete | DigitalOcean MI300X |
| Phase 3 (Joint) 0→1500 | 1,500 | ✅ Complete | DigitalOcean MI300X |
| Phase 3 (Joint) 1500→2500 | 1,000 | ✅ Complete | Modal A100-40GB |

**Final loss:** 0.25–0.40 range  
**VRAM:** 17.2 GB on A100-40GB  
**All checkpoints:** Uploaded to `checkpoints/` in HF model repo

### HuggingFace Checkpoints (Mindigenous/MINDI-1.5-Vision-Coder)
- Phase 1: 16 checkpoints (step250 → step5000)
- Phase 2: 10 checkpoints (step250 → step2500)
- Phase 3: `phase3_all_step500`, `step1000`, `step1500`, `step2000`, `phase3_all_step2500_final`, `phase3_final`

---

## 3. LIVE API — HuggingFace SPACE

**Space URL:** `https://mindigenous-mindi-chat.hf.space`  
**Space ID:** `Mindigenous/mindi-chat`  
**Framework:** Gradio 5.23.0 (ZeroGPU)  
**Protocol:** SSE v3 — two-step: POST to submit → GET to stream result

### API Call Pattern (Gradio 5.x SSE v3)

```javascript
// Step 1: Submit job
POST https://mindigenous-mindi-chat.hf.space/gradio_api/call/chat_fn
Headers: { "Content-Type": "application/json", "Authorization": "Bearer hf_..." }
Body: { "data": [prompt, imageArg, temperature, maxTokens, historyJson] }
Response: { "event_id": "..." }

// Step 2: Stream result
GET https://mindigenous-mindi-chat.hf.space/gradio_api/call/chat_fn/{event_id}
Parse SSE: find "event: complete" → next line "data: [...]"
Parse data[0] as JSON: { response: "...", sections: {} }
```

### ZeroGPU Quota
- **Anonymous users:** Very low quota (hits "GPU task aborted" error quickly)
- **Authenticated users (HF token):** ~8× higher quota
- **Quota errors throw as exceptions** with message containing "GPU task aborted" or "zerogpu"
- **Fix:** Always send `Authorization: Bearer <HF_TOKEN>` header

### Gradio Function Signature
```python
# hf_space/app.py — chat_fn
def chat_fn(prompt: str, image: dict|None, temperature: float, max_tokens: int, history_json: str) -> str:
    # Returns JSON string: {"response": "...", "sections": {...}}
```

---

## 4. FRONTEND — NEW VITE + REACT WEBSITE BUILDER ⭐ (Session 5 Work)

### What Was Built (May 2, 2026)

The old vanilla HTML/CSS/JS chat frontend was completely replaced with a **professional 3-panel website builder IDE** (similar to Bolt.new / v0.dev), built with Vite + React.

**The old frontend is backed up in:** `frontend/_legacy/`

### How to Run

```powershell
cd "d:\Desktop 31st Jan 2026\MINDI 1.5 vision-coder\frontend"
npm install          # only first time
npm run dev          # starts at http://localhost:5173
```

### New Frontend Structure

```
frontend/
├── index.html                    # Shell with Google Fonts (Inter + JetBrains Mono)
├── package.json                  # Vite 8.x + React 19 + prismjs + lucide-react
├── vite.config.js                # Vite config
├── _legacy/                      # Old vanilla JS chat frontend (backed up)
└── src/
    ├── main.jsx                  # React entry point
    ├── index.css                 # Design system (CSS tokens, reset, animations)
    ├── App.jsx                   # Main app — all state management + generation flow
    ├── App.css                   # All layout + component styles (3-panel IDE)
    ├── components/
    │   ├── Sidebar.jsx           # File tree + Agent Progress + status indicator
    │   ├── Editor.jsx            # Code editor with line-by-line animation + tabs
    │   ├── Preview.jsx           # Always-visible iframe preview + Console panel
    │   ├── PromptBar.jsx         # Bottom prompt input (auto-resize, send/stop)
    │   ├── PlanModal.jsx         # Clarifying questions (tech stack, design style)
    │   ├── SettingsModal.jsx     # API URL, HF token, temperature, max tokens
    │   └── Toasts.jsx            # Toast notifications
    └── services/
        ├── api.js                # Gradio SSE v3 integration + auth + demo fallback
        ├── promptEnhancer.js     # Analyzes prompt → asks questions → structured prompt
        └── fileParser.js         # Extracts files from model response markdown
```

### Layout

```
┌──────────────┬─────────────────────────┬──────────────────┐
│  SIDEBAR     │     CODE EDITOR          │   LIVE PREVIEW   │
│  (260px)     │  (flex: 1)               │   (420px)        │
│              │                          │                  │
│  MINDI 1.5   │  🌐 index.html           │  ● Preview       │
│  brand       │  1  <!DOCTYPE html>      │  [Rendered HTML] │
│              │  2  <html lang="en">     │                  │
│  FILES (1)   │  3  <head>               │                  │
│  🌐 index.   │  ...                     │  CONSOLE         │
│  html        │                          │  > Page rendered │
│              │                          │                  │
│  AGENT       │                          │                  │
│  PROGRESS    │                          │                  │
│  ✅ Enhancing│                          │                  │
│  ✅ Generating│                         │                  │
│  ✅ Complete  │                         │                  │
│              │                          │                  │
│  ● MINDI ·   │                          │                  │
│    Connected │                          │                  │
├──────────────┴─────────────────────────┴──────────────────┤
│  [Describe what you want to build...]              [Send]  │
│  MINDI 1.5 Vision-Coder               Shift+Enter new line │
└────────────────────────────────────────────────────────────┘
```

### Key Features

1. **Plan Modal** — When user submits prompt without specifying tech stack or theme, a "Configure Your Project" modal appears with:
   - Tech stack: HTML+CSS+JS / React / Next.js / Vue
   - Design style: Dark / Light / Gradient / Minimal
   - "Skip & Generate" and "Generate ⚡" buttons

2. **Prompt Enhancer** (`src/services/promptEnhancer.js`) — Transforms raw input into structured prompts with design requirements, responsiveness rules, font choices, no-placeholder rules.

3. **Code Animation** — Lines appear one by one at 15ms intervals with `line-appear` CSS animation as code generates.

4. **File Tree** — Files parsed from model response appear in sidebar with fade-in animation. Click to switch active file in editor.

5. **Live Preview** — Always-visible iframe on right renders HTML output. "Open in new tab" and "Copy HTML" buttons.

6. **Demo Fallback** — When API quota exceeded or any error occurs, pre-built demo responses for common prompts (landing page, dashboard) render automatically. No white screen.

7. **Settings** — Click the MINDI logo (top-left) to open Settings: configure API URL, HF Token, Temperature, Max Tokens.

### Error Handling in api.js

```javascript
// Two separate detection mechanisms:
isQuotaError(result)     // Response-level: checks result.response + result.sections.error
isQuotaException(errMsg) // Exception-level: checks thrown error message

// Both match: zerogpu | gpu quota | gpu task aborted | task aborted | unlogged user
```

When quota error detected → immediately falls back to `generateDemo(prompt)` which returns pre-built HTML.

### Demo Responses Available
- `/landing|hero|page|website/i` → Lumina landing page (Tailwind, gradient, features section)
- `/dashboard|chart|analytics|admin/i` → Pulsegrid dashboard (sidebar, stat cards, bar chart)
- Default → Simple MINDI hello card

### Settings Persistence
Saved in `localStorage` under key `mindi.builder.v1`:
```json
{
  "apiUrl": "https://mindigenous-mindi-chat.hf.space",
  "hfToken": "hf_...",
  "temperature": 0.7,
  "maxTokens": 2048
}
```

---

## 5. DIRECTORY STRUCTURE (Full Project)

```
MINDI-1.5-Vision-Coder/
├── src/                          # Model source code
│   ├── model/
│   │   ├── architecture.py       # Qwen2.5-Coder + LoRA wrapper (NOT nn.Module)
│   │   ├── mindi_model.py        # MINDI15 main class (nn.Module)
│   │   ├── vision_encoder.py     # CLIP ViT-L/14 (frozen) + trainable projection
│   │   ├── fusion_layer.py       # VisionLanguageFusion with text_gate
│   │   └── __init__.py
│   ├── training/
│   │   ├── mindi_trainer.py      # MINDITrainer: 3-phase loop, streaming data
│   │   ├── data_pipeline.py      # Data processing pipeline
│   │   └── __init__.py
│   └── ...
├── scripts/
│   ├── train.py                  # Master training launcher
│   ├── download_websight.py
│   ├── upload_websight_images.py
│   └── gpu_diagnostic.py
├── hf_space/
│   ├── app.py                    # Gradio Space — live at Mindigenous/mindi-chat
│   └── requirements.txt
├── frontend/                     # ⭐ NEW: Vite + React website builder
│   ├── index.html
│   ├── package.json
│   ├── _legacy/                  # Old vanilla JS chat (backup)
│   └── src/                      # (see Section 4 above)
├── api/                          # FastAPI endpoints (future)
├── modal_api.py                  # Modal.com A100 API server
├── modal_train.py                # Modal.com training script
├── data/                         # Local training data
├── configs/                      # Training configs
├── context.md                    # ← THIS FILE
└── ...
```

---

## 6. ARCHITECTURE DETAILS

### 6.1 Model Components

| Component | Class | File | Params | Trainable |
|-----------|-------|------|--------|-----------|
| Base LLM | `MINDIArchitecture` | `architecture.py` | 7.62B | No (frozen) |
| LoRA | via PEFT | `architecture.py` | 161.5M | Yes |
| CLIP Vision | `VisionEncoder` | `vision_encoder.py` | 304M | 4.2M (projection only) |
| Fusion | `VisionLanguageFusion` | `fusion_layer.py` | 16.8M | Yes |
| **Total** | `MINDI15` | `mindi_model.py` | **8.1B** | **182.5M (2.25%)** |

### 6.2 CRITICAL Architecture Notes

1. **`MINDIArchitecture` is NOT an `nn.Module`** — it's a plain Python wrapper. The actual trainable PeftModel is accessed via `self.architecture.get_model()` and registered as `self.llm` in `MINDI15.__init__()`.

2. **`self.llm = self.architecture.get_model()`** — Required so `model.parameters()` finds LoRA params.

3. **Fusion layer has `text_gate`** — Learnable scalar (init=0) for gradient flow during text-only batches.

### 6.3 MINDI Special Tokens (22 total, 11 pairs)

```
<|think_start|> / <|think_end|>         — Internal reasoning
<|code_start|> / <|code_end|>           — Generated code blocks
<|file_start|> / <|file_end|>           — File references
<|critique_start|> / <|critique_end|>   — Self-critique
<|suggest_start|> / <|suggest_end|>     — Suggestions
<|search_start|> / <|search_end|>       — Search context
<|error_start|> / <|error_end|>         — Error messages
<|fix_start|> / <|fix_end|>             — Fix attempts
<|vision_start|> / <|vision_end|>       — Vision input markers
<|sandbox_start|> / <|sandbox_end|>     — Sandbox execution
<|context_start|> / <|context_end|>     — Context block
```

---

## 7. HF SPACE — app.py KEY DETAILS

**File:** `hf_space/app.py`

### System Prompt (no identity hallucination fix)
The system prompt explicitly states: "You are MINDI 1.5 Vision-Coder, created by Mindigenous. You are NOT GPT-4, Claude, or any other AI..."

### chat_fn Signature
```python
@spaces.GPU(duration=60)
def chat_fn(prompt, image, temperature, max_tokens, history_json):
    # history_json is a JSON string of [{"role": ..., "content": ...}, ...]
    # Returns: JSON string {"response": "...", "sections": {...}}
```

### Gradio Interface
```python
gr.Interface(
    fn=chat_fn,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Image(type="filepath", label="Image"),
        gr.Slider(0, 2, value=0.7, label="Temperature"),
        gr.Slider(128, 4096, value=2048, label="Max Tokens"),
        gr.Textbox(label="History JSON", visible=False),
    ],
    outputs=gr.Textbox(label="Response"),
    api_name="chat_fn"
)
```

---

## 8. KNOWN ERRORS & FIXES HISTORY

### Training Errors (all fixed ✅)
| # | Error | Fix |
|---|-------|-----|
| 6.1 | GPU hang — HSA_OVERRIDE_GFX_VERSION | Do NOT set this var on ROCm 7.0 |
| 6.2 | No trainable params in optimizer | `self.llm = self.architecture.get_model()` |
| 6.3 | extra_special_tokens format error | Changed from list to dict in tokenizer_config.json |
| 6.4 | Phase 2 gradient flow crash | Added `text_gate` residual in VisionLanguageFusion |
| 6.5 | Git LFS push failures | `.gitattributes` + `git lfs migrate import` |
| 6.6 | HF auth for MI300X clone | Use token as both username+password in git URL |
| 6.7 | GPU hang after heavy I/O | PCI reset: `echo 1 > /sys/bus/pci/devices/0000:83:00.0/reset` |
| 6.8 | HF upload limits (10K/dir, 25K/commit) | Reorganized images into 6 subdirs |
| 6.9 | snapshot_download HTTP 429 | Use `git clone` instead |
| 6.10 | Bash history expansion `!'` | Use multi-line python or single-quoted strings |
| 6.11 | Data dir already exists on clone | `rm -rf data` before cloning dataset repo |

### Frontend API Errors (all fixed ✅)
| # | Error | Fix |
|---|-------|-----|
| 6.12 | `handleSend` ReferenceError in old app.js | `let activeSend = send` pattern (now in _legacy) |
| 6.13 | Gradio 3.x → 5.x API mismatch (404 on /api/predict) | Rewrote to SSE v3 two-step flow |
| 6.14 | Health check misdetects Space as offline | Use `fetch(base, {mode:'no-cors'})` for HF Spaces |
| 6.15 | GPU quota blocks demo — no fallback | `isQuotaError()` + `isQuotaException()` → auto demo |
| 6.16 | handlePlanSubmit catch had no demo fallback | Added demo fallback to all catch blocks in App.jsx |

---

## 9. SESSION HISTORY

| Session | Date | Key Work |
|---------|------|----------|
| 1 | April 15, 2026 | Phase 1 dry run. GPU hang resolved. |
| 2 | April 16, 2026 | Phase 1 training 0→4250. WebSight data uploaded. |
| 3 | April 19–28, 2026 | Phase 1→2→3 complete. Model deployed to HF Space. |
| 4 | April 30, 2026 | Fixed Gradio API protocol. HF token auth. ZeroGPU quota handling. Agent scaffolded. |
| 5 | May 2, 2026 | **Rebuilt frontend as Vite+React 3-panel IDE.** Prompt enhancer, plan modal, code animation, live preview, file tree, demo fallback. |

---

## 10. WHAT WORKS ✅

1. **Model training** — All 3 phases complete, checkpoints on HF
2. **HF Space** — Live at `Mindigenous/mindi-chat`, Gradio 5.x SSE v3
3. **New Frontend (Vite+React)** — `http://localhost:5173`
   - 3-panel IDE (Sidebar | Editor | Preview)
   - Plan Modal (tech stack + design style questions)
   - Prompt Enhancer (raw input → structured prompt)
   - Code animation (line-by-line fade-in)
   - File tree (real-time population during generation)
   - Live preview (always-visible iframe)
   - Demo fallback (landing page + dashboard demos)
   - Settings modal (API URL, HF token, temperature)
   - ZeroGPU quota detection + auto-fallback
4. **Build** — `npm run build` → 222KB JS (70KB gzip), 3.25s

---

## 11. WHAT REMAINS ❌

### High Priority
1. **Add HF token to Settings** — Without token, demo fallback always used. Real MINDI output requires `hf_...` token in Settings modal.
2. **Make suggestion pills clickable** — "Landing Page", "Dashboard" etc. chips on welcome screen should trigger generation when clicked.
3. **Syntax highlighting** — Add Prism.js token coloring to the code editor.

### Medium Priority
4. **Vision loop** — Feed preview screenshots back to MINDI for automated visual QA (captureScreenshot → base64 → callMINDI).
5. **Multi-file support** — Model generates single-file HTML currently. Add prompt instruction for `// filename:` markers to split into HTML/CSS/JS.
6. **Download project button** — Let user download generated files as a ZIP.

### Low Priority
7. **WebContainer SDK** — For projects that need Node.js execution (Next.js, npm packages).
8. **Fine-tuning for multi-file output** — Train on structured output format with `// filename:` markers.
9. **Deploy frontend** — Host on Vercel or GitHub Pages (free).

---

## 12. NEXT SESSION CHECKLIST

When starting a new AI assistant session:

1. **Read this file** first (most important)
2. **Run frontend:**
   ```powershell
   cd "d:\Desktop 31st Jan 2026\MINDI 1.5 vision-coder\frontend"
   npm run dev
   # Opens at http://localhost:5173
   ```
3. **Add HF token** in Settings (click MINDI logo → Settings → paste `hf_...` token)
4. **Test with real MINDI model** — type "landing page", skip plan modal, verify real response comes back
5. **Continue from "What Remains" section** above — start with suggestion chips or syntax highlighting

---

## 13. COMMANDS REFERENCE

### Frontend (Windows PowerShell)
```powershell
# Run dev server
cd "d:\Desktop 31st Jan 2026\MINDI 1.5 vision-coder\frontend"
npm run dev                    # http://localhost:5173

# Build for production
npm run build                  # dist/ folder

# Check build
npx vite build 2>&1 | Select-Object -Last 10
```

### Git
```powershell
git add -A
git commit -m "..."
git push origin master         # GitHub
git push hf master:main        # HuggingFace
```

### Local (Windows, PowerShell, in venv)
```powershell
& ".\venv\Scripts\Activate.ps1"
$env:HF_TOKEN="<your-hf-token>"
python scripts/download_websight.py --num_train 50000 --num_val 2500
python scripts/upload_websight_images.py
```

### MI300X (if spinning up again)
```bash
export HF_TOKEN=<your-hf-token>
export PYTORCH_ROCM_ARCH=gfx942
export TOKENIZERS_PARALLELISM=false
# DO NOT SET: HSA_OVERRIDE_GFX_VERSION

# GPU test
python3 -c "import torch; print('GPU:', torch.cuda.get_device_name(0)); x=torch.randn(100,device='cuda'); print('OK:', x.sum().item())"

# Full training
nohup python3 scripts/train.py --no_wandb > /workspace/training.log 2>&1 &
```

---

## 14. DESIGN SYSTEM (Frontend)

CSS variables defined in `src/index.css`:

```css
--bg-0: #07070c;          /* Deepest background */
--bg-1: #0a0a12;
--panel: #111120;          /* Sidebar, modals */
--border: rgba(255,255,255,.06);
--text: #ececf1;
--text-2: #b4b4c4;
--text-mute: #7a7a8c;
--purple: #7c3aed;
--purple-light: #a78bfa;
--blue: #2563eb;
--grad: linear-gradient(135deg, #7c3aed 0%, #2563eb 100%);
--sans: 'Inter', ...;
--mono: 'JetBrains Mono', ...;
--sidebar-w: 260px;
```

Key animations: `fadeIn`, `line-appear`, `float`, `pulse`, `spin`, `pop-in`, `toast-in`

---

## 15. MODEL QUALITY NOTES

MINDI 1.5 is a 7B model with ~10K training steps. Known characteristics:

| Issue | Status | Mitigation |
|-------|--------|-----------|
| Identity hallucination ("I am GPT-4") | ✅ Fixed via system prompt | Strong MINDI identity in `hf_space/app.py` |
| Basic/simple HTML output | ⚠️ Expected for 7B | Prompt enhancer adds design requirements |
| Weak image understanding | ⚠️ Only 2.5K vision steps | Prompt still works for text-only generation |
| No multi-file output | ⚠️ Not trained on it | Single complete file works fine |

**The prompt enhancer compensates for most quality issues** by structuring prompts with explicit design requirements (fonts, colors, responsiveness, no-placeholders rule, complete code requirement).

---

*Updated May 2, 2026 — Session 5: Rebuilt frontend as Vite+React 3-panel website builder IDE.*
*Previous sessions: April 15–30, 2026 — Model training (3 phases), HF Space deployment, API fixes.*
