/* =============================================================
   MINDI 1.5 — Vision-Coder · Frontend logic
   ============================================================= */

(() => {
  'use strict';

  // ----------------------------------------------------------------
  // Constants
  // ----------------------------------------------------------------
  const API_DEFAULT = 'https://mindigenous-mindi-chat.hf.space';
  const STORAGE_KEY = 'mindi.v1.state';
  const MAX_TEXTAREA = 200;
  const COLD_START_HINT_MS = 10_000; // show cold-start hint after 10s

  const SECTION_ORDER = ['thinking', 'code', 'critique', 'fix', 'error', 'suggest', 'file'];
  const SECTION_LABELS = {
    thinking: 'Thinking',
    code:     'Code',
    critique: 'Critique',
    fix:      'Fix',
    error:    'Error',
    suggest:  'Suggestion',
    file:     'File',
  };
  // mapping from raw token name → sections key
  const TOKEN_TO_KEY = {
    think: 'thinking',
    code:  'code',
    critique: 'critique',
    fix:   'fix',
    error: 'error',
    suggest: 'suggest',
    file:  'file',
  };

  // ----------------------------------------------------------------
  // State (persisted to localStorage)
  // ----------------------------------------------------------------
  const defaultState = () => ({
    apiUrl:      API_DEFAULT,
    hfToken:     '',          // optional HF PRO token to bypass anonymous ZeroGPU quota
    temperature: 0.7,
    maxTokens:   2048,
    chats:       [],          // [{id, title, createdAt, updatedAt, messages: [{role, content, images?}]}]
    currentId:   null,
  });

  const state = loadState();

  function loadState() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return defaultState();
      const parsed = JSON.parse(raw);
      return Object.assign(defaultState(), parsed);
    } catch {
      return defaultState();
    }
  }

  function saveState() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        apiUrl:      state.apiUrl,
        hfToken:     state.hfToken,
        temperature: state.temperature,
        maxTokens:   state.maxTokens,
        chats:       state.chats,
        currentId:   state.currentId,
      }));
    } catch (e) {
      console.warn('[mindi] failed to save state', e);
    }
  }

  // Runtime-only state (not persisted)
  const runtime = {
    status:        'connecting',  // connecting | online | demo | offline
    authBlocked:   false,         // true if last API call hit a quota/auth error
    pendingImages: [],            // [{name, dataUrl}]
    isSending:     false,
    lastCode:      null,          // {language, code}
    lastSections:  null,          // {thinking: [], ...}
  };

  // ----------------------------------------------------------------
  // DOM
  // ----------------------------------------------------------------
  const $  = (s) => document.querySelector(s);
  const $$ = (s) => Array.from(document.querySelectorAll(s));

  const els = {
    body:           document.body,
    sidebar:        $('#sidebar'),
    scrim:          $('#scrim'),
    brand:          $('#brand'),
    newChatBtn:     $('#new-chat-btn'),
    search:         $('#search'),
    history:        $('#chat-history'),
    historyEmpty:   $('#history-empty'),
    statusDot:      $('#status-dot'),
    statusText:     $('#status-text'),

    chat:           $('#chat'),
    hamburger:      $('#hamburger'),
    chatTitle:      $('#chat-title'),
    togglePreview:  $('#toggle-preview'),

    welcome:        $('#welcome'),
    quickCards:     $$('.quick-card'),
    messages:       $('#messages'),

    composer:       $('#composer'),
    composerImages: $('#composer-images'),
    attachBtn:      $('#attach-btn'),
    fileInput:      $('#file-input'),
    promptInput:    $('#prompt-input'),
    sendBtn:        $('#send-btn'),

    preview:        $('#preview'),
    tabs:           $$('.tab'),
    panes:          $$('.preview-pane'),
    copyCode:       $('#copy-code'),
    downloadCode:   $('#download-code'),
    codeOut:        $('#code-out'),
    codeOutInner:   $('#code-out-inner'),
    emptyCode:      $('#empty-code'),
    liveFrame:      $('#live-frame'),
    emptyLive:      $('#empty-live'),
    sections:       $('#sections'),
    emptySections:  $('#empty-sections'),

    settingsModal:  $('#settings-modal'),
    settingsUrl:    $('#settings-url'),
    settingsHfToken:$('#settings-hf-token'),
    hfTokenStatus:  $('#hf-token-status'),
    settingsTemp:   $('#settings-temp'),
    settingsTokens: $('#settings-tokens'),
    tempVal:        $('#temp-val'),
    tokensVal:      $('#tokens-val'),
    saveSettings:   $('#save-settings'),

    toasts:         $('#toasts'),
  };

  // ----------------------------------------------------------------
  // Utilities
  // ----------------------------------------------------------------
  function uid() {
    return 'c-' + Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
  }
  function escapeHtml(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
                     .replace(/"/g, '&quot;').replace(/'/g, '&#039;');
  }
  function escapeAttr(s) {
    return escapeHtml(s);
  }
  function fileToDataUrl(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }
  function debounce(fn, ms) {
    let t = null;
    return (...args) => {
      clearTimeout(t);
      t = setTimeout(() => fn(...args), ms);
    };
  }
  function downloadFile(filename, content) {
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }
  function relativeDateGroup(ts) {
    const d   = new Date(ts);
    const now = new Date();
    const start = (x) => { const z = new Date(x); z.setHours(0,0,0,0); return z; };
    const today     = start(now);
    const yesterday = new Date(today); yesterday.setDate(today.getDate() - 1);
    const week      = new Date(today); week.setDate(today.getDate() - 7);

    if (d >= today)     return 'Today';
    if (d >= yesterday) return 'Yesterday';
    if (d >= week)      return 'This Week';
    return 'Earlier';
  }
  function languageFromCode(code) {
    const trimmed = code.trim();
    if (/^<!doctype|^<html|^<\w+[\s>]/i.test(trimmed)) return 'markup';
    if (/^(import|from|def|class|print|if __name__)/m.test(trimmed)) return 'python';
    if (/^(import|export|const|function|class|let|var|=>)/m.test(trimmed)) return 'javascript';
    if (/^[\s\S]*\{[\s\S]*\}\s*$/.test(trimmed) && /^\s*"\w+"\s*:/m.test(trimmed)) return 'json';
    if (/(SELECT|INSERT|UPDATE|DELETE|CREATE TABLE)/i.test(trimmed)) return 'sql';
    if (/^[\.#]?[\w-]+\s*\{[^}]*:\s*[^;]+;/.test(trimmed)) return 'css';
    return 'plaintext';
  }

  // ----------------------------------------------------------------
  // Output cleaning + section parsing
  // ----------------------------------------------------------------
  // Strip all special tokens for chat display
  function cleanForDisplay(raw) {
    if (!raw) return '';
    let t = String(raw);

    // Section start/end tokens
    Object.keys(TOKEN_TO_KEY).forEach((tok) => {
      t = t.replace(new RegExp(`<\\|${tok}_start\\|>`, 'g'), '');
      t = t.replace(new RegExp(`<\\|${tok}_end\\|>`, 'g'), '');
    });

    // Conversation tokens
    t = t.replace(/<\|im_start\|>/g, '');
    t = t.replace(/<\|im_end\|>/g, '');
    t = t.replace(/<\|endoftext\|>/g, '');

    // Role prefixes at line starts
    t = t.replace(/^(system|user|assistant)\s*\n/gim, '');

    return t.trim();
  }

  // Parse the special token sections out of the raw response
  function parseSections(raw) {
    const sections = {};
    SECTION_ORDER.forEach((k) => { sections[k] = []; });
    if (!raw) return sections;

    const text = String(raw);
    Object.entries(TOKEN_TO_KEY).forEach(([tok, key]) => {
      const re = new RegExp(`<\\|${tok}_start\\|>([\\s\\S]*?)<\\|${tok}_end\\|>`, 'g');
      let m;
      while ((m = re.exec(text)) !== null) {
        const body = m[1].trim();
        if (body) sections[key].push(body);
      }
    });
    return sections;
  }

  // Merge API-provided sections with parsed ones (API wins, parsed fills gaps)
  function mergeSections(api, parsed) {
    const merged = {};
    SECTION_ORDER.forEach((k) => {
      const a = (api && Array.isArray(api[k])) ? api[k] : [];
      const p = (parsed && Array.isArray(parsed[k])) ? parsed[k] : [];
      merged[k] = a.length ? a : p;
    });
    return merged;
  }

  // Extract last fenced code block from the response text
  function extractLastCodeBlock(text) {
    if (!text) return null;
    const re = /```(\w+)?\s*\n([\s\S]*?)```/g;
    let last = null, m;
    while ((m = re.exec(text)) !== null) {
      last = { language: (m[1] || '').toLowerCase() || null, code: m[2] };
    }
    if (last) {
      if (!last.language) last.language = languageFromCode(last.code);
      return last;
    }
    return null;
  }

  // ----------------------------------------------------------------
  // Markdown renderer (limited: paragraphs, fenced code, inline code, bold/italic)
  // ----------------------------------------------------------------
  function renderMarkdown(text) {
    if (!text) return '';

    // Tokenize: split into fenced-code parts and text parts
    const segments = [];
    const re = /```(\w+)?\s*\n([\s\S]*?)```/g;
    let lastIdx = 0, m;
    while ((m = re.exec(text)) !== null) {
      if (m.index > lastIdx) {
        segments.push({ type: 'text', value: text.slice(lastIdx, m.index) });
      }
      segments.push({ type: 'code', lang: (m[1] || '').toLowerCase() || null, value: m[2] });
      lastIdx = re.lastIndex;
    }
    if (lastIdx < text.length) {
      segments.push({ type: 'text', value: text.slice(lastIdx) });
    }

    return segments.map((seg) => {
      if (seg.type === 'code') {
        const lang = seg.lang || languageFromCode(seg.value);
        const safe = escapeHtml(seg.value);
        const dataCode = escapeAttr(seg.value);
        return (
          `<pre class="md-code-block">` +
            `<div class="md-code-head">` +
              `<span>${escapeHtml(lang)}</span>` +
              `<button class="md-copy" data-code="${dataCode}" type="button">Copy</button>` +
            `</div>` +
            `<code class="language-${escapeHtml(lang)}">${safe}</code>` +
          `</pre>`
        );
      }
      // text segment
      let h = seg.value.trim();
      if (!h) return '';
      h = escapeHtml(h);
      h = h.replace(/`([^`\n]+)`/g, '<code class="md-inline">$1</code>');
      h = h.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
      h = h.replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s).,!?:;]|$)/g, '$1<em>$2</em>');
      return h.split(/\n{2,}/)
        .map((p) => '<p>' + p.replace(/\n/g, '<br>') + '</p>')
        .join('');
    }).join('');
  }

  // ----------------------------------------------------------------
  // API client
  // ----------------------------------------------------------------
  function authHeaders(extra) {
    const h = Object.assign({}, extra || {});
    if (state.hfToken) h['Authorization'] = `Bearer ${state.hfToken}`;
    return h;
  }

  // Detect responses that came back as a quota / auth error from the
  // backend's chat_fn try/except, so we can show actionable UX.
  function detectAuthError(result) {
    if (!result) return null;
    const text = String(result.response || '');
    const errs = (result.sections && result.sections.error) || [];
    const blob = (text + ' ' + errs.join(' ')).toLowerCase();
    if (/zerogpu|gpu quota|out of .* quota|exceeded .* quota|unlogged user/.test(blob)) {
      return state.hfToken
        ? 'Your HF token hit its ZeroGPU quota. Wait for the daily reset or use a PRO token.'
        : 'Anonymous ZeroGPU quota exhausted. Open Settings (double-click the MINDI logo) and paste your HF token.';
    }
    return null;
  }

  // Build a friendly assistant message shown inline in the chat when the
  // request is (or would be) blocked by the ZeroGPU quota / auth wall.
  // ZeroGPU quota is per-user — there is no way to bypass it without a
  // logged-in HF token. See https://huggingface.co/docs/hub/en/spaces-zerogpu
  function makeAuthBlockedResponse() {
    if (state.hfToken) {
      return {
        response:
`**Your HF token hit its daily ZeroGPU quota.**

ZeroGPU enforces a per-user GPU-time budget that resets every 24 hours after first use. Free HF accounts get a small daily allowance, **PRO accounts get 8\u00d7 more**, and PRO/Team/Enterprise can also top up with [pre-paid credits](https://huggingface.co/settings/billing) at \\$1 per 10 GPU-minutes.

**What to do:**
- Wait for the daily reset (24h after your first call today), **or**
- Top up credits in your HF billing settings, **or**
- Open **Settings** (double-click the MINDI logo) and paste a different PRO token.

I'll keep further messages local until you update the token \u2014 sending them now would just hit the same wall.`,
        sections: {},
      };
    }
    return {
      response:
`**Sign-in needed to use the live model.**

The MINDI 1.5 backend runs on HuggingFace ZeroGPU, which gives every IP a tiny anonymous quota (~3 minutes / day) before blocking further requests. That's why the first message worked but the next one didn't.

**To unlock real generation:**
1. Get a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (free account is fine; **PRO** gives 8\u00d7 more).
2. **Double-click the MINDI logo** \u2192 paste the token in the *HuggingFace token* field \u2192 *Save settings*.
3. Re-send your message.

Your token is stored only in this browser's local storage and sent as an \`Authorization: Bearer\` header to the Space.`,
      sections: {},
    };
  }

  async function pingHealth() {
    // If a previous request was blocked by ZeroGPU quota / auth, stay in
    // 'Auth required' until the user adds a token (applySettings clears it).
    // Otherwise this would silently flip back to 'online' every 60s and the
    // next user message would hit the same quota wall.
    if (runtime.authBlocked) {
      setStatus('demo', state.hfToken ? 'Quota exhausted' : 'Auth required');
      return;
    }
    if (!state.apiUrl) {
      setStatus('demo', 'Demo Mode');
      return;
    }
    try {
      const base = state.apiUrl.replace(/\/$/, '');
      const isGradio = base.includes('hf.space') || base.includes('huggingface.co');

      if (isGradio) {
        // For Gradio/HF Spaces: check the root URL which returns the Gradio page
        const res = await fetch(base, { method: 'HEAD', mode: 'no-cors' }).catch(() => null);
        // no-cors always returns opaque response, so we check for network errors
        if (res) {
          setStatus('online', 'MINDI · HF Space');
        } else {
          setStatus('demo', 'Demo Mode (Space unreachable)');
        }
      } else {
        // Direct REST API health check
        try {
          const res = await fetch(`${base}/api/health`, { method: 'GET', headers: { 'Accept': 'application/json' } });
          if (res.ok) {
            const d = await res.json().catch(() => ({}));
            setStatus('online', `${d.model || 'MINDI'} · online`);
          } else {
            setStatus('demo', 'Demo Mode');
          }
        } catch {
          setStatus('demo', 'Demo Mode');
        }
      }
    } catch {
      setStatus('demo', 'Demo Mode');
    }
  }

  async function callGenerate(prompt, image, signal) {
    const base = state.apiUrl.replace(/\/$/, '');

    // Detect if this is a Gradio HF Space
    const isGradio = base.includes('hf.space') || base.includes('huggingface.co/spaces');

    if (isGradio) {
      // Gradio 5.x SSE v3 protocol — two-step:
      // 1. POST /gradio_api/call/{api_name} → get event_id
      // 2. GET  /gradio_api/call/{api_name}/{event_id} → stream result

      // Step 1: Submit the request
      const submitRes = await fetch(`${base}/gradio_api/call/chat_fn`, {
        method: 'POST',
        headers: authHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          data: [prompt, image || null, state.temperature, state.maxTokens],
        }),
        signal,
      });
      if (!submitRes.ok) {
        const txt = await submitRes.text().catch(() => '');
        throw new Error(`API submit ${submitRes.status}: ${txt.slice(0, 200) || 'request failed'}`);
      }
      const { event_id } = await submitRes.json();
      if (!event_id) {
        throw new Error('No event_id returned from Gradio API');
      }

      // Step 2: Get the result via SSE stream
      const resultRes = await fetch(`${base}/gradio_api/call/chat_fn/${event_id}`, {
        method: 'GET',
        headers: authHeaders(),
        signal,
      });
      if (!resultRes.ok) {
        const txt = await resultRes.text().catch(() => '');
        throw new Error(`API result ${resultRes.status}: ${txt.slice(0, 200) || 'request failed'}`);
      }

      // Parse SSE response — look for the "complete" event with data
      const sseText = await resultRes.text();
      const lines = sseText.split('\n');
      let raw = null;
      for (let i = 0; i < lines.length; i++) {
        if (lines[i].startsWith('event: complete')) {
          // Next line(s) starting with "data: " contain the result
          const dataLine = lines[i + 1];
          if (dataLine && dataLine.startsWith('data: ')) {
            try {
              const parsed = JSON.parse(dataLine.slice(6));
              // Gradio wraps in array
              raw = Array.isArray(parsed) ? parsed[0] : parsed;
            } catch {
              raw = dataLine.slice(6);
            }
          }
          break;
        }
        if (lines[i].startsWith('event: error')) {
          const dataLine = lines[i + 1];
          const errMsg = dataLine?.startsWith('data: ') ? dataLine.slice(6) : 'Unknown Gradio error';
          throw new Error(`Gradio error: ${errMsg.slice(0, 300)}`);
        }
      }

      if (raw === null) {
        throw new Error('No complete event found in Gradio SSE response');
      }

      // raw is a JSON string from our chat_fn
      try {
        return JSON.parse(raw);
      } catch {
        return { response: String(raw), sections: {} };
      }

    } else {
      // Direct REST API (Modal or custom)
      const body = { prompt, temperature: state.temperature, max_tokens: state.maxTokens };
      if (image) body.image = image;
      const res = await fetch(`${base}/api/generate`, {
        method: 'POST',
        headers: authHeaders({ 'Content-Type': 'application/json', 'Accept': 'application/json' }),
        body: JSON.stringify(body),
        signal,
      });
      if (!res.ok) {
        const txt = await res.text().catch(() => '');
        throw new Error(`API ${res.status}: ${txt.slice(0, 200) || 'request failed'}`);
      }
      return res.json();
    }
  }

  // ----------------------------------------------------------------
  // Demo / Fallback responses
  // ----------------------------------------------------------------
  const DEMO_RESPONSES = [
    {
      match: /landing|hero|next\.?js/i,
      response:
`Here's a clean Next.js landing page using Tailwind CSS:

\`\`\`tsx
// app/page.tsx
export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-b from-slate-950 to-slate-900 text-white">
      <section className="max-w-6xl mx-auto px-6 py-24 text-center">
        <span className="inline-block px-3 py-1 rounded-full bg-violet-500/15 text-violet-300 text-xs font-mono tracking-widest uppercase mb-6">
          Now in beta
        </span>
        <h1 className="text-5xl md:text-7xl font-semibold tracking-tight">
          Build faster.<br/>
          <span className="bg-gradient-to-r from-violet-400 to-blue-400 bg-clip-text text-transparent">
            Ship sooner.
          </span>
        </h1>
        <p className="mt-6 text-lg text-slate-300 max-w-2xl mx-auto">
          The frontend you'd build if you had unlimited time, in a single prompt.
        </p>
        <div className="mt-10 flex justify-center gap-3">
          <a className="px-6 py-3 rounded-full bg-gradient-to-r from-violet-600 to-blue-600 font-medium" href="#cta">
            Get started
          </a>
          <a className="px-6 py-3 rounded-full border border-white/10 hover:bg-white/5" href="#features">
            See features
          </a>
        </div>
      </section>
    </main>
  );
}
\`\`\`

This sets up a hero section with a gradient headline, a kicker badge, and two CTAs. Drop in an \`<Image>\` background or particle layer next.`,
      sections: {
        thinking: ['User wants a Next.js landing page. Producing a single-file app/page.tsx using Tailwind for the hero, with two CTAs and accessible markup.'],
        code: ['app/page.tsx generated with hero section + gradient headline.'],
        critique: [],
        fix: [],
      },
    },
    {
      match: /dashboard|chart|analytics/i,
      response:
`Here's a self-contained dashboard UI in vanilla HTML/CSS:

\`\`\`html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>Pulsegrid · Dashboard</title>
<style>
  :root { --bg:#0b0b14; --panel:#14141f; --border:rgba(255,255,255,.08); --text:#ececf1; --mute:#8b94a7; --acc:#7c3aed; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); font:14px/1.55 'Inter', sans-serif; min-height:100vh; display:grid; grid-template-columns:240px 1fr; }
  aside { background:var(--panel); border-right:1px solid var(--border); padding:20px; }
  aside h1 { font-size:18px; background:linear-gradient(135deg,#7c3aed,#2563eb); -webkit-background-clip:text; color:transparent; margin-bottom:24px; }
  nav a { display:block; padding:10px 12px; border-radius:8px; color:var(--mute); text-decoration:none; margin-bottom:2px; }
  nav a.active { background:rgba(124,58,237,.15); color:#fff; }
  main { padding:24px; overflow-y:auto; }
  .stats { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:20px; }
  .stat { background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:16px; }
  .stat .v { font-size:24px; font-weight:600; margin-top:6px; }
  .stat .l { color:var(--mute); font-size:12px; text-transform:uppercase; letter-spacing:.1em; }
  .chart { background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:18px; height:260px; display:flex; align-items:end; gap:8px; }
  .bar { flex:1; background:linear-gradient(180deg,#7c3aed,#2563eb); border-radius:6px 6px 0 0; }
</style>
</head>
<body>
<aside>
  <h1>Pulsegrid</h1>
  <nav>
    <a class="active">Overview</a>
    <a>Customers</a>
    <a>Revenue</a>
    <a>Settings</a>
  </nav>
</aside>
<main>
  <div class="stats">
    <div class="stat"><div class="l">Revenue</div><div class="v">$48,210</div></div>
    <div class="stat"><div class="l">Active users</div><div class="v">12,840</div></div>
    <div class="stat"><div class="l">Conversion</div><div class="v">4.2%</div></div>
    <div class="stat"><div class="l">Churn</div><div class="v">1.1%</div></div>
  </div>
  <div class="chart">
    <div class="bar" style="height:40%"></div>
    <div class="bar" style="height:65%"></div>
    <div class="bar" style="height:30%"></div>
    <div class="bar" style="height:80%"></div>
    <div class="bar" style="height:55%"></div>
    <div class="bar" style="height:90%"></div>
    <div class="bar" style="height:70%"></div>
  </div>
</main>
</body>
</html>
\`\`\`

Hit the **Preview** tab to see it rendered live.`,
      sections: {
        thinking: ['User wants a dashboard. Building a self-contained HTML page with sidebar nav, stat cards, and a CSS-only bar chart so it can render in the iframe preview.'],
        code: ['Single-file dashboard.html with grid layout, stats, sidebar.'],
        critique: ['No real charting library — bars are static. For production, swap in Recharts/Chart.js.'],
        fix: [],
      },
    },
    {
      match: /api|fastapi|backend|jwt|postgres/i,
      response:
`Here's a minimal but production-shaped FastAPI service:

\`\`\`python
# main.py
from datetime import datetime, timedelta
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import jwt
from pydantic import BaseModel

from .database import SessionLocal, engine
from . import models, schemas

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Notes API")
SECRET_KEY = "change-me"
ALGORITHM  = "HS256"
EXPIRES    = timedelta(hours=24)
pwd        = CryptContext(schemes=["bcrypt"])
oauth2     = OAuth2PasswordBearer(tokenUrl="/auth/login")

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

def make_token(sub: str) -> str:
    return jwt.encode({"sub": sub, "exp": datetime.utcnow() + EXPIRES}, SECRET_KEY, ALGORITHM)

@app.post("/auth/login")
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter_by(email=form.username).first()
    if not user or not pwd.verify(form.password, user.hashed_password):
        raise HTTPException(401, "Invalid credentials")
    return {"access_token": make_token(user.email), "token_type": "bearer"}

@app.get("/notes", response_model=list[schemas.Note])
def list_notes(db: Session = Depends(get_db), token: str = Depends(oauth2)):
    payload = jwt.decode(token, SECRET_KEY, [ALGORITHM])
    return db.query(models.Note).filter_by(owner_email=payload["sub"]).all()
\`\`\`

Pair this with \`models.py\`, \`schemas.py\`, and a \`.env\` containing \`DATABASE_URL=postgresql://…\`.`,
      sections: {
        thinking: ['User asked for FastAPI + JWT + Postgres. Sketching the service entrypoint with auth flow and a protected GET endpoint.'],
        code: ['main.py with /auth/login and /notes endpoints.'],
        critique: ['SECRET_KEY hardcoded — move to env. bcrypt rounds are default — increase for production.'],
        suggest: ['Add Alembic migrations and pytest fixtures next.'],
      },
    },
    {
      match: /bug|fix|debug|divide/i,
      response:
`Two bugs in that snippet:

1. **No zero-divisor guard** — \`divide_list([1,2], 0)\` throws \`ZeroDivisionError\`.
2. **No type validation** — non-numeric items raise \`TypeError\` deep in the loop.

Here's the fix:

\`\`\`python
def divide_list(numbers, divisor):
    if divisor == 0:
        raise ValueError("divisor must be non-zero")
    result = []
    for n in numbers:
        if not isinstance(n, (int, float)):
            raise TypeError(f"non-numeric item: {n!r}")
        result.append(n / divisor)
    return result
\`\`\`

For very large lists, switch to a generator (\`yield\` instead of \`append\`) to keep memory flat.`,
      sections: {
        thinking: ['Two issues: zero-divisor crash, non-numeric crash. Adding explicit guards and a clearer error message.'],
        critique: ['Function silently coerces booleans because bool ⊂ int in Python — may want to exclude them explicitly.'],
        fix: ['Added divisor==0 guard, added isinstance validation, kept the original signature.'],
        code: ['Patched divide_list with safe input checking.'],
      },
    },
  ];

  const DEFAULT_DEMO = {
    response:
`I'm running in **Demo Mode** because the live API isn't reachable.

Try one of the quick-action prompts on the welcome screen, or open Settings (double-click the brand logo) and paste your MINDI API URL.

\`\`\`javascript
// You sent a prompt and I'm a placeholder.
// Connect the real API to see actual generations.
console.log("MINDI 1.5 — awaiting connection");
\`\`\``,
    sections: {
      thinking: ['No API URL configured or the endpoint is unreachable. Returning a demo response.'],
      code: ['Stub response.'],
    },
  };

  function pickDemo(prompt) {
    const found = DEMO_RESPONSES.find((d) => d.match.test(prompt));
    return found || DEFAULT_DEMO;
  }

  async function generateDemo(prompt) {
    await new Promise((r) => setTimeout(r, 1200 + Math.random() * 700));
    const demo = pickDemo(prompt);
    return { response: demo.response, sections: demo.sections || {} };
  }

  // ----------------------------------------------------------------
  // Status
  // ----------------------------------------------------------------
  function setStatus(status, label) {
    runtime.status = status;
    els.statusDot.classList.remove('status-dot--gray', 'status-dot--green', 'status-dot--yellow', 'status-dot--red');
    const map = { connecting: 'gray', online: 'green', demo: 'yellow', offline: 'red' };
    els.statusDot.classList.add(`status-dot--${map[status] || 'gray'}`);
    els.statusText.textContent = label;
  }

  // ----------------------------------------------------------------
  // Toasts
  // ----------------------------------------------------------------
  function toast(msg, kind = 'info', ms = 2400) {
    const el = document.createElement('div');
    el.className = `toast toast--${kind}`;
    el.innerHTML = `<span class="toast-icon"></span><span>${escapeHtml(msg)}</span>`;
    els.toasts.appendChild(el);
    setTimeout(() => {
      el.classList.add('is-leaving');
      setTimeout(() => el.remove(), 260);
    }, ms);
  }

  // ----------------------------------------------------------------
  // Chat data helpers
  // ----------------------------------------------------------------
  function currentChat() {
    return state.chats.find((c) => c.id === state.currentId) || null;
  }
  function ensureChat() {
    let chat = currentChat();
    if (!chat) {
      chat = { id: uid(), title: 'New chat', createdAt: Date.now(), updatedAt: Date.now(), messages: [] };
      state.chats.unshift(chat);
      state.currentId = chat.id;
    }
    return chat;
  }
  function deriveTitle(text) {
    const t = (text || '').replace(/\s+/g, ' ').trim();
    if (!t) return 'New chat';
    return t.length > 42 ? t.slice(0, 42).trim() + '…' : t;
  }

  // ----------------------------------------------------------------
  // Render: messages
  // ----------------------------------------------------------------
  function renderMessages() {
    const chat = currentChat();
    const hasMessages = !!(chat && chat.messages.length);
    els.chat.classList.toggle('has-messages', hasMessages);
    els.messages.innerHTML = '';
    if (!hasMessages) return;

    chat.messages.forEach((m) => els.messages.appendChild(renderMessageEl(m)));

    // Highlight after insertion
    if (window.Prism) {
      try { Prism.highlightAllUnder(els.messages); } catch { /* noop */ }
    }
    scrollMessagesToBottom();
  }

  function renderMessageEl(m) {
    const wrap = document.createElement('div');
    wrap.className = `msg msg--${m.role === 'user' ? 'user' : 'asst'}`;

    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';
    avatar.textContent = m.role === 'user' ? 'You'.slice(0,1) : 'M';

    const body = document.createElement('div');
    body.className = 'msg-body';

    const meta = document.createElement('div');
    meta.className = 'msg-meta';
    meta.innerHTML = `<span class="msg-meta-name">${m.role === 'user' ? 'You' : 'MINDI 1.5'}</span>`;
    body.appendChild(meta);

    if (Array.isArray(m.images) && m.images.length) {
      const imgsWrap = document.createElement('div');
      imgsWrap.className = 'msg-images';
      m.images.forEach((src) => {
        const img = document.createElement('img');
        img.src = src;
        img.alt = 'Attached image';
        imgsWrap.appendChild(img);
      });
      body.appendChild(imgsWrap);
    }

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    if (m.loading) {
      bubble.innerHTML = `<span>${escapeHtml(m.content || 'Thinking')}</span><span class="dots"><span></span><span></span><span></span></span>`;
      wrap.classList.add('msg-loading');
    } else {
      bubble.innerHTML = renderMarkdown(cleanForDisplay(m.content));
    }
    body.appendChild(bubble);

    wrap.appendChild(avatar);
    wrap.appendChild(body);
    return wrap;
  }

  function scrollMessagesToBottom() {
    requestAnimationFrame(() => {
      els.messages.scrollTop = els.messages.scrollHeight;
    });
  }

  // ----------------------------------------------------------------
  // Render: history sidebar
  // ----------------------------------------------------------------
  function renderHistory() {
    const q = (els.search.value || '').toLowerCase().trim();
    const filtered = q
      ? state.chats.filter((c) => c.title.toLowerCase().includes(q) ||
          c.messages.some((m) => (m.content || '').toLowerCase().includes(q)))
      : state.chats;

    els.history.innerHTML = '';

    if (!filtered.length) {
      els.history.innerHTML = `
        <div class="history-empty">
          <p>${q ? 'No matches.' : 'No chats yet.'}</p>
          <p class="muted">${q ? 'Try a different search.' : 'Start a conversation to see it here.'}</p>
        </div>`;
      return;
    }

    // Group by date
    const groupOrder = ['Today', 'Yesterday', 'This Week', 'Earlier'];
    const groups = {};
    filtered.forEach((c) => {
      const g = relativeDateGroup(c.updatedAt || c.createdAt);
      (groups[g] ||= []).push(c);
    });

    groupOrder.forEach((g) => {
      if (!groups[g]) return;
      const wrap = document.createElement('div');
      wrap.className = 'history-group';
      wrap.innerHTML = `<div class="history-group-title">${g}</div>`;
      groups[g].forEach((c) => {
        const btn = document.createElement('button');
        btn.className = 'history-item';
        if (c.id === state.currentId) btn.classList.add('is-active');
        btn.textContent = c.title || 'New chat';
        btn.title = c.title;
        btn.addEventListener('click', () => loadChat(c.id));
        wrap.appendChild(btn);
      });
      els.history.appendChild(wrap);
    });
  }

  function loadChat(id) {
    state.currentId = id;
    const chat = currentChat();
    if (chat) {
      els.chatTitle.textContent = chat.title || 'New chat';
      // recompute preview panels from last assistant message
      const lastAssistant = [...chat.messages].reverse().find((m) => m.role === 'assistant' && !m.loading);
      if (lastAssistant) updatePreviewFromAssistant(lastAssistant);
      else clearPreview();
    }
    renderMessages();
    renderHistory();
    saveState();
    closeMobileSidebar();
  }

  // ----------------------------------------------------------------
  // Preview panel updates
  // ----------------------------------------------------------------
  function clearPreview() {
    runtime.lastCode = null;
    runtime.lastSections = null;
    els.codeOut.hidden = true;
    els.emptyCode.hidden = false;
    els.liveFrame.hidden = true;
    els.emptyLive.hidden = false;
    els.sections.hidden = true;
    els.emptySections.hidden = false;
    els.sections.innerHTML = '';
    els.codeOutInner.textContent = '';
  }

  function updatePreviewFromAssistant(msg) {
    const cleaned = cleanForDisplay(msg.content);
    const block = extractLastCodeBlock(cleaned);
    runtime.lastCode = block;
    if (block) renderCodeOut(block);
    else { els.codeOut.hidden = true; els.emptyCode.hidden = false; }

    // Live HTML preview
    if (block && /^(markup|html)$/i.test(block.language || '')) {
      renderLivePreview(block.code);
    } else {
      els.liveFrame.hidden = true;
      els.emptyLive.hidden = false;
    }

    // Sections
    const apiSections = msg.sections || {};
    const parsedSections = parseSections(msg.content);
    runtime.lastSections = mergeSections(apiSections, parsedSections);
    renderSections(runtime.lastSections);
  }

  function renderCodeOut(block) {
    const lang = block.language || 'plaintext';
    els.codeOutInner.className = `language-${lang}`;
    els.codeOutInner.textContent = block.code;
    els.emptyCode.hidden = true;
    els.codeOut.hidden = false;
    if (window.Prism) {
      try { Prism.highlightElement(els.codeOutInner); } catch { /* noop */ }
    }
  }

  function renderLivePreview(html) {
    els.emptyLive.hidden = true;
    els.liveFrame.hidden = false;
    const doc = els.liveFrame.contentDocument || els.liveFrame.contentWindow.document;
    doc.open();
    doc.write(html);
    doc.close();
  }

  function renderSections(sections) {
    const hasAny = SECTION_ORDER.some((k) => (sections[k] || []).length);
    if (!hasAny) {
      els.sections.hidden = true;
      els.emptySections.hidden = false;
      els.sections.innerHTML = '';
      return;
    }
    els.emptySections.hidden = true;
    els.sections.hidden = false;
    els.sections.innerHTML = '';

    SECTION_ORDER.forEach((kind) => {
      const items = sections[kind] || [];
      items.forEach((body, i) => {
        const card = document.createElement('div');
        card.className = 'section-card';
        card.dataset.kind = kind;
        card.innerHTML = `
          <div class="section-card-head">
            <span class="section-tag">${SECTION_LABELS[kind]}</span>
            <span>${items.length > 1 ? `${i + 1} / ${items.length}` : ''}</span>
          </div>
          <div class="section-card-body">${escapeHtml(body)}</div>
        `;
        els.sections.appendChild(card);
      });
    });
  }

  // ----------------------------------------------------------------
  // Send flow
  // ----------------------------------------------------------------
  async function send() {
    if (runtime.isSending) return;
    const text = els.promptInput.value.trim();
    if (!text && !runtime.pendingImages.length) return;

    const chat = ensureChat();
    const wasEmpty = chat.messages.length === 0;

    const userMsg = {
      role:    'user',
      content: text,
      images:  runtime.pendingImages.map((p) => p.dataUrl),
      ts:      Date.now(),
    };
    chat.messages.push(userMsg);
    chat.updatedAt = Date.now();

    if (wasEmpty) {
      chat.title = deriveTitle(text);
      els.chatTitle.textContent = chat.title;
    }

    // Reset input
    const imageForApi = runtime.pendingImages[0]?.dataUrl || null;
    els.promptInput.value = '';
    autosizeTextarea();
    clearPendingImages();
    updateSendEnabled();

    // Loading message
    const loadingMsg = { role: 'assistant', content: 'Thinking', loading: true, ts: Date.now() };
    chat.messages.push(loadingMsg);
    renderMessages();
    renderHistory();
    saveState();

    runtime.isSending = true;
    let coldStartTimer = setTimeout(() => {
      loadingMsg.content = 'Cold start — booting the GPU. First request can take ~4 minutes';
      renderMessages();
    }, COLD_START_HINT_MS);

    let result, errored = null;
    try {
      // If we already know auth is blocked, don't call the API again — the
      // request would just consume more anonymous quota and return the same
      // error. Show the inline 'add your token' card instead.
      if (runtime.authBlocked) {
        result = makeAuthBlockedResponse();
      } else if (runtime.status === 'demo' || !state.apiUrl) {
        result = await generateDemo(text);
      } else {
        result = await callGenerate(text, imageForApi);
      }
    } catch (e) {
      errored = e;
      // Auto-fallback to demo so the UI never feels dead
      result = await generateDemo(text).catch(() => ({ response: 'Generation failed.' }));
      if (!/cold|abort|signal/i.test(String(e?.message || ''))) {
        toast('API error — falling back to demo', 'error', 3500);
      }
    } finally {
      clearTimeout(coldStartTimer);
      runtime.isSending = false;
    }

    // If the API returned a quota / auth error, surface it clearly and
    // stop calling the API on subsequent messages until the user adds a token.
    const authMsg = detectAuthError(result);
    if (authMsg) {
      runtime.authBlocked = true;
      toast(authMsg, 'error', 7000);
      setStatus('demo', state.hfToken ? 'Quota exhausted' : 'Auth required');
      // Replace the raw backend error with a friendlier inline card so the
      // chat doesn't show a wall of quota-error text.
      result = makeAuthBlockedResponse();
    }

    // Remove loading, push assistant
    const idx = chat.messages.indexOf(loadingMsg);
    if (idx !== -1) chat.messages.splice(idx, 1);

    const assistantMsg = {
      role:     'assistant',
      content:  result?.response || '(no response)',
      sections: result?.sections || null,
      ts:       Date.now(),
    };
    chat.messages.push(assistantMsg);
    chat.updatedAt = Date.now();

    renderMessages();
    renderHistory();
    updatePreviewFromAssistant(assistantMsg);
    saveState();

    if (errored) console.warn('[mindi] generate error:', errored);
  }

  // ----------------------------------------------------------------
  // Composer interactions
  // ----------------------------------------------------------------
  function autosizeTextarea() {
    const ta = els.promptInput;
    ta.style.height = 'auto';
    const next = Math.min(ta.scrollHeight, MAX_TEXTAREA);
    ta.style.height = next + 'px';
  }
  function updateSendEnabled() {
    const has = els.promptInput.value.trim().length > 0 || runtime.pendingImages.length > 0;
    els.sendBtn.disabled = !has || runtime.isSending;
  }
  function clearPendingImages() {
    runtime.pendingImages = [];
    renderPendingImages();
  }
  function renderPendingImages() {
    if (!runtime.pendingImages.length) {
      els.composerImages.hidden = true;
      els.composerImages.innerHTML = '';
      return;
    }
    els.composerImages.hidden = false;
    els.composerImages.innerHTML = '';
    runtime.pendingImages.forEach((p, i) => {
      const tile = document.createElement('div');
      tile.className = 'composer-image';
      tile.style.backgroundImage = `url("${p.dataUrl}")`;
      tile.title = p.name;
      const rm = document.createElement('button');
      rm.className = 'composer-image-remove';
      rm.setAttribute('aria-label', 'Remove image');
      rm.textContent = '×';
      rm.addEventListener('click', () => {
        runtime.pendingImages.splice(i, 1);
        renderPendingImages();
        updateSendEnabled();
      });
      tile.appendChild(rm);
      els.composerImages.appendChild(tile);
    });
  }

  async function handleFileChosen(file) {
    if (!file || !file.type.startsWith('image/')) {
      toast('Only image files are supported.', 'error');
      return;
    }
    if (file.size > 6 * 1024 * 1024) {
      toast('Image too large (max 6 MB).', 'error');
      return;
    }
    try {
      const dataUrl = await fileToDataUrl(file);
      runtime.pendingImages = [{ name: file.name, dataUrl }]; // single image per request
      renderPendingImages();
      updateSendEnabled();
    } catch {
      toast('Could not read that image.', 'error');
    }
  }

  // ----------------------------------------------------------------
  // Tabs
  // ----------------------------------------------------------------
  function activateTab(tabName) {
    els.tabs.forEach((t) => {
      const on = t.dataset.tab === tabName;
      t.classList.toggle('is-active', on);
      t.setAttribute('aria-selected', on ? 'true' : 'false');
    });
    els.panes.forEach((p) => {
      p.classList.toggle('is-active', p.dataset.pane === tabName);
    });
  }

  // ----------------------------------------------------------------
  // Settings modal
  // ----------------------------------------------------------------
  function maskToken(t) {
    if (!t) return 'none';
    if (t.length <= 8) return 'set';
    return `${t.slice(0, 4)}…${t.slice(-4)}`;
  }
  function refreshTokenStatus() {
    if (els.hfTokenStatus) els.hfTokenStatus.textContent = maskToken(state.hfToken);
  }
  function openSettings() {
    els.settingsUrl.value      = state.apiUrl || '';
    if (els.settingsHfToken) els.settingsHfToken.value = state.hfToken || '';
    els.settingsTemp.value     = state.temperature;
    els.settingsTokens.value   = state.maxTokens;
    els.tempVal.textContent    = Number(state.temperature).toFixed(2);
    els.tokensVal.textContent  = state.maxTokens;
    refreshTokenStatus();
    els.settingsModal.hidden   = false;
    setTimeout(() => els.settingsUrl.focus(), 50);
  }
  function closeSettings() {
    els.settingsModal.hidden = true;
  }
  function applySettings() {
    const url    = els.settingsUrl.value.trim();
    const token  = els.settingsHfToken ? els.settingsHfToken.value.trim() : '';
    const temp   = parseFloat(els.settingsTemp.value);
    const tokens = parseInt(els.settingsTokens.value, 10);
    const tokenChanged = token !== state.hfToken;
    state.apiUrl      = url || API_DEFAULT;
    state.hfToken     = token;
    state.temperature = isFinite(temp) ? temp : 0.7;
    state.maxTokens   = isFinite(tokens) ? tokens : 2048;
    // If the user just saved a new (non-empty) token, give the API another shot.
    if (tokenChanged && token) {
      runtime.authBlocked = false;
    }
    saveState();
    refreshTokenStatus();
    closeSettings();
    toast(tokenChanged && token ? 'Token saved — retrying API' : 'Settings saved', 'success');
    pingHealth();
  }

  // ----------------------------------------------------------------
  // Mobile sidebar / preview toggling
  // ----------------------------------------------------------------
  function openMobileSidebar()  { els.body.classList.add('sidebar-open'); }
  function closeMobileSidebar() { els.body.classList.remove('sidebar-open'); }
  function togglePreview() {
    if (window.matchMedia('(max-width: 1024px)').matches) {
      els.body.classList.toggle('preview-open');
    } else {
      els.body.classList.toggle('preview-hidden');
    }
  }

  // ----------------------------------------------------------------
  // Copy / download from preview panel
  // ----------------------------------------------------------------
  async function copyLastCode() {
    if (!runtime.lastCode) {
      toast('No code to copy yet', 'info');
      return;
    }
    try {
      await navigator.clipboard.writeText(runtime.lastCode.code);
      toast('Copied to clipboard', 'success', 1600);
    } catch {
      toast('Clipboard unavailable', 'error');
    }
  }
  function downloadLastCode() {
    if (!runtime.lastCode) {
      toast('No code to download yet', 'info');
      return;
    }
    const ext = (() => {
      const m = { javascript: 'js', typescript: 'ts', tsx: 'tsx', jsx: 'jsx',
                  python: 'py', markup: 'html', html: 'html', css: 'css',
                  json: 'json', sql: 'sql', bash: 'sh' };
      return m[runtime.lastCode.language] || 'txt';
    })();
    downloadFile(`mindi-output.${ext}`, runtime.lastCode.code);
  }

  // ----------------------------------------------------------------
  // Bind events
  // ----------------------------------------------------------------
  // The active send handler — overridden by agent init if available
  let activeSend = send;

  function bind() {
    // Composer
    els.promptInput.addEventListener('input', () => { autosizeTextarea(); updateSendEnabled(); });
    els.promptInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        activeSend();
      }
    });
    els.sendBtn.addEventListener('click', () => activeSend());

    // Attach
    els.attachBtn.addEventListener('click', () => els.fileInput.click());
    els.fileInput.addEventListener('change', (e) => {
      const f = e.target.files?.[0];
      if (f) handleFileChosen(f);
      e.target.value = '';
    });

    // Drag & drop on composer
    ['dragenter', 'dragover'].forEach((ev) => {
      els.composer.addEventListener(ev, (e) => { e.preventDefault(); els.composer.style.borderColor = 'rgba(124, 58, 237, .6)'; });
    });
    ['dragleave', 'drop'].forEach((ev) => {
      els.composer.addEventListener(ev, (e) => { e.preventDefault(); els.composer.style.borderColor = ''; });
    });
    els.composer.addEventListener('drop', (e) => {
      const file = e.dataTransfer?.files?.[0];
      if (file) handleFileChosen(file);
    });

    // Quick action cards
    els.quickCards.forEach((card) => {
      card.addEventListener('click', () => {
        els.promptInput.value = card.dataset.prompt || '';
        autosizeTextarea();
        updateSendEnabled();
        els.promptInput.focus();
      });
    });

    // Sidebar
    els.newChatBtn.addEventListener('click', () => {
      state.currentId = null;
      ensureChat();
      els.chatTitle.textContent = 'New chat';
      clearPreview();
      renderMessages();
      renderHistory();
      saveState();
      els.promptInput.focus();
      closeMobileSidebar();
    });
    els.search.addEventListener('input', debounce(renderHistory, 120));
    els.hamburger.addEventListener('click', openMobileSidebar);
    els.scrim.addEventListener('click', () => { closeMobileSidebar(); els.body.classList.remove('preview-open'); });
    els.togglePreview.addEventListener('click', togglePreview);

    // Tabs
    els.tabs.forEach((t) => t.addEventListener('click', () => activateTab(t.dataset.tab)));

    // Code copy / download
    els.copyCode.addEventListener('click', copyLastCode);
    els.downloadCode.addEventListener('click', downloadLastCode);

    // Click handler for inline copy buttons inside messages (delegated)
    els.messages.addEventListener('click', async (e) => {
      const btn = e.target.closest('.md-copy');
      if (!btn) return;
      try {
        await navigator.clipboard.writeText(btn.dataset.code || '');
        const prev = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(() => { btn.textContent = prev; }, 1400);
      } catch {
        toast('Clipboard unavailable', 'error');
      }
    });

    // Brand → settings (double-click)
    els.brand.addEventListener('dblclick', openSettings);

    // Settings modal
    els.settingsModal.addEventListener('click', (e) => {
      if (e.target.matches('[data-close]') || e.target.closest('[data-close]')) closeSettings();
    });
    els.settingsTemp.addEventListener('input', () => {
      els.tempVal.textContent = Number(els.settingsTemp.value).toFixed(2);
    });
    els.settingsTokens.addEventListener('input', () => {
      els.tokensVal.textContent = els.settingsTokens.value;
    });
    els.saveSettings.addEventListener('click', applySettings);

    // Esc to close modal / mobile drawers
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        if (!els.settingsModal.hidden) closeSettings();
        closeMobileSidebar();
        els.body.classList.remove('preview-open');
      }
      // Cmd/Ctrl + K → focus search
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        els.search.focus();
      }
    });

    // Keep textarea sized after window resize (font reflow)
    window.addEventListener('resize', autosizeTextarea);
  }

  // ----------------------------------------------------------------
  // Agent integration
  // ----------------------------------------------------------------
  const agentEls = {
    log:         document.getElementById('agent-log'),
    sandbox:     document.getElementById('agent-sandbox'),
    console:     document.getElementById('agent-console'),
    consoleBody: document.getElementById('agent-console-body'),
    emptyAgent:  document.getElementById('empty-agent'),
  };

  const STEP_ICONS = {
    plan:     '📋',
    generate: '⚡',
    execute:  '▶️',
    verify:   '✅',
    fix:      '🔧',
    done:     '🎉',
    error:    '❌',
  };

  const STEP_LABELS = {
    plan:     'Planning',
    generate: 'Generating Code',
    execute:  'Executing',
    verify:   'Verifying Output',
    fix:      'Fixing Error',
    done:     'Complete',
    error:    'Error',
  };

  function isCodeRequest(text) {
    return /\b(build|create|make|write|generate|code|html|css|app|page|website|component|function|class|api|dashboard|landing|todo|form|navbar|button|layout|design)\b/i.test(text);
  }

  function renderAgentStep(run, step) {
    if (!agentEls.log) return;

    // Show agent panel, hide empty state
    agentEls.emptyAgent && (agentEls.emptyAgent.hidden = true);
    agentEls.log.hidden = false;

    // Switch to agent tab
    const agentTab = document.querySelector('.tab[data-tab="agent"]');
    if (agentTab && !agentTab.classList.contains('is-active')) {
      agentTab.click();
    }

    // Find or create step element
    let el = agentEls.log.querySelector(`[data-step-id="${step.id}"]`);
    if (!el) {
      el = document.createElement('div');
      el.className = 'agent-step';
      el.dataset.stepId = step.id;
      el.innerHTML = `
        <div class="agent-step-icon"></div>
        <div class="agent-step-body">
          <div class="agent-step-title"></div>
          <div class="agent-step-detail"></div>
        </div>`;
      agentEls.log.appendChild(el);
    }

    // Update status class
    el.className = `agent-step agent-step--${step.status}`;

    // Update icon
    const iconEl = el.querySelector('.agent-step-icon');
    const statusIcons = { running: '⏳', success: '✅', failed: '❌', pending: '⏺' };
    iconEl.textContent = step.status === 'success' || step.status === 'failed'
      ? statusIcons[step.status]
      : (STEP_ICONS[step.type] || '⏳');

    // Update title
    el.querySelector('.agent-step-title').textContent = STEP_LABELS[step.type] || step.type;

    // Update detail
    el.querySelector('.agent-step-detail').textContent = step.detail || '';

    // Auto-scroll
    agentEls.log.scrollTop = agentEls.log.scrollHeight;

    // Show sandbox and console when executing
    if (step.type === 'execute') {
      agentEls.sandbox.hidden = false;
      agentEls.console.hidden = false;
    }

    // Update console on execution results
    if (step.type === 'execute' && (step.status === 'success' || step.status === 'failed')) {
      const run_ = run;  // closure
      if (run_.currentCode) {
        // Show code in the Code tab too
        const block = { language: run_.language || 'javascript', code: run_.currentCode };
        runtime.lastCode = block;
        renderCodeOut(block);
      }
    }

    // When done, show final code in preview
    if (step.type === 'done' && step.status === 'success' && run.currentCode) {
      const lang = run.language || 'javascript';
      if (/^(html|markup)$/i.test(lang)) {
        // Render in live preview
        els.liveFrame.hidden = false;
        const emptyLive = document.getElementById('empty-live');
        if (emptyLive) emptyLive.hidden = true;
        els.liveFrame.srcdoc = run.currentCode;
      }
    }
  }

  function clearAgentUI() {
    if (agentEls.log) {
      agentEls.log.innerHTML = '';
      agentEls.log.hidden = true;
    }
    if (agentEls.sandbox) {
      agentEls.sandbox.innerHTML = '';
      agentEls.sandbox.hidden = true;
    }
    if (agentEls.console) {
      agentEls.console.hidden = true;
    }
    if (agentEls.consoleBody) {
      agentEls.consoleBody.textContent = '';
    }
    if (agentEls.emptyAgent) {
      agentEls.emptyAgent.hidden = false;
    }
  }

  async function runAgent(prompt, image) {
    clearAgentUI();

    const apiCall = async (p, img) => {
      if (runtime.status === 'demo' || !state.apiUrl) {
        return generateDemo(p);
      }
      return callGenerate(p, img);
    };

    const result = await MINDIAgent.run(prompt, {
      apiCall,
      sandboxContainer: agentEls.sandbox,
      image,
      onStep: (run, step) => {
        renderAgentStep(run, step);

        // Update console output
        if (step.type === 'execute' && agentEls.consoleBody) {
          const detail = step.detail || '';
          if (step.status === 'failed') {
            agentEls.consoleBody.innerHTML += `<span class="console-error">${escapeHtml(detail)}</span>\n`;
          } else if (step.status === 'success') {
            agentEls.consoleBody.textContent += detail + '\n';
          }
        }
      },
    });

    return result;
  }

  // Agent-aware send handler: delegates to agent for code requests, standard send otherwise
  async function handleSendWithAgent() {
    const text = els.promptInput.value.trim();
    if (!text && !runtime.pendingImages.length) return;

    // Determine if this should use the agent
    const useAgent = typeof MINDIAgent !== 'undefined' && isCodeRequest(text);

    // If we know auth is blocked, the agent loop would just call the API
    // multiple times and fail every iteration. Fall back to send(), which
    // now renders the friendly inline 'add your token' card instead.
    if (!useAgent || runtime.authBlocked) {
      return send();
    }

    // Agent mode
    const chat = ensureChat();
    const wasEmpty = chat.messages.length === 0;

    const userMsg = {
      role:    'user',
      content: text,
      images:  runtime.pendingImages.map((p) => p.dataUrl),
      ts:      Date.now(),
    };
    chat.messages.push(userMsg);
    chat.updatedAt = Date.now();

    if (wasEmpty) {
      chat.title = deriveTitle(text);
      els.chatTitle.textContent = chat.title;
    }

    const imageForApi = runtime.pendingImages[0]?.dataUrl || null;
    els.promptInput.value = '';
    autosizeTextarea();
    clearPendingImages();
    updateSendEnabled();

    // Show loading
    const loadingMsg = { role: 'assistant', content: '🤖 Agent working', loading: true, ts: Date.now() };
    chat.messages.push(loadingMsg);
    renderMessages();
    renderHistory();
    saveState();

    runtime.isSending = true;

    try {
      const agentResult = await runAgent(text, imageForApi);

      // Remove loading
      const idx = chat.messages.indexOf(loadingMsg);
      if (idx !== -1) chat.messages.splice(idx, 1);

      // Build response from agent
      const iterations = agentResult.iteration + 1;
      const status = agentResult.status === 'success' ? '✅' : '❌';
      let responseText = `${status} Agent completed in ${iterations} iteration(s).\n\n`;

      if (agentResult.currentCode) {
        const lang = agentResult.language || 'javascript';
        responseText += `\`\`\`${lang}\n${agentResult.currentCode}\n\`\`\``;
      } else {
        // Fallback: get the last generate step's detail
        const lastGen = [...agentResult.steps].reverse().find(s => s.type === 'generate' || s.type === 'fix');
        if (lastGen) responseText += lastGen.detail || 'No code generated.';
      }

      const assistantMsg = {
        role:     'assistant',
        content:  responseText,
        ts:       Date.now(),
      };
      chat.messages.push(assistantMsg);
      chat.updatedAt = Date.now();

      renderMessages();
      renderHistory();
      updatePreviewFromAssistant(assistantMsg);
      saveState();

    } catch (e) {
      const idx = chat.messages.indexOf(loadingMsg);
      if (idx !== -1) chat.messages.splice(idx, 1);

      const errorMsg = { role: 'assistant', content: `Agent error: ${e.message}`, ts: Date.now() };
      chat.messages.push(errorMsg);
      renderMessages();
      toast('Agent encountered an error', 'error');
    } finally {
      runtime.isSending = false;
    }
  }

  // ----------------------------------------------------------------
  // Init
  // ----------------------------------------------------------------
  function init() {
    bind();

    // Override the active send handler with agent-aware version
    if (typeof MINDIAgent !== 'undefined') {
      activeSend = handleSendWithAgent;
    }

    renderHistory();
    renderMessages();

    // Restore current chat title
    const c = currentChat();
    if (c) {
      els.chatTitle.textContent = c.title || 'New chat';
      const lastAssistant = [...c.messages].reverse().find((m) => m.role === 'assistant' && !m.loading);
      if (lastAssistant) updatePreviewFromAssistant(lastAssistant);
    }

    autosizeTextarea();
    updateSendEnabled();

    // Health check (after a tick so UI paints first)
    setTimeout(pingHealth, 80);

    // Periodically re-check health (every 60s)
    setInterval(pingHealth, 60_000);

    console.log('[MINDI] Agent system loaded:', typeof MINDIAgent !== 'undefined' ? '✅' : '❌');
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }
})();
