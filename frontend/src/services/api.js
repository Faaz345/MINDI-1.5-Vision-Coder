/* =============================================================
   MINDI API Service — Gradio SSE v3 integration
   Connects to HuggingFace-hosted MINDI 1.5 Vision-Coder
   ============================================================= */

const API_DEFAULT = 'https://mindigenous-mindi-chat.hf.space';

function authHeaders(hfToken, extra = {}) {
  const h = { ...extra };
  if (hfToken) h['Authorization'] = `Bearer ${hfToken}`;
  return h;
}

function dataUrlToBlob(dataUrl) {
  const match = /^data:([^;]+);base64,(.+)$/.exec(dataUrl || '');
  if (!match) throw new Error('Invalid image data URL');
  const bytes = Uint8Array.from(atob(match[2]), c => c.charCodeAt(0));
  return { blob: new Blob([bytes], { type: match[1] }), mime: match[1] };
}

async function uploadImageToGradio(base, dataUrl, hfToken, signal) {
  const { blob, mime } = dataUrlToBlob(dataUrl);
  const ext = (mime.split('/')[1] || 'png').replace('+xml', '').split(';')[0];
  const filename = `mindi-upload-${Date.now()}.${ext}`;
  const formData = new FormData();
  formData.append('files', blob, filename);

  const headers = authHeaders(hfToken);
  delete headers['Content-Type'];

  const res = await fetch(`${base}/gradio_api/upload`, {
    method: 'POST', headers, body: formData, signal,
  });
  if (!res.ok) throw new Error(`Image upload ${res.status}`);
  const result = await res.json();
  const filePath = Array.isArray(result) ? result[0] : result?.files?.[0];
  if (!filePath) throw new Error('Upload failed');
  return filePath;
}

export async function callMINDI({ prompt, image, temperature = 0.7, maxTokens = 2048, history = [], hfToken = '', apiUrl = API_DEFAULT, signal }) {
  const base = (apiUrl || API_DEFAULT).replace(/\/$/, '');
  const isGradio = base.includes('hf.space') || base.includes('huggingface.co');
  const historyJson = history.length ? JSON.stringify(history) : '';

  if (isGradio) {
    let imageArg = null;
    if (image && image.startsWith('data:')) {
      try {
        const filePath = await uploadImageToGradio(base, image, hfToken, signal);
        imageArg = { path: filePath, meta: { _type: 'gradio.FileData' }, orig_name: filePath.split('/').pop() };
      } catch { imageArg = null; }
    }

    const submitRes = await fetch(`${base}/gradio_api/call/chat_fn`, {
      method: 'POST',
      headers: authHeaders(hfToken, { 'Content-Type': 'application/json' }),
      body: JSON.stringify({ data: [prompt, imageArg, temperature, maxTokens, historyJson] }),
      signal,
    });
    if (!submitRes.ok) {
      const txt = await submitRes.text().catch(() => '');
      throw new Error(`API ${submitRes.status}: ${txt.slice(0, 200)}`);
    }
    const { event_id } = await submitRes.json();
    if (!event_id) throw new Error('No event_id returned');

    const resultRes = await fetch(`${base}/gradio_api/call/chat_fn/${event_id}`, {
      method: 'GET', headers: authHeaders(hfToken), signal,
    });
    if (!resultRes.ok) throw new Error(`API result ${resultRes.status}`);

    const sseText = await resultRes.text();
    const lines = sseText.split('\n');
    for (let i = 0; i < lines.length; i++) {
      if (lines[i].startsWith('event: complete')) {
        const dataLine = lines[i + 1];
        if (dataLine?.startsWith('data: ')) {
          try {
            const parsed = JSON.parse(dataLine.slice(6));
            const raw = Array.isArray(parsed) ? parsed[0] : parsed;
            try { return JSON.parse(raw); } catch { return { response: String(raw), sections: {} }; }
          } catch { return { response: dataLine.slice(6), sections: {} }; }
        }
        break;
      }
      if (lines[i].startsWith('event: error')) {
        const errMsg = lines[i + 1]?.startsWith('data: ') ? lines[i + 1].slice(6) : 'Gradio error';
        throw new Error(errMsg.slice(0, 300));
      }
    }
    throw new Error('No complete event in response');
  } else {
    const body = { prompt, temperature, max_tokens: maxTokens, history };
    if (image) body.image = image;
    const res = await fetch(`${base}/api/generate`, {
      method: 'POST',
      headers: authHeaders(hfToken, { 'Content-Type': 'application/json', 'Accept': 'application/json' }),
      body: JSON.stringify(body), signal,
    });
    if (!res.ok) throw new Error(`API ${res.status}`);
    return res.json();
  }
}

export async function pingAPI(apiUrl, hfToken) {
  const base = (apiUrl || API_DEFAULT).replace(/\/$/, '');
  try {
    const res = await fetch(base, { method: 'HEAD', mode: 'no-cors' }).catch(() => null);
    return !!res;
  } catch { return false; }
}

export function isQuotaError(result) {
  if (!result) return false;
  const text = String(result.response || '');
  const errs = result.sections?.error || [];
  const blob = (text + ' ' + errs.join(' ')).toLowerCase();
  return /zerogpu|gpu quota|out of .* quota|exceeded .* quota|unlogged user|gpu task aborted|task aborted/.test(blob);
}

export function isQuotaException(errMessage) {
  const msg = (errMessage || '').toLowerCase();
  return /gpu quota|zerogpu|gpu task aborted|task aborted|unlogged user|out of .* quota|exceeded .* quota/.test(msg);
}

// Demo responses
const DEMOS = [
  {
    match: /landing|hero|page|website/i,
    response: `Here's a complete landing page:\n\n\`\`\`html\n<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n<meta name="viewport" content="width=device-width, initial-scale=1.0">\n<title>Lumina — Future of Design</title>\n<script src="https://cdn.tailwindcss.com"><\/script>\n<style>\n@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');\nbody { font-family: 'Inter', sans-serif; }\n.gradient-bg { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }\n.glow { box-shadow: 0 0 40px rgba(124, 58, 237, 0.3); }\n.card-hover:hover { transform: translateY(-4px); box-shadow: 0 20px 40px rgba(0,0,0,0.3); }\n</style>\n</head>\n<body class="gradient-bg text-white min-h-screen">\n<nav class="flex items-center justify-between px-8 py-5 max-w-7xl mx-auto">\n  <div class="text-2xl font-bold bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">Lumina</div>\n  <div class="hidden md:flex gap-8 text-sm text-gray-300">\n    <a href="#features" class="hover:text-white transition">Features</a>\n    <a href="#pricing" class="hover:text-white transition">Pricing</a>\n    <a href="#about" class="hover:text-white transition">About</a>\n  </div>\n  <button class="px-5 py-2 bg-purple-600 rounded-full text-sm font-medium hover:bg-purple-500 transition glow">Get Started</button>\n</nav>\n<main class="max-w-7xl mx-auto px-8">\n  <section class="py-24 text-center">\n    <span class="inline-block px-4 py-1.5 bg-purple-500/20 border border-purple-500/30 rounded-full text-purple-300 text-xs font-medium tracking-wider uppercase mb-6">Now in Beta</span>\n    <h1 class="text-5xl md:text-7xl font-extrabold leading-tight mb-6">\n      Build faster.<br>\n      <span class="bg-gradient-to-r from-purple-400 via-pink-400 to-blue-400 bg-clip-text text-transparent">Ship smarter.</span>\n    </h1>\n    <p class="text-lg text-gray-400 max-w-2xl mx-auto mb-10">The next-generation platform that turns your ideas into reality. No complexity, just results.</p>\n    <div class="flex justify-center gap-4">\n      <button class="px-8 py-3 bg-gradient-to-r from-purple-600 to-blue-600 rounded-full font-semibold hover:shadow-lg hover:shadow-purple-500/25 transition-all">Start Free Trial</button>\n      <button class="px-8 py-3 border border-white/20 rounded-full font-medium hover:bg-white/5 transition">Watch Demo</button>\n    </div>\n  </section>\n  <section id="features" class="py-20 grid md:grid-cols-3 gap-6">\n    <div class="p-8 bg-white/5 border border-white/10 rounded-2xl card-hover transition-all">\n      <div class="w-12 h-12 bg-purple-500/20 rounded-xl flex items-center justify-center text-2xl mb-4">⚡</div>\n      <h3 class="text-lg font-semibold mb-2">Lightning Fast</h3>\n      <p class="text-gray-400 text-sm">Deploy in seconds. Our edge network ensures your app loads instantly worldwide.</p>\n    </div>\n    <div class="p-8 bg-white/5 border border-white/10 rounded-2xl card-hover transition-all">\n      <div class="w-12 h-12 bg-blue-500/20 rounded-xl flex items-center justify-center text-2xl mb-4">🔒</div>\n      <h3 class="text-lg font-semibold mb-2">Enterprise Security</h3>\n      <p class="text-gray-400 text-sm">SOC 2 compliant with end-to-end encryption. Your data is always protected.</p>\n    </div>\n    <div class="p-8 bg-white/5 border border-white/10 rounded-2xl card-hover transition-all">\n      <div class="w-12 h-12 bg-pink-500/20 rounded-xl flex items-center justify-center text-2xl mb-4">🎨</div>\n      <h3 class="text-lg font-semibold mb-2">Beautiful UI</h3>\n      <p class="text-gray-400 text-sm">Pre-built components that look stunning out of the box. Customize everything.</p>\n    </div>\n  </section>\n</main>\n<footer class="border-t border-white/10 py-8 text-center text-gray-500 text-sm">\n  <p>&copy; 2026 Lumina. Crafted with AI.</p>\n</footer>\n</body>\n</html>\n\`\`\``,
  },
  {
    match: /dashboard|chart|analytics|admin/i,
    response: `Here's a dashboard UI:\n\n\`\`\`html\n<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">\n<title>Dashboard</title>\n<style>\n:root{--bg:#0b0b14;--panel:#14141f;--border:rgba(255,255,255,.08);--text:#ececf1;--mute:#8b94a7;--acc:#7c3aed}\n*{box-sizing:border-box;margin:0;padding:0}\nbody{background:var(--bg);color:var(--text);font:14px/1.55 'Inter',sans-serif;min-height:100vh;display:grid;grid-template-columns:240px 1fr}\naside{background:var(--panel);border-right:1px solid var(--border);padding:20px}\naside h1{font-size:18px;background:linear-gradient(135deg,#7c3aed,#2563eb);-webkit-background-clip:text;color:transparent;margin-bottom:24px}\nnav a{display:block;padding:10px 12px;border-radius:8px;color:var(--mute);text-decoration:none;margin-bottom:2px}\nnav a.active{background:rgba(124,58,237,.15);color:#fff}\nmain{padding:24px;overflow-y:auto}\n.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:20px}\n.stat{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:16px}\n.stat .v{font-size:24px;font-weight:600;margin-top:6px}\n.stat .l{color:var(--mute);font-size:12px;text-transform:uppercase;letter-spacing:.1em}\n.chart{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:18px;height:260px;display:flex;align-items:end;gap:8px}\n.bar{flex:1;background:linear-gradient(180deg,#7c3aed,#2563eb);border-radius:6px 6px 0 0;transition:height .5s}\n</style>\n</head>\n<body>\n<aside><h1>Pulsegrid</h1>\n<nav><a class="active">Overview</a><a>Customers</a><a>Revenue</a><a>Settings</a></nav>\n</aside>\n<main>\n<div class="stats">\n<div class="stat"><div class="l">Revenue</div><div class="v">$48,210</div></div>\n<div class="stat"><div class="l">Users</div><div class="v">12,840</div></div>\n<div class="stat"><div class="l">Conversion</div><div class="v">4.2%</div></div>\n<div class="stat"><div class="l">Churn</div><div class="v">1.1%</div></div>\n</div>\n<div class="chart">\n<div class="bar" style="height:40%"></div><div class="bar" style="height:65%"></div>\n<div class="bar" style="height:30%"></div><div class="bar" style="height:80%"></div>\n<div class="bar" style="height:55%"></div><div class="bar" style="height:90%"></div>\n<div class="bar" style="height:70%"></div>\n</div>\n</main>\n</body>\n</html>\n\`\`\``,
  },
];

const DEFAULT_DEMO = {
  response: `Here's a starter template:\n\n\`\`\`html\n<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">\n<title>MINDI Generated</title>\n<style>\n*{margin:0;padding:0;box-sizing:border-box}\nbody{min-height:100vh;background:#0f0c29;color:#fff;font-family:Inter,sans-serif;display:grid;place-items:center}\n.card{text-align:center;padding:48px;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.1);border-radius:20px;backdrop-filter:blur(10px)}\nh1{font-size:2.5rem;margin-bottom:12px;background:linear-gradient(135deg,#7c3aed,#2563eb);-webkit-background-clip:text;color:transparent}\np{color:#a0a0b8;font-size:1.1rem}\n</style>\n</head>\n<body>\n<div class="card">\n<h1>Hello from MINDI</h1>\n<p>Describe what you want to build and I'll generate it.</p>\n</div>\n</body>\n</html>\n\`\`\``,
  sections: {},
};

export async function generateDemo(prompt) {
  await new Promise(r => setTimeout(r, 800 + Math.random() * 600));
  const found = DEMOS.find(d => d.match.test(prompt));
  return { response: (found || DEFAULT_DEMO).response, sections: {} };
}
