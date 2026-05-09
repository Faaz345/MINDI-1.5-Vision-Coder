# Codex Prompt: Wire MINDIGENOUS 2.0 Frontend to MINDI 1.5 Backend

## Your Goal
Connect the MINDIGENOUS 2.0 React frontend to the live MINDI 1.5 backend API. Replace all local `/api/*` stub calls with real API calls to the HuggingFace Space.

## Project Location
You are working in: `d:\Desktop 31st Jan 2026\MINDIGENOUS 2.0\`
This is a Vite + React + Tailwind + TypeScript project.

## Backend API Details

### API URL
```
https://mindigenous-mindi-chat.hf.space
```

### Auth Token (HF Token)
```
# Get from .env.local or HF account settings
# Do NOT commit tokens to git
```
Send as: `Authorization: Bearer <token>` header on every request.

### Protocol: Gradio 5.x SSE v3
The backend is a Gradio 5.x Space on HuggingFace ZeroGPU. It uses a two-step SSE protocol:

**Step 1 — Submit:**
```
POST https://mindigenous-mindi-chat.hf.space/gradio_api/call/chat_fn
Headers:
  Content-Type: application/json
  Authorization: Bearer $HF_TOKEN
Body:
  { "data": [prompt, imageArg, temperature, maxTokens, historyJson] }
Response:
  { "event_id": "abc123" }
```

- `prompt`: string — the user's message
- `imageArg`: `null` OR a Gradio FileData object `{path: "/tmp/gradio/...", meta: {_type: "gradio.FileData"}}`
- `temperature`: number (e.g. 0.7)
- `maxTokens`: number (e.g. 2048)
- `historyJson`: string — JSON-encoded chat history `[{role, content}, ...]` or `""`

**Image upload (required before Step 1 if user attached an image):**
```
POST https://mindigenous-mindi-chat.hf.space/gradio_api/upload
Headers: Authorization: Bearer <token>
Body: multipart/form-data with field `files` containing the image blob
Response: ["/tmp/gradio/.../image.png"] — use this path in the imageArg above
```

**Step 2 — Stream Result:**
```
GET https://mindigenous-mindi-chat.hf.space/gradio_api/call/chat_fn/{event_id}
Headers: Authorization: Bearer <token>
Response: text/event-stream (SSE)
```

Parse SSE lines:
- `event: complete` → next line starts with `data: [...]` → the response is inside
- `event: error` → error occurred
- The `data:` payload is a JSON array; index 0 is the response string
- Response format: `{"response": "...markdown text...", "sections": {"thinking": [...], "code": [...]}}`

### Expected Response Format
```json
{
  "response": "Here's your code:\n\n```tsx\nexport default function Page() { ... }\n```",
  "sections": {
    "thinking": ["User wants a Next.js landing page..."],
    "code": ["Generated page.tsx with hero section"],
    "critique": [],
    "fix": [],
    "error": [],
    "suggest": [],
    "file": []
  }
}
```

The `response` field contains markdown with fenced code blocks. Extract the last ` ```lang\n...\n``` ` block for the code preview.

---

## Files to Modify

### 1. `vite.config.js` — Remove or update the proxy

**Current:**
```javascript
server: {
  proxy: {
    "/api": {
      target: "http://127.0.0.1:8000",
      changeOrigin: true,
    },
  },
},
```

**Change to:** Remove the proxy entirely. The frontend will call HF Space directly using full URLs.

### 2. `src/lib/mindiApi.ts` — Rewrite API calls

**Current:** Calls local `/api/workflow`, `/api/chat`, `/api/web-search`.

**Replace with:**

Create a `HfSpaceClient` class or functions that:
1. Store the API base URL and HF token
2. Provide `submitChat(prompt, image?, temperature, maxTokens, history?)`:
   - If image provided, upload via `/gradio_api/upload` first
   - Build the `data` array: `[prompt, imageArg, temp, maxTokens, historyJson]`
   - POST to `/gradio_api/call/chat_fn`
   - Return `event_id`
3. Provide `streamResult(eventId, onChunk, onDone, onError)`:
   - GET `/gradio_api/call/chat_fn/{event_id}`
   - Parse SSE stream
   - Extract the complete event data
   - Parse JSON response
   - Call callbacks
4. Export a convenience `sendMessage()` that does both steps

**Key implementation notes:**
- Use `fetch()` with `ReadableStream` for SSE parsing
- The SSE response from Gradio doesn't use standard `event:` prefix in some versions — may just be lines starting with `data:`
- On HF Spaces, CORS is handled — `mode: 'cors'` should work
- Add timeout handling (first request can take 4+ min for cold GPU start)
- Handle `Quota exceeded` / `Unlogged user` errors gracefully

### 3. `src/hooks/useAgentStream.ts` — Adapt to new API

**Current:** Uses `streamWorkflow()` from `mindiApi.ts`.

**Keep the hook interface** (`isStreaming`, `streamText`, `events`, `startWorkflow`, etc.) but change the internal call from `streamWorkflow()` to the new HF Space chat function.

**Adaptation mapping:**
- `WorkflowRequest.prompt` → chat `prompt`
- `WorkflowRequest.project_id` → ignore (not used by HF Space)
- `WorkflowRequest.files` → ignore for now
- `onToken(text)` → append text from `response` field as it streams (note: HF Space is NOT streaming, it's a single response. So you'll get the full response at once. You can simulate streaming by tokenizing the response string and yielding chunks.)
- `onDone(data)` → call with the parsed response object
- `onError` → handle quota/auth errors

**Important:** The HF Space does NOT stream tokens. It returns the complete response in one SSE complete event. To maintain the streaming UX:
1. Receive the full response
2. Split by words
3. Yield words with a small delay (e.g. 15ms/word) to simulate typing
4. This keeps the UI feeling interactive

### 4. `src/App.jsx` — Connect the chat component

**Current:** Imports `VercelV0Chat` from `@/components/ui/v0-ai-chat`.

**Ensure:** The chat component's `onSubmit` handler calls `startWorkflow()` from `useAgentStream()`, which now uses the HF Space API.

**No changes needed if** the chat component already calls the hook correctly. Just verify the data flows through.

### 5. Handle Demo Mode / Missing Checkpoints

**Critical:** The MINDI 1.5 model checkpoints are MISSING from HuggingFace (confirmed 2026-05-09). The HF Space currently:
- Has the code and tokenizer
- Does NOT have the trained LoRA weights
- Will either return a "quota exceeded" error OR fall back to demo responses

**Add a `DEMO_MODE` flag:**
```typescript
const DEMO_MODE = true; // Set false when checkpoints are restored
```

When `DEMO_MODE = true`:
- Skip the HF API call entirely
- Return mock responses based on prompt keywords
- Include realistic markdown with code blocks
- This lets you test the UI without burning GPU quota

Add mock responses for:
- "Next.js" / "landing page" → return TSX hero component
- "React" → return JSX component
- "HTML" → return full HTML document
- "Python" → return Python function
- Default → return a friendly message about demo mode

### 6. Add a Settings Panel for API Config

Add to the UI (or a `.env` file):
- API URL field (default: `https://mindigenous-mindi-chat.hf.space`)
- HF Token field (default: the token above)
- Demo mode toggle
- Temperature slider (0.0 - 1.0)
- Max tokens slider (256 - 4096)

Store these in `localStorage` so they persist across reloads.

---

## Implementation Order

1. **Step 1:** Create `src/lib/hfSpaceClient.ts` with the Gradio 5.x API adapter
2. **Step 2:** Update `src/lib/mindiApi.ts` to use the new client (or replace it)
3. **Step 3:** Update `src/hooks/useAgentStream.ts` to call the new API
4. **Step 4:** Add DEMO_MODE with mock responses
5. **Step 5:** Add settings panel for API config
6. **Step 6:** Update `vite.config.js` — remove proxy
7. **Step 7:** Test with a prompt like "Build a Next.js landing page"

---

## Code Snippets for Reference

### HF Space Chat Function (from old frontend)
```javascript
// From MINDI 1.5 vision-coder/frontend/app.js
const API_DEFAULT = 'https://mindigenous-mindi-chat.hf.space';

async function callGenerate(prompt, image, signal) {
  const base = API_DEFAULT;
  const history = []; // Build from chat history
  const historyJson = history.length ? JSON.stringify(history) : '';
  
  // Upload image if present
  let imageArg = null;
  if (image && image.startsWith('data:')) {
    const filePath = await uploadImageToGradio(base, image, signal);
    imageArg = { path: filePath, meta: { _type: 'gradio.FileData' } };
  }
  
  // Step 1: Submit
  const submitRes = await fetch(`${base}/gradio_api/call/chat_fn`, {
    method: 'POST',
    headers: { 
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${process.env.VITE_HF_TOKEN || ''}`
    },
    body: JSON.stringify({
      data: [prompt, imageArg, 0.7, 2048, historyJson],
    }),
    signal,
  });
  const { event_id } = await submitRes.json();
  
  // Step 2: Stream
  const resultRes = await fetch(`${base}/gradio_api/call/chat_fn/${event_id}`, {
    headers: { 'Authorization': `Bearer ${process.env.VITE_HF_TOKEN || ''}` },
    signal,
  });
  const sseText = await resultRes.text();
  // Parse SSE lines for "event: complete" and extract data
}
```

### Demo Response (from old frontend)
```javascript
const DEMO_RESPONSES = [
  {
    match: /landing|hero|next\.?js/i,
    response: `Here's a clean Next.js landing page:\n\n\`\`\`tsx\n// app/page.tsx\nexport default function Home() { ... }\n\`\`\``,
    sections: { thinking: [...], code: [...] },
  },
  // ... more patterns
];
```

---

## Testing Checklist

After wiring, test these flows:

- [ ] Type "hello" → should get a response (demo mode or real)
- [ ] Type "Build a Next.js landing page" → should return TSX code block
- [ ] The code block should appear in the chat UI with syntax highlighting
- [ ] The code should also appear in the preview/code panel
- [ ] Attach an image → should upload to HF Space first, then send prompt
- [ ] Clear chat history → should reset local state
- [ ] Change temperature in settings → should persist in localStorage
- [ ] Toggle demo mode → should switch between mock and real API

---

## Notes

- **Do NOT** expose the HF token in committed code. Use `.env` + `import.meta.env`.
- The current `vite.config.js` has `proxy: {"/api": ...}` — remove this.
- The `mindiApi.ts` types (`MindiStreamEvent`, `MindiEvent`, etc.) are good — keep them.
- The `useAgentStream` hook interface is good — keep it, just change the internal API call.
- For demo mode, reuse the `DEMO_RESPONSES` array from the old project (`MINDI 1.5 vision-coder/frontend/app.js`).

## Context Files for Reference

If you need to see the old implementation:
- `d:\Desktop 31st Jan 2026\MINDI 1.5 vision-coder\frontend\app.js` — old vanilla JS frontend with full API logic
- `d:\Desktop 31st Jan 2026\MINDI 1.5 vision-coder\hf_space\app.py` — backend with SYSTEM_MSG and chat_fn
