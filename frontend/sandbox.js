/* =============================================================
   MINDI Agent — Code Sandbox
   Executes code in isolated environments (iframe for HTML/JS, 
   Pyodide for Python). Captures output, errors, and screenshots.
   ============================================================= */

const CodeSandbox = (() => {
  'use strict';

  // ── Pyodide loader ─────────────────────────────────────
  let pyodideInstance = null;
  let pyodideLoading = false;

  async function loadPyodide() {
    if (pyodideInstance) return pyodideInstance;
    if (pyodideLoading) {
      // Wait for existing load
      while (pyodideLoading) await new Promise(r => setTimeout(r, 200));
      return pyodideInstance;
    }
    pyodideLoading = true;
    try {
      // Load Pyodide from CDN
      if (!window.loadPyodide) {
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js';
        document.head.appendChild(script);
        await new Promise((res, rej) => { script.onload = res; script.onerror = rej; });
      }
      pyodideInstance = await window.loadPyodide();
      console.log('[Sandbox] Pyodide loaded');
      return pyodideInstance;
    } finally {
      pyodideLoading = false;
    }
  }

  // ── HTML/JS execution in sandboxed iframe ──────────────
  function executeHTML(code, containerEl) {
    return new Promise((resolve) => {
      const logs = [];
      const errors = [];
      const startTime = Date.now();

      // Create sandboxed iframe
      let iframe = containerEl.querySelector('.sandbox-iframe');
      if (iframe) iframe.remove();

      iframe = document.createElement('iframe');
      iframe.className = 'sandbox-iframe';
      iframe.sandbox = 'allow-scripts allow-modals';
      iframe.style.cssText = 'width:100%;height:100%;border:none;background:#fff;border-radius:8px;';
      containerEl.appendChild(iframe);

      // Inject console capture + error handling into the code
      const wrappedCode = `
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<script>
  // Override console to send messages to parent
  const _origLog = console.log;
  const _origErr = console.error;
  const _origWarn = console.warn;
  
  function _send(type, args) {
    try {
      parent.postMessage({ 
        type: 'sandbox-' + type, 
        data: Array.from(args).map(a => {
          try { return typeof a === 'object' ? JSON.stringify(a, null, 2) : String(a); }
          catch { return String(a); }
        }).join(' ')
      }, '*');
    } catch {}
  }
  
  console.log = function() { _origLog.apply(console, arguments); _send('log', arguments); };
  console.error = function() { _origErr.apply(console, arguments); _send('error', arguments); };
  console.warn = function() { _origWarn.apply(console, arguments); _send('warn', arguments); };
  
  window.onerror = function(msg, src, line, col, err) {
    _send('error', [msg + ' (line ' + line + ')']);
    return true;
  };
  window.addEventListener('unhandledrejection', function(e) {
    _send('error', ['Unhandled promise rejection: ' + e.reason]);
  });
  
  // Signal ready after load
  window.addEventListener('load', function() {
    _send('ready', ['Page loaded in ' + (performance.now()|0) + 'ms']);
  });
</script>
</head>
<body>
${code.includes('<body') ? code.replace(/.*<body[^>]*>/is, '').replace(/<\/body>.*/is, '') : (code.includes('<html') ? '' : code)}
</body>
</html>`;

      // If code is a full HTML document, use it directly with console injection
      const finalCode = code.includes('<!DOCTYPE') || code.includes('<html')
        ? code.replace('<head>', `<head>
<script>
  const _origLog = console.log;
  const _origErr = console.error;
  function _send(t, a) { try { parent.postMessage({type:'sandbox-'+t,data:Array.from(a).map(x=>{try{return typeof x==='object'?JSON.stringify(x,null,2):String(x)}catch{return String(x)}}).join(' ')},'*'); } catch{} }
  console.log = function() { _origLog.apply(console,arguments); _send('log',arguments); };
  console.error = function() { _origErr.apply(console,arguments); _send('error',arguments); };
  window.onerror = function(m,s,l) { _send('error',[m+' (line '+l+')']); return true; };
  window.addEventListener('load', function() { _send('ready', ['loaded']); });
</script>`)
        : wrappedCode;

      // Listen for messages from iframe
      const handler = (event) => {
        if (!event.data || !event.data.type) return;
        const { type, data } = event.data;
        if (type === 'sandbox-log') logs.push(data);
        else if (type === 'sandbox-error') errors.push(data);
        else if (type === 'sandbox-warn') logs.push(`[warn] ${data}`);
        else if (type === 'sandbox-ready') {
          // Give a moment for rendering, then resolve
          setTimeout(() => {
            window.removeEventListener('message', handler);
            resolve({
              success: errors.length === 0,
              logs,
              errors,
              duration: Date.now() - startTime,
              iframe,
            });
          }, 500);
        }
      };
      window.addEventListener('message', handler);

      // Set content
      iframe.srcdoc = finalCode;

      // Timeout fallback (10 seconds)
      setTimeout(() => {
        window.removeEventListener('message', handler);
        resolve({
          success: errors.length === 0,
          logs,
          errors: errors.length ? errors : ['Timeout: page did not signal ready within 10s'],
          duration: Date.now() - startTime,
          iframe,
        });
      }, 10000);
    });
  }

  // ── JavaScript execution ───────────────────────────────
  function executeJS(code) {
    return new Promise((resolve) => {
      const logs = [];
      const errors = [];
      const startTime = Date.now();

      // Create a sandboxed execution context
      const origLog = console.log;
      const origErr = console.error;

      console.log = (...args) => {
        logs.push(args.map(a => typeof a === 'object' ? JSON.stringify(a, null, 2) : String(a)).join(' '));
      };
      console.error = (...args) => {
        errors.push(args.map(String).join(' '));
      };

      try {
        // Execute in indirect eval (global scope)
        const result = (0, eval)(code);
        if (result !== undefined) logs.push(String(result));
      } catch (e) {
        errors.push(`${e.name}: ${e.message}`);
      } finally {
        console.log = origLog;
        console.error = origErr;
      }

      resolve({
        success: errors.length === 0,
        logs,
        errors,
        duration: Date.now() - startTime,
      });
    });
  }

  // ── Python execution via Pyodide ───────────────────────
  async function executePython(code) {
    const logs = [];
    const errors = [];
    const startTime = Date.now();

    try {
      const pyodide = await loadPyodide();

      // Redirect stdout/stderr
      pyodide.runPython(`
import sys, io
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()
`);

      try {
        await pyodide.runPythonAsync(code);
      } catch (e) {
        errors.push(String(e));
      }

      // Capture output
      const stdout = pyodide.runPython('sys.stdout.getvalue()');
      const stderr = pyodide.runPython('sys.stderr.getvalue()');
      if (stdout) logs.push(stdout);
      if (stderr) errors.push(stderr);

      // Reset streams
      pyodide.runPython(`
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
`);
    } catch (e) {
      errors.push(`Pyodide error: ${e.message}`);
    }

    return {
      success: errors.length === 0,
      logs,
      errors,
      duration: Date.now() - startTime,
    };
  }

  // ── Screenshot capture ─────────────────────────────────
  async function captureScreenshot(iframe) {
    try {
      if (!window.html2canvas) {
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js';
        document.head.appendChild(script);
        await new Promise((res, rej) => { script.onload = res; script.onerror = rej; });
      }
      const canvas = await html2canvas(iframe, { useCORS: true, scale: 1 });
      return canvas.toDataURL('image/png');
    } catch (e) {
      console.warn('[Sandbox] Screenshot failed:', e);
      return null;
    }
  }

  // ── Language detection ─────────────────────────────────
  function detectLanguage(code) {
    const t = code.trim();
    if (/^<!doctype|^<html|^<div|^<section|^<main/i.test(t)) return 'html';
    if (/^<style|^[.#]?\w+\s*\{/.test(t)) return 'css';
    if (/^(import|from|def |class |print\(|if __name__)/m.test(t)) return 'python';
    if (/^(import |export |const |function |class |let |var |=>)/m.test(t)) return 'javascript';
    if (/^(package |func |type |import ")/m.test(t)) return 'go';
    if (/^(use |fn |let |struct |impl |pub )/m.test(t)) return 'rust';
    return 'javascript'; // default
  }

  // ── Main execute function ──────────────────────────────
  async function execute(code, language = null, containerEl = null) {
    const lang = language || detectLanguage(code);

    switch (lang) {
      case 'html':
      case 'markup':
      case 'css':
        if (!containerEl) {
          return { success: false, errors: ['No container for HTML preview'], logs: [], duration: 0 };
        }
        return executeHTML(code, containerEl);

      case 'python':
        return executePython(code);

      case 'javascript':
      case 'typescript':
      case 'js':
      case 'ts':
        return executeJS(code);

      default:
        return {
          success: false,
          logs: [],
          errors: [`Language "${lang}" execution not supported in browser. Supported: HTML, JavaScript, Python.`],
          duration: 0,
        };
    }
  }

  return { execute, executeHTML, executeJS, executePython, detectLanguage, captureScreenshot };
})();

// Export for module usage
if (typeof module !== 'undefined') module.exports = CodeSandbox;
