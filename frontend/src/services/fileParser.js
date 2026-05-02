/* File Parser — Extracts files from model response */

export function parseFiles(responseText) {
  if (!responseText) return [];
  const files = [];
  const re = /```(\w+)?\s*\n([\s\S]*?)```/g;
  let m, idx = 0;
  while ((m = re.exec(responseText)) !== null) {
    const lang = (m[1] || '').toLowerCase();
    const code = m[2];
    const filename = detectFilename(code, lang, idx);
    files.push({ id: `file-${idx}`, path: filename, content: code, language: lang || detectLang(code) });
    idx++;
  }
  if (files.length === 0 && responseText.trim()) {
    files.push({ id: 'file-0', path: 'index.html', content: responseText, language: 'html' });
  }
  return files;
}

function detectFilename(code, lang, idx) {
  // Check for filename comments
  const fnMatch = code.match(/^\/\/\s*([\w/.-]+\.\w+)/m) || code.match(/^<!--\s*([\w/.-]+\.\w+)/m) || code.match(/^\/\*\s*([\w/.-]+\.\w+)/m);
  if (fnMatch) return fnMatch[1];
  const extMap = { html: 'index.html', markup: 'index.html', css: 'styles.css', javascript: 'script.js', js: 'script.js', typescript: 'index.ts', tsx: 'page.tsx', jsx: 'App.jsx', python: 'main.py', json: 'package.json', vue: 'App.vue' };
  if (idx === 0 && /<!doctype|<html/i.test(code)) return 'index.html';
  return extMap[lang] || `file${idx}.${lang || 'txt'}`;
}

function detectLang(code) {
  const t = code.trim();
  if (/^<!doctype|^<html/i.test(t)) return 'html';
  if (/^import.*from|^export|^const |^function /m.test(t)) return 'javascript';
  if (/^from |^import |^def |^class /m.test(t)) return 'python';
  if (/^\{[\s\S]*\}$/.test(t)) return 'json';
  return 'plaintext';
}

export function buildPreviewHTML(files) {
  const htmlFile = files.find(f => f.language === 'html' || f.path.endsWith('.html'));
  if (htmlFile) return htmlFile.content;
  const cssFile = files.find(f => f.language === 'css');
  const jsFile = files.find(f => f.language === 'javascript' || f.language === 'js');
  if (cssFile || jsFile) {
    return `<!DOCTYPE html><html><head><meta charset="UTF-8"><style>${cssFile?.content || ''}</style></head><body><script>${jsFile?.content || ''}<\/script></body></html>`;
  }
  return null;
}

export function getFileIcon(path) {
  const ext = path.split('.').pop().toLowerCase();
  const icons = { html: '🌐', css: '🎨', js: '⚡', jsx: '⚛️', tsx: '⚛️', ts: '📘', py: '🐍', json: '📋', vue: '💚', md: '📝', svg: '🖼️' };
  return icons[ext] || '📄';
}
