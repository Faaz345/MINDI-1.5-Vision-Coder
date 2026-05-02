import { useRef, useEffect } from 'react';

export default function Preview({ html, consoleOutput, isGenerating }) {
  const iframeRef = useRef(null);

  useEffect(() => {
    if (iframeRef.current && html) {
      iframeRef.current.srcdoc = html;
    }
  }, [html]);

  return (
    <div className="preview-panel">
      <div className="preview-header">
        <div className="preview-title">
          {html && <span className="preview-dot" />}
          <span>Preview</span>
        </div>
        <div className="preview-actions">
          {html && (
            <button className="preview-btn" title="Open in new tab" onClick={() => {
              const w = window.open('', '_blank');
              if (w) { w.document.open(); w.document.write(html); w.document.close(); }
            }}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6" />
                <polyline points="15 3 21 3 21 9" />
                <line x1="10" y1="14" x2="21" y2="3" />
              </svg>
            </button>
          )}
          {html && (
            <button className="preview-btn" title="Copy HTML" onClick={() => {
              navigator.clipboard.writeText(html).catch(() => {});
            }}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="9" y="9" width="13" height="13" rx="2" />
                <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" />
              </svg>
            </button>
          )}
        </div>
      </div>

      <div className="preview-frame-wrap">
        {html ? (
          <iframe
            ref={iframeRef}
            className="preview-frame"
            title="Live Preview"
            sandbox="allow-scripts allow-same-origin"
          />
        ) : (
          <div className="preview-empty">
            <div className="preview-empty-icon">
              {isGenerating ? '⏳' : '👁️'}
            </div>
            <p>{isGenerating ? 'Generating preview...' : 'Live Preview'}</p>
            <p>{isGenerating ? 'Your website will appear here' : 'Generated code will render here in real-time'}</p>
          </div>
        )}
      </div>

      <div className="console-panel">
        <div className="console-header">Console</div>
        <div className="console-body">
          {consoleOutput.length === 0 ? (
            <span style={{ color: 'var(--text-dim)' }}>No output yet</span>
          ) : (
            consoleOutput.map((entry, i) => (
              <div key={i} className={entry.type === 'error' ? 'error' : ''}>
                {entry.text}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
