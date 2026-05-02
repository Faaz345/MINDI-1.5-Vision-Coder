import { getFileIcon } from '../services/fileParser';

export default function Editor({ file, codeLines, isGenerating, generationProgress, files, activeFile, onFileSelect }) {
  if (!file && !isGenerating) {
    return (
      <div className="editor">
        <div className="editor-welcome">
          <div className="welcome-logo">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M12 2L2 7l10 5 10-5-10-5z" />
              <path d="M2 17l10 5 10-5" />
              <path d="M2 12l10 5 10-5" />
            </svg>
          </div>
          <h1 className="welcome-title">
            What do you want to <span className="grad-text">build</span>?
          </h1>
          <p className="welcome-sub">
            Describe your website, app, or component and MINDI will generate production-ready code with live preview.
          </p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', justifyContent: 'center', maxWidth: '400px' }}>
            {['Landing Page', 'Dashboard', 'Portfolio', 'E-commerce'].map(tag => (
              <span key={tag} style={{
                padding: '6px 14px', borderRadius: '20px', fontSize: '11px',
                background: 'rgba(124,58,237,.1)', border: '1px solid rgba(124,58,237,.2)',
                color: 'var(--purple-light)', fontWeight: 500,
              }}>{tag}</span>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="editor">
      {/* Tabs */}
      {files.length > 0 && (
        <div className="editor-tabs">
          {files.map(f => (
            <button
              key={f.id}
              className={`editor-tab ${activeFile === f.id ? 'active' : ''}`}
              onClick={() => onFileSelect(f.id)}
            >
              <span className="tab-icon">{getFileIcon(f.path)}</span>
              {f.path}
            </button>
          ))}
        </div>
      )}

      {/* Generating indicator */}
      {isGenerating && (
        <div className="generating-indicator">
          <div className="gen-spinner" />
          <span>{generationProgress || 'Generating...'}</span>
        </div>
      )}

      {/* Code */}
      <div className="editor-content">
        {codeLines.map((line, i) => (
          <div key={line.id} className="code-line" style={{ animationDelay: `${Math.min(i * 12, 600)}ms` }}>
            <span className="line-number">{i + 1}</span>
            <span className="line-content">{line.text}</span>
          </div>
        ))}
        {isGenerating && codeLines.length === 0 && (
          <div style={{ padding: '20px 64px', color: 'var(--text-dim)', fontFamily: 'var(--mono)', fontSize: '12px' }}>
            Waiting for code...
          </div>
        )}
      </div>
    </div>
  );
}
