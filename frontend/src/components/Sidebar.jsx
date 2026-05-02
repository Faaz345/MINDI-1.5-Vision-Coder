import { getFileIcon } from '../services/fileParser';

const STEP_ICONS = { plan: '📋', enhance: '✨', generate: '⚡', parse: '📦', preview: '👁️', done: '🎉', stop: '⏹️' };
const STEP_LABELS = { plan: 'Planning', enhance: 'Enhancing', generate: 'Generating', parse: 'Parsing', preview: 'Previewing', done: 'Complete', stop: 'Stopped' };

export default function Sidebar({ files, activeFile, onFileSelect, agentSteps, status, isGenerating, onSettingsOpen }) {
  return (
    <aside className="sidebar">
      <button className="sidebar-brand" onClick={onSettingsOpen} title="Settings">
        <div className="brand-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 2L2 7l10 5 10-5-10-5z" /><path d="M2 17l10 5 10-5" /><path d="M2 12l10 5 10-5" /></svg>
        </div>
        <div className="brand-info">
          <div className="brand-name">MINDI<span className="grad-text"> 1.5</span></div>
          <div className="brand-sub">VISION-CODER • AI BUILDER</div>
        </div>
      </button>

      {/* File Tree */}
      <div className="sidebar-section">
        <div className="sidebar-section-title">
          {files.length > 0 ? `Files (${files.length})` : 'Project'}
        </div>
        <div className="file-tree">
          {files.length === 0 ? (
            <div style={{ padding: '8px 16px', fontSize: '11px', color: 'var(--text-dim)' }}>
              {isGenerating ? 'Generating files...' : 'No files yet. Describe what to build.'}
            </div>
          ) : (
            files.map((f, i) => (
              <button
                key={f.id}
                className={`file-item ${activeFile === f.id ? 'active' : ''}`}
                onClick={() => onFileSelect(f.id)}
                style={{ animationDelay: `${i * 80}ms` }}
              >
                <span className="file-icon">{getFileIcon(f.path)}</span>
                <span className="file-name">{f.path}</span>
              </button>
            ))
          )}
        </div>
      </div>

      {/* Agent Steps */}
      {agentSteps.length > 0 && (
        <div className="sidebar-section" style={{ borderTop: '1px solid var(--border)', paddingTop: '8px' }}>
          <div className="sidebar-section-title">Agent Progress</div>
          <div className="agent-steps">
            {agentSteps.map((step, i) => (
              <div key={step.id} className="agent-step" style={{ animationDelay: `${i * 60}ms` }}>
                <div className={`step-icon ${step.status}`}>
                  {step.status === 'running' ? '⏳' : step.status === 'success' ? '✅' : step.status === 'failed' ? '❌' : (STEP_ICONS[step.type] || '⏺')}
                </div>
                <div>
                  <div style={{ fontWeight: 500, color: 'var(--text)' }}>{STEP_LABELS[step.type] || step.type}</div>
                  <div className="step-detail">{step.detail}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div style={{ flex: 1 }} />

      {/* Footer */}
      <div className="sidebar-footer">
        <div className={`status-dot ${status}`} />
        <span>{status === 'online' ? 'MINDI · Connected' : status === 'demo' ? 'Demo Mode' : 'Connecting...'}</span>
      </div>
    </aside>
  );
}
