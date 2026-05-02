import { useState } from 'react';

export default function SettingsModal({ settings, onSave, onClose }) {
  const [form, setForm] = useState({ ...settings });

  const update = (key, val) => setForm(prev => ({ ...prev, [key]: val }));

  return (
    <div className="modal-overlay" onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="modal-card">
        <div className="modal-header">
          <h3>⚙️ Settings</h3>
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>

        <div className="modal-body">
          <div className="settings-field">
            <label>API Endpoint</label>
            <input className="settings-input" type="url" value={form.apiUrl} onChange={e => update('apiUrl', e.target.value)} placeholder="https://mindigenous-mindi-chat.hf.space" />
            <div className="settings-hint">HuggingFace Space or custom API URL</div>
          </div>

          <div className="settings-field">
            <label>HuggingFace Token</label>
            <input className="settings-input" type="password" value={form.hfToken} onChange={e => update('hfToken', e.target.value)} placeholder="hf_..." />
            <div className="settings-hint">Required for ZeroGPU access. Get one at huggingface.co/settings/tokens</div>
          </div>

          <div className="settings-field">
            <label>Temperature</label>
            <div className="settings-range-wrap">
              <input className="settings-range" type="range" min="0" max="2" step="0.05" value={form.temperature} onChange={e => update('temperature', parseFloat(e.target.value))} />
              <span className="settings-range-val">{Number(form.temperature).toFixed(2)}</span>
            </div>
          </div>

          <div className="settings-field">
            <label>Max Tokens</label>
            <div className="settings-range-wrap">
              <input className="settings-range" type="range" min="128" max="4096" step="128" value={form.maxTokens} onChange={e => update('maxTokens', parseInt(e.target.value))} />
              <span className="settings-range-val">{form.maxTokens}</span>
            </div>
          </div>
        </div>

        <div className="modal-footer">
          <button className="modal-btn ghost" onClick={onClose}>Cancel</button>
          <button className="modal-btn primary" onClick={() => onSave(form)}>Save Settings</button>
        </div>
      </div>
    </div>
  );
}
