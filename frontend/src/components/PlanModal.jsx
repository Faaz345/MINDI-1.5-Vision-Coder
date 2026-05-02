import { useState } from 'react';

export default function PlanModal({ userPrompt, questions, onSubmit, onClose }) {
  const [answers, setAnswers] = useState(() => {
    const init = {};
    questions.forEach(q => { init[q.id] = q.default; });
    return init;
  });

  return (
    <div className="modal-overlay" onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="modal-card">
        <div className="modal-header">
          <h3>🎯 Configure Your Project</h3>
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>

        <div className="modal-body">
          <div style={{ padding: '10px 14px', background: 'rgba(124,58,237,.08)', borderRadius: 'var(--r-md)', border: '1px solid rgba(124,58,237,.15)', fontSize: '12px', color: 'var(--text-2)', lineHeight: '1.5' }}>
            <strong style={{ color: 'var(--text)' }}>Your prompt:</strong> {userPrompt}
          </div>

          {questions.map(q => (
            <div key={q.id} className="modal-field">
              <label>{q.question}</label>
              <div className="modal-options">
                {q.options.map(opt => (
                  <button
                    key={opt.value}
                    className={`modal-option ${answers[q.id] === opt.value ? 'selected' : ''}`}
                    onClick={() => setAnswers(prev => ({ ...prev, [q.id]: opt.value }))}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>

        <div className="modal-footer">
          <button className="modal-btn ghost" onClick={() => { onSubmit(userPrompt, {}); }}>
            Skip & Generate
          </button>
          <button className="modal-btn primary" onClick={() => { onSubmit(userPrompt, answers); }}>
            Generate ⚡
          </button>
        </div>
      </div>
    </div>
  );
}
