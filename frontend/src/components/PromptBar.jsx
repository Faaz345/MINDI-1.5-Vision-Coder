import { useState, useRef, useEffect } from 'react';

export default function PromptBar({ onSubmit, onStop, isGenerating, generationProgress, status }) {
  const [input, setInput] = useState('');
  const textareaRef = useRef(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 120) + 'px';
    }
  }, [input]);

  const handleSubmit = () => {
    if (!input.trim() || isGenerating) return;
    onSubmit(input.trim());
    setInput('');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="prompt-bar">
      <div className="prompt-input-wrap">
        <textarea
          ref={textareaRef}
          className="prompt-input"
          placeholder="Describe what you want to build..."
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={1}
          disabled={isGenerating}
        />
        {isGenerating ? (
          <button className="prompt-stop" onClick={onStop} title="Stop generation">
            <svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2" /></svg>
          </button>
        ) : (
          <button
            className="prompt-send"
            onClick={handleSubmit}
            disabled={!input.trim()}
            title="Generate"
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          </button>
        )}
      </div>
      <div className="prompt-footer">
        <span>
          {status === 'demo' ? '⚠ Demo mode · add HF token in Settings' : 'MINDI 1.5 Vision-Coder'}
        </span>
        <span>
          {isGenerating ? generationProgress : 'Shift+Enter for new line'}
        </span>
      </div>
    </div>
  );
}
