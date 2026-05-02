import { useState, useCallback, useRef, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import Editor from './components/Editor';
import Preview from './components/Preview';
import PromptBar from './components/PromptBar';
import PlanModal from './components/PlanModal';
import SettingsModal from './components/SettingsModal';
import Toasts from './components/Toasts';
import { callMINDI, generateDemo, isQuotaError, isQuotaException, pingAPI } from './services/api';
import { analyzePrompt, enhancePrompt, getQuickEnhancement } from './services/promptEnhancer';
import { parseFiles, buildPreviewHTML } from './services/fileParser';
import './App.css';

const STORAGE_KEY = 'mindi.builder.v1';

function loadSettings() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {};
  } catch { return {}; }
}

export default function App() {
  const saved = loadSettings();
  const [settings, setSettings] = useState({
    apiUrl: saved.apiUrl || 'https://mindigenous-mindi-chat.hf.space',
    hfToken: saved.hfToken || '',
    temperature: saved.temperature ?? 0.7,
    maxTokens: saved.maxTokens ?? 2048,
  });
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [planModal, setPlanModal] = useState(null);
  const [toasts, setToasts] = useState([]);
  const [status, setStatus] = useState('connecting');
  const [files, setFiles] = useState([]);
  const [activeFile, setActiveFile] = useState(null);
  const [previewHTML, setPreviewHTML] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationProgress, setGenerationProgress] = useState('');
  const [agentSteps, setAgentSteps] = useState([]);
  const [codeLines, setCodeLines] = useState([]);
  const [consoleOutput, setConsoleOutput] = useState([]);
  const [history, setHistory] = useState([]);
  const abortRef = useRef(null);

  const addToast = useCallback((msg, type = 'info', ms = 3000) => {
    const id = Date.now() + Math.random();
    setToasts(t => [...t, { id, msg, type }]);
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), ms);
  }, []);

  const saveSettings = useCallback((s) => {
    setSettings(s);
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(s)); } catch {}
  }, []);

  // Health check
  useEffect(() => {
    const check = async () => {
      const ok = await pingAPI(settings.apiUrl, settings.hfToken);
      setStatus(ok ? 'online' : 'demo');
    };
    check();
    const iv = setInterval(check, 60000);
    return () => clearInterval(iv);
  }, [settings.apiUrl, settings.hfToken]);

  const addAgentStep = useCallback((type, detail, status = 'running') => {
    const step = { id: Date.now(), type, detail, status, time: new Date() };
    setAgentSteps(prev => [...prev, step]);
    return step.id;
  }, []);

  const updateAgentStep = useCallback((id, updates) => {
    setAgentSteps(prev => prev.map(s => s.id === id ? { ...s, ...updates } : s));
  }, []);

  // Animate code appearing line by line
  const animateCode = useCallback((code, fileList) => {
    const lines = code.split('\n');
    setCodeLines([]);
    let i = 0;
    const interval = setInterval(() => {
      if (i < lines.length) {
        setCodeLines(prev => [...prev, { text: lines[i], id: i, visible: true }]);
        i++;
      } else {
        clearInterval(interval);
      }
    }, 15); // ~15ms per line for smooth animation
    return () => clearInterval(interval);
  }, []);

  // Main generate function
  const handleGenerate = useCallback(async (userPrompt, skipPlan = false) => {
    if (!userPrompt.trim() || isGenerating) return;

    // Analyze prompt for planning
    if (!skipPlan) {
      const analysis = analyzePrompt(userPrompt);
      if (analysis.questions.length > 0) {
        setPlanModal({ userPrompt, questions: analysis.questions });
        return;
      }
    }

    setIsGenerating(true);
    setAgentSteps([]);
    setConsoleOutput([]);
    setCodeLines([]);
    setFiles([]);
    setPreviewHTML(null);
    abortRef.current = new AbortController();

    // Step 1: Plan
    const planId = addAgentStep('plan', 'Analyzing your request...');
    setGenerationProgress('Planning...');
    await new Promise(r => setTimeout(r, 400));
    updateAgentStep(planId, { status: 'success', detail: 'Requirements analyzed' });

    // Step 2: Enhance prompt
    const enhanceId = addAgentStep('enhance', 'Enhancing prompt for best output...');
    const enhanced = getQuickEnhancement(userPrompt);
    await new Promise(r => setTimeout(r, 300));
    updateAgentStep(enhanceId, { status: 'success', detail: 'Prompt optimized' });

    // Step 3: Generate
    const genId = addAgentStep('generate', 'Generating code with MINDI 1.5...');
    setGenerationProgress('Generating code...');

    let result;
    try {
      if (status === 'demo' || !settings.apiUrl) {
        result = await generateDemo(userPrompt);
      } else {
        result = await callMINDI({
          prompt: enhanced,
          temperature: settings.temperature,
          maxTokens: settings.maxTokens,
          history,
          hfToken: settings.hfToken,
          apiUrl: settings.apiUrl,
          signal: abortRef.current.signal,
        });
      }

      if (isQuotaError(result)) {
        updateAgentStep(genId, { status: 'failed', detail: 'GPU quota — using demo fallback' });
        addToast('GPU quota exceeded — showing demo. Add HF token in Settings for real generation.', 'error', 5000);
        result = await generateDemo(userPrompt);
      }

      updateAgentStep(genId, { status: 'success', detail: `Response received (${(result.response || '').length} chars)` });

      // Step 4: Parse files
      const parseId = addAgentStep('parse', 'Extracting files...');
      const parsedFiles = parseFiles(result.response);
      setFiles(parsedFiles);
      if (parsedFiles.length > 0) {
        setActiveFile(parsedFiles[0].id);
        // Animate the code
        animateCode(parsedFiles[0].content, parsedFiles);
      }
      updateAgentStep(parseId, { status: 'success', detail: `${parsedFiles.length} file(s) extracted` });

      // Step 5: Preview
      const previewId = addAgentStep('preview', 'Rendering preview...');
      const html = buildPreviewHTML(parsedFiles);
      if (html) {
        setPreviewHTML(html);
        updateAgentStep(previewId, { status: 'success', detail: 'Preview rendered' });
        setConsoleOutput(prev => [...prev, { type: 'log', text: '✓ Page rendered successfully' }]);
      } else {
        updateAgentStep(previewId, { status: 'success', detail: 'No HTML to preview' });
      }

      // Update history
      setHistory(prev => [
        ...prev.slice(-18),
        { role: 'user', content: userPrompt },
        { role: 'assistant', content: result.response },
      ]);

      // Done
      addAgentStep('done', 'Generation complete!', 'success');
      setGenerationProgress('');

    } catch (err) {
      updateAgentStep(genId, { status: 'failed', detail: err.message });
      addToast(`Error: ${err.message}`, 'error');

      // Fallback to demo
      try {
        result = await generateDemo(userPrompt);
        const parsedFiles = parseFiles(result.response);
        setFiles(parsedFiles);
        if (parsedFiles.length > 0) {
          setActiveFile(parsedFiles[0].id);
          animateCode(parsedFiles[0].content, parsedFiles);
        }
        const html = buildPreviewHTML(parsedFiles);
        if (html) setPreviewHTML(html);
        addAgentStep('done', 'Demo response used as fallback', 'success');
      } catch {}
    }

    setIsGenerating(false);
    setGenerationProgress('');
  }, [isGenerating, settings, status, history, addToast, addAgentStep, updateAgentStep, animateCode]);

  const handlePlanSubmit = useCallback((userPrompt, answers) => {
    setPlanModal(null);
    const enhanced = enhancePrompt(userPrompt, answers);
    // Re-call generate with the enhanced prompt, skipping plan
    setIsGenerating(true);
    setAgentSteps([]);
    setConsoleOutput([]);
    setCodeLines([]);
    setFiles([]);
    setPreviewHTML(null);

    (async () => {
      const genId = addAgentStep('generate', 'Generating with your preferences...');
      setGenerationProgress('Generating...');
      let result;
      try {
        if (status === 'demo') {
          result = await generateDemo(userPrompt);
        } else {
          result = await callMINDI({ prompt: enhanced, temperature: settings.temperature, maxTokens: settings.maxTokens, history, hfToken: settings.hfToken, apiUrl: settings.apiUrl });
        }
        if (isQuotaError(result)) {
          updateAgentStep(genId, { status: 'failed', detail: 'GPU quota — using demo' });
          addToast('GPU quota exceeded — showing demo.', 'error', 4000);
          result = await generateDemo(userPrompt);
        }
        updateAgentStep(genId, { status: 'success', detail: 'Code generated' });
        const parsedFiles = parseFiles(result.response);
        setFiles(parsedFiles);
        if (parsedFiles.length > 0) { setActiveFile(parsedFiles[0].id); animateCode(parsedFiles[0].content, parsedFiles); }
        const html = buildPreviewHTML(parsedFiles);
        if (html) setPreviewHTML(html);
        setHistory(prev => [...prev.slice(-18), { role: 'user', content: userPrompt }, { role: 'assistant', content: result.response }]);
        addAgentStep('done', 'Complete!', 'success');
      } catch (err) {
        updateAgentStep(genId, { status: 'failed', detail: err.message });
        // Fallback to demo on any error
        try {
          if (isQuotaException(err.message)) {
            addToast('GPU quota exceeded — showing demo.', 'error', 4000);
          }
          result = await generateDemo(userPrompt);
          const parsedFiles = parseFiles(result.response);
          setFiles(parsedFiles);
          if (parsedFiles.length > 0) { setActiveFile(parsedFiles[0].id); animateCode(parsedFiles[0].content, parsedFiles); }
          const html = buildPreviewHTML(parsedFiles);
          if (html) { setPreviewHTML(html); setConsoleOutput(prev => [...prev, { type: 'log', text: '✓ Demo preview rendered' }]); }
          addAgentStep('done', 'Demo fallback used', 'success');
        } catch {}
      }
      setIsGenerating(false);
      setGenerationProgress('');
    })();
  }, [settings, status, history, addToast, addAgentStep, updateAgentStep, animateCode]);

  const handleFileSelect = useCallback((fileId) => {
    setActiveFile(fileId);
    const file = files.find(f => f.id === fileId);
    if (file) {
      setCodeLines(file.content.split('\n').map((text, i) => ({ text, id: i, visible: true })));
    }
  }, [files]);

  const handleStop = useCallback(() => {
    abortRef.current?.abort();
    setIsGenerating(false);
    setGenerationProgress('');
    addAgentStep('stop', 'Generation stopped by user', 'failed');
  }, [addAgentStep]);

  const activeFileData = files.find(f => f.id === activeFile);

  return (
    <div className="app-shell">
      <div className="ambient">
        <div className="grid-pattern" />
        <div className="blob blob--purple" />
        <div className="blob blob--blue" />
      </div>

      <Sidebar
        files={files}
        activeFile={activeFile}
        onFileSelect={handleFileSelect}
        agentSteps={agentSteps}
        status={status}
        isGenerating={isGenerating}
        onSettingsOpen={() => setSettingsOpen(true)}
      />

      <main className="main-area">
        <Editor
          file={activeFileData}
          codeLines={codeLines}
          isGenerating={isGenerating}
          generationProgress={generationProgress}
          files={files}
          activeFile={activeFile}
          onFileSelect={handleFileSelect}
        />

        <PromptBar
          onSubmit={handleGenerate}
          onStop={handleStop}
          isGenerating={isGenerating}
          generationProgress={generationProgress}
          status={status}
        />
      </main>

      <Preview
        html={previewHTML}
        consoleOutput={consoleOutput}
        isGenerating={isGenerating}
      />

      {planModal && (
        <PlanModal
          userPrompt={planModal.userPrompt}
          questions={planModal.questions}
          onSubmit={handlePlanSubmit}
          onClose={() => setPlanModal(null)}
        />
      )}

      {settingsOpen && (
        <SettingsModal
          settings={settings}
          onSave={(s) => { saveSettings(s); setSettingsOpen(false); addToast('Settings saved', 'success'); }}
          onClose={() => setSettingsOpen(false)}
        />
      )}

      <Toasts toasts={toasts} />
    </div>
  );
}
