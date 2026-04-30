/* =============================================================
   MINDI Agent — Orchestrator
   Plan → Generate → Execute → Verify → Fix loop.
   Turns raw MINDI model into an autonomous coding agent.
   ============================================================= */

const MINDIAgent = (() => {
  'use strict';

  const MAX_RETRIES = 3;
  const STEP_TYPES = {
    PLAN:     'plan',
    GENERATE: 'generate',
    EXECUTE:  'execute',
    VERIFY:   'verify',
    FIX:      'fix',
    DONE:     'done',
    ERROR:    'error',
  };

  const STATUS = { PENDING: 'pending', RUNNING: 'running', SUCCESS: 'success', FAILED: 'failed' };

  // ── Agent state ────────────────────────────────────────
  function createRun() {
    return {
      id: 'run-' + Date.now().toString(36),
      steps: [],
      currentCode: null,
      language: null,
      iteration: 0,
      startTime: Date.now(),
      status: 'running',
    };
  }

  function addStep(run, type, status = STATUS.RUNNING, detail = '') {
    const step = {
      id: run.steps.length,
      type,
      status,
      detail,
      startTime: Date.now(),
      endTime: null,
    };
    run.steps.push(step);
    return step;
  }

  function completeStep(step, status, detail = '') {
    step.status = status;
    step.endTime = Date.now();
    if (detail) step.detail = detail;
  }

  // ── Prompt templates ───────────────────────────────────
  function planPrompt(userRequest) {
    return `Break this coding request into clear, numbered implementation steps (max 5 steps). Only list the steps, nothing else.

Request: ${userRequest}`;
  }

  function generatePrompt(userRequest, plan, previousCode, previousError) {
    let prompt = `Write COMPLETE, WORKING code for this request. Include ALL necessary HTML, CSS, and JavaScript in a single file. Do NOT leave any placeholders, TODOs, or "add more here" comments. Every feature must work.

Request: ${userRequest}`;

    if (plan) prompt += `\n\nPlan:\n${plan}`;

    if (previousCode && previousError) {
      prompt += `\n\nPrevious code had this error:\n${previousError}\n\nPrevious code:\n\`\`\`\n${previousCode}\n\`\`\`\n\nFix the error and return the COMPLETE corrected code.`;
    }

    return prompt;
  }

  function verifyPrompt(code, output, errors, screenshotDescription) {
    let prompt = `Review this code and its execution result. Is it working correctly?

Code:
\`\`\`
${code.slice(0, 3000)}
\`\`\`

Console output: ${output || '(none)'}
Errors: ${errors || '(none)'}`;

    if (screenshotDescription) {
      prompt += `\nScreenshot shows: ${screenshotDescription}`;
    }

    prompt += `\n\nRespond with either:
- "PASS" if the code works correctly
- "FAIL: <description of what's wrong>" if there are issues`;

    return prompt;
  }

  // ── Extract code from response ─────────────────────────
  function extractCode(response) {
    // Try fenced code blocks first
    const re = /```(\w+)?\s*\n([\s\S]*?)```/g;
    let last = null, m;
    while ((m = re.exec(response)) !== null) {
      last = { language: (m[1] || '').toLowerCase(), code: m[2] };
    }
    if (last) return last;

    // Try special tokens
    const codeMatch = response.match(/<\|code_start\|>([\s\S]*?)<\|code_end\|>/);
    if (codeMatch) {
      return { language: '', code: codeMatch[1].trim() };
    }

    return null;
  }

  // ── Main agent run ─────────────────────────────────────
  async function run(userPrompt, options = {}) {
    const {
      apiCall,          // async (prompt, image?) => {response, sections}
      sandboxContainer, // DOM element for iframe preview
      onStep,           // (run, step) => void — UI callback
      image = null,     // optional image for vision
    } = options;

    const agentRun = createRun();
    const notify = (step) => onStep && onStep(agentRun, step);

    try {
      // ── Step 1: PLAN ──────────────────────────────────
      const planStep = addStep(agentRun, STEP_TYPES.PLAN);
      notify(planStep);

      let plan = null;
      try {
        const planResult = await apiCall(planPrompt(userPrompt), image);
        plan = planResult.response;
        completeStep(planStep, STATUS.SUCCESS, plan.split('\n').filter(l => /^\d/.test(l.trim())).length + ' steps identified');
      } catch (e) {
        completeStep(planStep, STATUS.FAILED, e.message);
        // Continue without plan
      }
      notify(planStep);

      // ── Step 2+: GENERATE → EXECUTE → VERIFY → FIX loop
      let previousCode = null;
      let previousError = null;

      for (let iteration = 0; iteration <= MAX_RETRIES; iteration++) {
        agentRun.iteration = iteration;

        // ── GENERATE ──────────────────────────────────
        const genStep = addStep(agentRun, iteration === 0 ? STEP_TYPES.GENERATE : STEP_TYPES.FIX);
        genStep.detail = iteration === 0 ? 'Generating code...' : `Fixing (attempt ${iteration}/${MAX_RETRIES})...`;
        notify(genStep);

        let codeResult;
        try {
          const genResult = await apiCall(
            generatePrompt(userPrompt, plan, previousCode, previousError),
            image
          );
          codeResult = extractCode(genResult.response);

          if (!codeResult) {
            // No code block found — use entire response as code
            codeResult = { language: '', code: genResult.response };
          }

          agentRun.currentCode = codeResult.code;
          agentRun.language = codeResult.language || CodeSandbox.detectLanguage(codeResult.code);
          const lines = codeResult.code.split('\n').length;
          completeStep(genStep, STATUS.SUCCESS, `${lines} lines of ${agentRun.language}`);
        } catch (e) {
          completeStep(genStep, STATUS.FAILED, e.message);
          notify(genStep);
          break;
        }
        notify(genStep);

        // ── EXECUTE ───────────────────────────────────
        const execStep = addStep(agentRun, STEP_TYPES.EXECUTE);
        execStep.detail = `Running ${agentRun.language} code...`;
        notify(execStep);

        let execResult;
        try {
          execResult = await CodeSandbox.execute(
            codeResult.code,
            agentRun.language,
            sandboxContainer
          );

          const output = execResult.logs.join('\n') || '(no output)';
          if (execResult.success) {
            completeStep(execStep, STATUS.SUCCESS, `Ran in ${execResult.duration}ms — ${output.slice(0, 100)}`);
          } else {
            completeStep(execStep, STATUS.FAILED, execResult.errors.join('\n').slice(0, 200));
          }
        } catch (e) {
          execResult = { success: false, errors: [e.message], logs: [] };
          completeStep(execStep, STATUS.FAILED, e.message);
        }
        notify(execStep);

        // ── VERIFY ────────────────────────────────────
        if (execResult.success) {
          const verifyStep = addStep(agentRun, STEP_TYPES.VERIFY);
          verifyStep.detail = 'Checking output...';
          notify(verifyStep);

          // Try to take a screenshot for visual verification
          let screenshot = null;
          if (execResult.iframe && agentRun.language === 'html') {
            try {
              screenshot = await CodeSandbox.captureScreenshot(execResult.iframe);
            } catch { /* ignore */ }
          }

          // Simple check: if no errors and has output, consider it passing
          // For a more thorough check, we'd send screenshot back to MINDI
          const hasOutput = execResult.logs.length > 0 || agentRun.language === 'html';
          if (hasOutput) {
            completeStep(verifyStep, STATUS.SUCCESS, 'Code runs without errors ✓');
            notify(verifyStep);

            // DONE!
            const doneStep = addStep(agentRun, STEP_TYPES.DONE);
            completeStep(doneStep, STATUS.SUCCESS, `Completed in ${iteration + 1} iteration(s)`);
            agentRun.status = 'success';
            notify(doneStep);
            break;
          } else {
            completeStep(verifyStep, STATUS.FAILED, 'No output produced');
            previousCode = codeResult.code;
            previousError = 'Code produced no output';
            notify(verifyStep);
          }
        } else {
          // Execution failed — prepare for retry
          previousCode = codeResult.code;
          previousError = execResult.errors.join('\n');

          if (iteration === MAX_RETRIES) {
            const errStep = addStep(agentRun, STEP_TYPES.ERROR);
            completeStep(errStep, STATUS.FAILED, `Failed after ${MAX_RETRIES + 1} attempts: ${previousError.slice(0, 200)}`);
            agentRun.status = 'failed';
            notify(errStep);
          }
        }
      }
    } catch (e) {
      const errStep = addStep(agentRun, STEP_TYPES.ERROR);
      completeStep(errStep, STATUS.FAILED, e.message);
      agentRun.status = 'failed';
      notify(errStep);
    }

    agentRun.endTime = Date.now();
    return agentRun;
  }

  return { run, STEP_TYPES, STATUS, extractCode };
})();

if (typeof module !== 'undefined') module.exports = MINDIAgent;
