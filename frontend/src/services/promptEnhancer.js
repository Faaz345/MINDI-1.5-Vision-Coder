/* Prompt Enhancer — Transforms user input into structured prompts */

const TECH_CONFIGS = {
  html: { label: 'HTML + CSS + JS', instructions: 'Output a SINGLE complete <!DOCTYPE html> document. Include ALL CSS in <style> and JS in <script>. Use Tailwind CDN when appropriate.' },
  react: { label: 'React', instructions: 'Output a single React JSX component with export default. Use hooks when needed.' },
  nextjs: { label: 'Next.js', instructions: 'Output a single app/page.tsx. Use client directive only if needed. Use Tailwind classes.' },
  vue: { label: 'Vue', instructions: 'Output a single .vue SFC with template, script setup, style scoped.' },
};

const DESIGN_PRESETS = {
  dark: 'Dark theme with deep navy backgrounds, subtle gradients, glassmorphism.',
  light: 'Light theme with clean white/gray backgrounds, subtle shadows.',
  gradient: 'Rich gradient backgrounds, purple-to-blue or teal-to-emerald.',
  minimal: 'Minimalist with whitespace, clean typography, elegant simplicity.',
};

export function analyzePrompt(userInput) {
  const input = userInput.toLowerCase();
  const questions = [];
  const hasTech = /\b(html|react|next\.?js|vue|svelte|tailwind)\b/i.test(input);
  if (!hasTech) {
    questions.push({ id: 'tech', question: 'Tech stack?', options: Object.entries(TECH_CONFIGS).map(([k, v]) => ({ value: k, label: v.label })), default: 'html' });
  }
  const hasTheme = /\b(dark|light|gradient|minimal)\b/i.test(input);
  if (!hasTheme) {
    questions.push({ id: 'theme', question: 'Design style?', options: Object.entries(DESIGN_PRESETS).map(([k]) => ({ value: k, label: k.charAt(0).toUpperCase() + k.slice(1) })), default: 'dark' });
  }
  return { questions, hasTech, hasTheme, detectedTech: detectTech(input) };
}

function detectTech(input) {
  if (/next\.?js/i.test(input)) return 'nextjs';
  if (/\breact\b/i.test(input)) return 'react';
  if (/\bvue\b/i.test(input)) return 'vue';
  return 'html';
}

export function enhancePrompt(userInput, answers = {}) {
  const tech = answers.tech || detectTech(userInput);
  const theme = answers.theme || 'dark';
  const config = TECH_CONFIGS[tech] || TECH_CONFIGS.html;
  const design = DESIGN_PRESETS[theme] || DESIGN_PRESETS.dark;
  return `${userInput}\n\n--- REQUIREMENTS ---\nTech: ${config.label}\n${config.instructions}\nDesign: ${design}\nRules: Production-ready, responsive, Inter font, smooth animations, no placeholders, complete code.`;
}

export function getQuickEnhancement(userInput) {
  return enhancePrompt(userInput, {});
}
