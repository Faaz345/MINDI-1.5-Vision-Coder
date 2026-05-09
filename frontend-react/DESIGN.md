---
version: "alpha"
name: "MINDIGENOUS Matrix Builder"
description: "A cybernetic AI web-builder interface with a matrix-code atmosphere, dark translucent work surfaces, emerald/cyan accents, and a prompt-first creation flow."
colors:
  background: "#05060A"
  on-background: "#F5F7FB"
  surface: "#0D1018"
  surface-dim: "#05070B"
  surface-bright: "#17221F"
  surface-container-lowest: "#040609"
  surface-container-low: "#05080C"
  surface-container: "#0E1716"
  surface-container-high: "#10221A"
  surface-container-highest: "#1A2B29"
  on-surface: "#F1F7F5"
  on-surface-variant: "#C9D1DD"
  outline: "#255047"
  outline-variant: "#1A413B"
  primary: "#8FF7E6"
  on-primary: "#06110A"
  primary-container: "#10B981"
  on-primary-container: "#EFFFF9"
  secondary: "#22D3EE"
  on-secondary: "#06120F"
  secondary-container: "#0E7490"
  on-secondary-container: "#E6FDFF"
  tertiary: "#A7F3BF"
  on-tertiary: "#062015"
  error: "#F87171"
  on-error: "#1B0707"
  brand-start: "#79F4BD"
  brand-middle: "#D8FFF1"
  brand-end: "#6EEFFF"
  text-muted: "#9BA4B5"
  text-dim: "#6F7888"
  code-text: "#D9E2EF"
  preview-canvas: "#FFFFFF"
typography:
  display:
    fontFamily: "Inter, ui-sans-serif, system-ui, sans-serif"
    fontSize: 34px
    fontWeight: "720"
    lineHeight: 37px
    letterSpacing: "0"
  title:
    fontFamily: "Inter, ui-sans-serif, system-ui, sans-serif"
    fontSize: 15px
    fontWeight: "720"
    lineHeight: 18px
    letterSpacing: "0"
  body:
    fontFamily: "Inter, ui-sans-serif, system-ui, sans-serif"
    fontSize: 13px
    fontWeight: "400"
    lineHeight: 20px
    letterSpacing: "0"
  input:
    fontFamily: "Inter, ui-sans-serif, system-ui, sans-serif"
    fontSize: 16px
    fontWeight: "400"
    lineHeight: 26px
    letterSpacing: "0"
  input-compact:
    fontFamily: "Inter, ui-sans-serif, system-ui, sans-serif"
    fontSize: 14px
    fontWeight: "400"
    lineHeight: 22px
    letterSpacing: "0"
  label:
    fontFamily: "Inter, ui-sans-serif, system-ui, sans-serif"
    fontSize: 12px
    fontWeight: "720"
    lineHeight: 16px
    letterSpacing: "0"
  wordmark:
    fontFamily: "BankGothic Md BT, Eurostile, Arial Black, Impact, sans-serif"
    fontSize: 24px
    fontWeight: "900"
    lineHeight: 23px
    letterSpacing: "0"
  wordmark-compact:
    fontFamily: "BankGothic Md BT, Eurostile, Arial Black, Impact, sans-serif"
    fontSize: 15px
    fontWeight: "900"
    lineHeight: 14px
    letterSpacing: "0"
  code:
    fontFamily: "SFMono-Regular, Consolas, Liberation Mono, monospace"
    fontSize: 12.5px
    fontWeight: "400"
    lineHeight: 21px
    letterSpacing: "0"
spacing:
  unit: 8px
  xs: 4px
  sm: 8px
  md: 12px
  lg: 14px
  xl: 16px
  "2xl": 18px
  "3xl": 22px
  "4xl": 24px
  "5xl": 28px
  prompt-padding-x: 24px
  prompt-padding-y: 12px
  panel-padding: 16px
  workspace-padding: 14px
  workspace-gap: 14px
  drawer-gap: 14px
  icon-gap: 8px
rounded:
  xs: 8px
  sm: 10px
  md: 12px
  lg: 14px
  DEFAULT: 16px
  prompt: 24px
  xl: 24px
  "2xl": 28px
  full: 9999px
radii:
  icon-button: "{rounded.md}"
  prompt-tool-button: "{rounded.DEFAULT}"
  prompt-card: "{rounded.prompt}"
  prompt-card-compact: "{rounded.DEFAULT}"
  workspace-panel: "{rounded.xl}"
  sidebar: "{rounded.2xl}"
  editor-window: "{rounded.xl}"
  preview-frame: 18px
shadows:
  prompt: "0 24px 90px rgba(0, 0, 0, 0.48)"
  prompt-elevated: "0 46px 132px rgba(0, 0, 0, 0.72), 0 20px 48px rgba(0, 0, 0, 0.48)"
  workspace-panel: "0 24px 84px rgba(0, 0, 0, 0.36)"
  sidebar: "0 24px 72px rgba(0, 0, 0, 0.34)"
  transition-frame: "0 0 28px rgba(85, 242, 214, 0.18), 0 36px 126px rgba(0, 0, 0, 0.66)"
  toast: "0 24px 90px rgba(0, 0, 0, 0.48)"
elevation:
  base:
    backgroundColor: "{colors.background}"
    textColor: "{colors.on-background}"
  floating-prompt:
    backgroundColor: "{colors.surface-container}"
    textColor: "{colors.on-surface}"
    rounded: "{rounded.prompt}"
    padding: "{spacing.prompt-padding-x}"
  workspace-panel:
    backgroundColor: "{colors.surface-container-low}"
    textColor: "{colors.on-surface}"
    rounded: "{rounded.xl}"
    padding: "{spacing.panel-padding}"
  overlay-drawer:
    backgroundColor: "{colors.surface-container-lowest}"
    textColor: "{colors.on-surface}"
    rounded: "{rounded.xl}"
    padding: "{spacing.panel-padding}"
motion:
  instant: 180ms
  fast: 240ms
  menu-expand: 280ms
  layout-shift: 440ms
  prompt-enter: 480ms
  morph-frame: 720ms
  return-home: 880ms
  workspace-settle: 980ms
  easing-standard: "ease-in-out"
  easing-emphasized: "cubic-bezier(0.22, 1, 0.36, 1)"
  easing-morph: "cubic-bezier(0.76, 0, 0.24, 1)"
components:
  app-background:
    backgroundColor: "{colors.background}"
    textColor: "{colors.on-background}"
  wordmark:
    textColor: "{colors.brand-middle}"
    typography: "{typography.wordmark}"
    rounded: "{rounded.xs}"
    padding: 4px 7px 7px
  prompt-card:
    backgroundColor: "{colors.surface-container}"
    textColor: "{colors.on-surface}"
    typography: "{typography.input}"
    rounded: "{rounded.prompt}"
    padding: 12px 16px
  prompt-card-compact:
    backgroundColor: "{colors.surface-container}"
    textColor: "{colors.on-surface}"
    typography: "{typography.input-compact}"
    rounded: "{rounded.DEFAULT}"
    padding: 10px 16px
  prompt-tool-cluster:
    backgroundColor: "{colors.surface-container-low}"
    textColor: "{colors.text-muted}"
    rounded: "{rounded.DEFAULT}"
    padding: 4px
  prompt-tool-button:
    backgroundColor: transparent
    textColor: "{colors.text-muted}"
    rounded: "{rounded.DEFAULT}"
    size: 36px
  prompt-tool-button-hover:
    backgroundColor: "{colors.surface-container-highest}"
    textColor: "{colors.on-surface}"
    rounded: "{rounded.DEFAULT}"
    size: 36px
  send-button:
    backgroundColor: "{colors.primary-container}"
    textColor: "{colors.on-primary}"
    rounded: "{rounded.md}"
    size: 44px
  workspace-panel:
    backgroundColor: "{colors.surface-container-low}"
    textColor: "{colors.on-surface}"
    rounded: "{rounded.xl}"
    padding: "{spacing.panel-padding}"
  chat-message:
    backgroundColor: "{colors.surface-container-high}"
    textColor: "{colors.on-surface-variant}"
    typography: "{typography.body}"
    rounded: "{rounded.DEFAULT}"
    padding: 12px
  preview-frame:
    backgroundColor: "{colors.preview-canvas}"
    textColor: "#07080B"
    rounded: 18px
    padding: 12px
  sidebar:
    backgroundColor: "{colors.surface-container-lowest}"
    textColor: "{colors.on-surface}"
    rounded: "{rounded.2xl}"
    padding: 14px
  file-row:
    backgroundColor: transparent
    textColor: "{colors.text-muted}"
    typography: "{typography.body}"
    rounded: "{rounded.lg}"
    height: 34px
    padding: 0 9px
  file-row-active:
    backgroundColor: "{colors.surface-container-highest}"
    textColor: "{colors.on-surface}"
    typography: "{typography.body}"
    rounded: "{rounded.lg}"
    height: 34px
    padding: 0 9px
  code-editor:
    backgroundColor: "{colors.surface-dim}"
    textColor: "{colors.code-text}"
    typography: "{typography.code}"
    rounded: 0 0 30px 30px
    padding: 18px
  transition-frame:
    backgroundColor: transparent
    textColor: "{colors.primary}"
    rounded: "{rounded.2xl}"
    padding: 0
---

## Overview

MINDIGENOUS is a prompt-first AI web-builder interface with a cybernetic, cinematic identity. The visual system combines a full-screen matrix-code background, a dark center fade, and translucent emerald/cyan glass panels. The first impression should feel like a focused build console rather than a marketing page: one wordmark, one question, one prompt box.

The product moves from a quiet home state into a workspace through a visible morphing border animation. The prompt box is the origin of the experience. It should feel like the workspace grows from that input, not like the user is sent to a separate page.

## Colors

The palette is almost entirely dark graphite, black-green, emerald, and cyan. The matrix background provides energy, while the interface surfaces restrain that energy with tinted panels and strong center vignettes.

- **Background:** Use near-black `#05060A` as the foundation. Matrix symbols should be visible but sit behind dark radial fades so the prompt remains readable.
- **Surfaces:** Use green-black surfaces from `#05080C` through `#10221A`, never plain gray. Panels should feel technical, dampened, and slightly glassy.
- **Text:** Use `#F5F7FB` or `#F1F7F5` for primary text. Use muted blue-gray values for labels, captions, and helper text.
- **Accent:** Emerald and cyan are reserved for send actions, borders, active states, wordmark highlights, and technical status details. Avoid adding extra accent families.
- **Preview Canvas:** Generated website previews can render on white or their own internal theme; they should remain framed inside the darker builder shell.

## Typography

The core UI uses Inter for clarity and dense tool readability. The wordmark uses a techno-display fallback stack that approximates a squared, futuristic logo style. The wordmark must be upright, not italic, not skewed, and not overly spaced.

Headings are compact and centered in the home screen. Workspace labels are small, sturdy, and utilitarian. Code uses a monospace face with a tight 12.5px size and generous enough line height to scan long HTML/CSS/JS snippets.

## Layout

The home screen is intentionally sparse: brand text in the top-left, the build question centered, and the prompt box directly below. The background is visible around the composition, but the center fade must create enough contrast behind the prompt.

The workspace is preview-first. The main default state uses a left chat/context panel and a right live-preview panel. Files, code, and explorer details live in a sliding inspection window so the primary flow remains focused on prompting and checking output.

Spacing follows an 8px rhythm with practical 14px gutters in the workspace. Panels should align tightly to the viewport with little unused bottom space. The bottom prompt dock floats above the workspace and remains the primary command surface.

## Elevation & Depth

Depth is created with three layers: animated code background, dark radial fades, and floating glass panels. Prompt cards have the deepest shadow because they are the primary action surface. Workspace panels are flatter, with subtle shadows and thin emerald-tinted borders.

Glass effects should be controlled. Use translucency and blur to make panels feel integrated with the matrix background, but do not let background glyphs interfere with text entry or code reading. The prompt card should never become transparent enough that the matrix text dominates it.

## Shapes

The shape language is rounded, compact, and technical. The prompt card uses a large 22px radius. Workspace panels use 24px. Sidebar and editor containers can go up to 28px where they need to feel like distinct sliding surfaces.

Small icon buttons should be soft and rounded, not square. Voice, attachment, link, repository, design-file, and hamburger buttons all use rounded 16px corners. Project type choices live in the expandable prompt menu so the footer does not duplicate controls.

## Components

### Prompt Box

The prompt box is the anchor component. It contains a multi-line textarea, a left group of icon tools, a voice button, an expandable project-type menu, a character counter, a Shift + Enter hint, and a high-contrast send button. The home prompt is taller, more spacious, and visibly 24px rounded; the workspace prompt dock is thinner and compact at 16px.

### Wordmark

The MINDIGENOUS wordmark sits top-left on the home screen and inside navigation surfaces in compact form. The home wordmark is larger and brighter than compact instances so it remains visible against the matrix field. It uses a green-to-cyan metallic gradient, but the text itself remains upright and readable. Do not add a separate icon mark unless the surface requires an app icon.

### Workspace Panels

Workspace panels are dark, rounded, and lightly glassed. Chat messages use small avatars and restrained card backgrounds. The preview panel should dominate the workspace because output validation is the user's main job after prompting.

### Files & Code Drawer

The inspection window slides in with a left file explorer and right editor. The file explorer uses folder/file icons and clear active rows. The editor uses tabs for HTML, CSS, JavaScript, and preview. Keep this drawer secondary to the live preview.

### Motion

Motion should explain state changes. Use the morphing prompt border for mode switches, a slide-in motion for workspace panels, and short ease-in-out transitions for menus and controls. Avoid decorative motion that competes with the matrix background.

## Do's and Don'ts

- Do keep the home screen minimal, with only the top-left wordmark, centered question, and prompt box.
- Do keep the matrix background behind strong center fades so text and controls stay legible.
- Do make all prompt icon buttons rounded and consistent.
- Do keep preview and prompt as the primary workflow surfaces.
- Do use emerald and cyan sparingly for action, selection, and brand highlights.
- Do not add sidebars, dashboards, or dense controls to the home screen.
- Do not introduce marketing hero sections into the builder UI.
- Do not make the wordmark italic, skewed, or paired with the old standalone M mark.
- Do not stack multiple animated background effects.
- Do not use heavy glows around every component; reserve depth for the prompt and transition frame.
