# MINDI 1.5 Vision-Coder

**Built by [MINDIGENOUS.AI](https://mindigenous.ai)**  
**Builder:** Faaz ([@Mindigenous](https://huggingface.co/Mindigenous) on HuggingFace)  
**Started:** April 14, 2026 | **Training Complete:** April 28, 2026 | **Frontend v2:** May 2, 2026

---

## What is MINDI 1.5?

MINDI 1.5 Vision-Coder is a multimodal AI model that generates frontend code (HTML/CSS/JS, React, Next.js, Tailwind) from text prompts and UI screenshots.

- **Architecture:** Qwen2.5-Coder-7B-Instruct + LoRA + CLIP ViT-L/14 + Vision-Language Fusion
- **Training:** 10,000 steps across 3 phases on AMD MI300X 192GB
- **Live API:** [Mindigenous/mindi-chat](https://huggingface.co/spaces/Mindigenous/mindi-chat) on HuggingFace Spaces

## Frontend — AI Website Builder

A professional 3-panel IDE (Bolt.new-style) for interacting with MINDI:

```powershell
cd frontend
npm install
npm run dev   # → http://localhost:5173
```

Features: Plan modal, prompt enhancement, code animation, live preview, file tree, demo fallback.  
**Read `context.md` for full architecture and next steps.**

## HuggingFace

- **Model:** `Mindigenous/MINDI-1.5-Vision-Coder` (private)
- **Dataset:** `Mindigenous/MINDI-1.5-training-data` (private)
- **Space:** `Mindigenous/mindi-chat` (live, ZeroGPU)

## License

Apache 2.0
