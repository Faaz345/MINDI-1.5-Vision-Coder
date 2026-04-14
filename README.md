# MINDI 1.5 Vision-Coder

**Built by [MINDIGENOUS.AI](https://mindigenous.ai)**

**Builder:** Faaz ([@Mindigenous](https://huggingface.co/Mindigenous) on HuggingFace)

**Started:** April 14, 2026

**Target Launch:** May 5, 2026

---

## What is MINDI 1.5?

MINDI 1.5 Vision-Coder is a multimodal agentic AI coding model that:

- Generates production-ready Next.js 14 + Tailwind CSS + TypeScript code
- Sees its own output via vision capabilities (CLIP ViT-L/14)
- Critiques its own UI/UX design and iterates
- Searches the internet for latest packages and documentation
- Tests code in an isolated sandbox environment
- Fixes its own errors automatically
- Suggests improvements to the user

## Architecture

- **Base Model:** Open-source coding model (3B-7B parameters, Apache 2.0 / MIT)
- **Fine-tuning:** LoRA on AMD MI300X 192GB VRAM
- **Vision Encoder:** CLIP ViT-L/14
- **Agents:** Search + Sandbox + UI Critic + Code Generation
- **Training Data:** 500,000+ curated examples
- **Backend:** FastAPI
- **Output Format:** Next.js 14 + Tailwind CSS + TypeScript

## HuggingFace

Final model will be published at: `Mindigenous/MINDI-1.5-Vision-Coder`

## License

Apache 2.0
