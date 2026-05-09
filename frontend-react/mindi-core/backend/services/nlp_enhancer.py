from __future__ import annotations


class NlpEnhancer:
    async def enhance(self, prompt: str, intent: str, design_settings: dict | None = None) -> str:
        if intent != "build_ui":
            return prompt.strip()

        settings = design_settings or {}
        accent = settings.get("accent") or "controlled neon accent"
        return (
            f"{prompt.strip()}\n\n"
            "Build a production-ready React/Vite compatible UI. "
            "Keep the MINDIGENOUS design language: minimal dark structure, VS Code-inspired layout, "
            f"accessible controls, responsive behavior, and {accent}. "
            "Return concrete files and explain implementation choices briefly."
        )
