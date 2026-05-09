from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ValidationResult:
    ok: bool
    warnings: list[str] = field(default_factory=list)


class ValidationAgent:
    async def validate_files(self, files: dict[str, str]) -> ValidationResult:
        warnings: list[str] = []
        if "index.html" in files and "script.js" in files["index.html"] and "script.js" not in files:
            warnings.append("index.html references script.js but no script.js was generated.")
        if "index.html" in files and "styles.css" in files["index.html"] and "styles.css" not in files:
            warnings.append("index.html references styles.css but no styles.css was generated.")
        if any("TODO_MINDI" in content for content in files.values()):
            warnings.append("Generated files contain unresolved placeholders.")
        return ValidationResult(ok=not warnings, warnings=warnings)
