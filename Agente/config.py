from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    repo_root: Path

    @property
    def recetas_json(self) -> Path:
        return self.repo_root / "Jsons_Scrappeados" / "recetas.json"

    @property
    def utensilios_json(self) -> Path:
        return self.repo_root / "Jsons_Scrappeados" / "utensilios.json"

    @property
    def artifacts_dir(self) -> Path:
        return self.repo_root / "Red Neuronal" / "artifacts"


DEFAULT_PATHS = Paths(repo_root=Path(__file__).resolve().parents[1])
