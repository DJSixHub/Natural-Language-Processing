from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Recipe:
    id: int
    nombre: str
    comensales: str
    dificultad: str
    tiempo_raw: str
    ingredientes: list[str]
    instrucciones: list[str]


@dataclass(frozen=True)
class Utensil:
    nombre: str
    url: str | None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_recipes(recetas_json: Path) -> list[Recipe]:
    raw = _read_json(recetas_json)
    recipes: list[Recipe] = []
    for idx, r in enumerate(raw):
        recipes.append(
            Recipe(
                id=idx,
                nombre=str(r.get("Nombre", "")).strip(),
                comensales=str(r.get("Comensales", "")).strip(),
                dificultad=str(r.get("Dificultad", "")).strip(),
                tiempo_raw=str(r.get("Tiempo", "")).strip(),
                ingredientes=[str(x) for x in (r.get("Ingredientes") or [])],
                instrucciones=[str(x) for x in (r.get("Instrucciones") or [])],
            )
        )
    return recipes


def load_utensils(utensilios_json: Path) -> list[Utensil]:
    raw = _read_json(utensilios_json)
    utensils: list[Utensil] = []
    for u in raw:
        utensils.append(Utensil(nombre=str(u.get("nombre", "")).strip(), url=u.get("url")))
    return utensils
