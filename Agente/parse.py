from __future__ import annotations

import re
from dataclasses import dataclass

from .normalization import normalize


@dataclass(frozen=True)
class QueryConstraints:
    diet: str | None
    include_utensils: set[str]
    exclude_utensils: set[str]
    include_ingredients: set[str]
    exclude_ingredients: set[str]
    max_minutes: int | None


_DIET_KEYWORDS: dict[str, list[str]] = {
    "vegan": ["vegano", "vegana", "vegan"],
    "vegetarian": ["vegetariano", "vegetariana"],
    "paleo": ["paleo"],
    "keto": ["keto", "cetogenica", "cetogenico"],
    "low_carb": ["bajo en carbohidratos", "low carb", "pocos carbohidratos"],
    "high_protein": ["alto en proteina", "alta en proteina", "high protein", "proteico"],
    "halal_friendly": ["halal"],
    "kosher_friendly": ["kosher"],
    "no_requirements": ["sin restricciones", "cualquiera", "normal", "regular"],
}


def _extract_diet(text_n: str) -> str | None:
    for diet, kws in _DIET_KEYWORDS.items():
        for kw in kws:
            if normalize(kw) in text_n:
                return diet
    return None


def _extract_time(text_n: str) -> int | None:
    # "en menos de 30 min", "<=45m", "45 minutos"
    m = re.search(r"(\d{1,3})\s*(?:minutos|min|m)\b", text_n)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d{1,2})\s*(?:horas|hora|h)\b", text_n)
    if m:
        return int(m.group(1)) * 60
    return None


def parse_constraints(user_text: str) -> QueryConstraints:
    t = normalize(user_text)

    diet = _extract_diet(t)
    max_minutes = _extract_time(t)

    include_utensils: set[str] = set()
    exclude_utensils: set[str] = set()
    include_ingredients: set[str] = set()
    exclude_ingredients: set[str] = set()

    # very lightweight patterns: "sin X", "no use X", "con X"
    for m in re.finditer(r"\b(sin|no|evita|evitar)\s+([a-z0-9\s]{2,40})", t):
        phrase = m.group(2).strip()
        # split by conjunctions
        for part in re.split(r"\b(y|e|o)\b|,", phrase):
            p = (part or "").strip()
            if not p or p in {"y", "e", "o"}:
                continue
            exclude_ingredients.add(p)
            exclude_utensils.add(p)

    for m in re.finditer(r"\b(con|usa|utiliza|que use)\s+([a-z0-9\s]{2,40})", t):
        phrase = m.group(2).strip()
        for part in re.split(r"\b(y|e|o)\b|,", phrase):
            p = (part or "").strip()
            if not p or p in {"y", "e", "o"}:
                continue
            include_ingredients.add(p)
            include_utensils.add(p)

    return QueryConstraints(
        diet=diet,
        include_utensils=include_utensils,
        exclude_utensils=exclude_utensils,
        include_ingredients=include_ingredients,
        exclude_ingredients=exclude_ingredients,
        max_minutes=max_minutes,
    )
