from __future__ import annotations

from dataclasses import dataclass

from .normalization import normalize


@dataclass(frozen=True)
class UtensilLexicon:
    canonical: list[str]


_CORE_KEYWORDS = {
    "sarten": "sartén",
    "olla": "olla",
    "cacerola": "cacerola",
    "horno": "horno",
    "microondas": "microondas",
    "licuadora": "licuadora",
    "batidora": "batidora",
    "comal": "comal",
    "cuchillo": "cuchillo",
    "tabla": "tabla",
    "rallador": "rallador",
    "colador": "colador",
    "coladera": "colador",
    "bol": "bol",
    "bowl": "bol",
    "cuchara": "cuchara",
    "cucharones": "cucharón",
    "cucharon": "cucharón",
    "tenedor": "tenedor",
    "espumadera": "espumadera",
    "pinzas": "pinzas",
    "tijeras": "tijeras",
    "molinillo": "molinillo",
    "mortero": "mortero",
    "exprimidor": "exprimidor",
    "bandeja": "bandeja",
}


def build_lexicon_from_utensilios(utensilios: list[str]) -> UtensilLexicon:
    found: set[str] = set()
    for raw in utensilios:
        t = normalize(raw)
        for k, canon in _CORE_KEYWORDS.items():
            if k in t:
                found.add(canon)
    # always include a few common ones even if absent
    found.update({"sartén", "olla", "horno", "cuchillo", "tabla"})
    return UtensilLexicon(canonical=sorted(found))


def extract_utensils_for_recipe(lexicon: UtensilLexicon, recipe_text: str) -> set[str]:
    t = normalize(recipe_text)
    present = set()
    for canon in lexicon.canonical:
        if normalize(canon) in t:
            present.add(canon)
    return present
