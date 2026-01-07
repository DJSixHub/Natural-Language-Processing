from __future__ import annotations

from dataclasses import dataclass

from .normalization import normalize, normalize_ingredient_line


@dataclass(frozen=True)
class DietFlags:
    vegan: bool
    vegetarian: bool
    low_carb: bool
    keto: bool
    high_protein: bool
    paleo: bool
    halal_friendly: bool
    kosher_friendly: bool


_ANIMAL = {
    "pollo",
    "carne",
    "res",
    "cerdo",
    "puerco",
    "ternera",
    "cordero",
    "pescado",
    "atun",
    "sardina",
    "salmon",
    "camarones",
    "cangrejo",
    "langosta",
    "marisco",
    "huevo",
    "leche",
    "queso",
    "mantequilla",
    "yogur",
    "crema",
    "miel",
    "gelatina",
}

_PORK = {"cerdo", "puerco", "jamon", "tocino", "chorizo", "salchicha"}

_SHELLFISH = {"camarones", "camaron", "langosta", "cangrejo", "marisco", "mejillon", "ostras"}

_ALCOHOL = {"vino", "cerveza", "ron", "whisky", "vodka", "licor"}

_GRAINS_LEGUMES_DAIRY = {
    "harina",
    "trigo",
    "arroz",
    "maiz",
    "tortilla",
    "pan",
    "pasta",
    "avena",
    "frijol",
    "lenteja",
    "garbanzo",
    "soya",
    "leche",
    "queso",
    "yogur",
    "mantequilla",
}

_HIGH_CARB = {"azucar", "harina", "trigo", "arroz", "maiz", "tortilla", "pan", "pasta", "patata", "papa"}

_HIGH_PROTEIN = {"pollo", "pavo", "atun", "pescado", "huevo", "carne", "res", "ternera", "camarones", "tofu", "soya", "lenteja"}


def infer_diets(ingredientes: list[str], instrucciones: list[str] | None = None) -> DietFlags:
    # combine ingredient lines primarily
    items = []
    for line in ingredientes:
        t = normalize_ingredient_line(line)
        if t:
            items.append(t)

    joined = " ".join(items)
    toks = set(normalize(joined).split())

    has_animal = any(w in toks for w in _ANIMAL)
    has_dairy_or_eggs = any(w in toks for w in {"huevo", "leche", "queso", "mantequilla", "yogur", "crema"})
    has_pork = any(w in toks for w in _PORK)
    has_shellfish = any(w in toks for w in _SHELLFISH)
    has_alcohol = any(w in toks for w in _ALCOHOL)

    vegan = not has_animal
    vegetarian = (not has_animal) or (has_dairy_or_eggs and not any(w in toks for w in _PORK | {"pollo", "carne", "res", "ternera", "cordero", "pescado", "atun", "salmon", "camarones", "marisco"}))

    low_carb = not any(w in toks for w in _HIGH_CARB)
    keto = low_carb  # simple proxy

    high_protein = any(w in toks for w in _HIGH_PROTEIN)

    paleo = not any(w in toks for w in _GRAINS_LEGUMES_DAIRY)

    # "friendly" because we can't guarantee slaughter/handling
    halal_friendly = (not has_pork) and (not has_alcohol)

    # kosher rules are complex; we approximate: no pork/shellfish and avoid obvious meat+dairy combos
    has_meat = any(w in toks for w in {"pollo", "carne", "res", "ternera", "cordero", "pescado", "atun", "salmon", "camarones", "marisco"})
    has_dairy = any(w in toks for w in {"leche", "queso", "mantequilla", "yogur", "crema"})
    kosher_friendly = (not has_pork) and (not has_shellfish) and not (has_meat and has_dairy)

    return DietFlags(
        vegan=vegan,
        vegetarian=vegetarian,
        low_carb=low_carb,
        keto=keto,
        high_protein=high_protein,
        paleo=paleo,
        halal_friendly=halal_friendly,
        kosher_friendly=kosher_friendly,
    )
