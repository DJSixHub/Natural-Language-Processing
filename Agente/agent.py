from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import DEFAULT_PATHS, Paths
from .data import load_recipes, load_utensils, Recipe
from .diets import infer_diets
from .intents import IntentModel, fallback_intent
from .link_utensils import build_lexicon_from_utensilios, extract_utensils_for_recipe
from .parse import parse_constraints
from .retrieval import load_index, query_index
from .timeparse import parse_minutes


from .retrieval import RecipeIndex
from .link_utensils import UtensilLexicon


@dataclass(frozen=True)
class AgentResponse:
    text: str


class CookingAgent:
    def __init__(self, paths: Paths = DEFAULT_PATHS):
        self.paths = paths
        self._intent_model: IntentModel | None = None
        self._index: RecipeIndex | None = None
        self._recipes: list[Recipe] | None = None
        self._utensils = None
        self._lexicon: UtensilLexicon | None = None
        self._recipe_utensils: dict[int, set[str]] = {}

    def _lazy_load(self) -> None:
        if self._recipes is None:
            self._recipes = load_recipes(self.paths.recetas_json)
        if self._utensils is None:
            self._utensils = load_utensils(self.paths.utensilios_json)
        if self._lexicon is None:
            self._lexicon = build_lexicon_from_utensilios([u.nombre for u in self._utensils])
        if self._index is None:
            # If artifacts missing, user must run build_index.py
            self._index = load_index(self.paths.artifacts_dir)
        if self._intent_model is None:
            intent_path = self.paths.artifacts_dir / "intent_model.joblib"
            if intent_path.exists():
                self._intent_model = IntentModel.load(intent_path)

    def respond(self, user_text: str) -> AgentResponse:
        self._lazy_load()

        # Inform the type-checker: lazy load guarantees these
        assert self._index is not None
        assert self._lexicon is not None

        intent = self._intent_model.predict(user_text) if self._intent_model else fallback_intent(user_text)
        constraints = parse_constraints(user_text)

        # retrieve candidates
        candidates = query_index(self._index, user_text, top_k=12)
        filtered: list[Recipe] = []

        for r, _score in candidates:
            if constraints.max_minutes is not None:
                mins = parse_minutes(r.tiempo_raw)
                if mins is not None and mins > constraints.max_minutes:
                    continue

            diets = infer_diets(r.ingredientes, r.instrucciones)
            if constraints.diet and constraints.diet not in {"no_requirements", None}:
                if not getattr(diets, constraints.diet, False):
                    continue

            # ingredients include/exclude (very approximate substring match)
            ing_text = " ".join(r.ingredientes).lower()
            if any(x for x in constraints.include_ingredients if x.lower() not in ing_text):
                continue
            if any(x for x in constraints.exclude_ingredients if x.lower() in ing_text):
                continue

            # utensils extracted from name+instructions
            if r.id not in self._recipe_utensils:
                recipe_text = "\n".join([r.nombre] + r.instrucciones)
                self._recipe_utensils[r.id] = extract_utensils_for_recipe(self._lexicon, recipe_text)
            u_set = self._recipe_utensils[r.id]

            if any(x for x in constraints.include_utensils if x.lower() not in " ".join(u_set).lower()):
                continue
            if any(x for x in constraints.exclude_utensils if x.lower() in " ".join(u_set).lower()):
                continue

            filtered.append(r)

        if intent.intent == "saludo":
            return AgentResponse("¡Hola! Dime qué tipo de receta buscas (dieta, tiempo, ingredientes o utensilios).")

        if not filtered:
            msg = "No encontré recetas que cumplan esas condiciones."
            if constraints.diet and constraints.diet != "no_requirements":
                msg += f" (dieta: {constraints.diet})"
            if constraints.max_minutes:
                msg += f" (tiempo <= {constraints.max_minutes} min)"
            msg += " Prueba con menos restricciones o con otra palabra clave."
            return AgentResponse(msg)

        top = filtered[:5]
        lines = ["Aquí tienes algunas opciones:"]
        for i, r in enumerate(top, 1):
            lines.append(f"{i}. {r.nombre} — {r.tiempo_raw or 'tiempo N/D'} — {r.dificultad or 'dificultad N/D'}")

        # Add utensil summary for the first result
        r0 = top[0]
        u0 = sorted(self._recipe_utensils.get(r0.id, set()))
        if u0:
            lines.append("")
            lines.append(f"Utensilios detectados para la primera opción: {', '.join(u0)}")

        # Safety note for halal/kosher
        if constraints.diet in {"halal_friendly", "kosher_friendly"}:
            lines.append("")
            lines.append("Nota: halal/kosher aquí se interpreta como 'compatible' por ingredientes; no puedo garantizar certificación.")

        return AgentResponse("\n".join(lines))
