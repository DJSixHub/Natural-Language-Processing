from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib


@dataclass(frozen=True)
class IntentResult:
    intent: str
    confidence: float


class IntentModel:
    def __init__(self, model):
        self._model = model

    @staticmethod
    def load(path: Path) -> "IntentModel":
        return IntentModel(joblib.load(path))

    def predict(self, text: str) -> IntentResult:
        # For sklearn pipelines; use predict_proba when available
        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba([text])[0]
            idx = int(proba.argmax())
            return IntentResult(intent=str(self._model.classes_[idx]), confidence=float(proba[idx]))
        pred = self._model.predict([text])[0]
        return IntentResult(intent=str(pred), confidence=0.5)


def fallback_intent(text: str) -> IntentResult:
    t = text.lower()
    if any(x in t for x in ["hola", "buenas", "hey"]):
        return IntentResult("saludo", 0.9)
    if any(x in t for x in ["utensilio", "utensilios", "sarten", "olla", "horno"]):
        return IntentResult("utensilios", 0.6)
    if any(x in t for x in ["ingrediente", "ingredientes"]):
        return IntentResult("ingredientes", 0.6)
    if any(x in t for x in ["tiempo", "minutos", "hora", "horas"]):
        return IntentResult("tiempo", 0.6)
    return IntentResult("recomendar", 0.4)
