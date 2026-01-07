from __future__ import annotations

import re
import unicodedata


_WS_RE = re.compile(r"\s+")


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = strip_accents(text)
    text = re.sub(r"[^a-z0-9áéíóúüñ\s]", " ", text, flags=re.IGNORECASE)
    text = strip_accents(text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def normalize_ingredient_line(line: str) -> str:
    """Heuristic cleanup to get ingredient 'name-ish' from raw ingredient lines."""
    t = normalize(line)
    # Remove common quantity patterns
    t = re.sub(r"\b\d+(?:[\.,]\d+)?\b", " ", t)
    t = re.sub(r"\b(taza|tazas|cucharada|cucharadas|cucharadita|cucharaditas|gramo|gramos|kg|kilo|kilos|ml|mililitro|mililitros|litro|litros|onza|onzas|pieza|piezas|diente|dientes|pizca|pizcas)\b", " ", t)
    t = re.sub(r"\b(de|del|la|el|los|las|un|una|unos|unas|al|a|para)\b", " ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t
