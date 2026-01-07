from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
from scipy import sparse

from .data import Recipe
from .normalization import normalize


@dataclass(frozen=True)
class RecipeIndex:
    recipes: list[Recipe]
    vectorizer: "VectorizerLike"
    matrix: sparse.csr_matrix


class VectorizerLike(Protocol):
    def transform(self, raw_documents: list[str]) -> sparse.spmatrix:
        ...


def recipe_to_document(r: Recipe) -> str:
    parts = [r.nombre]
    parts.extend(r.ingredientes)
    parts.extend(r.instrucciones)
    return "\n".join(parts)


def load_index(artifacts_dir: Path) -> RecipeIndex:
    meta_path = artifacts_dir / "recipes_meta.json"
    vec_path = artifacts_dir / "tfidf_vectorizer.joblib"
    mat_path = artifacts_dir / "tfidf_matrix.npz"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    recipes = [Recipe(**m) for m in meta]

    vectorizer = joblib.load(vec_path)
    matrix = sparse.csr_matrix(sparse.load_npz(mat_path))

    return RecipeIndex(recipes=recipes, vectorizer=vectorizer, matrix=matrix)


def query_index(index: RecipeIndex, query: str, top_k: int = 8) -> list[tuple[Recipe, float]]:
    q = normalize(query)
    q_vec: sparse.csr_matrix = sparse.csr_matrix(index.vectorizer.transform([q]))
    scores = (index.matrix @ q_vec.T).toarray().ravel()
    if scores.size == 0:
        return []
    top_idx = np.argpartition(-scores, min(top_k, scores.size - 1))[:top_k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [(index.recipes[i], float(scores[i])) for i in top_idx]
