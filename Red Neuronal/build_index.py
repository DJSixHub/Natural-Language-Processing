from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Agente.config import DEFAULT_PATHS
from Agente.data import load_recipes
from Agente.retrieval import recipe_to_document


def main() -> None:
    paths = DEFAULT_PATHS
    artifacts = paths.artifacts_dir
    artifacts.mkdir(parents=True, exist_ok=True)

    recipes = load_recipes(paths.recetas_json)
    docs = [recipe_to_document(r) for r in recipes]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_features=250_000,
    )
    matrix = vectorizer.fit_transform(docs)

    joblib.dump(vectorizer, artifacts / "tfidf_vectorizer.joblib")
    sparse.save_npz(artifacts / "tfidf_matrix.npz", sparse.csr_matrix(matrix))

    # store recipe metadata for inference without re-reading the full json
    meta = [
        {
            "id": r.id,
            "nombre": r.nombre,
            "comensales": r.comensales,
            "dificultad": r.dificultad,
            "tiempo_raw": r.tiempo_raw,
            "ingredientes": r.ingredientes,
            "instrucciones": r.instrucciones,
        }
        for r in recipes
    ]
    (artifacts / "recipes_meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    print(f"OK: índice creado. Recetas: {len(recipes)}")
    print(f"Artefactos en: {artifacts}")


if __name__ == "__main__":
    main()
