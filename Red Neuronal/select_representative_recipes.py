from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _norm(s: str) -> str:
    return " ".join(s.lower().strip().split())


def _recipe_text(r: dict[str, Any]) -> str:
    nombre = str(r.get("Nombre", ""))
    ings = r.get("Ingredientes", [])
    inst = r.get("Instrucciones", [])
    if not isinstance(ings, list):
        ings = []
    if not isinstance(inst, list):
        inst = []
    parts = [nombre] + [str(x) for x in ings] + [str(x) for x in inst]
    return "\n".join(p for p in parts if str(p).strip())


def _load_recipes(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"recetas.json must be a list, got: {type(data)}")
    out: list[dict[str, Any]] = []
    for r in data:
        if not isinstance(r, dict):
            continue
        out.append(r)
    return out


def _load_utensils(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    out: list[str] = []
    for u in data:
        if not isinstance(u, dict):
            continue
        name = u.get("nombre")
        if isinstance(name, str) and name.strip():
            out.append(name.strip())
    return out


def _literal_utensils_in_text(text: str, utensil_names: list[str]) -> set[str]:
    # Strictly dataset-only literal matching (no synonym lists).
    t = _norm(text)
    found: set[str] = set()
    for u in utensil_names:
        un = _norm(u)
        if un and un in t:
            found.add(u)
    return found


@dataclass
class SelectionResult:
    selected_indices: list[int]
    missing_utensils: set[str]


def select_representatives(
    recipes: list[dict[str, Any]],
    utensil_names: list[str],
    k: int,
    svd_dim: int,
    seed: int,
    ensure_utensil_coverage: bool,
    max_extra: int,
) -> SelectionResult:
    if k <= 0:
        raise SystemExit("k must be > 0")
    if k >= len(recipes):
        return SelectionResult(selected_indices=list(range(len(recipes))), missing_utensils=set())

    # Import locally so this script is optional.
    from sklearn.cluster import MiniBatchKMeans  # type: ignore
    from sklearn.decomposition import TruncatedSVD  # type: ignore
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.preprocessing import normalize  # type: ignore

    texts = [_recipe_text(r) for r in recipes]

    # TF-IDF "embeddings" (offline) + SVD -> dense low-dim vectors.
    vec = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        max_features=120_000,
        ngram_range=(1, 2),
        min_df=2,
    )
    X = vec.fit_transform(texts)

    dim = min(int(svd_dim), max(2, min(X.shape) - 1))
    svd = TruncatedSVD(n_components=dim, random_state=seed)
    Z = svd.fit_transform(X)
    Z = normalize(Z)

    km = MiniBatchKMeans(
        n_clusters=int(k),
        random_state=seed,
        batch_size=2048,
        n_init="auto",
        max_iter=200,
    )
    labels = km.fit_predict(Z)
    centers = km.cluster_centers_

    # Pick medoid: argmin distance to centroid per cluster.
    best_idx: dict[int, int] = {}
    best_dist: dict[int, float] = {}
    for i, c in enumerate(labels):
        center = centers[int(c)]
        # squared euclidean
        d = float(((Z[i] - center) ** 2).sum())
        if c not in best_dist or d < best_dist[c]:
            best_dist[c] = d
            best_idx[c] = i

    selected = sorted(set(best_idx.values()))

    if not ensure_utensil_coverage or not utensil_names:
        return SelectionResult(selected_indices=selected, missing_utensils=set())

    selected_texts = [_recipe_text(recipes[i]) for i in selected]
    covered: set[str] = set()
    for t in selected_texts:
        covered |= _literal_utensils_in_text(t, utensil_names)

    all_u = set(utensil_names)
    missing = all_u - covered

    if not missing:
        return SelectionResult(selected_indices=selected, missing_utensils=set())

    # Greedy add recipes that cover the most missing utensils.
    remaining_budget = max(0, int(max_extra))
    if remaining_budget == 0:
        return SelectionResult(selected_indices=selected, missing_utensils=missing)

    # Precompute literal utensils per recipe once.
    per_recipe_ut: list[set[str]] = []
    for r in recipes:
        per_recipe_ut.append(_literal_utensils_in_text(_recipe_text(r), utensil_names))

    selected_set = set(selected)
    while missing and remaining_budget > 0:
        best_i = -1
        best_gain = 0
        for i in range(len(recipes)):
            if i in selected_set:
                continue
            gain = len(per_recipe_ut[i] & missing)
            if gain > best_gain:
                best_gain = gain
                best_i = i
        if best_i == -1 or best_gain == 0:
            break
        selected_set.add(best_i)
        selected.append(best_i)
        missing -= per_recipe_ut[best_i]
        remaining_budget -= 1

    selected = sorted(selected_set)
    return SelectionResult(selected_indices=selected, missing_utensils=missing)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--recipes-json",
        default=str(Path(__file__).resolve().parents[1] / "Jsons_Scrappeados" / "recetas.json"),
        help="Path to recetas.json",
    )
    ap.add_argument(
        "--utensils-json",
        default=str(Path(__file__).resolve().parents[1] / "Jsons_Scrappeados" / "utensilios.json"),
        help="Path to utensilios.json",
    )
    ap.add_argument("--k", type=int, default=500, help="Number of clusters/representatives")
    ap.add_argument("--svd-dim", type=int, default=64, help="SVD embedding dim")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--ensure-utensil-coverage",
        action="store_true",
        help="Greedily add extra recipes to cover missing canonical utensils (literal coverage).",
    )
    ap.add_argument(
        "--max-extra",
        type=int,
        default=250,
        help="Maximum extra recipes to add for utensil coverage.",
    )
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[0] / "representative_recipes.json"),
        help="Output JSON containing representative recipes (list of recipe dicts).",
    )
    args = ap.parse_args()

    recipes_path = Path(args.recipes_json)
    utensils_path = Path(args.utensils_json)
    out_path = Path(args.out)

    recipes = _load_recipes(recipes_path)
    if not recipes:
        raise SystemExit("No recipes loaded")

    utensil_names = _load_utensils(utensils_path)

    res = select_representatives(
        recipes=recipes,
        utensil_names=utensil_names,
        k=int(args.k),
        svd_dim=int(args.svd_dim),
        seed=int(args.seed),
        ensure_utensil_coverage=bool(args.ensure_utensil_coverage),
        max_extra=int(args.max_extra),
    )

    selected_recipes = [recipes[i] for i in res.selected_indices]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(selected_recipes, ensure_ascii=False), encoding="utf-8")

    print("OK")
    print("recipes_total:", len(recipes))
    print("selected:", len(selected_recipes))
    print("k:", int(args.k))
    if args.ensure_utensil_coverage:
        print("missing_utensils:", len(res.missing_utensils))
        if res.missing_utensils:
            sample = sorted(list(res.missing_utensils))[:25]
            print("missing_sample:", sample)


if __name__ == "__main__":
    main()
