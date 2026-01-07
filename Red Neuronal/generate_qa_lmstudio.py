from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import time
import unicodedata
from concurrent.futures import Future, ThreadPoolExecutor, wait
from concurrent.futures import FIRST_COMPLETED
from collections import deque
from pathlib import Path
from typing import Any, cast

import requests

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


def extract_json_object(text: str) -> str:
    """Extract the first JSON object from a string (handles ```json fences)."""
    if not text:
        return text
    # Strip common markdown fences
    t = text.strip()
    if t.startswith("```"):
        # remove first fence line
        first_nl = t.find("\n")
        t = t[first_nl + 1 :] if first_nl != -1 else t
        # remove ending fence
        if t.rstrip().endswith("```"):
            t = t.rstrip()[: -3]
        t = t.strip()

    # Find first {...} block
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return t[start : end + 1]
    return t


def norm(s: str) -> str:
    s = s.lower().strip()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = re.sub(r"\s+", " ", s)
    return s


def is_valid_utensil_list(utensils: list[str], allowed_exact: set[str]) -> bool:
    return all(u in allowed_exact for u in utensils)


def evidence_is_valid(evidence: list[str], src: set[str]) -> bool:
    return all(ev in src for ev in evidence)


def ingredient_entities_valid(entities_ings: list[str], ingredientes: list[str]) -> bool:
    # strict mode: ingredient entities must be exact strings from receta.ingredientes
    src = set(ingredientes)
    return all(ing in src for ing in entities_ings)


def _coerce_list_items_to_strings_from_source(items: object, source: list[str]) -> list[str]:
    """Accept either exact strings from source OR integer indices into source.

    This keeps grounding strict (final stored output uses exact source strings)
    while making the teacher's job easier.
    """
    if not isinstance(items, list):
        raise ValueError("must be list")
    out: list[str] = []
    for x in items:
        if isinstance(x, int):
            if x < 0 or x >= len(source):
                raise ValueError("index out of range")
            out.append(source[int(x)])
        else:
            s = str(x)
            if s not in source:
                raise ValueError("string not exact source line")
            out.append(s)
    return out





def _v1(base_url: str) -> str:
    # Accept either http://host:port or http://host:port/v1
    b = base_url.rstrip("/")
    return b if b.endswith("/v1") else b + "/v1"


def list_models(base_url: str) -> list[str]:
    url = _v1(base_url) + "/models"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    out: list[str] = []
    items = data.get("data") or []
    if not isinstance(items, list):
        return out
    for m in items:
        if not isinstance(m, dict):
            continue
        mid = m.get("id")
        if isinstance(mid, str) and mid:
            out.append(mid)
    return out


def _parse_teacher_json(content: str) -> object:
    return json.loads(extract_json_object(content))


def _parse_teacher_user_query_and_output(
    content: str,
    fallback_user_query: str,
) -> tuple[str, dict[str, object]]:
    """Parse teacher response.

    Accept:
    - {"user_query": "...", "output": {...}}
    - output-only object {...} (fallback uses provided fallback_user_query)
    """
    resp = _parse_teacher_json(content)
    if isinstance(resp, dict) and "output" in resp:
        resp_d = cast(dict[str, object], resp)
        uq_raw = resp_d.get("user_query")
        uq = str(uq_raw or "").strip()
        out_obj_raw = resp_d.get("output")
    else:
        uq = ""
        out_obj_raw = resp

    if not uq:
        uq = str(fallback_user_query or "").strip()
    if not uq:
        raise ValueError("missing user_query")

    if not isinstance(out_obj_raw, dict):
        raise ValueError("output must be a JSON object")
    out_obj = cast(dict[str, object], out_obj_raw)
    return uq, out_obj


def call_openai_compatible_chat(
    base_url: str,
    model: str,
    messages: list[dict],
    session: requests.Session,
    temperature: float = 0.1,
    max_tokens: int = 768,
    top_p: float | None = None,
    timeout_s: float = 120.0,
) -> str:
    url = _v1(base_url) + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if top_p is not None:
        payload["top_p"] = top_p
    r = session.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists() or path.is_dir():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                out.append(json.loads(s))
            except Exception:
                # tolerate a partially written final line
                continue
    return out


def _recipe_name_from_example(ex: dict) -> str:
    try:
        inp = ex.get("input")
        inp_obj = json.loads(inp) if isinstance(inp, str) else inp
        if isinstance(inp_obj, dict) and isinstance(inp_obj.get("receta"), dict):
            name = inp_obj["receta"].get("nombre")
            return str(name or "").strip()
    except Exception:
        return ""
    return ""


def _append_jsonl_line(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass


class _JsonlWriter:
    def __init__(self, train_path: Path, val_path: Path) -> None:
        self.train_path = train_path
        self.val_path = val_path
        self._lock = threading.Lock()
        self._train_f = train_path.open("a", encoding="utf-8", newline="\n")
        self._val_f = val_path.open("a", encoding="utf-8", newline="\n")
        self._since_flush = 0
        self._since_fsync = 0

    def close(self) -> None:
        with self._lock:
            try:
                self._train_f.flush()
                self._val_f.flush()
            finally:
                self._train_f.close()
                self._val_f.close()

    def append(self, which: str, obj: dict) -> None:
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        with self._lock:
            f = self._val_f if which == "val" else self._train_f
            f.write(line)
            self._since_flush += 1
            self._since_fsync += 1
            if self._since_flush >= 5:
                f.flush()
                self._since_flush = 0
            if self._since_fsync >= 25:
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
                self._since_fsync = 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:5000", help="LM Studio server base URL (optionally ending with /v1)")
    ap.add_argument("--model", default="", help="Model id as exposed by the server; empty = auto-detect first")
    ap.add_argument(
        "--recipes-json",
        default="",
        help="Path a recetas.json (o subset como representative_recipes.json). Vacío = Jsons_Scrappeados/recetas.json.",
    )
    ap.add_argument(
        "--utensils-json",
        default="",
        help="Path a utensilios.json. Vacío = Jsons_Scrappeados/utensilios.json.",
    )
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n", type=int, default=1000, help="Total QA pairs to generate")
    ap.add_argument("--recipe-frac", type=float, default=0.8, help="Fraction grounded in recipes")
    ap.add_argument("--mode", choices=["mixed", "all_recipes"], default="mixed")
    ap.add_argument("--per-recipe", type=int, default=1, help="Examples per recipe in all_recipes mode")
    ap.add_argument("--retries", type=int, default=4, help="Retries per example if validation fails")
    ap.add_argument("--refusal-frac", type=float, default=0.15, help="In mixed mode, fraction of refusal examples")
    ap.add_argument("--max-attempts", type=int, default=0, help="If >0, cap total attempts (useful if teacher is unstable)")
    ap.add_argument(
        "--utensil-infer-threshold",
        type=float,
        default=0.35,
        help="Umbral para aceptar utensilios inferidos por co-ocurrencia P(utensilio|token).",
    )
    ap.add_argument(
        "--utensil-infer-min-action-count",
        type=int,
        default=30,
        help="Mínimo de recetas donde aparece un token para considerarlo en inferencia.",
    )
    ap.add_argument(
        "--utensil-infer-min-joint",
        type=int,
        default=8,
        help="Mínimo de co-ocurrencias (token, utensilio) para considerarlo.",
    )
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[0] / "model" / "distill_train.jsonl"),
        help="Output JSONL path",
    )
    ap.add_argument("--val-out", default=str(Path(__file__).resolve().parents[0] / "model" / "distill_val.jsonl"))
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Trunca los archivos de salida existentes (en vez de reanudar).",
    )
    ap.add_argument("--workers", type=int, default=1, help="Workers concurrentes para llamadas al teacher (I/O-bound).")
    ap.add_argument("--temperature", type=float, default=0.2, help="Temperatura para el teacher.")
    ap.add_argument("--top-p", type=float, default=0.9, help="Top-p para el teacher.")
    ap.add_argument("--max-tokens", type=int, default=1100, help="Max tokens de salida del teacher (evita truncado).")
    ap.add_argument("--timeout-s", type=float, default=120.0, help="Timeout por request (segundos).")
    ap.add_argument(
        "--inflight",
        type=int,
        default=0,
        help="Máximo de requests en vuelo (0 = 2*workers).",
    )
    args = ap.parse_args()
    # Preflight: ensure server reachable and model id known
    try:
        models = list_models(args.base_url)
    except Exception as e:
        raise SystemExit(
            "No pude conectar al servidor OpenAI-compatible.\n"
            f"- base_url: {args.base_url}\n"
            "- Asegúrate de que LM Studio 'Local Server' esté activo y exponga /v1/models y /v1/chat/completions.\n"
            "- Si tu puerto es distinto (LM Studio suele usar 1234), pásalo con --base-url.\n"
            f"Detalle: {e}"
        )

    model_id = args.model.strip() or (models[0] if models else "")
    if not model_id:
        raise SystemExit("El servidor respondió, pero no encontré modelos en /v1/models. Revisa LM Studio.")
    print("Usando modelo:", model_id)


    random.seed(args.seed)

    repo_root = Path(__file__).resolve().parents[1]
    recetas_path = Path(args.recipes_json) if str(args.recipes_json).strip() else (repo_root / "Jsons_Scrappeados" / "recetas.json")
    utensilios_path = Path(args.utensils_json) if str(args.utensils_json).strip() else (repo_root / "Jsons_Scrappeados" / "utensilios.json")

    recetas = json.loads(recetas_path.read_text(encoding="utf-8"))
    utensilios = json.loads(utensilios_path.read_text(encoding="utf-8"))

    utensil_catalog: list[str] = []
    for u in utensilios:
        if not isinstance(u, dict):
            continue
        name = u.get("nombre")
        if isinstance(name, str) and name.strip():
            utensil_catalog.append(name.strip())

    # normalized -> all originals (keeps variants; still closed vocab)
    utensil_norm_to_names: dict[str, set[str]] = {}
    for u in utensil_catalog:
        k = norm(u)
        if not k:
            continue
        bucket = utensil_norm_to_names.get(k)
        if bucket is None:
            bucket = set()
            utensil_norm_to_names[k] = bucket
        bucket.add(u)

    system = (
        "Eres un asistente experto de cocina y soporte del agente de recetas. "
        "Responde SIEMPRE en español. "
        "REGLA CRÍTICA: Si la pregunta requiere datos que NO están en el contexto proporcionado, "
        "di explícitamente que no se puede determinar con el contexto. "
        "NO inventes ingredientes, pasos, cantidades, utensilios ni tiempos. "
        "\n\n"
        "Debes devolver SOLO un JSON válido con EXACTAMENTE estas claves en la raíz: user_query, output. "
        "- user_query: una consulta natural del usuario en español (puede usar regionalismos/sinónimos), no vacía. "
        "- output: un objeto JSON con claves: answer, evidence, intent, entities. "
        "\n\n"
        "GROUNDING ESTRICTO:\n"
        "- evidence: DEVUELVE una lista de ÍNDICES enteros (0..n-1) que apunten a receta.ingredientes o receta.instrucciones. "
        "  (Alternativa aceptada: strings exactos, pero se recomienda índices).\n"
        "- entities.utensilios: SOLO strings EXACTOS presentes en receta.utensilios_permitidos.\n"
        "- entities.ingredientes (STRICT): DEVUELVE lista de ÍNDICES enteros (0..n-1) apuntando a receta.ingredientes "
        "  (Alternativa aceptada: strings exactos).\n"
        "NO incluyas Markdown ni fences; SOLO JSON."
    )

    tasks = [
        "recomendar recetas según restricciones",
        "explicar sustituciones de ingredientes",
        "explicar utensilios necesarios",
        "responder si una receta es vegana/keto/paleo/alta en proteína",
        "estimar tiempo y pasos",
        "intención y enrutamiento del agente (rechazar/redirigir)",
    ]

    out_path = Path(args.out)
    val_path = Path(args.val_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        out_path.write_text("", encoding="utf-8")
        val_path.write_text("", encoding="utf-8")

    existing_train = _read_jsonl(out_path)
    existing_val = _read_jsonl(val_path)
    train_written = len(existing_train)
    val_written = len(existing_val)

    target_success = len(recetas) * args.per_recipe if args.mode == "all_recipes" else args.n
    val_target = max(1, int(target_success * args.val_frac))

    # For coverage mode, resume per-recipe counts from existing output.
    per_recipe_done: dict[str, int] = {}
    if args.mode == "all_recipes":
        for example in existing_train + existing_val:
            rn = _recipe_name_from_example(example)
            if rn:
                per_recipe_done[rn] = per_recipe_done.get(rn, 0) + 1

    # Progress bar
    pbar = None
    if tqdm is None:
        print("Nota: instala 'tqdm' para progress bar (pip install tqdm).")
    else:
        pbar = tqdm(
            total=target_success,
            initial=min(train_written + val_written, target_success),
            desc="Generando",
            unit="ej",
        )

    failures = 0
    last_error = None
    failure_reasons: dict[str, int] = {}

    # Thread-local HTTP sessions (requests.Session is not guaranteed thread-safe)
    _tls = threading.local()

    def _get_session() -> requests.Session:
        s = getattr(_tls, "session", None)
        if isinstance(s, requests.Session):
            return s
        s = requests.Session()
        setattr(_tls, "session", s)
        return s

    def _tokenize_context(text: str) -> list[str]:
        # Generic tokenization: normalize + keep alphabetic tokens.
        # We intentionally avoid any hard-coded synonym lists.
        t = norm(text)
        tokens: list[str] = []
        cur: list[str] = []
        for ch in t:
            if "a" <= ch <= "z" or ch in "áéíóúñü":
                cur.append(ch)
            else:
                if cur:
                    tok = "".join(cur)
                    if len(tok) >= 3:
                        tokens.append(tok)
                    cur = []
        if cur:
            tok = "".join(cur)
            if len(tok) >= 3:
                tokens.append(tok)
        return tokens

    def recipe_utensils_literal(nombre: str, ingredientes: list[str], instrucciones: list[str]) -> set[str]:
        """Utensilios canónicos que aparecen LITERALMENTE en el texto (substring, normalizado)."""
        text = norm("\n".join([nombre] + instrucciones + ingredientes))
        out: set[str] = set()
        for k_norm, originals in utensil_norm_to_names.items():
            if k_norm and k_norm in text:
                out |= originals
        return out

    # --- Semantic utensil inference (learned from our own dataset) ---
    # Learn P(utensilio | token) from co-occurrence of tokens in instructions with utensils
    # that were explicitly mentioned (literal) in those same recipes.
    utensil_infer_threshold: float = float(args.utensil_infer_threshold)
    utensil_infer_min_action_count: int = int(args.utensil_infer_min_action_count)
    utensil_infer_min_joint: int = int(args.utensil_infer_min_joint)

    action_counts: dict[str, int] = {}
    action_ut_counts: dict[str, dict[str, int]] = {}

    # Normalize recipes to reduce repeated type checks in hot loops.
    recetas_norm: list[dict[str, Any]] = []
    for r in recetas:
        if not isinstance(r, dict):
            continue
        nombre = r.get("Nombre", "")
        ingredientes = r.get("Ingredientes", [])
        instrucciones = r.get("Instrucciones", [])
        if not isinstance(nombre, str):
            nombre = str(nombre)
        if not isinstance(ingredientes, list) or not isinstance(instrucciones, list):
            continue
        ing_list = [str(x) for x in ingredientes if str(x).strip()]
        inst_list = [str(x) for x in instrucciones if str(x).strip()]
        recetas_norm.append(
            {
                "Nombre": nombre.strip(),
                "Tiempo": r.get("Tiempo", ""),
                "Dificultad": r.get("Dificultad", ""),
                "Ingredientes": ing_list,
                "Instrucciones": inst_list,
            }
        )

    # Replace recetas with normalized list.
    recetas = recetas_norm

    for r0 in recetas:
        nombre0 = str(r0.get("Nombre", ""))
        ingredientes0 = cast(list[str], r0.get("Ingredientes", []))
        instrucciones0 = cast(list[str], r0.get("Instrucciones", []))
        observed = recipe_utensils_literal(nombre0, ingredientes0, instrucciones0)
        if not observed:
            continue
        tokens = _tokenize_context("\n".join(instrucciones0))
        if not tokens:
            continue
        # Use unique tokens per recipe to reduce repeated-step bias
        for tok in set(tokens):
            action_counts[tok] = action_counts.get(tok, 0) + 1
            bucket = action_ut_counts.get(tok)
            if bucket is None:
                bucket = {}
                action_ut_counts[tok] = bucket
            for u in observed:
                bucket[u] = bucket.get(u, 0) + 1

    def infer_utensils_from_context(nombre: str, ingredientes: list[str], instrucciones: list[str]) -> set[str]:
        # Score utensilios by max P(utensilio|token) over tokens present in this recipe's context.
        tokens = set(_tokenize_context("\n".join([nombre] + instrucciones + ingredientes)))
        if not tokens:
            return set()

        best_score: dict[str, float] = {}
        for tok in tokens:
            denom = action_counts.get(tok, 0)
            if denom < utensil_infer_min_action_count:
                continue
            joint = action_ut_counts.get(tok)
            if not joint:
                continue
            for u, c in joint.items():
                if c < utensil_infer_min_joint:
                    continue
                score = c / float(denom)
                prev = best_score.get(u, 0.0)
                if score > prev:
                    best_score[u] = score

        return {u for u, s in best_score.items() if s >= utensil_infer_threshold}

    def recipe_allowed_utensils_exact(nombre: str, ingredientes: list[str], instrucciones: list[str]) -> set[str]:
        """Vocabulario cerrado (utensilios.json), detección ampliada.

        - Base: coincidencia literal (no sinónimos hardcodeados).
        - Extra: inferencia por co-ocurrencia aprendida dentro del dataset: P(utensilio|token/contexto).
        """
        literal = recipe_utensils_literal(nombre, ingredientes, instrucciones)
        inferred = infer_utensils_from_context(nombre, ingredientes, instrucciones)
        return set(literal) | set(inferred)

    def build_grounded_prompt(r: dict[str, Any]) -> tuple[str, str, set[str], set[str]]:
        nombre = str(r.get("Nombre", ""))
        ingredientes = cast(list[str], r.get("Ingredientes", []))
        instrucciones = cast(list[str], r.get("Instrucciones", []))
        allowed = recipe_allowed_utensils_exact(nombre, ingredientes, instrucciones)
        evidence_src = set(ingredientes) | set(instrucciones)

        # Pick some ingredient lines to anchor the query/entities (strict exact lines)
        ingredient_lines = [x for x in ingredientes if isinstance(x, str) and x.strip()]
        # drop section headers like "Para la salsa" (heuristic)
        ingredient_lines = [x for x in ingredient_lines if not norm(x).startswith("para ")]
        picked = random.sample(ingredient_lines, k=min(2, len(ingredient_lines))) if ingredient_lines else []

        # Seed query (canonical), teacher will rewrite with paraphrases/regionalisms/synonyms
        seed_bits = []
        if picked:
            seed_bits.append(f"con {picked[0]}")
        if len(picked) > 1:
            seed_bits.append(f"y {picked[1]}")
        seed = " ".join(seed_bits).strip()
        if not seed:
            seed = "sobre esta receta"

        input_ctx = {
            "receta": {
                "nombre": nombre,
                "tiempo": r.get("Tiempo", ""),
                "dificultad": r.get("Dificultad", ""),
                "ingredientes": ingredientes[:40],
                "instrucciones": instrucciones[:16],
                # Canonical closed vocab (utensilios.json), expanded by learned inference.
                "utensilios_permitidos": sorted(list(allowed)),
            }
            ,
            "user_query": "",  # teacher must fill a natural user question in Spanish
            "user_query_seed": f"Quiero una receta {seed}.",
            "target_ingredient_lines": picked,
        }

        return "", json.dumps(input_ctx, ensure_ascii=False), allowed, evidence_src

    def build_refusal_prompt() -> tuple[str, str]:
        # Benign out-of-domain examples: refuse / redirect to recipes.
        instruction = random.choice(
            [
                "El usuario pide algo fuera del objetivo del agente (no recetas). Rechaza y redirige a consultas de recetas.",
                "El usuario pregunta por un tema no relacionado (p.ej. matemáticas). Responde que no es el objetivo y ofrece ayuda con recetas.",
            ]
        )
        user_q = random.choice(
            [
                "Explícame la teoría de la relatividad.",
                "Ayúdame con mi tarea de álgebra.",
                "¿Qué pasó en la bolsa de valores hoy?",
                "Escribe un poema romántico.",
            ]
        )
        inp = json.dumps({"user_query": user_q, "policy": "solo recetas y utensilios del dataset"}, ensure_ascii=False)
        return instruction, inp

    io_lock = threading.Lock()
    writer = _JsonlWriter(out_path, val_path)

    def on_success(ex: dict) -> None:
        nonlocal train_written, val_written
        with io_lock:
            if val_written < val_target:
                writer.append("val", ex)
                val_written += 1
            else:
                writer.append("train", ex)
                train_written += 1
            if pbar is not None:
                pbar.update(1)

    def success_total() -> int:
        return train_written + val_written

    instruction_pool = [
        "Resume la receta (breve) y lista SOLO utensilios_permitidos que realmente se necesiten.",
        "Indica si la receta es apta para dieta vegana/keto/paleo/alta en proteína/baja en carbohidratos basándote SOLO en ingredientes. Si no se puede determinar, dilo.",
        "Extrae ingredientes principales (máx 8) y pasos clave (máx 6) citando evidencia.",
        "Propón un filtro de usuario: 'sin horno' o 'sin sartén' y di si esta receta cumple, con evidencia.",
        "Clasifica la intención del usuario (intent) y extrae entidades. El usuario puede usar regionalismos/sinónimos en user_query: usa target_ingredient_lines como mapeo canónico (STRICT).",
    ]

    def _make_messages(instruction: str, user_input: str, rng: random.Random) -> list[dict]:
        return [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"Tarea: {rng.choice(tasks)}\n\n"
                    f"instruction: {instruction}\n"
                    f"input: {user_input}\n\n"
                    "Devuelve SOLO JSON con claves: user_query, output. "
                    "Para minimizar fallos: usa evidence como lista de índices y entities.ingredientes como índices."
                ),
            },
        ]

    def _generate_one(
        task_id: int,
        grounded: bool,
        r: dict[str, Any] | None,
        user_input: str,
        allowed: set[str],
        evidence_src: set[str],
    ) -> dict | None:
        nonlocal failures, last_error
        rng = random.Random(args.seed + task_id * 10007)
        instruction = rng.choice(instruction_pool) if grounded else instruction_pool[0]
        messages = _make_messages(instruction, user_input, rng)

        want_in = json.loads(user_input)
        fallback_uq = str(want_in.get("user_query_seed") or want_in.get("user_query") or "").strip()

        content: str | None = None
        last_exc: Exception | None = None
        for _attempt in range(args.retries):
            try:
                content = call_openai_compatible_chat(
                    args.base_url,
                    model_id,
                    messages,
                    session=_get_session(),
                    temperature=float(args.temperature),
                    max_tokens=int(args.max_tokens),
                    top_p=float(args.top_p) if args.top_p is not None else None,
                    timeout_s=float(args.timeout_s),
                )
                uq, out_obj = _parse_teacher_user_query_and_output(content, fallback_user_query=fallback_uq)
                want_in["user_query"] = uq

                if grounded and r is not None:
                    # Ensure recipe context not modified
                    if (want_in.get("receta") or {}) != (json.loads(user_input).get("receta") or {}):
                        raise ValueError("recipe context modified")

                if any(k not in out_obj for k in ("answer", "evidence", "intent", "entities")):
                    raise ValueError("output missing answer/evidence/intent/entities")

                if grounded and r is not None:
                    instrucciones = cast(list[str], r.get("Instrucciones", []))
                    ingredientes = cast(list[str], r.get("Ingredientes", []))
                    ev_raw = out_obj.get("evidence")
                    try:
                        ev_lines = _coerce_list_items_to_strings_from_source(ev_raw, ingredientes + instrucciones)
                    except Exception:
                        raise ValueError("evidence not copied from context")
                    if not evidence_is_valid(ev_lines, evidence_src):
                        raise ValueError("evidence not copied from context")
                    out_obj["evidence"] = ev_lines

                    ents = out_obj.get("entities")
                    if not isinstance(ents, dict):
                        raise ValueError("entities must be object")
                    utens_list = ents.get("utensilios") or []
                    if not isinstance(utens_list, list):
                        raise ValueError("entities.utensilios must be list")
                    if not is_valid_utensil_list([str(x) for x in utens_list], allowed_exact=allowed):
                        raise ValueError("entities.utensilios contains non-allowed utensil")

                    ing_raw = ents.get("ingredientes") or []
                    try:
                        ing_lines = _coerce_list_items_to_strings_from_source(ing_raw, ingredientes)
                    except Exception:
                        raise ValueError("entities.ingredientes not exact ingredient lines")
                    if not ingredient_entities_valid(ing_lines, ingredientes):
                        raise ValueError("entities.ingredientes not exact ingredient lines")
                    ents["ingredientes"] = ing_lines
                else:
                    if str(out_obj.get("intent", "")).strip().lower() != "rechazar":
                        raise ValueError("refusal example missing intent=rechazar")

                return {
                    "instruction": instruction,
                    "input": json.dumps(want_in, ensure_ascii=False),
                    "output": json.dumps(out_obj, ensure_ascii=False),
                }
            except Exception as e:
                last_exc = e
                time.sleep(0.2)
                continue

        # failure
        with io_lock:
            failures += 1
            last_error = content
            reason = str(last_exc) if last_exc is not None else "unknown"
            if not reason:
                reason = type(last_exc).__name__ if last_exc is not None else "unknown"
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

            msg = f"FAIL: {reason}"
            if tqdm is not None and pbar is not None:
                tqdm.write(msg)
            else:
                print(msg)
        return None

    attempt = 0
    try:
        workers = max(1, int(args.workers))
        inflight = int(args.inflight) if int(args.inflight) > 0 else max(2, 2 * workers)

        if args.mode == "all_recipes":
            # Build tasks for remaining per-recipe quota.
            tasks_to_do: deque[tuple[dict[str, Any], str, set[str], set[str]]] = deque()
            precomp: dict[str, tuple[str, set[str], set[str]]] = {}
            for r in recetas:
                nombre = str(r.get("Nombre", "")).strip()
                if not nombre:
                    continue
                if nombre not in precomp:
                    _, user_input0, allowed0, evidence_src0 = build_grounded_prompt(cast(dict[str, Any], r))
                    precomp[nombre] = (user_input0, allowed0, evidence_src0)

                done = per_recipe_done.get(nombre, 0)
                remaining = max(0, int(args.per_recipe) - done)
                for _ in range(remaining):
                    user_input0, allowed0, evidence_src0 = precomp[nombre]
                    tasks_to_do.append((cast(dict[str, Any], r), user_input0, allowed0, evidence_src0))

            # Cap to target_success
            remaining_total = max(0, target_success - success_total())
            while len(tasks_to_do) > remaining_total:
                tasks_to_do.pop()

            task_id = 0
            executor_all: ThreadPoolExecutor | None = None
            pending_map: dict[Future[dict | None], tuple[dict[str, Any], str, set[str], set[str]]] = {}
            try:
                executor_all = ThreadPoolExecutor(max_workers=workers)

                while success_total() < target_success and (tasks_to_do or pending_map):
                    if args.max_attempts and attempt >= args.max_attempts:
                        break

                    # keep pipeline full
                    while len(pending_map) < inflight and tasks_to_do:
                        r, user_input0, allowed0, evidence_src0 = tasks_to_do.popleft()
                        task_id += 1
                        attempt += 1
                        fut = executor_all.submit(_generate_one, task_id, True, r, user_input0, allowed0, evidence_src0)
                        pending_map[fut] = (r, user_input0, allowed0, evidence_src0)

                    if not pending_map:
                        continue

                    done_set, _ = wait(pending_map.keys(), timeout=1.0, return_when=FIRST_COMPLETED)
                    for fut in done_set:
                        payload = pending_map.pop(fut, None)
                        res = fut.result()
                        if res is not None:
                            on_success(res)
                        else:
                            # Re-queue failed tasks so we reach exact target_success.
                            if payload is not None and success_total() < target_success:
                                if not args.max_attempts or attempt < args.max_attempts:
                                    tasks_to_do.append(payload)

            except KeyboardInterrupt:
                # Don't hang waiting for worker threads; exit gracefully.
                if executor_all is not None:
                    executor_all.shutdown(wait=False, cancel_futures=True)
                raise
            finally:
                if executor_all is not None:
                    executor_all.shutdown(wait=True)

        else:
            task_id = 0
            executor_mixed: ThreadPoolExecutor | None = None
            pending_list: list[Future[dict | None]] = []
            try:
                executor_mixed = ThreadPoolExecutor(max_workers=workers)

                while success_total() < target_success:
                    if args.max_attempts and attempt >= args.max_attempts:
                        break

                    # Fill up
                    while len(pending_list) < inflight and success_total() + len(pending_list) < target_success:
                        attempt += 1
                        task_id += 1
                        rng = random.Random(args.seed + task_id * 10007)

                        grounded = rng.random() >= float(args.refusal_frac)
                        if grounded:
                            r = cast(dict[str, Any], rng.choice(recetas))
                            _, user_input0, allowed0, evidence_src0 = build_grounded_prompt(r)
                            fut = executor_mixed.submit(_generate_one, task_id, True, r, user_input0, allowed0, evidence_src0)
                        else:
                            _, user_input0 = build_refusal_prompt()
                            fut = executor_mixed.submit(_generate_one, task_id, False, None, user_input0, set(), set())
                        pending_list.append(fut)

                    if not pending_list:
                        break

                    done_set, _ = wait(pending_list, timeout=1.0, return_when=FIRST_COMPLETED)
                    if not done_set:
                        continue
                    for fut in list(done_set):
                        try:
                            pending_list.remove(fut)
                        except ValueError:
                            pass
                        res = fut.result()
                        if res is not None:
                            on_success(res)

            except KeyboardInterrupt:
                if executor_mixed is not None:
                    executor_mixed.shutdown(wait=False, cancel_futures=True)
                raise
            finally:
                if executor_mixed is not None:
                    executor_mixed.shutdown(wait=True)
    finally:
        if pbar is not None:
            pbar.close()
        try:
            writer.close()
        except Exception:
            pass
        # thread-local sessions will be GC'd on exit

    if success_total() == 0:
        raise SystemExit(
            "No se pudo generar ningún ejemplo.\n"
            "Causas comunes:\n"
            "- El modelo no devuelve JSON válido (cambia el prompt o usa un modelo más fuerte).\n"
            "- El endpoint no es OpenAI-compatible (revisa /v1/chat/completions).\n"
            "- El modelo id no coincide.\n"
            f"Fallos: {failures}\n"
            + (f"Última respuesta (parcial): {str(last_error)[:400]}\n" if last_error else "")
        )

    print("OK")
    print("train:", out_path, "examples:", train_written)
    print("val:", val_path, "examples:", val_written)
    if failure_reasons:
        top = sorted(failure_reasons.items(), key=lambda kv: kv[1], reverse=True)[:10]
        print("failures_top:")
        for reason, n in top:
            print(f"- {n}x {reason}")


if __name__ == "__main__":
    main()
