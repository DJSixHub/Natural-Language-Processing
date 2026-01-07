from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from red_neuronal.data.jsonl_dataset import format_prompt_and_answer


def _jsonl_has_any_row(path: Path) -> bool:
    if not path.exists() or path.is_dir():
        return False
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return True
    return False


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists() or path.is_dir():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _iter_training_texts(paths: list[Path]):
    for p in paths:
        if not p.exists() or p.is_dir():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    ex = json.loads(s)
                except Exception:
                    continue

                try:
                    prompt, answer = format_prompt_and_answer(ex)
                except Exception:
                    # Fallback: at least learn from raw line.
                    yield s
                    continue

                if prompt:
                    yield prompt
                if answer:
                    yield answer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--val-jsonl", required=True)
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[0] / "model"))
    ap.add_argument("--vocab-size", type=int, default=32000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(args.train_jsonl)
    val_path = Path(args.val_jsonl)
    if not _jsonl_has_any_row(train_path) or not _jsonl_has_any_row(val_path):
        raise SystemExit(
            "Los JSONL de entrenamiento/validación están vacíos (o no existen).\n"
            f"- train: {train_path}\n"
            f"- val:   {val_path}\n"
            "Primero genera datos con generate_qa_lmstudio.py."
        )

    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
    )

    jsonl_paths = [train_path, val_path]
    total_rows = sum(_count_jsonl_rows(p) for p in jsonl_paths)
    # We yield up to 2 strings per row (prompt + answer).
    approx_length = max(1, total_rows * 2)
    
    print(f"Entrenando tokenizer BPE...")
    print(f"  - Ejemplos JSONL: {total_rows} (~{approx_length} textos)")
    print(f"  - Vocab size: {args.vocab_size}")
    print(f"  - Archivos: {[str(p) for p in jsonl_paths]}")
    print("Procesando (puede tardar 1-3 minutos)...\n")
    
    tok.train_from_iterator(_iter_training_texts(jsonl_paths), trainer=trainer, length=approx_length)

    print("Entrenamiento completo. Guardando tokenizer...")
    tok_path = out_dir / "tokenizer.json"
    tmp_path = out_dir / "tokenizer.json.tmp"
    tok.save(str(tmp_path))
    tmp_path.replace(tok_path)

    print(f"✓ Tokenizer guardado en: {tok_path}")
    print(f"  Tokens en vocabulario: {tok.get_vocab_size()}")


if __name__ == "__main__":
    main()
