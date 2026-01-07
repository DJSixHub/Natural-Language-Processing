from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


def _try_parse_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def _to_json_str(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def format_prompt_and_answer(ex: dict) -> tuple[str, str]:
    """Return (prompt, answer).

    The generator writes:
      - instruction: string
      - input: JSON string (often includes receta + user_query)
      - output: JSON string (assistant output with intent/entities/etc)

    We train the model as an agentic chat:
      [context: receta] + [user_query] + [instruction] -> [assistant JSON]
    """
    instruction = (ex.get("instruction") or "").strip()

    raw_inp = ex.get("input")
    if raw_inp is None:
        raw_inp = ""
    inp_str = raw_inp if isinstance(raw_inp, str) else _to_json_str(raw_inp)
    inp_obj = _try_parse_json(inp_str) if isinstance(inp_str, str) else None

    user_query = ""
    receta_ctx = ""

    if isinstance(inp_obj, dict):
        user_query = str(inp_obj.get("user_query") or "").strip()
        if "receta" in inp_obj:
            receta_ctx = _to_json_str(inp_obj.get("receta"))
        else:
            # non-recipe inputs (e.g., refusal) keep full JSON as context
            receta_ctx = _to_json_str(inp_obj)
    else:
        receta_ctx = (inp_str or "").strip()

    raw_out = ex.get("output")
    if raw_out is None:
        raw_out = ""
    answer = raw_out if isinstance(raw_out, str) else _to_json_str(raw_out)
    answer = (answer or "").strip()

    # Prompt style: RAG-ish: context + user + instruction.
    # We explicitly ask for JSON to support agentic routing.
    parts = [
        "### Sistema:\nEres un agente de recetas offline. Responde SIEMPRE en español.\n",
    ]

    if receta_ctx:
        parts.append(f"### Contexto (dataset):\n{receta_ctx}\n")
    if user_query:
        parts.append(f"### Usuario:\n{user_query}\n")
    if instruction:
        parts.append(f"### Instrucción:\n{instruction}\n")

    parts.append("### Asistente (SOLO JSON):\n")
    prompt = "\n".join(parts)
    return prompt, answer


@dataclass(frozen=True)
class TokenizedBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor


class JsonlSFTDataset(Dataset):
    def __init__(self, jsonl_path: Path, tokenizer, max_seq_len: int):
        self.examples = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.examples.append(json.loads(line))
        if not self.examples:
            raise ValueError(
                f"JSONL vacío: {jsonl_path}. Genera datos primero (p.ej. generate_qa_lmstudio.py)."
            )
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> TokenizedBatch:
        prompt, answer = format_prompt_and_answer(self.examples[idx])

        pad_id = self.tokenizer.token_to_id("<pad>")
        bos_id = self.tokenizer.token_to_id("<bos>")
        eos_id = self.tokenizer.token_to_id("<eos>")
        if pad_id is None or bos_id is None or eos_id is None:
            raise RuntimeError("Tokenizer must contain <pad>, <bos>, <eos>.")

        prompt_ids = self.tokenizer.encode(prompt).ids
        answer_ids = self.tokenizer.encode(answer).ids

        # Truncation strategy:
        # Keep at least a small tail of the prompt and at least 1 answer token.
        # This avoids batches where all labels are -100 (can yield NaN losses).
        max_content = max(1, self.max_seq_len - 2)  # excluding <bos>/<eos>
        if len(prompt_ids) + len(answer_ids) > max_content:
            # Prefer keeping more answer than prompt, but never drop prompt completely.
            min_answer = min(64, max_content - 1)  # keep up to 64 answer tokens when possible
            if len(answer_ids) > min_answer:
                answer_ids = answer_ids[:min_answer]

            prompt_budget = max_content - len(answer_ids)
            if prompt_budget < 1:
                # Still too long: force 1 token of prompt and fit answer to the rest.
                prompt_budget = 1
                answer_ids = answer_ids[: max(1, max_content - prompt_budget)]

            if len(prompt_ids) > prompt_budget:
                # Keep the tail (closest to the assistant turn marker).
                prompt_ids = prompt_ids[-prompt_budget:]

        # Build final sequence: <bos> prompt answer <eos>
        ids = [bos_id] + prompt_ids + answer_ids + [eos_id]
        ids = ids[: self.max_seq_len]

        # Mask loss on prompt tokens: only learn to generate answer + eos
        # labels align with ids; training code shifts by 1.
        labels = [-100] * len(ids)

        # Determine answer span start in the truncated sequence.
        answer_start = 1 + len(prompt_ids)
        for i in range(answer_start, len(ids)):
            labels[i] = ids[i]

        # Safety: ensure at least one supervised target.
        if all(x == -100 for x in labels):
            labels[-1] = ids[-1]

        input_ids = torch.tensor(ids, dtype=torch.long)
        labels_t = torch.tensor(labels, dtype=torch.long)
        return TokenizedBatch(input_ids=input_ids, labels=labels_t)


def collate_pad(batch: list[TokenizedBatch], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(x.input_ids.numel() for x in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, ex in enumerate(batch):
        n = ex.input_ids.numel()
        input_ids[i, :n] = ex.input_ids
        labels[i, :n] = ex.labels

    return {"input_ids": input_ids, "labels": labels}
