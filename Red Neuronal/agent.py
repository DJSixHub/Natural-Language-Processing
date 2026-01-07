"""
Unified inference module for recipe agent
Supports both standalone neural inference and hybrid TF-IDF + neural
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

# Add paths for imports
RED_NEURONAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = RED_NEURONAL_DIR.parent

for p in [RED_NEURONAL_DIR, REPO_ROOT]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from red_neuronal.modeling.model import DecoderOnlyTransformerLM, ModelConfig
from Agente.retrieval import load_index, query_index, RecipeIndex
from Agente.data import load_recipes, Recipe


class RecipeAgentInference:
    """Offline neural recipe agent inference."""

    def __init__(
        self,
        model_path: Path | str,
        tokenizer_path: Path | str,
        device: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        # Load tokenizer
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        self.bos_id = self.tokenizer.token_to_id("<bos>")
        self.eos_id = self.tokenizer.token_to_id("<eos>")

        if self.pad_id is None or self.bos_id is None or self.eos_id is None:
            raise RuntimeError("Tokenizer missing <pad>, <bos>, or <eos>")

        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load model
        ckpt = torch.load(self.model_path, map_location="cpu", weights_only=False)
        cfg_dict = ckpt.get("config")
        if cfg_dict is None:
            raise RuntimeError(f"Checkpoint missing 'config': {self.model_path}")
        
        self.config = ModelConfig(**cfg_dict)
        self.model = DecoderOnlyTransformerLM(self.config).to(self.device)
        self.model.load_state_dict(ckpt["model_state"], strict=True)
        self.model.eval()

        print(f"Modelo cargado: {self.model_path}")
        print(f"  Device: {self.device}")
        print(f"  Parámetros: {sum(p.numel() for p in self.model.parameters()):,}")

    def format_prompt(
        self,
        user_query: str,
        receta: dict[str, Any] | None = None,
        instruction: str = "Responde en JSON con: intent, entities (ingredientes, utensilios), answer, evidence.",
    ) -> str:
        """Format prompt matching training style."""
        parts = [
            "### Sistema:\nEres un agente de recetas offline. Responde SIEMPRE en español.\n",
        ]

        if receta:
            receta_json = json.dumps(receta, ensure_ascii=False, separators=(",", ":"))
            parts.append(f"### Contexto (dataset):\n{receta_json}\n")

        parts.append(f"### Usuario:\n{user_query}\n")
        parts.append(f"### Instrucción:\n{instruction}\n")
        parts.append("### Asistente (SOLO JSON):\n")

        return "\n".join(parts)

    @torch.inference_mode()
    def generate(self, prompt: str) -> str:
        """Generate response from prompt."""
        # Tokenize
        prompt_ids = self.tokenizer.encode(prompt).ids
        input_ids = [self.bos_id] + prompt_ids

        # Truncate if needed
        if len(input_ids) > self.config.max_seq_len - self.max_new_tokens:
            input_ids = [self.bos_id] + prompt_ids[-(self.config.max_seq_len - self.max_new_tokens - 1):]

        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # Generate
        for _ in range(self.max_new_tokens):
            if input_tensor.size(1) >= self.config.max_seq_len:
                break

            logits = self.model(input_tensor)
            next_logits = logits[0, -1, :]

            # Temperature + top-p sampling
            if self.temperature > 0:
                next_logits = next_logits / self.temperature
                probs = torch.softmax(next_logits, dim=-1)

                # Top-p (nucleus) sampling
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                cutoff_mask = cumsum_probs > self.top_p
                if cutoff_mask.any():
                    cutoff_idx = cutoff_mask.nonzero(as_tuple=True)[0][0].item()
                    sorted_probs[cutoff_idx + 1:] = 0.0
                    sorted_probs = sorted_probs / sorted_probs.sum()

                next_token = sorted_indices[torch.multinomial(sorted_probs, 1)].item()
            else:
                next_token = next_logits.argmax().item()

            if next_token == self.eos_id:
                break

            input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]], device=self.device)], dim=1)

        # Decode (skip prompt)
        generated_ids = input_tensor[0, len(input_ids):].tolist()
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def answer_query(
        self,
        user_query: str,
        receta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Answer a user query about a recipe.

        Returns dict with keys:
          - raw_output: the model's generated text
          - parsed: parsed JSON if valid, else None
          - error: error message if parsing failed
        """
        prompt = self.format_prompt(user_query, receta)
        raw_output = self.generate(prompt).strip()

        result: dict[str, Any] = {"raw_output": raw_output, "parsed": None, "error": None}

        # Try multiple JSON extraction strategies
        parsed = self._try_parse_json(raw_output)
        if parsed:
            result["parsed"] = parsed
        else:
            result["error"] = "Could not parse valid JSON from model output"

        return result

    def _try_parse_json(self, text: str) -> dict[str, Any] | None:
        """Try multiple strategies to extract valid JSON"""
        # Strategy 1: Direct parse
        try:
            return json.loads(text)
        except Exception:
            pass

        # Strategy 2: Extract first complete {...}
        try:
            start = text.find("{")
            if start != -1:
                # Find matching closing brace
                depth = 0
                for i in range(start, len(text)):
                    if text[i] == "{":
                        depth += 1
                    elif text[i] == "}":
                        depth -= 1
                        if depth == 0:
                            candidate = text[start:i+1]
                            return json.loads(candidate)
        except Exception:
            pass

        # Strategy 3: Remove markdown fences
        if "```" in text:
            try:
                lines = text.split("\n")
                json_lines = []
                in_fence = False
                for line in lines:
                    if line.strip().startswith("```"):
                        in_fence = not in_fence
                        continue
                    if in_fence or ("{" in line or "}" in line or ": " in line):
                        json_lines.append(line)
                cleaned = "\n".join(json_lines)
                return json.loads(cleaned)
            except Exception:
                pass

        return None


class HybridRecipeAgent:
    """Combines TF-IDF retrieval with neural response generation."""

    def __init__(
        self,
        recetas_json: Path | str,
        artifacts_dir: Path | str,
        model_path: Path | str,
        tokenizer_path: Path | str,
        use_neural: bool = True,
    ):
        self.recipes = load_recipes(Path(recetas_json))
        self.index = load_index(Path(artifacts_dir))
        self.use_neural = use_neural
        self.neural: RecipeAgentInference | None = None

        if use_neural:
            self.neural = RecipeAgentInference(
                model_path=model_path,
                tokenizer_path=tokenizer_path,
                device="auto",
                temperature=0.3,
                top_p=0.9,
            )
            print("Modo: Híbrido (TF-IDF + Neural LM)")
        else:
            print("Modo: Solo TF-IDF (baseline)")

    def query(self, user_query: str, top_k: int = 5) -> dict[str, Any]:
        """
        Process user query with hybrid approach:
        1. Retrieve relevant recipes with TF-IDF
        2. Generate response with neural LM (if enabled)
        """
        # Step 1: Retrieval
        candidates = query_index(self.index, user_query, top_k=top_k)
        
        retrieval_results = [
            {
                "receta": {
                    "nombre": recipe.nombre,
                    "ingredientes": recipe.ingredientes,
                    "instrucciones": recipe.instrucciones,
                    "tiempo": recipe.tiempo_raw,
                    "dificultad": recipe.dificultad,
                },
                "score": float(score),
            }
            for recipe, score in candidates
        ]

        if not retrieval_results:
            return {
                "intent": "sin_resultados",
                "answer": "No encontré recetas que coincidan con tu consulta.",
                "recipes": [],
                "neural_output": None,
            }

        # Step 2: Neural generation (if enabled)
        if self.use_neural and self.neural is not None and retrieval_results:
            # Use top recipe as context
            top_recipe = retrieval_results[0]["receta"]
            neural_result = self.neural.answer_query(user_query, receta=top_recipe)

            return {
                "retrieval": retrieval_results,
                "neural_raw": neural_result["raw_output"],
                "neural_parsed": neural_result["parsed"],
                "neural_error": neural_result["error"],
            }
        else:
            # Baseline: just return retrieval results
            return {
                "retrieval": retrieval_results,
                "neural_raw": None,
                "neural_parsed": None,
                "neural_error": None,
            }


def demo_standalone():
    """Demo of standalone neural inference."""
    import sys
    if len(sys.argv) < 2:
        print("Usage: python agent.py <query>")
        sys.exit(1)

    agent = RecipeAgentInference(
        model_path="model/final.pt",
        tokenizer_path="model/tokenizer.json",
        device="auto",
        temperature=0.3,
    )

    query = " ".join(sys.argv[1:])
    print(f"\nConsulta: {query}\n")

    result = agent.answer_query(query)
    print("=== Raw Output ===")
    print(result["raw_output"])
    print("\n=== Parsed JSON ===")
    if result["parsed"]:
        print(json.dumps(result["parsed"], ensure_ascii=False, indent=2))
    else:
        print(f"Error: {result['error']}")


def demo_hybrid():
    """Interactive demo of hybrid agent."""
    recetas = REPO_ROOT / "Jsons_Scrappeados" / "recetas.json"
    artifacts = RED_NEURONAL_DIR / "artifacts"
    model = RED_NEURONAL_DIR / "model" / "final.pt"
    tokenizer = RED_NEURONAL_DIR / "model" / "tokenizer.json"

    agent = HybridRecipeAgent(
        recetas_json=recetas,
        artifacts_dir=artifacts,
        model_path=model,
        tokenizer_path=tokenizer,
        use_neural=True,
    )

    print("\n" + "="*60)
    print("AGENTE HÍBRIDO DE RECETAS (TF-IDF + Neural LM)")
    print("="*60)
    print("Escribe tu consulta (o 'salir' para terminar)\n")

    while True:
        query = input(">>> ").strip()
        if not query or query.lower() in ["salir", "exit", "quit"]:
            break

        result = agent.query(query, top_k=3)

        print("\n" + "-"*60)
        print("RETRIEVAL (Top 3):")
        for i, res in enumerate(result["retrieval"][:3], 1):
            print(f"  {i}. {res['receta']['nombre']} (score: {res['score']:.3f})")

        print("\nNEURAL OUTPUT:")
        if result["neural_parsed"]:
            print(json.dumps(result["neural_parsed"], ensure_ascii=False, indent=2))
        elif result["neural_error"]:
            print(f"Error: {result['neural_error']}")
            print(f"Raw: {result['neural_raw'][:200]}...")
        else:
            print("(neural generation disabled)")

        print("-"*60 + "\n")


if __name__ == "__main__":
    # Choose demo based on context
    if len(sys.argv) > 1 and sys.argv[1] in ["--interactive", "-i"]:
        demo_hybrid()
    else:
        demo_standalone()
