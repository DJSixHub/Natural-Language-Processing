# Natural-Language-Processing

Proyecto: agente conversacional offline en español para responder preguntas sobre recetas usando los datasets:
- `Jsons_Scrappeados/recetas.json`
- `Jsons_Scrappeados/utensilios.json`

Estructura requerida:
- `Red Neuronal/`: scripts de entrenamiento y artefactos
- `Agente/`: lógica del agente (intenciones, parsing, filtros, recuperación)
- `Interfaz/`: UI Streamlit tipo chat

## Instalación

```powershell
cd E:\Escuela\NLP\Natural-Language-Processing
E:\Escuela\NLP\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Entrenamiento (offline)

1) Construir índice TF‑IDF sobre las 10k recetas:

```powershell
E:\Escuela\NLP\.venv\Scripts\python.exe "Red Neuronal\build_index.py"
```

2) (Opcional) Entrenar clasificador de intención para mejorar detección:

```powershell
E:\Escuela\NLP\.venv\Scripts\python.exe "Red Neuronal\train_intents.py"
```

Los artefactos quedan en `Red Neuronal/artifacts/`.

## Ejecutar

CLI:

```powershell
E:\Escuela\NLP\.venv\Scripts\python.exe Agente\main.py
```

Streamlit:

```powershell
E:\Escuela\NLP\.venv\Scripts\python.exe -m streamlit run Interfaz\app.py
```

## Red Neuronal (entrenamiento propio)

En `Red Neuronal/` hay un Transformer *definido por nosotros* (RoPE + RMSNorm + SwiGLU) y scripts para entrenarlo con **destilación** (generación de Q/A sintéticos) usando un modelo *teacher* local vía LM Studio.

1) Instalar dependencias de entrenamiento (aparte):

```powershell
cd E:\Escuela\NLP\Natural-Language-Processing
E:\Escuela\NLP\.venv\Scripts\python.exe -m pip install -r "Red Neuronal\requirements_neural.txt"
```

Nota: `torch` se recomienda instalarlo manualmente según tu CUDA/CPU.

2) Generar dataset de destilación (requiere LM Studio server activo en `http://127.0.0.1:5000`):

```powershell
E:\Escuela\NLP\.venv\Scripts\python.exe "Red Neuronal\generate_qa_lmstudio.py" --n 10000 --mode all_recipes --per-recipe 1
```

Tip: puedes omitir `--model` para auto-detectar el primero en `/v1/models`, o pasar el id exacto expuesto por LM Studio (p.ej. `--model "google/gemma-3-4b"`).

Robustez ante cortes:
- El generador escribe en `distill_train.jsonl` / `distill_val.jsonl` **mientras genera** (con progress bar).
- Si se interrumpe (luz/PC), al relanzarlo **reanuda** y continúa agregando líneas.
- Para reiniciar desde cero, usa `--overwrite`.

Esto crea:
- `Red Neuronal/model/distill_train.jsonl`
- `Red Neuronal/model/distill_val.jsonl`

3) Entrenar tokenizer (BPE) propio:

```powershell
E:\Escuela\NLP\.venv\Scripts\python.exe "Red Neuronal\train_tokenizer.py" --train-jsonl "Red Neuronal\model\distill_train.jsonl" --val-jsonl "Red Neuronal\model\distill_val.jsonl"
```

4) Entrenar el modelo con checkpoints y validación **cada 2 épocas**:

```powershell
E:\Escuela\NLP\.venv\Scripts\python.exe "Red Neuronal\train_lm.py" --train-jsonl "Red Neuronal\model\distill_train.jsonl" --val-jsonl "Red Neuronal\model\distill_val.jsonl" --tokenizer "Red Neuronal\model\tokenizer.json" --epochs 8 --checkpoint-every 2 --validate-every 2
```

Salida:
- Modelo final: `Red Neuronal/model/final.pt`
- Config: `Red Neuronal/model/config.json`
- Checkpoints: `Red Neuronal/model/checkpoints/epoch_0002.pt`, `epoch_0004.pt`, ...

## Ejemplos de prompts

- "Recomiéndame una receta vegana en menos de 30 minutos"
- "Quiero algo keto sin queso"
- "Dame recetas que usen horno"
- "Recetas sin sartén" (aproximado por detección de utensilios en instrucciones)

Notas:
- Las restricciones halal/kosher se interpretan como "compatibles" por ingredientes; no se garantiza certificación.
