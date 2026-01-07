# Neural Recipe Agent - Production System

## 📁 Estructura del Proyecto

```
Natural-Language-Processing/
├── Interfaz/
│   └── app.py              # UI unificada (Streamlit) - Baseline + Neural
├── Agente/                  # Baseline TF-IDF agent
│   ├── agent.py            # Agente principal con filtros
│   ├── retrieval.py        # Índice TF-IDF
│   ├── data.py             # Carga de recetas
│   ├── diets.py            # Inferencia de dietas
│   ├── parse.py            # Parser de constraints
│   └── ...                 # Utilidades
├── Red Neuronal/
│   ├── agent.py            # ⭐ Agente neural unificado (inference + hybrid)
│   ├── train_lm.py         # Entrenamiento del LM
│   ├── train_tokenizer.py  # Entrenamiento del tokenizer
│   ├── build_index.py      # Construcción del índice TF-IDF
│   ├── generate_qa_lmstudio.py  # Generación de dataset de distilación
│   ├── select_representative_recipes.py  # Clustering para reducir dataset
│   ├── red_neuronal/       # Package del modelo
│   │   ├── modeling/
│   │   │   └── transformer.py  # Arquitectura del Transformer
│   │   └── data/
│   │       └── jsonl_dataset.py  # Dataset loader
│   └── model/              # Artefactos entrenados
│       ├── final.pt        # Modelo final
│       ├── tokenizer.json  # Tokenizer BPE
│       └── checkpoints/    # Checkpoints de entrenamiento
├── Jsons_Scrappeados/
│   ├── recetas.json        # Dataset de recetas
│   └── utensilios.json     # Dataset de utensilios
└── Scrappers/              # Scripts de scraping originales
```

## 🚀 Uso del Sistema

### 1. Interfaz Web (Producción)

```bash
cd Natural-Language-Processing
python -m streamlit run Interfaz/app.py
```

**Características:**
- Modo Baseline (TF-IDF): rápido, CPU-efficient
- Modo Neural (Híbrido): TF-IDF + modelo neural 17M params
- Toggle entre modos en tiempo real
- Chat persistente con historial

### 2. CLI Interactivo

```bash
cd "Red Neuronal"

# Modo híbrido interactivo
python agent.py --interactive

# Consulta directa
python agent.py "¿Qué receta vegana puedo hacer en 30 minutos?"
```

### 3. Pipeline de Entrenamiento (si necesitas reentrenar)

```bash
cd "Red Neuronal"

# 1. Generar índice TF-IDF (requerido)
python build_index.py

# 2. Opcional: Reducir dataset con clustering
python select_representative_recipes.py

# 3. Generar datos de distilación (requiere LM Studio con teacher model)
python generate_qa_lmstudio.py --base-url http://127.0.0.1:5000

# 4. Entrenar tokenizer
python train_tokenizer.py --train-jsonl model/distill_train.jsonl --val-jsonl model/distill_val.jsonl

# 5. Entrenar modelo neural
python train_lm.py --train-jsonl model/distill_train.jsonl --val-jsonl model/distill_val.jsonl --tokenizer model/tokenizer.json --epochs 8
```

## 📊 Artefactos Necesarios

### Para Baseline (TF-IDF):
- `Red Neuronal/artifacts/` - Índice TF-IDF
  - `recipes_meta.json`
  - `tfidf_matrix.npz`
  - `vectorizer.joblib`

### Para Neural (Híbrido):
- Todo lo de Baseline +
- `Red Neuronal/model/final.pt` - Modelo entrenado (17M params)
- `Red Neuronal/model/tokenizer.json` - Tokenizer BPE

## 🔧 Scripts de Utilidad

### build_index.py
Construye el índice TF-IDF necesario para la recuperación.

```bash
python build_index.py
```

### select_representative_recipes.py
Reduce el dataset de 10k a ~500 recetas representativas usando clustering.

```bash
python select_representative_recipes.py --k 500
```

### generate_qa_lmstudio.py
Genera dataset de distilación usando un teacher model en LM Studio.

```bash
python generate_qa_lmstudio.py \
  --base-url http://127.0.0.1:5000 \
  --recipes-json representative_recipes.json \
  --workers 4 \
  --target-train 1000
```

## 🎯 Modelo Neural

**Arquitectura:**
- Decoder-only Transformer
- 6 capas, 6 heads, 384 dim, FFN 1024
- 17.1M parámetros
- RoPE, RMSNorm, SwiGLU
- fp16 mixed precision

**Entrenamiento:**
- 8 epochs
- 1,078 ejemplos (distilación)
- Train loss: 2.85
- Val loss: 3.71
- GPU: NVIDIA RTX 4050
- Tiempo: ~2 minutos total

## 📦 Dependencias

Ver `requirements.txt` (baseline) y `Red Neuronal/requirements_neural.txt` (neural)

Principales:
- `streamlit` - UI
- `scikit-learn` - TF-IDF
- `torch` - Neural model
- `tokenizers` - BPE tokenizer

## 🗑️ Archivos Obsoletos/Redundantes Eliminados

Durante la refactorización se consolidaron y eliminaron:

1. ~~`Interfaz/app_neural.py`~~ → Merge to `app.py`
2. ~~`Agente/main.py`~~ → Streamlit es la UI principal
3. ~~`Red Neuronal/inference.py`~~ → Consolidado en `agent.py`
4. ~~`Red Neuronal/hybrid_agent.py`~~ → Consolidado en `agent.py`

## ⚡ Optimizaciones Realizadas

- **UI unificada**: Un solo archivo Streamlit con toggle Baseline/Neural
- **Módulo consolidado**: `Red Neuronal/agent.py` maneja inferencia standalone y hybrid
- **Lazy loading**: El modelo neural solo se carga si se selecciona el modo Neural
- **Cache eficiente**: Streamlit cachea el agente para evitar recargas
- **Imports simplificados**: Path resolution centralizado

## 🔄 Flujo de Trabajo Típico

1. **Desarrollo**: Usa modo Baseline para iteración rápida
2. **Testing Neural**: Cambia a modo Neural en la UI
3. **Reentreno**: Si necesitas mejorar el modelo, ejecuta el pipeline completo
4. **Deploy**: Corre Streamlit con el modelo final.pt

## 📝 Notas

- El sistema es 100% offline después de entrenar
- No requiere internet en runtime
- GPU opcional pero recomendado para modo Neural
- Modo Baseline funciona perfecto solo con CPU
