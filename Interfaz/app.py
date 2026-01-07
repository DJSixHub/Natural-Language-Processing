"""
Agente de Recetas - Interfaz de Usuario Unificada
Soporta modo baseline (TF-IDF) y modo neural (Híbrido)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import streamlit as st

# Setup paths
REPO_ROOT = Path(__file__).resolve().parents[1]
RED_NEURONAL_DIR = REPO_ROOT / "Red Neuronal"

for p in [REPO_ROOT, RED_NEURONAL_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Import agents
from Agente.agent import CookingAgent


def load_hybrid_agent():
    """Lazy load hybrid agent to avoid loading neural model unnecessarily."""
    spec = importlib.util.spec_from_file_location("agent_module", RED_NEURONAL_DIR / "agent.py")
    if spec and spec.loader:
        agent_module = importlib.util.module_from_spec(spec)
        sys.modules["agent_module"] = agent_module
        spec.loader.exec_module(agent_module)
        return agent_module.HybridRecipeAgent
    raise ImportError(f"Cannot load agent from {RED_NEURONAL_DIR}")


# Page config
st.set_page_config(
    page_title="Agente de Recetas",
    page_icon="🍳",
    layout="wide",
)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuración")
    
    agent_mode = st.radio(
        "Modo del Agente",
        options=["Baseline (TF-IDF)", "Neural (Híbrido)"],
        index=1,
        help="Baseline: solo recuperación TF-IDF.\nNeural: TF-IDF + modelo neural 17M params"
    )
    
    use_neural = agent_mode == "Neural (Híbrido)"
    
    if use_neural:
        top_k = st.slider(
            "Top-K recetas",
            min_value=1,
            max_value=10,
            value=3,
            help="Número de recetas a recuperar"
        )
    
    st.divider()
    
    if use_neural:
        st.markdown("""
        ### 🤖 Modelo Neural
        - **Parámetros**: 17.1M
        - **Arquitectura**: Decoder-only Transformer  
        - **Entrenamiento**: 8 epochs, 1078 ejemplos
        - **Val Loss**: 3.71
        
        ### 💾 Características
        ✅ Offline (sin internet)  
        ✅ GPU acelerado (CUDA)  
        ✅ Español nativo
        """)
    else:
        st.markdown("""
        ### 📊 Modo Baseline
        - **Recuperación**: TF-IDF
        - **Filtros**: dietas, tiempo, ingredientes
        - **Velocidad**: Ultra-rápido
        
        ### 💾 Características
        ✅ Offline (sin internet)  
        ✅ CPU eficiente  
        ✅ Español nativo
        """)
    
    if st.button("🗑️ Limpiar conversación"):
        st.session_state.messages = []
        st.rerun()


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize or switch agent
mode_key = f"agent_mode_{use_neural}"
if "current_mode" not in st.session_state or st.session_state.current_mode != mode_key:
    with st.spinner("Cargando modelo neural..." if use_neural else "Cargando agente baseline..."):
        if use_neural:
            try:
                HybridRecipeAgent = load_hybrid_agent()
                st.session_state.agent = HybridRecipeAgent(
                    recetas_json=REPO_ROOT / "Jsons_Scrappeados" / "recetas.json",
                    artifacts_dir=RED_NEURONAL_DIR / "artifacts",
                    model_path=RED_NEURONAL_DIR / "model" / "final.pt",
                    tokenizer_path=RED_NEURONAL_DIR / "model" / "tokenizer.json",
                    use_neural=True,
                )
                st.session_state.agent_type = "hybrid"
            except Exception as e:
                st.error(f"Error cargando modelo neural: {e}")
                st.info("Cambiando a modo baseline...")
                st.session_state.agent = CookingAgent()
                st.session_state.agent_type = "baseline"
                use_neural = False
        else:
            st.session_state.agent = CookingAgent()
            st.session_state.agent_type = "baseline"
        
        st.session_state.current_mode = mode_key

# Title
if use_neural:
    st.title("🍳 Agente Neural de Recetas")
    st.caption("TF-IDF Retrieval + Neural Language Model · Offline · Español")
else:
    st.title("🍳 Agente de Recetas (Baseline)")
    st.caption("TF-IDF Retrieval con filtros inteligentes · Offline · Español")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["content"])
        else:
            # Check if it's a hybrid response with structured data
            if use_neural and st.session_state.agent_type == "hybrid" and "retrieval" in message:
                st.markdown("**📋 Recetas Encontradas:**")
                for i, res in enumerate(message["retrieval"][:top_k], 1):
                    recipe = res["receta"]
                    score = res["score"]
                    with st.expander(f"{i}. {recipe['nombre']} (relevancia: {score:.2f})", expanded=(i == 1)):
                        st.markdown(f"**Tiempo:** {recipe.get('tiempo', 'N/D')}")
                        st.markdown(f"**Dificultad:** {recipe.get('dificultad', 'N/D')}")
                        
                        st.markdown("**Ingredientes:**")
                        for ing in recipe["ingredientes"][:8]:
                            st.markdown(f"- {ing}")
                        if len(recipe["ingredientes"]) > 8:
                            st.caption(f"... y {len(recipe['ingredientes']) - 8} más")
                
                if message.get("neural_parsed"):
                    st.divider()
                    st.markdown("**🤖 Respuesta Neural:**")
                    parsed = message["neural_parsed"]
                    
                    if "intent" in parsed:
                        st.info(f"**Intención:** {parsed['intent']}")
                    
                    if "answer" in parsed:
                        st.markdown(parsed["answer"])
                    
                    if "entities" in parsed and isinstance(parsed["entities"], dict):
                        ents = parsed["entities"]
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if ings := ents.get("ingredientes"):
                                st.markdown("**Ingredientes mencionados:**")
                                for ing in ings[:5]:
                                    st.caption(f"• {ing}")
                        
                        with col2:
                            if utils := ents.get("utensilios"):
                                st.markdown("**Utensilios:**")
                                for u in utils[:5]:
                                    st.caption(f"• {u}")
                    
                    with st.expander("Ver JSON completo"):
                        st.json(parsed)
                
                elif message.get("neural_error"):
                    st.warning(f"⚠️ {message['neural_error']}")
                    with st.expander("Ver salida raw"):
                        st.code(message.get("neural_raw", ""), language="text")
            else:
                # Baseline response
                st.write(message.get("content", ""))

# Chat input
if prompt := st.chat_input("Pregunta sobre recetas..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Procesando..." if not use_neural else "Analizando con neural LM..."):
            try:
                if use_neural and st.session_state.agent_type == "hybrid":
                    # Hybrid mode
                    result = st.session_state.agent.query(prompt, top_k=top_k)
                    
                    if result.get("retrieval"):
                        st.markdown("**📋 Recetas Encontradas:**")
                        for i, res in enumerate(result["retrieval"][:top_k], 1):
                            recipe = res["receta"]
                            score = res["score"]
                            with st.expander(f"{i}. {recipe['nombre']} (relevancia: {score:.2f})", expanded=(i == 1)):
                                st.markdown(f"**Tiempo:** {recipe.get('tiempo', 'N/D')}")
                                st.markdown(f"**Dificultad:** {recipe.get('dificultad', 'N/D')}")
                                
                                st.markdown("**Ingredientes:**")
                                for ing in recipe["ingredientes"][:8]:
                                    st.markdown(f"- {ing}")
                                if len(recipe["ingredientes"]) > 8:
                                    st.caption(f"... y {len(recipe['ingredientes']) - 8} más")
                        
                        if result.get("neural_parsed"):
                            st.divider()
                            st.markdown("**🤖 Respuesta Neural:**")
                            parsed = result["neural_parsed"]
                            
                            if "intent" in parsed:
                                st.info(f"**Intención:** {parsed['intent']}")
                            
                            if "answer" in parsed:
                                st.markdown(parsed["answer"])
                            
                            if "entities" in parsed and isinstance(parsed["entities"], dict):
                                ents = parsed["entities"]
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    if ings := ents.get("ingredientes"):
                                        st.markdown("**Ingredientes mencionados:**")
                                        for ing in ings[:5]:
                                            st.caption(f"• {ing}")
                                
                                with col2:
                                    if utils := ents.get("utensilios"):
                                        st.markdown("**Utensilios:**")
                                        for u in utils[:5]:
                                            st.caption(f"• {u}")
                            
                            with st.expander("Ver JSON completo"):
                                st.json(parsed)
                        
                        elif result.get("neural_error"):
                            st.warning(f"⚠️ {result['neural_error']}")
                            with st.expander("Ver salida raw"):
                                st.code(result.get("neural_raw", ""), language="text")
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "retrieval": result.get("retrieval"),
                            "neural_parsed": result.get("neural_parsed"),
                            "neural_raw": result.get("neural_raw"),
                            "neural_error": result.get("neural_error"),
                        })
                    else:
                        msg = result.get("answer", "No encontré recetas relevantes.")
                        st.write(msg)
                        st.session_state.messages.append({"role": "assistant", "content": msg})
                
                else:
                    # Baseline mode
                    resp = st.session_state.agent.respond(prompt)
                    st.write(resp.text)
                    st.session_state.messages.append({"role": "assistant", "content": resp.text})
            
            except FileNotFoundError as e:
                error_msg = (
                    f"❌ Faltan artefactos. Ejecuta primero:\n"
                    f"`python \"Red Neuronal/build_index.py\"`\n\n"
                    f"Detalle: {e}"
                )
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
            except Exception as e:
                error_msg = f"❌ Error: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.divider()
if use_neural:
    st.caption("Powered by: TF-IDF + Custom Transformer (17M params) · PyTorch + CUDA · 100% Offline")
else:
    st.caption("Powered by: TF-IDF + Scikit-learn · Python · 100% Offline")
