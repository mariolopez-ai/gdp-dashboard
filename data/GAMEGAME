import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="Máquina de Aprendizaje Manual (Perceptrón)", layout="wide")

st.title("🧠 Perceptrón Interactivo: Estilo Welch Labs")
st.write("Ajusta los pesos manualmente para clasificar los patrones. ¡Tú eres el algoritmo de entrenamiento!")

# --- BARRA LATERAL: Configuración de Objetivos ---
st.sidebar.header("1. Definir Objetivos (Etiquetas)")
st.sidebar.write("Indica qué salida quieres para cada combinación:")

targets = []
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]

for i, inp in enumerate(inputs):
    target = st.sidebar.radio(f"Entrada {inp}:", options=[0, 1], key=f"target_{i}", horizontal=True)
    targets.append(target)

# --- CUERPO PRINCIPAL: Perillas (Weights & Bias) ---
st.header("2. Ajustar Perillas (Parámetros)")
col1, col2, col3 = st.columns(3)

with col1:
    w1 = st.slider("Peso $w_1$", -2.0, 2.0, 0.0, 0.1)
with col2:
    w2 = st.slider("Peso $w_2$", -2.0, 2.0, 0.0, 0.1)
with col3:
    bias = st.slider("Bias (Sesgo)", -2.0, 2.0, 0.0, 0.1)

# --- CÁLCULOS ---
def step_function(z):
    return 1 if z > 0 else 0

correct_count = 0
results = []

for i, x in enumerate(inputs):
    z = (x[0] * w1) + (x[1] * w2) + bias
    prediction = step_function(z)
    is_correct = prediction == targets[i]
    if is_correct:
        correct_count += 1
    results.append({
        "input": x,
        "target": targets[i],
        "z": round(z, 2),
        "output": prediction,
        "correct": is_correct
    })

# --- VISUALIZACIÓN ---
st.header("3. Visualización en Tiempo Real")
c_plot, c_stats = st.columns([2, 1])

with c_stats:
    st.subheader("Estado del Perceptrón")
    st.metric("Patrones Correctos", f"{correct_count}/4")
    
    for res in results:
        icon = "✅" if res['correct'] else "❌"
        st.write(f"{icon} Entrada {res['input']}: Suma={res['z']} ➔ Salida={res['output']} (Deseado={res['target']})")

with c_plot:
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Dibujar puntos
    for res in results:
        color = 'blue' if res['target'] == 1 else 'red'
        marker = 'o' if res['output'] == res['target'] else 'x'
        ax.scatter(res['input'][0], res['input'][1], c=color, s=200, edgecolors='black', label=f"Clase {res['target']}")
    
    # Dibujar Frontera de Decisión: w1*x1 + w2*x2 + b = 0
    # Despejando x2: x2 = (-w1*x1 - b) / w2
    x_vals = np.array([-0.5, 1.5])
    if w2 != 0:
        y_vals = (-w1 * x_vals - bias) / w2
        ax.plot(x_vals, y_vals, '--g', label="Frontera de decisión")
    elif w1 != 0:
        # Caso línea vertical
        ax.axvline(-bias/w1, color='g', linestyle='--', label="Frontera de decisión")

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("Entrada $x_1$")
    ax.set_ylabel("Entrada $x_2$")
    ax.grid(True, alpha=0.3)
    ax.set_title("Plano de Clasificación 2D")
    st.pyplot(fig)

st.markdown("---")
st.info("**Tip:** Intenta replicar una compuerta AND (solo [1,1] es azul) o una OR (todos menos [0,0] son azules).")
