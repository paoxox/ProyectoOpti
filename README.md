# ProyectoOpti

AUXILIOOOOOOOOOOOOOOOO

import streamlit as st

# Configuración de página (debe ser lo primero)
st.set_page_config(
    page_title="Mi Primera App",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título y encabezados
st.title("Laboratorio de IQ 🧪")
st.header("Reactor Batch")
st.subheader("Condiciones Operativas")

# Texto simple
st.text("Simulación de reactor tipo batch")

# Markdown
st.markdown("""
### Parámetros:
- **Temperatura**: 350 K
- **Presión**: 2.5 atm
""")

# Ecuaciones LaTeX
st.latex(r'''
    \frac{dC_A}{dt} = -k \cdot C_A^n
''')

# Ejecutar: streamlit run app.py
