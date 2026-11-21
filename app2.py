import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Configuración general de la página
st.set_page_config(
    page_title="Simulación y Optimización de HDS",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items = {
        "Get help": "https://docs.streamlit.io",
        "Report a bug": None,
        "About": "Sección interna de simulación, versión 1.0",
    },
)
