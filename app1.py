import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ==========================================
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS
# ==========================================
st.set_page_config(
    page_title="Simulación y Optimización HDS",
    page_icon="⚗️", # Icono de la pestaña
    layout="wide"
)

# Inyectar CSS personalizado para colores pastel y diseño limpio
st.markdown("""
    <style>
    /* Fondo principal */
    .stApp {
        background-color: #f4f8fb; /* Azul pastel muy claro */
    }
    /* Barra lateral */
    [data-testid="stSidebar"] {
        background-color: #e1ecf4; /* Azul pastel un poco más oscuro */
    }
    /* Títulos */
    h1, h2, h3 {
        color: #2c4a6b !important; /* Azul marino suave */
        font-family: 'Helvetica Neue', sans-serif;
    }
    /* Cajas de métricas */
    [data-testid="stMetricValue"] {
        color: #1d5b8c !important;
    }
    [data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #c9ddec;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    /* Pestañas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #e1ecf4;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        color: #2c4a6b;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #1d5b8c !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. FUNCIONES MATEMÁTICAS
# ==========================================
def calcular_k0(Ea, tau):
    R = 8.314
    return -np.log(1 - 0.985) / (tau * np.exp(-Ea / (R * 600)))

def calcular_modelo(Tin, eps, mdot, Cp, Ea, tau, Sin, k0):
    R = 8.314
    k = k0 * np.exp(-Ea / (R * Tin))
    Xs = 1 - np.exp(-k * tau)
    Sout = Sin * (1 - Xs)
    Delta_Tad = 15 * Xs
    Tefluente = Tin + Delta_Tad
    Tintercambiador = 300 + eps * (Tefluente - 300)
    Qhorno = mdot * Cp * (Tin - Tintercambiador)
    return Xs, Sout, Tefluente, Tintercambiador, Qhorno

def calcular_pareto(Tin_arr, eps_arr, params):
    T_grid, E_grid = np.meshgrid(Tin_arr, eps_arr)
    T_flat, E_flat = T_grid.flatten(), E_grid.flatten()
    
    Xs, Sout, _, _, Qhorno = calcular_modelo(T_flat, E_flat, **params)
    
    df = pd.DataFrame({
        'Tin': T_flat, 'eps': E_flat, 'Xs': Xs, 'Sout': Sout, 'Qhorno': Qhorno,
        'Factible': Sout <= 10
    })
    
    df_fact = df[df['Factible']].copy()
    if df_fact.empty:
        return df, pd.DataFrame(), None
    
    df_fact = df_fact.sort_values(by=['Xs', 'Qhorno'], ascending=[False, True])
    pareto_indices = []
    min_q = float('inf')
    for idx, row in df_fact.iterrows():
        if row['Qhorno'] < min_q:
            pareto_indices.append(idx)
            min_q = row['Qhorno']
            
    df_pareto = df_fact.loc[pareto_indices].sort_values(by='Qhorno').reset_index(drop=True)
    
    if len(df_pareto) > 1:
        x_norm = (1 - df_pareto['Xs'] - (1 - df_pareto['Xs']).min()) / ((1 - df_pareto['Xs']).max() - (1 - df_pareto['Xs']).min())
        y_norm = (df_pareto['Qhorno'] - df_pareto['Qhorno'].min()) / (df_pareto['Qhorno'].max() - df_pareto['Qhorno'].min())
        distancias = np.abs(x_norm + y_norm - 1) / np.sqrt(2)
        knee_idx = distancias.idxmax()
        knee_point = df_pareto.loc[knee_idx]
    else:
        knee_point = df_pareto.iloc[0]
        
    return df, df_pareto, knee_point

# ==========================================
# 3. CABECERA (Logo y Título)
# ==========================================
col_logo, col_title = st.columns([1, 8])
with col_logo:
    # Puedes reemplazar esta URL por la ruta de tu propio logo, ej: "logo_ucv.png"
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063822.png", width=80) 
with col_title:
    st.title("Simulación y Optimización de Hidrodesulfuración (HDS)")
    st.markdown("**Curso de Optimización de Procesos — Ingeniería Química**")

# ==========================================
# 4. CONTROLES (Barra Lateral)
# ==========================================
st.sidebar.header("⚙️ Parámetros de Operación")
Tin_min, Tin_max = st.sidebar.slider("Rango de Temperatura Tin (K)", 550, 650, (550, 650))
eps_min, eps_max = st.sidebar.slider("Rango de Efectividad ε", 0.60, 0.90, (0.60, 0.90), step=0.01)

st.sidebar.header("🔧 Perturbaciones del Modelo")
mdot = st.sidebar.number_input("Flujo másico (kg/h)", value=100000, step=1000)
Cp = st.sidebar.number_input("Cp (kJ/kg·K)", value=2.5, step=0.1)
Ea = st.sidebar.number_input("Ea (J/mol)", value=80000, step=1000)
tau = st.sidebar.number_input("Tiempo espacial τ (h)", value=0.5, step=0.01)
Sin = st.sidebar.number_input("Azufre entrada (ppm)", value=500, step=10)

# Cálculos base
k0 = calcular_k0(Ea, tau)
params = {'mdot': mdot, 'Cp': Cp, 'Ea': Ea, 'tau': tau, 'Sin': Sin, 'k0': k0}

# Grid Search
n_puntos = 60 
Tin_arr = np.linspace(Tin_min, Tin_max, n_puntos)
eps_arr = np.linspace(eps_min, eps_max, n_puntos)
df_full, df_pareto, knee = calcular_pareto(Tin_arr, eps_arr, params)

# ==========================================
# 5. CUERPO PRINCIPAL (Pestañas)
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Optimización y Pareto", "✅ Validación (Sanity Checks)", "📉 Análisis de Sensibilidad"])

# --- PESTAÑA 1: OPTIMIZACIÓN ---
with tab1:
    col_metrics, col_plot = st.columns([1, 2])
    
    with col_metrics:
        st.subheader("Punto de Rodilla (Knee Point)")
        if knee is not None:
            st.metric("Temperatura (Tin)", f"{knee['Tin']:.1f} K")
            st.metric("Efectividad (ε)", f"{knee['eps']:.3f}")
            st.metric("Conversión (Xs)", f"{knee['Xs']*100:.2f} %")
            st.metric("Sout", f"{knee['Sout']:.2f} ppm")
            st.metric("Carga Horno (Q)", f"{knee['Qhorno']:,.0f} kJ/h")
        else:
            st.error("No hay puntos factibles con los parámetros actuales.")

    with col_plot:
        fig = go.Figure()
        infact = df_full[~df_full['Factible']]
        fig.add_trace(go.Scatter(x=infact['Xs'], y=infact['Qhorno'], mode='markers',
                                 marker=dict(color='#d3d3d3', size=3), name='Infactible (Sout > 10)'))
        
        fact = df_full[df_full['Factible']]
        fig.add_trace(go.Scatter(x=fact['Xs'], y=fact['Qhorno'], mode='markers',
                                 marker=dict(color='#a1c4fd', size=4), name='Factible'))
        
        if not df_pareto.empty:
            fig.add_trace(go.Scatter(x=df_pareto['Xs'], y=df_pareto['Qhorno'], mode='lines+markers',
                                     line=dict(color='#2c4a6b', width=3), marker=dict(size=6), name='Frontera de Pareto'))
            fig.add_trace(go.Scatter(x=[knee['Xs']], y=[knee['Qhorno']], mode='markers',
                                     marker=dict(color='#ff6b6b', size=15, symbol='star'), name='Knee Point'))
            
        fig.update_layout(
            xaxis_title="Conversión de Azufre (Xs)", yaxis_title="Carga Térmica (kJ/h)",
            plot_bgcolor='white', paper_bgcolor='white', hovermode="closest"
        )
        st.plotly_chart(fig, use_container_width=True)

# --- PESTAÑA 2: SANITY CHECKS (Requisito 5 del PDF) ---
with tab2:
    st.subheader("Validación del Modelo")
    st.markdown("Comprobación de los escenarios exigidos en el proyecto:")
    
    # Check 1
    Xs_1, Sout_1, _, _, Q_1 = calcular_modelo(650, 0.90, **params)
    st.info(f"**Check 1 (Tin=650 K, ε=0.90):** Xs = {Xs_1:.4f}, Sout = {Sout_1:.2f} ppm, Q = {Q_1:,.0f} kJ/h. "
            f"**Factible:** {'Sí' if Sout_1 <= 10 else 'No'}")
    
    # Check 2
    Xs_2, Sout_2, _, _, Q_2 = calcular_modelo(550, 0.75, **params)
    st.warning(f"**Check 2 (Tin=550 K, ε=0.75):** Sout = {Sout_2:.2f} ppm. "
               f"**¿Es factible?:** {'Sí' if Sout_2 <= 10 else 'No'}")
    
    # Check 3 (Límite eps -> 1)
    _, _, Tef_3, Tint_3, Q_3 = calcular_modelo(600, 1.0, **params)
    st.success(f"**Check 3 (ε → 1):** Tintercambiador = {Tint_3:.2f} K (≈ Tefluente = {Tef_3:.2f} K). "
               f"Q_horno = {Q_3:,.0f} kJ/h. *Interpretación: Q negativo indica excedente de calor.*")

# --- PESTAÑA 3: SENSIBILIDAD (Requisito 5 del PDF) ---
with tab3:
    st.subheader("Análisis de Sensibilidad (±10%)")
    st.markdown("Evaluación del efecto de perturbar los parámetros en el **Punto de Rodilla** actual.")
    
    if knee is not None:
        # Punto base
        base_T = knee['Tin']
        base_eps = knee['eps']
        
        def evaluar_perturbacion(nombre_param, valor_pert, factor):
            # Copia de parámetros
            p_temp = params.copy()
            p_temp[nombre_param] = valor_pert
            # Recalcular k0 si se perturba Ea o tau
            if nombre_param in ['Ea', 'tau']:
                p_temp['k0'] = calcular_k0(p_temp['Ea'], p_temp['tau'])
                
            _, Sout, _, _, Qhorno = calcular_modelo(base_T, base_eps, **p_temp)
            return Qhorno, Sout

        # Evaluar perturbaciones
        escenarios = [
            ("Base", params['mdot'], params['Cp'], params['Ea'], params['k0'], params['tau'], params['Sin']),
            ("Flujo +10%", params['mdot']*1.1, params['Cp'], params['Ea'], params['k0'], params['tau'], params['Sin']),
            ("Flujo -10%", params['mdot']*0.9, params['Cp'], params['Ea'], params['k0'], params['tau'], params['Sin']),
            ("Cp +10%", params['mdot'], params['Cp']*1.1, params['Ea'], params['k0'], params['tau'], params['Sin']),
            ("Cp -10%", params['mdot'], params['Cp']*0.9, params['Ea'], params['k0'], params['tau'], params['Sin']),
            ("Ea +10%", params['mdot'], params['Cp'], params['Ea']*1.1, calcular_k0(params['Ea']*1.1, params['tau']), params['tau'], params['Sin']),
            ("Ea -10%", params['mdot'], params['Cp'], params['Ea']*0.9, calcular_k0(params['Ea']*0.9, params['tau']), params['tau'], params['Sin']),
            ("k0 +10%", params['mdot'], params['Cp'], params['Ea'], params['k0']*1.1, params['tau'], params['Sin']),
            ("k0 -10%", params['mdot'], params['Cp'], params['Ea'], params['k0']*0.9, params['tau'], params['Sin']),
            ("tau +10%", params['mdot'], params['Cp'], params['Ea'], calcular_k0(params['Ea'], params['tau']*1.1), params['tau']*1.1, params['Sin']),
            ("tau -10%", params['mdot'], params['Cp'], params['Ea'], calcular_k0(params['Ea'], params['tau']*0.9), params['tau']*0.9, params['Sin']),
            ("Sin +10%", params['mdot'], params['Cp'], params['Ea'], params['k0'], params['tau'], params['Sin']*1.1),
            ("Sin -10%", params['mdot'], params['Cp'], params['Ea'], params['k0'], params['tau'], params['Sin']*0.9),
        ]

        resultados = []
        for nombre, m, cp, e, k, t, s in escenarios:
            p = {'mdot': m, 'Cp': cp, 'Ea': e, 'k0': k, 'tau': t, 'Sin': s}
            _, Sout, _, _, Q = calcular_modelo(base_T, base_eps, **p)
            resultados.append({
                "Escenario": nombre,
                "Q_horno (kJ/h)": f"{Q:,.0f}",
                "Sout (ppm)": f"{Sout:.2f}"
            })
            
        st.dataframe(pd.DataFrame(resultados), use_container_width=True, hide_index=True)
    else:
        st.warning("Ajusta los parámetros para encontrar un punto factible primero.")

# Pie de página
st.markdown("---")
st.caption("Desarrollado para el Proyecto Final de Optimización de Procesos. 2026.")
