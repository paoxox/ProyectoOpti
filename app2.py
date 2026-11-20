import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# 1. Configuración general de la página
st.set_page_config(
    page_title="Simulación y Optimización de HDS",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# CABECERA CON LOGO
# ==========================================
col_logo, col_titulo = st.columns([1, 8], vertical_alignment="center")
with col_logo:
    st.image("https://img.magnific.com/premium-photo/nuclear-power-plant-with-nuclear-reactor-illuminated-by-lights-middle-night_124507-45320.jpg?semt=ais_hybrid&w=740&q=80", width=90)
with col_titulo:
    st.title("Simulación y Optimización de Hidrodesulfuración (HDS)")
    st.caption(
        "Evaluación de conversión de azufre ($X_{s}$) y carga térmica del horno ($Q_{horno}$) "
        "bajo restricción de Azufre en la salida($S_{out}$) ≤ 10 ppm."
    )

# Constantes fijas del modelo
R_GAS = 8.314
T_ALIM = 300.0
X_CALIB = 0.985
T_CALIB = 600.0
TAU_CALIB = 0.5
EA_CALIB = 80000.0
DT_AD_MAX = 15.0

def calcular_k0(ea: float = EA_CALIB, tau: float = TAU_CALIB, x_ref: float = X_CALIB, t_ref: float = T_CALIB) -> float:
    return -np.log(1.0 - x_ref) / (tau * np.exp(-ea / (R_GAS * t_ref)))

def evaluar_hds(t_in, epsilon, m_dot, s_in, cp, ea, k0, tau):
    k_t = k0 * np.exp(-ea / (R_GAS * t_in))
    x_s = 1.0 - np.exp(-k_t * tau)
    s_out = s_in * (1.0 - x_s)
    factible = s_out <= 10.0

    delta_t_ad = DT_AD_MAX * x_s
    t_efluente = t_in + delta_t_ad
    t_intercambiador = T_ALIM + epsilon * (t_efluente - T_ALIM)

    q_horno_kjh = m_dot * cp * (t_in - t_intercambiador)
    q_horno_kw = q_horno_kjh / 3600.0

    return {
        "Tin_K": t_in, "epsilon": epsilon, "k_h-1": k_t, "Xs": x_s,
        "Sout_ppm": s_out, "factible": factible, "dT_ad_K": delta_t_ad,
        "T_efluente_K": t_efluente, "T_intercambiador_K": t_intercambiador,
        "Q_horno_kJ_h": q_horno_kjh, "Q_horno_kW": q_horno_kw,
    }

# 2. Controles en barra lateral
with st.sidebar:
    st.header("Variables de decisión")
    t_in = st.slider("Temperatura entrada reactor $T_{in}$(K)", 550.0, 650.0, 600.0, 1.0)
    epsilon = st.slider("Efectividad intercambiador ε", 0.60, 0.90, 0.75, 0.01)

    st.header("Parámetros del modelo")
    m_dot = st.number_input("Flujo másico ṁ (kg/h)", 90000.0, 110000.0, 100000.0, 1000.0)
    s_in = st.number_input("Azufre alimentación $S_{in}$ (ppm)", 450.0, 550.0, 500.0, 5.0)
    cp = st.number_input("Capacidad calorífica $Cp_{}$ (kJ/(kg·K))", 2.25, 2.75, 2.50, 0.05)
    ea = st.number_input("Energía de activación Ea (J/mol)", 72000.0, 88000.0, 80000.0, 500.0)
    tau = st.number_input("Tiempo espacial τ (h)", 0.45, 0.55, 0.50, 0.01)

    k0_base = calcular_k0(ea=ea, tau=tau)
    k0 = st.number_input(
        "Factor preexponencial k0 (h⁻¹)",
        min_value=float(k0_base * 0.90),
        max_value=float(k0_base * 1.10),
        value=float(k0_base),
        format="%.4e",
    )

# 3. Evaluación del punto actual (Sidebar)
res_actual = evaluar_hds(t_in, epsilon, m_dot, s_in, cp, ea, k0, tau)

st.subheader("Punto operativo seleccionado")
if res_actual["factible"]:
    st.success(f"Cumple especificación de calidad: $S_{{out}}$  = {res_actual['Sout_ppm']:.2f} ppm (≤ 10 ppm).", icon="✅")
else:
    st.warning(f"No cumple especificación de calidad: $S_{{out}}$  = {res_actual['Sout_ppm']:.2f} ppm (> 10 ppm).", icon="⚠️")

# Etiquetas con subíndices HTML (formato exacto de la imagen)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Conversión $X_{s}$", f"{res_actual['Xs']*100:.2f}%")
c2.metric("Azufre salida $S_{out}$", f"{res_actual['Sout_ppm']:.2f} ppm", delta_color="inverse")
c3.metric("Carga horno $Q_{horno}$", f"{res_actual['Q_horno_kW']:.2f} kW")
c4.metric("Carga horno $Q_{horno}$", f"{res_actual['Q_horno_kJ_h']:,.0f} kJ/h") # Corregido para que no se corte

st.markdown("---")

# 4. Generación de la Malla (Grid Search) y Pareto VECTORIZADO
@st.cache_data
def calcular_malla_y_pareto(n_puntos=50, m_dot=100000.0, s_in=500.0, cp=2.5, ea=EA_CALIB, k0=None, tau=TAU_CALIB):
    grid_tin = np.linspace(550.0, 650.0, n_puntos)
    grid_eps = np.linspace(0.60, 0.90, n_puntos)
    T_grid, E_grid = np.meshgrid(grid_tin, grid_eps)
    T_flat, E_flat = T_grid.flatten(), E_grid.flatten()

    res = evaluar_hds(t_in=T_flat, epsilon=E_flat, m_dot=m_dot, s_in=s_in, cp=cp, ea=ea, k0=k0, tau=tau)
    
    df_grid = pd.DataFrame(res)
    df_factible = df_grid[df_grid["factible"]].copy()

    if df_factible.empty:
        return df_grid, pd.DataFrame(), None

    df_factible["obj_xs"] = 1.0 - df_factible["Xs"]
    df_factible["obj_q"] = df_factible["Q_horno_kW"]
    df_sorted = df_factible.sort_values(by=["obj_xs", "obj_q"], ascending=[True, True]).reset_index(drop=True)

    pareto_indices = []
    min_q_visto = float("inf")
    for i, row in df_sorted.iterrows():
        if row["obj_q"] < min_q_visto:
            pareto_indices.append(i)
            min_q_visto = row["obj_q"]

    df_pareto = df_sorted.loc[pareto_indices].copy()

    xs_min, xs_max = df_pareto["obj_xs"].min(), df_pareto["obj_xs"].max()
    q_min, q_max = df_pareto["obj_q"].min(), df_pareto["obj_q"].max()

    denom_xs = (xs_max - xs_min) if (xs_max - xs_min) > 0 else 1.0
    denom_q = (q_max - q_min) if (q_max - q_min) > 0 else 1.0

    df_pareto["xs_norm"] = (df_pareto["obj_xs"] - xs_min) / denom_xs
    df_pareto["q_norm"] = (df_pareto["obj_q"] - q_min) / denom_q

    x1, y1 = df_pareto.iloc[0]["xs_norm"], df_pareto.iloc[0]["q_norm"]
    x2, y2 = df_pareto.iloc[-1]["xs_norm"], df_pareto.iloc[-1]["q_norm"]

    a = y2 - y1
    b = -(x2 - x1)
    c = x2 * y1 - y2 * x1
    norma = np.sqrt(a**2 + b**2)

    distancias = []
    for _, row in df_pareto.iterrows():
        x0, y0 = row["xs_norm"], row["q_norm"]
        d = abs(a * x0 + b * y0 + c) / norma if norma != 0 else 0.0
        distancias.append(d)

    df_pareto["distancia_knee"] = distancias
    knee_point = df_pareto.loc[df_pareto["distancia_knee"].idxmax()].to_dict()

    return df_grid, df_pareto, knee_point

# Ejecutar grid search
df_grid, df_pareto, knee_point = calcular_malla_y_pareto(m_dot=m_dot, s_in=s_in, cp=cp, ea=ea, k0=k0, tau=tau)

# 5. Visualización con Pestañas
tab_pareto, tab_sanity, tab_sensibilidad = st.tabs(
    ["📈 Frontera de Pareto", "✅ Sanity Checks y Malla", "🧪 Sensibilidad"]
)

with tab_pareto:
    if df_pareto.empty:
        st.error("No se encontraron puntos factibles en el dominio especificado.")
    else:
        col_graph, col_knee = st.columns([2.5, 1])

        with col_graph:
            fig = go.Figure()
            df_fact = df_grid[df_grid["factible"]]
            df_infact = df_grid[~df_grid["factible"]]

            # Puntos Infactibles
            fig.add_trace(go.Scatter(x=df_infact["Q_horno_kW"], y=df_infact["Xs"]*100, mode="markers",
                                     name="Infactibles (S<sub>out</sub> > 10)", marker=dict(color="rgba(200, 200, 200, 0.3)", size=4)))
            # Puntos Factibles
            fig.add_trace(go.Scatter(x=df_fact["Q_horno_kW"], y=df_fact["Xs"]*100, mode="markers",
                                     name="Factibles", marker=dict(color="#A0C4FF", size=5, opacity=0.6),
                                     hovertemplate="Q<sub>horno</sub>: %{x:.1f} kW<br>X<sub>s</sub>: %{y:.2f}%<br>T<sub>in</sub>: %{customdata[0]:.1f} K<br>ε: %{customdata[1]:.2f}<extra></extra>",
                                     customdata=df_fact[["Tin_K", "epsilon"]].values))
            # Frontera
            fig.add_trace(go.Scatter(x=df_pareto["Q_horno_kW"], y=df_pareto["Xs"]*100, mode="lines+markers",
                                     name="Frontera de Pareto", line=dict(color="#023E8A", width=3), marker=dict(color="#023E8A", size=5),
                                     hovertemplate="<b>Pareto</b><br>Q<sub>horno</sub>: %{x:.1f} kW<br>X<sub>s</sub>: %{y:.2f}%<br>T<sub>in</sub>: %{customdata[0]:.1f} K<br>ε: %{customdata[1]:.2f}<extra></extra>",
                                     customdata=df_pareto[["Tin_K", "epsilon"]].values))
            # Knee Point (Punto de compromiso, sin la "estrella cochina")
            fig.add_trace(go.Scatter(x=[knee_point["Q_horno_kW"]], y=[knee_point["Xs"]*100], mode="markers",
                                     name="Knee Point", marker=dict(color="#FF006E", size=14, symbol="diamond", line=dict(color="white", width=2)),
                                     hovertemplate="<b>Knee Point</b><br>Q<sub>horno</sub>: %{x:.1f} kW<br>X<sub>s</sub>: %{y:.2f}%<extra></extra>"))

            fig.update_layout(title="Optimización Multiobjetivo", xaxis_title="Carga Térmica Q<sub>horno</sub> (kW)",
                              yaxis_title="Conversión X<sub>s</sub> (%)", legend=dict(x=0.01, y=0.01, bgcolor="rgba(255,255,255,0.8)"), height=500)
            st.plotly_chart(fig, use_container_width=True)

        with col_knee:
            st.markdown("### 📍 Knee Point")
            st.info("Punto de equilibrio ideal: la mejor conversión posible con el menor consumo de energía.")
            st.metric("Temperatura $T_{in}$", f"{knee_point['Tin_K']:.1f} K", f"{knee_point['Tin_K']-273.15:.1f} °C")
            st.metric("Efectividad ε", f"{knee_point['epsilon']:.2f}")
            st.metric("Conversión $X_{s}$", f"{knee_point['Xs']*100:.3f}%")
            st.metric("$S_{out}$", f"{knee_point['Sout_ppm']:.2f} ppm")
            st.metric("Carga Horno $Q_{horno}$", f"{knee_point['Q_horno_kW']:.2f} kW")

with tab_sanity:
    st.subheader("Sanity Checks (Valores Base)")
    
    sc1 = evaluar_hds(t_in=650.0, epsilon=0.90, m_dot=100000.0, s_in=500.0, cp=2.5, ea=80000.0, k0=calcular_k0(80000.0, 0.5), tau=0.5)
    sc2 = evaluar_hds(t_in=550.0, epsilon=0.75, m_dot=100000.0, s_in=500.0, cp=2.5, ea=80000.0, k0=calcular_k0(80000.0, 0.5), tau=0.5)
    sc3 = evaluar_hds(t_in=600.0, epsilon=0.999999, m_dot=100000.0, s_in=500.0, cp=2.5, ea=80000.0, k0=calcular_k0(80000.0, 0.5), tau=0.5)

    # En tablas usamos caracteres Unicode para los subíndices (Xₛ, Sₒᵤₜ, Qₕₒᵣₙₒ) ya que Dataframe no lee HTML
    df_sanity = pd.DataFrame([
        {"Caso": "1) Tin = 650 K, ε = 0.90", "Xₛ": f"{sc1['Xs']:.4f}", "Sₒᵤₜ (ppm)": f"{sc1['Sout_ppm']:.4f}", "Factible": "Sí" if sc1["factible"] else "No", "Qₕₒᵣₙₒ (kW)": f"{sc1['Q_horno_kW']:.2f}"},
        {"Caso": "2) Tin = 550 K, ε = 0.75", "Xₛ": f"{sc2['Xs']:.4f}", "Sₒᵤₜ (ppm)": f"{sc2['Sout_ppm']:.4f}", "Factible": "Sí" if sc2["factible"] else "No", "Qₕₒᵣₙₒ (kW)": f"{sc2['Q_horno_kW']:.2f}"},
        {"Caso": "3) Límite ε → 1 (Tin = 600 K)", "Xₛ": f"{sc3['Xs']:.4f}", "Sₒᵤₜ (ppm)": f"{sc3['Sout_ppm']:.4f}", "Factible": "Sí" if sc3["factible"] else "No", "Qₕₒᵣₙₒ (kW)": f"{sc3['Q_horno_kW']:.2f}"},
    ])
    st.dataframe(df_sanity, use_container_width=True, hide_index=True)
    
    st.markdown("#### Estadísticas de la Malla")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Puntos Malla", f"{len(df_grid):,}")
    m2.metric("Puntos Factibles", f"{len(df_grid[df_grid['factible']]):,}")
    m3.metric("Fracción Factible", f"{(len(df_grid[df_grid['factible']])/len(df_grid))*100:.1f}%")
    m4.metric("Puntos en Pareto", f"{len(df_pareto):,}")

with tab_sensibilidad:
    st.subheader("Análisis de Sensibilidad Paramétrica")
    st.caption("Evaluado alrededor del punto seleccionado en la barra lateral.")
    
    @st.cache_data
    def calcular_sensibilidad(t_in_base, eps_base, m_dot_base, s_in_base, cp_base, ea_base, k0_base, tau_base):
        caso_base = evaluar_hds(t_in_base, eps_base, m_dot_base, s_in_base, cp_base, ea_base, k0_base, tau_base)
        parametros = [("Flujo másico (ṁ)", "m_dot", m_dot_base), ("Capacidad calorífica (Cp)", "cp", cp_base), 
                      ("Energía de activación (Ea)", "ea", ea_base), ("Factor preexponencial (k0)", "k0", k0_base),
                      ("Tiempo espacial (tau)", "tau", tau_base), ("Efectividad (ε)", "epsilon", eps_base)]
        filas = []
        for nombre, clave, val_base in parametros:
            for factor, etiqueta in [(0.90, "-10%"), (1.10, "+10%")]:
                kwargs = {"t_in": t_in_base, "epsilon": eps_base, "m_dot": m_dot_base, "s_in": s_in_base, "cp": cp_base, "ea": ea_base, "k0": k0_base, "tau": tau_base}
                val_pert = val_base * factor
                if clave == "epsilon": val_pert = min(max(val_pert, 0.0), 1.0)
                kwargs[clave] = val_pert
                
                if clave in ["ea", "tau"]: kwargs["k0"] = calcular_k0(kwargs["ea"], kwargs["tau"])
                
                res = evaluar_hds(**kwargs)
                delta_q_pct = ((res["Q_horno_kW"] - caso_base["Q_horno_kW"]) / abs(caso_base["Q_horno_kW"]) * 100.0) if abs(caso_base["Q_horno_kW"]) > 1e-9 else 0.0
                delta_s_pct = ((res["Sout_ppm"] - caso_base["Sout_ppm"]) / caso_base["Sout_ppm"] * 100.0) if caso_base["Sout_ppm"] > 0 else 0.0

                filas.append({"Parámetro": nombre, "Perturbación": etiqueta, "Valor": f"{val_pert:.4g}", 
                              "Qₕₒᵣₙₒ (kW)": round(res["Q_horno_kW"], 2), "ΔQₕₒᵣₙₒ (%)": round(delta_q_pct, 2),
                              "Sₒᵤₜ (ppm)": round(res["Sout_ppm"], 3), "ΔSₒᵤₜ (%)": round(delta_s_pct, 2), "Factible": "Sí" if res["factible"] else "No"})
        return pd.DataFrame(filas)

    df_sens = calcular_sensibilidad(t_in, epsilon, m_dot, s_in, cp, ea, k0, tau)
    st.dataframe(df_sens, use_container_width=True, hide_index=True)
