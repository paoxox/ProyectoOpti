import numpy as np
import pandas as pd
import streamlit as st

# 1. Configuración general de la página
st.set_page_config(
    page_title="Proyecto HDS - Optimización Multiobjetivo",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Unidad de Hidrodesulfuración (HDS) — Optimización Multiobjetivo")
st.caption(
    "Evaluación de conversión de azufre (Xs) y carga térmica del horno (Q_horno)"
    " bajo restricción Sout <= 10 ppm."
)

# Constantes fijas del modelo
R_GAS = 8.314  # J/(mol*K)
T_ALIM = 300.0  # K
X_CALIB = 0.985
T_CALIB = 600.0  # K
TAU_CALIB = 0.5  # h
EA_CALIB = 80000.0  # J/mol
DT_AD_MAX = 15.0  # K a conversión completa (Xs = 1)


def calcular_k0(
    ea: float = EA_CALIB,
    tau: float = TAU_CALIB,
    x_ref: float = X_CALIB,
    t_ref: float = T_CALIB,
) -> float:
  """Calcula el factor preexponencial calibrado k0 [h^-1]."""
  return -np.log(1.0 - x_ref) / (tau * np.exp(-ea / (R_GAS * t_ref)))


def evaluar_hds(
    t_in: float,
    epsilon: float,
    m_dot: float = 100000.0,
    s_in: float = 500.0,
    cp: float = 2.5,
    ea: float = EA_CALIB,
    k0: float | None = None,
    tau: float = TAU_CALIB,
) -> dict:
  """Evalúa cinética, calidad de nafta, balance térmico y carga del horno."""
  if k0 is None:
    k0 = calcular_k0(ea=ea, tau=tau)

  # Cinética aparente de primer orden y calidad
  k_t = k0 * np.exp(-ea / (R_GAS * t_in))
  x_s = 1.0 - np.exp(-k_t * tau)
  s_out = s_in * (1.0 - x_s)
  factible = bool(s_out <= 10.0)

  # Balance térmico simplificado (reactor adiabático + intercambiador E-101)
  delta_t_ad = DT_AD_MAX * x_s
  t_efluente = t_in + delta_t_ad
  t_intercambiador = T_ALIM + epsilon * (t_efluente - T_ALIM)

  # Carga térmica del horno H-101: kJ/h y kW (sin recortar negativos)
  q_horno_kjh = m_dot * cp * (t_in - t_intercambiador)
  q_horno_kw = q_horno_kjh / 3600.0

  return {
      "Tin_K": t_in,
      "epsilon": epsilon,
      "k_h-1": k_t,
      "Xs": x_s,
      "Sout_ppm": s_out,
      "factible": factible,
      "dT_ad_K": delta_t_ad,
      "T_efluente_K": t_efluente,
      "T_intercambiador_K": t_intercambiador,
      "Q_horno_kJ_h": q_horno_kjh,
      "Q_horno_kW": q_horno_kw,
  }


# 2. Controles en barra lateral
with st.sidebar:
  st.header("Variables de decisión")
  t_in = st.slider(
      "Temperatura entrada reactor Tin (K)",
      min_value=550.0,
      max_value=650.0,
      value=600.0,
      step=1.0,
  )
  epsilon = st.slider(
      "Efectividad intercambiador ε",
      min_value=0.60,
      max_value=0.90,
      value=0.75,
      step=0.01,
  )

  st.header("Parámetros del modelo")
  m_dot = st.number_input(
      "Flujo másico ṁ (kg/h)",
      min_value=90000.0,
      max_value=110000.0,
      value=100000.0,
      step=1000.0,
  )
  s_in = st.number_input(
      "Azufre alimentación Sin (ppm)",
      min_value=450.0,
      max_value=550.0,
      value=500.0,
      step=5.0,
  )
  cp = st.number_input(
      "Capacidad calorífica Cp (kJ/(kg·K))",
      min_value=2.25,
      max_value=2.75,
      value=2.50,
      step=0.05,
  )
  ea = st.number_input(
      "Energía de activación Ea (J/mol)",
      min_value=72000.0,
      max_value=88000.0,
      value=80000.0,
      step=500.0,
  )
  tau = st.number_input(
      "Tiempo espacial τ (h)",
      min_value=0.45,
      max_value=0.55,
      value=0.50,
      step=0.01,
  )

  k0_base = calcular_k0(ea=ea, tau=tau)
  k0 = st.number_input(
      "Factor preexponencial k0 (h⁻¹)",
      min_value=float(k0_base * 0.90),
      max_value=float(k0_base * 1.10),
      value=float(k0_base),
      format="%.4e",
  )

# 3. Evaluación del punto actual
res_actual = evaluar_hds(
    t_in=t_in,
    epsilon=epsilon,
    m_dot=m_dot,
    s_in=s_in,
    cp=cp,
    ea=ea,
    k0=k0,
    tau=tau,
)

st.subheader("Punto operativo seleccionado")
if res_actual["factible"]:
  st.success(
      f"Cumple especificación de calidad: Sout = {res_actual['Sout_ppm']:.2f}"
      " ppm (<= 10 ppm)."
  )
else:
  st.warning(
      f"No cumple especificación de calidad: Sout = {res_actual['Sout_ppm']:.2f}"
      " ppm (> 10 ppm)."
  )

c1, c2, c3, c4 = st.columns(4)
c1.metric("Conversión Xs", f"{res_actual['Xs']*100:.2f}%")
c2.metric(
    "Azufre salida Sout",
    f"{res_actual['Sout_ppm']:.2f} ppm",
    delta_color="inverse",
)
c3.metric("Carga horno Q_horno", f"{res_actual['Q_horno_kW']:.2f} kW")
c4.metric("Carga horno Q_horno", f"{res_actual['Q_horno_kJ_h']:,.0f} kJ/h")

# 4. Sanity checks requeridos en la guía
st.subheader("Sanity Checks del modelo (valores base)")
sc1 = evaluar_hds(t_in=650.0, epsilon=0.90)
sc2 = evaluar_hds(t_in=550.0, epsilon=0.75)
sc3 = evaluar_hds(t_in=600.0, epsilon=0.999999)

df_sanity = pd.DataFrame([
    {
        "Caso": "1) Tin = 650 K, ε = 0.90",
        "Xs": f"{sc1['Xs']:.4f}",
        "Sout (ppm)": f"{sc1['Sout_ppm']:.4f}",
        "Factible (<=10 ppm)": "Sí" if sc1["factible"] else "No",
        "Q_horno (kW)": f"{sc1['Q_horno_kW']:.2f}",
        "Q_horno (kJ/h)": f"{sc1['Q_horno_kJ_h']:,.0f}",
    },
    {
        "Caso": "2) Tin = 550 K, ε = 0.75",
        "Xs": f"{sc2['Xs']:.4f}",
        "Sout (ppm)": f"{sc2['Sout_ppm']:.4f}",
        "Factible (<=10 ppm)": "Sí" if sc2["factible"] else "No",
        "Q_horno (kW)": f"{sc2['Q_horno_kW']:.2f}",
        "Q_horno (kJ/h)": f"{sc2['Q_horno_kJ_h']:,.0f}",
    },
    {
        "Caso": "3) Límite ε → 1 (Tin = 600 K)",
        "Xs": f"{sc3['Xs']:.4f}",
        "Sout (ppm)": f"{sc3['Sout_ppm']:.4f}",
        "Factible (<=10 ppm)": "Sí" if sc3["factible"] else "No",
        "Q_horno (kW)": f"{sc3['Q_horno_kW']:.2f}",
        "Q_horno (kJ/h)": f"{sc3['Q_horno_kJ_h']:,.0f}",
    },
])

import plotly.graph_objects as go
# 5. Generación de la Malla (Grid Search) y Pareto
@st.cache_data
def calcular_malla_y_pareto(
    n_tin: int = 50,
    n_eps: int = 50,
    m_dot: float = 100000.0,
    s_in: float = 500.0,
    cp: float = 2.5,
    ea: float = EA_CALIB,
    k0: float | None = None,
    tau: float = TAU_CALIB,
):
  """Genera el dominio completo, filtra factibles, calcula la frontera de Pareto y halla el Knee Point."""
  grid_tin = np.linspace(550.0, 650.0, n_tin)
  grid_eps = np.linspace(0.60, 0.90, n_eps)

  resultados = []
  for t in grid_tin:
    for e in grid_eps:
      res = evaluar_hds(
          t_in=t,
          epsilon=e,
          m_dot=m_dot,
          s_in=s_in,
          cp=cp,
          ea=ea,
          k0=k0,
          tau=tau,
      )
      resultados.append(res)

  df_grid = pd.DataFrame(resultados)
  df_factible = df_grid[df_grid["factible"]].copy()

  if df_factible.empty:
    return df_grid, pd.DataFrame(), None

  # Filtrado de la Frontera de Pareto (no dominados)
  # Objetivos: Minimizar (1 - Xs) y Minimizar Q_horno_kW
  df_factible["obj_xs"] = 1.0 - df_factible["Xs"]
  df_factible["obj_q"] = df_factible["Q_horno_kW"]

  # Ordenar por obj_xs ascendente
  df_sorted = df_factible.sort_values(
      by=["obj_xs", "obj_q"], ascending=[True, True]
  ).reset_index(drop=True)

  pareto_indices = []
  min_q_visto = float("inf")

  for i, row in df_sorted.iterrows():
    if row["obj_q"] < min_q_visto:
      pareto_indices.append(i)
      min_q_visto = row["obj_q"]

  df_pareto = df_sorted.loc[pareto_indices].copy()

  # Cálculo del Knee Point (distancia máxima a la recta entre extremos normalizados)
  xs_min, xs_max = df_pareto["obj_xs"].min(), df_pareto["obj_xs"].max()
  q_min, q_max = df_pareto["obj_q"].min(), df_pareto["obj_q"].max()

  denom_xs = (xs_max - xs_min) if (xs_max - xs_min) > 0 else 1.0
  denom_q = (q_max - q_min) if (q_max - q_min) > 0 else 1.0

  # Normalización Min-Max (0 a 1)
  df_pareto["xs_norm"] = (df_pareto["obj_xs"] - xs_min) / denom_xs
  df_pareto["q_norm"] = (df_pareto["obj_q"] - q_min) / denom_q

  # Puntos extremos normalizados (x1, y1) y (x2, y2)
  x1, y1 = df_pareto.iloc[0]["xs_norm"], df_pareto.iloc[0]["q_norm"]
  x2, y2 = df_pareto.iloc[-1]["xs_norm"], df_pareto.iloc[-1]["q_norm"]

  # Ecuación de la recta en 2D: (y2 - y1)*x - (x2 - x1)*y + (x2*y1 - y2*x1) = 0
  # ax + by + c = 0
  a = y2 - y1
  b = -(x2 - x1)
  c = x2 * y1 - y2 * x1
  norma = np.sqrt(a**2 + b**2)

  distancias = []
  for _, row in df_pareto.iterrows():
    x0, y0 = row["xs_norm"], row["q_norm"]
    if norma == 0:
      d = 0.0
    else:
      # Distancia de un punto (x0, y0) a la recta ax + by + c = 0
      d = abs(a * x0 + b * y0 + c) / norma
    distancias.append(d)

  df_pareto["distancia_knee"] = distancias
  idx_knee = df_pareto["distancia_knee"].idxmax()
  knee_point = df_pareto.loc[idx_knee].to_dict()

  return df_grid, df_pareto, knee_point

# Generar malla y Pareto con parámetros de la barra lateral
df_grid, df_pareto, knee_point = calcular_malla_y_pareto(
    n_tin=50,
    n_eps=50,
    m_dot=m_dot,
    s_in=s_in,
    cp=cp,
    ea=ea,
    k0=k0,
    tau=tau,
)

# 6. Visualización con Pestañas
st.markdown("---")
tab_pareto, tab_malla = st.tabs(
    ["📈 Frontera de Pareto y Knee Point", "🔍 Análisis de la Malla"]
)

with tab_pareto:
  st.subheader("Optimización Multiobjetivo: Conversión vs Carga Térmica")

  if df_pareto.empty:
    st.error(
        "No se encontraron puntos factibles en el dominio especificado."
    )
  else:
    col_graph, col_knee = st.columns([2, 1])

    with col_graph:
      # Crear gráfico interactivo de Pareto
      fig = go.Figure()

      # Puntos Factibles
      df_fact = df_grid[df_grid["factible"]]
      fig.add_trace(
          go.Scatter(
              x=df_fact["Q_horno_kW"],
              y=df_fact["Xs"] * 100,
              mode="markers",
              name="Puntos Factibles (Sout ≤ 10 ppm)",
              marker=dict(color="lightskyblue", size=5, opacity=0.5),
              hovertemplate=(
                  "Q: %{x:.1f} kW<br>Xs: %{y:.2f}%<br>Tin: %{customdata[0]:.1f}"
                  " K<br>ε: %{customdata[1]:.2f}<extra></extra>"
              ),
              customdata=df_fact[["Tin_K", "epsilon"]].values,
          )
      )

      # Puntos Infactibles
      df_infact = df_grid[~df_grid["factible"]]
      fig.add_trace(
          go.Scatter(
              x=df_infact["Q_horno_kW"],
              y=df_infact["Xs"] * 100,
              mode="markers",
              name="Puntos Infactibles (Sout > 10 ppm)",
              marker=dict(color="lightgray", size=4, opacity=0.3),
              hovertemplate="Q: %{x:.1f} kW<br>Xs: %{y:.2f}%<extra></extra>",
          )
      )

      # Frontera de Pareto
      fig.add_trace(
          go.Scatter(
              x=df_pareto["Q_horno_kW"],
              y=df_pareto["Xs"] * 100,
              mode="lines+markers",
              name="Frontera de Pareto",
              line=dict(color="royalblue", width=3),
              marker=dict(size=6),
              hovertemplate=(
                  "<b>Pareto</b><br>Q: %{x:.1f} kW<br>Xs:"
                  " %{y:.2f}%<br>Tin: %{customdata[0]:.1f} K<br>ε:"
                  " %{customdata[1]:.2f}<extra></extra>"
              ),
              customdata=df_pareto[["Tin_K", "epsilon"]].values,
          )
      )

      # Knee Point
      fig.add_trace(
          go.Scatter(
              x=[knee_point["Q_horno_kW"]],
              y=[knee_point["Xs"] * 100],
              mode="markers",
              name="Knee Point (Rodilla)",
              marker=dict(
                  color="crimson",
                  size=14,
                  symbol="star",
                  line=dict(color="black", width=1),
              ),
              hovertemplate=(
                  "<b>Knee Point</b><br>Q: %{x:.1f} kW<br>Xs:"
                  " %{y:.2f}%<br>Tin: %{customdata[0]:.1f} K<br>ε:"
                  " %{customdata[1]:.2f}<extra></extra>"
              ),
              customdata=[[knee_point["Tin_K"], knee_point["epsilon"]]],
          )
      )

      fig.update_layout(
          title="Frontera de Pareto y Punto de Rodilla",
          xaxis_title="Carga Térmica del Horno Q_horno (kW)",
          yaxis_title="Conversión de Azufre Xs (%)",
          legend=dict(x=0.01, y=0.01, bgcolor="rgba(255,255,255,0.8)"),
          height=500,
      )
      st.plotly_chart(fig, use_container_width=True)

    with col_knee:
      st.markdown("### 📍 Resultados del Knee Point")
      st.success("Solución de compromiso óptima entre conversión y energía.")

      st.metric(
          "Temperatura Tin",
          f"{knee_point['Tin_K']:.1f} K",
          f"{knee_point['Tin_K']-273.15:.1f} °C",
      )
      st.metric("Efectividad ε", f"{knee_point['epsilon']:.2f}")
      st.metric("Conversión Xs", f"{knee_point['Xs']*100:.3f}%")
      st.metric(
          "Azufre Salida Sout",
          f"{knee_point['Sout_ppm']:.2f} ppm",
          delta_color="inverse",
      )
      st.metric("Carga Horno Q_horno", f"{knee_point['Q_horno_kW']:.2f} kW")
      st.metric("Carga Horno Q_horno", f"{knee_point['Q_horno_kJ_h']:,.0f} kJ/h")

with tab_malla:
  st.subheader("Resumen Estadístico de la Malla de Evaluación")

  total_puntos = len(df_grid)
  factibles_puntos = len(df_fact)
  fraccion_factible = (factibles_puntos / total_puntos) * 100

  m1, m2, m3, m4 = st.columns(4)
  m1.metric("Total Puntos Malla", f"{total_puntos:,}")
  m2.metric("Puntos Factibles", f"{factibles_puntos:,}")
  m3.metric("Fracción Factible", f"{fraccion_factible:.1f}%")
  m4.metric("Puntos en Pareto", f"{len(df_pareto):,}")

  st.markdown("#### Tabla de Puntos Representativos de la Frontera de Pareto")
  st.dataframe(
      df_pareto[[
          "Tin_K",
          "epsilon",
          "Xs",
          "Sout_ppm",
          "Q_horno_kW",
          "Q_horno_kJ_h",
      ]],
      use_container_width=True,
      hide_index=True,
  )

# 7. Análisis de Sensibilidad (±10% y exploración de Sin)
@st.cache_data
def calcular_sensibilidad(
    t_in_base: float,
    eps_base: float,
    m_dot_base: float,
    s_in_base: float,
    cp_base: float,
    ea_base: float,
    k0_base: float,
    tau_base: float,
):
  """Evalúa variaciones de -10% y +10% en m_dot, Cp, Ea, k0 y epsilon alrededor del punto actual."""
  caso_base = evaluar_hds(
      t_in=t_in_base,
      epsilon=eps_base,
      m_dot=m_dot_base,
      s_in=s_in_base,
      cp=cp_base,
      ea=ea_base,
      k0=k0_base,
      tau=tau_base,
  )

  parametros = [
      ("Flujo másico (ṁ)", "m_dot", m_dot_base),
      ("Capacidad calorífica (Cp)", "cp", cp_base),
      ("Energía de activación (Ea)", "ea", ea_base),
      ("Factor preexponencial (k0)", "k0", k0_base),
      ("Efectividad (ε)", "epsilon", eps_base),
  ]

  filas = []
  for nombre, clave, val_base in parametros:
    for factor, etiqueta in [(0.90, "-10%"), (1.10, "+10%")]:
      kwargs = {
          "t_in": t_in_base,
          "epsilon": eps_base,
          "m_dot": m_dot_base,
          "s_in": s_in_base,
          "cp": cp_base,
          "ea": ea_base,
          "k0": k0_base,
          "tau": tau_base,
      }
      val_pert = val_base * factor
      # Respetar límites físicos de efectividad ε en [0, 1] si aplica
      if clave == "epsilon":
        val_pert = min(max(val_pert, 0.0), 1.0)
      kwargs[clave] = val_pert

      res = evaluar_hds(**kwargs)
      delta_q_pct = (
          ((res["Q_horno_kW"] - caso_base["Q_horno_kW"]) / abs(caso_base["Q_horno_kW"]) * 100.0)
          if abs(caso_base["Q_horno_kW"]) > 1e-9
          else 0.0
      )
      delta_s_pct = (
          ((res["Sout_ppm"] - caso_base["Sout_ppm"]) / caso_base["Sout_ppm"] * 100.0)
          if caso_base["Sout_ppm"] > 0
          else 0.0
      )

      filas.append({
          "Parámetro": nombre,
          "Perturbación": etiqueta,
          "Valor perturbado": f"{val_pert:.4g}",
          "Q_horno (kW)": round(res["Q_horno_kW"], 2),
          "ΔQ_horno (%)": round(delta_q_pct, 2),
          "Sout (ppm)": round(res["Sout_ppm"], 3),
          "ΔSout (%)": round(delta_s_pct, 2),
          "Factible": "Sí" if res["factible"] else "No",
      })

  return pd.DataFrame(filas)


df_sens = calcular_sensibilidad(
    t_in_base=t_in,
    eps_base=epsilon,
    m_dot_base=m_dot,
    s_in_base=s_in,
    cp_base=cp,
    ea_base=ea,
    k0_base=k0,
    tau_base=tau,
)

with st.expander("🧪 Análisis de Sensibilidad (±10%) y Azufre de Alimentación (Sin)", expanded=True):
  st.caption(
      "Evaluado alrededor de la temperatura Tin y efectividad ε seleccionadas en la barra lateral."
  )
  st.dataframe(df_sens, use_container_width=True, hide_index=True)

  sin_rango = st.slider(
      "Explorar azufre de alimentación Sin (ppm)",
      min_value=450.0,
      max_value=550.0,
      value=float(s_in),
      step=5.0,
      key="slider_sin_sensibilidad",
  )
  res_sin = evaluar_hds(
      t_in=t_in,
      epsilon=epsilon,
      m_dot=m_dot,
      s_in=sin_rango,
      cp=cp,
      ea=ea,
      k0=k0,
      tau=tau,
  )
  st.write(
      f"Para **Sin = {sin_rango:.0f} ppm** (con Tin = {t_in:.0f} K y ε = {epsilon:.2f}): "
      f"**Sout = {res_sin['Sout_ppm']:.2f} ppm** | "
      f"**Factible (≤ 10 ppm):** {'Sí' if res_sin['factible'] else 'No'} | "
      f"**Q_horno = {res_sin['Q_horno_kW']:.2f} kW**."
  )
