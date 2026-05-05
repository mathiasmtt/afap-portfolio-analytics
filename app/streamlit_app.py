"""Dashboard Streamlit — AFAP Portfolio Analytics.

Demo end-to-end para la postulación a Analista de Datos Comercial — AFAP Itaú.
Consume exclusivamente las funciones puras de ``src/`` (single source of truth).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics import (  # noqa: E402
    LABELS_ANTIGUEDAD,
    LABELS_EDAD,
    LABELS_SALDO,
    aplicar_segmentaciones,
    concentracion_cartera,
    cross_sell_resumen,
    oportunidades_cross_sell,
    pareto_cartera,
    resumen_calidad,
    segmentos_reglas,
)
from src.data_loader import (  # noqa: E402
    DEPARTAMENTO_MAP,
    GENERO_MAP,
    RENAME_MAP,
    cargar_csv,
)
from src.models.churn_logit import (  # noqa: E402
    drivers_principales,
    entrenar_logit,
    scorear_afiliados,
)
from src.reporting import generar_excel  # noqa: E402

# ---------------------------------------------------------------------------
# Paleta Itaú
# ---------------------------------------------------------------------------
ITAU_ORANGE = "#EC7000"
ITAU_BLUE = "#003B71"
ITAU_ORANGE_LIGHT = "#FAA61A"
GREY_NEUTRAL = "#D2D2D2"
TEXT_COLOR = "#32373C"

ITAU_ORANGE_SCALE = [
    [0.0, "#FFFFFF"],
    [0.5, ITAU_ORANGE_LIGHT],
    [1.0, ITAU_ORANGE],
]
ITAU_DIVERGING = [
    [0.0, ITAU_BLUE],
    [0.5, "#FFFFFF"],
    [1.0, ITAU_ORANGE],
]

CSV_PATH = ROOT / "data" / "raw" / "Customer-Churn-Records.csv"

# ---------------------------------------------------------------------------
# Configuración global de la página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AFAP Portfolio Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Data / modelo cacheados
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Cargando dataset de afiliados...")
def _cargar_dataset() -> pd.DataFrame:
    if not CSV_PATH.exists():
        return pd.DataFrame()
    return cargar_csv(CSV_PATH)


@st.cache_resource(show_spinner="Entrenando modelo de fuga...")
def _entrenar_modelo(df: pd.DataFrame):
    return entrenar_logit(df, seed=42)


@st.cache_data(show_spinner="Generando Excel...")
def _excel_bytes(_df: pd.DataFrame, _model) -> bytes:
    return generar_excel(_df, _model, top_riesgo_n=100)


# ---------------------------------------------------------------------------
# Filtros (sidebar)
# ---------------------------------------------------------------------------


def _sidebar_header() -> None:
    """Branding + descripción corta en el tope del sidebar."""
    st.sidebar.markdown(
        f"""
        <div style="text-align:center; padding:8px 0 4px 0;">
            <div style="font-size:28px; line-height:1; margin-bottom:4px;">📊</div>
            <div style="font-weight:700; font-size:18px; color:{ITAU_BLUE};
                        letter-spacing:-0.3px; line-height:1.2;">
                AFAP Portfolio<br>Analytics
            </div>
            <div style="font-size:11px; color:#6B7280; margin-top:6px;">
                Demo comercial · cartera de afiliados
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        f"<hr style='margin:8px 0 12px 0; border:none; border-top:2px solid {ITAU_ORANGE};'/>",
        unsafe_allow_html=True,
    )


def _aplicar_filtros(df: pd.DataFrame) -> pd.DataFrame:
    """Construye el sidebar de filtros y devuelve el DataFrame filtrado."""
    seg = aplicar_segmentaciones(df)
    _sidebar_header()

    # Botón de reset: re-inicializa las claves de los widgets.
    col_reset, col_ayuda = st.sidebar.columns([3, 1])
    with col_reset:
        if st.button("↺ Resetear filtros", width="stretch", key="btn_reset"):
            for clave in ("f_dpto", "f_edad", "f_antig", "f_saldo", "f_activos"):
                st.session_state.pop(clave, None)
            st.rerun()
    with col_ayuda:
        with st.popover("ℹ️", width="stretch"):
            st.markdown(
                "**Cómo usar el dashboard**\n\n"
                "Ajustá los filtros para acotar la cartera. "
                "Los KPIs, gráficos y el Excel se recalculan en vivo.\n\n"
                "- **Departamento** · geografía del afiliado.\n"
                "- **Tramo etario / antigüedad / saldo** · segmentos canónicos.\n"
                "- **Solo activos** · excluye cotizantes inactivos.\n\n"
                "El modelo de scoring se entrena sobre la población completa "
                "para evitar sesgo por los filtros."
            )

    departamentos = sorted(df["departamento"].unique())

    with st.sidebar.expander("🗺️ Geografía", expanded=True):
        sel_dpto = st.multiselect(
            "Departamento",
            departamentos,
            default=st.session_state.get("f_dpto", departamentos),
            key="f_dpto",
            label_visibility="collapsed",
        )

    with st.sidebar.expander("👥 Segmentos del afiliado", expanded=True):
        st.caption("Tramo etario")
        sel_edad = st.pills(
            "Tramo etario",
            options=LABELS_EDAD,
            selection_mode="multi",
            default=st.session_state.get("f_edad", LABELS_EDAD),
            key="f_edad",
            label_visibility="collapsed",
        )
        st.caption("Antigüedad (años)")
        sel_antig = st.pills(
            "Antigüedad",
            options=LABELS_ANTIGUEDAD,
            selection_mode="multi",
            default=st.session_state.get("f_antig", LABELS_ANTIGUEDAD),
            key="f_antig",
            label_visibility="collapsed",
        )
        st.caption("Tramo de saldo")
        sel_saldo = st.pills(
            "Saldo",
            options=LABELS_SALDO,
            selection_mode="multi",
            default=st.session_state.get("f_saldo", LABELS_SALDO),
            key="f_saldo",
            label_visibility="collapsed",
        )

    with st.sidebar.expander("⚡ Actividad", expanded=False):
        solo_activos = st.toggle(
            "Solo aportantes activos",
            value=st.session_state.get("f_activos", False),
            key="f_activos",
            help="Excluye cotizantes inactivos (no generan fee de administración).",
        )

    mask = (
        seg["departamento"].isin(sel_dpto)
        & seg["tramo_edad"].isin(sel_edad)
        & seg["tramo_antiguedad"].isin(sel_antig)
        & seg["tramo_saldo"].isin(sel_saldo)
    )
    if solo_activos:
        mask &= seg["aportante_activo"] == 1

    filtrado = seg.loc[mask].drop(
        columns=["tramo_edad", "tramo_antiguedad", "tramo_saldo"]
    )

    _sidebar_resumen(df, filtrado)
    _sidebar_footer()
    return filtrado


def _sidebar_resumen(df_total: pd.DataFrame, df_filtrado: pd.DataFrame) -> None:
    """Muestra en vivo cuántos afiliados y qué % de saldo sobrevivieron al filtro."""
    n_total = len(df_total)
    n_sel = len(df_filtrado)
    saldo_total = df_total["saldo_cuenta"].sum()
    saldo_sel = df_filtrado["saldo_cuenta"].sum() if n_sel else 0.0
    pct_n = (n_sel / n_total * 100) if n_total else 0.0
    pct_saldo = (saldo_sel / saldo_total * 100) if saldo_total else 0.0

    n_sel_fmt = f"{n_sel:,}".replace(",", ".")
    n_total_fmt = f"{n_total:,}".replace(",", ".")
    st.sidebar.markdown(
        f"""
        <div style='margin-top:14px; padding:12px; border-radius:8px;
                    background:#F5F5F5; border-left:3px solid {ITAU_ORANGE};'>
            <div style='font-size:11px; color:#6B7280; text-transform:uppercase;
                        letter-spacing:0.5px; margin-bottom:6px;'>
                Selección actual
            </div>
            <div style='font-size:18px; font-weight:700; color:{TEXT_COLOR};
                        line-height:1.1;'>
                {n_sel_fmt} <span style='color:#9CA3AF; font-weight:500;
                font-size:14px;'>/ {n_total_fmt} afiliados</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.progress(min(pct_n / 100, 1.0))
    st.sidebar.caption(
        f"**{pct_n:.1f}%** de afiliados · **{pct_saldo:.1f}%** del saldo"
    )
    if n_sel == 0:
        st.sidebar.warning("La selección actual no contiene afiliados.")


def _sidebar_footer() -> None:
    """Links y metadatos al pie del sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "**Dataset**: Bank Customer Churn (Kaggle, 10 k clientes) reframeado "
        "al vocabulario AFAP.\n\n"
        "[📄 Código fuente](https://github.com/mathiasmtt/afap-portfolio-analytics)"
    )


# ---------------------------------------------------------------------------
# Helpers de presentación
# ---------------------------------------------------------------------------


_PLOTLY_LAYOUT = dict(
    margin=dict(l=30, r=20, t=30, b=40),
    font=dict(family="Arial, sans-serif", size=13, color=TEXT_COLOR),
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    hoverlabel=dict(bgcolor="white", font_size=12),
)


def _fmt_int(n: float) -> str:
    return f"{int(round(n)):,}".replace(",", ".")


def _fmt_uyu(v: float) -> str:
    return f"UYU {v:,.0f}".replace(",", ".")


def _como_leer(texto: str) -> None:
    """Caption sobrio debajo de cada gráfico, con el mismo estilo que el resto.

    Formato fijo: `st.caption` con prefijo **Lectura:** en negrita seguido del
    texto explicativo. Mismo gris tenue y tipografía en toda la app.
    """
    st.caption(f"**Lectura:** {texto}")


def _bloque_titulo(capitulo: str, titulo: str, bajada: str) -> None:
    """Encabezado narrativo de una sección: 'Capítulo' + título + bajada explicativa."""
    st.markdown(
        f"""
        <div style='margin-top:8px; margin-bottom:8px;'>
            <div style='font-size:11px; font-weight:700; color:{ITAU_ORANGE};
                        text-transform:uppercase; letter-spacing:1.5px;'>
                {capitulo}
            </div>
            <div style='font-size:22px; font-weight:700; color:{ITAU_BLUE};
                        line-height:1.2; margin-top:2px;'>
                {titulo}
            </div>
            <div style='font-size:14px; color:#4B5563; margin-top:4px;'>
                {bajada}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Capítulo 1 — Calidad de datos y EDA (describir)
# ---------------------------------------------------------------------------


# Descripciones canónicas por columna AFAP (para la tabla de schema).
_DESCRIPCIONES_AFAP: dict[str, str] = {
    "afiliado_id": "Identificador único del afiliado.",
    "apellido": "Apellido (pseudonimizado).",
    "score_interno": "Score crediticio / comportamental (350-850).",
    "departamento": "Departamento de residencia (Montevideo, Canelones, Maldonado).",
    "genero": "Género: M / F.",
    "edad": "Edad en años.",
    "anios_afiliacion": "Años desde que se afilió a la AFAP.",
    "saldo_cuenta": "Saldo acumulado en la cuenta individual (UYU).",
    "productos_contratados": "Número de productos AFAP contratados (1-4).",
    "tiene_producto_adicional": "Tiene producto adicional del grupo Itaú (0/1).",
    "aportante_activo": "Aportó en el mes corriente (0/1).",
    "salario_nominal": "Salario nominal estimado (UYU).",
    "traspaso": "Target: se traspasó a otra AFAP (0/1).",
}


def _seccion_schema_y_reframe() -> None:
    """Tablas explícitas del schema y del reframe bancario → AFAP."""
    st.markdown(
        f"<div style='font-weight:600; color:{TEXT_COLOR}; margin:16px 0 6px 0;'>"
        f"Schema del dataset · 13 columnas trabajadas</div>",
        unsafe_allow_html=True,
    )
    schema_df = pd.DataFrame(
        [
            {
                "Columna AFAP": afap,
                "Tipo": _tipo_de(afap),
                "Descripción": _DESCRIPCIONES_AFAP.get(afap, ""),
            }
            for _orig, afap in RENAME_MAP.items()
        ]
    )
    st.dataframe(schema_df, width="stretch", hide_index=True)

    st.markdown(
        f"<div style='font-weight:600; color:{TEXT_COLOR}; margin:18px 0 6px 0;'>"
        f"Qué se cambió · reframe bancario → AFAP</div>",
        unsafe_allow_html=True,
    )
    reframe_rows = [
        {
            "Origen bancario": orig,
            "→": "→",
            "Destino AFAP": afap,
            "Tipo de cambio": _tipo_cambio(orig),
        }
        for orig, afap in RENAME_MAP.items()
    ]
    # Agrego las 2 filas de mapeos de valores (geografía y género).
    geo_str = " · ".join(f"{k} → {v}" for k, v in DEPARTAMENTO_MAP.items())
    gen_str = " · ".join(f"{k} → {v}" for k, v in GENERO_MAP.items())
    reframe_rows.append(
        {
            "Origen bancario": f"valores de Geography ({geo_str})",
            "→": "→",
            "Destino AFAP": "valores mapeados a departamentos de Uruguay",
            "Tipo de cambio": "Recoding de valores",
        }
    )
    reframe_rows.append(
        {
            "Origen bancario": f"valores de Gender ({gen_str})",
            "→": "→",
            "Destino AFAP": "códigos M / F",
            "Tipo de cambio": "Recoding de valores",
        }
    )
    reframe_df = pd.DataFrame(reframe_rows)
    st.dataframe(reframe_df, width="stretch", hide_index=True)

    _como_leer(
        "La primera tabla muestra el schema AFAP tal como lo consume el resto de "
        "la app. La segunda tabla es auditable: cada fila documenta un cambio "
        "concreto aplicado al CSV original de Kaggle. Sólo se renombran columnas "
        "y se recodifican dos variables categóricas — **no se modifican valores "
        "numéricos, no se filtran filas, no se imputa**."
    )


def _tipo_de(col: str) -> str:
    if col in {"afiliado_id", "edad", "anios_afiliacion", "score_interno",
               "productos_contratados"}:
        return "Entero"
    if col in {"saldo_cuenta", "salario_nominal"}:
        return "Decimal (UYU)"
    if col in {"traspaso", "aportante_activo", "tiene_producto_adicional"}:
        return "Binario 0/1"
    if col == "departamento":
        return "Categórica (3 valores)"
    if col == "genero":
        return "Categórica (2 valores)"
    if col == "apellido":
        return "Texto"
    return "—"


def _tipo_cambio(col_origen: str) -> str:
    if col_origen == "Geography":
        return "Rename + recoding geográfico"
    if col_origen == "Gender":
        return "Rename + recoding de valores"
    return "Rename (sólo cambio de nombre)"


def _seccion_calidad_eda(df: pd.DataFrame) -> None:
    """¿Con qué estamos trabajando? Snapshot de calidad + distribuciones clave."""
    _bloque_titulo(
        "Capítulo 1 · Describir",
        "¿Con qué datos estamos trabajando?",
        "Antes de cualquier análisis, una foto de la calidad del dataset y la "
        "distribución de las variables que van a aparecer en el resto de la historia.",
    )

    r = resumen_calidad(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas", _fmt_int(r["n_filas"]), help="Un afiliado por fila.")
    c2.metric("Columnas", _fmt_int(r["n_columnas"]))
    c3.metric(
        "Valores faltantes",
        f"{r['pct_missing']:.2f} %",
        help="Porcentaje sobre el total de celdas. <0.1% es muy bueno.",
    )
    c4.metric(
        "Duplicados por ID",
        _fmt_int(r["n_duplicados"]),
        help="Afiliados con el mismo afiliado_id. Debe ser 0.",
    )

    # Tablas de schema + reframe.
    _seccion_schema_y_reframe()

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown(
            f"<div style='font-weight:600; color:{TEXT_COLOR}; margin-top:8px;'>"
            f"Distribución de edad</div>",
            unsafe_allow_html=True,
        )
        fig = px.histogram(
            df, x="edad", nbins=30,
            color_discrete_sequence=[ITAU_ORANGE],
        )
        fig.update_traces(marker_line_color="white", marker_line_width=1)
        fig.update_layout(
            **_PLOTLY_LAYOUT, height=280, bargap=0.02,
            xaxis=dict(title="Edad (años)", gridcolor="#F0F0F0"),
            yaxis=dict(title="Afiliados", gridcolor="#F0F0F0"),
        )
        st.plotly_chart(fig, width="stretch")
        _como_leer(
            "Cada barra agrupa afiliados por tramo de edad. La altura es la cantidad. "
            "Una distribución concentrada entre 30-50 años indica que el grueso de "
            "la cartera está en edad laboral activa."
        )

    with col_b:
        st.markdown(
            f"<div style='font-weight:600; color:{TEXT_COLOR}; margin-top:8px;'>"
            f"Distribución de saldo de cuenta individual</div>",
            unsafe_allow_html=True,
        )
        fig = px.histogram(
            df, x="saldo_cuenta", nbins=40,
            color_discrete_sequence=[ITAU_BLUE],
        )
        fig.update_traces(marker_line_color="white", marker_line_width=1)
        fig.update_layout(
            **_PLOTLY_LAYOUT, height=280, bargap=0.02,
            xaxis=dict(title="Saldo (UYU)", gridcolor="#F0F0F0"),
            yaxis=dict(title="Afiliados", gridcolor="#F0F0F0"),
        )
        st.plotly_chart(fig, width="stretch")
        _como_leer(
            "Distribución del saldo individual. El pico en cero corresponde a "
            "afiliados sin saldo acumulado todavía. La cola derecha (saldos altos) "
            "es el segmento clave de retención — pocos afiliados, mucho dinero."
        )

    # Balance del target: clave para leer el resto del análisis.
    col_c, col_d = st.columns([1, 1])
    with col_c:
        tasa = r["tasa_target"] * 100
        st.markdown(
            f"<div style='font-weight:600; color:{TEXT_COLOR}; margin-top:8px;'>"
            f"Balance del target · traspaso</div>",
            unsafe_allow_html=True,
        )
        fig = go.Figure(
            data=[
                go.Bar(
                    x=["Retenido", "Traspasado"],
                    y=[100 - tasa, tasa],
                    marker_color=[ITAU_BLUE, ITAU_ORANGE],
                    text=[f"{100 - tasa:.1f}%", f"{tasa:.1f}%"],
                    textposition="outside",
                )
            ]
        )
        fig.update_layout(
            **_PLOTLY_LAYOUT, height=260, showlegend=False,
            yaxis=dict(title="% de afiliados", ticksuffix="%",
                       gridcolor="#F0F0F0", range=[0, 100]),
            xaxis=dict(title=""),
        )
        st.plotly_chart(fig, width="stretch")
        _como_leer(
            "Proporción de afiliados que se traspasaron a otra AFAP (naranja) vs. "
            "retenidos (azul). Un target desbalanceado (<30% positivos) es típico "
            "en fuga y condiciona cómo se lee el modelo del capítulo 5."
        )

    with col_d:
        st.markdown(
            f"<div style='font-weight:600; color:{TEXT_COLOR}; margin-top:8px;'>"
            f"Rangos observados</div>",
            unsafe_allow_html=True,
        )
        rangos_df = pd.DataFrame(r["rangos"]).T
        rangos_df.index = [
            {"edad": "Edad (años)", "anios_afiliacion": "Antigüedad (años)",
             "saldo_cuenta": "Saldo (UYU)", "salario_nominal": "Salario (UYU)"}[i]
            for i in rangos_df.index
        ]
        st.dataframe(
            rangos_df.style.format("{:,.0f}"),
            width="stretch",
        )

    _como_leer(
        f"dataset con **{_fmt_int(r['n_filas'])}** afiliados, "
        f"**sin duplicados ni datos faltantes**, con tasa de traspaso del "
        f"**{r['tasa_target'] * 100:.1f} %** — desbalanceado pero suficiente para "
        f"modelado. Los números siguientes se apoyan en esta base."
    )


def _seccion_pareto(df: pd.DataFrame) -> None:
    """Capítulo 2 — Priorizar: concentración de cartera (curva de Lorenz)."""
    _bloque_titulo(
        "Capítulo 2 · Priorizar",
        "Pocos afiliados concentran la mayor parte del saldo",
        "La curva acumulada ordena a los afiliados de mayor a menor saldo. "
        "Dónde está el dinero = dónde va primero el equipo de retención.",
    )

    p = pareto_cartera(df)
    if p.empty:
        st.info("Sin datos para los filtros seleccionados.")
        return

    top20 = concentracion_cartera(df, 0.20) * 100
    top50 = concentracion_cartera(df, 0.50) * 100

    # Curva de Lorenz: % de afiliados en X, % de saldo acumulado en Y.
    pct_afil = p["pct_afiliados"].to_numpy() * 100
    pct_saldo = p["pct_acumulado"].to_numpy() * 100

    fig = go.Figure()

    # Línea de igualdad (distribución uniforme) como referencia visual.
    fig.add_trace(
        go.Scatter(
            x=[0, 100],
            y=[0, 100],
            mode="lines",
            line=dict(color=GREY_NEUTRAL, width=1, dash="dash"),
            name="Distribución uniforme",
            hoverinfo="skip",
        )
    )

    # Curva real: cuánto del saldo concentra el top X% de afiliados.
    fig.add_trace(
        go.Scatter(
            x=pct_afil,
            y=pct_saldo,
            mode="lines",
            line=dict(color=ITAU_ORANGE, width=3),
            fill="tozeroy",
            fillcolor="rgba(236, 112, 0, 0.12)",
            name="Concentración real",
            hovertemplate="Top %{x:.0f}% de afiliados<br>"
                          "concentra <b>%{y:.1f}%</b> del saldo<extra></extra>",
        )
    )

    # Marcadores + anotaciones en Top 20 y Top 50.
    for x_marker, y_marker, etiqueta in [
        (20, top20, f"<b>Top 20%</b><br>{top20:.1f}% del saldo"),
        (50, top50, f"<b>Top 50%</b><br>{top50:.1f}% del saldo"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=[x_marker],
                y=[y_marker],
                mode="markers",
                marker=dict(size=11, color=ITAU_BLUE, line=dict(color="white", width=2)),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_annotation(
            x=x_marker, y=y_marker, text=etiqueta, showarrow=True,
            arrowhead=0, arrowcolor=ITAU_BLUE, ax=40, ay=-40,
            font=dict(size=12, color=ITAU_BLUE),
            bgcolor="rgba(255,255,255,0.9)", bordercolor=ITAU_BLUE, borderwidth=1,
            borderpad=4,
        )

    fig.update_layout(
        **_PLOTLY_LAYOUT,
        xaxis=dict(
            title="% de afiliados (ordenados por saldo, de mayor a menor)",
            ticksuffix="%", range=[0, 100], gridcolor="#F0F0F0",
        ),
        yaxis=dict(
            title="% del saldo total acumulado",
            ticksuffix="%", range=[0, 101], gridcolor="#F0F0F0",
        ),
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
        height=440,
    )
    st.plotly_chart(fig, width="stretch")

    _como_leer(
        f"El eje X ordena a los afiliados de mayor a menor saldo (0% = el más rico). "
        f"El eje Y muestra cuánto del saldo total se acumula. "
        f"La línea punteada es la distribución uniforme hipotética. "
        f"Cuanto más curva esté la línea naranja por encima, más concentrada está "
        f"la cartera. Un traspaso en el top 20% pesa **{top20 / 20:.1f}×** uno "
        f"en un afiliado promedio."
    )


def _seccion_segmentacion_reglas(df: pd.DataFrame) -> None:
    """Capítulo 3 — Diagnosticar: reglas compuestas que identifican al segmento fugado."""
    _bloque_titulo(
        "Capítulo 3 · Diagnosticar",
        "¿Quién se me va a ir? Reglas que lo identifican",
        "Cruce de tres dimensiones (edad × departamento × actividad). "
        "Cada fila es un segmento accionable: si respeta una regla simple, "
        "el equipo puede filtrar su CRM sin necesidad del modelo.",
    )

    reglas = segmentos_reglas(df, min_n=30)
    if reglas.empty:
        st.info("Sin segmentos suficientemente representativos para la selección actual.")
        return

    tasa_global = float(df["traspaso"].mean())

    # Top 8 segmentos con mayor fuga.
    top = reglas.head(8).copy()
    top["segmento"] = top.apply(
        lambda r: (
            f"Edad {r['tramo_edad']} · {r['departamento']} · "
            f"{'Activo' if r['aportante_activo'] else 'Inactivo'}"
        ),
        axis=1,
    )

    fig = px.bar(
        top.sort_values("tasa_fuga"),
        x="tasa_fuga",
        y="segmento",
        orientation="h",
        color="lift",
        color_continuous_scale=ITAU_ORANGE_SCALE,
        text=top.sort_values("tasa_fuga")["tasa_fuga"].map(lambda x: f"{x * 100:.0f}%"),
        hover_data={"n_afiliados": True, "lift": ":.2f", "saldo_total": ":,.0f"},
    )
    fig.update_traces(textposition="outside", textfont=dict(size=12))
    fig.add_vline(
        x=tasa_global, line_dash="dash", line_color=GREY_NEUTRAL,
        annotation_text=f"Promedio {tasa_global * 100:.1f}%",
        annotation_position="top right", annotation_font_size=11,
    )
    fig.update_layout(
        **_PLOTLY_LAYOUT, height=440,
        xaxis=dict(title="Tasa de fuga", tickformat=".0%", gridcolor="#F0F0F0",
                   range=[0, max(top["tasa_fuga"].max() * 1.25, 0.05)]),
        yaxis=dict(title=""),
        coloraxis_colorbar=dict(title="Lift", tickformat=".1f"),
    )
    st.plotly_chart(fig, width="stretch")

    peor = reglas.iloc[0]
    _como_leer(
        f"Cada barra es un segmento (cruce edad × departamento × actividad) con "
        f"al menos 30 afiliados. La longitud es la tasa de fuga del segmento; "
        f"el color indica el **lift** (cuántas veces fuga más que el promedio). "
        f"La línea punteada gris es la media de la cartera filtrada. "
        f"**Diagnóstico top:** el segmento **edad {peor['tramo_edad']} × "
        f"{peor['departamento']} × "
        f"{'activo' if peor['aportante_activo'] else 'inactivo'}** fuga al "
        f"**{peor['tasa_fuga'] * 100:.1f}%** (lift {peor['lift']:.2f}×), con "
        f"{_fmt_int(peor['n_afiliados'])} afiliados y "
        f"UYU {peor['saldo_total'] / 1e6:,.1f} M en juego."
    )


def _seccion_cross_sell(df: pd.DataFrame) -> None:
    """Capítulo 4 — Oportunidad: cross-sell y riesgo de sobre-producto."""
    _bloque_titulo(
        "Capítulo 4 · Oportunidad",
        "¿Cuántos productos tiene cada afiliado? Y cómo se comporta",
        "El número de productos contratados es a la vez una palanca de fidelización "
        "y una señal de alarma: el *sobre-producto* (3-4) suele correlacionar con fuga.",
    )

    cs = cross_sell_resumen(df)
    if cs.empty:
        st.info("Sin datos.")
        return

    # Gráfico: barras (n_afiliados) + línea (tasa_fuga) por # productos.
    fig = go.Figure()
    fig.add_bar(
        x=cs["productos_contratados"].astype(str),
        y=cs["n_afiliados"],
        marker_color=ITAU_ORANGE_LIGHT,
        name="Afiliados",
        text=cs["n_afiliados"].map(lambda x: f"{int(x):,}".replace(",", ".")),
        textposition="outside",
        hovertemplate="%{x} producto(s)<br>%{y:,.0f} afiliados<extra></extra>",
    )
    fig.add_trace(
        go.Scatter(
            x=cs["productos_contratados"].astype(str),
            y=cs["tasa_fuga"],
            mode="lines+markers+text",
            line=dict(color=ITAU_BLUE, width=3),
            marker=dict(size=10),
            name="Tasa de fuga",
            text=cs["tasa_fuga"].map(lambda x: f"{x * 100:.0f}%"),
            textposition="top center",
            textfont=dict(color=ITAU_BLUE, size=12),
            yaxis="y2",
            hovertemplate="%{x} producto(s)<br>Fuga %{y:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        **_PLOTLY_LAYOUT, height=400,
        xaxis=dict(title="Productos contratados"),
        yaxis=dict(title="N° afiliados", gridcolor="#F0F0F0"),
        yaxis2=dict(
            title="Tasa de fuga", overlaying="y", side="right",
            tickformat=".0%",
            range=[0, max(cs["tasa_fuga"].max() * 1.4, 0.05)],
            showgrid=False,
        ),
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig, width="stretch")

    # Oportunidades accionables.
    oport = oportunidades_cross_sell(df, saldo_minimo=100_000)
    n_op = len(oport)

    # Insight narrativo automático (mismo formato "Cómo leerlo" unificado).
    productos_peor = cs.loc[cs["tasa_fuga"].idxmax(), "productos_contratados"]
    fuga_peor = cs["tasa_fuga"].max() * 100
    productos_mejor = cs.loc[cs["tasa_fuga"].idxmin(), "productos_contratados"]
    fuga_mejor = cs["tasa_fuga"].min() * 100

    if n_op == 0:
        _como_leer(
            f"Las barras naranjas son afiliados por # de productos; la línea azul "
            f"su tasa de fuga. La fuga es mínima con **{productos_mejor} producto(s)** "
            f"({fuga_mejor:.1f}%) y máxima con **{productos_peor} producto(s)** "
            f"({fuga_peor:.1f}%). En la selección actual no hay candidatos a cross-sell "
            f"(activo + 1 producto + saldo ≥ UYU 100k + no traspasado)."
        )
        return

    saldo_op = oport["saldo_cuenta"].sum()
    _como_leer(
        f"Las barras naranjas son afiliados por # de productos; la línea azul su "
        f"tasa de fuga. La fuga es mínima con **{productos_mejor} producto(s)** "
        f"({fuga_mejor:.1f}%) y máxima con **{productos_peor} producto(s)** "
        f"({fuga_peor:.1f}%). **{_fmt_int(n_op)} afiliados** activos tienen hoy un "
        f"solo producto y saldo ≥ UYU 100k — **UYU {saldo_op / 1e6:,.1f} M** en juego "
        f"si les sumamos un producto voluntario."
    )

    st.markdown("**Top 20 candidatos a cross-sell** (ordenados por saldo):")
    tabla = oport.head(20).rename(
        columns={
            "afiliado_id": "ID",
            "apellido": "Apellido",
            "departamento": "Departamento",
            "edad": "Edad",
            "saldo_cuenta": "Saldo (UYU)",
            "anios_afiliacion": "Años afiliación",
            "salario_nominal": "Salario nominal (UYU)",
        }
    )
    st.dataframe(
        tabla,
        width="stretch",
        hide_index=True,
        column_config={
            "Saldo (UYU)": st.column_config.NumberColumn(format="%.0f"),
            "Salario nominal (UYU)": st.column_config.NumberColumn(format="%.0f"),
        },
    )


def _seccion_modelo(df: pd.DataFrame, model) -> None:
    """Capítulo 5 — Cuantificar: modelo logístico + intersección Pareto × riesgo."""
    scores = scorear_afiliados(model, df)
    if len(scores) == 0:
        st.info("Sin afiliados para scorear.")
        return

    # Enriquecer con saldo, departamento, apellido, edad.
    enr = scores.merge(
        df[["afiliado_id", "apellido", "departamento", "edad", "saldo_cuenta"]],
        on="afiliado_id",
        how="left",
    )

    # Intersección: top-20% por saldo Y top-100 por riesgo.
    saldo_corte = df["saldo_cuenta"].quantile(0.80)
    n_top_riesgo = min(100, len(enr))
    top_riesgo = enr.head(n_top_riesgo)
    lista_dorada = top_riesgo[top_riesgo["saldo_cuenta"] >= saldo_corte].copy()

    saldo_dorado = lista_dorada["saldo_cuenta"].sum()

    _bloque_titulo(
        "Capítulo 5 · Cuantificar",
        f"{len(lista_dorada)} afiliados que valen UYU {saldo_dorado / 1e6:,.1f} M "
        f"y están por irse",
        "Intersección entre los dos hilos de la historia: afiliados del **top 20% "
        "de saldo** (capítulo 2) que además están en el **top 100 de riesgo** del "
        "modelo (capítulo 3 diagnosticado + capítulo 5 cuantificado).",
    )

    # KPIs de la lista dorada.
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Lista dorada",
        f"{len(lista_dorada)} afiliados",
        help="Top-100 riesgo ∩ Top-20% saldo.",
    )
    c2.metric(
        "Saldo en juego",
        f"UYU {saldo_dorado / 1e6:,.1f} M",
    )
    c3.metric(
        "Prob. fuga promedio",
        f"{lista_dorada['prob_fuga'].mean() * 100:.1f} %"
        if len(lista_dorada) else "—",
    )

    # Scatter: saldo vs prob_fuga, con cuadrante dorado sombreado.
    st.markdown(
        f"<div style='font-weight:600; color:{TEXT_COLOR}; margin-top:12px;'>"
        f"Cartera entera · saldo vs. probabilidad de fuga</div>",
        unsafe_allow_html=True,
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=enr["saldo_cuenta"],
            y=enr["prob_fuga"],
            mode="markers",
            marker=dict(
                size=5,
                color=enr["prob_fuga"],
                colorscale=ITAU_ORANGE_SCALE,
                opacity=0.55,
                line=dict(width=0),
            ),
            name="Afiliados",
            hovertemplate="Saldo UYU %{x:,.0f}<br>Prob. fuga %{y:.1%}<extra></extra>",
        )
    )
    # Marcar cuadrante dorado (saldo ≥ corte Y prob ≥ min de la lista dorada).
    prob_min_dorada = lista_dorada["prob_fuga"].min() if len(lista_dorada) else 1.0
    fig.add_shape(
        type="rect",
        x0=saldo_corte, x1=enr["saldo_cuenta"].max() * 1.02,
        y0=prob_min_dorada, y1=1.0,
        line=dict(color=ITAU_BLUE, width=2, dash="dot"),
        fillcolor="rgba(0, 59, 113, 0.08)",
    )
    fig.add_annotation(
        x=enr["saldo_cuenta"].max() * 0.98,
        y=min(prob_min_dorada + 0.05, 0.97),
        text=f"<b>Lista dorada</b><br>{len(lista_dorada)} afiliados",
        showarrow=False, font=dict(color=ITAU_BLUE, size=12),
        bgcolor="rgba(255,255,255,0.95)", bordercolor=ITAU_BLUE, borderwidth=1,
        borderpad=6, xanchor="right",
    )
    fig.update_layout(
        **_PLOTLY_LAYOUT, height=420, showlegend=False,
        xaxis=dict(title="Saldo de cuenta (UYU)", gridcolor="#F0F0F0"),
        yaxis=dict(title="Probabilidad de fuga", tickformat=".0%",
                   range=[0, 1], gridcolor="#F0F0F0"),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, width="stretch")
    _como_leer(
        "Cada punto es un afiliado. El eje X es su saldo, el eje Y la probabilidad "
        "de fuga estimada por el modelo. El **cuadrante azul arriba a la derecha** "
        "es la *lista dorada*: afiliados que simultáneamente tienen mucho saldo "
        "y alta probabilidad de irse. Ese cuadrante es la prioridad absoluta del "
        "equipo de retención — un puñado de llamadas que protegen millones."
    )

    # Lista dorada como tabla accionable.
    if len(lista_dorada) > 0:
        st.markdown(
            "**Llamadas prioritarias esta semana** · ordenadas por saldo (mayor → menor):"
        )
        vista = lista_dorada.sort_values("saldo_cuenta", ascending=False).rename(
            columns={
                "afiliado_id": "ID",
                "apellido": "Apellido",
                "departamento": "Departamento",
                "edad": "Edad",
                "saldo_cuenta": "Saldo (UYU)",
                "prob_fuga": "Prob. fuga",
            }
        )[["ID", "Apellido", "Departamento", "Edad", "Saldo (UYU)", "Prob. fuga"]]
        st.dataframe(
            vista,
            width="stretch",
            hide_index=True,
            column_config={
                "Saldo (UYU)": st.column_config.NumberColumn(format="%.0f"),
                "Prob. fuga": st.column_config.ProgressColumn(
                    format="%.1f%%", min_value=0.0, max_value=1.0
                ),
            },
        )

    # Drivers interpretables.
    st.markdown(
        f"<div style='margin-top:20px; font-size:14px; color:#4B5563;'>"
        f"<b style='color:{ITAU_BLUE};'>¿Por qué estos afiliados?</b> "
        f"Los drivers del modelo ordenados por magnitud del coeficiente "
        f"(naranja = aumenta la fuga, azul = la reduce):</div>",
        unsafe_allow_html=True,
    )
    drv = drivers_principales(model, top_n=8).sort_values("coef")
    fig = px.bar(
        drv,
        x="coef",
        y="feature",
        orientation="h",
        color="coef",
        color_continuous_scale=ITAU_DIVERGING,
        color_continuous_midpoint=0,
        text=drv["coef"].map(lambda x: f"{x:+.2f}"),
    )
    fig.update_traces(textposition="outside", textfont=dict(size=12))
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        height=340,
        xaxis=dict(title="Coeficiente logístico", gridcolor="#F0F0F0", zeroline=True,
                   zerolinecolor=GREY_NEUTRAL, zerolinewidth=1),
        yaxis=dict(title=""),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, width="stretch")
    _como_leer(
        "Cada barra es una variable del modelo. Cuanto más a la derecha, más "
        "**aumenta** la probabilidad de fuga; cuanto más a la izquierda, más la "
        "**reduce**. Las variables se muestran en escala estandarizada, así que "
        "la longitud de la barra es comparable entre ellas. Útil para comunicar "
        "al equipo comercial **por qué** el modelo marcó a cada afiliado."
    )


def _header_principal() -> None:
    st.markdown(
        f"""
        <div style='padding: 10px 0 0 0;'>
            <div style='font-size:12px; font-weight:700; color:{ITAU_ORANGE};
                        text-transform:uppercase; letter-spacing:2px;'>
                AFAP Portfolio Analytics
            </div>
            <h1 style='color:{ITAU_BLUE}; font-size:34px; margin:4px 0 6px 0;
                       line-height:1.15;'>
                De 10.000 afiliados a una lista<br>
                de llamadas esta semana.
            </h1>
            <p style='color:#4B5563; font-size:15px; margin:0; max-width:820px;'>
                Cinco capítulos que recorren el flujo completo del análisis comercial:
                <b>describir</b> los datos, <b>priorizar</b> la cartera,
                <b>diagnosticar</b> la fuga, detectar <b>oportunidades</b> de cross-sell
                y <b>cuantificar</b> el riesgo con un modelo interpretable.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _seccion_cierre(df: pd.DataFrame, modelo) -> None:
    """Cierre narrativo: mapea el dashboard al aviso de Analista de Datos Comercial."""
    _bloque_titulo(
        "Cierre · Por qué este demo encaja con el puesto",
        "Cada capítulo corresponde a una responsabilidad del aviso",
        "Lectura para el equipo de selección: qué parte del trabajo diario del rol "
        "Analista de Datos Comercial queda demostrada en cada capítulo.",
    )

    # 1) Mapeo capítulo → responsabilidad del aviso.
    mapeo = [
        (
            "Cap. 1 · Describir",
            "Control de calidad de datos",
            "Validación de schema, detección de missing/duplicados, rangos — la "
            "base de cualquier reporte confiable.",
        ),
        (
            "Cap. 2 · Priorizar",
            "Análisis de afiliados y seguimiento de performance",
            "Concentración de cartera (curva de Lorenz) — sabe dónde poner foco "
            "el equipo comercial.",
        ),
        (
            "Cap. 3 · Diagnosticar",
            "Relevamiento y análisis de información de mercado",
            "Reglas compuestas con lift — diagnóstico accionable por segmento "
            "sin necesidad de modelo.",
        ),
        (
            "Cap. 4 · Oportunidad",
            "Análisis de productos para detectar oportunidades",
            "Cross-sell cuantificado: UYU en juego si se suma un producto "
            "voluntario a los activos con un solo producto.",
        ),
        (
            "Cap. 5 · Cuantificar",
            "Apoyo al equipo comercial en la interpretación de datos",
            "Modelo logístico interpretable + intersección Pareto × riesgo → "
            "lista dorada de llamadas priorizadas.",
        ),
        (
            "Excel descargable",
            "Preparación y actualización de reportes comerciales",
            "Libro multi-hoja generado on-the-fly, reproducible con los filtros "
            "aplicados — automatización básica de procesos.",
        ),
    ]

    # Render de la tabla de mapeo.
    tabla_html = (
        "<table style='width:100%; border-collapse:collapse; margin:8px 0 20px 0; "
        "font-size:13px;'>"
        f"<thead><tr style='background:{ITAU_BLUE}; color:white;'>"
        "<th style='padding:10px 12px; text-align:left; width:18%;'>Capítulo</th>"
        "<th style='padding:10px 12px; text-align:left; width:32%;'>"
        "Responsabilidad del aviso</th>"
        "<th style='padding:10px 12px; text-align:left;'>Cómo queda demostrada</th>"
        "</tr></thead><tbody>"
    )
    for i, (cap, resp, explica) in enumerate(mapeo):
        bg = "#FFFFFF" if i % 2 == 0 else "#F8F8F8"
        tabla_html += (
            f"<tr style='background:{bg}; border-bottom:1px solid #E5E7EB;'>"
            f"<td style='padding:10px 12px; font-weight:600; color:{ITAU_ORANGE};'>"
            f"{cap}</td>"
            f"<td style='padding:10px 12px; color:{TEXT_COLOR};'>{resp}</td>"
            f"<td style='padding:10px 12px; color:#4B5563;'>{explica}</td>"
            "</tr>"
        )
    tabla_html += "</tbody></table>"
    st.markdown(tabla_html, unsafe_allow_html=True)

    # 2) Stack técnico vs requisitos del aviso — misma tabla sobria.
    stack = [
        ("SQL", "DuckDB", "4 scripts en `sql/` con test de paridad contra Python."),
        ("Python", "Módulos puros + 105 tests", "Cobertura en `tests/`, CI con ruff + pytest."),
        ("Visualización", "Streamlit + Plotly", "Equivalente funcional a Power BI."),
        ("Calidad", "Validación temprana",
         "Fail-fast con mensajes accionables en `data_loader.py`."),
    ]
    tabla2_html = (
        "<div style='font-weight:700; color:" + ITAU_BLUE + "; font-size:16px; "
        "margin:18px 0 6px 0;'>Requisitos técnicos del aviso · match del demo</div>"
        "<table style='width:100%; border-collapse:collapse; margin:0 0 20px 0; "
        "font-size:13px;'>"
        f"<thead><tr style='background:{ITAU_BLUE}; color:white;'>"
        "<th style='padding:10px 12px; text-align:left; width:18%;'>Requisito</th>"
        "<th style='padding:10px 12px; text-align:left; width:32%;'>Stack usado</th>"
        "<th style='padding:10px 12px; text-align:left;'>Evidencia en el repo</th>"
        "</tr></thead><tbody>"
    )
    for i, (req, st_used, ev) in enumerate(stack):
        bg = "#FFFFFF" if i % 2 == 0 else "#F8F8F8"
        tabla2_html += (
            f"<tr style='background:{bg}; border-bottom:1px solid #E5E7EB;'>"
            f"<td style='padding:10px 12px; font-weight:600; color:{ITAU_ORANGE};'>"
            f"{req}</td>"
            f"<td style='padding:10px 12px; color:{TEXT_COLOR};'>{st_used}</td>"
            f"<td style='padding:10px 12px; color:#4B5563;'>{ev}</td>"
            "</tr>"
        )
    tabla2_html += "</tbody></table>"
    st.markdown(tabla2_html, unsafe_allow_html=True)

    excel = _excel_bytes(df, modelo)
    st.download_button(
        "📥 Descargar reporte Excel",
        data=excel,
        file_name="afap_portfolio_analytics.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )

    # 4) Firma / contacto.
    st.markdown(
        f"<div style='margin-top:28px; padding:18px 0; border-top:1px solid #E5E7EB; "
        f"text-align:center; color:#6B7280; font-size:13px;'>"
        f"<div style='font-weight:700; color:{TEXT_COLOR}; font-size:15px;'>"
        f"Mathias · Postulación Analista de Datos Junior — Equipo Comercial</div>"
        f"<div style='margin-top:6px;'>"
        f"📧 <a href='mailto:seleccion@afapitau.com.uy' style='color:{ITAU_BLUE};'>"
        f"seleccion@afapitau.com.uy</a> · "
        f"📂 <a href='https://github.com/mathiasmtt/afap-portfolio-analytics' "
        f"style='color:{ITAU_BLUE};'>github.com/mathiasmtt/afap-portfolio-analytics</a>"
        f"</div>"
        f"<div style='margin-top:8px; font-size:11px; color:#9CA3AF;'>"
        f"Dataset: Bank Customer Churn (Kaggle) reframeado al vocabulario AFAP · "
        f"Stack: Python · SQL (DuckDB) · Streamlit · scikit-learn · "
        f"CI: GitHub Actions"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    _header_principal()

    df = _cargar_dataset()
    if df.empty:
        st.warning(
            "No se encontró `data/raw/Customer-Churn-Records.csv`. "
            "Descargá el dataset de Kaggle (Bank Customer Churn) y colocalo allí."
        )
        st.stop()

    filtrado = _aplicar_filtros(df)

    st.divider()
    _seccion_calidad_eda(filtrado)

    st.divider()
    _seccion_pareto(filtrado)

    st.divider()
    _seccion_segmentacion_reglas(filtrado)

    st.divider()
    _seccion_cross_sell(filtrado)

    st.divider()
    modelo = _entrenar_modelo(df)  # entrena sobre la población completa
    _seccion_modelo(filtrado, modelo)

    st.divider()
    _seccion_cierre(filtrado, modelo)


if __name__ == "__main__":
    main()
