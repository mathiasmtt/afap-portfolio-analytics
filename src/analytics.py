"""Funciones puras de analítica comercial sobre cartera AFAP.

Todas las funciones reciben un DataFrame ya reframeado al vocabulario AFAP
(ver :mod:`src.data_loader`). Ninguna función lee archivos ni muta estado
global; todas retornan estructuras nuevas.

Estas funciones son la **única fuente de verdad** de los KPIs del proyecto:
el notebook, la app Streamlit y los scripts SQL las consumen o las replican.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Segmentaciones canónicas (bins + labels)
# ---------------------------------------------------------------------------

BINS_EDAD: list[float] = [0, 35, 45, 55, 65, 200]
LABELS_EDAD: list[str] = ["<35", "35-45", "45-55", "55-65", "65+"]

BINS_ANTIGUEDAD: list[float] = [-1, 2, 5, 8, 100]
LABELS_ANTIGUEDAD: list[str] = ["0-2", "3-5", "6-8", "9+"]

BINS_SALDO: list[float] = [-1, 0, 50_000, 100_000, 150_000, float("inf")]
LABELS_SALDO: list[str] = ["Sin saldo", "Bajo", "Medio", "Alto", "Premium"]


# ---------------------------------------------------------------------------
# KPIs globales
# ---------------------------------------------------------------------------


def kpis_globales(df: pd.DataFrame) -> dict[str, float]:
    """KPIs de cabecera del dashboard."""
    if len(df) == 0:
        return {
            "n_afiliados": 0,
            "saldo_total": 0.0,
            "ticket_promedio": 0.0,
            "tasa_fuga": 0.0,
            "pct_activos": 0.0,
        }
    return {
        "n_afiliados": int(len(df)),
        "saldo_total": float(df["saldo_cuenta"].sum()),
        "ticket_promedio": float(df["saldo_cuenta"].mean()),
        "tasa_fuga": float(df["traspaso"].mean()),
        "pct_activos": float(df["aportante_activo"].mean()),
    }


# ---------------------------------------------------------------------------
# Pareto de cartera
# ---------------------------------------------------------------------------


def pareto_cartera(df: pd.DataFrame) -> pd.DataFrame:
    """Afiliados ordenados por saldo desc, con % acumulado de cartera.

    Columnas de salida: ``afiliado_id``, ``saldo_cuenta``, ``rank``,
    ``pct_acumulado``, ``pct_afiliados``, ``top_80``.
    """
    if len(df) == 0:
        return pd.DataFrame(
            columns=[
                "afiliado_id",
                "saldo_cuenta",
                "rank",
                "pct_acumulado",
                "pct_afiliados",
                "top_80",
            ]
        )
    ordenado = df[["afiliado_id", "saldo_cuenta"]].sort_values(
        "saldo_cuenta", ascending=False, kind="mergesort"
    ).reset_index(drop=True)
    total = ordenado["saldo_cuenta"].sum()
    ordenado["rank"] = np.arange(1, len(ordenado) + 1)
    ordenado["pct_acumulado"] = (
        ordenado["saldo_cuenta"].cumsum() / total if total > 0 else 0.0
    )
    ordenado["pct_afiliados"] = ordenado["rank"] / len(ordenado)
    ordenado["top_80"] = ordenado["pct_acumulado"] <= 0.80
    return ordenado


def concentracion_cartera(df: pd.DataFrame, tope_pct: float = 0.20) -> float:
    """% del saldo total concentrado en el ``tope_pct`` superior de afiliados."""
    if len(df) == 0 or tope_pct <= 0:
        return 0.0
    saldos = df["saldo_cuenta"].sort_values(ascending=False).reset_index(drop=True)
    total = saldos.sum()
    if total == 0:
        return 0.0
    n_top = max(1, int(round(len(saldos) * tope_pct)))
    return float(saldos.iloc[:n_top].sum() / total)


# ---------------------------------------------------------------------------
# Segmentaciones
# ---------------------------------------------------------------------------


def segmentacion_edad(df: pd.DataFrame) -> pd.Series:
    """Devuelve una Serie categórica con el tramo etario por afiliado."""
    return pd.cut(
        df["edad"], bins=BINS_EDAD, labels=LABELS_EDAD, right=False, include_lowest=True
    )


def segmentacion_antiguedad(df: pd.DataFrame) -> pd.Series:
    """Devuelve una Serie categórica con el tramo de antigüedad."""
    return pd.cut(
        df["anios_afiliacion"],
        bins=BINS_ANTIGUEDAD,
        labels=LABELS_ANTIGUEDAD,
        right=True,
    )


def segmentacion_saldo(df: pd.DataFrame) -> pd.Series:
    """Devuelve una Serie categórica con el tramo de saldo."""
    return pd.cut(
        df["saldo_cuenta"], bins=BINS_SALDO, labels=LABELS_SALDO, right=True
    )


def aplicar_segmentaciones(df: pd.DataFrame) -> pd.DataFrame:
    """Versión conveniente: agrega tres columnas al DataFrame."""
    out = df.copy()
    out["tramo_edad"] = segmentacion_edad(df)
    out["tramo_antiguedad"] = segmentacion_antiguedad(df)
    out["tramo_saldo"] = segmentacion_saldo(df)
    return out


# ---------------------------------------------------------------------------
# Tasa de fuga por segmento
# ---------------------------------------------------------------------------


def tasa_fuga_por_segmento(
    df: pd.DataFrame,
    col: str,
    min_n: int = 1,
) -> pd.DataFrame:
    """Tasa de fuga agregada por valor de ``col``.

    Columnas de salida: ``col``, ``n_afiliados``, ``n_fugas``, ``tasa_fuga``,
    ``saldo_total``. Filtra grupos con menos de ``min_n`` afiliados.
    """
    if col not in df.columns:
        raise KeyError(f"Columna '{col}' no está en el DataFrame")
    if len(df) == 0:
        return pd.DataFrame(
            columns=[col, "n_afiliados", "n_fugas", "tasa_fuga", "saldo_total"]
        )
    agg = (
        df.groupby(col, observed=True)
        .agg(
            n_afiliados=("afiliado_id", "count"),
            n_fugas=("traspaso", "sum"),
            saldo_total=("saldo_cuenta", "sum"),
        )
        .reset_index()
    )
    agg["tasa_fuga"] = agg["n_fugas"] / agg["n_afiliados"]
    agg = agg[agg["n_afiliados"] >= min_n].reset_index(drop=True)
    return agg.sort_values("tasa_fuga", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cohortes de afiliación
# ---------------------------------------------------------------------------


def cohortes_afiliacion(
    df: pd.DataFrame,
    metrica: Literal["saldo_total", "n_afiliados", "tasa_fuga"] = "saldo_total",
) -> pd.DataFrame:
    """Agregado por cohorte (``anios_afiliacion``).

    Retorna un DataFrame con columnas ``anios_afiliacion``, ``n_afiliados``,
    ``saldo_total``, ``tasa_fuga`` y ``metrica`` (alias de la métrica pedida).
    """
    if metrica not in {"saldo_total", "n_afiliados", "tasa_fuga"}:
        raise ValueError(f"Métrica desconocida: {metrica}")
    if len(df) == 0:
        return pd.DataFrame(
            columns=["anios_afiliacion", "n_afiliados", "saldo_total", "tasa_fuga", "metrica"]
        )
    agg = (
        df.groupby("anios_afiliacion", observed=True)
        .agg(
            n_afiliados=("afiliado_id", "count"),
            saldo_total=("saldo_cuenta", "sum"),
            tasa_fuga=("traspaso", "mean"),
        )
        .reset_index()
        .sort_values("anios_afiliacion")
        .reset_index(drop=True)
    )
    agg["metrica"] = agg[metrica]
    return agg


# ---------------------------------------------------------------------------
# Heatmap edad x antigüedad
# ---------------------------------------------------------------------------


def heatmap_edad_antiguedad(
    df: pd.DataFrame,
    valor: Literal["tasa_fuga", "n_afiliados", "saldo_total"] = "tasa_fuga",
) -> pd.DataFrame:
    """Matriz (filas=tramo_edad, columnas=tramo_antiguedad) con el valor pedido."""
    if valor not in {"tasa_fuga", "n_afiliados", "saldo_total"}:
        raise ValueError(f"Valor no soportado: {valor}")
    seg = aplicar_segmentaciones(df)
    if len(seg) == 0:
        return pd.DataFrame(index=LABELS_EDAD, columns=LABELS_ANTIGUEDAD, dtype=float)
    if valor == "tasa_fuga":
        mat = seg.pivot_table(
            index="tramo_edad",
            columns="tramo_antiguedad",
            values="traspaso",
            aggfunc="mean",
            observed=False,
        )
    elif valor == "n_afiliados":
        mat = seg.pivot_table(
            index="tramo_edad",
            columns="tramo_antiguedad",
            values="afiliado_id",
            aggfunc="count",
            observed=False,
        )
    else:
        mat = seg.pivot_table(
            index="tramo_edad",
            columns="tramo_antiguedad",
            values="saldo_cuenta",
            aggfunc="sum",
            observed=False,
        )
    return mat.reindex(index=LABELS_EDAD, columns=LABELS_ANTIGUEDAD)


# ---------------------------------------------------------------------------
# Capítulo 1 — Calidad de datos y EDA
# ---------------------------------------------------------------------------


COLUMNAS_CANONICAS_AFAP: tuple[str, ...] = (
    "afiliado_id",
    "apellido",
    "score_interno",
    "departamento",
    "genero",
    "edad",
    "anios_afiliacion",
    "saldo_cuenta",
    "productos_contratados",
    "tiene_producto_adicional",
    "aportante_activo",
    "salario_nominal",
    "traspaso",
)


def resumen_calidad(df: pd.DataFrame) -> dict:
    """Snapshot de calidad del dataset para el capítulo descriptivo.

    Retorna métricas que el equipo comercial puede leer de un vistazo para
    confiar (o no) en los números posteriores.
    """
    n_filas = int(len(df))
    n_cols = int(len(df.columns))
    missing_por_col = (
        df.isna().sum().to_dict() if n_filas else {c: 0 for c in df.columns}
    )
    total_missing = int(sum(missing_por_col.values()))
    pct_missing = (total_missing / (n_filas * n_cols) * 100) if n_filas and n_cols else 0.0
    n_duplicados = int(df.duplicated(subset=["afiliado_id"]).sum()) if "afiliado_id" in df else 0
    tasa_target = float(df["traspaso"].mean()) if "traspaso" in df and n_filas else 0.0
    schema_ok = set(COLUMNAS_CANONICAS_AFAP).issubset(set(df.columns))

    rangos = {}
    for col in ("edad", "anios_afiliacion", "saldo_cuenta", "salario_nominal"):
        if col in df.columns and n_filas:
            rangos[col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
            }

    return {
        "n_filas": n_filas,
        "n_columnas": n_cols,
        "n_duplicados": n_duplicados,
        "total_missing": total_missing,
        "pct_missing": pct_missing,
        "tasa_target": tasa_target,
        "schema_afap_ok": schema_ok,
        "missing_por_col": missing_por_col,
        "rangos": rangos,
    }


# ---------------------------------------------------------------------------
# Capítulo 3 — Segmentación por reglas (diagnóstico de fuga)
# ---------------------------------------------------------------------------


def segmentos_reglas(df: pd.DataFrame, min_n: int = 30) -> pd.DataFrame:
    """Cruza 3 dimensiones de segmentación y devuelve los segmentos con mayor
    tasa de fuga (filtrando los poco representativos).

    Útil para diagnosticar: "el problema está en este cruce concreto".
    """
    if len(df) == 0:
        return pd.DataFrame(
            columns=[
                "tramo_edad",
                "departamento",
                "aportante_activo",
                "n_afiliados",
                "n_fugas",
                "tasa_fuga",
                "saldo_total",
                "lift",
            ]
        )

    seg = aplicar_segmentaciones(df)
    agg = (
        seg.groupby(
            ["tramo_edad", "departamento", "aportante_activo"],
            observed=True,
        )
        .agg(
            n_afiliados=("afiliado_id", "count"),
            n_fugas=("traspaso", "sum"),
            saldo_total=("saldo_cuenta", "sum"),
        )
        .reset_index()
    )
    agg["tasa_fuga"] = agg["n_fugas"] / agg["n_afiliados"]
    agg = agg[agg["n_afiliados"] >= min_n].reset_index(drop=True)

    tasa_global = float(df["traspaso"].mean()) if float(df["traspaso"].mean()) > 0 else 1.0
    agg["lift"] = agg["tasa_fuga"] / tasa_global
    return agg.sort_values("tasa_fuga", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Capítulo 4 — Oportunidad de cross-sell
# ---------------------------------------------------------------------------


def cross_sell_resumen(df: pd.DataFrame) -> pd.DataFrame:
    """Tasa de fuga, saldo promedio y headcount por # productos contratados.

    Responde: ¿a qué número de productos se cruza la frontera entre engagement
    y sobreproducto? Input para decidir acciones de cross-sell.
    """
    if len(df) == 0:
        return pd.DataFrame(
            columns=[
                "productos_contratados",
                "n_afiliados",
                "tasa_fuga",
                "saldo_promedio",
                "saldo_total",
                "pct_activos",
            ]
        )
    agg = (
        df.groupby("productos_contratados", observed=True)
        .agg(
            n_afiliados=("afiliado_id", "count"),
            tasa_fuga=("traspaso", "mean"),
            saldo_promedio=("saldo_cuenta", "mean"),
            saldo_total=("saldo_cuenta", "sum"),
            pct_activos=("aportante_activo", "mean"),
        )
        .reset_index()
        .sort_values("productos_contratados")
        .reset_index(drop=True)
    )
    return agg


def oportunidades_cross_sell(
    df: pd.DataFrame,
    saldo_minimo: float = 100_000,
) -> pd.DataFrame:
    """Afiliados activos con saldo alto y un solo producto contratado.

    Segmento de oro para cross-sell: ya aportan, tienen plata, tienen margen
    para agregar producto voluntario / seguros.
    """
    if len(df) == 0:
        return df.assign(potencial_cross_sell=[])
    mask = (
        (df["aportante_activo"] == 1)
        & (df["productos_contratados"] == 1)
        & (df["saldo_cuenta"] >= saldo_minimo)
        & (df["traspaso"] == 0)  # no incluir a los que ya se fueron
    )
    out = df.loc[mask].copy()
    out = out.sort_values("saldo_cuenta", ascending=False).reset_index(drop=True)
    return out[
        [
            "afiliado_id",
            "apellido",
            "departamento",
            "edad",
            "saldo_cuenta",
            "anios_afiliacion",
            "salario_nominal",
        ]
    ]


# ---------------------------------------------------------------------------
# Variación entre segmentos
# ---------------------------------------------------------------------------


def variacion_entre_segmentos(
    df: pd.DataFrame,
    col: str,
    baseline: str | None = None,
) -> pd.DataFrame:
    """Devuelve delta absoluto y relativo de tasa de fuga vs un baseline.

    Si ``baseline`` es None se usa la tasa global como referencia.
    """
    base = tasa_fuga_por_segmento(df, col)
    if len(base) == 0:
        return base.assign(delta_abs=[], delta_rel=[])
    if baseline is not None:
        if baseline not in base[col].values:
            raise ValueError(f"Baseline '{baseline}' no está en los valores de '{col}'")
        ref = float(base.loc[base[col] == baseline, "tasa_fuga"].iloc[0])
    else:
        ref = float(df["traspaso"].mean())
    base = base.copy()
    base["delta_abs"] = base["tasa_fuga"] - ref
    base["delta_rel"] = np.where(ref > 0, base["delta_abs"] / ref, np.nan)
    return base.sort_values("delta_abs", ascending=False).reset_index(drop=True)
