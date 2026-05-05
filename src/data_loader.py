"""Carga, validación y reframe del dataset bancario al vocabulario AFAP.

Este módulo aísla la única frontera donde el proyecto toca el dataset
original de Kaggle (Bank Customer Churn). Todo el resto de la aplicación
trabaja con los nombres canónicos AFAP definidos en el proyecto.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constantes canónicas (reframe bancario -> AFAP)
# ---------------------------------------------------------------------------

COLUMNAS_ORIGINALES: tuple[str, ...] = (
    "CustomerId",
    "Surname",
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Exited",
)

RENAME_MAP: dict[str, str] = {
    "CustomerId": "afiliado_id",
    "Surname": "apellido",
    "CreditScore": "score_interno",
    "Geography": "departamento",
    "Gender": "genero",
    "Age": "edad",
    "Tenure": "anios_afiliacion",
    "Balance": "saldo_cuenta",
    "NumOfProducts": "productos_contratados",
    "HasCrCard": "tiene_producto_adicional",
    "IsActiveMember": "aportante_activo",
    "EstimatedSalary": "salario_nominal",
    "Exited": "traspaso",
}

DEPARTAMENTO_MAP: dict[str, str] = {
    "France": "Montevideo",
    "Germany": "Canelones",
    "Spain": "Maldonado",
}

GENERO_MAP: dict[str, str] = {
    "Male": "M",
    "Female": "F",
}

COLUMNAS_AFAP: tuple[str, ...] = tuple(RENAME_MAP.values())


# ---------------------------------------------------------------------------
# Validación
# ---------------------------------------------------------------------------


def validar_columnas(df: pd.DataFrame) -> None:
    """Falla rápido si el DataFrame no trae todas las columnas originales.

    Mensaje accionable con las columnas faltantes.
    """
    faltantes = [c for c in COLUMNAS_ORIGINALES if c not in df.columns]
    if faltantes:
        raise ValueError(
            "Columnas faltantes en el CSV de entrada: "
            f"{faltantes}. Se esperaba el schema de Kaggle 'Bank Customer Churn'."
        )


# ---------------------------------------------------------------------------
# Reframe puro
# ---------------------------------------------------------------------------


def reframe_bancario_a_afap(df: pd.DataFrame) -> pd.DataFrame:
    """Transforma un DataFrame con el schema bancario al vocabulario AFAP.

    Función pura: no muta ``df``.
    """
    validar_columnas(df)
    out = df[list(COLUMNAS_ORIGINALES)].copy()
    out = out.rename(columns=RENAME_MAP)
    out["departamento"] = out["departamento"].map(DEPARTAMENTO_MAP)
    out["genero"] = out["genero"].map(GENERO_MAP)

    # Tipos explícitos
    out["afiliado_id"] = out["afiliado_id"].astype("int64")
    out["edad"] = out["edad"].astype("int64")
    out["anios_afiliacion"] = out["anios_afiliacion"].astype("int64")
    out["productos_contratados"] = out["productos_contratados"].astype("int64")
    out["tiene_producto_adicional"] = out["tiene_producto_adicional"].astype("int8")
    out["aportante_activo"] = out["aportante_activo"].astype("int8")
    out["traspaso"] = out["traspaso"].astype("int8")
    out["score_interno"] = out["score_interno"].astype("int64")
    out["saldo_cuenta"] = out["saldo_cuenta"].astype("float64")
    out["salario_nominal"] = out["salario_nominal"].astype("float64")

    # Validaciones post-reframe
    if out["departamento"].isna().any():
        desconocidos = df.loc[out["departamento"].isna(), "Geography"].unique().tolist()
        raise ValueError(
            f"Valores de 'Geography' sin mapeo a departamento AFAP: {desconocidos}. "
            f"Mapeo esperado: {DEPARTAMENTO_MAP}"
        )
    if out["genero"].isna().any():
        desconocidos = df.loc[out["genero"].isna(), "Gender"].unique().tolist()
        raise ValueError(f"Valores de 'Gender' no reconocidos: {desconocidos}")

    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Carga desde archivo
# ---------------------------------------------------------------------------


def cargar_csv(path: str | Path) -> pd.DataFrame:
    """Lee el CSV de Kaggle y devuelve el DataFrame reframeado al vocabulario AFAP.

    Parameters
    ----------
    path: ruta al archivo ``Churn_Modelling.csv``.

    Returns
    -------
    DataFrame con las columnas AFAP en :data:`COLUMNAS_AFAP`.
    """
    ruta = Path(path)
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset en {ruta}. "
            "Descargá 'Churn_Modelling.csv' de Kaggle y guardalo en data/raw/."
        )
    df = pd.read_csv(ruta)
    return reframe_bancario_a_afap(df)
