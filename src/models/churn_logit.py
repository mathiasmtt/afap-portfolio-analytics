"""Logística interpretable para scoring de fuga de afiliados.

Se elige LogisticRegression sobre XGBoost deliberadamente: el equipo
comercial necesita poder **explicar** por qué un afiliado está en riesgo,
no maximizar AUC. Los coeficientes se traducen a lenguaje narrativo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Features del modelo (solo atributos del afiliado, nunca el target).
FEATURES_NUMERICAS: list[str] = [
    "edad",
    "anios_afiliacion",
    "saldo_cuenta",
    "productos_contratados",
    "score_interno",
    "salario_nominal",
]
FEATURES_CATEGORICAS: list[str] = ["departamento", "genero"]
FEATURES_BINARIAS: list[str] = ["aportante_activo", "tiene_producto_adicional"]

TARGET: str = "traspaso"


@dataclass
class ResultadoScoring:
    """Contenedor del DataFrame de scoring con metadatos."""

    scores: pd.DataFrame  # columnas: afiliado_id, prob_fuga
    umbral_top_5pct: float


def _build_pipeline(seed: int = 42) -> Pipeline:
    """Construye el pipeline de preprocesamiento + logística."""
    preproc = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), FEATURES_NUMERICAS),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                FEATURES_CATEGORICAS,
            ),
            ("bin", "passthrough", FEATURES_BINARIAS),
        ]
    )
    pipe = Pipeline(
        steps=[
            ("preproc", preproc),
            (
                "logit",
                LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )
    return pipe


def entrenar_logit(df: pd.DataFrame, seed: int = 42) -> Pipeline:
    """Entrena la logística sobre el DataFrame AFAP completo.

    No se hace split train/test aquí: el objetivo no es medir generalización
    sino generar un ranking interpretable para la acción comercial. Para
    auditar performance, ver el notebook ``01_insights_comerciales.ipynb``.
    """
    _validar_columnas(df)
    X = df[FEATURES_NUMERICAS + FEATURES_CATEGORICAS + FEATURES_BINARIAS]
    y = df[TARGET].astype(int)
    pipe = _build_pipeline(seed=seed)
    pipe.fit(X, y)
    return pipe


def scorear_afiliados(model: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """Predice probabilidad de fuga para cada afiliado.

    Retorna DataFrame con columnas: ``afiliado_id``, ``prob_fuga``,
    ordenado descendente por probabilidad.
    """
    _validar_columnas(df, requiere_target=False)
    X = df[FEATURES_NUMERICAS + FEATURES_CATEGORICAS + FEATURES_BINARIAS]
    probs = model.predict_proba(X)[:, 1]
    out = pd.DataFrame(
        {
            "afiliado_id": df["afiliado_id"].to_numpy(),
            "prob_fuga": probs,
        }
    )
    return out.sort_values("prob_fuga", ascending=False).reset_index(drop=True)


def drivers_principales(model: Pipeline, top_n: int = 8) -> pd.DataFrame:
    """Extrae los coeficientes más influyentes en formato narrativo.

    Retorna DataFrame con: ``feature``, ``coef``, ``abs_coef``, ``signo``,
    ``narrativa``. Ordenado por ``abs_coef`` desc.
    """
    logit: LogisticRegression = model.named_steps["logit"]
    preproc: ColumnTransformer = model.named_steps["preproc"]

    feature_names: list[str] = []
    for nombre, _transf, cols in preproc.transformers_:
        if nombre == "num":
            feature_names.extend(cols)
        elif nombre == "cat":
            ohe: OneHotEncoder = preproc.named_transformers_["cat"]
            feature_names.extend(ohe.get_feature_names_out(cols).tolist())
        elif nombre == "bin":
            feature_names.extend(cols)

    coefs = logit.coef_[0]
    assert len(coefs) == len(feature_names), (
        f"Mismatch de features: {len(coefs)} coefs vs {len(feature_names)} nombres"
    )

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "coef": coefs,
        }
    )
    df["abs_coef"] = df["coef"].abs()
    df["signo"] = np.where(df["coef"] >= 0, "aumenta", "reduce")
    df["narrativa"] = df.apply(
        lambda r: f"'{r['feature']}' {r['signo']} la probabilidad de fuga", axis=1
    )
    df = df.sort_values("abs_coef", ascending=False).reset_index(drop=True)
    return df.head(top_n)


def _validar_columnas(df: pd.DataFrame, requiere_target: bool = True) -> None:
    base = FEATURES_NUMERICAS + FEATURES_CATEGORICAS + FEATURES_BINARIAS + ["afiliado_id"]
    requeridas = set(base + ([TARGET] if requiere_target else []))
    faltantes = requeridas - set(df.columns)
    if faltantes:
        raise ValueError(f"Columnas faltantes para el modelo: {sorted(faltantes)}")
