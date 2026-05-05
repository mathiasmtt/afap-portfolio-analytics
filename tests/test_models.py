"""Tests para src.models.churn_logit."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.models.churn_logit import (
    FEATURES_BINARIAS,
    FEATURES_CATEGORICAS,
    FEATURES_NUMERICAS,
    drivers_principales,
    entrenar_logit,
    scorear_afiliados,
)


@pytest.fixture(scope="module")
def modelo(df_afap_module: pd.DataFrame) -> Pipeline:  # noqa: F811
    return entrenar_logit(df_afap_module, seed=42)


@pytest.fixture(scope="module")
def df_afap_module(df_bancario_sintetico: pd.DataFrame) -> pd.DataFrame:
    from src.data_loader import reframe_bancario_a_afap

    return reframe_bancario_a_afap(df_bancario_sintetico)


class TestEntrenarLogit:
    def test_devuelve_pipeline(self, modelo: Pipeline) -> None:
        assert isinstance(modelo, Pipeline)

    def test_tiene_logit(self, modelo: Pipeline) -> None:
        assert "logit" in modelo.named_steps

    def test_falla_si_falta_columna(self, df_afap_module: pd.DataFrame) -> None:
        df = df_afap_module.drop(columns=["edad"])
        with pytest.raises(ValueError, match="edad"):
            entrenar_logit(df)

    def test_determinismo_mismo_seed(self, df_afap_module: pd.DataFrame) -> None:
        m1 = entrenar_logit(df_afap_module, seed=7)
        m2 = entrenar_logit(df_afap_module, seed=7)
        c1 = m1.named_steps["logit"].coef_
        c2 = m2.named_steps["logit"].coef_
        assert (c1 == c2).all()


class TestScoreo:
    def test_columnas(self, modelo: Pipeline, df_afap_module: pd.DataFrame) -> None:
        s = scorear_afiliados(modelo, df_afap_module)
        assert list(s.columns) == ["afiliado_id", "prob_fuga"]

    def test_longitud(self, modelo: Pipeline, df_afap_module: pd.DataFrame) -> None:
        s = scorear_afiliados(modelo, df_afap_module)
        assert len(s) == len(df_afap_module)

    def test_probabilidades_en_rango(self, modelo: Pipeline, df_afap_module: pd.DataFrame) -> None:
        s = scorear_afiliados(modelo, df_afap_module)
        assert ((s["prob_fuga"] >= 0) & (s["prob_fuga"] <= 1)).all()

    def test_ordenado_desc(self, modelo: Pipeline, df_afap_module: pd.DataFrame) -> None:
        s = scorear_afiliados(modelo, df_afap_module)
        assert s["prob_fuga"].is_monotonic_decreasing

    def test_afiliados_son_los_mismos(
        self, modelo: Pipeline, df_afap_module: pd.DataFrame
    ) -> None:
        s = scorear_afiliados(modelo, df_afap_module)
        assert set(s["afiliado_id"]) == set(df_afap_module["afiliado_id"])

    def test_scoring_sin_target(self, modelo: Pipeline, df_afap_module: pd.DataFrame) -> None:
        df = df_afap_module.drop(columns=["traspaso"])
        s = scorear_afiliados(modelo, df)
        assert len(s) == len(df_afap_module)


class TestDrivers:
    def test_columnas(self, modelo: Pipeline) -> None:
        d = drivers_principales(modelo, top_n=5)
        assert {"feature", "coef", "abs_coef", "signo", "narrativa"}.issubset(d.columns)

    def test_top_n_respetado(self, modelo: Pipeline) -> None:
        assert len(drivers_principales(modelo, top_n=3)) == 3

    def test_ordenado_por_abs_coef(self, modelo: Pipeline) -> None:
        d = drivers_principales(modelo, top_n=8)
        assert d["abs_coef"].is_monotonic_decreasing

    def test_signos_validos(self, modelo: Pipeline) -> None:
        d = drivers_principales(modelo)
        assert set(d["signo"].unique()).issubset({"aumenta", "reduce"})

    def test_features_esperadas_cubiertas(self, modelo: Pipeline) -> None:
        d = drivers_principales(modelo, top_n=20)
        # Todos los features numéricos y binarios deberían aparecer al menos.
        esperadas = set(FEATURES_NUMERICAS + FEATURES_BINARIAS)
        assert esperadas.issubset(set(d["feature"])), (
            f"Falta alguna feature numérica/binaria en drivers: {esperadas - set(d['feature'])}"
        )

    def test_al_menos_una_categoria_ohe(self, modelo: Pipeline) -> None:
        d = drivers_principales(modelo, top_n=20)
        # El OneHotEncoder expande 'departamento' a 'departamento_Canelones', etc.
        prefijos = [f"{c}_" for c in FEATURES_CATEGORICAS]
        assert any(
            any(f.startswith(p) for p in prefijos) for f in d["feature"]
        ), "Ninguna categoría OHE aparece en los drivers"
