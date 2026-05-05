"""Paridad SQL/Python: cada .sql debe producir los mismos números que analytics.py."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src.analytics import kpis_globales, pareto_cartera, tasa_fuga_por_segmento

SQL_DIR = Path(__file__).parent.parent / "sql"


@pytest.fixture(scope="module")
def con(df_bancario_sintetico: pd.DataFrame) -> duckdb.DuckDBPyConnection:
    from src.data_loader import reframe_bancario_a_afap

    df_afap = reframe_bancario_a_afap(df_bancario_sintetico)
    conn = duckdb.connect(":memory:")
    conn.register("afiliados", df_afap)
    return conn


def _sql(nombre: str) -> str:
    return (SQL_DIR / nombre).read_text()


class TestKpisGlobales:
    def test_paridad(
        self, con: duckdb.DuckDBPyConnection, df_bancario_sintetico: pd.DataFrame
    ) -> None:
        from src.data_loader import reframe_bancario_a_afap

        df_afap = reframe_bancario_a_afap(df_bancario_sintetico)
        sql_row = con.execute(_sql("01_kpis_globales.sql")).fetchone()
        assert sql_row is not None
        n, saldo, ticket, tasa, pct_act = sql_row
        py = kpis_globales(df_afap)
        assert int(n) == py["n_afiliados"]
        assert float(saldo) == pytest.approx(py["saldo_total"])
        assert float(ticket) == pytest.approx(py["ticket_promedio"])
        assert float(tasa) == pytest.approx(py["tasa_fuga"])
        assert float(pct_act) == pytest.approx(py["pct_activos"])


class TestPareto:
    def test_paridad_primeros_10(
        self, con: duckdb.DuckDBPyConnection, df_bancario_sintetico: pd.DataFrame
    ) -> None:
        from src.data_loader import reframe_bancario_a_afap

        df_afap = reframe_bancario_a_afap(df_bancario_sintetico)
        sql_df = con.execute(_sql("02_pareto_cartera.sql")).fetchdf()
        py_df = pareto_cartera(df_afap)
        # Pueden diferir en filas con saldos iguales; comparar por saldo ordenado.
        pd.testing.assert_series_equal(
            sql_df["saldo_cuenta"].head(10).reset_index(drop=True),
            py_df["saldo_cuenta"].head(10).reset_index(drop=True),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            sql_df["pct_acumulado"].head(10).reset_index(drop=True).astype(float),
            py_df["pct_acumulado"].head(10).reset_index(drop=True).astype(float),
            check_names=False,
            atol=1e-9,
        )


class TestFugaPorSegmento:
    def test_paridad_departamento(
        self, con: duckdb.DuckDBPyConnection, df_bancario_sintetico: pd.DataFrame
    ) -> None:
        from src.data_loader import reframe_bancario_a_afap

        df_afap = reframe_bancario_a_afap(df_bancario_sintetico)
        sql_df = (
            con.execute(_sql("03_fuga_por_segmento.sql"))
            .fetchdf()
            .sort_values("departamento")
            .reset_index(drop=True)
        )
        py_df = (
            tasa_fuga_por_segmento(df_afap, "departamento")
            .sort_values("departamento")
            .reset_index(drop=True)
        )
        assert (sql_df["n_afiliados"].values == py_df["n_afiliados"].values).all()
        assert (
            sql_df["tasa_fuga"].round(8).values == py_df["tasa_fuga"].round(8).values
        ).all()


class TestTopRiesgo:
    def test_devuelve_filas(
        self, con: duckdb.DuckDBPyConnection, df_bancario_sintetico: pd.DataFrame
    ) -> None:
        df = con.execute(_sql("04_top_riesgo_afiliados.sql")).fetchdf()
        assert len(df) > 0
        assert ((df["score_heuristico"] >= 0) & (df["score_heuristico"] <= 1)).all()
        assert df["score_heuristico"].is_monotonic_decreasing
