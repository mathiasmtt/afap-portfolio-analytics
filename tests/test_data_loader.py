"""Tests para src.data_loader."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_loader import (
    COLUMNAS_AFAP,
    DEPARTAMENTO_MAP,
    GENERO_MAP,
    cargar_csv,
    reframe_bancario_a_afap,
    validar_columnas,
)


class TestValidarColumnas:
    def test_schema_completo_no_falla(self, df_bancario_sintetico: pd.DataFrame) -> None:
        validar_columnas(df_bancario_sintetico)

    def test_falta_columna_levanta_value_error(self, df_bancario_sintetico: pd.DataFrame) -> None:
        df = df_bancario_sintetico.drop(columns=["Age"])
        with pytest.raises(ValueError, match="Age"):
            validar_columnas(df)

    def test_mensaje_lista_todas_las_faltantes(self, df_bancario_sintetico: pd.DataFrame) -> None:
        df = df_bancario_sintetico.drop(columns=["Age", "Balance"])
        with pytest.raises(ValueError) as exc:
            validar_columnas(df)
        assert "Age" in str(exc.value) and "Balance" in str(exc.value)


class TestReframe:
    def test_columnas_resultantes(self, df_afap: pd.DataFrame) -> None:
        assert tuple(df_afap.columns) == COLUMNAS_AFAP

    def test_no_muta_el_input(self, df_bancario_sintetico: pd.DataFrame) -> None:
        original = df_bancario_sintetico.copy()
        _ = reframe_bancario_a_afap(df_bancario_sintetico)
        pd.testing.assert_frame_equal(df_bancario_sintetico, original)

    def test_cantidad_de_filas_se_preserva(
        self, df_bancario_sintetico: pd.DataFrame, df_afap: pd.DataFrame
    ) -> None:
        assert len(df_afap) == len(df_bancario_sintetico)

    def test_mapeo_departamento(self, df_afap: pd.DataFrame) -> None:
        departamentos = set(df_afap["departamento"].unique())
        assert departamentos.issubset(set(DEPARTAMENTO_MAP.values()))

    def test_mapeo_genero(self, df_afap: pd.DataFrame) -> None:
        generos = set(df_afap["genero"].unique())
        assert generos.issubset(set(GENERO_MAP.values()))

    def test_geography_desconocida_falla(self, df_bancario_sintetico: pd.DataFrame) -> None:
        df = df_bancario_sintetico.copy()
        df.loc[0, "Geography"] = "Narnia"
        with pytest.raises(ValueError, match="Narnia"):
            reframe_bancario_a_afap(df)

    def test_gender_desconocido_falla(self, df_bancario_sintetico: pd.DataFrame) -> None:
        df = df_bancario_sintetico.copy()
        df.loc[0, "Gender"] = "Unknown"
        with pytest.raises(ValueError, match="Unknown"):
            reframe_bancario_a_afap(df)

    def test_tipos_enteros(self, df_afap: pd.DataFrame) -> None:
        assert df_afap["afiliado_id"].dtype == "int64"
        assert df_afap["edad"].dtype == "int64"
        assert df_afap["traspaso"].dtype == "int8"
        assert df_afap["aportante_activo"].dtype == "int8"

    def test_tipos_flotantes(self, df_afap: pd.DataFrame) -> None:
        assert df_afap["saldo_cuenta"].dtype == "float64"
        assert df_afap["salario_nominal"].dtype == "float64"

    def test_traspaso_binario(self, df_afap: pd.DataFrame) -> None:
        assert set(df_afap["traspaso"].unique()).issubset({0, 1})

    def test_aportante_activo_binario(self, df_afap: pd.DataFrame) -> None:
        assert set(df_afap["aportante_activo"].unique()).issubset({0, 1})

    def test_mapeo_france_a_montevideo(self, df_bancario_sintetico: pd.DataFrame) -> None:
        df = df_bancario_sintetico.copy()
        df["Geography"] = "France"
        out = reframe_bancario_a_afap(df)
        assert (out["departamento"] == "Montevideo").all()

    def test_mapeo_germany_a_canelones(self, df_bancario_sintetico: pd.DataFrame) -> None:
        df = df_bancario_sintetico.copy()
        df["Geography"] = "Germany"
        out = reframe_bancario_a_afap(df)
        assert (out["departamento"] == "Canelones").all()

    def test_mapeo_spain_a_maldonado(self, df_bancario_sintetico: pd.DataFrame) -> None:
        df = df_bancario_sintetico.copy()
        df["Geography"] = "Spain"
        out = reframe_bancario_a_afap(df)
        assert (out["departamento"] == "Maldonado").all()


class TestCargarCsv:
    def test_archivo_inexistente_falla(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="Churn_Modelling"):
            cargar_csv(tmp_path / "no_existe.csv")

    def test_carga_csv_real(self, tmp_path, df_bancario_sintetico: pd.DataFrame) -> None:
        ruta = tmp_path / "Churn_Modelling.csv"
        df_bancario_sintetico.to_csv(ruta, index=False)
        out = cargar_csv(ruta)
        assert tuple(out.columns) == COLUMNAS_AFAP
        assert len(out) == len(df_bancario_sintetico)
