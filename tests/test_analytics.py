"""Tests para src.analytics. Objetivo: 30+ tests con fixtures sintéticas."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics import (
    LABELS_ANTIGUEDAD,
    LABELS_EDAD,
    LABELS_SALDO,
    aplicar_segmentaciones,
    cohortes_afiliacion,
    concentracion_cartera,
    cross_sell_resumen,
    heatmap_edad_antiguedad,
    kpis_globales,
    oportunidades_cross_sell,
    pareto_cartera,
    resumen_calidad,
    segmentacion_antiguedad,
    segmentacion_edad,
    segmentacion_saldo,
    segmentos_reglas,
    tasa_fuga_por_segmento,
    variacion_entre_segmentos,
)

# =============================================================================
# KPIs globales
# =============================================================================


class TestKpisGlobales:
    def test_claves_esperadas(self, df_afap: pd.DataFrame) -> None:
        k = kpis_globales(df_afap)
        assert set(k.keys()) == {
            "n_afiliados",
            "saldo_total",
            "ticket_promedio",
            "tasa_fuga",
            "pct_activos",
        }

    def test_n_afiliados_coincide(self, df_afap: pd.DataFrame) -> None:
        assert kpis_globales(df_afap)["n_afiliados"] == len(df_afap)

    def test_saldo_total_suma(self, df_afap: pd.DataFrame) -> None:
        assert kpis_globales(df_afap)["saldo_total"] == pytest.approx(
            df_afap["saldo_cuenta"].sum()
        )

    def test_tasa_fuga_en_rango(self, df_afap: pd.DataFrame) -> None:
        tf = kpis_globales(df_afap)["tasa_fuga"]
        assert 0.0 <= tf <= 1.0

    def test_pct_activos_en_rango(self, df_afap: pd.DataFrame) -> None:
        pa = kpis_globales(df_afap)["pct_activos"]
        assert 0.0 <= pa <= 1.0

    def test_ticket_promedio(self, df_afap: pd.DataFrame) -> None:
        k = kpis_globales(df_afap)
        assert k["ticket_promedio"] == pytest.approx(k["saldo_total"] / k["n_afiliados"])

    def test_dataframe_vacio(self) -> None:
        k = kpis_globales(pd.DataFrame(columns=["saldo_cuenta", "traspaso", "aportante_activo"]))
        assert k["n_afiliados"] == 0
        assert k["tasa_fuga"] == 0.0


# =============================================================================
# Pareto de cartera
# =============================================================================


class TestParetoCartera:
    def test_columnas_salida(self, df_afap: pd.DataFrame) -> None:
        p = pareto_cartera(df_afap)
        assert set(p.columns) == {
            "afiliado_id",
            "saldo_cuenta",
            "rank",
            "pct_acumulado",
            "pct_afiliados",
            "top_80",
        }

    def test_ordenado_por_saldo_desc(self, df_afap: pd.DataFrame) -> None:
        p = pareto_cartera(df_afap)
        assert p["saldo_cuenta"].is_monotonic_decreasing

    def test_pct_acumulado_termina_en_uno(self, df_afap: pd.DataFrame) -> None:
        p = pareto_cartera(df_afap)
        assert p["pct_acumulado"].iloc[-1] == pytest.approx(1.0)

    def test_rank_monotono(self, df_afap: pd.DataFrame) -> None:
        p = pareto_cartera(df_afap)
        assert (p["rank"].diff().dropna() == 1).all()

    def test_top_80_bool(self, df_afap: pd.DataFrame) -> None:
        p = pareto_cartera(df_afap)
        assert p["top_80"].dtype == bool

    def test_vacio(self) -> None:
        p = pareto_cartera(pd.DataFrame(columns=["afiliado_id", "saldo_cuenta"]))
        assert len(p) == 0

    def test_concentracion_en_rango(self, df_afap: pd.DataFrame) -> None:
        c = concentracion_cartera(df_afap, 0.2)
        assert 0.2 <= c <= 1.0  # el top 20% siempre concentra al menos su proporción

    def test_concentracion_100_pct_es_uno(self, df_afap: pd.DataFrame) -> None:
        assert concentracion_cartera(df_afap, 1.0) == pytest.approx(1.0)

    def test_concentracion_saldos_cero(self, df_afap: pd.DataFrame) -> None:
        df = df_afap.copy()
        df["saldo_cuenta"] = 0.0
        assert concentracion_cartera(df, 0.2) == 0.0


# =============================================================================
# Segmentaciones
# =============================================================================


class TestSegmentaciones:
    def test_edad_labels(self, df_afap: pd.DataFrame) -> None:
        s = segmentacion_edad(df_afap)
        assert set(s.dropna().unique()).issubset(set(LABELS_EDAD))

    def test_antiguedad_labels(self, df_afap: pd.DataFrame) -> None:
        s = segmentacion_antiguedad(df_afap)
        assert set(s.dropna().unique()).issubset(set(LABELS_ANTIGUEDAD))

    def test_saldo_labels(self, df_afap: pd.DataFrame) -> None:
        s = segmentacion_saldo(df_afap)
        assert set(s.dropna().unique()).issubset(set(LABELS_SALDO))

    def test_edad_30_es_menor_35(self) -> None:
        df = pd.DataFrame({"edad": [30]})
        assert segmentacion_edad(df).iloc[0] == "<35"

    def test_edad_50_es_45_55(self) -> None:
        df = pd.DataFrame({"edad": [50]})
        assert segmentacion_edad(df).iloc[0] == "45-55"

    def test_saldo_cero_sin_saldo(self) -> None:
        df = pd.DataFrame({"saldo_cuenta": [0.0]})
        assert segmentacion_saldo(df).iloc[0] == "Sin saldo"

    def test_saldo_200k_premium(self) -> None:
        df = pd.DataFrame({"saldo_cuenta": [200_000.0]})
        assert segmentacion_saldo(df).iloc[0] == "Premium"

    def test_aplicar_segmentaciones_agrega_columnas(self, df_afap: pd.DataFrame) -> None:
        seg = aplicar_segmentaciones(df_afap)
        assert {"tramo_edad", "tramo_antiguedad", "tramo_saldo"}.issubset(seg.columns)

    def test_aplicar_no_muta(self, df_afap: pd.DataFrame) -> None:
        original = df_afap.copy()
        _ = aplicar_segmentaciones(df_afap)
        pd.testing.assert_frame_equal(df_afap, original)


# =============================================================================
# Tasa de fuga por segmento
# =============================================================================


class TestTasaFugaPorSegmento:
    def test_columnas(self, df_afap: pd.DataFrame) -> None:
        t = tasa_fuga_por_segmento(df_afap, "departamento")
        assert set(t.columns) == {
            "departamento",
            "n_afiliados",
            "n_fugas",
            "tasa_fuga",
            "saldo_total",
        }

    def test_tasas_en_rango(self, df_afap: pd.DataFrame) -> None:
        t = tasa_fuga_por_segmento(df_afap, "departamento")
        assert ((t["tasa_fuga"] >= 0) & (t["tasa_fuga"] <= 1)).all()

    def test_suma_n_afiliados(self, df_afap: pd.DataFrame) -> None:
        t = tasa_fuga_por_segmento(df_afap, "departamento")
        assert int(t["n_afiliados"].sum()) == len(df_afap)

    def test_columna_inexistente(self, df_afap: pd.DataFrame) -> None:
        with pytest.raises(KeyError):
            tasa_fuga_por_segmento(df_afap, "no_existe")

    def test_min_n_filtra(self, df_afap: pd.DataFrame) -> None:
        t = tasa_fuga_por_segmento(df_afap, "afiliado_id", min_n=2)
        assert len(t) == 0  # cada afiliado es único

    def test_ordenado_por_tasa_desc(self, df_afap: pd.DataFrame) -> None:
        t = tasa_fuga_por_segmento(df_afap, "departamento")
        assert t["tasa_fuga"].is_monotonic_decreasing


# =============================================================================
# Cohortes
# =============================================================================


class TestCohortes:
    def test_columnas(self, df_afap: pd.DataFrame) -> None:
        c = cohortes_afiliacion(df_afap)
        assert {"anios_afiliacion", "n_afiliados", "saldo_total", "tasa_fuga", "metrica"}.issubset(
            c.columns
        )

    def test_ordenado_por_anios(self, df_afap: pd.DataFrame) -> None:
        c = cohortes_afiliacion(df_afap)
        assert c["anios_afiliacion"].is_monotonic_increasing

    def test_metrica_alias(self, df_afap: pd.DataFrame) -> None:
        c = cohortes_afiliacion(df_afap, metrica="tasa_fuga")
        assert (c["metrica"] == c["tasa_fuga"]).all()

    def test_metrica_invalida(self, df_afap: pd.DataFrame) -> None:
        with pytest.raises(ValueError):
            cohortes_afiliacion(df_afap, metrica="basura")  # type: ignore[arg-type]


# =============================================================================
# Heatmap edad x antigüedad
# =============================================================================


class TestHeatmap:
    def test_dimensiones(self, df_afap: pd.DataFrame) -> None:
        h = heatmap_edad_antiguedad(df_afap)
        assert list(h.index) == LABELS_EDAD
        assert list(h.columns) == LABELS_ANTIGUEDAD

    def test_valores_en_rango_tasa_fuga(self, df_afap: pd.DataFrame) -> None:
        h = heatmap_edad_antiguedad(df_afap)
        vals = h.to_numpy()
        mask = ~np.isnan(vals)
        assert ((vals[mask] >= 0) & (vals[mask] <= 1)).all()

    def test_n_afiliados_suma(self, df_afap: pd.DataFrame) -> None:
        h = heatmap_edad_antiguedad(df_afap, valor="n_afiliados")
        total = np.nansum(h.to_numpy())
        # puede haber NaN si algún bin está vacío, pero la suma debe ser ~len(df)
        assert int(total) == len(df_afap)

    def test_valor_invalido(self, df_afap: pd.DataFrame) -> None:
        with pytest.raises(ValueError):
            heatmap_edad_antiguedad(df_afap, valor="basura")  # type: ignore[arg-type]


# =============================================================================
# Variación entre segmentos
# =============================================================================


class TestVariacion:
    def test_columnas_delta(self, df_afap: pd.DataFrame) -> None:
        v = variacion_entre_segmentos(df_afap, "departamento")
        assert {"delta_abs", "delta_rel"}.issubset(v.columns)

    def test_baseline_global(self, df_afap: pd.DataFrame) -> None:
        v = variacion_entre_segmentos(df_afap, "departamento")
        # Delta absoluto debe sumar ~0 ponderado por n_afiliados
        suma_ponderada = (v["delta_abs"] * v["n_afiliados"]).sum() / v["n_afiliados"].sum()
        assert suma_ponderada == pytest.approx(0.0, abs=1e-9)

    def test_baseline_explicito(self, df_afap: pd.DataFrame) -> None:
        v = variacion_entre_segmentos(df_afap, "departamento", baseline="Montevideo")
        fila_base = v[v["departamento"] == "Montevideo"]
        assert fila_base["delta_abs"].iloc[0] == pytest.approx(0.0)

    def test_baseline_invalido(self, df_afap: pd.DataFrame) -> None:
        with pytest.raises(ValueError):
            variacion_entre_segmentos(df_afap, "departamento", baseline="Narnia")


# =============================================================================
# Cap. 1 — Calidad de datos
# =============================================================================


class TestResumenCalidad:
    def test_claves(self, df_afap: pd.DataFrame) -> None:
        r = resumen_calidad(df_afap)
        for k in (
            "n_filas",
            "n_columnas",
            "n_duplicados",
            "total_missing",
            "pct_missing",
            "tasa_target",
            "schema_afap_ok",
            "rangos",
        ):
            assert k in r

    def test_schema_ok_con_df_afap(self, df_afap: pd.DataFrame) -> None:
        assert resumen_calidad(df_afap)["schema_afap_ok"] is True

    def test_schema_detecta_faltantes(self, df_afap: pd.DataFrame) -> None:
        df = df_afap.drop(columns=["traspaso"])
        assert resumen_calidad(df)["schema_afap_ok"] is False

    def test_duplicados_cero_en_fixture(self, df_afap: pd.DataFrame) -> None:
        assert resumen_calidad(df_afap)["n_duplicados"] == 0

    def test_dataframe_vacio(self) -> None:
        r = resumen_calidad(pd.DataFrame(columns=["afiliado_id", "traspaso"]))
        assert r["n_filas"] == 0
        assert r["pct_missing"] == 0.0


# =============================================================================
# Cap. 3 — Segmentación por reglas
# =============================================================================


class TestSegmentosReglas:
    def test_columnas(self, df_afap: pd.DataFrame) -> None:
        s = segmentos_reglas(df_afap, min_n=1)
        assert {"tramo_edad", "departamento", "aportante_activo", "tasa_fuga", "lift"}.issubset(
            s.columns
        )

    def test_ordenado_por_tasa_desc(self, df_afap: pd.DataFrame) -> None:
        s = segmentos_reglas(df_afap, min_n=1)
        assert s["tasa_fuga"].is_monotonic_decreasing

    def test_filtrado_por_min_n(self, df_afap: pd.DataFrame) -> None:
        s = segmentos_reglas(df_afap, min_n=1000)
        assert len(s) == 0

    def test_lift_positivo(self, df_afap: pd.DataFrame) -> None:
        s = segmentos_reglas(df_afap, min_n=1)
        assert (s["lift"] >= 0).all()

    def test_vacio(self) -> None:
        s = segmentos_reglas(
            pd.DataFrame(
                columns=["afiliado_id", "departamento", "edad", "anios_afiliacion",
                         "saldo_cuenta", "aportante_activo", "traspaso"]
            )
        )
        assert len(s) == 0


# =============================================================================
# Cap. 4 — Cross-sell
# =============================================================================


class TestCrossSellResumen:
    def test_columnas(self, df_afap: pd.DataFrame) -> None:
        r = cross_sell_resumen(df_afap)
        assert {
            "productos_contratados",
            "n_afiliados",
            "tasa_fuga",
            "saldo_promedio",
            "pct_activos",
        }.issubset(r.columns)

    def test_ordenado_por_productos(self, df_afap: pd.DataFrame) -> None:
        r = cross_sell_resumen(df_afap)
        assert r["productos_contratados"].is_monotonic_increasing

    def test_tasas_en_rango(self, df_afap: pd.DataFrame) -> None:
        r = cross_sell_resumen(df_afap)
        assert ((r["tasa_fuga"] >= 0) & (r["tasa_fuga"] <= 1)).all()


class TestOportunidadesCrossSell:
    def test_columnas(self, df_afap: pd.DataFrame) -> None:
        o = oportunidades_cross_sell(df_afap, saldo_minimo=0)
        assert {"afiliado_id", "apellido", "saldo_cuenta", "anios_afiliacion"}.issubset(
            o.columns
        )

    def test_solo_activos_con_un_producto(self, df_afap: pd.DataFrame) -> None:
        o = oportunidades_cross_sell(df_afap, saldo_minimo=0)
        sub = df_afap[df_afap["afiliado_id"].isin(o["afiliado_id"])]
        assert (sub["aportante_activo"] == 1).all()
        assert (sub["productos_contratados"] == 1).all()
        assert (sub["traspaso"] == 0).all()

    def test_respeta_saldo_minimo(self, df_afap: pd.DataFrame) -> None:
        o = oportunidades_cross_sell(df_afap, saldo_minimo=50_000)
        assert (o["saldo_cuenta"] >= 50_000).all() if len(o) else True

    def test_ordenado_por_saldo_desc(self, df_afap: pd.DataFrame) -> None:
        o = oportunidades_cross_sell(df_afap, saldo_minimo=0)
        if len(o) >= 2:
            assert o["saldo_cuenta"].is_monotonic_decreasing
