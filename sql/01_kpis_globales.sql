-- KPIs globales de cartera AFAP.
-- Uso: duckdb -c "$(cat sql/01_kpis_globales.sql)"
-- Espera la vista/tabla 'afiliados' con el schema AFAP (ver src/data_loader.py).

SELECT
    COUNT(*)                             AS n_afiliados,
    SUM(saldo_cuenta)                    AS saldo_total,
    AVG(saldo_cuenta)                    AS ticket_promedio,
    AVG(traspaso::DOUBLE)                AS tasa_fuga,
    AVG(aportante_activo::DOUBLE)        AS pct_activos
FROM afiliados;
