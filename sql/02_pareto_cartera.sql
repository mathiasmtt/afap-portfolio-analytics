-- Pareto de cartera: afiliados ordenados por saldo desc con % acumulado.
-- Equivalente a src.analytics.pareto_cartera.

WITH ordenado AS (
    SELECT
        afiliado_id,
        saldo_cuenta,
        ROW_NUMBER() OVER (ORDER BY saldo_cuenta DESC) AS rank
    FROM afiliados
),
con_total AS (
    SELECT *, SUM(saldo_cuenta) OVER () AS saldo_total_cartera
    FROM ordenado
)
SELECT
    afiliado_id,
    saldo_cuenta,
    rank,
    CASE WHEN saldo_total_cartera > 0
         THEN SUM(saldo_cuenta) OVER (ORDER BY rank) / saldo_total_cartera
         ELSE 0 END                                     AS pct_acumulado,
    rank::DOUBLE / COUNT(*) OVER ()                     AS pct_afiliados
FROM con_total
ORDER BY rank;
