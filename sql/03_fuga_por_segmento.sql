-- Tasa de fuga por departamento.
-- Equivalente a src.analytics.tasa_fuga_por_segmento(df, 'departamento').

SELECT
    departamento,
    COUNT(*)                      AS n_afiliados,
    SUM(traspaso)                 AS n_fugas,
    AVG(traspaso::DOUBLE)         AS tasa_fuga,
    SUM(saldo_cuenta)             AS saldo_total
FROM afiliados
GROUP BY departamento
ORDER BY tasa_fuga DESC;
