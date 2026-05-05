-- Top-N afiliados de mayor riesgo de fuga según scoring heurístico.
--
-- Nota importante: el scoring "oficial" del proyecto vive en
-- src/models/churn_logit.py (regresión logística entrenada en Python).
-- Este SQL es un PROXY HEURÍSTICO basado en reglas derivadas de los
-- coeficientes típicos del modelo: más edad, inactivo, Canelones (proxy
-- de Germany del dataset original) y >=3 productos aumentan el riesgo.
-- Sirve para que el equipo pueda explorar en SQL puro sin Python,
-- y para el test de paridad SQL/Python (ver tests/test_sql_parity.py).

WITH scoring AS (
    SELECT
        afiliado_id,
        apellido,
        departamento,
        edad,
        saldo_cuenta,
        aportante_activo,
        productos_contratados,
        -- Heurística logística simplificada (sigmoide sobre combinación lineal).
        1.0 / (1.0 + EXP(
            -(
                -2.0
                + 0.04 * (edad - 40)
                + 0.9 * (departamento = 'Canelones')::INTEGER
                - 1.3 * aportante_activo
                + 0.3 * (productos_contratados >= 3)::INTEGER
            )
        )) AS score_heuristico
    FROM afiliados
)
SELECT *
FROM scoring
ORDER BY score_heuristico DESC
LIMIT 100;
