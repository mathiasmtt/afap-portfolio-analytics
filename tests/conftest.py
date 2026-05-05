"""Fixtures sintéticas para tests. No dependen del CSV real."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_loader import COLUMNAS_ORIGINALES, reframe_bancario_a_afap


def _dataset_bancario_sintetico(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Construye un DataFrame con el mismo schema que el CSV de Kaggle."""
    rng = np.random.default_rng(seed)
    geos = rng.choice(["France", "Germany", "Spain"], size=n, p=[0.5, 0.25, 0.25])
    genders = rng.choice(["Male", "Female"], size=n)
    ages = rng.integers(18, 85, size=n)
    tenures = rng.integers(0, 11, size=n)
    balances = np.round(rng.uniform(0, 250_000, size=n), 2)
    # Introducir ~30% de balances en 0 (como en el dataset real)
    balances[rng.random(n) < 0.3] = 0.0
    n_products = rng.integers(1, 5, size=n)
    has_card = rng.integers(0, 2, size=n)
    is_active = rng.integers(0, 2, size=n)
    salaries = np.round(rng.uniform(10_000, 200_000, size=n), 2)
    credit = rng.integers(350, 851, size=n)

    # Target con señal: mayores, inactivos y alemanes fugan más.
    logit = (
        -2.0
        + 0.04 * (ages - 40)
        + 0.9 * (geos == "Germany").astype(float)
        - 1.3 * is_active
        + 0.3 * (n_products >= 3).astype(float)
    )
    probs = 1 / (1 + np.exp(-logit))
    exited = (rng.random(n) < probs).astype(int)

    df = pd.DataFrame(
        {
            "RowNumber": np.arange(1, n + 1),
            "CustomerId": 15_000_000 + np.arange(n),
            "Surname": [f"Apellido{i:04d}" for i in range(n)],
            "CreditScore": credit,
            "Geography": geos,
            "Gender": genders,
            "Age": ages,
            "Tenure": tenures,
            "Balance": balances,
            "NumOfProducts": n_products,
            "HasCrCard": has_card,
            "IsActiveMember": is_active,
            "EstimatedSalary": salaries,
            "Exited": exited,
        }
    )
    assert set(COLUMNAS_ORIGINALES).issubset(df.columns)
    return df


@pytest.fixture(scope="session")
def df_bancario_sintetico() -> pd.DataFrame:
    """Dataset sintético con schema Kaggle (200 filas)."""
    return _dataset_bancario_sintetico(n=200, seed=42)


@pytest.fixture(scope="session")
def df_afap(df_bancario_sintetico: pd.DataFrame) -> pd.DataFrame:
    """Dataset sintético ya reframeado al vocabulario AFAP."""
    return reframe_bancario_a_afap(df_bancario_sintetico)


@pytest.fixture()
def df_afap_pequenio() -> pd.DataFrame:
    """Dataset mínimo (20 filas) para tests de edge cases."""
    return reframe_bancario_a_afap(_dataset_bancario_sintetico(n=20, seed=7))
