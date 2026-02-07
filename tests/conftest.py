"""Shared fixtures: real data when available, synthetic data always."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_loader import generate_synthetic, load_dataset, prepare


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    return prepare(generate_synthetic(n=600, seed=1))


@pytest.fixture(scope="session")
def app_df() -> pd.DataFrame:
    """Whatever the app would load (real CSV if present, else synthetic)."""
    df, _ = load_dataset()
    return df
