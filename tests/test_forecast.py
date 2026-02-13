"""Tests for src/forecast.py."""

from __future__ import annotations

import pandas as pd
import pytest

from src.forecast import Scenario, department_baseline, monte_carlo_band, monthly_rate, simulate, total_by_month


def test_monthly_rate_compounds_back_to_annual() -> None:
    m = monthly_rate(0.16)
    assert 1 - (1 - m) ** 12 == pytest.approx(0.16)


def test_full_backfill_zero_growth_is_flat(synthetic_df: pd.DataFrame) -> None:
    base = department_baseline(synthetic_df)
    tot = total_by_month(simulate(base, Scenario(months=12, backfill_rate=1.0)))
    assert tot["headcount"].iloc[-1] == pytest.approx(len(synthetic_df))


def test_growth_target_is_reached_with_full_backfill(synthetic_df: pd.DataFrame) -> None:
    base = department_baseline(synthetic_df)
    tot = total_by_month(simulate(base, Scenario(months=12, annual_growth=0.10, backfill_rate=1.0)))
    assert tot["headcount"].iloc[-1] == pytest.approx(len(synthetic_df) * 1.10, rel=1e-6)


def test_no_backfill_shrinks_headcount(synthetic_df: pd.DataFrame) -> None:
    base = department_baseline(synthetic_df)
    tot = total_by_month(simulate(base, Scenario(months=24, backfill_rate=0.0)))
    assert tot["headcount"].is_monotonic_decreasing


def test_monte_carlo_band_is_ordered_and_brackets_plan(synthetic_df: pd.DataFrame) -> None:
    base = department_baseline(synthetic_df)
    s = Scenario(months=24, annual_growth=0.05)
    band = monte_carlo_band(base, s, n_sims=300)
    plan = total_by_month(simulate(base, s))
    assert (band["p10"] <= band["p50"]).all() and (band["p50"] <= band["p90"]).all()
    assert band["p10"].iloc[-1] < plan["headcount"].iloc[-1] < band["p90"].iloc[-1]
