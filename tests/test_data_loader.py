"""Tests for src/data_loader.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import (
    IBM_COLUMNS, REQUIRED_COLUMNS, TENURE_LABELS, Filters, apply_filters, attrition_rate_by,
    generate_synthetic, kpis, load_raw, prepare, validate_schema,
)


def test_synthetic_has_full_schema_and_size() -> None:
    df = generate_synthetic(n=500, seed=0)
    assert list(df.columns) == IBM_COLUMNS
    assert len(df) == 500
    assert set(df["Attrition"]) <= {"Yes", "No"}


def test_synthetic_is_reproducible() -> None:
    pd.testing.assert_frame_equal(generate_synthetic(200, seed=3), generate_synthetic(200, seed=3))


def test_synthetic_values_are_in_valid_ranges() -> None:
    df = generate_synthetic(n=1000, seed=5)
    assert df["Age"].between(18, 60).all()
    assert df["JobLevel"].between(1, 5).all()
    assert df["MonthlyIncome"].between(1009, 19999).all()
    assert (df["YearsAtCompany"] <= df["TotalWorkingYears"]).all()
    assert (df["YearsSinceLastPromotion"] <= df["YearsAtCompany"]).all()
    # Attrition should be a realistic minority class.
    assert 0.08 < (df["Attrition"] == "Yes").mean() < 0.30


def test_synthetic_contains_known_driver() -> None:
    df = prepare(generate_synthetic(n=3000, seed=11))
    rates = df.groupby("OverTime")["AttritionFlag"].mean()
    assert rates["Yes"] > rates["No"]


def test_missing_csv_falls_back_to_synthetic(tmp_path: Path) -> None:
    result = load_raw(tmp_path / "does_not_exist.csv", n_synthetic=300)
    assert result.source == "synthetic"
    assert len(result.df) == 300


def test_csv_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "hr_attrition.csv"
    generate_synthetic(120, seed=2).to_csv(path, index=False)
    result = load_raw(path)
    assert result.source == "csv"
    assert len(result.df) == 120


def test_validate_schema_rejects_missing_columns() -> None:
    df = generate_synthetic(50).drop(columns=["OverTime"])
    with pytest.raises(ValueError, match="OverTime"):
        validate_schema(df)


def test_prepare_adds_derived_columns(synthetic_df: pd.DataFrame) -> None:
    for col in ["AttritionFlag", "TenureBand", "AgeBand", "SalaryBand", "PromotionGap"]:
        assert col in synthetic_df.columns
    assert synthetic_df["TenureBand"].notna().all()
    assert list(synthetic_df["TenureBand"].cat.categories) == TENURE_LABELS
    assert set(synthetic_df["AttritionFlag"]) <= {0, 1}
    # Constant columns are dropped.
    assert "EmployeeCount" not in synthetic_df.columns


def test_real_dataset_shape_if_present(app_df: pd.DataFrame) -> None:
    assert set(REQUIRED_COLUMNS) <= set(app_df.columns)
    assert len(app_df) > 0


def test_apply_filters(synthetic_df: pd.DataFrame) -> None:
    f = Filters(departments=("Sales",), overtime=("Yes",), age_range=(25, 40))
    out = apply_filters(synthetic_df, f)
    assert (out["Department"] == "Sales").all()
    assert (out["OverTime"] == "Yes").all()
    assert out["Age"].between(25, 40).all()
    # An empty filter returns everything.
    assert len(apply_filters(synthetic_df, Filters())) == len(synthetic_df)


def test_kpis_match_manual_calculation(synthetic_df: pd.DataFrame) -> None:
    k = kpis(synthetic_df)
    assert k["employees"] == len(synthetic_df)
    assert k["attrition_rate"] == pytest.approx(synthetic_df["AttritionFlag"].mean())
    assert kpis(synthetic_df.iloc[0:0])["employees"] == 0


def test_attrition_rate_by_sums_to_total(synthetic_df: pd.DataFrame) -> None:
    g = attrition_rate_by(synthetic_df, "Department")
    assert g["employees"].sum() == len(synthetic_df)
    assert g["leavers"].sum() == synthetic_df["AttritionFlag"].sum()
