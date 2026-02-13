"""Tests for sql/queries.sql and src/sql_engine.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import kpis
from src.sql_engine import (
    UnsafeQueryError, build_database, memory_connection, parse_queries, run_named, run_readonly, validate_select,
)

EXPECTED_QUERIES = {
    "kpi_summary", "dept_attrition", "role_rank", "income_quartile_within_dept",
    "tenure_cumulative", "tenure_year_lag", "overtime_satisfaction", "pay_vs_peers",
}


@pytest.fixture()
def conn(synthetic_df: pd.DataFrame):
    c = memory_connection(synthetic_df)
    yield c
    c.close()


def test_all_expected_queries_are_parsed() -> None:
    queries = parse_queries()
    assert EXPECTED_QUERIES <= set(queries)
    for q in queries.values():
        assert q.title and q.description
        assert q.sql.lstrip().upper().startswith(("SELECT", "WITH"))


@pytest.mark.parametrize("name", sorted(EXPECTED_QUERIES))
def test_every_query_runs_and_returns_rows(conn, name: str) -> None:
    assert not run_named(conn, name).empty


def test_kpi_summary_matches_pandas(conn, synthetic_df: pd.DataFrame) -> None:
    row = run_named(conn, "kpi_summary").iloc[0]
    k = kpis(synthetic_df)
    assert row["employees"] == k["employees"]
    assert row["leavers"] == k["leavers"]
    assert row["attrition_rate"] == pytest.approx(k["attrition_rate"])
    assert row["avg_income"] == pytest.approx(k["avg_income"])


def test_dept_attrition_matches_pandas_and_shares_sum_to_one(conn, synthetic_df: pd.DataFrame) -> None:
    res = run_named(conn, "dept_attrition").set_index("Department")
    expected = synthetic_df.groupby("Department")["AttritionFlag"].mean()
    for dept, rate in expected.items():
        assert res.loc[dept, "attrition_rate"] == pytest.approx(rate, abs=1e-4)
    assert res["share_of_all_leavers"].sum() == pytest.approx(1.0, abs=1e-3)


def test_role_rank_is_ordered(conn) -> None:
    res = run_named(conn, "role_rank")
    assert res["company_rank"].is_monotonic_increasing
    assert res["attrition_rate"].is_monotonic_decreasing
    assert (res["employees"] >= 10).all()


def test_tenure_cumulative_reaches_one(conn) -> None:
    res = run_named(conn, "tenure_cumulative")
    assert res["cumulative_share_of_leavers"].is_monotonic_increasing
    assert res["cumulative_share_of_leavers"].iloc[-1] == pytest.approx(1.0)


def test_income_quartiles_have_four_buckets_per_department(conn) -> None:
    res = run_named(conn, "income_quartile_within_dept")
    assert (res.groupby("Department")["pay_quartile"].nunique() == 4).all()


def test_on_disk_database_is_built(tmp_path: Path, synthetic_df: pd.DataFrame) -> None:
    db = build_database(synthetic_df, tmp_path / "hr.db")
    assert db.exists() and db.stat().st_size > 0


# --- read-only guard -------------------------------------------------------- #
def test_user_select_runs(conn) -> None:
    res, truncated = run_readonly(conn, "SELECT Department, COUNT(*) AS n FROM employees GROUP BY Department")
    assert not truncated and res["n"].sum() > 0


def test_cte_and_window_allowed(conn) -> None:
    sql = "WITH d AS (SELECT Department, MonthlyIncome FROM employees) " \
          "SELECT Department, RANK() OVER (ORDER BY AVG(MonthlyIncome) DESC) r FROM d GROUP BY Department"
    res, _ = run_readonly(conn, sql)
    assert len(res) > 0


@pytest.mark.parametrize("sql", [
    "DROP TABLE employees",
    "DELETE FROM employees",
    "UPDATE employees SET Age = 1",
    "INSERT INTO employees (Age) VALUES (1)",
    "SELECT 1; DROP TABLE employees",
    "PRAGMA table_info(employees)",
    "ATTACH DATABASE 'x.db' AS x",
    "WITH x AS (SELECT 1) DELETE FROM employees",
    "",
])
def test_writes_and_non_selects_are_blocked(conn, sql: str) -> None:
    with pytest.raises(UnsafeQueryError):
        run_readonly(conn, sql)
    # The table must be untouched afterwards.
    assert run_named(conn, "kpi_summary").iloc[0]["employees"] > 0


def test_row_cap_truncates(conn) -> None:
    sql = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c LIMIT 5000) SELECT x FROM c"
    res, truncated = run_readonly(conn, sql, max_rows=100)
    assert truncated and len(res) == 100


def test_validate_select_strips_comments() -> None:
    assert validate_select("-- a comment\nSELECT 1;").upper().startswith("SELECT")
