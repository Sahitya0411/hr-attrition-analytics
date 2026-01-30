"""Monthly headcount simulation by department.

Assumptions (shown in the app):
- The observed attrition rate per department is treated as an annual rate and
  converted to a monthly rate: m = 1 - (1 - a) ** (1 / 12).
- Each month: leavers = headcount * monthly attrition,
  hires = backfill% * leavers + growth hires.
- Growth hires compound so the department reaches its annual growth target if
  attrition is fully backfilled.
- Baseline = current attrition, 0% growth, same backfill rate.
- The uncertainty band (Monte Carlo) keeps the hiring plan fixed at the
  expected-value plan, while actual departures vary: each run draws a
  department's true attrition rate from a Beta posterior (rate uncertainty
  from a finite sample) and then binomial leavers each month.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Scenario:
    """User inputs for one forecast scenario."""

    months: int = 24
    annual_growth: float = 0.0       # e.g. 0.05 = +5% headcount per year
    attrition_change: float = 0.0    # relative, e.g. -0.2 = attrition 20% lower
    backfill_rate: float = 1.0       # share of leavers replaced each month


def monthly_rate(annual: float) -> float:
    """Convert an annual attrition rate to the equivalent monthly rate."""
    annual = float(np.clip(annual, 0.0, 0.999))
    return 1 - (1 - annual) ** (1 / 12)


def department_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Starting headcount and annual attrition rate per department."""
    g = df.groupby("Department")["AttritionFlag"].agg(start_headcount="size", annual_attrition="mean")
    return g.reset_index()


def simulate(
    base: pd.DataFrame, s: Scenario, start: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Deterministic (expected value) monthly simulation per department.

    Returns one row per department-month with headcount, hires and leavers.
    Month 0 is the starting point.
    """
    start = (start or pd.Timestamp.today()).to_period("M").to_timestamp()
    months = pd.date_range(start, periods=s.months + 1, freq="MS")
    growth_m = (1 + s.annual_growth) ** (1 / 12) - 1
    rows = []
    for _, d in base.iterrows():
        m_rate = monthly_rate(d["annual_attrition"] * (1 + s.attrition_change))
        hc = float(d["start_headcount"])
        rows.append((d["Department"], months[0], hc, 0.0, 0.0))
        for t in range(1, s.months + 1):
            leavers = hc * m_rate
            hires = s.backfill_rate * leavers + hc * growth_m
            hc = max(hc - leavers + hires, 0.0)
            rows.append((d["Department"], months[t], hc, hires, leavers))
    return pd.DataFrame(rows, columns=["Department", "month", "headcount", "hires", "leavers"])


def total_by_month(sim: pd.DataFrame) -> pd.DataFrame:
    """Company-wide totals per month."""
    return sim.groupby("month", as_index=False)[["headcount", "hires", "leavers"]].sum()


def monte_carlo_band(
    base: pd.DataFrame, s: Scenario, n_sims: int = 500, seed: int = 7,
    start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """P10 / P50 / P90 of total headcount when actual departures differ from plan.

    Hires follow the deterministic plan from `simulate` (recruiting is planned
    in advance from expected attrition). Leavers are random: each simulation
    draws a true annual rate per department from Beta(leavers + 1, stayers + 1),
    scaled by the scenario's attrition change, then binomial leavers each month.
    """
    rng = np.random.default_rng(seed)
    plan = simulate(base, s, start)
    months = plan["month"].drop_duplicates().sort_values().to_numpy()
    planned_hires = plan.pivot(index="month", columns="Department", values="hires")[base["Department"]].to_numpy()

    start_hc = base["start_headcount"].to_numpy(dtype=float)
    leavers0 = np.round(base["annual_attrition"].to_numpy() * start_hc)
    annual = rng.beta(leavers0 + 1, start_hc - leavers0 + 1, size=(n_sims, len(base)))
    annual = np.clip(annual * (1 + s.attrition_change), 0, 0.999)
    rates = 1 - (1 - annual) ** (1 / 12)

    hc = np.tile(start_hc, (n_sims, 1))  # sims x departments
    totals = np.empty((n_sims, s.months + 1))
    totals[:, 0] = hc.sum(axis=1)
    for t in range(1, s.months + 1):
        leavers = rng.binomial(np.round(hc).astype(int), rates)
        hc = np.maximum(hc - leavers + planned_hires[t], 0)
        totals[:, t] = hc.sum(axis=1)
    q = np.percentile(totals, [10, 50, 90], axis=0)
    return pd.DataFrame({"month": months, "p10": q[0], "p50": q[1], "p90": q[2]})


def summary(baseline: pd.DataFrame, scenario: pd.DataFrame) -> dict[str, float]:
    """End-of-horizon comparison used for the KPI row."""
    b_tot, s_tot = total_by_month(baseline), total_by_month(scenario)
    return {
        "start": float(s_tot["headcount"].iloc[0]),
        "baseline_end": float(b_tot["headcount"].iloc[-1]),
        "scenario_end": float(s_tot["headcount"].iloc[-1]),
        "scenario_hires": float(s_tot["hires"].sum()),
        "scenario_leavers": float(s_tot["leavers"].sum()),
        "baseline_leavers": float(b_tot["leavers"].sum()),
    }
