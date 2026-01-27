"""Plain-English insights computed from the (filtered) data.

Nothing here is hardcoded: every number and every segment in a sentence comes
from the DataFrame passed in, so the insights change with the sidebar filters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data_loader import attrition_rate_by

# Dimensions screened for high-risk segments, with readable descriptions.
# The template receives the segment value as {v}.
SEGMENT_DIMENSIONS: dict[str, str] = {
    "OverTime": "employees {ot}",
    "BusinessTravel": "employees who travel {v}",
    "MaritalStatus": "{v} employees",
    "JobRole": "{v}s",
    "Department": "employees in {v}",
    "TenureBand": "employees with {v} of tenure",
    "SalaryBand": "employees in pay quartile {v}",
    "AgeBand": "employees aged {v}",
    "JobSatisfactionLabel": "employees reporting {v} job satisfaction",
    "WorkLifeBalanceLabel": "employees rating work-life balance as '{v}'",
    "PromotionGap": "employees whose last promotion was {v} ago",
    "JobLevel": "job level {v} employees",
    "StockOptionLevel": "employees with stock option level {v}",
}

TRAVEL_TEXT = {"Travel_Frequently": "frequently", "Travel_Rarely": "rarely", "Non-Travel": "never"}


@dataclass(frozen=True)
class Insight:
    """One ranked insight."""

    headline: str
    detail: str
    dimension: str
    excess_leavers: float


def _describe_segment(dim: str, value: object) -> str:
    # iterrows() upcasts ints to floats in mixed rows; print 1.0 as 1.
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    v = str(value)
    if dim == "OverTime":
        return "employees working overtime" if v == "Yes" else "employees not working overtime"
    if dim == "BusinessTravel":
        v = TRAVEL_TEXT.get(v, v)
    if dim == "PromotionGap" and v == "Promoted this yr":
        return "employees promoted this year"
    if dim == "JobRole" and v.endswith("Human Resources"):
        return "HR staff"
    if dim == "JobRole" and v == "Manager":
        return "managers"
    if dim == "MaritalStatus":
        v = v.lower()
    return SEGMENT_DIMENSIONS[dim].format(v=v, ot=v)


def segment_table(df: pd.DataFrame, min_group: int | None = None) -> pd.DataFrame:
    """Every (dimension, value) segment with its rate, lift and excess leavers.

    Excess leavers = leavers above what the company-average rate would
    predict for a group that size. It ranks segments by business impact
    rather than by rate alone, so a tiny group with a high rate does not win.
    """
    n = len(df)
    overall = df["AttritionFlag"].mean() if n else 0.0
    min_group = min_group or max(20, int(0.03 * n))
    rows = []
    for dim in SEGMENT_DIMENSIONS:
        if dim not in df.columns or df[dim].nunique() < 2:
            continue  # filtered down to a single value: nothing to compare
        g = attrition_rate_by(df, dim)
        g = g[g["employees"] >= min_group]
        for _, r in g.iterrows():
            rows.append({
                "dimension": dim,
                "segment": r[dim],
                "employees": int(r["employees"]),
                "leavers": int(r["leavers"]),
                "attrition_rate": r["attrition_rate"],
                "lift": r["attrition_rate"] / overall if overall else np.nan,
                "excess_leavers": r["leavers"] - overall * r["employees"],
            })
    return pd.DataFrame(rows)


def top_insights(df: pd.DataFrame, k: int = 5) -> list[Insight]:
    """Top-k insights: one high-risk segment per dimension, plus a pay comparison."""
    n = len(df)
    if n < 30 or df["AttritionFlag"].sum() < 5:
        return []
    overall = df["AttritionFlag"].mean()
    seg = segment_table(df)
    insights: list[Insight] = []

    if not seg.empty:
        # Best segment per dimension, then the dimensions with the most excess leavers.
        best = (
            seg[seg["lift"] > 1.15]
            .sort_values("excess_leavers", ascending=False)
            .drop_duplicates("dimension")
        )
        for _, r in best.head(k).iterrows():
            who = _describe_segment(r["dimension"], r["segment"])
            insights.append(Insight(
                headline=f"{who[0].upper()}{who[1:]} leave at {r['attrition_rate']:.0%}, "
                         f"{r['lift']:.1f}x the {overall:.0%} average.",
                detail=f"{r['leavers']:,} of {r['employees']:,} left, about {r['excess_leavers']:.0f} more "
                       f"than the average rate would predict.",
                dimension=r["dimension"],
                excess_leavers=float(r["excess_leavers"]),
            ))

    pay = _pay_gap_insight(df)
    if pay is not None:
        # The pay comparison replaces the weakest segment insight if needed.
        insights = insights[: k - 1] + [pay] if len(insights) >= k else insights + [pay]
    return insights[:k]


def _pay_gap_insight(df: pd.DataFrame) -> Insight | None:
    """Median pay of leavers vs stayers at the same job level."""
    left = df[df["AttritionFlag"] == 1]
    stayed = df[df["AttritionFlag"] == 0]
    if len(left) < 5 or len(stayed) < 5:
        return None
    # Compare within job level so seniority mix does not drive the gap.
    med = df.groupby(["JobLevel", "Attrition"], observed=True)["MonthlyIncome"].median().unstack()
    if not {"Yes", "No"} <= set(med.columns):
        return None
    weights = left["JobLevel"].value_counts()
    med = med.dropna().loc[lambda m: m.index.isin(weights.index)]
    if med.empty:
        return None
    w = weights.reindex(med.index)
    gap = float(np.average(med["Yes"] / med["No"] - 1, weights=w))
    overall_gap = left["MonthlyIncome"].median() / stayed["MonthlyIncome"].median() - 1
    direction = "less" if gap < 0 else "more"
    return Insight(
        headline=f"Leavers earned {abs(gap):.0%} {direction} than stayers at the same job level.",
        detail=f"Without the job-level adjustment the median pay gap is {overall_gap:+.0%}, "
               f"so part of the raw gap reflects seniority rather than pay at the same level.",
        dimension="MonthlyIncome",
        excess_leavers=0.0,
    )


# Short, readable names for segment values in chart captions.
_SHORT_LABELS: dict[str, dict[str, str] | str] = {
    "OverTime": {"Yes": "Overtime workers", "No": "Non-overtime staff"},
    "BusinessTravel": {"Travel_Frequently": "Frequent travellers", "Travel_Rarely": "Occasional travellers",
                       "Non-Travel": "Non-travellers"},
    "JobLevel": "Job level {v}",
    "TenureBand": "Tenure {v}",
    "AgeBand": "Ages {v}",
    "SalaryBand": "Pay {v}",
    "JobSatisfactionLabel": "{v} job satisfaction",
    "WorkLifeBalanceLabel": "'{v}' work-life balance",
    "PromotionGap": {"Promoted this yr": "Promoted this year", "1-2 yrs": "Last promoted 1-2 yrs ago",
                     "3-5 yrs": "Last promoted 3-5 yrs ago", "6+ yrs": "Last promoted 6+ yrs ago"},
    "StockOptionLevel": "Stock level {v}",
}


def segment_label(dim: str, value: object) -> str:
    """Readable label for a segment value, e.g. ('OverTime', 'Yes') -> 'Overtime workers'."""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    rule = _SHORT_LABELS.get(dim)
    if isinstance(rule, dict):
        return rule.get(str(value), str(value))
    if isinstance(rule, str):
        return rule.format(v=value)
    return str(value)


def so_what_rate(df_rates: pd.DataFrame, category: str, overall: float) -> str:
    """One-line takeaway for a rate-by-category chart."""
    if df_rates.empty:
        return "Not enough data for this view with the current filters."
    hi = df_rates.loc[df_rates["attrition_rate"].idxmax()]
    lo = df_rates.loc[df_rates["attrition_rate"].idxmin()]
    if hi[category] == lo[category]:
        return f"Only one group in view, at {hi['attrition_rate']:.0%} attrition."
    hi_lbl, lo_lbl = segment_label(category, hi[category]), segment_label(category, lo[category])
    ratio = hi["attrition_rate"] / lo["attrition_rate"] if lo["attrition_rate"] > 0 else np.inf
    comparison = (f"{ratio:.1f}x the lowest group, {lo_lbl} ({lo['attrition_rate']:.0%})" if np.isfinite(ratio)
                  else f"against {lo['attrition_rate']:.0%} for {lo_lbl}")
    return (
        f"**So what:** attrition peaks in {hi_lbl} at {hi['attrition_rate']:.0%}, "
        f"{comparison}. The overall rate is {overall:.0%}."
    )


def strongest_correlations(corr: pd.Series, k: int = 3) -> str:
    """Describe the k features most correlated with attrition."""
    s = corr.drop(labels=["AttritionFlag"], errors="ignore").dropna()
    if s.empty:
        return "Not enough variation to compute correlations."
    top = s.reindex(s.abs().sort_values(ascending=False).index).head(k)
    parts = [f"{name} ({val:+.2f})" for name, val in top.items()]
    return (
        "**So what:** the features most linearly related to attrition are " + ", ".join(parts) +
        ". Positive means more of the feature goes with more attrition. All values are modest, so no "
        "single factor explains attrition alone."
    )
