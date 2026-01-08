"""Data loading, synthetic fallback, feature engineering and filtering.

The app expects the IBM HR Analytics Employee Attrition dataset. If no CSV is
found, a synthetic dataset with the same schema is generated so the app (and
the tests) still run end to end.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# The preferred file name first, then the original Kaggle download name.
CSV_CANDIDATES: tuple[Path, ...] = (
    DATA_DIR / "hr_attrition.csv",
    DATA_DIR / "WA_Fn-UseC_-HR-Employee-Attrition.csv",
)

# Full IBM schema, in the original column order.
IBM_COLUMNS: list[str] = [
    "Age", "Attrition", "BusinessTravel", "DailyRate", "Department",
    "DistanceFromHome", "Education", "EducationField", "EmployeeCount",
    "EmployeeNumber", "EnvironmentSatisfaction", "Gender", "HourlyRate",
    "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction", "MaritalStatus",
    "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked", "Over18", "OverTime",
    "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction",
    "StandardHours", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
]

# Columns the app cannot work without.
REQUIRED_COLUMNS: list[str] = [
    "Age", "Attrition", "BusinessTravel", "Department", "DistanceFromHome",
    "Education", "EducationField", "Gender", "JobRole", "JobLevel",
    "JobSatisfaction", "MaritalStatus", "MonthlyIncome", "OverTime",
    "TotalWorkingYears", "YearsAtCompany", "YearsSinceLastPromotion",
    "WorkLifeBalance", "PerformanceRating",
]

# Constant or ID-like columns with no analytical value.
DROP_COLUMNS: list[str] = ["EmployeeCount", "Over18", "StandardHours"]

# Ordinal survey scales used by the IBM dataset.
SATISFACTION_LABELS = {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}
WLB_LABELS = {1: "Bad", 2: "Good", 3: "Better", 4: "Best"}
EDUCATION_LABELS = {1: "Below College", 2: "College", 3: "Bachelor", 4: "Master", 5: "Doctor"}

# Band edges for derived features. Ordered labels keep charts sorted correctly.
TENURE_BINS = [-1, 1, 3, 5, 10, 100]
TENURE_LABELS = ["0-1 yrs", "2-3 yrs", "4-5 yrs", "6-10 yrs", "11+ yrs"]
AGE_BINS = [17, 25, 35, 45, 55, 100]
AGE_LABELS = ["18-25", "26-35", "36-45", "46-55", "56+"]
PROMO_BINS = [-1, 0, 2, 5, 100]
PROMO_LABELS = ["Promoted this yr", "1-2 yrs", "3-5 yrs", "6+ yrs"]
SALARY_LABELS = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]


@dataclass(frozen=True)
class LoadResult:
    """A loaded dataset plus where it came from."""

    df: pd.DataFrame
    source: str  # "csv" or "synthetic"
    path: str | None = None


@dataclass(frozen=True)
class Filters:
    """Global sidebar filters. Empty tuples mean "no filter"."""

    departments: tuple[str, ...] = ()
    job_roles: tuple[str, ...] = ()
    genders: tuple[str, ...] = ()
    overtime: tuple[str, ...] = ()
    age_range: tuple[int, int] = (18, 60)

    def describe(self) -> dict[str, str]:
        """Human-readable summary used in exports."""
        return {
            "Department": ", ".join(self.departments) or "All",
            "Job role": ", ".join(self.job_roles) or "All",
            "Gender": ", ".join(self.genders) or "All",
            "Overtime": ", ".join(self.overtime) or "All",
            "Age range": f"{self.age_range[0]}-{self.age_range[1]}",
        }


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def find_csv(candidates: tuple[Path, ...] = CSV_CANDIDATES) -> Path | None:
    """Return the first CSV that exists, or None."""
    return next((p for p in candidates if p.exists()), None)


def load_raw(path: Path | str | None = None, n_synthetic: int = 1470, seed: int = 42) -> LoadResult:
    """Load the raw dataset from CSV, falling back to synthetic data.

    Args:
        path: Explicit CSV path. If None, the default candidates are searched.
        n_synthetic: Row count for the synthetic fallback.
        seed: Random seed for the synthetic fallback.
    """
    csv_path = Path(path) if path is not None else find_csv()
    if csv_path is not None and csv_path.exists():
        # utf-8-sig strips the BOM that the Kaggle file starts with.
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        validate_schema(df)
        return LoadResult(df=df, source="csv", path=str(csv_path))
    return LoadResult(df=generate_synthetic(n_synthetic, seed), source="synthetic")


def validate_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if any required column is missing."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


# --------------------------------------------------------------------------- #
# Synthetic fallback
# --------------------------------------------------------------------------- #
def _choice(rng: np.random.Generator, options: dict[str, float], n: int) -> np.ndarray:
    keys = list(options)
    probs = np.array(list(options.values()), dtype=float)
    return rng.choice(keys, size=n, p=probs / probs.sum())


def generate_synthetic(n: int = 1470, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic dataset with the full IBM schema.

    Marginal distributions roughly follow the public dataset, and attrition is
    drawn from a logistic model so the known drivers (overtime, low job level,
    short tenure, single, frequent travel, low satisfaction) are present.
    """
    rng = np.random.default_rng(seed)

    department = _choice(rng, {"Research & Development": 0.65, "Sales": 0.30, "Human Resources": 0.05}, n)
    roles_by_dept = {
        "Research & Development": {
            "Research Scientist": 0.30, "Laboratory Technician": 0.27, "Manufacturing Director": 0.15,
            "Healthcare Representative": 0.14, "Research Director": 0.08, "Manager": 0.06,
        },
        "Sales": {"Sales Executive": 0.73, "Sales Representative": 0.19, "Manager": 0.08},
        "Human Resources": {"Human Resources": 0.83, "Manager": 0.17},
    }
    job_role = np.array([_choice(rng, roles_by_dept[d], 1)[0] for d in department])

    # Seniority is driven by role, then age and experience follow seniority.
    base_level = {
        "Research Scientist": 1, "Laboratory Technician": 1, "Sales Representative": 1,
        "Human Resources": 1, "Sales Executive": 2, "Healthcare Representative": 2,
        "Manufacturing Director": 2, "Research Director": 4, "Manager": 4,
    }
    job_level = np.clip(np.array([base_level[r] for r in job_role]) + rng.integers(0, 2, n), 1, 5)

    age = np.clip(rng.normal(27 + job_level * 4.5, 7, n), 18, 60).round().astype(int)
    total_working_years = np.clip(age - 18 - rng.integers(0, 6, n), 0, 40)
    years_at_company = np.clip((total_working_years * rng.beta(2, 2.5, n)).round(), 0, 40).astype(int)
    years_in_role = np.minimum(years_at_company, (years_at_company * rng.uniform(0.3, 1, n)).round()).astype(int)
    years_with_mgr = np.minimum(years_at_company, (years_at_company * rng.uniform(0.3, 1, n)).round()).astype(int)
    years_since_promo = np.minimum(years_at_company, rng.poisson(2.2, n))

    monthly_income = np.clip(
        (job_level * 3600 + total_working_years * 120 + rng.normal(0, 900, n)).round(), 1009, 19999
    ).astype(int)

    overtime = _choice(rng, {"Yes": 0.28, "No": 0.72}, n)
    travel = _choice(rng, {"Travel_Rarely": 0.71, "Travel_Frequently": 0.19, "Non-Travel": 0.10}, n)
    marital = _choice(rng, {"Married": 0.46, "Single": 0.32, "Divorced": 0.22}, n)
    job_sat = rng.integers(1, 5, n)
    env_sat = rng.integers(1, 5, n)
    wlb = _choice(rng, {"1": 0.05, "2": 0.23, "3": 0.61, "4": 0.11}, n).astype(int)
    stock = np.where(marital == "Single", 0, _choice(rng, {"0": 0.25, "1": 0.55, "2": 0.12, "3": 0.08}, n).astype(int))
    distance = np.clip(rng.gamma(1.4, 6.5, n).round(), 1, 29).astype(int)

    # Logistic attrition model; intercept tuned to an overall rate of ~16%.
    logit = (
        -1.85
        + 1.4 * (overtime == "Yes")
        + 0.8 * (travel == "Travel_Frequently")
        + 0.7 * (marital == "Single")
        - 0.35 * (job_level - 1)
        - 0.10 * np.minimum(years_at_company, 10)
        - 0.30 * (job_sat - 2.5)
        - 0.25 * (env_sat - 2.5)
        - 0.30 * (wlb - 2.5)
        - 0.30 * (stock > 0)
        + 0.03 * distance
        - 0.03 * (age - 35)
    )
    attrition = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")

    salary_hike = rng.integers(11, 26, n)
    df = pd.DataFrame({
        "Age": age,
        "Attrition": attrition,
        "BusinessTravel": travel,
        "DailyRate": rng.integers(102, 1500, n),
        "Department": department,
        "DistanceFromHome": distance,
        "Education": _choice(rng, {"1": 0.12, "2": 0.19, "3": 0.39, "4": 0.27, "5": 0.03}, n).astype(int),
        "EducationField": _choice(rng, {
            "Life Sciences": 0.41, "Medical": 0.32, "Marketing": 0.11,
            "Technical Degree": 0.09, "Other": 0.05, "Human Resources": 0.02,
        }, n),
        "EmployeeCount": 1,
        "EmployeeNumber": np.arange(1, n + 1),
        "EnvironmentSatisfaction": env_sat,
        "Gender": _choice(rng, {"Male": 0.6, "Female": 0.4}, n),
        "HourlyRate": rng.integers(30, 101, n),
        "JobInvolvement": _choice(rng, {"1": 0.06, "2": 0.26, "3": 0.59, "4": 0.09}, n).astype(int),
        "JobLevel": job_level,
        "JobRole": job_role,
        "JobSatisfaction": job_sat,
        "MaritalStatus": marital,
        "MonthlyIncome": monthly_income,
        "MonthlyRate": rng.integers(2094, 27000, n),
        "NumCompaniesWorked": np.minimum(rng.poisson(2.7, n), 9),
        "Over18": "Y",
        "OverTime": overtime,
        "PercentSalaryHike": salary_hike,
        # In the IBM data a rating of 4 corresponds to hikes of 20%+.
        "PerformanceRating": np.where(salary_hike >= 20, 4, 3),
        "RelationshipSatisfaction": rng.integers(1, 5, n),
        "StandardHours": 80,
        "StockOptionLevel": stock,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": np.clip(rng.poisson(2.8, n), 0, 6),
        "WorkLifeBalance": wlb,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_role,
        "YearsSinceLastPromotion": years_since_promo,
        "YearsWithCurrManager": years_with_mgr,
    })
    return df[IBM_COLUMNS]


# --------------------------------------------------------------------------- #
# Cleaning and feature engineering
# --------------------------------------------------------------------------- #
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw data and add the derived columns used across the app."""
    out = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns]).copy()
    out = out.drop_duplicates()
    out["Attrition"] = out["Attrition"].astype(str).str.strip().str.title()
    out["OverTime"] = out["OverTime"].astype(str).str.strip().str.title()
    out["AttritionFlag"] = (out["Attrition"] == "Yes").astype(int)

    out["TenureBand"] = pd.Categorical(
        pd.cut(out["YearsAtCompany"], TENURE_BINS, labels=TENURE_LABELS).astype(str),
        categories=TENURE_LABELS, ordered=True,
    )
    out["AgeBand"] = pd.Categorical(
        pd.cut(out["Age"], AGE_BINS, labels=AGE_LABELS).astype(str),
        categories=AGE_LABELS, ordered=True,
    )
    out["PromotionGap"] = pd.Categorical(
        pd.cut(out["YearsSinceLastPromotion"], PROMO_BINS, labels=PROMO_LABELS).astype(str),
        categories=PROMO_LABELS, ordered=True,
    )
    # Salary quartiles are computed on the full population so bands don't
    # shift when filters are applied.
    out["SalaryBand"] = pd.qcut(out["MonthlyIncome"], 4, labels=SALARY_LABELS)

    out["JobSatisfactionLabel"] = pd.Categorical(
        out["JobSatisfaction"].map(SATISFACTION_LABELS), categories=list(SATISFACTION_LABELS.values()), ordered=True
    )
    out["WorkLifeBalanceLabel"] = pd.Categorical(
        out["WorkLifeBalance"].map(WLB_LABELS), categories=list(WLB_LABELS.values()), ordered=True
    )
    if "EmployeeNumber" not in out.columns:
        out["EmployeeNumber"] = np.arange(1, len(out) + 1)
    return out.reset_index(drop=True)


def load_dataset(path: Path | str | None = None) -> tuple[pd.DataFrame, LoadResult]:
    """Load and prepare in one call. Returns (prepared_df, load_result)."""
    result = load_raw(path)
    return prepare(result.df), result


def apply_filters(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    """Apply the global sidebar filters to a prepared DataFrame."""
    mask = df["Age"].between(f.age_range[0], f.age_range[1])
    if f.departments:
        mask &= df["Department"].isin(f.departments)
    if f.job_roles:
        mask &= df["JobRole"].isin(f.job_roles)
    if f.genders:
        mask &= df["Gender"].isin(f.genders)
    if f.overtime:
        mask &= df["OverTime"].isin(f.overtime)
    return df.loc[mask].reset_index(drop=True)


def kpis(df: pd.DataFrame) -> dict[str, float]:
    """Headline workforce KPIs for a (filtered) DataFrame."""
    n = len(df)
    if n == 0:
        return {"employees": 0, "leavers": 0, "attrition_rate": 0.0, "avg_tenure": 0.0,
                "avg_income": 0.0, "overtime_pct": 0.0}
    return {
        "employees": n,
        "leavers": int(df["AttritionFlag"].sum()),
        "attrition_rate": float(df["AttritionFlag"].mean()),
        "avg_tenure": float(df["YearsAtCompany"].mean()),
        "avg_income": float(df["MonthlyIncome"].mean()),
        "overtime_pct": float((df["OverTime"] == "Yes").mean()),
    }


def attrition_rate_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Headcount, leavers and attrition rate for each value of `col`."""
    g = (
        df.groupby(col, observed=True)["AttritionFlag"]
        .agg(employees="size", leavers="sum")
        .reset_index()
    )
    g["attrition_rate"] = g["leavers"] / g["employees"]
    return g
