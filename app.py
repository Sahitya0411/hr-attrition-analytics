"""HR Attrition & Workforce Analytics: Streamlit entry point.

Run locally with:  streamlit run app.py
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import charts, forecast, insights, report
from src.data_loader import (
    IBM_COLUMNS, Filters, apply_filters, attrition_rate_by, kpis, load_dataset,
    EDUCATION_LABELS, SATISFACTION_LABELS, WLB_LABELS,
)
from src.model import LOGREG, XGBOOST, ModelBundle, build_input_row, predict_with_explanation, risk_band, train_models
from src.sql_engine import (
    TABLE_NAME, SqlQuery, UnsafeQueryError, build_database, memory_connection, parse_queries,
    run_query, run_readonly,
)

st.set_page_config(
    page_title="HR Attrition & Workforce Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Light CSS polish. Colors come from the active theme so dark mode still works.
st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; padding-bottom: 3rem; }
      .insight { border-left: 3px solid #eb6834; padding: 0.35rem 0 0.35rem 0.9rem; margin-bottom: 0.75rem; }
      .insight b { font-size: 1.0rem; }
      .insight span { opacity: 0.75; font-size: 0.9rem; }
      .subtle { opacity: 0.7; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================================== #
# Cached data, SQL and model layers
# =========================================================================== #
@st.cache_data(show_spinner="Loading data...")
def get_data() -> tuple[pd.DataFrame, str, str | None]:
    df, result = load_dataset()
    return df, result.source, result.path


@st.cache_resource(show_spinner=False)
def ensure_database(df: pd.DataFrame) -> str:
    """Build data/hr_attrition.db on first run (full, unfiltered data)."""
    return str(build_database(df))


@st.cache_data
def get_queries() -> dict[str, SqlQuery]:
    return parse_queries()


@st.cache_data(show_spinner=False)
def sql_result(df: pd.DataFrame, sql: str) -> pd.DataFrame:
    """Run a trusted query against an in-memory DB holding the filtered rows."""
    conn = memory_connection(df)
    try:
        return run_query(conn, sql)
    finally:
        conn.close()


@st.cache_resource(show_spinner="Training models (first run only)...")
def get_models(df: pd.DataFrame) -> ModelBundle:
    return train_models(df)


@st.cache_data(show_spinner=False)
def get_insights(df: pd.DataFrame) -> list[insights.Insight]:
    return insights.top_insights(df, k=5)


# Per-run counter so identical charts in different tabs get distinct element keys.
_chart_ids = itertools.count()


def show(fig: go.Figure, caption: str | None = None) -> None:
    """Render a Plotly figure with the Streamlit theme plus an optional caption."""
    st.plotly_chart(fig, theme="streamlit", config={"displaylogo": False}, key=f"chart_{next(_chart_ids)}")
    if caption:
        st.caption(caption)


def section(title: str, subtitle: str | None = None) -> None:
    st.markdown(f"#### {title}")
    if subtitle:
        st.markdown(f"<div class='subtle'>{subtitle}</div>", unsafe_allow_html=True)


# =========================================================================== #
# Sidebar: global filters
# =========================================================================== #
full_df, data_source, data_path = get_data()
ensure_database(full_df)
QUERIES = get_queries()

FILTER_KEYS = ["f_depts", "f_roles", "f_gender", "f_ot", "f_age"]


def reset_filters() -> None:
    for key in FILTER_KEYS:
        st.session_state.pop(key, None)


with st.sidebar:
    st.header("Filters")
    st.caption("Filters apply to every tab. Leave a box empty to include all.")

    depts = st.multiselect("Department", sorted(full_df["Department"].unique()), key="f_depts")
    role_pool = full_df[full_df["Department"].isin(depts)] if depts else full_df
    role_options = sorted(role_pool["JobRole"].unique())
    # Drop roles that are no longer valid after a department change.
    if "f_roles" in st.session_state:
        st.session_state["f_roles"] = [r for r in st.session_state["f_roles"] if r in role_options]
    roles = st.multiselect("Job role", role_options, key="f_roles")

    age_min, age_max = int(full_df["Age"].min()), int(full_df["Age"].max())
    age = st.slider("Age range", age_min, age_max, (age_min, age_max), key="f_age")
    genders = st.multiselect("Gender", sorted(full_df["Gender"].unique()), key="f_gender")
    ot = st.multiselect("Overtime", ["Yes", "No"], key="f_ot")
    st.button("Reset filters", on_click=reset_filters, width="stretch")

    st.divider()
    if data_source == "csv":
        st.caption(f"Data: IBM HR Analytics ({len(full_df):,} employees)")
    else:
        st.warning("CSV not found, so the app is showing **synthetic data** with the same schema. "
                   "Add `data/hr_attrition.csv` to use the real dataset.")

filters = Filters(
    departments=tuple(depts), job_roles=tuple(roles), genders=tuple(genders),
    overtime=tuple(ot), age_range=(int(age[0]), int(age[1])),
)
df = apply_filters(full_df, filters)
company = kpis(full_df)

# =========================================================================== #
# Header
# =========================================================================== #
st.title("HR Attrition & Workforce Analytics")
st.markdown(
    f"<div class='subtle'>Who is leaving, why, and what it means for headcount. "
    f"Showing <b>{len(df):,}</b> of {len(full_df):,} employees.</div>",
    unsafe_allow_html=True,
)

if df.empty:
    st.error("No employees match the current filters. Widen the filters in the sidebar.")
    st.stop()

k = kpis(df)
overall_rate = k["attrition_rate"]
is_filtered = len(df) != len(full_df)

tabs = st.tabs([
    "Overview", "Attrition Drivers", "SQL Explorer", "Risk Model", "Headcount Forecast", "Report Export",
])


# =========================================================================== #
# 1. Overview
# =========================================================================== #
with tabs[0]:
    # KPIs come from SQL (kpi_summary in sql/queries.sql).
    kpi = sql_result(df, QUERIES["kpi_summary"].sql).iloc[0]

    def delta(value: float, base: float, fmt: str) -> str | None:
        """Delta vs company only when filters are active."""
        return (fmt.format(value - base) + " vs company") if is_filtered else None

    c = st.columns(5)
    c[0].metric("Employees", f"{int(kpi.employees):,}", border=True,
                help="Headcount in the current filter.")
    c[1].metric("Attrition rate", f"{kpi.attrition_rate:.1%}",
                delta(kpi.attrition_rate * 100, company["attrition_rate"] * 100, "{:+.1f} pts"),
                delta_color="inverse", border=True, help=f"{int(kpi.leavers):,} leavers")
    c[2].metric("Avg tenure", f"{kpi.avg_tenure:.1f} yrs",
                delta(kpi.avg_tenure, company["avg_tenure"], "{:+.1f} yrs"), border=True)
    c[3].metric("Avg monthly income", f"${kpi.avg_income:,.0f}",
                delta(kpi.avg_income, company["avg_income"], "{:+,.0f}"), border=True)
    c[4].metric("Overtime", f"{kpi.overtime_pct:.1%}",
                delta(kpi.overtime_pct * 100, company["overtime_pct"] * 100, "{:+.1f} pts"),
                delta_color="inverse", border=True, help="Share of employees working overtime")

    st.write("")
    left, right = st.columns([1, 2])
    with left:
        section("Attrition split")
        show(charts.attrition_donut(k["leavers"], k["employees"] - k["leavers"]),
             f"**So what:** roughly 1 in {1 / max(overall_rate, 1e-9):.0f} employees in this view has left.")
    with right:
        section("By department")
        dept_rates = attrition_rate_by(df, "Department").sort_values("attrition_rate", ascending=False)
        show(charts.rate_bar(dept_rates, "Department", overall_rate),
             insights.so_what_rate(dept_rates, "Department", overall_rate))

    cols = st.columns(3)
    for col, (field, label) in zip(cols, [("OverTime", "By overtime"), ("AgeBand", "By age band"),
                                          ("MaritalStatus", "By marital status")]):
        with col:
            section(label)
            rates = attrition_rate_by(df, field)
            show(charts.rate_bar(rates, field, overall_rate, height=300),
                 insights.so_what_rate(rates, field, overall_rate))


# =========================================================================== #
# 2. Attrition drivers
# =========================================================================== #
with tabs[1]:
    section("Top 5 insights", "Computed from the filtered data and ranked by excess leavers: how many more people "
            "left from a group than the average rate would predict.")
    found = get_insights(df)
    if not found:
        st.info("Not enough employees or leavers in this filter to generate reliable insights.")
    for i, ins in enumerate(found, 1):
        st.markdown(f"<div class='insight'><b>{i}. {ins.headline}</b><br><span>{ins.detail}</span></div>",
                    unsafe_allow_html=True)

    st.divider()
    section("Attrition by job role")
    role_rates = attrition_rate_by(df, "JobRole").sort_values("attrition_rate", ascending=False)
    show(charts.rate_bar(role_rates, "JobRole", overall_rate, horizontal=True,
                         height=max(260, 34 * len(role_rates) + 60)),
         insights.so_what_rate(role_rates, "JobRole", overall_rate))

    driver_views = [
        ("OverTime", "Overtime"),
        ("TenureBand", "Tenure band (years at company)"),
        ("SalaryBand", "Salary band (company-wide quartiles)"),
        ("JobSatisfactionLabel", "Job satisfaction"),
        ("WorkLifeBalanceLabel", "Work-life balance"),
        ("PromotionGap", "Years since last promotion"),
        ("BusinessTravel", "Business travel"),
        ("JobLevel", "Job level"),
    ]
    for row in range(0, len(driver_views), 2):
        cols = st.columns(2)
        for col, (field, label) in zip(cols, driver_views[row:row + 2]):
            with col:
                section(label)
                rates = attrition_rate_by(df, field)
                show(charts.rate_bar(rates, field, overall_rate, height=300),
                     insights.so_what_rate(rates, field, overall_rate))

    section("Monthly income: leavers vs stayers")
    med_left = df.loc[df["AttritionFlag"] == 1, "MonthlyIncome"].median()
    med_stay = df.loc[df["AttritionFlag"] == 0, "MonthlyIncome"].median()
    income_caption = (
        f"**So what:** the median leaver earned ${med_left:,.0f} a month against ${med_stay:,.0f} for stayers. "
        "Leavers cluster at the low end of the pay scale."
        if not np.isnan(med_left) and not np.isnan(med_stay) else None
    )
    show(charts.distribution_by_attrition(df, "MonthlyIncome"), income_caption)

    section("Correlation heatmap", "Pearson correlation between numeric features, including attrition "
            "(1 = left) and overtime (1 = yes).")
    numeric = df.select_dtypes("number").drop(columns=["EmployeeNumber", "DailyRate", "HourlyRate", "MonthlyRate"],
                                              errors="ignore")
    numeric = numeric.assign(OverTimeFlag=(df["OverTime"] == "Yes").astype(int))
    numeric = numeric.loc[:, numeric.std() > 0]  # constant columns have no correlation
    corr = numeric.corr()
    show(charts.correlation_heatmap(corr),
         insights.strongest_correlations(corr["AttritionFlag"]) if "AttritionFlag" in corr else None)


# =========================================================================== #
# 3. SQL explorer
# =========================================================================== #
def _safe(fn: Callable[[pd.DataFrame], str]) -> Callable[[pd.DataFrame], str]:
    """So-what generators should never break the page on unusual filters."""
    def wrapper(res: pd.DataFrame) -> str:
        try:
            return fn(res)
        except (IndexError, KeyError, ValueError, ZeroDivisionError):
            return ""
    return wrapper


def _role_frame(res: pd.DataFrame) -> pd.DataFrame:
    out = res.copy()
    abbrev = {"Research & Development": "R&D", "Human Resources": "HR", "Sales": "Sales"}
    out["Role"] = out["JobRole"] + " (" + out["Department"].map(abbrev).fillna(out["Department"]) + ")"
    return out


def _strip_order_prefix(s: pd.Series) -> pd.Series:
    """'1 Below peers' -> 'Below peers'. The prefix only exists for SQL ordering."""
    return s.str.replace(r"^\d+\s", "", regex=True)


SQL_CHARTS: dict[str, Callable[[pd.DataFrame], go.Figure | None]] = {
    "kpi_summary": lambda r: None,
    "dept_attrition": lambda r: charts.rate_bar(r, "Department", overall_rate),
    "role_rank": lambda r: charts.rate_bar(_role_frame(r), "Role", overall_rate, horizontal=True,
                                           height=max(260, 30 * len(r) + 60)),
    "income_quartile_within_dept": lambda r: charts.grouped_bar(r, "pay_quartile", "attrition_rate", "Department"),
    "tenure_cumulative": lambda r: charts.line_with_bars(r, "tenure_band", "attrition_rate",
                                                         "cumulative_share_of_leavers"),
    "tenure_year_lag": lambda r: charts.simple_bar(r, "years_at_company", "attrition_rate", y_format=".0%"),
    "overtime_satisfaction": lambda r: charts.heatmap_matrix(
        r.assign(job_satisfaction=_strip_order_prefix(r["job_satisfaction"]))
        .pivot(index="OverTime", columns="job_satisfaction", values="attrition_rate")
        .reindex(columns=["Low", "Medium", "High", "Very High"]).dropna(axis=1, how="all")),
    "pay_vs_peers": lambda r: charts.rate_bar(r.assign(pay_position=_strip_order_prefix(r["pay_position"])),
                                              "pay_position", overall_rate),
}


@_safe
def _sw_dept(r: pd.DataFrame) -> str:
    top = r.iloc[0]
    return (f"**So what:** {top.Department} has the highest attrition ({top.attrition_rate:.0%}) and accounts for "
            f"{top.share_of_all_leavers:.0%} of all leavers in this view.")


@_safe
def _sw_role(r: pd.DataFrame) -> str:
    top, bottom = r.iloc[0], r.iloc[-1]
    return (f"**So what:** {top.JobRole} ({top.Department}) ranks #1 at {top.attrition_rate:.0%}, against "
            f"{bottom.attrition_rate:.0%} for {bottom.JobRole} ({bottom.Department}). Retention effort should start with the top-ranked roles.")


@_safe
def _sw_quartile(r: pd.DataFrame) -> str:
    p = r.pivot(index="Department", columns="pay_quartile", values="attrition_rate")
    gap = (p["Q1"] - p["Q4"]).sort_values(ascending=False)
    d = gap.index[0]
    return (f"**So what:** in {d}, the lowest-paid quarter leaves at {p.loc[d, 'Q1']:.0%} against "
            f"{p.loc[d, 'Q4']:.0%} for the highest-paid quarter, so pay relative to department peers matters.")


@_safe
def _sw_tenure(r: pd.DataFrame) -> str:
    early = r.loc[r.tenure_band == "2-3 yrs", "cumulative_share_of_leavers"].iat[0]
    return (f"**So what:** {early:.0%} of leavers left within their first 3 years. Onboarding and early-career "
            f"retention give the most leverage.")


@_safe
def _sw_lag(r: pd.DataFrame) -> str:
    drop = r.dropna(subset=["change_vs_prior_year"]).sort_values("change_vs_prior_year").iloc[0]
    return (f"**So what:** the biggest year-over-year drop in attrition comes at year {int(drop.years_at_company)} "
            f"({drop.change_vs_prior_year * 100:+.0f} pts). Getting people past that point is a key milestone.")


@_safe
def _sw_ot(r: pd.DataFrame) -> str:
    yes = r[r.OverTime == "Yes"].attrition_rate
    no = r[r.OverTime == "No"].attrition_rate
    return (f"**So what:** with overtime, attrition ranges from {yes.min():.0%} to {yes.max():.0%} across satisfaction "
            f"levels. Without it, the range is {no.min():.0%} to {no.max():.0%}. Overtime raises attrition at every "
            f"satisfaction level.")


@_safe
def _sw_peers(r: pd.DataFrame) -> str:
    below = r[r.pay_position.str.contains("Below")].iloc[0]
    within = r[r.pay_position.str.contains("Within")].iloc[0]
    return (f"**So what:** employees paid more than 10% below peers at the same level leave at "
            f"{below.attrition_rate:.0%}, against {within.attrition_rate:.0%} for those paid in line with peers.")


SQL_SO_WHAT: dict[str, Callable[[pd.DataFrame], str]] = {
    "dept_attrition": _sw_dept, "role_rank": _sw_role, "income_quartile_within_dept": _sw_quartile,
    "tenure_cumulative": _sw_tenure, "tenure_year_lag": _sw_lag, "overtime_satisfaction": _sw_ot,
    "pay_vs_peers": _sw_peers,
}

with tabs[2]:
    section("SQL Explorer", f"Each query runs live against SQLite. The <code>{TABLE_NAME}</code> table holds the "
            "rows that match the sidebar filters. All queries are in <code>sql/queries.sql</code>.")

    for name, q in QUERIES.items():
        with st.container(border=True):
            st.markdown(f"#### {q.title}")
            st.caption(q.description)
            left, right = st.columns([5, 6], gap="large")
            with left:
                st.code(q.sql, language="sql")
            with right:
                res = sql_result(df, q.sql)
                if res.empty:
                    st.info("No rows for the current filters.")
                    continue
                fig = SQL_CHARTS.get(name, lambda r: None)(res)
                if fig is not None:
                    show(fig)
                st.dataframe(res, hide_index=True, height=min(38 * (len(res) + 1), 320))
                so_what = SQL_SO_WHAT.get(name, lambda r: "")(res)
                if so_what:
                    st.markdown(so_what)

    st.divider()
    section("Run your own query", "Read-only: only a single SELECT (or WITH ... SELECT) runs. Writes, PRAGMA and "
            "ATTACH are blocked by a SQLite authorizer. Results are capped at 1,000 rows and 3 seconds.")
    with st.expander(f"Schema of `{TABLE_NAME}`"):
        schema = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str)})
        st.dataframe(schema[schema["column"].isin(IBM_COLUMNS)], hide_index=True)

    default_sql = (
        "SELECT EducationField,\n"
        "       COUNT(*) AS employees,\n"
        "       ROUND(AVG(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0 END), 3) AS attrition_rate,\n"
        "       RANK() OVER (ORDER BY AVG(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0 END) DESC) AS risk_rank\n"
        "FROM employees\n"
        "GROUP BY EducationField\n"
        "ORDER BY risk_rank;"
    )
    user_sql = st.text_area("SQL", value=default_sql, height=180, label_visibility="collapsed")
    if st.button("Run query", type="primary"):
        conn = memory_connection(df)
        try:
            result, truncated = run_readonly(conn, user_sql)
            st.success(f"{len(result):,} rows returned" + (" (truncated to 1,000)" if truncated else ""))
            st.dataframe(result, hide_index=True)
            # Quick chart: first text column against first numeric column.
            text_cols = result.select_dtypes(exclude="number").columns
            num_cols = result.select_dtypes("number").columns
            if len(text_cols) and len(num_cols) and len(result) <= 50:
                fmt = ".0%" if result[num_cols[-1]].between(0, 1).all() else ",.0f"
                show(charts.simple_bar(result, text_cols[0], num_cols[-1], y_format=fmt),
                     f"Auto-chart: {num_cols[-1]} by {text_cols[0]}.")
        except UnsafeQueryError as exc:
            st.error(f"Query blocked or failed: {exc}")
        except Exception as exc:  # surface SQL syntax errors etc. without crashing the app
            st.error(f"Query failed: {exc}")
        finally:
            conn.close()


# =========================================================================== #
# 4. Risk model
# =========================================================================== #
with tabs[3]:
    bundle = get_models(full_df)
    section("Attrition risk model", f"Trained on all {len(full_df):,} employees, not the filtered view, so the model "
            f"stays stable. 80/20 stratified split ({bundle.n_train:,} train / {bundle.n_test:,} test). Class "
            f"imbalance ({bundle.positive_rate:.0%} leavers) is handled with balanced class weights (logistic "
            "regression) and scale_pos_weight (XGBoost).")

    metrics = bundle.metrics
    st.dataframe(
        metrics.style.format("{:.3f}").highlight_max(axis=0, props="font-weight:700"),
    )
    best_auc = metrics["ROC-AUC"].idxmax()
    best_recall = metrics["Recall"].idxmax()
    recall_txt = (f"It also catches the most actual leavers (recall {metrics.loc[best_recall, 'Recall']:.0%})."
                  if best_recall == best_auc else
                  f"{best_recall} catches the most actual leavers (recall {metrics.loc[best_recall, 'Recall']:.0%}).")
    st.caption(
        f"**So what:** {best_auc} ranks employees by risk best (ROC-AUC {metrics.loc[best_auc, 'ROC-AUC']:.3f}). "
        f"{recall_txt} "
        "For retention outreach, recall usually matters more than accuracy: missing a leaver costs more than "
        "an extra check-in conversation."
    )

    model_name = st.radio("Model to inspect", [LOGREG, XGBOOST], horizontal=True, index=1)
    c1, c2 = st.columns(2)
    with c1:
        section("Confusion matrix (test set, 0.5 threshold)")
        cm = bundle.confusion[model_name]
        show(charts.confusion_matrix_fig(cm),
             f"**So what:** caught {cm[1, 1]} of {cm[1].sum()} actual leavers and flagged {cm[0, 1]} stayers "
             "as false alarms.")
    with c2:
        section("ROC curves")
        show(charts.roc_curves(bundle.roc),
             "The further a curve bows toward the top-left, the better the model separates leavers from stayers.")

    section("Feature importance", "Mean absolute contribution to the log-odds on the test set (exact SHAP values "
            "for XGBoost, coefficient x deviation from average for logistic regression). One-hot columns are "
            "summed back into their original feature.")
    imp = bundle.importance[model_name]
    top3 = ", ".join(imp["feature"].head(3))
    show(charts.importance_bar(imp), f"**So what:** {top3} move predicted risk the most in the {model_name} model.")

    st.divider()
    section("Predict employee risk", "Enter an employee profile. Fields not shown use the company median or most "
            "common value.")
    d = bundle.defaults
    with st.form("predict"):
        f1, f2, f3 = st.columns(3)
        with f1:
            p_age = st.number_input("Age", 18, 60, int(d["Age"]))
            p_ot = st.selectbox("Overtime", ["No", "Yes"])
            p_marital = st.selectbox("Marital status", sorted(full_df["MaritalStatus"].unique()))
            p_travel = st.selectbox("Business travel", sorted(full_df["BusinessTravel"].unique()),
                                    index=sorted(full_df["BusinessTravel"].unique()).index(d["BusinessTravel"]))
            p_dept = st.selectbox("Department", sorted(full_df["Department"].unique()),
                                  index=sorted(full_df["Department"].unique()).index(d["Department"]))
            p_role = st.selectbox("Job role", sorted(full_df["JobRole"].unique()),
                                  index=sorted(full_df["JobRole"].unique()).index(d["JobRole"]))
        with f2:
            p_level = st.slider("Job level", 1, 5, int(d["JobLevel"]))
            p_income = st.number_input("Monthly income ($)", 1000, 20000, int(d["MonthlyIncome"]), step=250)
            p_stock = st.slider("Stock option level", 0, 3, int(d["StockOptionLevel"]))
            p_total = st.number_input("Total working years", 0, 40, int(d["TotalWorkingYears"]))
            p_years = st.number_input("Years at company", 0, 40, int(d["YearsAtCompany"]))
            p_promo = st.number_input("Years since last promotion", 0, 15, int(d["YearsSinceLastPromotion"]))
        with f3:
            p_jobsat = st.select_slider("Job satisfaction", options=[1, 2, 3, 4], value=int(d["JobSatisfaction"]),
                                        format_func=SATISFACTION_LABELS.get)
            p_envsat = st.select_slider("Environment satisfaction", options=[1, 2, 3, 4],
                                        value=int(d["EnvironmentSatisfaction"]), format_func=SATISFACTION_LABELS.get)
            p_wlb = st.select_slider("Work-life balance", options=[1, 2, 3, 4], value=int(d["WorkLifeBalance"]),
                                     format_func=WLB_LABELS.get)
            p_dist = st.number_input("Distance from home (km)", 1, 30, int(d["DistanceFromHome"]))
            p_ncomp = st.number_input("Companies worked at before", 0, 9, int(d["NumCompaniesWorked"]))
            p_edu = st.select_slider("Education", options=[1, 2, 3, 4, 5], value=int(d["Education"]),
                                     format_func=EDUCATION_LABELS.get)
        submitted = st.form_submit_button("Predict risk", type="primary")

    if submitted:
        # Keep tenure fields consistent with each other.
        years = min(p_years, p_total)
        if years != p_years:
            st.warning("Years at company was capped at total working years.")
        st.session_state["prediction_inputs"] = {
            "Age": p_age, "OverTime": p_ot, "MaritalStatus": p_marital, "BusinessTravel": p_travel,
            "Department": p_dept, "JobRole": p_role, "JobLevel": p_level, "MonthlyIncome": p_income,
            "StockOptionLevel": p_stock, "TotalWorkingYears": p_total, "YearsAtCompany": years,
            "YearsSinceLastPromotion": min(p_promo, years), "JobSatisfaction": p_jobsat,
            "EnvironmentSatisfaction": p_envsat, "WorkLifeBalance": p_wlb, "DistanceFromHome": p_dist,
            "NumCompaniesWorked": p_ncomp, "Education": p_edu,
            "YearsInCurrentRole": min(d["YearsInCurrentRole"], years),
            "YearsWithCurrManager": min(d["YearsWithCurrManager"], years),
        }

    if "prediction_inputs" in st.session_state:
        row = build_input_row(bundle, st.session_state["prediction_inputs"])
        probs = {name: predict_with_explanation(bundle, name, row) for name in (LOGREG, XGBOOST)}
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{LOGREG} probability", f"{probs[LOGREG][0]:.0%}", border=True)
        m2.metric(f"{XGBOOST} probability", f"{probs[XGBOOST][0]:.0%}", border=True)
        avg_p = float(np.mean([p for p, _ in probs.values()]))
        m3.metric("Risk band (average)", risk_band(avg_p), border=True,
                  help="Low < 30%, Medium 30-60%, High >= 60%")

        proba, contrib = probs[model_name]
        ups = contrib[contrib["contribution"] > 0]["label"].head(3).tolist()
        downs = contrib[contrib["contribution"] < 0]["label"].head(3).tolist()
        section(f"Top contributing factors ({model_name})")
        show(charts.contribution_bar(contrib),
             ("**Raising risk:** " + (", ".join(ups) or "none") + ". **Lowering risk:** " + (", ".join(downs) or "none")
              + ". Because training reweights the classes, probabilities run higher than the true "
              f"{bundle.positive_rate:.0%} base rate. Treat them as a relative risk score, not a calibrated "
              "likelihood."))


# =========================================================================== #
# 5. Headcount forecast
# =========================================================================== #
with tabs[4]:
    section("Headcount forecast", "Monthly simulation by department based on observed attrition. Adjust the scenario "
            "below. The baseline keeps current attrition and zero growth.")
    s1, s2, s3, s4 = st.columns(4)
    horizon = s1.slider("Horizon (months)", 6, 36, 24, step=6)
    growth = s2.slider("Headcount growth target (% / yr)", -10, 30, 5)
    att_change = s3.slider("Attrition change (%)", -50, 50, -15,
                           help="Relative change, e.g. -20% means attrition falls from 16% to 12.8%.")
    backfill = s4.slider("Backfill rate (% of leavers replaced)", 0, 120, 100, step=5)

    base_tbl = forecast.department_baseline(df)
    base_scn = forecast.Scenario(months=horizon, annual_growth=0.0, attrition_change=0.0, backfill_rate=backfill / 100)
    scn = forecast.Scenario(months=horizon, annual_growth=growth / 100, attrition_change=att_change / 100,
                            backfill_rate=backfill / 100)
    sim_base, sim_scn = forecast.simulate(base_tbl, base_scn), forecast.simulate(base_tbl, scn)
    band = forecast.monte_carlo_band(base_tbl, scn)
    summ = forecast.summary(sim_base, sim_scn)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Starting headcount", f"{summ['start']:,.0f}", border=True)
    k2.metric(f"Baseline in {horizon} mo", f"{summ['baseline_end']:,.0f}",
              f"{summ['baseline_end'] - summ['start']:+,.0f}", border=True)
    k3.metric(f"Scenario in {horizon} mo", f"{summ['scenario_end']:,.0f}",
              f"{summ['scenario_end'] - summ['baseline_end']:+,.0f} vs baseline", border=True)
    k4.metric("Hires needed (scenario)", f"{summ['scenario_hires']:,.0f}",
              f"{summ['scenario_leavers'] - summ['baseline_leavers']:+,.0f} leavers vs baseline",
              delta_color="inverse", border=True)

    show(charts.forecast_fig(forecast.total_by_month(sim_base), forecast.total_by_month(sim_scn), band),
         f"**So what:** to reach {summ['scenario_end']:,.0f} employees in {horizon} months you need about "
         f"{summ['scenario_hires']:,.0f} hires, of which {summ['scenario_leavers'] * backfill / 100:,.0f} only replace "
         f"leavers. Every 1-point cut in attrition reduces that replacement load. The shaded band is the P10-P90 "
         "range from 500 Monte Carlo runs: the hiring plan is fixed and actual departures vary around it.")

    c1, c2 = st.columns([3, 2])
    with c1:
        section("Scenario headcount by department")
        show(charts.dept_forecast_fig(sim_scn))
    with c2:
        section("Department detail")
        end = sim_scn.groupby("Department").agg(end_headcount=("headcount", "last"), hires=("hires", "sum"),
                                                leavers=("leavers", "sum")).reset_index()
        tbl = base_tbl.merge(end, on="Department")
        tbl["annual_attrition"] = tbl["annual_attrition"].map("{:.1%}".format)
        st.dataframe(tbl.round(0), hide_index=True, column_config={
            "start_headcount": "Start", "annual_attrition": "Attrition", "end_headcount": "End",
            "hires": "Hires", "leavers": "Leavers",
        })
    st.caption("Assumptions: the observed attrition rate is treated as annual and converted to a monthly rate "
               "(1 - (1 - a)^(1/12)). Hires = backfill x leavers + growth hires. Small departments make the "
               "uncertainty band wider.")


# =========================================================================== #
# 6. Report export
# =========================================================================== #
@st.cache_data(show_spinner="Building Excel...")
def excel_bytes(data: pd.DataFrame, _k: dict, filt: dict, _ins: list) -> bytes:
    return report.build_excel(data, _k, filt, _ins)


@st.cache_data(show_spinner="Building PDF...")
def pdf_bytes(data: pd.DataFrame, _k: dict, filt: dict, _ins: list, _metrics: pd.DataFrame) -> bytes:
    return report.build_pdf(data, _k, filt, _ins, _metrics)


with tabs[5]:
    section("Report export", "Downloads reflect the current sidebar filters.")
    filt_desc = filters.describe()
    found = get_insights(df)
    st.json(filt_desc, expanded=False)

    e1, e2, e3 = st.columns(3)
    with e1, st.container(border=True):
        st.markdown("**Excel workbook**")
        st.caption("Summary sheet (filters, KPIs, insights), department and job-role breakdowns, and the full "
                   "filtered data.")
        st.download_button("Download .xlsx", excel_bytes(df, k, filt_desc, found),
                           file_name="hr_attrition_report.xlsx", type="primary",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with e2, st.container(border=True):
        st.markdown("**PDF summary**")
        st.caption("A one-to-two page brief: KPIs, top insights, department table and model performance.")
        st.download_button("Download .pdf", pdf_bytes(df, k, filt_desc, found, get_models(full_df).metrics),
                           file_name="hr_attrition_summary.pdf", mime="application/pdf", type="primary")
    with e3, st.container(border=True):
        st.markdown("**HTML summary**")
        st.caption("Same summary with interactive charts. Opens in any browser (loads plotly.js from a CDN).")
        dept_rates = attrition_rate_by(df, "Department").sort_values("attrition_rate", ascending=False)
        role_rates = attrition_rate_by(df, "JobRole").sort_values("attrition_rate", ascending=False)
        figs = [
            ("Attrition split", charts.attrition_donut(k["leavers"], k["employees"] - k["leavers"])),
            ("Attrition by department", charts.rate_bar(dept_rates, "Department", overall_rate)),
            ("Attrition by job role", charts.rate_bar(role_rates, "JobRole", overall_rate, horizontal=True,
                                                      height=max(260, 34 * len(role_rates) + 60))),
            ("Attrition by overtime", charts.rate_bar(attrition_rate_by(df, "OverTime"), "OverTime", overall_rate)),
        ]
        html_doc = report.build_html(k, filt_desc, found, figs, report.department_table(df))
        st.download_button("Download .html", html_doc, file_name="hr_attrition_summary.html", mime="text/html",
                           type="primary")

    st.divider()
    section("Filtered data preview")
    st.dataframe(df.head(200), hide_index=True)
    st.caption(f"Showing the first 200 of {len(df):,} rows. The Excel export contains every row.")
