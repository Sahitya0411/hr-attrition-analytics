-- =============================================================================
-- HR Attrition & Workforce Analytics: every SQL query used by the app.
--
-- Table: employees (one row per employee, raw IBM columns).
-- Each query starts with a metadata header that src/sql_engine.py parses:
--   -- name:  unique key the app uses to look the query up
--   -- title: display title
--   -- desc:  one-line description of the technique and the business question
-- Bands are derived in SQL with CASE so the SQL stands on its own.
-- =============================================================================


-- name: kpi_summary
-- title: Headline KPIs
-- desc: Aggregates with conditional averages. Feeds the Overview KPI cards.
SELECT
    COUNT(*)                                                    AS employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)          AS leavers,
    AVG(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0.0 END)      AS attrition_rate,
    AVG(YearsAtCompany)                                         AS avg_tenure,
    AVG(MonthlyIncome)                                          AS avg_income,
    AVG(CASE WHEN OverTime = 'Yes' THEN 1.0 ELSE 0.0 END)       AS overtime_pct
FROM employees;


-- name: dept_attrition
-- title: Attrition by department
-- desc: GROUP BY plus a window SUM() OVER () to show each department's share of all leavers.
SELECT
    Department,
    COUNT(*)                                                    AS employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)          AS leavers,
    ROUND(AVG(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0.0 END), 4) AS attrition_rate,
    ROUND(
        1.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)
        / NULLIF(SUM(SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)) OVER (), 0), 4
    )                                                           AS share_of_all_leavers
FROM employees
GROUP BY Department
ORDER BY attrition_rate DESC;


-- name: role_rank
-- title: Job roles ranked by attrition
-- desc: CTE + RANK() across the company and DENSE_RANK() within each department.
WITH role_stats AS (
    SELECT
        Department,
        JobRole,
        COUNT(*)                                                AS employees,
        SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)      AS leavers,
        AVG(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0.0 END)  AS attrition_rate,
        AVG(MonthlyIncome)                                      AS avg_income
    FROM employees
    GROUP BY Department, JobRole
    HAVING COUNT(*) >= 10          -- ignore tiny groups with unstable rates
)
SELECT
    RANK()       OVER (ORDER BY attrition_rate DESC)                          AS company_rank,
    DENSE_RANK() OVER (PARTITION BY Department ORDER BY attrition_rate DESC)  AS dept_rank,
    Department,
    JobRole,
    employees,
    leavers,
    ROUND(attrition_rate, 4)                                                  AS attrition_rate,
    ROUND(avg_income, 0)                                                      AS avg_income
FROM role_stats
ORDER BY company_rank;


-- name: income_quartile_within_dept
-- title: Attrition by pay quartile within department
-- desc: NTILE(4) partitioned by department, so pay is compared to department peers, not the whole company.
WITH ranked AS (
    SELECT
        Department,
        Attrition,
        MonthlyIncome,
        NTILE(4) OVER (PARTITION BY Department ORDER BY MonthlyIncome) AS pay_quartile
    FROM employees
)
SELECT
    Department,
    'Q' || pay_quartile                                         AS pay_quartile,
    COUNT(*)                                                    AS employees,
    MIN(MonthlyIncome)                                          AS min_income,
    MAX(MonthlyIncome)                                          AS max_income,
    ROUND(AVG(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0.0 END), 4) AS attrition_rate
FROM ranked
GROUP BY Department, pay_quartile
ORDER BY Department, pay_quartile;


-- name: tenure_cumulative
-- title: Where in the tenure curve people leave
-- desc: CASE banding + running SUM() OVER (ORDER BY ...) to get the cumulative share of leavers.
WITH banded AS (
    SELECT
        CASE
            WHEN YearsAtCompany <= 1  THEN 1
            WHEN YearsAtCompany <= 3  THEN 2
            WHEN YearsAtCompany <= 5  THEN 3
            WHEN YearsAtCompany <= 10 THEN 4
            ELSE 5
        END                                                     AS band_order,
        Attrition
    FROM employees
),
agg AS (
    SELECT
        band_order,
        COUNT(*)                                                AS employees,
        SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)      AS leavers
    FROM banded
    GROUP BY band_order
)
SELECT
    CASE band_order
        WHEN 1 THEN '0-1 yrs' WHEN 2 THEN '2-3 yrs' WHEN 3 THEN '4-5 yrs'
        WHEN 4 THEN '6-10 yrs' ELSE '11+ yrs'
    END                                                         AS tenure_band,
    employees,
    leavers,
    ROUND(1.0 * leavers / employees, 4)                         AS attrition_rate,
    ROUND(
        1.0 * SUM(leavers) OVER (ORDER BY band_order ROWS UNBOUNDED PRECEDING)
        / NULLIF(SUM(leavers) OVER (), 0), 4
    )                                                           AS cumulative_share_of_leavers
FROM agg
ORDER BY band_order;


-- name: tenure_year_lag
-- title: Year-over-year change in attrition by years at company
-- desc: LAG() compares each tenure year's attrition rate with the year before.
WITH by_year AS (
    SELECT
        YearsAtCompany                                          AS years_at_company,
        COUNT(*)                                                AS employees,
        AVG(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0.0 END)  AS attrition_rate
    FROM employees
    WHERE YearsAtCompany <= 10
    GROUP BY YearsAtCompany
)
SELECT
    years_at_company,
    employees,
    ROUND(attrition_rate, 4)                                    AS attrition_rate,
    ROUND(attrition_rate - LAG(attrition_rate) OVER (ORDER BY years_at_company), 4)
                                                                AS change_vs_prior_year
FROM by_year
ORDER BY years_at_company;


-- name: overtime_satisfaction
-- title: Overtime x job satisfaction
-- desc: Two-dimensional GROUP BY with CASE labels: does overtime hurt more when satisfaction is low?
SELECT
    OverTime,
    CASE JobSatisfaction
        WHEN 1 THEN '1 Low' WHEN 2 THEN '2 Medium'
        WHEN 3 THEN '3 High' ELSE '4 Very High'
    END                                                         AS job_satisfaction,
    COUNT(*)                                                    AS employees,
    ROUND(AVG(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0.0 END), 4) AS attrition_rate
FROM employees
GROUP BY OverTime, JobSatisfaction
ORDER BY OverTime, JobSatisfaction;


-- name: pay_vs_peers
-- title: Pay relative to peers (same department and job level)
-- desc: AVG() OVER (PARTITION BY ...) builds a peer benchmark for every employee without a self-join.
WITH peer AS (
    SELECT
        Attrition,
        MonthlyIncome,
        AVG(MonthlyIncome) OVER (PARTITION BY Department, JobLevel) AS peer_avg_income
    FROM employees
),
positioned AS (
    SELECT
        Attrition,
        CASE
            WHEN MonthlyIncome < 0.9 * peer_avg_income THEN '1 Below peers (>10%)'
            WHEN MonthlyIncome > 1.1 * peer_avg_income THEN '3 Above peers (>10%)'
            ELSE '2 Within +/-10% of peers'
        END                                                     AS pay_position
    FROM peer
)
SELECT
    pay_position,
    COUNT(*)                                                    AS employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)          AS leavers,
    ROUND(AVG(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0.0 END), 4) AS attrition_rate
FROM positioned
GROUP BY pay_position
ORDER BY pay_position;
