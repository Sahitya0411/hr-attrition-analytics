"""Report exports: Excel workbook, PDF summary and HTML summary."""

from __future__ import annotations

import html
import io
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from fpdf import FPDF

from src.data_loader import attrition_rate_by
from src.insights import Insight

# Derived columns exported as plain text instead of pandas categoricals.
_CATEGORICAL_EXPORT = ["TenureBand", "AgeBand", "PromotionGap", "SalaryBand",
                       "JobSatisfactionLabel", "WorkLifeBalanceLabel"]


def _kpi_rows(k: dict[str, float]) -> list[tuple[str, str]]:
    return [
        ("Employees", f"{k['employees']:,}"),
        ("Leavers", f"{k['leavers']:,}"),
        ("Attrition rate", f"{k['attrition_rate']:.1%}"),
        ("Average tenure (years)", f"{k['avg_tenure']:.1f}"),
        ("Average monthly income", f"${k['avg_income']:,.0f}"),
        ("Overtime share", f"{k['overtime_pct']:.1%}"),
    ]


def department_table(df: pd.DataFrame) -> pd.DataFrame:
    """Department breakdown used in every export."""
    t = attrition_rate_by(df, "Department").sort_values("attrition_rate", ascending=False)
    t["avg_income"] = df.groupby("Department")["MonthlyIncome"].mean().reindex(t["Department"]).to_numpy()
    return t.rename(columns={
        "employees": "Employees", "leavers": "Leavers",
        "attrition_rate": "Attrition rate", "avg_income": "Avg monthly income",
    })


# --------------------------------------------------------------------------- #
# Excel
# --------------------------------------------------------------------------- #
def build_excel(
    df: pd.DataFrame, k: dict[str, float], filters: dict[str, str], insights: list[Insight],
) -> bytes:
    """Workbook with Summary, By Department, By Job Role and Data sheets."""
    buf = io.BytesIO()
    data = df.copy()
    for c in _CATEGORICAL_EXPORT:
        if c in data:
            data[c] = data[c].astype(str)

    summary_rows = (
        [("Report generated", datetime.now().strftime("%Y-%m-%d %H:%M")), ("", "")]
        + [("FILTERS", "")] + list(filters.items()) + [("", "")]
        + [("KPIs", "")] + _kpi_rows(k) + [("", "")]
        + [("KEY INSIGHTS", "")] + [(f"{i + 1}. {ins.headline}", ins.detail) for i, ins in enumerate(insights)]
    )
    summary = pd.DataFrame(summary_rows, columns=["Item", "Value"])
    by_role = attrition_rate_by(df, "JobRole").sort_values("attrition_rate", ascending=False)

    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="Summary", index=False)
        department_table(df).to_excel(xw, sheet_name="By Department", index=False)
        by_role.to_excel(xw, sheet_name="By Job Role", index=False)
        data.to_excel(xw, sheet_name="Data", index=False)

        # Light formatting: column widths, percent formats, bold headers.
        for name, ws in xw.sheets.items():
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 70)
            ws.freeze_panes = "A2"
            for cell in ws[1]:
                cell.font = cell.font.copy(bold=True)
            if name in {"By Department", "By Job Role"}:
                for row in ws.iter_rows(min_row=2):
                    for cell in row:
                        header = ws.cell(row=1, column=cell.column).value
                        if header in {"Attrition rate", "attrition_rate"}:
                            cell.number_format = "0.0%"
                        elif header == "Avg monthly income":
                            cell.number_format = "#,##0"
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
def _latin1(text: str) -> str:
    """fpdf2 core fonts are latin-1 only. Swap common unicode punctuation."""
    for a, b in {"–": "-", "—": "-", "’": "'", "‘": "'", "“": '"', "”": '"', "×": "x", "…": "..."}.items():
        text = text.replace(a, b)
    return text.encode("latin-1", "replace").decode("latin-1")


class _Pdf(FPDF):
    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", size=8)
        self.set_text_color(130)
        self.cell(0, 8, f"HR Attrition & Workforce Analytics  |  page {self.page_no()}", align="C")


def build_pdf(
    df: pd.DataFrame, k: dict[str, float], filters: dict[str, str], insights: list[Insight],
    model_metrics: pd.DataFrame | None = None,
) -> bytes:
    """Short text-and-table PDF summary."""
    pdf = _Pdf(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    w = pdf.w - pdf.l_margin - pdf.r_margin

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(w, 10, "HR Attrition & Workforce Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=9)
    pdf.set_text_color(110)
    pdf.cell(w, 6, f"Generated {datetime.now():%d %b %Y, %H:%M}", new_x="LMARGIN", new_y="NEXT")
    pdf.multi_cell(w, 5, _latin1("Filters: " + "; ".join(f"{a}: {b}" for a, b in filters.items())),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0)
    pdf.ln(3)

    def heading(text: str) -> None:
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(w, 8, text, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=10)

    def table(headers: list[str], rows: list[list[str]], widths: list[float]) -> None:
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(235, 238, 244)
        for h, cw in zip(headers, widths):
            pdf.cell(cw, 7, _latin1(h), border="B", fill=True)
        pdf.ln()
        pdf.set_font("Helvetica", size=9)
        for r in rows:
            for v, cw in zip(r, widths):
                pdf.cell(cw, 6, _latin1(v), border="B")
            pdf.ln()

    heading("Key metrics")
    table(["Metric", "Value"], [[a, b] for a, b in _kpi_rows(k)], [w * 0.6, w * 0.4])

    heading("Top insights")
    if not insights:
        pdf.multi_cell(w, 5, "Not enough data in the current filter to generate insights.")
    for i, ins in enumerate(insights, 1):
        pdf.set_font("Helvetica", "B", 10)
        pdf.multi_cell(w, 5, _latin1(f"{i}. {ins.headline}"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=9)
        pdf.set_text_color(90)
        pdf.multi_cell(w, 5, _latin1(ins.detail), new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0)
        pdf.ln(1)

    heading("Attrition by department")
    dt = department_table(df)
    table(
        ["Department", "Employees", "Leavers", "Attrition", "Avg income"],
        [[r["Department"], f"{r['Employees']:,}", f"{r['Leavers']:,}", f"{r['Attrition rate']:.1%}",
          f"${r['Avg monthly income']:,.0f}"] for _, r in dt.iterrows()],
        [w * 0.36, w * 0.16, w * 0.16, w * 0.16, w * 0.16],
    )

    if model_metrics is not None and not model_metrics.empty:
        heading("Risk model performance (hold-out test set)")
        cols = ["Accuracy", "Precision", "Recall", "ROC-AUC"]
        table(["Model"] + cols,
              [[str(idx)] + [f"{row[c]:.3f}" for c in cols] for idx, row in model_metrics.iterrows()],
              [w * 0.32] + [w * 0.17] * 4)

    return bytes(pdf.output())


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
def build_html(
    k: dict[str, float], filters: dict[str, str], insights: list[Insight],
    figures: list[tuple[str, go.Figure]], dept: pd.DataFrame,
) -> str:
    """Self-contained HTML report with interactive Plotly charts (plotly.js via CDN)."""
    cards = "".join(
        f'<div class="card"><div class="lbl">{html.escape(a)}</div><div class="val">{html.escape(b)}</div></div>'
        for a, b in _kpi_rows(k)
    )
    ins = "".join(
        f"<li><b>{html.escape(i.headline)}</b><br><span>{html.escape(i.detail)}</span></li>" for i in insights
    ) or "<li>Not enough data in the current filter to generate insights.</li>"
    filt = " &middot; ".join(f"{html.escape(a)}: <b>{html.escape(b)}</b>" for a, b in filters.items())
    charts = ""
    for idx, (title, fig) in enumerate(figures):
        fig = go.Figure(fig).update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white")
        charts += f"<h3>{html.escape(title)}</h3>" + fig.to_html(
            full_html=False, include_plotlyjs="cdn" if idx == 0 else False
        )
    dept_fmt = dept.assign(**{
        "Attrition rate": dept["Attrition rate"].map("{:.1%}".format),
        "Avg monthly income": dept["Avg monthly income"].map("${:,.0f}".format),
    })
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HR Attrition Summary</title>
<style>
 body {{ font-family: system-ui, -apple-system, 'Segoe UI', sans-serif; margin: 0 auto; max-width: 1000px;
        padding: 24px 16px; color: #0b0b0b; background: #fcfcfb; }}
 .muted {{ color: #52514e; font-size: 14px; }}
 .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 16px 0; }}
 .card {{ border: 1px solid #e1e0d9; border-radius: 10px; padding: 12px 14px; background: #fff; }}
 .lbl {{ font-size: 12px; color: #52514e; }} .val {{ font-size: 22px; font-weight: 600; margin-top: 4px; }}
 li {{ margin-bottom: 10px; }} li span {{ color: #52514e; font-size: 14px; }}
 table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
 th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid #e1e0d9; }}
</style></head><body>
<h1>HR Attrition &amp; Workforce Summary</h1>
<div class="muted">Generated {datetime.now():%d %b %Y, %H:%M} &middot; {filt}</div>
<div class="grid">{cards}</div>
<h2>Top insights</h2><ol>{ins}</ol>
<h2>Attrition by department</h2>{dept_fmt.to_html(index=False, border=0)}
<h2>Charts</h2>{charts}
</body></html>"""
