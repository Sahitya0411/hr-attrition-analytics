"""Plotly chart factory with one consistent, colorblind-checked palette.

Charts leave background and font colors to the Streamlit theme
(`st.plotly_chart(..., theme="streamlit")`), so they work in light and dark
mode. Only data colors are set here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Categorical palette in a fixed order (validated for CVD separation).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
MUTED = "#898781"

# Attrition status always uses the same two colors throughout the app.
STAYED_COLOR, LEFT_COLOR = BLUE, ORANGE
ATTRITION_COLORS = {"No": STAYED_COLOR, "Yes": LEFT_COLOR, "Stayed": STAYED_COLOR, "Left": LEFT_COLOR}

# Sequential (one hue, light -> dark) and diverging (blue <-> gray <-> red).
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = [[0.0, "#184f95"], [0.25, "#6da7ec"], [0.5, "#9a9893"], [0.75, "#ec8a86"], [1.0, "#b8302f"]]

FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def _base_layout(fig: go.Figure, title: str | None = None, height: int = 360) -> go.Figure:
    """Common layout: thin margins, recessive grid, legend on top."""
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15)) if title else None,
        height=height,
        margin=dict(l=8, r=8, t=48 if title else 16, b=8),
        font=dict(family=FONT, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
        hoverlabel=dict(font=dict(family=FONT)),
        bargap=0.3,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridwidth=1, zeroline=False)
    return fig


def rate_bar(
    df: pd.DataFrame,
    category: str,
    overall_rate: float,
    title: str | None = None,
    horizontal: bool = False,
    height: int = 340,
) -> go.Figure:
    """Attrition rate per category with a dashed company-average reference.

    Expects the output of `data_loader.attrition_rate_by`. Bars above the
    average are highlighted so the takeaway is visible at a glance.
    """
    colors = [ORANGE if r > overall_rate else BLUE for r in df["attrition_rate"]]
    labels = [str(v) for v in df[category]]
    custom = np.stack([df["employees"], df["leavers"]], axis=-1)
    hover = (
        "<b>%{customdata[2]}</b><br>Attrition rate: %{" + ("x" if horizontal else "y") + ":.1%}"
        "<br>Leavers: %{customdata[1]:,} of %{customdata[0]:,}<extra></extra>"
    )
    custom = np.column_stack([custom, labels])
    bar = go.Bar(
        x=df["attrition_rate"] if horizontal else labels,
        y=labels if horizontal else df["attrition_rate"],
        orientation="h" if horizontal else "v",
        marker=dict(color=colors, cornerradius=4),
        text=[f"{r:.0%}" for r in df["attrition_rate"]],
        textposition="outside",
        cliponaxis=False,
        customdata=custom,
        hovertemplate=hover,
        showlegend=False,
    )
    fig = go.Figure(bar)
    ref = dict(line_dash="dash", line_color=MUTED, line_width=1.5,
               annotation_text=f"Company avg {overall_rate:.0%}", annotation_font_color=MUTED)
    # Headroom so the outside value labels are never clipped.
    top = max(float(df["attrition_rate"].max()), overall_rate) * 1.18 or 0.1
    if horizontal:
        fig.add_vline(x=overall_rate, annotation_position="top", **ref)
        fig.update_xaxes(tickformat=".0%", showgrid=True, range=[0, top])
        fig.update_yaxes(autorange="reversed", showgrid=False)
    else:
        fig.add_hline(y=overall_rate, annotation_position="top right", **ref)
        fig.update_yaxes(tickformat=".0%", range=[0, top])
    return _base_layout(fig, title, height)


def attrition_donut(leavers: int, stayers: int, title: str | None = None) -> go.Figure:
    """Share of leavers vs stayers."""
    fig = go.Figure(go.Pie(
        labels=["Stayed", "Left"],
        values=[stayers, leavers],
        hole=0.62,
        sort=False,
        marker=dict(colors=[STAYED_COLOR, LEFT_COLOR], line=dict(width=2, color="rgba(0,0,0,0)")),
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
    ))
    total = leavers + stayers
    rate = leavers / total if total else 0
    fig.add_annotation(text=f"<b>{rate:.1%}</b><br>attrition", showarrow=False, font=dict(size=16))
    fig.update_layout(showlegend=False)
    return _base_layout(fig, title, 320)


def stacked_share(df: pd.DataFrame, category: str, title: str | None = None, height: int = 320) -> go.Figure:
    """100% stacked bar of stayed vs left per category (shows composition)."""
    ct = pd.crosstab(df[category], df["Attrition"], normalize="index")
    counts = pd.crosstab(df[category], df["Attrition"])
    fig = go.Figure()
    for status, name in (("No", "Stayed"), ("Yes", "Left")):
        if status not in ct:
            continue
        fig.add_bar(
            y=[str(i) for i in ct.index], x=ct[status], name=name, orientation="h",
            marker=dict(color=ATTRITION_COLORS[status], line=dict(width=2, color="rgba(0,0,0,0)")),
            customdata=counts[status], text=[f"{v:.0%}" for v in ct[status]], textposition="inside",
            hovertemplate=f"%{{y}}<br>{name}: %{{x:.1%}} (%{{customdata:,}})<extra></extra>",
        )
    fig.update_layout(barmode="stack")
    fig.update_xaxes(tickformat=".0%")
    return _base_layout(fig, title, height)


def distribution_by_attrition(df: pd.DataFrame, col: str, title: str | None = None, nbins: int = 30) -> go.Figure:
    """Overlaid normalized histograms for stayers vs leavers."""
    fig = go.Figure()
    for status, name in (("No", "Stayed"), ("Yes", "Left")):
        sub = df.loc[df["Attrition"] == status, col]
        fig.add_histogram(
            x=sub, name=name, histnorm="percent", nbinsx=nbins, opacity=0.75,
            marker=dict(color=ATTRITION_COLORS[status]),
            hovertemplate=f"{name}<br>{col}: %{{x}}<br>%{{y:.1f}}% of group<extra></extra>",
        )
    fig.update_layout(barmode="overlay")
    fig.update_yaxes(ticksuffix="%")
    return _base_layout(fig, title, 320)


def correlation_heatmap(corr: pd.DataFrame, title: str | None = None) -> go.Figure:
    """Diverging correlation heatmap (blue = negative, red = positive)."""
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        zmin=-1, zmax=1, colorscale=DIVERGING,
        text=np.round(corr.values, 2), texttemplate="%{text}", textfont=dict(size=9),
        hovertemplate="%{y} vs %{x}<br>r = %{z:.2f}<extra></extra>",
        colorbar=dict(thickness=10, len=0.8),
        xgap=2, ygap=2,
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(tickangle=-45)
    return _base_layout(fig, title, 620)


def heatmap_matrix(pivot: pd.DataFrame, title: str | None = None, fmt: str = ".0%") -> go.Figure:
    """Sequential heatmap for a 2D pivot of rates."""
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=[str(c) for c in pivot.columns], y=[str(i) for i in pivot.index],
        colorscale=[[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)],
        texttemplate="%{z:" + fmt + "}", hovertemplate="%{y} / %{x}<br>%{z:" + fmt + "}<extra></extra>",
        colorbar=dict(thickness=10, tickformat=fmt), xgap=2, ygap=2,
    ))
    fig.update_yaxes(showgrid=False)
    return _base_layout(fig, title, 300)


def grouped_bar(
    df: pd.DataFrame, x: str, y: str, color: str, title: str | None = None,
    y_format: str = ".0%", height: int = 340,
) -> go.Figure:
    """Grouped bars with categorical colors assigned in fixed order."""
    fig = go.Figure()
    for i, (key, sub) in enumerate(df.groupby(color, sort=False, observed=True)):
        fig.add_bar(
            x=sub[x].astype(str), y=sub[y], name=str(key),
            marker=dict(color=CATEGORICAL[i % len(CATEGORICAL)], cornerradius=4),
            hovertemplate=f"{key}<br>%{{x}}: %{{y:{y_format}}}<extra></extra>",
        )
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickformat=y_format)
    return _base_layout(fig, title, height)


def simple_bar(df: pd.DataFrame, x: str, y: str, title: str | None = None,
               y_format: str = ",.0f", color: str = BLUE, height: int = 320) -> go.Figure:
    """Single-series bar chart for arbitrary query results."""
    fig = go.Figure(go.Bar(
        x=df[x].astype(str), y=df[y], marker=dict(color=color, cornerradius=4),
        hovertemplate="%{x}<br>%{y:" + y_format + "}<extra></extra>",
    ))
    fig.update_yaxes(tickformat=y_format)
    return _base_layout(fig, title, height)


def line_with_bars(df: pd.DataFrame, x: str, bar_y: str, line_y: str,
                   title: str | None = None, height: int = 340) -> go.Figure:
    """Rate bars plus a cumulative-share line, both on one 0-100% axis."""
    fig = go.Figure()
    fig.add_bar(x=df[x], y=df[bar_y], name="Attrition rate", marker=dict(color=BLUE, cornerradius=4),
                hovertemplate="%{x}<br>Attrition rate: %{y:.1%}<extra></extra>")
    fig.add_scatter(x=df[x], y=df[line_y], name="Cumulative share of leavers", mode="lines+markers",
                    line=dict(color=ORANGE, width=2), marker=dict(size=8),
                    hovertemplate="%{x}<br>Cumulative share: %{y:.0%}<extra></extra>")
    fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
    return _base_layout(fig, title, height)


# --------------------------------------------------------------------------- #
# Model charts
# --------------------------------------------------------------------------- #
def confusion_matrix_fig(cm: np.ndarray, title: str | None = None) -> go.Figure:
    labels = ["Stayed", "Left"]
    total = cm.sum()
    text = [[f"{cm[i, j]:,}<br>({cm[i, j] / total:.0%})" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(
        z=cm, x=[f"Predicted {lbl}" for lbl in labels], y=[f"Actual {lbl}" for lbl in labels],
        colorscale=[[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)],
        text=text, texttemplate="%{text}", showscale=False, xgap=3, ygap=3,
        hovertemplate="%{y}, %{x}: %{z}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return _base_layout(fig, title, 300)


def roc_curves(curves: dict[str, tuple[np.ndarray, np.ndarray, float]], title: str | None = None) -> go.Figure:
    """ROC curves for several models. curves = {name: (fpr, tpr, auc)}."""
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                    line=dict(color=MUTED, dash="dash", width=1), hoverinfo="skip")
    for i, (name, (fpr, tpr, auc)) in enumerate(curves.items()):
        fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC {auc:.3f})",
                        line=dict(color=CATEGORICAL[i], width=2),
                        hovertemplate="FPR %{x:.2f}<br>TPR %{y:.2f}<extra></extra>")
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate", range=[0, 1.02])
    return _base_layout(fig, title, 360)


def importance_bar(imp: pd.DataFrame, title: str | None = None, top_n: int = 12) -> go.Figure:
    """Horizontal bar of feature importance (expects columns feature, importance)."""
    top = imp.nlargest(top_n, "importance").iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top["importance"], y=top["feature"], orientation="h",
        marker=dict(color=BLUE, cornerradius=4),
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    return _base_layout(fig, title, max(300, 26 * len(top) + 60))


def contribution_bar(contrib: pd.DataFrame, title: str | None = None) -> go.Figure:
    """Signed contributions to one prediction (red raises risk, blue lowers it)."""
    d = contrib.iloc[::-1]
    colors = [RED if v > 0 else BLUE for v in d["contribution"]]
    fig = go.Figure(go.Bar(
        x=d["contribution"], y=d["label"], orientation="h", marker=dict(color=colors, cornerradius=4),
        hovertemplate="%{y}<br>Effect on log-odds: %{x:+.2f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=MUTED, line_width=1)
    fig.update_xaxes(title="Pushes risk down  ←  →  Pushes risk up", showgrid=True)
    return _base_layout(fig, title, max(260, 34 * len(d) + 80))


# --------------------------------------------------------------------------- #
# Forecast chart
# --------------------------------------------------------------------------- #
def forecast_fig(
    baseline: pd.DataFrame, scenario: pd.DataFrame, band: pd.DataFrame | None = None,
    title: str | None = None,
) -> go.Figure:
    """Baseline vs scenario total headcount with an optional P10-P90 band."""
    fig = go.Figure()
    if band is not None:
        fig.add_scatter(x=band["month"], y=band["p90"], mode="lines", line=dict(width=0),
                        showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=band["month"], y=band["p10"], mode="lines", line=dict(width=0),
                        fill="tonexty", fillcolor="rgba(235,104,52,0.15)", name="Scenario P10-P90",
                        hovertemplate="P10 %{y:,.0f}<extra></extra>")
    fig.add_scatter(x=baseline["month"], y=baseline["headcount"], name="Baseline", mode="lines",
                    line=dict(color=BLUE, width=2, dash="dash"),
                    hovertemplate="%{x|%b %Y}<br>Baseline: %{y:,.0f}<extra></extra>")
    fig.add_scatter(x=scenario["month"], y=scenario["headcount"], name="Scenario", mode="lines",
                    line=dict(color=ORANGE, width=2),
                    hovertemplate="%{x|%b %Y}<br>Scenario: %{y:,.0f}<extra></extra>")
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(tickformat=",.0f")
    return _base_layout(fig, title, 380)


def dept_forecast_fig(scenario_by_dept: pd.DataFrame, title: str | None = None) -> go.Figure:
    """Scenario headcount per department (small number of series, fixed colors)."""
    fig = go.Figure()
    for i, (dept, sub) in enumerate(scenario_by_dept.groupby("Department", sort=True)):
        fig.add_scatter(x=sub["month"], y=sub["headcount"], name=dept, mode="lines",
                        line=dict(color=CATEGORICAL[i % len(CATEGORICAL)], width=2),
                        hovertemplate=f"{dept}<br>%{{x|%b %Y}}: %{{y:,.0f}}<extra></extra>")
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(tickformat=",.0f")
    return _base_layout(fig, title, 340)
