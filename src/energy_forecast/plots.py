from datetime import date

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from energy_forecast.data import MIX, RENEWABLE

SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CATEGORICAL = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]
DIVERGING = ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f2a3a2", "#e34948", "#a8302f"]
INK = "#0b0b0b"
DRIVER_COLORS = {"Wind": CATEGORICAL[0], "Solar": CATEGORICAL[1], "Load": CATEGORICAL[2]}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _hour_month_grid(df: pl.DataFrame) -> np.ndarray:
    cells = df.group_by(hour=pl.col("local").dt.hour(), month=pl.col("local").dt.month()).agg(
        pl.col("price").median()
    )
    grid = np.full((24, 12), np.nan)
    grid[cells["hour"].to_numpy(), cells["month"].to_numpy() - 1] = cells["price"].to_numpy()
    return grid


def price_heatmap(df: pl.DataFrame, year: int) -> go.Figure:
    grid = _hour_month_grid(df.filter(pl.col("local").dt.year() == year))
    fig = px.imshow(
        grid,
        x=MONTHS,
        y=list(range(24)),
        color_continuous_scale=SEQUENTIAL,
        aspect="auto",
        labels={"x": "Month", "y": "Hour (Europe/Berlin)", "color": "€/MWh"},
        title=f"Median day-ahead price by hour and month, {year}",
    )
    fig.update_yaxes(dtick=3)
    fig.update_traces(hovertemplate="%{x}, %{y}:00<br>%{z:.0f} €/MWh<extra></extra>")
    return fig


def price_heatmap_by_year(df: pl.DataFrame) -> go.Figure:
    years = df["local"].dt.year().unique().sort().to_list()
    grids = np.stack([_hour_month_grid(df.filter(pl.col("local").dt.year() == y)) for y in years])
    fig = px.imshow(
        grids,
        x=MONTHS,
        y=list(range(24)),
        facet_col=0,
        facet_col_wrap=4,
        color_continuous_scale=SEQUENTIAL,
        aspect="auto",
        labels={"color": "€/MWh"},
        title="Median day-ahead price by hour and month",
    )
    fig.for_each_annotation(lambda a: a.update(text=str(years[int(a.text.split("=")[1])])))
    fig.update_traces(hovertemplate="%{x}, %{y}:00<br>%{z:.0f} €/MWh<extra></extra>")
    return fig


def mix_area(df: pl.DataFrame, start: date, end: date) -> go.Figure:
    d = df.filter(pl.col("local").dt.date().is_between(start, end))
    fig = go.Figure()
    for name, color in zip(MIX, CATEGORICAL, strict=True):
        fig.add_scatter(
            x=d["local"],
            y=d[name] / 1000,
            name=name,
            stackgroup="mix",
            mode="lines",
            line={"width": 0, "color": color},
            fillcolor=color,
        )
    fig.add_scatter(x=d["local"], y=d["load"] / 1000, name="Load", line={"width": 2, "color": INK})
    fig.update_layout(
        title=f"Generation by source, {start:%d %b %Y} – {end:%d %b %Y}",
        yaxis_title="GW",
        hovermode="x unified",
    )
    fig.update_traces(hovertemplate="%{y:.1f} GW")
    return fig


def price_line(df: pl.DataFrame, start: date, end: date) -> go.Figure:
    d = df.filter(pl.col("local").dt.date().is_between(start, end))
    fig = px.line(d, x="local", y="price", title="Day-ahead price")
    fig.update_traces(
        line={"width": 2, "color": INK}, hovertemplate="%{x|%d %b %H:00}: %{y:.1f} €/MWh"
    )
    fig.update_layout(xaxis_title="", yaxis_title="€/MWh", hovermode="x unified")
    return fig


def merit_order(df: pl.DataFrame) -> go.Figure:
    d = df.drop_nulls(["residual_load", "price"]).with_columns(
        year=pl.col("local").dt.year(), residual_gw=pl.col("residual_load") / 1000
    )
    fig = px.density_heatmap(
        d,
        x="residual_gw",
        y="price",
        facet_col="year",
        facet_col_wrap=4,
        nbinsx=50,
        nbinsy=60,
        color_continuous_scale=SEQUENTIAL,
        labels={"residual_gw": "Residual load (GW)", "price": "€/MWh", "year": ""},
        title="Residual load vs day-ahead price",
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.lstrip("=")))
    return fig


def negative_hours(df: pl.DataFrame) -> go.Figure:
    d = (
        df.group_by(year=pl.col("local").dt.year())
        .agg(hours=(pl.col("price") < 0).sum())
        .sort("year")
        .with_columns(pl.col("year").cast(pl.String))
    )
    fig = px.bar(
        d, x="year", y="hours", text_auto=True, title="Hours with negative day-ahead price"
    )
    fig.update_traces(marker_color=CATEGORICAL[0])
    fig.update_layout(xaxis_title="", yaxis_title="Hours")
    return fig


def renewable_share(df: pl.DataFrame) -> go.Figure:
    d = (
        df.drop_nulls(list(MIX))
        .sort("local")
        .group_by_dynamic("local", every="1mo")
        .agg(share=pl.sum_horizontal(RENEWABLE).sum() / pl.sum_horizontal(list(MIX)).sum())
    )
    fig = px.line(d, x="local", y="share", title="Renewable share of generation, monthly")
    fig.update_traces(
        line={"width": 2, "color": CATEGORICAL[0]}, hovertemplate="%{x|%b %Y}: %{y:.0%}"
    )
    fig.update_layout(xaxis_title="", yaxis_title="", yaxis_tickformat=".0%")
    return fig


def forecast_lines(df: pl.DataFrame, preds: pl.DataFrame, start: date, end: date) -> go.Figure:
    d = preds.join(df.select("time", "local", "price"), on="time").filter(
        pl.col("local").dt.date().is_between(start, end)
    )
    actual = d.unique("time").sort("time")
    fig = go.Figure()
    fig.add_scatter(
        x=actual["local"], y=actual["price"], name="Actual", line={"width": 2, "color": INK}
    )
    for name, color in zip(d["model"].unique().sort(), CATEGORICAL, strict=False):
        m = d.filter(pl.col("model") == name).sort("time")
        fig.add_scatter(x=m["local"], y=m["pred"], name=name, line={"width": 2, "color": color})
    fig.update_layout(
        title=f"Day-ahead forecasts, {start:%d %b %Y} – {end:%d %b %Y}",
        yaxis_title="€/MWh",
        hovermode="x unified",
    )
    return fig


def mae_by(df: pl.DataFrame, preds: pl.DataFrame, by: str) -> go.Figure:
    d = (
        preds.join(df.select("time", "price", "local"), on="time")
        .with_columns(hour=pl.col("local").dt.hour(), month=pl.col("local").dt.strftime("%Y-%m"))
        .group_by(by, "model")
        .agg(mae=(pl.col("pred") - pl.col("price")).abs().mean())
        .sort(by, "model")
    )
    models = d["model"].unique().sort().to_list()
    fig = px.line(
        d,
        x=by,
        y="mae",
        color="model",
        category_orders={"model": models},
        color_discrete_sequence=CATEGORICAL,
        markers=True,
        title=f"MAE by {by}",
    )
    fig.update_layout(yaxis_title="MAE (€/MWh)", xaxis_title="")
    return fig


def shap_importance(imp: pl.DataFrame, top: int = 15) -> go.Figure:
    d = imp.head(top).reverse()
    fig = px.bar(d, x="mean_abs_shap", y="feature", orientation="h", title="Mean |SHAP| value")
    fig.update_traces(marker_color=CATEGORICAL[0], hovertemplate="%{y}: %{x:.1f} €/MWh")
    fig.update_layout(xaxis_title="€/MWh", yaxis_title="", height=40 + 28 * len(d))
    return fig


def shap_dependence(
    features: pl.DataFrame, shap: pl.DataFrame, feature: str, n: int = 5000
) -> go.Figure:
    d = pl.DataFrame({"value": features[feature], "shap": shap[feature]}).sample(
        min(n, features.height), seed=42
    )
    fig = px.scatter(d, x="value", y="shap", opacity=0.3, title=f"SHAP dependence: {feature}")
    fig.update_traces(marker={"size": 4, "color": CATEGORICAL[0]})
    fig.update_layout(xaxis_title=feature, yaxis_title="SHAP (€/MWh)")
    return fig


def shap_day(shap: pl.DataFrame, features: list[str], day: date, top: int = 8) -> go.Figure:
    d = shap.filter(pl.col("local").dt.date() == day).select(*features, "base", "anchor").mean()
    contrib = sorted(((f, d[f][0]) for f in features), key=lambda x: abs(x[1]), reverse=True)
    rest = sum(v for _, v in contrib[top:])
    start = d["base"][0] + d["anchor"][0]
    names = ["prev-day mean + base", *(f for f, _ in contrib[:top]), "other", "forecast"]
    values = [start, *(v for _, v in contrib[:top]), rest, 0]
    fig = go.Figure(
        go.Waterfall(
            x=names,
            y=values,
            measure=["absolute", *["relative"] * (top + 1), "total"],
            increasing={"marker": {"color": CATEGORICAL[7]}},
            decreasing={"marker": {"color": CATEGORICAL[0]}},
            totals={"marker": {"color": INK}},
            hovertemplate="%{x}: %{y:.1f} €/MWh<extra></extra>",
        )
    )
    fig.update_layout(title=f"Daily mean forecast explained, {day:%d %b %Y}", yaxis_title="€/MWh")
    return fig


def driver_profile(drivers: pl.DataFrame) -> go.Figure:
    names = list(DRIVER_COLORS)
    cells = drivers.group_by(hour=pl.col("local").dt.hour(), month=pl.col("local").dt.month()).agg(
        pl.col(names).mean()
    )
    limit = float(cells.select(pl.max_horizontal(pl.col(names).abs().max())).item())
    fig = make_subplots(rows=1, cols=len(names), subplot_titles=names, shared_yaxes=True)
    for i, name in enumerate(names, 1):
        grid = np.full((24, 12), np.nan)
        grid[cells["hour"].to_numpy(), cells["month"].to_numpy() - 1] = cells[name].to_numpy()
        fig.add_heatmap(
            z=grid,
            x=MONTHS,
            y=list(range(24)),
            coloraxis="coloraxis",
            hovertemplate=f"{name}<br>%{{x}}, %{{y}}:00<br>%{{z:+.1f}} €/MWh<extra></extra>",
            row=1,
            col=i,
        )
    fig.update_layout(
        title="Mean price effect by hour and month (vs. average hour)",
        coloraxis={
            "colorscale": DIVERGING,
            "cmin": -limit,
            "cmax": limit,
            "colorbar": {"title": "€/MWh"},
        },
        height=480,
    )
    fig.update_yaxes(autorange="reversed", dtick=3)
    fig.update_yaxes(title_text="Hour", row=1, col=1)
    return fig


def driver_day(drivers: pl.DataFrame, day: date) -> go.Figure:
    d = drivers.filter(pl.col("local").dt.date() == day)
    fig = go.Figure()
    for name, color in DRIVER_COLORS.items():
        fig.add_bar(
            x=d["local"],
            y=d[name],
            name=name,
            marker={"color": color, "line": {"width": 2, "color": "white"}},
            hovertemplate=f"{name}: %{{y:+.1f}} €/MWh<extra></extra>",
        )
    fig.add_scatter(
        x=d["local"],
        y=d["pred"] - d["reference"],
        name="Total",
        mode="lines+markers",
        line={"width": 2, "color": INK},
        marker={"size": 8},
        hovertemplate="Total: %{y:+.1f} €/MWh<extra></extra>",
    )
    fig.update_layout(
        barmode="relative",
        title=f"Price effect of wind, solar and load, {day:%d %b %Y}",
        yaxis_title="€/MWh vs. average hour",
        hovermode="x unified",
    )
    return fig
