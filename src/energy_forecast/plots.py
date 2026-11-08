from datetime import date

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl

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
INK = "#0b0b0b"
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
