import sys
from datetime import timedelta
from pathlib import Path

import plotly.io as pio
import polars as pl
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from energy_forecast import plots

DATA = ROOT / "data/app"

st.set_page_config(page_title="German day-ahead power prices", layout="wide")
pio.templates.default = "plotly_white"


@st.cache_data
def read(name: str) -> pl.DataFrame:
    return pl.read_parquet(DATA / f"{name}.parquet")


def show(fig) -> None:
    st.plotly_chart(fig, theme=None, width="stretch")


energy, preds, scores = read("energy"), read("backtest"), read("scores")
importance, drivers = read("importance"), read("drivers")
first, last = energy["local"].min().date(), energy["local"].max().date()
test_start, test_end = preds["time"].min(), preds["time"].max()

st.title("German day-ahead power prices")
st.caption(
    f"Bidding zone DE-LU · hourly SMARD data {first:%b %Y} – {last:%d %b %Y} · "
    "forecast for delivery day D issued at 12:00 on D-1"
)

mix_tab, forecast_tab, drivers_tab = st.tabs(["Energy mix", "Forecast", "Drivers"])

with mix_tab:
    period = st.date_input(
        "Period", (last - timedelta(days=6), last), min_value=first, max_value=last
    )
    if len(period) == 2:
        show(plots.mix_area(energy, *period))
    years = energy["local"].dt.year().unique().sort().to_list()
    year = st.select_slider("Year", years, value=years[-1])
    show(plots.price_heatmap(energy, year))
    left, right = st.columns(2)
    with left:
        show(plots.negative_hours(energy))
    with right:
        show(plots.renewable_share(energy))
    show(plots.merit_order(energy))

with forecast_tab:
    st.markdown(
        f"Walk-forward backtest {test_start:%b %Y} – {test_end:%b %Y}: monthly refit, "
        "only information available at gate closure. Ensemble = mean of LightGBM and "
        "Chronos-2 with covariates."
    )
    for col, row in zip(st.columns(scores.height), scores.iter_rows(named=True), strict=True):
        col.metric(row["model"], f"{row['mae']:.1f} €/MWh", help="Mean absolute error")
    days = preds["time"].dt.convert_time_zone("Europe/Berlin").dt.date()
    week = st.date_input(
        "Week starting", days.max() - timedelta(days=6), min_value=days.min(), max_value=days.max()
    )
    show(plots.forecast_lines(energy, preds, week, week + timedelta(days=6)))
    left, right = st.columns(2)
    with left:
        show(plots.mae_by(energy, preds, "month"))
    with right:
        show(plots.mae_by(energy, preds, "hour"))
    st.dataframe(scores.with_columns(pl.col("mae", "rmse", "bias").round(2)), hide_index=True)

with drivers_tab:
    st.markdown(
        "How wind, solar and load move the LightGBM forecast. Each TSO forecast is replaced by "
        "its test-year mean in all combinations (exact Shapley values), so effects are "
        "relative to an average hour and add up to the forecast."
    )
    show(plots.driver_profile(drivers))
    daily = drivers.group_by(day=pl.col("local").dt.date()).agg(pl.col("pred").mean()).sort("pred")
    day = st.date_input(
        "Day",
        daily["day"][0],
        min_value=daily["day"].min(),
        max_value=daily["day"].max(),
    )
    show(plots.driver_day(drivers, day))
    show(plots.shap_importance(importance))
