import argparse
import subprocess
import sys
from datetime import date, timedelta

import polars as pl

from energy_forecast import explain, features, forecast
from energy_forecast.data import MIX, ROOT, TEST, load

OUT = ROOT / "data/app"
TSO = ["load_fc", "wind_onshore_fc", "wind_offshore_fc", "solar_fc"]

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="mps")
args = parser.parse_args()


def latest_forecast_day(raw: pl.DataFrame) -> date:
    return (
        raw.group_by(day=pl.col("local").dt.date())
        .agg(
            complete=pl.all_horizontal(pl.col(TSO).is_not_null()).all()
            & (pl.col("local").dt.hour().min() == 0)
            & (pl.col("local").dt.hour().max() == 23)
        )
        .filter("complete")["day"]
        .max()
    )


def next_day(
    raw: pl.DataFrame, df: pl.DataFrame, cfg: forecast.Config, day: date
) -> tuple[pl.DataFrame, pl.DataFrame]:
    model = forecast.fit(forecast.train_window(df, day, cfg), cfg)
    target = df.filter(pl.col("day") == day)
    lgbm = pl.DataFrame(
        {"time": target["time"], "pred": forecast.predict(model, target, cfg), "model": "lightgbm"}
    )
    path = ROOT / "data/processed/chronos_next.parquet"
    cmd = [sys.executable, ROOT / "scripts/run_chronos.py", "--device", args.device, "--out", path]
    subprocess.run([*map(str, cmd), "--start", str(day), "--end", str(day)], check=True)
    chronos = pl.read_parquet(path).filter(pl.col("model") == "chronos2_cov")
    preds = pl.concat([lgbm, chronos])
    preds = pl.concat([preds, forecast.ensemble(preds, ["lightgbm", "chronos2_cov"])])
    reference = (day - timedelta(days=365), day - timedelta(days=1))
    drivers = explain.drivers(model, raw, (day, day), cfg, reference)
    return preds.with_columns(target=pl.lit(day)), drivers


raw = load()
df = features.build(raw)
cfg = forecast.DEFAULT
model = forecast.fit(forecast.train_window(df, TEST[0], cfg), cfg)
test = df.filter(pl.col("day").is_between(*TEST))
preds = pl.read_parquet(ROOT / f"data/processed/benchmark_{TEST[0]}_{TEST[1]}.parquet")

day = latest_forecast_day(raw)
latest, next_drivers = next_day(raw, df, cfg, day)
archive = OUT / "forecasts.parquet"
if archive.exists():
    latest = pl.concat([pl.read_parquet(archive).filter(pl.col("target") != day), latest])

tables = {
    "energy": raw.select("time", "local", "price", "load", "residual_load", *MIX),
    "backtest": preds,
    "scores": forecast.score(df, preds),
    "importance": explain.importance(explain.shap_values(model, test, cfg), cfg),
    "drivers": explain.drivers(model, raw, TEST, cfg),
    "forecasts": latest.sort("target", "model", "time"),
    "next_drivers": next_drivers,
}
OUT.mkdir(parents=True, exist_ok=True)
for name, table in tables.items():
    path = OUT / f"{name}.parquet"
    table.write_parquet(path)
    print(f"{name}: {table.height} rows, {path.stat().st_size / 1e6:.2f} MB")
print(f"next-day forecast for {day}")
