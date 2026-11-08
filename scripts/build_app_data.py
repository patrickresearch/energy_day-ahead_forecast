import polars as pl

from energy_forecast import explain, features, forecast
from energy_forecast.data import MIX, ROOT, TEST, load

OUT = ROOT / "data/app"

raw = load()
df = features.build(raw)
cfg = forecast.DEFAULT
model = forecast.fit(forecast.train_window(df, TEST[0], cfg), cfg)
test = df.filter(pl.col("day").is_between(*TEST))
preds = pl.read_parquet(ROOT / f"data/processed/benchmark_{TEST[0]}_{TEST[1]}.parquet")

tables = {
    "energy": raw.select("time", "local", "price", "load", "residual_load", *MIX),
    "backtest": preds,
    "scores": forecast.score(df, preds),
    "importance": explain.importance(explain.shap_values(model, test, cfg), cfg),
    "drivers": explain.drivers(model, raw, TEST, cfg),
}
OUT.mkdir(parents=True, exist_ok=True)
for name, table in tables.items():
    path = OUT / f"{name}.parquet"
    table.write_parquet(path)
    print(f"{name}: {table.height} rows, {path.stat().st_size / 1e6:.2f} MB")
