import argparse
from dataclasses import replace
from datetime import date

import polars as pl

from energy_forecast import explain, features, forecast
from energy_forecast.data import ROOT, load

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=date.fromisoformat, default=date(2025, 3, 1))
parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 10, 31))
parser.add_argument("--base", default="lgbm_all")
parser.add_argument("--ks", type=int, nargs="+", default=[30, 24, 20, 16, 12, 10, 8, 6])
args = parser.parse_args()

period = (args.start, args.end)
name = f"{args.start}_{args.end}"
df = features.build(load())
cfg = forecast.CONFIGS[args.base]

model = forecast.fit(forecast.train_window(df, args.start, cfg), cfg)
held_out = df.filter(pl.col("day").is_between(*period))
imp = explain.importance(explain.shap_values(model, held_out, cfg), cfg)
pl.Config.set_tbl_rows(-1)
print(imp)

preds = []
for k in args.ks:
    top = replace(cfg, name=f"top{k}", features=tuple(imp["feature"].head(k)))
    preds.append(forecast.backtest(df, top, period))
    print(f"top{k} done", flush=True)

scores = forecast.score(df, pl.concat(preds))
(ROOT / "results").mkdir(exist_ok=True)
imp.write_csv(ROOT / f"results/shap_{args.base}_{name}.csv")
scores.write_csv(ROOT / f"results/prune_{args.base}_{name}.csv")
print(scores)
