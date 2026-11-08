import argparse
import subprocess
import sys
from datetime import date

import polars as pl

from energy_forecast import features, forecast
from energy_forecast.data import ROOT, TEST, load

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=date.fromisoformat, default=TEST[0])
parser.add_argument("--end", type=date.fromisoformat, default=TEST[1])
parser.add_argument(
    "--models", nargs="+", default=["naive_24h", "lightgbm", "chronos2_cov", "ensemble"]
)
parser.add_argument("--device", default="mps")
args = parser.parse_args()

period = (args.start, args.end)
name = f"{args.start}_{args.end}"
df = features.build(load())

preds = []
for model in args.models:
    if model.startswith("naive_"):
        preds.append(forecast.seasonal_naive(df, int(model.removeprefix("naive_")[:-1]), period))
    elif model in forecast.CONFIGS:
        preds.append(forecast.backtest(df, forecast.CONFIGS[model], period))
        print(f"{model} done", flush=True)

chronos = [m for m in args.models if m.startswith("chronos2")]
if chronos:
    path = ROOT / f"data/processed/chronos_{name}.parquet"
    cmd = [sys.executable, ROOT / "scripts/run_chronos.py", "--device", args.device, "--out", path]
    subprocess.run([*map(str, cmd), "--start", str(args.start), "--end", str(args.end)], check=True)
    preds.append(pl.read_parquet(path).filter(pl.col("model").is_in(chronos)))

preds = pl.concat(preds)
if "ensemble" in args.models:
    preds = pl.concat([preds, forecast.ensemble(preds, ["lightgbm", "chronos2_cov"])])

scores = forecast.score(df, preds)
(ROOT / "results").mkdir(exist_ok=True)
preds.write_parquet(ROOT / f"data/processed/benchmark_{name}.parquet")
scores.write_csv(ROOT / f"results/scores_{name}.csv")
print(scores)
