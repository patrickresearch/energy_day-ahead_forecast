import argparse
import subprocess
import sys
from datetime import date

import polars as pl

from energy_forecast import evaluate, features, forecast, plots, tracking
from energy_forecast.data import ROOT, TEST, load

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=date.fromisoformat, default=TEST[0])
parser.add_argument("--end", type=date.fromisoformat, default=TEST[1])
parser.add_argument(
    "--models", nargs="+", default=["naive_24h", "lightgbm", "chronos2_cov", "ensemble"]
)
parser.add_argument("--device", default="mps")
parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args()

period = (args.start, args.end)
name = f"{args.start}_{args.end}"
lgbm = [forecast.CONFIGS[m] for m in args.models if m in forecast.CONFIGS]
config = {**vars(args), "lightgbm": tracking.configs(lgbm)}
run = tracking.start("benchmark", name, config, args.wandb)
df = features.build(load())


def external(script: str, models: list[str]) -> pl.DataFrame:
    path = ROOT / f"data/processed/{script}_{name}.parquet"
    cmd = [
        sys.executable,
        ROOT / f"scripts/run_{script}.py",
        "--device",
        args.device,
        "--out",
        path,
    ]
    period_args = ["--start", args.start, "--end", args.end]
    extra = ["--models", *models] if script == "tft" else []
    subprocess.run([str(c) for c in [*cmd, *period_args, *extra]], check=True)
    return (
        pl.read_parquet(path).filter(pl.col("model").is_in(models)).select("time", "pred", "model")
    )


preds = []
for model in args.models:
    if model.startswith("naive_"):
        preds.append(forecast.seasonal_naive(df, int(model.removeprefix("naive_")[:-1]), period))
    elif model in forecast.CONFIGS:
        preds.append(forecast.backtest(df, forecast.CONFIGS[model], period))
        print(f"{model} done", flush=True)

for script, prefix in [("chronos", "chronos2"), ("tft", "tft")]:
    models = [m for m in args.models if m.startswith(prefix)]
    if models:
        preds.append(external(script, models))

preds = pl.concat(preds)
if "ensemble" in args.models:
    preds = pl.concat([preds, forecast.ensemble(preds, ["lightgbm", "chronos2_cov"])])

scores = forecast.score(df, preds)
dm = evaluate.dm_matrix(df, preds, scores["model"].to_list())
(ROOT / "results").mkdir(exist_ok=True)
preds.write_parquet(ROOT / f"data/processed/benchmark_{name}.parquet")
scores.write_csv(ROOT / f"results/scores_{name}.csv")
dm.write_csv(ROOT / f"results/dm_{name}.csv")
tracking.log_scores(run, scores)
run.log({"diebold_mariano": tracking.table(dm), "dm_heatmap": plots.dm_heatmap(dm)})
run.log({"mae_by_month": plots.mae_by(df, preds, "month")})
run.log({"mae_by_hour": plots.mae_by(df, preds, "hour")})
run.finish()
pl.Config.set_tbl_cols(-1)
print(scores)
print("Diebold-Mariano p-values (H1: column model more accurate than row model)")
print(dm.with_columns(pl.exclude("model").round(4)))
