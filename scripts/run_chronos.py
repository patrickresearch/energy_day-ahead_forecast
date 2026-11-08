import argparse
from datetime import date
from pathlib import Path

import polars as pl

from energy_forecast import features, foundation
from energy_forecast.data import ROOT, TEST, load

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=date.fromisoformat, default=TEST[0])
parser.add_argument("--end", type=date.fromisoformat, default=TEST[1])
parser.add_argument("--device", default="mps")
parser.add_argument("--out", type=Path, default=ROOT / "data/processed/chronos.parquet")
args = parser.parse_args()

df = features.build(load())
pipeline = foundation.load_pipeline(args.device)
period = (args.start, args.end)
preds = pl.concat([foundation.backtest(pipeline, df, period, covariates=c) for c in (False, True)])

args.out.parent.mkdir(parents=True, exist_ok=True)
preds.write_parquet(args.out)
print(preds.group_by("model").agg(hours=pl.len(), nulls=pl.col("pred").null_count()))
