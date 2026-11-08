import argparse
from datetime import date
from pathlib import Path

import polars as pl

from energy_forecast import features, tft
from energy_forecast.data import ROOT, TEST, load

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=date.fromisoformat, default=TEST[0])
parser.add_argument("--end", type=date.fromisoformat, default=TEST[1])
parser.add_argument("--models", nargs="+", default=list(tft.CONFIGS))
parser.add_argument("--device", default="mps")
parser.add_argument("--out", type=Path, default=ROOT / "data/processed/tft.parquet")
args = parser.parse_args()

df = features.build(load())
period = (args.start, args.end)
results = [tft.backtest(df, tft.CONFIGS[m], period, args.device) for m in args.models]
preds, vsn, lags = (pl.concat(parts) for parts in zip(*results, strict=True))

args.out.parent.mkdir(parents=True, exist_ok=True)
preds.write_parquet(args.out)
vsn.write_parquet(args.out.with_name(f"{args.out.stem}_vsn.parquet"))
lags.write_parquet(args.out.with_name(f"{args.out.stem}_attention.parquet"))
print(preds.group_by("model").agg(hours=pl.len(), nulls=pl.col("pred").null_count()))
