import argparse

import polars as pl

from energy_forecast import features, foundation
from energy_forecast.data import ROOT, TEST, load

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="mps")
args = parser.parse_args()

df = features.build(load())
pipeline = foundation.load_pipeline(args.device)
preds = pl.concat([foundation.backtest(pipeline, df, TEST, covariates=c) for c in (False, True)])

out = ROOT / "data/processed/chronos.parquet"
out.parent.mkdir(parents=True, exist_ok=True)
preds.write_parquet(out)
print(preds.group_by("model").agg(hours=pl.len(), nulls=pl.col("pred").null_count()))
