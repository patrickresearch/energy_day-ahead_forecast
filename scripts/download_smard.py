import argparse
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from energy_forecast.smard import download

parser = argparse.ArgumentParser()
parser.add_argument("--start", default="2019-01-01")
parser.add_argument("--out", type=Path, default=Path("data/raw/smard_hourly.parquet"))
args = parser.parse_args()

df = download(datetime.fromisoformat(args.start).replace(tzinfo=UTC))
args.out.parent.mkdir(parents=True, exist_ok=True)
df.write_parquet(args.out)

span = df.select(pl.col("time").min().alias("first"), pl.col("time").max().alias("last"))
expected = int((span["last"][0] - span["first"][0]).total_seconds() // 3600) + 1
print(span)
print(f"rows={df.height} expected_hours={expected} duplicates={df['time'].is_duplicated().sum()}")
print(df.null_count().transpose(include_header=True, column_names=["nulls"]))
