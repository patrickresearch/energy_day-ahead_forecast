from datetime import date

import numpy as np
import polars as pl
from chronos import Chronos2Pipeline

from energy_forecast.forecast import TEST

COVARIATES = ["load_fc", "wind_fc", "solar_fc", "residual_load_fc"]
CONTEXT = 2048
HORIZON = 25


def load_pipeline(device: str = "mps") -> Chronos2Pipeline:
    return Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device)


def backtest(
    pipeline: Chronos2Pipeline,
    df: pl.DataFrame,
    period: tuple[date, date] = TEST,
    covariates: bool = True,
) -> pl.DataFrame:
    df = df.sort("time").with_row_index("row")
    days = (
        df.filter(pl.col("day").is_between(*period))
        .group_by("day")
        .agg(pl.col("row").min(), n=pl.len())
        .sort("day")
    )
    price = df["price"].to_numpy()
    cov = {c: df[c].to_numpy() for c in COVARIATES}
    inputs = []
    for i in days["row"].to_list():
        item = {"target": price[i - CONTEXT : i]}
        if covariates:
            item["past_covariates"] = {c: v[i - CONTEXT : i] for c, v in cov.items()}
            item["future_covariates"] = {c: v[i : i + HORIZON] for c, v in cov.items()}
        inputs.append(item)
    quantiles, _ = pipeline.predict_quantiles(
        inputs, prediction_length=HORIZON, quantile_levels=[0.5]
    )
    starts, lengths = days["row"].to_list(), days["n"].to_list()
    rows = np.concatenate([np.arange(i, i + n) for i, n in zip(starts, lengths, strict=True)])
    pred = np.concatenate(
        [q[0, :n, 0].float().cpu().numpy() for q, n in zip(quantiles, lengths, strict=True)]
    )
    return pl.DataFrame(
        {
            "time": df["time"].gather(rows),
            "pred": pred,
            "model": "chronos2_cov" if covariates else "chronos2",
        }
    )
