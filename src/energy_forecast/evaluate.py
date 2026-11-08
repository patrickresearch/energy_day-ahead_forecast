import math

import polars as pl


def daily_mae(df: pl.DataFrame, preds: pl.DataFrame) -> pl.DataFrame:
    return (
        preds.join(df.select("time", "price", "local"), on="time")
        .group_by(pl.col("local").dt.date().alias("day"), "model")
        .agg((pl.col("pred") - pl.col("price")).abs().mean())
        .pivot(on="model", index="day", values="pred")
        .drop_nulls()
        .sort("day")
    )


def diebold_mariano(daily: pl.DataFrame, a: str, b: str) -> float:
    d = (daily[a] - daily[b]).to_numpy()
    stat = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))
    return 0.5 * math.erfc(-stat / math.sqrt(2))


def dm_matrix(df: pl.DataFrame, preds: pl.DataFrame, models: list[str]) -> pl.DataFrame:
    daily = daily_mae(df, preds)
    return pl.DataFrame(
        [
            {
                "model": row,
                **{c: None if c == row else diebold_mariano(daily, c, row) for c in models},
            }
            for row in models
        ],
        schema={"model": pl.String, **dict.fromkeys(models, pl.Float64)},
    )
