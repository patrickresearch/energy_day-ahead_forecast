from datetime import date, timedelta

import lightgbm as lgb
import polars as pl

from energy_forecast.features import FEATURES

TEST_START = date(2025, 11, 1)
TEST_END = date(2026, 10, 31)

PARAMS = {
    "objective": "l1",
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}


def _test(df: pl.DataFrame, start: date, end: date) -> pl.DataFrame:
    return df.filter(pl.col("day").is_between(start, end))


def seasonal_naive(
    df: pl.DataFrame, lag: int, start: date = TEST_START, end: date = TEST_END
) -> pl.DataFrame:
    return _test(df, start, end).select(
        "time", pred=pl.col(f"price_lag{lag}"), model=pl.lit(f"naive_{lag}h")
    )


def fit(train: pl.DataFrame, params: dict = PARAMS) -> lgb.LGBMRegressor:
    train = train.drop_nulls("price")
    return lgb.LGBMRegressor(**params).fit(
        train.select(FEATURES).to_numpy(), train["price"].to_numpy(), feature_name=FEATURES
    )


def lightgbm_backtest(
    df: pl.DataFrame, start: date = TEST_START, end: date = TEST_END, params: dict = PARAMS
) -> pl.DataFrame:
    bounds = [*pl.date_range(start, end, "1mo", eager=True).to_list(), end + timedelta(days=1)]
    preds = []
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=False):
        model = fit(df.filter(pl.col("day") < lo), params)
        test = _test(df, lo, hi - timedelta(days=1))
        preds.append(
            test.select("time", model=pl.lit("lightgbm")).with_columns(
                pred=pl.Series(model.predict(test.select(FEATURES).to_numpy()))
            )
        )
    return pl.concat(preds).select("time", "pred", "model")


def score(df: pl.DataFrame, preds: pl.DataFrame) -> pl.DataFrame:
    err = preds.join(df.select("time", "price"), on="time").with_columns(
        err=pl.col("pred") - pl.col("price")
    )
    return (
        err.group_by("model")
        .agg(
            mae=pl.col("err").abs().mean(),
            rmse=pl.col("err").pow(2).mean().sqrt(),
            hours=pl.len(),
        )
        .sort("mae")
    )
