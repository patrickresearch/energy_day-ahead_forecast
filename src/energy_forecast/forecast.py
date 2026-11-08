from dataclasses import dataclass, field
from datetime import date, timedelta

import lightgbm as lgb
import numpy as np
import polars as pl

from energy_forecast.features import FEATURES

VAL = (date(2024, 11, 1), date(2025, 10, 31))
TEST = (date(2025, 11, 1), date(2026, 10, 31))

PARAMS = {
    "objective": "l1",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "feature_fraction": 0.8,
    "num_threads": 4,
    "force_col_wise": True,
    "seed": 42,
    "verbose": -1,
}


@dataclass(frozen=True)
class Config:
    name: str = "lightgbm"
    features: tuple[str, ...] = tuple(FEATURES)
    anchor: str | None = "prev_day_mean"
    window_days: int | None = 2 * 365
    params: dict = field(default_factory=lambda: dict(PARAMS))
    rounds: int = 1000


DEFAULT = Config()


def _period(df: pl.DataFrame, period: tuple[date, date]) -> pl.DataFrame:
    return df.filter(pl.col("day").is_between(*period))


def _anchor(df: pl.DataFrame, cfg: Config) -> np.ndarray | float:
    return df[cfg.anchor].to_numpy() if cfg.anchor else 0.0


def seasonal_naive(df: pl.DataFrame, lag: int, period: tuple[date, date] = TEST) -> pl.DataFrame:
    return _period(df, period).select(
        "time", pred=pl.col(f"price_lag{lag}"), model=pl.lit(f"naive_{lag}h")
    )


def fit(train: pl.DataFrame, cfg: Config = DEFAULT) -> lgb.Booster:
    train = train.drop_nulls(["price", cfg.anchor] if cfg.anchor else ["price"])
    data = lgb.Dataset(
        train.select(cfg.features).to_numpy(),
        train["price"].to_numpy() - _anchor(train, cfg),
        feature_name=list(cfg.features),
    )
    return lgb.train(cfg.params, data, num_boost_round=cfg.rounds)


def predict(model: lgb.Booster, df: pl.DataFrame, cfg: Config = DEFAULT) -> np.ndarray:
    return model.predict(df.select(cfg.features).to_numpy()) + _anchor(df, cfg)


def train_window(df: pl.DataFrame, before: date, cfg: Config) -> pl.DataFrame:
    train = df.filter(pl.col("day") < before)
    if cfg.window_days:
        train = train.filter(pl.col("day") >= before - timedelta(days=cfg.window_days))
    return train


def backtest(
    df: pl.DataFrame, cfg: Config = DEFAULT, period: tuple[date, date] = TEST
) -> pl.DataFrame:
    start, end = period
    bounds = [*pl.date_range(start, end, "1mo", eager=True).to_list(), end + timedelta(days=1)]
    preds = []
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=False):
        model = fit(train_window(df, lo, cfg), cfg)
        test = _period(df, (lo, hi - timedelta(days=1)))
        preds.append(
            test.select("time", model=pl.lit(cfg.name)).with_columns(
                pred=pl.Series(predict(model, test, cfg))
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
            bias=pl.col("err").mean(),
            hours=pl.len(),
        )
        .sort("mae")
    )
