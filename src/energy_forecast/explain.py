from datetime import date
from itertools import combinations
from math import factorial

import lightgbm as lgb
import numpy as np
import polars as pl

from energy_forecast import features, forecast
from energy_forecast.data import residual_load_fc
from energy_forecast.forecast import DEFAULT, Config

DRIVERS = {
    "Wind": ["wind_onshore_fc", "wind_offshore_fc"],
    "Solar": ["solar_fc"],
    "Load": ["load_fc"],
}


def shap_values(model: lgb.Booster, df: pl.DataFrame, cfg: Config = DEFAULT) -> pl.DataFrame:
    contrib = model.predict(df.select(cfg.features).to_numpy(), pred_contrib=True)
    return pl.DataFrame(contrib, schema=[*cfg.features, "base"], orient="row").with_columns(
        anchor=df[cfg.anchor] if cfg.anchor else pl.lit(0.0),
        time=df["time"],
        local=df["local"],
    )


def importance(shap: pl.DataFrame, cfg: Config = DEFAULT) -> pl.DataFrame:
    return (
        shap.select(pl.col(cfg.features).abs().mean())
        .transpose(include_header=True, header_name="feature", column_names=["mean_abs_shap"])
        .sort("mean_abs_shap", descending=True)
    )


def _predict_with(
    model: lgb.Booster,
    raw: pl.DataFrame,
    reference: dict[str, float],
    present: frozenset[str],
    period: tuple[date, date],
    cfg: Config,
) -> np.ndarray:
    absent = [c for name, cols in DRIVERS.items() if name not in present for c in cols]
    df = raw.with_columns(pl.lit(reference[c]).alias(c) for c in absent).with_columns(
        residual_load_fc=residual_load_fc()
    )
    return forecast.predict(
        model, features.build(df).filter(pl.col("day").is_between(*period)), cfg
    )


def drivers(
    model: lgb.Booster,
    raw: pl.DataFrame,
    period: tuple[date, date],
    cfg: Config = DEFAULT,
    reference_period: tuple[date, date] | None = None,
) -> pl.DataFrame:
    days = raw["local"].dt.date()
    in_period = raw.filter(days.is_between(*period))
    cols = [c for cols in DRIVERS.values() for c in cols]
    reference = (
        raw.filter(days.is_between(*(reference_period or period)))
        .select(pl.col(cols).mean())
        .row(0, named=True)
    )
    names, n = list(DRIVERS), len(DRIVERS)
    value = {
        frozenset(s): _predict_with(model, raw, reference, frozenset(s), period, cfg)
        for r in range(n + 1)
        for s in combinations(names, r)
    }
    effects = {}
    for name in names:
        others = [o for o in names if o != name]
        effects[name] = sum(
            factorial(r)
            * factorial(n - r - 1)
            / factorial(n)
            * (value[frozenset((*s, name))] - value[frozenset(s)])
            for r in range(n)
            for s in combinations(others, r)
        )
    return (
        in_period.sort("time")
        .select("time", "local")
        .with_columns(reference=value[frozenset()], pred=value[frozenset(names)], **effects)
    )
