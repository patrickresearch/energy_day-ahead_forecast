import holidays
import polars as pl

LAGS = [24, 48, 168]

BASE_FEATURES = [
    "hour",
    "dow",
    "month",
    "holiday",
    "load_fc",
    "wind_fc",
    "solar_fc",
    "residual_load_fc",
    *(f"price_lag{lag}" for lag in LAGS),
    "prev_day_mean",
    "prev_day_min",
    "prev_day_max",
]

FEATURES = [
    *BASE_FEATURES,
    "price_7d_mean",
    "res_share_fc",
    "residual_load_fc_delta24",
    "residual_load_fc_day_mean",
    "residual_load_fc_day_min",
    "residual_load_fc_day_max",
    "residual_load_fc_rel_day",
    "residual_load_fc_day_delta",
]


def build(df: pl.DataFrame) -> pl.DataFrame:
    years = df["local"].dt.year().unique().to_list()
    holiday_dates = list(holidays.Germany(years=years))
    df = df.sort("time").with_columns(
        *(pl.col("price").shift(lag).alias(f"price_lag{lag}") for lag in LAGS),
        day=pl.col("local").dt.date(),
        hour=pl.col("local").dt.hour(),
        dow=pl.col("local").dt.weekday(),
        month=pl.col("local").dt.month(),
        holiday=pl.col("local").dt.date().is_in(holiday_dates).cast(pl.Int8),
        wind_fc=pl.col("wind_onshore_fc") + pl.col("wind_offshore_fc"),
    )
    daily = (
        df.group_by("day")
        .agg(
            price_mean=pl.col("price").mean(),
            price_min=pl.col("price").min(),
            price_max=pl.col("price").max(),
            residual_load_fc_day_mean=pl.col("residual_load_fc").mean(),
            residual_load_fc_day_min=pl.col("residual_load_fc").min(),
            residual_load_fc_day_max=pl.col("residual_load_fc").max(),
        )
        .sort("day")
        .with_columns(
            prev_day_mean=pl.col("price_mean").shift(1),
            prev_day_min=pl.col("price_min").shift(1),
            prev_day_max=pl.col("price_max").shift(1),
            price_7d_mean=pl.col("price_mean").rolling_mean(7).shift(1),
            residual_load_fc_day_delta=pl.col("residual_load_fc_day_mean").diff(),
        )
        .drop("price_mean", "price_min", "price_max")
    )
    return (
        df.join(daily, on="day", how="left")
        .sort("time")
        .with_columns(
            res_share_fc=(pl.col("wind_fc") + pl.col("solar_fc")) / pl.col("load_fc"),
            residual_load_fc_delta24=pl.col("residual_load_fc")
            - pl.col("residual_load_fc").shift(24),
            residual_load_fc_rel_day=pl.col("residual_load_fc")
            - pl.col("residual_load_fc_day_mean"),
        )
    )
