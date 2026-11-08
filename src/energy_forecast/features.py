import holidays
import polars as pl

LAGS = [24, 48, 168]

FEATURES = [
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


def build(df: pl.DataFrame) -> pl.DataFrame:
    years = df["local"].dt.year().unique().to_list()
    holiday_dates = list(holidays.Germany(years=years))
    prev_day = (
        df.group_by(day=pl.col("local").dt.date())
        .agg(
            prev_day_mean=pl.col("price").mean(),
            prev_day_min=pl.col("price").min(),
            prev_day_max=pl.col("price").max(),
        )
        .with_columns(pl.col("day").dt.offset_by("1d"))
    )
    return (
        df.sort("time")
        .with_columns(
            *(pl.col("price").shift(lag).alias(f"price_lag{lag}") for lag in LAGS),
            day=pl.col("local").dt.date(),
            hour=pl.col("local").dt.hour(),
            dow=pl.col("local").dt.weekday(),
            month=pl.col("local").dt.month(),
            holiday=pl.col("local").dt.date().is_in(holiday_dates).cast(pl.Int8),
            wind_fc=pl.col("wind_onshore_fc") + pl.col("wind_offshore_fc"),
        )
        .join(prev_day, on="day", how="left")
    )
