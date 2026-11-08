from datetime import date
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/smard_hourly.parquet"
WEATHER = ROOT / "data/raw/weather_hourly.parquet"
MACRO = ROOT / "data/raw/macro_daily.parquet"
TZ = "Europe/Berlin"
VAL = (date(2024, 11, 1), date(2025, 10, 31))
TEST = (date(2025, 11, 1), date(2026, 10, 31))

MIX = {
    "Wind": ["wind_onshore", "wind_offshore"],
    "Solar": ["solar"],
    "Biomass": ["biomass"],
    "Hydro & other RE": ["hydro", "other_renewable"],
    "Gas": ["gas"],
    "Hard coal": ["hard_coal"],
    "Lignite": ["lignite"],
    "Other conventional": ["nuclear", "other_conventional", "pumped_storage"],
}
RENEWABLE = ["Wind", "Solar", "Biomass", "Hydro & other RE"]


def _total(cols: list[str]) -> pl.Expr:
    return sum(pl.col(c) for c in cols)


def load(path: Path = RAW) -> pl.DataFrame:
    return (
        pl.read_parquet(path)
        .with_columns(pl.col("nuclear").fill_null(0))
        .with_columns(
            *(_total(cols).alias(name) for name, cols in MIX.items()),
            residual_load=pl.col("load") - _total(["wind_onshore", "wind_offshore", "solar"]),
            residual_load_fc=pl.col("load_fc")
            - _total(["wind_onshore_fc", "wind_offshore_fc", "solar_fc"]),
            local=pl.col("time").dt.convert_time_zone(TZ).dt.replace_time_zone(None),
        )
    )
