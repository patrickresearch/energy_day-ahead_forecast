from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import polars as pl
import requests

BASE = "https://www.smard.de/app/chart_data"

SERIES = {
    "price": (4169, "DE-LU"),
    "load": (410, "DE"),
    "lignite": (1223, "DE"),
    "nuclear": (1224, "DE"),
    "hard_coal": (4069, "DE"),
    "gas": (4071, "DE"),
    "other_conventional": (1227, "DE"),
    "pumped_storage": (4070, "DE"),
    "wind_onshore": (4067, "DE"),
    "wind_offshore": (1225, "DE"),
    "solar": (4068, "DE"),
    "biomass": (4066, "DE"),
    "hydro": (1226, "DE"),
    "other_renewable": (1228, "DE"),
    "load_fc": (411, "DE"),
    "wind_onshore_fc": (123, "DE"),
    "wind_offshore_fc": (3791, "DE"),
    "solar_fc": (125, "DE"),
}


def _get(session: requests.Session, url: str) -> dict:
    response = session.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_series(session: requests.Session, name: str, start: datetime) -> pl.DataFrame:
    filter_id, region = SERIES[name]
    root = f"{BASE}/{filter_id}/{region}"
    cutoff = (start - timedelta(days=7)).timestamp() * 1000
    stamps = [t for t in _get(session, f"{root}/index_hour.json")["timestamps"] if t >= cutoff]
    urls = [f"{root}/{filter_id}_{region}_hour_{t}.json" for t in stamps]
    with ThreadPoolExecutor(8) as pool:
        chunks = pool.map(lambda u: _get(session, u)["series"], urls)
        rows = [row for chunk in chunks for row in chunk]
    return pl.DataFrame(
        rows, schema={"ms": pl.Int64, "value": pl.Float64}, orient="row", strict=False
    ).with_columns(series=pl.lit(name))


def download(start: datetime) -> pl.DataFrame:
    with requests.Session() as session:
        long = pl.concat([fetch_series(session, name, start) for name in SERIES])
    return (
        long.unique(["ms", "series"])
        .pivot(on="series", index="ms", values="value")
        .select(
            pl.from_epoch("ms", time_unit="ms").dt.replace_time_zone("UTC").alias("time"),
            *SERIES,
        )
        .filter(pl.col("time") >= start)
        .filter(pl.any_horizontal(pl.exclude("time").is_not_null()))
        .sort("time")
    )
