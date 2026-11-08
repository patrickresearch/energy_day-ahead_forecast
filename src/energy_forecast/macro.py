from datetime import UTC, date, datetime

import polars as pl
import requests

URL = "https://query1.finance.yahoo.com/v8/finance/chart/{}"
TICKERS = {"co2": "CO2.L", "coal": "MTF=F"}


def _fetch(name: str, start: date) -> pl.DataFrame:
    params = {
        "period1": int(datetime.combine(start, datetime.min.time(), UTC).timestamp()),
        "period2": int(datetime.now(UTC).timestamp()),
        "interval": "1d",
    }
    response = requests.get(
        URL.format(TICKERS[name]), params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=30
    )
    response.raise_for_status()
    result = response.json()["chart"]["result"][0]
    return (
        pl.DataFrame(
            {"day": result["timestamp"], name: result["indicators"]["quote"][0]["close"]},
            schema={"day": pl.Int64, name: pl.Float64},
        )
        .with_columns(pl.from_epoch("day").dt.date())
        .drop_nulls()
        .unique("day", keep="last")
    )


def download(start: date) -> pl.DataFrame:
    frames = [_fetch(name, start) for name in TICKERS]
    out = frames[0]
    for frame in frames[1:]:
        out = out.join(frame, on="day", how="full", coalesce=True)
    return out.sort("day").with_columns(pl.exclude("day").forward_fill())
