from datetime import date

import polars as pl
import requests

URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

LOCATIONS = [
    (54.4, 9.2),
    (53.3, 7.9),
    (52.6, 13.0),
    (51.4, 7.2),
    (51.3, 12.4),
    (50.1, 8.7),
    (48.8, 9.2),
    (48.4, 11.1),
]

VARIABLES = {
    "temperature_2m_previous_day2": "temp_wx",
    "wind_speed_100m_previous_day2": "wind_wx",
    "shortwave_radiation_previous_day2": "solar_wx",
}


def _fetch(start: date, end: date) -> pl.DataFrame:
    params = {
        "latitude": ",".join(str(lat) for lat, _ in LOCATIONS),
        "longitude": ",".join(str(lon) for _, lon in LOCATIONS),
        "hourly": ",".join(VARIABLES),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "timezone": "GMT",
    }
    response = requests.get(URL, params=params, timeout=120)
    response.raise_for_status()
    schema = {"time": pl.String, **dict.fromkeys(VARIABLES, pl.Float64)}
    return pl.concat(pl.DataFrame(site["hourly"], schema=schema) for site in response.json())


def download(start: date, end: date) -> pl.DataFrame:
    years = range(start.year, end.year + 1)
    chunks = [_fetch(max(start, date(y, 1, 1)), min(end, date(y, 12, 31))) for y in years]
    return (
        pl.concat(chunks)
        .group_by("time")
        .agg(pl.col(src).mean().alias(dst) for src, dst in VARIABLES.items())
        .with_columns(pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M", time_zone="UTC"))
        .sort("time")
    )
