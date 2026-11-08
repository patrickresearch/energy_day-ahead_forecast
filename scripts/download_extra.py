from datetime import UTC, date, datetime, timedelta

from energy_forecast import macro, weather
from energy_forecast.data import MACRO, WEATHER

tomorrow = datetime.now(UTC).date() + timedelta(days=1)
sources = [
    (WEATHER, weather.download(date(2024, 1, 1), tomorrow), "time"),
    (MACRO, macro.download(date(2019, 1, 1)), "day"),
]
for path, df, key in sources:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    print(path.name, df.height, df[key].min(), df[key].max())
    print(df.null_count())
