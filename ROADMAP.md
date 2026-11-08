# Roadmap

Decisions: hourly resolution (SMARD aggregates 15-min prices since 2025-10), SMARD as source, UTC internally, Europe/Berlin for display only, uv + Python 3.12, weather later.

Leakage rule: features for delivery day D must be known at 12:00 D-1 (gate closure). Use TSO forecasts (load_fc, wind_*_fc, solar_fc), never actuals. Price lags >= 24h from D-1 auction results.

1. Data — `scripts/download_smard.py` → `data/raw/smard_hourly.parquet`. Done when: rows == expected_hours, no duplicates, nulls explained.
2. EDA — `notebooks/01_eda.ipynb`: price heatmap hour x month, renewable share, negative-price hours per year, summer/winter stacked mix, residual load vs price.
3. Baseline — seasonal naive (D-1, D-7), MAE/RMSE on one-year walk-forward backtest.
4. LightGBM — calendar, residual load forecast, price lags. Daily refit or rolling window.
5. Chronos-Bolt — zero-shot, with and without covariates.
6. SHAP — `notebooks/03_explain.ipynb`.
7. Streamlit app — period selection, mix + price, precomputed 24h forecast. Deploy on Streamlit Community Cloud.
8. README — hero heatmap, MAE table, negative hours per year.
9. Extensions — GitHub Actions daily update, battery LP (pulp), Open-Meteo weather.
