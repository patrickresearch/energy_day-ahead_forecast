import lightgbm as lgb
import polars as pl

from energy_forecast.forecast import DEFAULT, Config


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
