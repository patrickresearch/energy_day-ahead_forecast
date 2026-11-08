from dataclasses import asdict

import polars as pl
import wandb

from energy_forecast.forecast import Config

ENTITY = "kahala_patrick-stanford-university"
PROJECT = "energy-forecast"


def start(job_type: str, name: str, config: dict, enabled: bool = True) -> wandb.Run:
    return wandb.init(
        entity=ENTITY,
        project=PROJECT,
        job_type=job_type,
        name=name,
        config=config,
        mode="online" if enabled else "disabled",
    )


def configs(cfgs: list[Config]) -> dict:
    return {cfg.name: asdict(cfg) for cfg in cfgs}


def table(df: pl.DataFrame) -> wandb.Table:
    return wandb.Table(columns=df.columns, data=df.rows())


def log_scores(run: wandb.Run, scores: pl.DataFrame) -> None:
    run.log({"scores": table(scores)})
    for row in scores.iter_rows(named=True):
        for metric in ("mae", "rmse", "bias"):
            run.summary[f"{row['model']}/{metric}"] = row[metric]
