import math
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.nn import functional as F

from energy_forecast.data import TEST

ENCODER = 168
HORIZON = 25
QUANTILES = (0.1, 0.5, 0.9)
CALENDAR = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "holiday"]
TSO = ["load_fc", "wind_fc", "solar_fc", "residual_load_fc"]
ENGINEERED = [
    "res_share_fc",
    "residual_load_fc_rel_day",
    "residual_load_fc_day_delta",
    "residual_load_fc_delta24",
    "residual_load_fc_day_max",
]
EXTERNAL = ["temp_wx", "solar_wx", "co2", "coal"]


@dataclass(frozen=True)
class Config:
    name: str
    known: tuple[str, ...]
    window_days: int = 2 * 365
    hidden: int = 32
    heads: int = 4
    dropout: float = 0.1
    lr: float = 1e-3
    batch: int = 256
    samples: int = 4096
    epochs: int = 30
    patience: int = 3
    val_days: int = 30


CONFIGS = {
    cfg.name: cfg
    for cfg in [
        Config("tft", tuple(CALENDAR)),
        Config("tft_cov", (*CALENDAR, *TSO)),
        Config("tft_eng", (*CALENDAR, *TSO, *ENGINEERED)),
        Config("tft_all", (*CALENDAR, *TSO, *ENGINEERED, *EXTERNAL)),
    ]
}


class GRN(nn.Module):
    def __init__(self, d_in: int, d: int, d_out: int | None = None, dropout: float = 0.1):
        super().__init__()
        d_out = d_out or d
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.fc1 = nn.Linear(d_in, d)
        self.fc2 = nn.Linear(d, d)
        self.gate = nn.Linear(d, 2 * d_out)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.fc2(F.elu(self.fc1(x))))
        return self.norm(self.skip(x) + F.glu(self.gate(h)))


class GateAddNorm(nn.Module):
    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.gate = nn.Linear(d, 2 * d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.norm(residual + F.glu(self.gate(self.drop(x))))


class VariableSelection(nn.Module):
    def __init__(self, n: int, d: int, dropout: float):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n, d) * 0.1)
        self.bias = nn.Parameter(torch.zeros(n, d))
        self.select = GRN(n * d, d, n, dropout)
        self.fc1 = nn.Parameter(torch.randn(n, d, d) / math.sqrt(d))
        self.b1 = nn.Parameter(torch.zeros(n, d))
        self.fc2 = nn.Parameter(torch.randn(n, d, d) / math.sqrt(d))
        self.b2 = nn.Parameter(torch.zeros(n, d))
        self.gate = nn.Parameter(torch.randn(n, d, 2 * d) / math.sqrt(d))
        self.bg = nn.Parameter(torch.zeros(n, 2 * d))
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e = x.unsqueeze(-1) * self.weight + self.bias
        w = torch.softmax(self.select(e.flatten(-2)), -1)
        h = F.elu(torch.einsum("btnd,nde->btne", e, self.fc1) + self.b1)
        h = self.drop(torch.einsum("btnd,nde->btne", h, self.fc2) + self.b2)
        h = self.norm(e + F.glu(torch.einsum("btnd,nde->btne", h, self.gate) + self.bg))
        return (w.unsqueeze(-1) * h).sum(-2), w


class InterpretableAttention(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.heads, self.dh = heads, d // heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, self.dh)
        self.out = nn.Linear(self.dh, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.heads, self.dh).transpose(1, 2)
        k = self.k(x).view(b, t, self.heads, self.dh).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.dh)
        mask = torch.ones(t, t, dtype=torch.bool, device=x.device).triu(1)
        a = torch.softmax(scores.masked_fill(mask, float("-inf")), -1).mean(1)
        return self.out(self.drop(a) @ self.v(x)), a


class TFT(nn.Module):
    def __init__(self, n_known: int, d: int, heads: int, dropout: float):
        super().__init__()
        self.past = VariableSelection(n_known + 1, d, dropout)
        self.future = VariableSelection(n_known, d, dropout)
        self.encoder = nn.LSTM(d, d, batch_first=True)
        self.decoder = nn.LSTM(d, d, batch_first=True)
        self.post_lstm = GateAddNorm(d, dropout)
        self.enrich = GRN(d, d, dropout=dropout)
        self.attention = InterpretableAttention(d, heads, dropout)
        self.post_attention = GateAddNorm(d, dropout)
        self.ff = GRN(d, d, dropout=dropout)
        self.post_ff = GateAddNorm(d, dropout)
        self.head = nn.Linear(d, len(QUANTILES))

    def forward(self, past: torch.Tensor, future: torch.Tensor):
        p, past_w = self.past(past)
        f, future_w = self.future(future)
        enc, state = self.encoder(p)
        dec, _ = self.decoder(f, state)
        temporal = self.post_lstm(torch.cat([enc, dec], 1), torch.cat([p, f], 1))
        x = self.enrich(temporal)
        a, attention = self.attention(x)
        x = self.post_ff(self.ff(self.post_attention(a, x)), temporal)
        return self.head(x[:, -future.shape[1] :]), past_w, future_w, attention


def _inputs(df: pl.DataFrame) -> pl.DataFrame:
    hour = 2 * math.pi * pl.col("hour") / 24
    dow = 2 * math.pi * (pl.col("dow") - 1) / 7
    return df.sort("time").with_columns(
        hour_sin=hour.sin(), hour_cos=hour.cos(), dow_sin=dow.sin(), dow_cos=dow.cos()
    )


class Data:
    def __init__(self, df: pl.DataFrame, cfg: Config, device: str):
        known = df.select(cfg.known).to_numpy().astype(np.float32)
        self.raw_price = df["price"].to_numpy().astype(np.float32)
        self.raw_known = np.vstack([known, np.full((HORIZON, known.shape[1]), np.nan, np.float32)])
        price = np.concatenate([self.raw_price, np.full(HORIZON, np.nan, np.float32)])
        self.price = torch.tensor(price, device=device)
        self.missing = np.concatenate([[0], np.cumsum(np.isnan(price))])
        self.device = device

    def normalize(self, lo: int, cut: int) -> None:
        train = self.raw_known[lo:cut]
        mean, std = np.nanmean(train, 0), np.nanstd(train, 0) + 1e-6
        known = np.nan_to_num((self.raw_known - mean) / std)
        self.known = torch.tensor(known, dtype=torch.float32, device=self.device)
        self.scale = float(np.nanstd(self.raw_price[lo:cut]))

    def starts(self, lo: int, cut: int) -> torch.Tensor:
        s = np.arange(lo + ENCODER, cut - HORIZON + 1)
        valid = self.missing[s + HORIZON] - self.missing[s - ENCODER] == 0
        return torch.tensor(s[valid], device=self.device)

    def batch(self, starts: torch.Tensor):
        past_idx = starts[:, None] + torch.arange(-ENCODER, 0, device=self.device)
        future_idx = starts[:, None] + torch.arange(HORIZON, device=self.device)
        y_past = self.price[past_idx]
        center = y_past[:, -24:].mean(1, keepdim=True)
        past = torch.cat([((y_past - center) / self.scale).unsqueeze(-1), self.known[past_idx]], -1)
        target = (self.price[future_idx] - center) / self.scale
        return past, self.known[future_idx], center, target


def _loss(net: TFT, data: Data, starts: torch.Tensor) -> torch.Tensor:
    past, future, _, target = data.batch(starts)
    out = net(past, future)[0]
    q = torch.tensor(QUANTILES, device=out.device)
    err = target.unsqueeze(-1) - out
    return torch.maximum(q * err, (q - 1) * err).mean()


def fit(data: Data, lo: int, cut: int, cfg: Config) -> tuple[TFT, int, float]:
    torch.manual_seed(42)
    net = TFT(len(cfg.known), cfg.hidden, cfg.heads, cfg.dropout).to(data.device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    starts = data.starts(lo, cut)
    train, val = starts[: -cfg.val_days * 24], starts[-cfg.val_days * 24 :]
    best, best_epoch, state, wait = float("inf"), 0, None, 0
    for epoch in range(1, cfg.epochs + 1):
        net.train()
        for idx in train[torch.randperm(len(train), device=data.device)[: cfg.samples]].split(
            cfg.batch
        ):
            loss = _loss(net, data, idx)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            score = float(torch.stack([_loss(net, data, i) for i in val.split(cfg.batch)]).mean())
        if score < best:
            best, best_epoch, state, wait = score, epoch, deepcopy(net.state_dict()), 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    net.load_state_dict(state)
    return net, best_epoch, best


def backtest(
    df: pl.DataFrame, cfg: Config, period: tuple[date, date] = TEST, device: str = "mps"
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    df = _inputs(df).with_row_index("row")
    data = Data(df, cfg, device)
    days = df.group_by("day").agg(pl.col("row").min(), n=pl.len()).sort("day")
    months = pl.date_range(period[0], period[1], "1mo", eager=True).to_list()
    preds, past_w, future_w, attention, count = [], 0, 0, 0, 0
    for start in months:
        end = min(_month_end(start), period[1])
        lo = days.filter(pl.col("day") >= start - timedelta(days=cfg.window_days))["row"][0]
        cut = days.filter(pl.col("day") == start)["row"][0]
        data.normalize(lo, cut)
        net, epochs, val = fit(data, lo, cut, cfg)
        month = days.filter(pl.col("day").is_between(start, end))
        starts = torch.tensor(month["row"].to_numpy().astype(np.int64), device=device)
        net.eval()
        with torch.no_grad():
            past, future, center, _ = data.batch(starts)
            out, pw, fw, att = net(past, future)
        quantiles = (center.unsqueeze(-1) + out * data.scale).cpu().numpy().astype(np.float64)
        for row, n, q in zip(month["row"], month["n"], quantiles, strict=True):
            preds.append(
                pl.DataFrame(
                    {
                        "time": df["time"][row : row + n],
                        "pred": q[:n, 1],
                        "q10": q[:n, 0],
                        "q90": q[:n, 2],
                    }
                )
            )
        past_w = past_w + pw.sum((0, 1)).cpu().numpy()
        future_w = future_w + fw.sum((0, 1)).cpu().numpy()
        attention = attention + att[:, ENCODER:].sum((0, 1)).cpu().numpy()
        count += len(starts)
        print(f"{cfg.name} {start:%Y-%m}: best_epoch={epochs} val={val:.4f}", flush=True)
    preds = pl.concat(preds).with_columns(model=pl.lit(cfg.name))
    vsn = pl.DataFrame(
        {
            "variable": ["price", *cfg.known],
            "past": past_w / (count * ENCODER),
            "future": [None, *(future_w / (count * HORIZON))],
        }
    ).with_columns(model=pl.lit(cfg.name))
    lags = pl.DataFrame(
        {"lag": np.arange(-ENCODER, HORIZON), "weight": attention / (count * HORIZON)}
    ).with_columns(model=pl.lit(cfg.name))
    return preds, vsn, lags


def _month_end(start: date) -> date:
    return (start.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
