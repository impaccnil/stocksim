from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    sma_fast: int = 20
    sma_slow: int = 50
    ema_fast: int = 20
    roc_n: int = 10
    atr_n: int = 14
    z_n: int = 20
    vol_n: int = 20
    volume_n: int = 20
    swing_lookback: int = 20


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _trend_slope(y: pd.Series, n: int) -> float:
    y = y.dropna().astype(float)
    if len(y) < n:
        return 0.0
    w = y.iloc[-n:].values
    x = np.arange(len(w), dtype=float)
    x = (x - x.mean()) / (x.std() + 1e-12)
    w = (w - w.mean()) / (w.std() + 1e-12)
    return float(np.polyfit(x, w, 1)[0])


def compute_features(df: pd.DataFrame, cfg: FeatureConfig | None = None) -> pd.DataFrame:
    """
    Input: OHLCV indexed by UTC timestamp with columns open/high/low/close/volume.
    Output: feature DataFrame aligned to df index (NaNs early).
    """
    cfg = cfg or FeatureConfig()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    vol = df["volume"].astype(float)

    ret1 = close.pct_change()
    sma_fast = close.rolling(cfg.sma_fast).mean()
    sma_slow = close.rolling(cfg.sma_slow).mean()
    ema_fast = _ema(close, cfg.ema_fast)

    roc = close.pct_change(cfg.roc_n)
    vol_std = ret1.rolling(cfg.vol_n).std(ddof=0)
    atr = _atr(df, cfg.atr_n)

    vol_avg = vol.rolling(cfg.volume_n).mean()
    vol_ratio = vol / (vol_avg + 1e-12)

    # Statistical: z-score vs SMA
    mean = close.rolling(cfg.z_n).mean()
    std = close.rolling(cfg.z_n).std(ddof=0)
    z = (close - mean) / (std + 1e-12)

    # Wyckoff-ish: compression & contraction
    hl_range = (high - low) / (close.abs() + 1e-12)
    compression = (hl_range.rolling(cfg.z_n).mean() / (hl_range.rolling(cfg.z_n * 2).mean() + 1e-12)).clip(0, 3)
    contraction = (hl_range.rolling(cfg.z_n).mean().pct_change()).fillna(0.0)
    breakout_strength = (close - close.rolling(cfg.z_n).max().shift(1)) / (atr + 1e-12)

    # SMC-ish: liquidity sweep + structure break
    swing_high = high.rolling(cfg.swing_lookback).max().shift(1)
    swing_low = low.rolling(cfg.swing_lookback).min().shift(1)
    wick_up = (high > swing_high) & (close < swing_high)  # swept above then closed back below
    wick_down = (low < swing_low) & (close > swing_low)  # swept below then closed back above
    liq_sweep = wick_down.astype(float) - wick_up.astype(float)  # +1 bullish sweep, -1 bearish sweep

    bos_up = (close > swing_high).astype(float)
    bos_down = (close < swing_low).astype(float)
    structure_break = bos_up - bos_down

    # Trend slope proxy (computed as rolling apply, small cost at 1d)
    slope_20 = close.rolling(cfg.sma_fast).apply(lambda s: _trend_slope(s, min(len(s), cfg.sma_fast)), raw=False)

    out = pd.DataFrame(
        {
            "ret1": ret1,
            "sma_fast": sma_fast,
            "sma_slow": sma_slow,
            "ema_fast": ema_fast,
            "roc": roc,
            "vol_std": vol_std,
            "atr": atr,
            "vol_ratio": vol_ratio,
            "zscore": z,
            "compression": compression,
            "contraction": contraction,
            "breakout_strength": breakout_strength,
            "liq_sweep": liq_sweep,
            "structure_break": structure_break,
            "slope_20": slope_20,
        }
    )
    return out

