from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalConfig:
    z_entry: float = 1.5
    z_exit: float = 0.3


def zscore_mean_reversion_signal(feat: pd.DataFrame, cfg: SignalConfig | None = None) -> pd.Series:
    """
    Returns a continuous signal in [-1, 1]:
    +1 means buy/cover (too cheap), -1 means sell/short (too rich).
    """
    cfg = cfg or SignalConfig()
    z = feat["zscore"].astype(float)
    sig = pd.Series(0.0, index=feat.index)
    sig[z <= -cfg.z_entry] = 1.0
    sig[z >= cfg.z_entry] = -1.0
    # soften when close to mean
    near = z.abs() <= cfg.z_exit
    sig[near] = 0.0
    return sig


def momentum_signal(feat: pd.DataFrame) -> pd.Series:
    """
    Continuous momentum in [-1,1] based on ROC, slope, MA alignment.
    """
    roc = feat["roc"].astype(float).clip(-0.2, 0.2) / 0.2
    slope = feat["slope_20"].astype(float).clip(-1.5, 1.5) / 1.5
    ma = (feat["sma_fast"] > feat["sma_slow"]).astype(float) * 2 - 1
    sig = 0.45 * roc + 0.35 * slope + 0.20 * ma
    return sig.clip(-1.0, 1.0)


def wyckoff_signal(feat: pd.DataFrame) -> pd.Series:
    """
    +1: accumulation/compression breakout bias
    -1: distribution/compression breakdown bias
    """
    comp = feat["compression"].astype(float)
    breakout = feat["breakout_strength"].astype(float)
    vol_ratio = feat["vol_ratio"].astype(float)
    # compression signal: low range (comp < 1) + rising volume ratio
    acc = ((comp < 0.95) & (vol_ratio > 1.05)).astype(float)
    dist = ((comp < 0.95) & (vol_ratio > 1.05)).astype(float)  # symmetric placeholder
    # breakout strength pushes direction
    sig = acc * np.tanh(breakout.fillna(0.0)) - dist * np.tanh((-breakout).fillna(0.0))
    return sig.clip(-1.0, 1.0)


def smc_signal(feat: pd.DataFrame) -> pd.Series:
    """
    SMC-ish:
    - liquidity sweep reversal
    - structure break confirmation
    """
    sweep = feat["liq_sweep"].astype(float)
    bos = feat["structure_break"].astype(float)
    sig = 0.65 * sweep + 0.35 * bos
    return sig.clip(-1.0, 1.0)


def vol_risk_score(feat: pd.DataFrame) -> pd.Series:
    """
    Higher means riskier. Uses rolling std and ATR normalization.
    """
    v = feat["vol_std"].astype(float).fillna(0.0)
    # map typical daily std 0-5% to 0-1
    return (v / 0.05).clip(0.0, 2.0)

