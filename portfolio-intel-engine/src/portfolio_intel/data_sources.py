from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Protocol

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketSnapshot:
    as_of: date
    prices: dict[str, float]
    returns_20d: dict[str, float]
    vol_20d: dict[str, float]
    sector_proxy: dict[str, str]  # ticker -> sector/cluster label


@dataclass(frozen=True)
class MacroSnapshot:
    as_of: date
    fed_policy_bias: str
    inflation_trend: str
    rates_regime: str
    oil_shock_risk: str
    geopolitics_risk: str


@dataclass(frozen=True)
class SentimentSnapshot:
    as_of: date
    bias: dict[str, str]  # ticker -> bullish/neutral/bearish
    narrative: dict[str, str]  # ticker -> short note


class MarketDataProvider(Protocol):
    def get_snapshot(self, tickers: list[str], as_of: date) -> MarketSnapshot: ...


class MacroDataProvider(Protocol):
    def get_snapshot(self, as_of: date) -> MacroSnapshot: ...


class SentimentProvider(Protocol):
    def get_snapshot(self, tickers: list[str], as_of: date) -> SentimentSnapshot: ...


def _stable_hash(s: str) -> int:
    return sum((i + 1) * ord(c) for i, c in enumerate(s))


class MockMarketDataProvider:
    """
    Offline provider: deterministic pseudo-market data so the system runs immediately.
    Replace with real providers later (e.g., yfinance, paid feeds).
    """

    def get_snapshot(self, tickers: list[str], as_of: date) -> MarketSnapshot:
        prices: dict[str, float] = {}
        returns_20d: dict[str, float] = {}
        vol_20d: dict[str, float] = {}

        for t in tickers:
            h = _stable_hash(t)
            base = (h % 500) + 10
            drift = ((h % 31) - 15) / 100.0
            prices[t] = float(base + drift * 10.0)
            returns_20d[t] = float(np.clip(drift * 2.5, -0.35, 0.35))
            vol_20d[t] = float(np.clip(0.15 + (h % 17) / 100.0, 0.12, 0.55))

        # crude clustering — user can replace with GICS mapping later
        def cluster(t: str) -> str:
            u = t.upper()
            if u in {"QQQ", "SMH"}:
                return "ETF"
            if u in {"NVDA", "AVGO", "MU", "MRVL", "AMKR", "CLS", "AAOI", "GLW", "VRT", "NVTS"}:
                return "Semis/Hardware"
            if u in {"CRWD", "PLTR", "ORCL", "GOOGL", "APP", "TTD", "UBER", "NFLX", "SOFI", "HIMS"}:
                return "Software/Internet"
            if u in {"CVX", "SLB", "TOTDY"}:
                return "Energy"
            if u in {"PFE", "SNY", "ABCL", "UNH"}:
                return "Healthcare/Biotech"
            if u in {"OKLO", "CEG", "BE"}:
                return "Power/Nuclear"
            return "Other"

        sector_proxy = {t: cluster(t) for t in tickers}
        return MarketSnapshot(
            as_of=as_of,
            prices=prices,
            returns_20d=returns_20d,
            vol_20d=vol_20d,
            sector_proxy=sector_proxy,
        )


class MockMacroDataProvider:
    def get_snapshot(self, as_of: date) -> MacroSnapshot:
        # Deterministic "regime" based on date for repeatability
        day = as_of.toordinal() % 4
        rates_regime = ["falling", "rising", "sticky-high", "volatile"][day]
        fed_policy_bias = ["dovish", "hawkish", "data-dependent", "restrictive"][day]
        inflation_trend = ["cooling", "re-accelerating", "sticky", "disinflation"][day]
        oil_shock_risk = ["low", "medium", "high", "medium"][day]
        geopolitics_risk = ["medium", "high", "medium", "low"][day]
        return MacroSnapshot(
            as_of=as_of,
            fed_policy_bias=fed_policy_bias,
            inflation_trend=inflation_trend,
            rates_regime=rates_regime,
            oil_shock_risk=oil_shock_risk,
            geopolitics_risk=geopolitics_risk,
        )


class MockSentimentProvider:
    def get_snapshot(self, tickers: list[str], as_of: date) -> SentimentSnapshot:
        bias: dict[str, str] = {}
        narrative: dict[str, str] = {}
        for t in tickers:
            h = _stable_hash(t) % 3
            b = ["bearish", "neutral", "bullish"][h]
            bias[t] = b
            narrative[t] = {
                "bearish": "Narrative weakening; higher headline sensitivity.",
                "neutral": "Mixed narratives; wait for confirmation.",
                "bullish": "Narrative strong; supportive flows possible.",
            }[b]
        return SentimentSnapshot(as_of=as_of, bias=bias, narrative=narrative)


def default_as_of(as_of: str | None) -> date:
    if as_of:
        return date.fromisoformat(as_of)
    return (pd.Timestamp.utcnow().date())  # type: ignore[no-any-return]

