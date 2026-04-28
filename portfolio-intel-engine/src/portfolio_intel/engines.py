from __future__ import annotations

from dataclasses import dataclass

from portfolio_intel.data_sources import MacroSnapshot, MarketSnapshot, SentimentSnapshot
from portfolio_intel.models import EngineScore
from portfolio_intel.scoring import clamp_0_100


@dataclass(frozen=True)
class EngineContext:
    market: MarketSnapshot
    macro: MacroSnapshot
    sentiment: SentimentSnapshot


class TechnicalEngine:
    """
    Lightweight technical proxy:
    - uses 20d return (trend proxy)
    - penalizes high 20d vol (fragility proxy)
    """

    def score(self, ticker: str, ctx: EngineContext) -> EngineScore:
        r = ctx.market.returns_20d.get(ticker, 0.0)
        v = ctx.market.vol_20d.get(ticker, 0.25)
        # map return to [0,100], vol as penalty
        trend = clamp_0_100(50.0 + 120.0 * r)
        vol_penalty = clamp_0_100((v - 0.20) * 120.0)  # >20% vol starts to hurt
        val = clamp_0_100(trend - 0.6 * vol_penalty)
        summary = f"20D return {r:+.1%}, 20D vol {v:.1%} (trend/vol proxy)."
        return EngineScore(value=val, summary=summary)


class FundamentalEngine:
    """
    Placeholder fundamental model (no live financial statements by default).
    Uses sector proxy as a valuation/growth roughness input; real provider can
    replace with revenue/margins/valuation metrics.
    """

    def score(self, ticker: str, ctx: EngineContext) -> EngineScore:
        cluster = ctx.market.sector_proxy.get(ticker, "Other")
        # crude priors: defensive sectors slightly higher baseline
        base = {
            "ETF": 55,
            "Semis/Hardware": 52,
            "Software/Internet": 50,
            "Energy": 54,
            "Healthcare/Biotech": 53,
            "Power/Nuclear": 51,
            "Other": 50,
        }.get(cluster, 50)
        # small adjustment from returns (momentum sometimes coincides with fundamentals)
        r = ctx.market.returns_20d.get(ticker, 0.0)
        val = clamp_0_100(base + 35.0 * r)
        summary = f"Baseline fundamental prior for {cluster}; adjusted by 20D return proxy."
        return EngineScore(value=val, summary=summary)


class MacroRegimeEngine:
    """
    Scores how supportive the current macro snapshot is for the ticker's cluster.
    """

    def score(self, ticker: str, ctx: EngineContext) -> EngineScore:
        cluster = ctx.market.sector_proxy.get(ticker, "Other")
        rates = ctx.macro.rates_regime

        # simplistic mapping
        score = 50.0
        if rates in {"rising", "sticky-high"} and cluster in {"Semis/Hardware", "Software/Internet"}:
            score -= 8.0
        if rates in {"falling"} and cluster in {"Semis/Hardware", "Software/Internet"}:
            score += 8.0
        if cluster in {"Energy"}:
            if ctx.macro.oil_shock_risk == "high":
                score += 6.0
            if ctx.macro.geopolitics_risk == "high":
                score += 4.0
        if cluster in {"Healthcare/Biotech"} and ctx.macro.fed_policy_bias in {"restrictive", "hawkish"}:
            score -= 3.0
        if cluster in {"Power/Nuclear"} and ctx.macro.rates_regime in {"falling"}:
            score += 4.0

        val = clamp_0_100(score)
        summary = (
            f"Rates: {ctx.macro.rates_regime}; Fed: {ctx.macro.fed_policy_bias}; "
            f"Inflation: {ctx.macro.inflation_trend}; Oil shock: {ctx.macro.oil_shock_risk}; "
            f"Geopolitics: {ctx.macro.geopolitics_risk}."
        )
        return EngineScore(value=val, summary=summary)


class SentimentEngine:
    def score(self, ticker: str, ctx: EngineContext) -> EngineScore:
        b = ctx.sentiment.bias.get(ticker, "neutral")
        val = {"bearish": 35.0, "neutral": 50.0, "bullish": 65.0}.get(b, 50.0)
        summary = f"{b.capitalize()} bias. {ctx.sentiment.narrative.get(ticker, '')}".strip()
        return EngineScore(value=val, summary=summary)

