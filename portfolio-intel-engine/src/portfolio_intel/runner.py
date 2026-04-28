from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from portfolio_intel.data_sources import (
    MockMacroDataProvider,
    MockMarketDataProvider,
    MockSentimentProvider,
    default_as_of,
)
from portfolio_intel.decision import DecisionEngine
from portfolio_intel.engines import EngineContext
from portfolio_intel.io import ensure_dir, load_portfolio
from portfolio_intel.models import DailyReport
from portfolio_intel.reporting import email_subject, render_email
from portfolio_intel.risk import PortfolioRiskEngine
from portfolio_intel.sim import SimulationEngine


def _market_regime_text(macro) -> str:
    # simple classification string
    if macro.rates_regime in {"falling"} and macro.fed_policy_bias in {"dovish"}:
        regime = "risk-on bull / liquidity easing"
    elif macro.rates_regime in {"rising", "sticky-high"} and macro.fed_policy_bias in {"hawkish", "restrictive"}:
        regime = "tightening / bearish transition"
    else:
        regime = "mixed / range-bound"
    return (
        f"{regime}. Rates: {macro.rates_regime}; Fed: {macro.fed_policy_bias}; "
        f"Inflation: {macro.inflation_trend}; Oil shock risk: {macro.oil_shock_risk}; "
        f"Geopolitics risk: {macro.geopolitics_risk}."
    )


def _portfolio_health_score(fragility_score: float, avg_decision_conf: float) -> float:
    # health is inverse fragility, modestly boosted by clarity (confidence) but capped
    return max(0.0, min(100.0, (100.0 - fragility_score) * 0.85 + avg_decision_conf * 0.15))


def run_daily(*, portfolio_path: Path, out_dir: Path, as_of: str | None = None) -> DailyReport:
    asof: date = default_as_of(as_of)
    state = load_portfolio(portfolio_path)

    tickers = state.tickers()
    market = MockMarketDataProvider().get_snapshot(tickers, asof)
    macro = MockMacroDataProvider().get_snapshot(asof)
    sentiment = MockSentimentProvider().get_snapshot(tickers, asof)

    ctx = EngineContext(market=market, macro=macro, sentiment=sentiment)
    dec_engine = DecisionEngine()
    decisions = [dec_engine.decide(t, ctx) for t in tickers]

    # rank for actions
    trims = [d for d in decisions if d.action in {"TRIM", "EXIT"}]
    accumulates = [d for d in decisions if d.action == "BUY"]

    trims = sorted(trims, key=lambda d: (d.action != "EXIT", -d.confidence))
    accumulates = sorted(accumulates, key=lambda d: -d.confidence)

    risk = PortfolioRiskEngine().analyze(state, market)
    sim_summary = SimulationEngine().simulate_one_day_delta(state, decisions, market)

    avg_conf = sum(d.confidence for d in decisions) / max(len(decisions), 1)
    health = _portfolio_health_score(risk.fragility_score, avg_conf)

    top_risks = (risk.warnings or [])[:5]
    top_opps = [f"{d.ticker}: {d.reasoning}" for d in accumulates[:5]]

    subj = email_subject(asof)
    report = DailyReport(
        as_of=asof,
        market_regime=_market_regime_text(macro),
        portfolio_health_score=health,
        top_risks=top_risks,
        top_opportunities=top_opps,
        trims=trims,
        accumulates=accumulates,
        decisions=decisions,
        risk=risk,
        sim_vs_real_delta_summary=sim_summary,
        email_subject=subj,
        email_body="",
    )
    subject, body = render_email(report, to_email="impacc.nil@gmail.com")
    report.email_subject = subject
    report.email_body = body

    ensure_dir(out_dir)
    out_md = out_dir / f"report_{asof.isoformat()}.txt"
    out_json = out_dir / f"report_{asof.isoformat()}.json"
    out_md.write_text(body, encoding="utf-8")
    out_json.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    return report

