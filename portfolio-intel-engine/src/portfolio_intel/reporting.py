from __future__ import annotations

from datetime import date

from portfolio_intel.models import DailyReport, StockDecision


def _bullet(lines: list[str]) -> str:
    if not lines:
        return "- (none)\n"
    return "".join([f"- {l}\n" for l in lines])


def _dec_line(d: StockDecision) -> str:
    return f"{d.ticker}: {d.action} (conf {d.confidence:.0f}, risk {d.risk_level}) — {d.reasoning}"


def render_email(report: DailyReport, *, to_email: str) -> tuple[str, str]:
    subject = report.email_subject
    body = f"""To: {to_email}
Subject: {subject}

PORTFOLIO INTELLIGENCE REPORT — {report.as_of.isoformat()}

## Market regime
{report.market_regime}

## Portfolio health
Health score: {report.portfolio_health_score:.0f}/100
Fragility score: {report.risk.fragility_score:.0f}/100

## Top risks
{_bullet(report.top_risks)}

## Top opportunities
{_bullet(report.top_opportunities)}

## Stocks to trim immediately
{_bullet([_dec_line(d) for d in report.trims[:8]])}

## Stocks to accumulate
{_bullet([_dec_line(d) for d in report.accumulates[:8]])}

## Risk warnings
{_bullet(report.risk.warnings[:10])}

## Concentrations
Top positions: {", ".join(report.risk.top_concentrations) if report.risk.top_concentrations else "(n/a)"}
Cluster exposures: {", ".join(report.risk.sector_overexposures) if report.risk.sector_overexposures else "(n/a)"}

## Macro alert summary
{report.market_regime}

## Simulated vs real performance delta
{report.sim_vs_real_delta_summary}

---
Notes:
- This system is analysis-only and **does not execute trades**.
- All outputs are probabilistic; uncertainty remains high without live fundamentals/news/historical backtests.
"""
    return subject, body


def email_subject(as_of: date) -> str:
    return f"PORTFOLIO INTELLIGENCE REPORT – {as_of.isoformat()}"

