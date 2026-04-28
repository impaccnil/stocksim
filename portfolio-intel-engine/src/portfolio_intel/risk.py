from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from portfolio_intel.data_sources import MarketSnapshot
from portfolio_intel.models import PortfolioRiskReport, PortfolioState
from portfolio_intel.scoring import clamp_0_100


@dataclass(frozen=True)
class RiskConfig:
    max_single_name_weight: float = 0.18
    max_cluster_weight: float = 0.45


def _portfolio_market_value(state: PortfolioState, mkt: MarketSnapshot) -> float:
    v = float(state.cash)
    for p in state.positions:
        px = mkt.prices.get(p.ticker.upper())
        if px is None:
            continue
        v += float(p.shares) * float(px)
    return max(v, 1.0)


def _weights_by_ticker(state: PortfolioState, mkt: MarketSnapshot) -> dict[str, float]:
    total = _portfolio_market_value(state, mkt)
    w: dict[str, float] = {}
    for p in state.positions:
        t = p.ticker.upper()
        px = mkt.prices.get(t)
        if px is None:
            continue
        w[t] = (float(p.shares) * float(px)) / total
    return w


def _weights_by_cluster(w_ticker: dict[str, float], mkt: MarketSnapshot) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for t, w in w_ticker.items():
        out[mkt.sector_proxy.get(t, "Other")] += w
    return dict(out)


def _correlation_proxy_notes(w_ticker: dict[str, float], mkt: MarketSnapshot) -> list[str]:
    """
    Without a price history provider, approximate correlation risk by:
    - identifying high-weight tickers that share the same cluster label
    - flagging high-vol clusters with many constituents
    """
    cluster_members: dict[str, list[str]] = defaultdict(list)
    for t in w_ticker:
        cluster_members[mkt.sector_proxy.get(t, "Other")].append(t)

    notes: list[str] = []
    for cl, members in sorted(cluster_members.items(), key=lambda kv: -len(kv[1])):
        if len(members) < 3:
            continue
        vols = [mkt.vol_20d.get(t, 0.25) for t in members]
        avg_vol = float(np.mean(vols)) if vols else 0.25
        if avg_vol >= 0.28:
            notes.append(
                f"Correlation proxy: {cl} has {len(members)} names with avg 20D vol {avg_vol:.1%} (clustered risk)."
            )
    return notes[:5]


class PortfolioRiskEngine:
    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()

    def analyze(self, state: PortfolioState, mkt: MarketSnapshot) -> PortfolioRiskReport:
        w_ticker = _weights_by_ticker(state, mkt)
        w_cluster = _weights_by_cluster(w_ticker, mkt)

        warnings: list[str] = []
        top_conc = sorted(w_ticker.items(), key=lambda kv: -kv[1])[:5]
        cluster_conc = sorted(w_cluster.items(), key=lambda kv: -kv[1])[:5]

        for t, w in top_conc:
            if w >= self.config.max_single_name_weight:
                warnings.append(f"Single-name concentration: {t} at {w:.1%} of portfolio value.")

        for cl, w in cluster_conc:
            if w >= self.config.max_cluster_weight:
                warnings.append(f"Cluster overexposure: {cl} at {w:.1%} of portfolio value.")

        # high-vol clustering
        high_vol = [t for t, w in w_ticker.items() if mkt.vol_20d.get(t, 0.25) >= 0.35 and w >= 0.04]
        if len(high_vol) >= 3:
            warnings.append(
                f"High-vol clustering: {len(high_vol)} meaningful weights in ≥35% 20D vol names ({', '.join(high_vol[:6])})."
            )

        corr_notes = _correlation_proxy_notes(w_ticker, mkt)
        if corr_notes:
            warnings.append("Hidden correlation risk likely elevated due to cluster overlap.")

        # fragility score: concentration + vol clustering + cluster overlap penalty
        w_max = max(w_ticker.values(), default=0.0)
        top_cluster = max(w_cluster.values(), default=0.0)
        vol_penalty = float(np.mean([mkt.vol_20d.get(t, 0.25) for t in w_ticker])) if w_ticker else 0.25

        fragility = clamp_0_100(
            20.0
            + 220.0 * max(0.0, w_max - 0.10)
            + 120.0 * max(0.0, top_cluster - 0.30)
            + 80.0 * max(0.0, vol_penalty - 0.22)
            + (10.0 if corr_notes else 0.0)
        )

        sector_over = [f"{cl}: {w:.1%}" for cl, w in cluster_conc if w >= 0.25]
        top_conc_fmt = [f"{t}: {w:.1%}" for t, w in top_conc]

        return PortfolioRiskReport(
            fragility_score=fragility,
            warnings=warnings[:10],
            top_concentrations=top_conc_fmt,
            sector_overexposures=sector_over[:5],
            correlation_notes=corr_notes,
        )

