from __future__ import annotations

from dataclasses import dataclass

from portfolio_intel.data_sources import MarketSnapshot
from portfolio_intel.models import PortfolioState, StockDecision


@dataclass(frozen=True)
class SimConfig:
    """
    Very lightweight simulation rules:
    - BUY: add notional exposure by reallocating from cash (if available)
    - TRIM: reduce exposure by moving a fraction into cash
    - EXIT: fully move exposure into cash

    This is not an execution engine; it's an accounting thought experiment.
    """

    buy_cash_fraction_per_name: float = 0.10
    trim_fraction: float = 0.25


def _position_value(state: PortfolioState, mkt: MarketSnapshot, ticker: str) -> float:
    t = ticker.upper()
    px = mkt.prices.get(t)
    if px is None:
        return 0.0
    for p in state.positions:
        if p.ticker.upper() == t:
            return float(p.shares) * float(px)
    return 0.0


class SimulationEngine:
    def __init__(self, cfg: SimConfig | None = None) -> None:
        self.cfg = cfg or SimConfig()

    def simulate_one_day_delta(
        self,
        real: PortfolioState,
        decisions: list[StockDecision],
        mkt: MarketSnapshot,
    ) -> str:
        """
        Produces a plain-English delta summary rather than claiming precise alpha.
        Realistic backtesting requires price history, slippage, and constraints.
        """
        trims = [d for d in decisions if d.action in {"TRIM", "EXIT"}]
        buys = [d for d in decisions if d.action == "BUY"]

        trim_notional = sum(_position_value(real, mkt, d.ticker) for d in trims)
        buy_capacity = float(real.cash)

        return (
            "Simulation (proxy-mode): "
            f"{len(trims)} de-risk signals covering ~${trim_notional:,.0f} notional; "
            f"{len(buys)} accumulate signals with ${buy_capacity:,.0f} cash capacity. "
            "Backtest-grade results require enabling historical price providers."
        )

