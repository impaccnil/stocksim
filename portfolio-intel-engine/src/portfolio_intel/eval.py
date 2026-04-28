from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from portfolio_intel.models import TradeRecord


@dataclass(frozen=True)
class PerformanceMetrics:
    trades: int
    win_rate: float
    avg_pnl: float
    total_pnl: float
    max_drawdown: float


def compute_trade_metrics(trades: list[TradeRecord]) -> PerformanceMetrics:
    realized = [t.pnl_realized for t in trades if t.pnl_realized is not None]
    if not realized:
        return PerformanceMetrics(trades=0, win_rate=0.0, avg_pnl=0.0, total_pnl=0.0, max_drawdown=0.0)
    r = np.array(realized, dtype=float)
    wins = float((r > 0).mean())
    total = float(r.sum())
    avg = float(r.mean())
    # equity curve on realized pnl only (proxy)
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    max_dd = float(dd.max()) if len(dd) else 0.0
    return PerformanceMetrics(trades=len(r), win_rate=wins, avg_pnl=avg, total_pnl=total, max_drawdown=max_dd)

