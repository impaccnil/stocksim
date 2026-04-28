from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd

from portfolio_intel.marketdata import OHLCVRequest, OHLCVProvider
from portfolio_intel.models import Holding, ManagedPortfolio


@dataclass(frozen=True)
class HoldingMetrics:
    symbol: str
    name: str | None
    quantity: float
    avg_buy_price: float
    current_price: float
    price_change: float
    price_change_pct: float
    market_value: float
    cost_basis: float
    gain_loss: float
    gain_loss_pct: float
    allocation_pct: float
    reinvest: bool


@dataclass(frozen=True)
class PortfolioMetrics:
    total_value: float
    cash: float
    total_cost_basis: float
    total_return: float
    total_return_pct: float
    daily_change: float
    daily_change_pct: float
    diversification_score: float  # 0-100 heuristic
    top_gainers: list[HoldingMetrics]
    top_losers: list[HoldingMetrics]
    overexposed: list[HoldingMetrics]
    holdings: list[HoldingMetrics]


def _last_two_closes(ohlcv: OHLCVProvider, symbol: str) -> tuple[float, float | None]:
    now = datetime.now(tz=timezone.utc)
    req = OHLCVRequest(symbol, start=now - timedelta(days=10), end=now + timedelta(days=1), timeframe="1d")
    df = ohlcv.get_ohlcv(req)
    if df.empty:
        raise ValueError(f"No OHLCV for {symbol}")
    close = df["close"].astype(float)
    last = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) >= 2 else None
    return last, prev


def compute_portfolio_metrics(
    p: ManagedPortfolio,
    ohlcv: OHLCVProvider,
    *,
    max_position_weight: float,
) -> PortfolioMetrics:
    rows: list[HoldingMetrics] = []
    cash = float(p.cash)

    # First pass: get prices and values
    total_mkt = cash
    temp: list[tuple[Holding, float, float | None]] = []
    for h in p.holdings:
        px, prev = _last_two_closes(ohlcv, h.symbol)
        mv = float(h.quantity) * px
        total_mkt += mv
        temp.append((h, px, prev))

    total_cost = cash
    total_daily = 0.0

    for h, px, prev in temp:
        qty = float(h.quantity)
        cost = qty * float(h.avg_buy_price)
        mv = qty * px
        total_cost += cost

        if prev is not None:
            total_daily += qty * (px - prev)

        price_chg = (px - prev) if prev is not None else 0.0
        price_chg_pct = (price_chg / prev) if prev not in (None, 0.0) else 0.0
        gl = mv - cost
        gl_pct = (gl / cost) if cost > 0 else 0.0
        alloc = (mv / total_mkt) if total_mkt > 0 else 0.0

        rows.append(
            HoldingMetrics(
                symbol=h.symbol.upper(),
                name=h.name,
                quantity=qty,
                avg_buy_price=float(h.avg_buy_price),
                current_price=px,
                price_change=price_chg,
                price_change_pct=price_chg_pct,
                market_value=mv,
                cost_basis=cost,
                gain_loss=gl,
                gain_loss_pct=gl_pct,
                allocation_pct=alloc,
                reinvest=h.reinvest,
            )
        )

    total_return = total_mkt - total_cost
    total_return_pct = (total_return / total_cost) if total_cost > 0 else 0.0
    daily_change_pct = (total_daily / (total_mkt - total_daily)) if (total_mkt - total_daily) > 0 else 0.0

    # Diversification heuristic: penalize concentration using Herfindahl index on allocations
    w = [r.allocation_pct for r in rows if r.market_value > 0]
    hhi = sum(x * x for x in w)
    diversification = max(0.0, min(100.0, 100.0 * (1.0 - hhi)))  # higher is better

    # Sorters
    gainers = sorted(rows, key=lambda r: r.price_change_pct, reverse=True)[:5]
    losers = sorted(rows, key=lambda r: r.price_change_pct)[:5]
    over = [r for r in sorted(rows, key=lambda r: r.allocation_pct, reverse=True) if r.allocation_pct >= max_position_weight]

    return PortfolioMetrics(
        total_value=total_mkt,
        cash=cash,
        total_cost_basis=total_cost,
        total_return=total_return,
        total_return_pct=total_return_pct,
        daily_change=total_daily,
        daily_change_pct=daily_change_pct,
        diversification_score=diversification,
        top_gainers=gainers,
        top_losers=losers,
        overexposed=over[:10],
        holdings=sorted(rows, key=lambda r: r.market_value, reverse=True),
    )

