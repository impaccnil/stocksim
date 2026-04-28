from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from portfolio_intel.models import Holding, ManagedPortfolio, TradeRecord
from portfolio_intel.risk_modes import cfg_for
from portfolio_intel.store import PortfolioStore, utc_now


Action = Literal["BUY", "SELL", "TRIM", "EXIT", "REBALANCE", "HOLD"]


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    action: Action
    quantity: float
    price: float
    reason: str
    features_snapshot: dict[str, float] | None = None


def _get_holding(p: ManagedPortfolio, symbol: str) -> Holding | None:
    s = symbol.upper()
    for h in p.holdings:
        if h.symbol.upper() == s:
            return h
    return None


def _upsert_holding(p: ManagedPortfolio, holding: Holding) -> None:
    s = holding.symbol.upper()
    for i, h in enumerate(p.holdings):
        if h.symbol.upper() == s:
            p.holdings[i] = holding
            return
    p.holdings.append(holding)


class PaperBroker:
    """
    Simulated execution + accounting engine.
    No real orders. Applies intents to a ManagedPortfolio and logs TradeRecords.
    """

    def __init__(self, store: PortfolioStore) -> None:
        self.store = store

    def apply_intent(self, portfolio: ManagedPortfolio, intent: OrderIntent) -> TradeRecord:
        sym = intent.symbol.upper()
        price = float(intent.price)
        qty = float(intent.quantity)
        if qty <= 0:
            raise ValueError("quantity must be > 0")
        if price <= 0:
            raise ValueError("price must be > 0")

        cfg = cfg_for(portfolio.risk_mode)
        holding = _get_holding(portfolio, sym)

        ts = utc_now()
        pnl_realized: float | None = None

        # Special case: TRIM can be specified as a fraction (0<qty<1) of current position
        if intent.action == "TRIM":
            holding = _get_holding(portfolio, sym)
            if holding is None:
                raise ValueError(f"No existing holding for {sym}")
            if 0.0 < qty < 1.0:
                qty = float(holding.quantity) * qty

        if intent.action in {"BUY"}:
            notional = qty * price
            if notional > float(portfolio.cash) + 1e-9:
                raise ValueError("Insufficient cash for BUY intent.")

            # Create or update average buy price with VWAP-style blending
            if holding is None:
                new_qty = qty
                new_avg = price
                new_reinv = False
                new_name = None
            else:
                new_qty = float(holding.quantity) + qty
                if new_qty <= 0:
                    new_avg = 0.0
                else:
                    new_avg = (float(holding.avg_buy_price) * float(holding.quantity) + price * qty) / new_qty
                new_reinv = holding.reinvest
                new_name = holding.name

            portfolio.cash = float(portfolio.cash) - notional  # type: ignore[assignment]
            _upsert_holding(
                portfolio,
                Holding(
                    symbol=sym,
                    name=new_name,
                    quantity=new_qty,
                    avg_buy_price=new_avg,
                    reinvest=new_reinv,
                ),
            )
            qdelta = qty

        elif intent.action in {"SELL", "TRIM", "EXIT"}:
            if holding is None:
                raise ValueError(f"No existing holding for {sym}")

            if intent.action == "EXIT":
                sell_qty = float(holding.quantity)
            else:
                sell_qty = qty

            if sell_qty <= 0 or sell_qty - float(holding.quantity) > 1e-9:
                raise ValueError("Sell quantity exceeds position size.")

            notional = sell_qty * price
            portfolio.cash = float(portfolio.cash) + notional  # type: ignore[assignment]

            # realized PnL vs avg cost
            pnl_realized = (price - float(holding.avg_buy_price)) * sell_qty

            remaining = float(holding.quantity) - sell_qty
            if remaining <= 1e-9:
                portfolio.holdings = [h for h in portfolio.holdings if h.symbol.upper() != sym]
            else:
                _upsert_holding(
                    portfolio,
                    Holding(
                        symbol=sym,
                        name=holding.name,
                        quantity=remaining,
                        avg_buy_price=float(holding.avg_buy_price),
                        reinvest=holding.reinvest,
                    ),
                )
            qdelta = -sell_qty

        else:
            # HOLD/REBALANCE log-only for transparency
            notional = 0.0
            qdelta = 0.0

        trade = TradeRecord(
            portfolio_id=portfolio.portfolio_id,
            timestamp=ts,
            symbol=sym,
            action=intent.action,
            quantity_delta=qdelta,
            price=price,
            notional=notional,
            reason=intent.reason,
            features_snapshot=intent.features_snapshot or {},
            pnl_realized=pnl_realized,
        )

        self.store.save_portfolio(portfolio)
        self.store.append_trade(trade)
        return trade

