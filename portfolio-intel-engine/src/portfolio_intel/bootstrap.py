from __future__ import annotations

from pathlib import Path

from portfolio_intel.io import load_portfolio
from portfolio_intel.models import Holding, ManagedPortfolio
from portfolio_intel.store import PortfolioStore


def ensure_default_portfolios(store: PortfolioStore, *, portfolio_json: Path = Path("data/portfolio.json")) -> None:
    """
    Creates baseline portfolios on first run:
    - user-imported: derived from data/portfolio.json (if present)
    - bot-strategy: empty starter portfolio
    """
    existing = {p.portfolio_id for p in store.list_portfolios()}

    if "user-imported" not in existing and portfolio_json.exists():
        st = load_portfolio(portfolio_json)
        holdings = [
            Holding(symbol=pos.ticker.upper(), quantity=float(pos.shares), avg_buy_price=float(pos.avg_cost))
            for pos in st.positions
        ]
        store.save_portfolio(
            ManagedPortfolio(
                portfolio_id="user-imported",
                name="User Imported Portfolio",
                risk_mode="medium",
                cash=float(st.cash),
                holdings=holdings,
            )
        )

    if "bot-strategy" not in existing:
        store.save_portfolio(
            ManagedPortfolio(
                portfolio_id="bot-strategy",
                name="Bot Strategy Portfolio",
                risk_mode="medium",
                cash=10_000.0,
                holdings=[],
            )
        )

