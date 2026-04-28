from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from portfolio_intel.models import ManagedPortfolio, TradeRecord


class PortfolioStore:
    """
    Simple JSON-on-disk store for multiple portfolios + trade history.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or (Path("data") / "store")
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "portfolios").mkdir(exist_ok=True)
        (self.root / "trades").mkdir(exist_ok=True)

    def _portfolio_path(self, portfolio_id: str) -> Path:
        return self.root / "portfolios" / f"{portfolio_id}.json"

    def _trades_path(self, portfolio_id: str) -> Path:
        return self.root / "trades" / f"{portfolio_id}.jsonl"

    def list_portfolios(self) -> list[ManagedPortfolio]:
        out: list[ManagedPortfolio] = []
        for p in sorted((self.root / "portfolios").glob("*.json")):
            out.append(ManagedPortfolio.model_validate_json(p.read_text(encoding="utf-8")))
        return out

    def load_portfolio(self, portfolio_id: str) -> ManagedPortfolio:
        p = self._portfolio_path(portfolio_id)
        if not p.exists():
            raise FileNotFoundError(f"Portfolio not found: {portfolio_id}")
        return ManagedPortfolio.model_validate_json(p.read_text(encoding="utf-8"))

    def save_portfolio(self, portfolio: ManagedPortfolio) -> None:
        p = self._portfolio_path(portfolio.portfolio_id)
        p.write_text(portfolio.model_dump_json(indent=2), encoding="utf-8")

    def append_trade(self, trade: TradeRecord) -> None:
        path = self._trades_path(trade.portfolio_id)
        line = json.dumps(trade.model_dump(mode="json"), ensure_ascii=False)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def load_trades(self, portfolio_id: str, limit: int = 2000) -> list[TradeRecord]:
        path = self._trades_path(portfolio_id)
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        if limit > 0:
            lines = lines[-limit:]
        out: list[TradeRecord] = []
        for ln in lines:
            if not ln.strip():
                continue
            out.append(TradeRecord.model_validate_json(ln))
        return out


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)

