from __future__ import annotations

import re
from dataclasses import dataclass

from portfolio_intel.models import Holding


@dataclass(frozen=True)
class ImportResult:
    holdings: list[Holding]
    errors: list[str]


_LINE_RE = re.compile(
    r"^\s*(?P<symbol>[A-Za-z.\-]+)\s*,\s*(?P<qty>[-+]?\d+(\.\d+)?)\s*,\s*(?P<price>[-+]?\d+(\.\d+)?)\s*$"
)


def parse_holdings_csv(text: str) -> ImportResult:
    """
    Parses lines like:
      AAPL, 10, 150
      NVDA, 5, 800
    Returns holdings + per-line errors (does not raise).
    """
    holdings: list[Holding] = []
    errors: list[str] = []
    for i, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        m = _LINE_RE.match(line)
        if not m:
            errors.append(f"Line {i}: could not parse '{raw}'. Expected: SYMBOL, QTY, BUY_PRICE")
            continue
        sym = m.group("symbol").upper()
        qty = float(m.group("qty"))
        price = float(m.group("price"))
        if qty <= 0:
            errors.append(f"Line {i}: quantity must be > 0")
            continue
        if price < 0:
            errors.append(f"Line {i}: buy price must be >= 0")
            continue
        holdings.append(Holding(symbol=sym, quantity=qty, avg_buy_price=price))
    return ImportResult(holdings=holdings, errors=errors)

