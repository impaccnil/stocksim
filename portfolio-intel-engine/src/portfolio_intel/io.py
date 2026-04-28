from __future__ import annotations

import json
from pathlib import Path

from portfolio_intel.models import PortfolioState


def load_portfolio(path: Path) -> PortfolioState:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return PortfolioState.model_validate(raw)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

