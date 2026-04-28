from __future__ import annotations

import argparse
from pathlib import Path

from portfolio_intel.runner import run_daily


def main() -> int:
    p = argparse.ArgumentParser(description="AI Portfolio Intelligence Engine (non-trading)")
    p.add_argument("--portfolio", type=Path, default=Path("data/portfolio.json"))
    p.add_argument("--out", type=Path, default=Path("reports"))
    p.add_argument("--as-of", dest="as_of", type=str, default=None, help="YYYY-MM-DD (optional)")
    args = p.parse_args()

    run_daily(portfolio_path=args.portfolio, out_dir=args.out, as_of=args.as_of)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

