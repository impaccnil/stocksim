from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from portfolio_intel.features import compute_features
from portfolio_intel.learner import OnlineLogisticLearner
from portfolio_intel.marketdata import OHLCVRequest, OHLCVProvider


def train_learner_on_symbol(
    learner: OnlineLogisticLearner,
    ohlcv: OHLCVProvider,
    symbol: str,
    *,
    lookback_days: int = 365 * 3,
) -> int:
    """
    Trains the logistic learner using next-day direction as target.
    Returns number of training rows used (approx).
    """
    end = datetime.now(tz=timezone.utc) + timedelta(days=1)
    start = end - timedelta(days=lookback_days)
    df = ohlcv.get_ohlcv(OHLCVRequest(symbol, start=start, end=end, timeframe="1d"))
    if df.empty:
        return 0
    feat = compute_features(df)
    close = df["close"].astype(float)
    future_ret = close.pct_change().shift(-1).dropna()
    learner.partial_fit(feat.loc[future_ret.index], future_ret)
    return int(len(future_ret))

