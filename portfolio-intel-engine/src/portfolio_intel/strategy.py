from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from portfolio_intel.broker import OrderIntent
from portfolio_intel.features import FeatureConfig, compute_features
from portfolio_intel.learner import OnlineLogisticLearner
from portfolio_intel.marketdata import OHLCVRequest, OHLCVProvider
from portfolio_intel.models import ManagedPortfolio
from portfolio_intel.risk_modes import cfg_for
from portfolio_intel.signals import (
    SignalConfig,
    momentum_signal,
    smc_signal,
    vol_risk_score,
    wyckoff_signal,
    zscore_mean_reversion_signal,
)


@dataclass(frozen=True)
class StrategyConfig:
    score_threshold: float = 0.35
    buy_threshold: float = 0.55
    sell_threshold: float = -0.55
    rebalance_to_cash_floor: float = 0.02
    max_new_buys_per_step: int = 3
    trim_fraction: float = 0.25

    # weights from your spec
    w_momentum: float = 0.30
    w_wyckoff: float = 0.20
    w_smc: float = 0.20
    w_stat: float = 0.20
    w_fundamental: float = 0.10  # placeholder until fundamentals implemented


def _asof_end() -> tuple[datetime, datetime]:
    now = datetime.now(tz=timezone.utc)
    end = now + timedelta(days=1)
    start = now - timedelta(days=180)
    return start, end


class HybridStrategy:
    """
    Rule + model hybrid:
    - features -> signals
    - logistic learner outputs P(up)
    - combine into final_score:
        0.3 momentum + 0.2 wyckoff + 0.2 smc + 0.2 statistical + 0.1 fundamental
    """

    def __init__(
        self,
        ohlcv: OHLCVProvider,
        *,
        cfg: StrategyConfig | None = None,
        feat_cfg: FeatureConfig | None = None,
        sig_cfg: SignalConfig | None = None,
        learner: OnlineLogisticLearner | None = None,
    ) -> None:
        self.ohlcv = ohlcv
        self.cfg = cfg or StrategyConfig()
        self.feat_cfg = feat_cfg or FeatureConfig()
        self.sig_cfg = sig_cfg or SignalConfig()
        self.learner = learner or OnlineLogisticLearner()

    def score_symbol(self, symbol: str) -> tuple[float, dict[str, float], str]:
        start, end = _asof_end()
        df = self.ohlcv.get_ohlcv(OHLCVRequest(symbol, start=start, end=end, timeframe="1d"))
        feat = compute_features(df, self.feat_cfg)
        if feat.empty:
            return 0.0, {}, "No OHLCV."
        row = feat.iloc[-1]

        mom = float(momentum_signal(feat).iloc[-1])
        wyk = float(wyckoff_signal(feat).iloc[-1])
        smc = float(smc_signal(feat).iloc[-1])
        stat = float(zscore_mean_reversion_signal(feat, self.sig_cfg).iloc[-1])

        # logistic regression probability
        p_up = self.learner.predict_proba_up(row)
        model_sig = (p_up - 0.5) * 2.0  # map to [-1,1]

        # fundamental placeholder (neutral)
        fund = 0.0

        final = (
            self.cfg.w_momentum * mom
            + self.cfg.w_wyckoff * wyk
            + self.cfg.w_smc * smc
            + self.cfg.w_stat * (0.6 * stat + 0.4 * model_sig)
            + self.cfg.w_fundamental * fund
        )
        final = float(np.clip(final, -1.0, 1.0))

        snap = {
            "momentum": mom,
            "wyckoff": wyk,
            "smc": smc,
            "stat": stat,
            "p_up": float(p_up),
            "final": final,
            "zscore": float(row.get("zscore", 0.0)),
            "vol_std": float(row.get("vol_std", 0.0)),
            "vol_ratio": float(row.get("vol_ratio", 1.0)),
        }
        reason = (
            f"Score={final:+.2f} (mom {mom:+.2f}, wyk {wyk:+.2f}, smc {smc:+.2f}, stat {stat:+.2f}, P(up) {p_up:.2f})."
        )
        return final, snap, reason

    def propose_intents(self, portfolio: ManagedPortfolio, universe: list[str]) -> list[OrderIntent]:
        cfg_mode = cfg_for(portfolio.risk_mode)

        # Current symbols held
        held = {h.symbol.upper() for h in portfolio.holdings}
        candidates = sorted({u.upper() for u in universe})

        scored: list[tuple[str, float, dict[str, float], str]] = []
        for sym in candidates:
            s, snap, reason = self.score_symbol(sym)
            scored.append((sym, s, snap, reason))

        # sell/trim signals for held positions
        intents: list[OrderIntent] = []
        for sym, score, snap, reason in scored:
            if sym not in held:
                continue
            if score <= self.cfg.sell_threshold:
                intents.append(
                    OrderIntent(
                        symbol=sym,
                        action="TRIM",
                        quantity=self.cfg.trim_fraction,  # interpreted as fraction later by executor if desired
                        price=self._latest_price(sym),
                        reason=f"TRIM signal. {reason}",
                        features_snapshot=snap,
                    )
                )

        # buy signals for not-held
        buys = [(sym, sc, snap, reason) for (sym, sc, snap, reason) in scored if sym not in held and sc >= self.cfg.buy_threshold]
        buys = sorted(buys, key=lambda x: -x[1])[: self.cfg.max_new_buys_per_step]
        for sym, score, snap, reason in buys:
            intents.append(
                OrderIntent(
                    symbol=sym,
                    action="BUY",
                    quantity=1.0,  # placeholder; sizing layer will convert
                    price=self._latest_price(sym),
                    reason=f"BUY signal. {reason}",
                    features_snapshot=snap,
                )
            )

        return intents

    def _latest_price(self, symbol: str) -> float:
        start, end = _asof_end()
        df = self.ohlcv.get_ohlcv(OHLCVRequest(symbol, start=end - timedelta(days=7), end=end, timeframe="1d"))
        if df.empty:
            raise ValueError(f"No price for {symbol}")
        return float(df["close"].astype(float).iloc[-1])

