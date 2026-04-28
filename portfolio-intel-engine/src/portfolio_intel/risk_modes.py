from __future__ import annotations

from dataclasses import dataclass

from portfolio_intel.models import RiskMode


@dataclass(frozen=True)
class RiskModeConfig:
    mode: RiskMode
    max_position_weight: float
    max_positions: int
    target_turnover_per_day: int
    vol_target: float  # annualized vol target proxy for sizing
    allow_high_vol: bool


RISK_MODE_CONFIGS: dict[RiskMode, RiskModeConfig] = {
    "aggressive": RiskModeConfig(
        mode="aggressive",
        max_position_weight=0.25,
        max_positions=25,
        target_turnover_per_day=8,
        vol_target=0.35,
        allow_high_vol=True,
    ),
    "medium": RiskModeConfig(
        mode="medium",
        max_position_weight=0.15,
        max_positions=35,
        target_turnover_per_day=4,
        vol_target=0.25,
        allow_high_vol=True,
    ),
    "low": RiskModeConfig(
        mode="low",
        max_position_weight=0.08,
        max_positions=45,
        target_turnover_per_day=2,
        vol_target=0.18,
        allow_high_vol=False,
    ),
}


def cfg_for(mode: RiskMode) -> RiskModeConfig:
    return RISK_MODE_CONFIGS[mode]

