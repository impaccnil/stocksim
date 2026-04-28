from __future__ import annotations

from dataclasses import dataclass

from portfolio_intel.models import EngineScore


def clamp_0_100(x: float) -> float:
    return max(0.0, min(100.0, x))


@dataclass(frozen=True)
class Weights:
    technical: float = 0.35
    fundamental: float = 0.35
    macro: float = 0.20
    sentiment: float = 0.10

    def validate(self) -> None:
        s = self.technical + self.fundamental + self.macro + self.sentiment
        if abs(s - 1.0) > 1e-9:
            raise ValueError(f"Weights must sum to 1.0, got {s}")


def weighted_score(scores: dict[str, EngineScore], w: Weights) -> float:
    w.validate()
    return clamp_0_100(
        scores["technical"].value * w.technical
        + scores["fundamental"].value * w.fundamental
        + scores["macro"].value * w.macro
        + scores["sentiment"].value * w.sentiment
    )

