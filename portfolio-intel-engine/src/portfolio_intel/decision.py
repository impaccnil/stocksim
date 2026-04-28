from __future__ import annotations

from dataclasses import dataclass

from portfolio_intel.engines import EngineContext, FundamentalEngine, MacroRegimeEngine, SentimentEngine, TechnicalEngine
from portfolio_intel.models import EngineScore, StockDecision
from portfolio_intel.scoring import Weights, clamp_0_100, weighted_score


@dataclass(frozen=True)
class DecisionThresholds:
    buy: float = 67.0
    hold: float = 52.0
    trim: float = 40.0
    # below trim => exit


def _risk_level(technical: EngineScore, macro: EngineScore) -> str:
    # high risk if technical weak and macro unsupportive
    if technical.value < 40 and macro.value < 45:
        return "high"
    if technical.value < 48 or macro.value < 48:
        return "medium"
    return "low"


class DecisionEngine:
    def __init__(
        self,
        weights: Weights | None = None,
        thresholds: DecisionThresholds | None = None,
        *,
        technical: TechnicalEngine | None = None,
        fundamental: FundamentalEngine | None = None,
        macro: MacroRegimeEngine | None = None,
        sentiment: SentimentEngine | None = None,
    ) -> None:
        self.weights = weights or Weights()
        self.thresholds = thresholds or DecisionThresholds()
        self.technical = technical or TechnicalEngine()
        self.fundamental = fundamental or FundamentalEngine()
        self.macro = macro or MacroRegimeEngine()
        self.sentiment = sentiment or SentimentEngine()

    def decide(self, ticker: str, ctx: EngineContext) -> StockDecision:
        t = ticker.upper()
        scores: dict[str, EngineScore] = {
            "technical": self.technical.score(t, ctx),
            "fundamental": self.fundamental.score(t, ctx),
            "macro": self.macro.score(t, ctx),
            "sentiment": self.sentiment.score(t, ctx),
        }
        total = weighted_score(scores, self.weights)

        if total >= self.thresholds.buy:
            action = "BUY"
        elif total >= self.thresholds.hold:
            action = "HOLD"
        elif total >= self.thresholds.trim:
            action = "TRIM"
        else:
            action = "EXIT"

        # confidence grows with distance from the hold boundary and score dispersion
        confidence = clamp_0_100(50.0 + abs(total - self.thresholds.hold) * 1.2)
        risk_level = _risk_level(scores["technical"], scores["macro"])
        reasoning = (
            f"Weighted score {total:.1f}/100 → {action}. "
            f"Tech {scores['technical'].value:.0f}, Fund {scores['fundamental'].value:.0f}, "
            f"Macro {scores['macro'].value:.0f}, Sent {scores['sentiment'].value:.0f}. "
            "Uncertainty: fundamentals/news are proxy-mode unless real providers are enabled."
        )

        return StockDecision(
            ticker=t,
            action=action,  # guidance only; no execution
            confidence=confidence,
            risk_level=risk_level,  # type: ignore[arg-type]
            reasoning=reasoning,
            technical=scores["technical"],
            fundamental=scores["fundamental"],
            macro=scores["macro"],
            sentiment=scores["sentiment"],
        )

