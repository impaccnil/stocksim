from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, PositiveInt, confloat


Action = Literal["BUY", "HOLD", "TRIM", "EXIT"]
RiskLevel = Literal["low", "medium", "high"]
RiskMode = Literal["aggressive", "medium", "low"]


class Holding(BaseModel):
    """
    Expanded holding structure for UI.
    """

    symbol: str = Field(min_length=1)
    name: str | None = None
    quantity: confloat(gt=0)  # type: ignore[valid-type]
    avg_buy_price: confloat(ge=0)  # type: ignore[valid-type]
    reinvest: bool = False


class Position(BaseModel):
    ticker: str = Field(min_length=1)
    shares: PositiveInt
    avg_cost: confloat(ge=0)  # type: ignore[valid-type]


class PortfolioState(BaseModel):
    as_of: date | None = None
    cash: confloat(ge=0)  # type: ignore[valid-type]
    positions: list[Position]

    def tickers(self) -> list[str]:
        return sorted({p.ticker.upper() for p in self.positions})


class EngineScore(BaseModel):
    value: confloat(ge=0, le=100)  # type: ignore[valid-type]
    summary: str


class StockDecision(BaseModel):
    ticker: str
    action: Action
    confidence: confloat(ge=0, le=100)  # type: ignore[valid-type]
    risk_level: RiskLevel
    reasoning: str
    technical: EngineScore
    fundamental: EngineScore
    macro: EngineScore
    sentiment: EngineScore


class PortfolioRiskReport(BaseModel):
    fragility_score: confloat(ge=0, le=100)  # type: ignore[valid-type]
    warnings: list[str]
    top_concentrations: list[str]
    sector_overexposures: list[str]
    correlation_notes: list[str]


class DailyReport(BaseModel):
    as_of: date
    market_regime: str
    portfolio_health_score: confloat(ge=0, le=100)  # type: ignore[valid-type]
    top_risks: list[str]
    top_opportunities: list[str]
    trims: list[StockDecision]
    accumulates: list[StockDecision]
    decisions: list[StockDecision]
    risk: PortfolioRiskReport
    sim_vs_real_delta_summary: str
    email_subject: str
    email_body: str


class TradeRecord(BaseModel):
    portfolio_id: str
    timestamp: datetime
    symbol: str
    action: Literal["BUY", "SELL", "TRIM", "EXIT", "REBALANCE", "HOLD"]
    quantity_delta: float
    price: float
    notional: float
    reason: str
    features_snapshot: dict[str, float] = Field(default_factory=dict)
    pnl_realized: float | None = None


class ManagedPortfolio(BaseModel):
    portfolio_id: str
    name: str
    risk_mode: RiskMode = "medium"
    cash: confloat(ge=0)  # type: ignore[valid-type]
    holdings: list[Holding] = Field(default_factory=list)

