from __future__ import annotations


class NonTradingPolicyError(RuntimeError):
    pass


class NonTradingPolicy:
    """
    Guardrail: this system is analysis-only.
    """

    @staticmethod
    def assert_no_trade_intent(action: str) -> None:
        action_u = action.strip().upper()
        if action_u in {"BUY", "SELL", "SHORT", "COVER", "ORDER", "TRADE"}:
            raise NonTradingPolicyError(
                "Trading intent detected. This system must never execute trades."
            )

