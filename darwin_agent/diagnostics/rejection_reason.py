"""
Darwin v4 — Rejection Reason Enum.

Enumerates every discrete reason a trade signal can be rejected
before reaching the exchange. Used by the diagnostics layer for
transparent logging. Does NOT modify any trading logic.
"""
from __future__ import annotations

from enum import Enum


class RejectionReason(Enum):
    """Explicit rejection reasons for signal diagnostics."""

    # Signal quality
    SCORE_BELOW_THRESHOLD = "SCORE_BELOW_THRESHOLD"

    # Market conditions
    ATR_TOO_LOW = "ATR_TOO_LOW"
    TREND_MISALIGNED = "TREND_MISALIGNED"

    # Risk engine
    RISK_ENGINE_BLOCKED = "RISK_ENGINE_BLOCKED"
    DRAWDOWN_LOCK = "DRAWDOWN_LOCK"

    # Order sizing
    MIN_NOTIONAL_FAIL = "MIN_NOTIONAL_FAIL"
    INSUFFICIENT_MARGIN = "INSUFFICIENT_MARGIN"

    # Exchange constraints
    LEVERAGE_MISMATCH = "LEVERAGE_MISMATCH"

    # Funding filter
    FUNDING_FILTER_BLOCK = "FUNDING_FILTER_BLOCK"

    # Pre-trade validation failures
    BALANCE_FETCH_FAILED = "BALANCE_FETCH_FAILED"
    LEVERAGE_SET_FAILED = "LEVERAGE_SET_FAILED"
    ZERO_QUANTITY = "ZERO_QUANTITY"

    # No rejection (trade approved)
    NONE = "NONE"

    @classmethod
    def from_risk_verdict(cls, reason: str) -> "RejectionReason":
        """
        Map a PortfolioRiskEngine verdict reason string to a
        RejectionReason enum. Best-effort mapping; defaults to
        RISK_ENGINE_BLOCKED for unrecognized reasons.
        """
        if not reason:
            return cls.NONE

        lower = reason.lower()

        # Drawdown / halt related
        if "halted" in lower or "drawdown" in lower:
            return cls.DRAWDOWN_LOCK

        # Daily loss
        if "daily_loss" in lower:
            return cls.DRAWDOWN_LOCK

        # Consecutive losses
        if "consecutive_loss" in lower:
            return cls.RISK_ENGINE_BLOCKED

        # Signal quality
        if "signal_strength" in lower or "weak_signal" in lower:
            return cls.SCORE_BELOW_THRESHOLD

        if "low_confidence" in lower or "confidence" in lower:
            return cls.SCORE_BELOW_THRESHOLD

        # Exposure limits
        if "exposure" in lower:
            return cls.INSUFFICIENT_MARGIN

        # Position limits
        if "max_positions" in lower or "duplicate_symbol" in lower:
            return cls.RISK_ENGINE_BLOCKED

        # VaR
        if "var" in lower:
            return cls.RISK_ENGINE_BLOCKED

        # Directional concentration
        if "directional" in lower:
            return cls.RISK_ENGINE_BLOCKED

        # Capital
        if "zero_capital" in lower:
            return cls.INSUFFICIENT_MARGIN

        # Critical / close-only
        if "critical" in lower or "close_only" in lower:
            return cls.DRAWDOWN_LOCK

        return cls.RISK_ENGINE_BLOCKED
