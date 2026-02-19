"""
Darwin v4 — Signal Diagnostic Dataclass.

Immutable snapshot capturing the full state of a signal evaluation
for a single symbol on a single tick. Used for logging and Telegram
debug reports. Does NOT modify any trading logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from darwin_agent.diagnostics.rejection_reason import RejectionReason


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class SignalDiagnostic:
    """Per-symbol, per-tick diagnostic snapshot."""

    # Identification
    symbol: str = ""
    timestamp: datetime = field(default_factory=_utcnow)

    # Signal metrics
    signal_score: float = 0.0
    signal_threshold: float = 0.0

    # Market conditions
    atr_value: float = 0.0
    trend_alignment: bool = False
    volatility_ok: bool = True
    funding_ok: bool = True

    # Risk state
    risk_allowed: bool = True

    # Position sizing
    computed_position_size_usdt: float = 0.0
    computed_quantity: float = 0.0
    min_notional_required: float = 5.0
    final_notional: float = 0.0

    # Outcome
    rejection_reason: RejectionReason = RejectionReason.NONE

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "signal_score": round(self.signal_score, 6),
            "signal_threshold": round(self.signal_threshold, 6),
            "atr_value": round(self.atr_value, 6),
            "trend_alignment": self.trend_alignment,
            "volatility_ok": self.volatility_ok,
            "funding_ok": self.funding_ok,
            "risk_allowed": self.risk_allowed,
            "computed_position_size_usdt": round(self.computed_position_size_usdt, 4),
            "computed_quantity": round(self.computed_quantity, 8),
            "min_notional_required": round(self.min_notional_required, 4),
            "final_notional": round(self.final_notional, 4),
            "rejection_reason": self.rejection_reason.value,
        }
