"""
Darwin v4 — Signal Diagnostics Layer.

Pure observability. Does NOT modify trading logic, risk engine math,
entry logic, RQRE, GMRT, or strategy weights.

Components:
    RejectionReason     Enum of all reasons a trade may be rejected.
    SignalDiagnostic     Per-tick diagnostic snapshot for a symbol.
    SignalDiagnosticsLogger   Structured logger for signal diagnostics.
    PreTradeValidator    Re-validates account state before order placement.
    TelegramDebugNotifier  Sends debug-level signal reports to Telegram.
"""

from darwin_agent.diagnostics.rejection_reason import RejectionReason
from darwin_agent.diagnostics.signal_diagnostic import SignalDiagnostic
from darwin_agent.diagnostics.signal_diagnostics_logger import SignalDiagnosticsLogger
from darwin_agent.diagnostics.pre_trade_validator import (
    PreTradeValidator,
    PreTradeResult,
)
from darwin_agent.diagnostics.telegram_debug import TelegramDebugNotifier

__all__ = [
    "RejectionReason",
    "SignalDiagnostic",
    "SignalDiagnosticsLogger",
    "PreTradeValidator",
    "PreTradeResult",
    "TelegramDebugNotifier",
]
