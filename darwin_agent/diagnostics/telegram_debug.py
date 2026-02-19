"""
Darwin v4 — Telegram Debug Mode.

When TELEGRAM_DEBUG=true, sends diagnostic messages to Telegram for:
  - Signals near threshold (score >= 0.8 * threshold)
  - Rejected trades (with rejection reason)
  - Executed trades

Message format:
    [Darwin Signal]
    Symbol: BTCUSDT
    Score: 0.72 / 0.75
    ATR: 1.21
    Trend: aligned
    Size: 14.2 USDT
    Status: REJECTED (SCORE_BELOW_THRESHOLD)

All sends are fire-and-forget with error suppression.
Does NOT modify any trading logic.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

from darwin_agent.diagnostics.rejection_reason import RejectionReason
from darwin_agent.diagnostics.signal_diagnostic import SignalDiagnostic

logger = logging.getLogger("darwin.diagnostics.telegram")


class TelegramDebugNotifier:
    """
    Sends signal diagnostic messages to Telegram when debug mode
    is enabled. All sends are non-blocking and error-suppressed.

    Requires a send_fn callable: (text: str) -> None
    This is typically TelegramNotifier.send from the existing
    infrastructure, keeping this class fully decoupled.
    """

    __slots__ = ("_send_fn", "_enabled", "_lock")

    def __init__(
        self,
        send_fn=None,
        enabled: bool = False,
    ) -> None:
        self._send_fn = send_fn
        self._enabled = enabled and send_fn is not None
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _safe_send(self, text: str) -> None:
        """Fire-and-forget send. Never raises, never blocks."""
        if not self._enabled or self._send_fn is None:
            return
        try:
            with self._lock:
                self._send_fn(text)
        except Exception as exc:
            logger.warning("telegram debug send failed: %s", exc)

    def notify_signal_near_threshold(self, diag: SignalDiagnostic) -> None:
        """
        Send when signal_score >= 0.8 * signal_threshold.
        This fires for signals that are close to but may not reach threshold.
        """
        if not self._enabled:
            return

        threshold = diag.signal_threshold
        if threshold <= 0:
            return
        if diag.signal_score < 0.8 * threshold:
            return

        trend_str = "aligned" if diag.trend_alignment else "misaligned"
        status = "WATCHING (near threshold)"

        text = (
            f"[Darwin Signal]\n"
            f"Symbol: {diag.symbol}\n"
            f"Score: {diag.signal_score:.2f} / {threshold:.2f}\n"
            f"ATR: {diag.atr_value:.2f}\n"
            f"Trend: {trend_str}\n"
            f"Size: {diag.computed_position_size_usdt:.1f} USDT\n"
            f"Status: {status}"
        )
        self._safe_send(text)

    def notify_trade_rejected(self, diag: SignalDiagnostic) -> None:
        """Send when a trade is rejected for any reason."""
        if not self._enabled:
            return

        if diag.rejection_reason == RejectionReason.NONE:
            return

        threshold = diag.signal_threshold
        trend_str = "aligned" if diag.trend_alignment else "misaligned"

        text = (
            f"[Darwin Signal]\n"
            f"Symbol: {diag.symbol}\n"
            f"Score: {diag.signal_score:.2f} / {threshold:.2f}\n"
            f"ATR: {diag.atr_value:.2f}\n"
            f"Trend: {trend_str}\n"
            f"Size: {diag.computed_position_size_usdt:.1f} USDT\n"
            f"Status: REJECTED ({diag.rejection_reason.value})"
        )
        self._safe_send(text)

    def notify_trade_executed(
        self,
        diag: SignalDiagnostic,
        filled_qty: float = 0.0,
        filled_price: float = 0.0,
    ) -> None:
        """Send when a trade is successfully executed."""
        if not self._enabled:
            return

        threshold = diag.signal_threshold
        trend_str = "aligned" if diag.trend_alignment else "misaligned"

        text = (
            f"[Darwin Signal]\n"
            f"Symbol: {diag.symbol}\n"
            f"Score: {diag.signal_score:.2f} / {threshold:.2f}\n"
            f"ATR: {diag.atr_value:.2f}\n"
            f"Trend: {trend_str}\n"
            f"Size: {diag.computed_position_size_usdt:.1f} USDT\n"
            f"Qty: {filled_qty:.6f} @ {filled_price:.2f}\n"
            f"Status: EXECUTED"
        )
        self._safe_send(text)

    def notify_pretrade_failure(
        self,
        symbol: str,
        reason: RejectionReason,
        detail: str = "",
    ) -> None:
        """Send when pre-trade validation fails."""
        if not self._enabled:
            return

        text = (
            f"[Darwin Pre-Trade]\n"
            f"Symbol: {symbol}\n"
            f"Status: ABORTED ({reason.value})\n"
            f"Detail: {detail}"
        )
        self._safe_send(text)
