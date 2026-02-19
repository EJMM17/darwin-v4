"""
Darwin v5 — TelemetryReporter.

Structured JSON logging and optional Telegram diagnostics.
Provides centralized observability for all v5 modules.

Usage:
    telemetry = TelemetryReporter(telegram_notifier=tg, log_dir="/app/logs")
    telemetry.log_signal(signal_data)
    telemetry.log_rejection(symbol, reason, details)
    telemetry.log_execution(order_data)
    telemetry.log_heartbeat(state)
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("darwin.v5.telemetry")


class TelemetryReporter:
    """
    Centralized telemetry for the v5 engine.

    Writes structured JSON log lines and sends Telegram notifications
    for key events.

    Parameters
    ----------
    telegram_notifier : TelegramNotifier, optional
        For sending Telegram alerts.
    log_dir : str
        Directory for structured log files.
    signal_threshold_notify : float
        Notify Telegram when signal confidence >= this fraction of threshold.
    """

    def __init__(
        self,
        telegram_notifier: Any = None,
        log_dir: str = "/app/logs",
        signal_threshold_notify: float = 0.8,
    ) -> None:
        self._telegram = telegram_notifier
        self._log_dir = Path(log_dir)
        self._signal_threshold_notify = signal_threshold_notify
        self._event_count: int = 0
        self._jsonl_logger: Optional[logging.Logger] = None

        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            self._jsonl_logger = logging.getLogger("darwin.v5.telemetry.jsonl")
            self._jsonl_logger.setLevel(logging.INFO)
            self._jsonl_logger.propagate = False
            # Only add handler if none exist
            if not self._jsonl_logger.handlers:
                from logging.handlers import RotatingFileHandler
                handler = RotatingFileHandler(
                    self._log_dir / "v5_telemetry.jsonl",
                    maxBytes=10_000_000,
                    backupCount=5,
                )
                handler.setFormatter(logging.Formatter("%(message)s"))
                self._jsonl_logger.addHandler(handler)
        except Exception as exc:
            logger.warning("failed to set up JSONL logger: %s", exc)

    def _emit(self, event_type: str, data: Dict[str, Any]) -> None:
        """Write a structured JSON log line."""
        self._event_count += 1
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            "seq": self._event_count,
            **data,
        }
        if self._jsonl_logger:
            try:
                self._jsonl_logger.info(json.dumps(record, default=str))
            except Exception:
                pass
        logger.debug("telemetry: %s", event_type)

    def _notify(self, text: str) -> None:
        """Send Telegram notification (best-effort)."""
        if self._telegram:
            try:
                self._telegram.send(text)
            except Exception as exc:
                logger.warning("telegram notify failed: %s", exc)

    # ── Signal Events ─────────────────────────────────────

    def log_signal_generated(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        threshold: float,
        regime: str,
        factors: Dict[str, float],
        position_size_usdt: float,
    ) -> None:
        """Log when a signal is generated."""
        self._emit("signal_generated", {
            "symbol": symbol,
            "direction": direction,
            "confidence": round(confidence, 4),
            "threshold": round(threshold, 4),
            "regime": regime,
            "factors": {k: round(v, 4) for k, v in factors.items()},
            "position_size_usdt": round(position_size_usdt, 2),
        })
        if confidence >= self._signal_threshold_notify * threshold:
            self._notify(
                f"Signal: {symbol} {direction}\n"
                f"Confidence: {confidence:.3f} (threshold: {threshold:.3f})\n"
                f"Regime: {regime}\n"
                f"Factors: {', '.join(f'{k}={v:.2f}' for k, v in factors.items())}\n"
                f"Size: ${position_size_usdt:.2f}"
            )

    def log_signal_rejected(
        self,
        symbol: str,
        reason: str,
        confidence: float = 0.0,
        regime: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log when a signal is rejected."""
        self._emit("signal_rejected", {
            "symbol": symbol,
            "reason": reason,
            "confidence": round(confidence, 4),
            "regime": regime,
            **(details or {}),
        })
        self._notify(
            f"Signal REJECTED: {symbol}\n"
            f"Reason: {reason}\n"
            f"Confidence: {confidence:.3f}\n"
            f"Regime: {regime}"
        )

    # ── Execution Events ──────────────────────────────────

    def log_order_placed(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        leverage: int,
        order_type: str = "MARKET",
    ) -> None:
        """Log when an order is placed."""
        self._emit("order_placed", {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": round(price, 8),
            "leverage": leverage,
            "order_type": order_type,
        })
        self._notify(
            f"ORDER PLACED: {side} {symbol}\n"
            f"Qty: {quantity} @ ${price:.2f}\n"
            f"Leverage: {leverage}x"
        )

    def log_order_filled(
        self,
        symbol: str,
        side: str,
        filled_qty: float,
        avg_price: float,
        fee: float,
        order_id: str,
        slippage_bps: float = 0.0,
    ) -> None:
        """Log when an order is filled."""
        self._emit("order_filled", {
            "symbol": symbol,
            "side": side,
            "filled_qty": filled_qty,
            "avg_price": round(avg_price, 8),
            "fee": round(fee, 8),
            "order_id": order_id,
            "slippage_bps": round(slippage_bps, 2),
        })
        self._notify(
            f"ORDER FILLED: {side} {symbol}\n"
            f"Filled: {filled_qty} @ ${avg_price:.2f}\n"
            f"Fee: ${fee:.4f}\n"
            f"Slippage: {slippage_bps:.1f}bps"
        )

    def log_position_closed(
        self,
        symbol: str,
        side: str,
        pnl: float,
        pnl_pct: float,
        close_reason: str,
        equity_after: float,
    ) -> None:
        """Log when a position is closed."""
        self._emit("position_closed", {
            "symbol": symbol,
            "side": side,
            "pnl": round(pnl, 4),
            "pnl_pct": round(pnl_pct, 2),
            "close_reason": close_reason,
            "equity_after": round(equity_after, 2),
        })
        emoji = "+" if pnl >= 0 else ""
        self._notify(
            f"POSITION CLOSED: {symbol}\n"
            f"PnL: {emoji}${pnl:.2f} ({emoji}{pnl_pct:.1f}%)\n"
            f"Reason: {close_reason}\n"
            f"Equity: ${equity_after:.2f}"
        )

    # ── Heartbeat ─────────────────────────────────────────

    def log_heartbeat(
        self,
        equity: float,
        wallet_balance: float,
        unrealized_pnl: float,
        open_positions: int,
        regime: str,
        rbe_mult: float = 1.0,
        gmrt_mult: float = 1.0,
        drawdown_pct: float = 0.0,
    ) -> None:
        """Log periodic heartbeat status (every 60s)."""
        self._emit("heartbeat", {
            "equity": round(equity, 4),
            "wallet_balance": round(wallet_balance, 4),
            "unrealized_pnl": round(unrealized_pnl, 4),
            "open_positions": open_positions,
            "regime": regime,
            "rbe_mult": round(rbe_mult, 4),
            "gmrt_mult": round(gmrt_mult, 4),
            "drawdown_pct": round(drawdown_pct, 2),
        })

    # ── Risk Events ───────────────────────────────────────

    def log_risk_event(
        self,
        event_type: str,
        details: Dict[str, Any],
    ) -> None:
        """Log risk-related events (daily loss cap, drawdown, etc.)."""
        self._emit(f"risk_{event_type}", details)
        self._notify(f"RISK EVENT: {event_type}\n{json.dumps(details, indent=2, default=str)}")

    # ── Monte Carlo ───────────────────────────────────────

    def log_monte_carlo_result(
        self,
        n_simulations: int,
        mean_pnl: float,
        p5_pnl: float,
        p95_pnl: float,
        edge_ratio: float,
        max_drawdown_p95: float,
    ) -> None:
        """Log Monte Carlo validation results."""
        self._emit("monte_carlo_result", {
            "n_simulations": n_simulations,
            "mean_pnl": round(mean_pnl, 4),
            "p5_pnl": round(p5_pnl, 4),
            "p95_pnl": round(p95_pnl, 4),
            "edge_ratio": round(edge_ratio, 4),
            "max_drawdown_p95": round(max_drawdown_p95, 4),
        })

    @property
    def event_count(self) -> int:
        return self._event_count
