"""
Darwin v4 — Audit Logger.

Structured JSON logging for the execution audit layer.
One file per trading day. UTC timestamps. Zero external dependencies.

Thread-safe, non-blocking, deterministic ordering.

Usage:
    logger = AuditLogger(log_dir="/var/log/darwin")
    logger.log_order_intent(...)
    logger.log_fill(...)
    logger.flush()  # call at end of each bar
"""

from __future__ import annotations

import json
import os
import threading
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Deque, Dict, Optional


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    ALERT = "ALERT"


class AuditLogger:
    """
    Structured JSON logger.

    Writes one file per UTC day: audit_2026-02-16.jsonl
    Each line is a single JSON object. JSONL format for streaming ingestion.
    """

    __slots__ = (
        "_log_dir", "_buffer", "_lock", "_current_date",
        "_current_file", "_fh", "_metrics_counters",
    )

    def __init__(self, log_dir: str = "logs/audit"):
        self._log_dir = log_dir
        self._buffer: Deque[Dict[str, Any]] = deque()
        self._lock = threading.Lock()
        self._current_date: str = ""
        self._current_file: str = ""
        self._fh = None
        self._metrics_counters: Dict[str, float] = {}

        os.makedirs(log_dir, exist_ok=True)

    def _utc_now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _ensure_file(self) -> None:
        """Rotate to new file if UTC date changed."""
        today = self._utc_now().strftime("%Y-%m-%d")
        if today != self._current_date:
            if self._fh is not None:
                self._fh.close()
            self._current_date = today
            self._current_file = os.path.join(
                self._log_dir, "audit_{}.jsonl".format(today)
            )
            self._fh = open(self._current_file, "a")

    def _emit(self, record: Dict[str, Any]) -> None:
        """Write a single record. Thread-safe."""
        with self._lock:
            self._ensure_file()
            line = json.dumps(record, separators=(",", ":"), default=str)
            self._fh.write(line + "\n")

    def log(
        self,
        event: str,
        level: LogLevel = LogLevel.INFO,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a structured log entry."""
        record = {
            "ts": self._utc_now().isoformat(),
            "event": event,
            "level": level.value,
        }
        if data:
            record["data"] = data
        self._emit(record)

    # ── Convenience methods ──

    def log_order_intent(
        self,
        symbol: str,
        side: str,
        intended_qty: float,
        intended_price: float,
        leverage: float,
        signal_strength: float,
        bar_index: int,
    ) -> None:
        self.log("ORDER_INTENT", LogLevel.INFO, {
            "symbol": symbol,
            "side": side,
            "intended_qty": round(intended_qty, 8),
            "intended_price": round(intended_price, 4),
            "leverage": round(leverage, 4),
            "signal_strength": round(signal_strength, 6),
            "bar": bar_index,
        })

    def log_fill(
        self,
        symbol: str,
        side: str,
        intended_qty: float,
        filled_qty: float,
        intended_price: float,
        avg_fill_price: float,
        fee: float,
        order_id: str,
        latency_ms: float,
        partial: bool,
        rejected: bool,
    ) -> None:
        slippage_bps = 0.0
        slippage_pct = 0.0
        if intended_price > 0:
            slippage_pct = abs(avg_fill_price - intended_price) / intended_price * 100.0
            slippage_bps = slippage_pct * 100.0  # 1% = 100bps

        fill_ratio = filled_qty / intended_qty if intended_qty > 0 else 0.0

        level = LogLevel.INFO
        if rejected:
            level = LogLevel.ERROR
        elif partial:
            level = LogLevel.WARN
        elif slippage_pct > 0.5:
            level = LogLevel.ALERT

        self.log("ORDER_FILL", level, {
            "symbol": symbol,
            "side": side,
            "intended_qty": round(intended_qty, 8),
            "filled_qty": round(filled_qty, 8),
            "fill_ratio": round(fill_ratio, 4),
            "intended_price": round(intended_price, 4),
            "avg_fill_price": round(avg_fill_price, 4),
            "slippage_bps": round(slippage_bps, 2),
            "slippage_pct": round(slippage_pct, 4),
            "fee": round(fee, 6),
            "order_id": order_id,
            "latency_ms": round(latency_ms, 1),
            "partial": partial,
            "rejected": rejected,
        })

    def log_risk_state(
        self,
        equity: float,
        peak: float,
        drawdown_pct: float,
        rbe_mult: float,
        gmrt_mult: float,
        effective_leverage: float,
        configured_lev_cap: float,
        positions: Dict[str, float],
        bar_index: int,
    ) -> None:
        level = LogLevel.INFO
        if effective_leverage > configured_lev_cap:
            level = LogLevel.ALERT

        self.log("RISK_STATE", level, {
            "equity": round(equity, 2),
            "peak": round(peak, 2),
            "drawdown_pct": round(drawdown_pct, 4),
            "rbe_mult": round(rbe_mult, 4),
            "gmrt_mult": round(gmrt_mult, 4),
            "effective_leverage": round(effective_leverage, 4),
            "configured_lev_cap": round(configured_lev_cap, 4),
            "positions": {s: round(v, 4) for s, v in sorted(positions.items())},
            "bar": bar_index,
        })

    def log_circuit_breaker(
        self,
        trigger_bar: int,
        equity_before: float,
        equity_after: float,
        loss_pct: float,
        positions_closed: int,
        orphan_positions: int,
        cooldown_bars: int,
    ) -> None:
        level = LogLevel.ALERT
        if orphan_positions > 0:
            level = LogLevel.ERROR

        self.log("CIRCUIT_BREAKER", level, {
            "trigger_bar": trigger_bar,
            "equity_before": round(equity_before, 2),
            "equity_after": round(equity_after, 2),
            "loss_pct": round(loss_pct, 4),
            "positions_closed": positions_closed,
            "orphan_positions": orphan_positions,
            "cooldown_bars": cooldown_bars,
        })

    def log_exposure_drift(
        self,
        symbol: str,
        simulated_size: float,
        actual_size: float,
        drift_pct: float,
        mark_price: float,
        liquidation_price: float,
        margin_ratio: float,
        funding_rate: float,
        bar_index: int,
    ) -> None:
        level = LogLevel.INFO
        if abs(drift_pct) > 5.0:
            level = LogLevel.WARN
        if abs(drift_pct) > 10.0 or (
            liquidation_price > 0 and mark_price > 0 and
            abs(mark_price - liquidation_price) / mark_price < 0.15
        ):
            level = LogLevel.ALERT

        self.log("EXPOSURE_DRIFT", level, {
            "symbol": symbol,
            "simulated_size": round(simulated_size, 8),
            "actual_size": round(actual_size, 8),
            "drift_pct": round(drift_pct, 4),
            "mark_price": round(mark_price, 4),
            "liquidation_price": round(liquidation_price, 4),
            "margin_ratio": round(margin_ratio, 4),
            "funding_rate": round(funding_rate, 6),
            "bar": bar_index,
        })

    def log_alert(self, alert_type: str, details: Dict[str, Any]) -> None:
        self.log("ALERT", LogLevel.ALERT, {
            "alert_type": alert_type,
            "details": details,
        })

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
