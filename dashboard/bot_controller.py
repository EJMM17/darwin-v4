"""
Darwin v4 Dashboard — Bot Controller.

Thread-safe bot lifecycle management.
Reads state from bot + ExecutionAudit. Does NOT modify trading logic.

States: STOPPED → RUNNING → STOPPED
Emergency close: RUNNING → EMERGENCY_LOCKED (requires manual unlock)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class BotState(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    EMERGENCY_LOCKED = "emergency_locked"
    ERROR = "error"


@dataclass
class BotStatus:
    """Read-only snapshot of current bot state."""
    state: BotState = BotState.STOPPED
    mode: str = "paper"
    equity: float = 0.0
    peak_equity: float = 0.0
    drawdown_pct: float = 0.0
    leverage: float = 0.0
    rbe_mult: float = 1.0
    gmrt_mult: float = 1.0
    pnl_today: float = 0.0
    rqre_days_remaining: int = 30
    exposure_by_symbol: Dict[str, float] = field(default_factory=dict)
    max_dd_rolling: float = 0.0
    liq_distance_pct: float = 100.0
    margin_usage_pct: float = 0.0
    cb_triggers: int = 0
    halted_bars: int = 0
    last_alert: str = ""
    started_at: str = ""
    uptime_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "running": self.state == BotState.RUNNING,
            "state": self.state.value,
            "mode": self.mode,
            "equity": round(self.equity, 2),
            "peak_equity": round(self.peak_equity, 2),
            "drawdown_pct": round(self.drawdown_pct, 2),
            "leverage": round(self.leverage, 4),
            "rbe_mult": round(self.rbe_mult, 4),
            "gmrt_mult": round(self.gmrt_mult, 4),
            "pnl_today": round(self.pnl_today, 2),
            "rqre_days_remaining": self.rqre_days_remaining,
            "exposure_by_symbol": {
                k: round(v, 4) for k, v in sorted(self.exposure_by_symbol.items())
            },
            "max_dd_rolling": round(self.max_dd_rolling, 2),
            "liq_distance_pct": round(self.liq_distance_pct, 2),
            "margin_usage_pct": round(self.margin_usage_pct, 2),
            "cb_triggers": self.cb_triggers,
            "halted_bars": self.halted_bars,
            "last_alert": self.last_alert,
            "started_at": self.started_at,
            "uptime_seconds": round(self.uptime_seconds, 1),
        }


class BotController:
    """
    Thread-safe bot lifecycle controller.

    Manages bot start/stop/emergency without touching trading logic.
    The bot_runner_fn is the user-provided callable that runs the actual bot.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._status = BotStatus()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._emergency_lock = threading.Event()
        self._runner_fn: Optional[Callable] = None
        self._started_at: float = 0.0

    @property
    def status(self) -> BotStatus:
        with self._lock:
            if self._status.state == BotState.RUNNING:
                self._status.uptime_seconds = time.time() - self._started_at
            return BotStatus(**{
                f.name: getattr(self._status, f.name)
                for f in self._status.__dataclass_fields__.values()
            })

    def update_status(self, **kwargs) -> None:
        """Called by the bot thread to push state updates. Thread-safe."""
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self._status, k):
                    setattr(self._status, k, v)

    def mark_started(self, mode: str) -> None:
        """Mark runtime start metadata when startup bypasses controller.start()."""
        with self._lock:
            self._started_at = time.time()
            self._status.mode = mode
            self._status.started_at = datetime.now(timezone.utc).isoformat()

    def mark_stopped(self, state: BotState = BotState.STOPPED) -> None:
        """Normalize stop metadata when runtime handles lifecycle directly."""
        with self._lock:
            self._status.state = state
            self._status.uptime_seconds = 0.0

    def set_runner(self, fn: Callable) -> None:
        """Set the bot runner function. Must accept (stop_event, controller) args."""
        self._runner_fn = fn

    def start(self, mode: str = "paper") -> Dict[str, Any]:
        """Start the bot in a background thread."""
        with self._lock:
            if self._status.state == BotState.RUNNING:
                return {"ok": False, "error": "Bot already running"}
            if self._status.state == BotState.EMERGENCY_LOCKED:
                return {"ok": False, "error": "Emergency lock active. Unlock first."}
            if self._runner_fn is None:
                return {"ok": False, "error": "No bot runner configured"}

            self._stop_event.clear()
            self._emergency_lock.clear()
            self._status.state = BotState.STARTING
            self._status.mode = mode
            self._started_at = time.time()
            self._status.started_at = datetime.now(timezone.utc).isoformat()

        def _run():
            try:
                with self._lock:
                    self._status.state = BotState.RUNNING
                self._runner_fn(self._stop_event, self)
            except Exception as e:
                with self._lock:
                    self._status.last_alert = "Bot crashed: {}".format(str(e)[:200])
            finally:
                with self._lock:
                    if self._status.state != BotState.EMERGENCY_LOCKED:
                        self._status.state = BotState.STOPPED

        self._thread = threading.Thread(target=_run, daemon=True, name="darwin-bot")
        self._thread.start()
        return {"ok": True, "mode": mode}

    def stop(self) -> Dict[str, Any]:
        """Graceful stop. Signals the bot thread to finish current bar and exit."""
        with self._lock:
            if self._status.state not in (BotState.RUNNING, BotState.STARTING):
                return {"ok": False, "error": "Bot not running"}
            self._status.state = BotState.STOPPING

        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=30)

        with self._lock:
            self._status.state = BotState.STOPPED
        return {"ok": True}

    def emergency_close(self, close_fn: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Emergency: stop bot + close all positions + lock.

        close_fn: Optional async/sync callable that cancels orders and closes positions.
                  Called AFTER stopping the bot thread.
        """
        with self._lock:
            self._status.state = BotState.EMERGENCY_LOCKED
            self._status.last_alert = "EMERGENCY CLOSE triggered"

        self._stop_event.set()
        self._emergency_lock.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=15)

        result = {"ok": True, "positions_closed": 0}

        if close_fn:
            try:
                close_result = close_fn()
                if isinstance(close_result, dict):
                    result["positions_closed"] = close_result.get("closed", 0)
            except Exception as e:
                result["close_error"] = str(e)[:200]

        return result

    def unlock(self) -> Dict[str, Any]:
        """Unlock after emergency close. Allows bot to be started again."""
        with self._lock:
            if self._status.state != BotState.EMERGENCY_LOCKED:
                return {"ok": False, "error": "Not in emergency lock"}
            self._status.state = BotState.STOPPED
            self._emergency_lock.clear()
        return {"ok": True}

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._status.state == BotState.RUNNING

    @property
    def is_locked(self) -> bool:
        with self._lock:
            return self._status.state == BotState.EMERGENCY_LOCKED

    @property
    def should_stop(self) -> bool:
        """Bot thread checks this each iteration."""
        return self._stop_event.is_set()
