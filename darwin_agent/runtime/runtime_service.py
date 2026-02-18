from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from darwin_agent.core.engine import DarwinCoreEngine, EquitySnapshot
from darwin_agent.infrastructure.binance_client import BinanceFuturesClient
from darwin_agent.infrastructure.telegram_notifier import TelegramNotifier


@dataclass(slots=True)
class RuntimeStatus:
    running: bool
    retries: int
    safe_shutdown: bool


@dataclass(slots=True)
class RuntimeState:
    equity: float = 0.0
    wallet_balance: float = 0.0
    unrealized_pnl: float = 0.0
    positions: list[dict[str, Any]] = field(default_factory=list)
    drawdown_pct: float = 0.0


class RuntimeService:
    def __init__(
        self,
        engine: DarwinCoreEngine,
        binance_client: BinanceFuturesClient,
        telegram_notifier: TelegramNotifier,
        symbols: list[str],
        poll_interval_s: float = 5.0,
        max_retries: int = 5,
        safe_shutdown_flag: bool = True,
    ) -> None:
        self._logger = logging.getLogger("darwin.runtime")
        self._engine = engine
        self._binance = binance_client
        self._telegram = telegram_notifier
        self._symbols = symbols
        self._poll_interval_s = poll_interval_s
        self._max_retries = max_retries
        self._safe_shutdown_flag = safe_shutdown_flag
        self._stop_event = asyncio.Event()
        self._running = False
        self._retry_count = 0
        self._state = RuntimeState()
        self._startup_state: dict[str, Any] = {}
        self._leverage = 5
        self._tick_count = 0

    def status(self) -> RuntimeStatus:
        return RuntimeStatus(running=self._running, retries=self._retry_count, safe_shutdown=self._safe_shutdown_flag)

    def state(self) -> RuntimeState:
        return self._state

    def stop(self) -> None:
        self._stop_event.set()
        self._running = False

    async def run_forever(self) -> int:
        self._stop_event.clear()
        self._running = True
        self._telegram.notify_engine_started()
        self._tick_count = 0
        self._logger.info("runtime loop started", extra={"event": "engine_started", "symbols": list(self._symbols), "leverage": self._leverage})
        try:
            while not self._stop_event.is_set():
                try:
                    await self._run_once()
                    self._retry_count = 0
                    await asyncio.sleep(self._poll_interval_s)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._retry_count += 1
                    self._logger.error(
                        "runtime tick exception",
                        extra={"event": "runtime_tick_error", "retry": self._retry_count, "error": str(exc)},
                        exc_info=True,
                    )
                    self._telegram.notify_error(str(exc))
                    if self._retry_count > self._max_retries:
                        self._logger.error(
                            "max retries exceeded",
                            extra={"event": "runtime_fatal_stop", "retry": self._retry_count},
                        )
                        return 1
                    self._telegram.notify_reconnect_attempt(self._retry_count, self._max_retries)
                    backoff = min(2 ** self._retry_count, 60)
                    await asyncio.sleep(backoff)
            return 0
        finally:
            self._running = False
            self._telegram.notify_engine_stopped()

    async def _run_once(self) -> None:
        self._tick_count += 1
        wallet_balance = await asyncio.to_thread(self._binance.get_wallet_balance)
        upnl = await asyncio.to_thread(self._binance.get_unrealized_pnl)
        positions = await asyncio.to_thread(self._binance.get_open_positions)
        snapshot = EquitySnapshot(wallet_balance=wallet_balance, unrealized_pnl=upnl)
        ctx = self._engine.evaluate(snapshot, positions)

        self._state.wallet_balance = wallet_balance
        self._state.unrealized_pnl = upnl
        self._state.equity = snapshot.equity
        self._state.positions = list(positions)
        self._state.drawdown_pct = float(ctx.get("drawdown_pct", 0.0))

        if ctx.get("drawdown_alert"):
            self._telegram.notify_error("drawdown > 10%")

        self._logger.info(
            "runtime tick",
            extra={
                "event": "runtime_tick",
                "equity": self._state.equity,
                "positions": len(self._state.positions),
                "drawdown_pct": self._state.drawdown_pct,
                "symbols": list(self._symbols),
                "leverage": self._leverage,
                "tick_count": self._tick_count,
            },
        )

        if self._tick_count % 60 == 0:
            wallet = await asyncio.to_thread(self._binance.get_wallet_balance)
            positions = await asyncio.to_thread(self._binance.get_open_positions)
            self._logger.info("Heartbeat: wallet=%.4f open_positions=%d", wallet, len(positions))
