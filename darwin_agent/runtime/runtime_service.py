from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from darwin_agent.core.engine import DarwinCoreEngine, EquitySnapshot
from darwin_agent.infrastructure.binance_client import BinanceFuturesClient
from darwin_agent.infrastructure.telegram_notifier import TelegramNotifier


@dataclass(slots=True)
class RuntimeStatus:
    running: bool
    retries: int
    safe_shutdown: bool


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
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._running = False
        self._retry_count = 0

    def start(self) -> None:
        if self._running:
            return
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="darwin-runtime")
        self._thread.start()
        self._notify("engine start")

    def stop(self) -> None:
        self._stop_event.set()
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._notify("fatal stop")

    def restart(self) -> None:
        self.stop()
        self.start()

    def status(self) -> RuntimeStatus:
        return RuntimeStatus(running=self._running, retries=self._retry_count, safe_shutdown=self._safe_shutdown_flag)

    def _notify(self, message: str) -> None:
        try:
            self._telegram.send(f"Darwin Engine: {message}")
        except Exception as exc:
            self._logger.error("telegram notification failed: %s", exc)

    def _run_once(self) -> None:
        wallet_balance = self._binance.get_wallet_balance()
        upnl = self._binance.get_unrealized_pnl()
        positions = self._binance.get_open_positions()
        snapshot = EquitySnapshot(wallet_balance=wallet_balance, unrealized_pnl=upnl)
        ctx = self._engine.evaluate(snapshot, positions)

        if ctx.get("drawdown_alert"):
            self._notify("drawdown > 10%")

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._run_once()
                self._retry_count = 0
                time.sleep(self._poll_interval_s)
            except Exception as exc:
                self._logger.error("runtime exception: %s", exc, exc_info=True)
                self._notify(f"runtime exception: {exc}")
                self._retry_count += 1
                if self._retry_count > self._max_retries:
                    self._notify("fatal stop")
                    self._running = False
                    raise SystemExit(1)
                backoff = min(2 ** self._retry_count, 60)
                self._notify(f"restart attempt {self._retry_count}/{self._max_retries} in {backoff}s")
                time.sleep(backoff)
