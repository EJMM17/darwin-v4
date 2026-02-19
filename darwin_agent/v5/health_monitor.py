"""
Darwin v5 — HealthMonitor.

Pings the exchange every 30s, detects connectivity loss, and triggers
reconnection. Runs as an async background task.

Usage:
    monitor = HealthMonitor(binance_client, telegram_notifier)
    asyncio.create_task(monitor.run())
    monitor.stop()
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("darwin.v5.health")


@dataclass(slots=True)
class HealthStatus:
    """Point-in-time health snapshot."""
    connected: bool = False
    last_ping_ms: float = 0.0
    last_success_ts: float = 0.0
    consecutive_failures: int = 0
    total_pings: int = 0
    total_failures: int = 0
    uptime_pct: float = 100.0


class HealthMonitor:
    """
    Background exchange health monitor.

    Pings the exchange periodically and tracks connectivity.
    On failure, attempts reconnection with exponential backoff.

    Parameters
    ----------
    binance_client : BinanceFuturesClient
        Exchange client with ping_futures() method.
    telegram_notifier : TelegramNotifier, optional
        For alerting on connectivity loss.
    ping_interval_s : float
        Seconds between health pings.
    max_consecutive_failures : int
        Alert threshold for consecutive failures.
    """

    def __init__(
        self,
        binance_client: Any,
        telegram_notifier: Any = None,
        ping_interval_s: float = 30.0,
        max_consecutive_failures: int = 3,
    ) -> None:
        self._client = binance_client
        self._telegram = telegram_notifier
        self._ping_interval_s = ping_interval_s
        self._max_consecutive_failures = max_consecutive_failures
        self._stop_event = asyncio.Event()
        self._status = HealthStatus()

    @property
    def status(self) -> HealthStatus:
        return self._status

    @property
    def is_healthy(self) -> bool:
        return self._status.connected

    def stop(self) -> None:
        self._stop_event.set()

    async def run(self) -> None:
        """Run the health monitor loop. Call as asyncio.create_task()."""
        logger.info("health monitor started (interval=%.0fs)", self._ping_interval_s)
        while not self._stop_event.is_set():
            await self._ping()
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._ping_interval_s,
                )
                break  # stop_event was set
            except asyncio.TimeoutError:
                pass  # normal: continue to next ping
        logger.info("health monitor stopped")

    async def _ping(self) -> None:
        """Execute a single health ping."""
        self._status.total_pings += 1
        t_start = time.time()

        try:
            await asyncio.to_thread(self._client.ping_futures)
            latency_ms = (time.time() - t_start) * 1000.0

            was_disconnected = not self._status.connected
            self._status.connected = True
            self._status.last_ping_ms = latency_ms
            self._status.last_success_ts = time.time()
            self._status.consecutive_failures = 0

            if was_disconnected:
                logger.info("exchange connectivity restored (latency=%.0fms)", latency_ms)
                if self._telegram:
                    try:
                        self._telegram.send("Exchange connectivity restored")
                    except Exception:
                        pass

        except Exception as exc:
            self._status.consecutive_failures += 1
            self._status.total_failures += 1
            self._status.connected = False

            logger.warning(
                "health ping failed (%d consecutive): %s",
                self._status.consecutive_failures,
                exc,
            )

            if self._status.consecutive_failures >= self._max_consecutive_failures:
                logger.error(
                    "exchange connectivity lost (%d consecutive failures)",
                    self._status.consecutive_failures,
                )
                if self._telegram:
                    try:
                        self._telegram.send(
                            f"Exchange connectivity lost ({self._status.consecutive_failures} failures)"
                        )
                    except Exception:
                        pass

        # Update uptime percentage
        total = self._status.total_pings
        if total > 0:
            self._status.uptime_pct = (
                (total - self._status.total_failures) / total * 100.0
            )

    async def wait_for_connection(self, timeout_s: float = 60.0) -> bool:
        """
        Wait until exchange is reachable.

        Returns True if connected, False if timeout.
        """
        deadline = time.time() + timeout_s
        backoff = 1.0
        while time.time() < deadline:
            await self._ping()
            if self._status.connected:
                return True
            await asyncio.sleep(min(backoff, deadline - time.time()))
            backoff = min(backoff * 2, 16.0)
        return False

    def get_diagnostics(self) -> dict:
        """Return diagnostic info for telemetry."""
        s = self._status
        return {
            "connected": s.connected,
            "last_ping_ms": round(s.last_ping_ms, 1),
            "consecutive_failures": s.consecutive_failures,
            "total_pings": s.total_pings,
            "total_failures": s.total_failures,
            "uptime_pct": round(s.uptime_pct, 2),
        }
