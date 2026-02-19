"""
Darwin v5 — TimeSyncManager.

Eliminates Binance timestamp drift errors on signed requests by
computing and maintaining a local offset against exchange server time.

Usage:
    sync = TimeSyncManager(binance_client)
    sync.sync()                   # calibrate once at startup
    ts = sync.synced_timestamp()  # use for all signed requests
"""
from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("darwin.v5.time_sync")


class TimeSyncManager:
    """
    Maintains time offset between local clock and Binance server.

    Binance rejects signed requests when |local_time - server_time| > recvWindow.
    This module periodically re-syncs to prevent 400 timestamp errors.

    Parameters
    ----------
    binance_client : BinanceFuturesClient
        Exchange client with access to /fapi/v1/time endpoint.
    max_drift_ms : int
        Maximum allowed drift before forcing re-sync.
    sync_interval_s : float
        Seconds between automatic re-syncs.
    """

    def __init__(
        self,
        binance_client: Any,
        max_drift_ms: int = 1000,
        sync_interval_s: float = 300.0,
    ) -> None:
        self._client = binance_client
        self._max_drift_ms = max_drift_ms
        self._sync_interval_s = sync_interval_s
        self._offset_ms: int = 0
        self._last_sync: float = 0.0
        self._sync_count: int = 0
        self._last_rtt_ms: float = 0.0

    @property
    def offset_ms(self) -> int:
        """Current time offset in milliseconds (server - local)."""
        return self._offset_ms

    @property
    def sync_count(self) -> int:
        return self._sync_count

    @property
    def last_rtt_ms(self) -> float:
        return self._last_rtt_ms

    def sync(self) -> int:
        """
        Synchronize with Binance server time.

        Returns the computed offset in milliseconds.
        """
        try:
            session = self._client._session
            base_url = self._client._base_url
            timeout = self._client._timeout_s

            t_before = time.time()
            response = session.get(
                f"{base_url}/fapi/v1/time", timeout=timeout
            )
            t_after = time.time()
            response.raise_for_status()

            server_time = int(response.json()["serverTime"])
            rtt_ms = (t_after - t_before) * 1000.0
            local_time = int(((t_before + t_after) / 2) * 1000)

            self._offset_ms = server_time - local_time
            self._last_rtt_ms = rtt_ms
            self._last_sync = time.time()
            self._sync_count += 1

            logger.info(
                "time sync completed: offset=%dms rtt=%.0fms sync_count=%d",
                self._offset_ms,
                rtt_ms,
                self._sync_count,
            )
            return self._offset_ms

        except Exception as exc:
            logger.warning("time sync failed: %s (keeping offset=%dms)", exc, self._offset_ms)
            return self._offset_ms

    def synced_timestamp(self) -> int:
        """
        Return current timestamp adjusted by server offset.

        Automatically re-syncs if interval has elapsed.
        """
        now = time.time()
        if now - self._last_sync > self._sync_interval_s:
            self.sync()
        return int(now * 1000) + self._offset_ms

    def needs_sync(self) -> bool:
        """Check if re-sync is needed based on interval."""
        return (time.time() - self._last_sync) > self._sync_interval_s

    def force_sync_on_error(self, error_code: int) -> bool:
        """
        Force re-sync when a timestamp-related error occurs.

        Parameters
        ----------
        error_code : int
            HTTP status code from the failed request.

        Returns
        -------
        bool
            True if re-sync was performed.
        """
        if error_code in (400, 401):
            logger.warning("forcing time re-sync due to error code %d", error_code)
            self.sync()
            return True
        return False

    def get_diagnostics(self) -> dict:
        """Return diagnostic info for telemetry."""
        return {
            "offset_ms": self._offset_ms,
            "last_rtt_ms": round(self._last_rtt_ms, 1),
            "sync_count": self._sync_count,
            "seconds_since_sync": round(time.time() - self._last_sync, 1) if self._last_sync > 0 else -1,
        }
