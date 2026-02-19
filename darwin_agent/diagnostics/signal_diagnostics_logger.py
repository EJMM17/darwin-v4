"""
Darwin v4 — Signal Diagnostics Logger.

Structured JSONL logging of per-symbol, per-tick signal diagnostics.
Writes to logs/diagnostics/ with daily rotation. Thread-safe,
non-blocking, no crash on logging failures.

Does NOT modify any trading logic.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone

from darwin_agent.diagnostics.signal_diagnostic import SignalDiagnostic

logger = logging.getLogger("darwin.diagnostics")


class SignalDiagnosticsLogger:
    """
    Append-only JSONL logger for signal diagnostics.

    One file per UTC day: signal_diagnostics_YYYY-MM-DD.jsonl
    Each line is a JSON object representing one SignalDiagnostic.

    Safe to call from any thread. All I/O errors are caught and logged
    at WARNING level — never raises, never blocks the trading path.
    """

    __slots__ = (
        "_log_dir", "_lock", "_current_date", "_fh", "_enabled",
    )

    def __init__(self, log_dir: str = "logs/diagnostics", enabled: bool = True) -> None:
        self._log_dir = log_dir
        self._lock = threading.Lock()
        self._current_date: str = ""
        self._fh = None
        self._enabled = enabled

        if enabled:
            try:
                os.makedirs(log_dir, exist_ok=True)
            except OSError as exc:
                logger.warning("cannot create diagnostics log dir %s: %s", log_dir, exc)
                self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _utcnow(self) -> datetime:
        return datetime.now(timezone.utc)

    def _ensure_file(self) -> bool:
        """Rotate to new file if UTC date changed. Returns True on success."""
        today = self._utcnow().strftime("%Y-%m-%d")
        if today != self._current_date:
            if self._fh is not None:
                try:
                    self._fh.close()
                except OSError:
                    pass
            self._current_date = today
            filename = os.path.join(
                self._log_dir,
                f"signal_diagnostics_{today}.jsonl",
            )
            try:
                self._fh = open(filename, "a")
            except OSError as exc:
                logger.warning("cannot open diagnostics log %s: %s", filename, exc)
                self._fh = None
                return False
        return self._fh is not None

    def log(self, diagnostic: SignalDiagnostic) -> None:
        """
        Write a single diagnostic entry. Non-blocking, never raises.
        """
        if not self._enabled:
            return
        try:
            record = diagnostic.to_dict()
            with self._lock:
                if not self._ensure_file():
                    return
                line = json.dumps(record, separators=(",", ":"), default=str)
                self._fh.write(line + "\n")
                self._fh.flush()
        except Exception as exc:
            # Never crash the trading path for a logging failure
            logger.warning("diagnostics log write failed: %s", exc)

    def close(self) -> None:
        """Flush and close the current file handle."""
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.close()
                except OSError:
                    pass
                self._fh = None
