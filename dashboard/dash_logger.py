"""
Darwin v4 Dashboard — Action Logger.

Logs all dashboard actions to daily JSONL files.
logs/dashboard-YYYY-MM-DD.jsonl
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone


class DashboardLogger:
    """Structured JSONL logger for dashboard actions."""

    def __init__(self, log_dir: str = "logs"):
        self._log_dir = log_dir
        self._lock = threading.Lock()
        self._fh = None
        self._current_date = ""
        os.makedirs(log_dir, exist_ok=True)

    def _ensure_file(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._current_date:
            if self._fh:
                self._fh.close()
            self._current_date = today
            path = os.path.join(self._log_dir, "dashboard-{}.jsonl".format(today))
            self._fh = open(path, "a")

    def log(
        self,
        user: str,
        action: str,
        ip: str = "",
        success: bool = True,
        details: dict = None,
    ) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user": user,
            "action": action,
            "ip": ip,
            "success": success,
        }
        if details:
            record["details"] = details

        with self._lock:
            self._ensure_file()
            self._fh.write(
                json.dumps(record, separators=(",", ":"), default=str) + "\n"
            )
            self._fh.flush()

    def close(self):
        with self._lock:
            if self._fh:
                self._fh.close()
                self._fh = None
