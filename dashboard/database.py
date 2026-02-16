"""
Darwin v4 Dashboard — Database Layer.

SQLite database for credential storage and dashboard state.
Thread-safe. No ORM dependency.

Tables:
    - api_credentials: Encrypted exchange API keys
    - dashboard_users: Hashed login credentials
    - bot_events: Audit trail of dashboard actions
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


DB_PATH = os.environ.get("DASHBOARD_DB_PATH", "data/dashboard.db")


class Database:
    """Thread-safe SQLite database."""

    def __init__(self, db_path: str = DB_PATH):
        self._db_path = db_path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        with self._lock, self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS api_credentials (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    exchange TEXT NOT NULL,
                    encrypted_api_key TEXT NOT NULL,
                    encrypted_secret_key TEXT NOT NULL,
                    encrypted_passphrase TEXT DEFAULT NULL,
                    testnet INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS dashboard_users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS bot_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    username TEXT NOT NULL,
                    action TEXT NOT NULL,
                    ip_address TEXT DEFAULT '',
                    success INTEGER DEFAULT 1,
                    details TEXT DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS equity_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    equity REAL NOT NULL,
                    peak REAL NOT NULL,
                    drawdown_pct REAL NOT NULL,
                    leverage REAL NOT NULL DEFAULT 0,
                    pnl_today REAL NOT NULL DEFAULT 0
                );
            """)

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # ── Credentials ──

    def save_credential(
        self,
        exchange: str,
        encrypted_api_key: str,
        encrypted_secret_key: str,
        encrypted_passphrase: Optional[str] = None,
        testnet: bool = True,
    ) -> int:
        now = self._utc_now()
        with self._lock, self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO api_credentials
                   (exchange, encrypted_api_key, encrypted_secret_key,
                    encrypted_passphrase, testnet, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (exchange, encrypted_api_key, encrypted_secret_key,
                 encrypted_passphrase, int(testnet), now, now),
            )
            return cur.lastrowid

    def list_credentials(self) -> List[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT id, exchange, testnet, created_at, updated_at FROM api_credentials"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_credential(self, cred_id: int) -> Optional[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM api_credentials WHERE id = ?", (cred_id,)
            ).fetchone()
            return dict(row) if row else None

    def delete_credential(self, cred_id: int) -> bool:
        with self._lock, self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM api_credentials WHERE id = ?", (cred_id,)
            )
            return cur.rowcount > 0

    # ── Users ──

    def create_user(self, username: str, password_hash: str) -> int:
        now = self._utc_now()
        with self._lock, self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO dashboard_users (username, password_hash, created_at) VALUES (?, ?, ?)",
                (username, password_hash, now),
            )
            return cur.lastrowid

    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM dashboard_users WHERE username = ?", (username,)
            ).fetchone()
            return dict(row) if row else None

    def user_count(self) -> int:
        with self._lock, self._conn() as conn:
            row = conn.execute("SELECT COUNT(*) as c FROM dashboard_users").fetchone()
            return row["c"]

    # ── Events ──

    def log_event(
        self,
        username: str,
        action: str,
        ip_address: str = "",
        success: bool = True,
        details: Optional[Dict] = None,
    ) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                """INSERT INTO bot_events
                   (timestamp, username, action, ip_address, success, details)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (self._utc_now(), username, action, ip_address,
                 int(success), json.dumps(details or {})),
            )

    def recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM bot_events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    # ── Equity Snapshots ──

    def save_equity_snapshot(
        self, equity: float, peak: float, drawdown_pct: float,
        leverage: float = 0.0, pnl_today: float = 0.0,
    ) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                """INSERT INTO equity_snapshots
                   (timestamp, equity, peak, drawdown_pct, leverage, pnl_today)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (self._utc_now(), equity, peak, drawdown_pct, leverage, pnl_today),
            )

    def get_equity_history(self, limit: int = 500) -> List[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                """SELECT timestamp, equity, peak, drawdown_pct, leverage, pnl_today
                   FROM equity_snapshots ORDER BY id DESC LIMIT ?""",
                (limit,),
            ).fetchall()
            return [dict(r) for r in reversed(rows)]
