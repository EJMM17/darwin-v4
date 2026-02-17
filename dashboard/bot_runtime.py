"""Production runtime manager decoupled from dashboard request/websocket lifecycle."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from darwin_agent.config import DarwinConfig, ExchangeConfig
from darwin_agent.main import run as darwin_run
from darwin_agent.monitoring.execution_audit import ExecutionAudit
from darwin_agent.exchanges.binance import BinanceAdapter

from dashboard.alerts import send_telegram_alert
from dashboard.bot_controller import BotController, BotState
from dashboard.config_loader import RuntimeConfigLoader
from dashboard.crypto_vault import CryptoVault
from dashboard.database import Database

MAX_LEVERAGE = 5


class RuntimeManager:
    """Owns trading runtime lifecycle in a dedicated background thread."""

    def __init__(self, controller: BotController, audit: ExecutionAudit, config: Optional[DarwinConfig] = None):
        self.controller = controller
        self.audit = audit
        self.config = config or RuntimeConfigLoader().load()

        self.stop_event = threading.Event()
        self._lock = threading.RLock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._last_start_error = ""
        self._logger = logging.getLogger("darwin.runtime")
        self._runtime_started_monotonic = 0.0
        self._status: Dict[str, Any] = {
            "is_running": False,
            "current_equity": 0.0,
            "wallet_balance": 0.0,
            "positions": [],
            "drawdown": 0.0,
            "last_update": "",
            "mode": "TEST",
        }

    @property
    def last_start_error(self) -> str:
        return self._last_start_error

    def is_running(self) -> bool:
        with self._lock:
            return self._running and self._thread is not None and self._thread.is_alive()

    def get_runtime_status(self) -> Dict[str, Any]:
        with self._lock:
            out = dict(self._status)
            out["is_running"] = self.is_running()
            return out

    def _load_live_binance_credentials(self) -> ExchangeConfig:
        db = Database()
        vault = CryptoVault()
        creds = sorted(db.list_credentials(), key=lambda c: c.get("id", 0), reverse=True)
        rec = next((c for c in creds if str(c.get("exchange", "")).lower() == "binance" and not bool(c.get("testnet", 1))), None)
        if not rec:
            raise RuntimeError("Live Binance credentials are missing")
        full = db.get_credential(int(rec["id"]))
        if not full:
            raise RuntimeError("Failed to load live Binance credential record")
        return ExchangeConfig(
            exchange_id="binance",
            api_key=vault.decrypt(full["encrypted_api_key"]),
            api_secret=vault.decrypt(full["encrypted_secret_key"]),
            testnet=False,
            enabled=True,
            leverage=MAX_LEVERAGE,
            symbols=list(self.config.exchanges[0].symbols if self.config.exchanges else ["BTCUSDT"]),
        )

    async def _validate_credentials(self, mode: str) -> None:
        if mode != "live":
            return
        ex_cfg = self._load_live_binance_credentials()
        adapter = BinanceAdapter(api_key=ex_cfg.api_key, api_secret=ex_cfg.api_secret, testnet=False)
        try:
            await adapter.validate_live_credentials()
            await adapter.get_wallet_balance_and_upnl()
            self._logger.info("exchange connection success")
        finally:
            await adapter.close()

    async def _set_hard_leverage(self) -> None:
        ex = next((e for e in self.config.exchanges if e.enabled), None)
        if ex is None:
            return
        ex.leverage = MAX_LEVERAGE
        adapter = BinanceAdapter(api_key=ex.api_key, api_secret=ex.api_secret, testnet=bool(ex.testnet))
        try:
            for symbol in ex.symbols:
                await adapter.set_leverage(symbol, MAX_LEVERAGE)
                self._logger.info("leverage set symbol=%s leverage=%s", symbol, MAX_LEVERAGE)
        finally:
            await adapter.close()

    def calculate_live_equity(self, wallet_balance: float, unrealized_pnl: float) -> float:
        return float(wallet_balance) + float(unrealized_pnl)

    def start(self, mode: str = "test") -> bool:
        with self._lock:
            if self.is_running():
                return False
            self._last_start_error = ""
            self.config = RuntimeConfigLoader().load()
            mode_value = (mode or self.config.mode or "test").lower()
            self.config.mode = "live" if mode_value == "live" else "test"
            try:
                if self.config.mode == "live":
                    self.config.exchanges = [self._load_live_binance_credentials()]
                self._run_async_in_thread(self._validate_credentials(self.config.mode))
            except Exception as exc:
                self._last_start_error = str(exc)
                self.controller.update_status(state=BotState.ERROR, last_alert=self._last_start_error)
                self._logger.error("exchange connection failure: %s", exc)
                return False

            self.stop_event.clear()
            self._runtime_started_monotonic = time.monotonic()
            self._running = True
            self._status["mode"] = "LIVE" if self.config.mode == "live" else "TEST"
            self._status["is_running"] = True
            self.controller.mark_started(mode=self._status["mode"])
            self.controller.update_status(state=BotState.STARTING)
            self._thread = threading.Thread(target=self._thread_main, daemon=True, name="darwin-runtime")
            self._thread.start()
            send_telegram_alert("Darwin bot started")
            return True

    def _run_async_in_thread(self, coro: Any) -> None:
        holder: Dict[str, Any] = {"error": None}

        def _runner() -> None:
            try:
                asyncio.run(coro)
            except Exception as exc:
                holder["error"] = exc

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        t.join()
        if holder["error"] is not None:
            raise holder["error"]

    def stop(self, timeout: float = 30.0) -> bool:
        with self._lock:
            if not self._running:
                return False
            self.stop_event.set()
            lp = self._loop
        if lp and lp.is_running():
            lp.call_soon_threadsafe(self._cancel_task)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout)
        with self._lock:
            self._running = False
            self._status["is_running"] = False
        self.controller.mark_stopped(BotState.STOPPED)
        send_telegram_alert("Darwin bot stopped")
        return True

    def emergency_close(self) -> bool:
        return self.stop()

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            self._task = loop.create_task(self._runtime_loop())
            loop.run_until_complete(self._task)
        except BaseException as exc:
            if not isinstance(exc, asyncio.CancelledError):
                self._record_error("runtime_thread_exception", Exception(str(exc)))
        finally:
            pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
            with self._lock:
                self._running = False
                self._status["is_running"] = False
            if self.controller.status.state not in (BotState.ERROR, BotState.EMERGENCY_LOCKED):
                self.controller.mark_stopped(BotState.STOPPED)

    def _cancel_task(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()

    async def _runtime_loop(self) -> None:
        reconnect_delay = 2.0
        while not self.stop_event.is_set():
            try:
                await self._set_hard_leverage()
                await self._update_live_snapshot()
                await darwin_run(self.config)
                if not self.stop_event.is_set():
                    self._logger.warning("runtime stopped unexpectedly; reconnect attempt")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._record_error("runtime_exception", exc, fatal=False)
                self._logger.error("reconnect attempt in %.1fs", reconnect_delay)
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(30.0, reconnect_delay * 2)
                continue
            await asyncio.sleep(1.0)

    async def _update_live_snapshot(self) -> None:
        ex = next((e for e in self.config.exchanges if e.enabled), None)
        if ex is None:
            return
        adapter = BinanceAdapter(api_key=ex.api_key, api_secret=ex.api_secret, testnet=bool(ex.testnet))
        try:
            positions = await adapter.get_positions()
            wallet_balance, unrealized_pnl = await adapter.get_wallet_balance_and_upnl()
            equity = self.calculate_live_equity(wallet_balance, unrealized_pnl) if self.config.mode == "live" else float(self.config.starting_capital)
            position_payload = [
                {
                    "symbol": str(getattr(p, "symbol", "")),
                    "size": float(getattr(p, "size", 0.0)),
                    "side": str(getattr(p, "side", "")),
                    "leverage": float(getattr(p, "leverage", 0.0)),
                }
                for p in positions
            ]
            peak = max(float(self.controller.status.peak_equity), equity)
            dd = ((peak - equity) / peak * 100.0) if peak > 0 else 0.0
            now = datetime.now(timezone.utc).isoformat()
            with self._lock:
                self._status.update({
                    "current_equity": float(equity),
                    "wallet_balance": float(wallet_balance),
                    "positions": position_payload,
                    "drawdown": float(dd),
                    "last_update": now,
                })
            self.controller.update_status(
                state=BotState.RUNNING,
                mode=("LIVE" if self.config.mode == "live" else "TEST"),
                equity=float(equity),
                peak_equity=float(peak),
                drawdown_pct=float(dd),
                exposure_by_symbol={p["symbol"]: p["size"] for p in position_payload},
                uptime_seconds=max(0.0, time.monotonic() - self._runtime_started_monotonic),
            )
            self._logger.info("balance fetch wallet=%.6f upnl=%.6f equity=%.6f", wallet_balance, unrealized_pnl, equity)
        except Exception as exc:
            self._record_error("balance_fetch_failed", exc, fatal=False)
        finally:
            await adapter.close()

    def _record_error(self, alert_type: str, exc: Exception, fatal: bool = True) -> None:
        try:
            self.audit._fire_alert(alert_type, {"error": str(exc)[:400]})
        except Exception:
            pass
        self._logger.exception("exception caught: %s", exc)
        self.controller.update_status(last_alert=str(exc)[:200])
        if fatal:
            self.controller.update_status(state=BotState.ERROR)


DarwinRuntime = RuntimeManager
