"""Production runtime bridge for running Darwin async engine from dashboard."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from darwin_agent.config import DarwinConfig, ExchangeConfig, load_config
from darwin_agent.main import run as darwin_run
from darwin_agent.monitoring.execution_audit import ExecutionAudit

from dashboard.alerts import send_telegram_alert
from dashboard.bot_controller import BotController, BotState
from dashboard.database import Database
from dashboard.crypto_vault import CryptoVault
from darwin_agent.exchanges.binance import BinanceAdapter


class DarwinRuntime:
    """Runs Darwin async engine in a dedicated event loop thread."""

    def __init__(self, controller: BotController, audit: ExecutionAudit, config: Optional[DarwinConfig] = None):
        self.controller = controller
        self.audit = audit
        self.config = config or load_config(os.getenv("DARWIN_CONFIG", "config.yaml"))

        self.thread: Optional[threading.Thread] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.stop_event = threading.Event()
        self.running = False

        self._lock = threading.RLock()
        self._main_task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._last_status_update = time.monotonic()
        self._last_account_sync = 0.0
        self._last_seen_audit_alert = 0
        self._runtime_started_monotonic = 0.0
        self._starting_capital = max(self.config.starting_capital, 0.01)
        self._logger = logging.getLogger("darwin.runtime")

    def _load_live_binance_credentials(self) -> ExchangeConfig:
        db = Database(os.environ.get("DASHBOARD_DB_PATH", "data/dashboard.db"))
        vault = CryptoVault()

        creds = sorted(db.list_credentials(), key=lambda c: c.get("id", 0), reverse=True)
        binance_live = next(
            (c for c in creds if str(c.get("exchange", "")).lower() == "binance" and not bool(c.get("testnet", 1))),
            None,
        )
        if binance_live is None:
            raise RuntimeError("No live Binance credentials found in dashboard store")

        record = db.get_credential(int(binance_live["id"]))
        if not record:
            raise RuntimeError("Failed to load Binance credential record")

        return ExchangeConfig(
            exchange_id="binance",
            api_key=vault.decrypt(record["encrypted_api_key"]),
            api_secret=vault.decrypt(record["encrypted_secret_key"]),
            testnet=False,
            enabled=True,
            leverage=int(self.config.exchanges[0].leverage if self.config.exchanges else 20),
            symbols=list(self.config.exchanges[0].symbols) if self.config.exchanges else ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        )

    def _enforce_live_binance(self) -> None:
        live_exchange = self._load_live_binance_credentials()
        self.config.mode = "live"
        self.config.exchanges = [live_exchange]

    def is_running(self) -> bool:
        with self._lock:
            return self.running and self.thread is not None and self.thread.is_alive()

    def start(self, mode: str = "live") -> bool:
        with self._lock:
            if self.is_running():
                return False

            requested_mode = (mode or "live").strip().lower()
            if requested_mode == "live":
                self._enforce_live_binance()
                dashboard_mode = "live"
            else:
                # Runtime currently only supports Binance adapter execution.
                # Keep mode marker aligned with dashboard UX for non-live starts.
                self.config.mode = "test"
                dashboard_mode = "paper"

            self.controller.mark_started(mode=dashboard_mode)
            self.controller.update_status(state=BotState.STARTING)
            self.stop_event.clear()
            self._runtime_started_monotonic = time.monotonic()
            self.running = True
            self.thread = threading.Thread(target=self._thread_main, daemon=True, name="darwin-runtime")
            self.thread.start()
        send_telegram_alert("Darwin bot started")
        return True

    def stop(self, timeout: float = 30.0) -> bool:
        with self._lock:
            if not self.running:
                return False
            self.stop_event.set()
            loop = self.loop

        if loop and loop.is_running():
            loop.call_soon_threadsafe(self._request_stop)

        thread = self.thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)

        with self._lock:
            self.running = False

        self.controller.mark_stopped(BotState.STOPPED)
        send_telegram_alert("Darwin bot stopped")
        return True

    def _request_stop(self) -> None:
        if self._main_task and not self._main_task.done():
            self._main_task.cancel()
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()

    def emergency_close(self) -> bool:
        with self._lock:
            if not self.running:
                self.controller.update_status(
                    state=BotState.EMERGENCY_LOCKED,
                    last_alert="EMERGENCY CLOSE triggered",
                )
                return False
            self.stop_event.set()
            self.controller.update_status(
                state=BotState.EMERGENCY_LOCKED,
                last_alert="EMERGENCY CLOSE triggered",
                uptime_seconds=0.0,
            )
            loop = self.loop

        if loop and loop.is_running():
            loop.call_soon_threadsafe(self._request_stop)

        self._logger.warning("Emergency close requested from dashboard")
        send_telegram_alert("Emergency close triggered from dashboard")
        return True

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        with self._lock:
            self.loop = loop

        try:
            loop.run_until_complete(self._runtime_main())
        except Exception as exc:
            self._record_error("runtime_thread_exception", exc)
        finally:
            pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            with self._lock:
                self.loop = None
                self.running = False
            if self.controller.status.state not in (BotState.ERROR, BotState.EMERGENCY_LOCKED):
                self.controller.mark_stopped(BotState.STOPPED)

    async def _runtime_main(self) -> None:
        self._main_task = asyncio.create_task(self._engine_wrapper(), name="darwin-main")
        self._monitor_task = asyncio.create_task(self._status_monitor(), name="darwin-monitor")

        done, pending = await asyncio.wait(
            {self._main_task, self._monitor_task},
            return_when=asyncio.FIRST_EXCEPTION,
        )

        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        for task in done:
            if task.cancelled():
                continue
            exc = task.exception()
            if exc and not isinstance(exc, asyncio.CancelledError):
                raise exc

    async def _engine_wrapper(self) -> None:
        original_signal = signal.signal

        def _safe_signal(sig, handler):
            try:
                return original_signal(sig, handler)
            except ValueError:
                return None

        signal.signal = _safe_signal
        try:
            await darwin_run(self.config)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._record_error("runtime_exception", exc)
            raise
        finally:
            signal.signal = original_signal

    async def _status_monitor(self) -> None:
        while not self.stop_event.is_set():
            if self.controller.status.state == BotState.ERROR:
                return
            s = self.controller.status

            equity = float(s.equity)
            peak = float(s.peak_equity)
            drawdown_pct = float(s.drawdown_pct)
            leverage = float(s.leverage)
            pnl_today = float(s.pnl_today)
            open_positions = len(s.exposure_by_symbol)
            uptime_seconds = (
                max(0.0, time.monotonic() - self._runtime_started_monotonic)
                if self._runtime_started_monotonic
                else float(s.uptime_seconds)
            )

            self.controller.update_status(
                equity=equity,
                peak_equity=peak,
                drawdown_pct=drawdown_pct,
                leverage=leverage,
                pnl_today=pnl_today,
                exposure_by_symbol=dict(s.exposure_by_symbol),
                state=BotState.RUNNING,
                mode="live" if self.config.mode == "live" else "paper",
                started_at=s.started_at or datetime.now(timezone.utc).isoformat(),
                uptime_seconds=uptime_seconds,
            )
            self._logger.debug(
                "update_status heartbeat equity=%.4f peak=%.4f dd=%.4f lev=%.2f positions=%d mode=%s uptime=%.1f",
                equity,
                peak,
                drawdown_pct,
                leverage,
                open_positions,
                "live" if self.config.mode == "live" else "paper",
                uptime_seconds,
            )

            # Periodically sync real account data for dashboard visibility.
            if self.config.mode == "live" and (time.monotonic() - self._last_account_sync) >= 10.0:
                self._last_account_sync = time.monotonic()
                await self._sync_live_account_snapshot()

            self._last_status_update = time.monotonic()

            if drawdown_pct > 30.0:
                send_telegram_alert("Drawdown above 30%", {"drawdown_pct": round(drawdown_pct, 2)})

            if equity > 0 and equity < self._starting_capital * 0.5:
                send_telegram_alert("Equity floor guard triggered", {"equity": round(equity, 2)})
                self.stop_event.set()
                self._request_stop()
                return

            max_lev = max((float(ex.leverage) for ex in self.config.exchanges if ex.enabled), default=0.0)
            if leverage > max_lev > 0:
                send_telegram_alert("Leverage guard exceeded", {"leverage": leverage, "max": max_lev})

            if (time.monotonic() - self._last_status_update) > 600:
                send_telegram_alert("Heartbeat watchdog triggered", {"reason": "no status update >10m"})

            for alert in self.audit.get_recent_alerts(20):
                alert_num = int(alert.get("alert_number", 0))
                if alert_num <= self._last_seen_audit_alert:
                    continue
                self._last_seen_audit_alert = alert_num
                a_type = str(alert.get("alert_type", ""))
                details = alert.get("details", {})
                if a_type.startswith("CB"):
                    send_telegram_alert("Circuit breaker fired", details if isinstance(details, dict) else None)
                if "REJECT" in a_type:
                    send_telegram_alert("Order rejected", details if isinstance(details, dict) else None)

            await asyncio.sleep(2.0)

    async def _sync_live_account_snapshot(self) -> None:
        ex_cfg = next((ex for ex in self.config.exchanges if ex.enabled), None)
        if ex_cfg is None:
            return

        adapter = BinanceAdapter(
            api_key=ex_cfg.api_key,
            api_secret=ex_cfg.api_secret,
            testnet=bool(ex_cfg.testnet),
        )
        try:
            positions = await adapter.get_positions()
            wallet_balance = float(await adapter.get_balance())
            unrealized_pnl = 0.0
            exposure_by_symbol = {}
            leverage_values = []
            for p in positions:
                side_mult = -1.0 if str(getattr(p, "side", "")).endswith("SELL") else 1.0
                size = float(getattr(p, "size", 0.0))
                exposure_by_symbol[str(getattr(p, "symbol", ""))] = side_mult * size
                unrealized_pnl += float(getattr(p, "unrealized_pnl", 0.0))
                leverage_values.append(float(getattr(p, "leverage", 0.0)))

            equity = wallet_balance + unrealized_pnl
            if not (equity == equity):  # NaN guard
                equity = wallet_balance
            peak = max(float(self.controller.status.peak_equity), equity)
            drawdown_pct = ((peak - equity) / peak * 100.0) if peak > 0 else 0.0
            leverage = max(leverage_values) if leverage_values else float(ex_cfg.leverage)
            uptime_seconds = (
                max(0.0, time.monotonic() - self._runtime_started_monotonic)
                if self._runtime_started_monotonic
                else float(self.controller.status.uptime_seconds)
            )

            self._logger.debug(
                "Live account sync wallet_balance=%.4f unrealized_pnl=%.4f equity=%.4f positions=%d",
                wallet_balance,
                unrealized_pnl,
                equity,
                len(exposure_by_symbol),
            )
            self.controller.update_status(
                equity=equity,
                peak_equity=peak,
                drawdown_pct=drawdown_pct,
                leverage=leverage,
                pnl_today=unrealized_pnl,
                exposure_by_symbol=exposure_by_symbol,
                state=BotState.RUNNING,
                mode="live",
                started_at=self.controller.status.started_at or datetime.now(timezone.utc).isoformat(),
                uptime_seconds=uptime_seconds,
            )
            self._logger.debug(
                "Live update_status pushed equity=%.4f peak=%.4f dd=%.4f lev=%.2f uptime=%.1f",
                equity,
                peak,
                drawdown_pct,
                leverage,
                uptime_seconds,
            )
        except Exception as exc:
            self._record_error("account_sync_failed", exc)
            self.controller.update_status(last_alert=f"Live account sync failed: {str(exc)[:160]}")
        finally:
            await adapter.close()

    def _record_error(self, alert_type: str, exc: Exception) -> None:
        details = {"error": str(exc)[:400]}
        try:
            self.audit._fire_alert(alert_type, details)
        except Exception:
            pass
        self._logger.exception("Darwin runtime failure: %s", exc)
        send_telegram_alert("Exception in trading loop", details)
        self.controller.update_status(state=BotState.ERROR, last_alert=str(exc)[:200])
