import os
import threading
import time

from fastapi.testclient import TestClient


def test_telegram_alert_mock_send(monkeypatch):
    from dashboard import alerts

    sent = []

    def fake_post(url, json, timeout):
        sent.append((url, json, timeout))
        class R:
            status_code = 200
        return R()

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat")
    monkeypatch.setattr(alerts.requests, "post", fake_post)

    alerts.send_telegram_alert("hello", {"x": 1})
    time.sleep(0.1)
    assert sent
    assert "sendMessage" in sent[0][0]


def test_runtime_start_stop_and_no_thread_leak(monkeypatch):
    import dashboard.bot_runtime as br
    from dashboard.bot_controller import BotController
    from darwin_agent.monitoring.execution_audit import ExecutionAudit

    async def fake_run(config):
        while True:
            await br.asyncio.sleep(0.05)

    monkeypatch.setattr(br, "darwin_run", fake_run)
    monkeypatch.setattr(br.DarwinRuntime, "_enforce_live_binance", lambda self: None)

    ctrl = BotController()
    audit = ExecutionAudit(log_dir="logs/audit-test")
    runtime = br.DarwinRuntime(controller=ctrl, audit=audit)

    before = {t.name for t in threading.enumerate()}
    assert runtime.start() is True
    time.sleep(0.2)
    assert runtime.is_running()

    assert runtime.stop() is True
    time.sleep(0.2)
    assert not runtime.is_running()
    after = {t.name for t in threading.enumerate()}
    assert "darwin-runtime" not in after - before


def test_runtime_graceful_cancellation_and_audit_trigger(monkeypatch):
    import dashboard.bot_runtime as br
    from dashboard.bot_controller import BotController, BotState
    from darwin_agent.monitoring.execution_audit import ExecutionAudit

    async def broken_run(config):
        raise RuntimeError("boom")

    monkeypatch.setattr(br, "darwin_run", broken_run)
    monkeypatch.setattr(br.DarwinRuntime, "_enforce_live_binance", lambda self: None)

    ctrl = BotController()
    audit = ExecutionAudit(log_dir="logs/audit-test-2")
    runtime = br.DarwinRuntime(controller=ctrl, audit=audit)

    runtime.start()
    time.sleep(0.3)
    assert ctrl.status.state == BotState.ERROR
    assert audit.get_metrics()["darwin_alerts_total"] >= 1


def test_metrics_endpoint(monkeypatch, tmp_path):
    from cryptography.fernet import Fernet

    monkeypatch.setenv("DASHBOARD_SECRET_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("DASHBOARD_DB_PATH", str(tmp_path / "db.sqlite"))
    monkeypatch.setenv("DASHBOARD_ADMIN_PASSWORD", "pw")

    from importlib import reload
    import dashboard.database
    import dashboard.app
    reload(dashboard.database)
    reload(dashboard.app)

    client = TestClient(dashboard.app.app)
    with client:
        r = client.get("/metrics")
        assert r.status_code == 200
        text = r.text
        assert "darwin_equity" in text
        assert "darwin_drawdown_pct" in text


def test_runtime_enforces_live_binance_from_dashboard_credentials(monkeypatch, tmp_path):
    from cryptography.fernet import Fernet

    monkeypatch.setenv("DASHBOARD_SECRET_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("DASHBOARD_DB_PATH", str(tmp_path / "db.sqlite"))

    from dashboard.database import Database
    from dashboard.crypto_vault import CryptoVault
    import dashboard.bot_runtime as br
    from dashboard.bot_controller import BotController
    from darwin_agent.monitoring.execution_audit import ExecutionAudit

    db = Database(str(tmp_path / "db.sqlite"))
    vault = CryptoVault()
    db.save_credential(
        exchange="binance",
        encrypted_api_key=vault.encrypt("live-key"),
        encrypted_secret_key=vault.encrypt("live-secret"),
        testnet=False,
    )

    ctrl = BotController()
    audit = ExecutionAudit(log_dir="logs/audit-test-3")
    runtime = br.DarwinRuntime(controller=ctrl, audit=audit)

    runtime._enforce_live_binance()

    assert runtime.config.mode == "live"
    assert len(runtime.config.exchanges) == 1
    assert runtime.config.exchanges[0].exchange_id == "binance"
    assert runtime.config.exchanges[0].testnet is False
    assert runtime.config.exchanges[0].api_key == "live-key"
    assert runtime.config.exchanges[0].api_secret == "live-secret"


def test_runtime_sets_started_at_and_uptime(monkeypatch):
    import dashboard.bot_runtime as br
    from dashboard.bot_controller import BotController
    from darwin_agent.monitoring.execution_audit import ExecutionAudit

    async def fake_run(config):
        while True:
            await br.asyncio.sleep(0.05)

    monkeypatch.setattr(br, "darwin_run", fake_run)
    monkeypatch.setattr(br.DarwinRuntime, "_enforce_live_binance", lambda self: None)

    ctrl = BotController()
    audit = ExecutionAudit(log_dir="logs/audit-test-uptime")
    runtime = br.DarwinRuntime(controller=ctrl, audit=audit)

    assert runtime.start(mode="paper") is True
    time.sleep(0.3)

    status = ctrl.status
    assert status.mode == "paper"
    assert status.started_at
    assert status.uptime_seconds > 0

    assert runtime.stop() is True


def test_live_sync_sets_positive_equity_and_calls_update_status(monkeypatch, tmp_path):
    from cryptography.fernet import Fernet

    monkeypatch.setenv("DASHBOARD_SECRET_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("DASHBOARD_DB_PATH", str(tmp_path / "db.sqlite"))

    import dashboard.bot_runtime as br
    from dashboard.database import Database
    from dashboard.crypto_vault import CryptoVault
    from dashboard.bot_controller import BotController
    from darwin_agent.monitoring.execution_audit import ExecutionAudit

    db = Database(str(tmp_path / "db.sqlite"))
    vault = CryptoVault()
    db.save_credential(
        exchange="binance",
        encrypted_api_key=vault.encrypt("live-key"),
        encrypted_secret_key=vault.encrypt("live-secret"),
        testnet=False,
    )

    class FakeAdapter:
        def __init__(self, *args, **kwargs):
            pass

        async def get_positions(self):
            from darwin_agent.interfaces.types import Position
            from darwin_agent.interfaces.enums import OrderSide, ExchangeID
            return [
                Position(
                    symbol="BTCUSDT",
                    side=OrderSide.BUY,
                    size=0.01,
                    entry_price=50000,
                    current_price=51000,
                    unrealized_pnl=5.0,
                    leverage=10,
                    exchange_id=ExchangeID.BINANCE,
                )
            ]

        async def get_wallet_balance_and_upnl(self):
            return 100.0, 5.0

        async def close(self):
            return None

    monkeypatch.setattr(br, "BinanceAdapter", FakeAdapter)

    ctrl = BotController()
    audit = ExecutionAudit(log_dir="logs/audit-test-live-sync")
    runtime = br.DarwinRuntime(controller=ctrl, audit=audit)
    runtime._enforce_live_binance()

    calls = []
    orig_update = ctrl.update_status

    def spy_update_status(**kwargs):
        calls.append(kwargs)
        orig_update(**kwargs)

    monkeypatch.setattr(ctrl, "update_status", spy_update_status)

    import asyncio
    asyncio.run(runtime._sync_live_account_snapshot())

    assert ctrl.status.equity > 0
    assert any({"equity", "peak_equity", "drawdown_pct", "exposure_by_symbol", "leverage", "mode", "uptime_seconds"}.issubset(set(c.keys())) for c in calls)


def test_live_start_rejects_invalid_binance_credentials(monkeypatch, tmp_path):
    from cryptography.fernet import Fernet

    monkeypatch.setenv("DASHBOARD_SECRET_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("DASHBOARD_DB_PATH", str(tmp_path / "db.sqlite"))
    monkeypatch.setenv("DASHBOARD_ADMIN_PASSWORD", "pw")

    from importlib import reload
    import dashboard.database
    import dashboard.app
    reload(dashboard.database)
    reload(dashboard.app)

    db = dashboard.app.db
    vault = dashboard.app.CryptoVault()
    db.save_credential(
        exchange="binance",
        encrypted_api_key=vault.encrypt("bad-key"),
        encrypted_secret_key=vault.encrypt("bad-secret"),
        testnet=False,
    )

    async def bad_validate(self):
        raise RuntimeError("Invalid API-key, IP, or permissions for action")

    monkeypatch.setattr(dashboard.app.BinanceAdapter, "validate_live_credentials", bad_validate)

    client = TestClient(dashboard.app.app)
    with client:
        r = client.post("/api/login", json={"username": "admin", "password": "pw"})
        csrf = r.json()["csrf_token"]
        r = client.post("/bot/start", json={"mode": "live"}, headers={"x-csrf-token": csrf})
        assert r.status_code in (400, 422)
        assert "failed" in r.json()["detail"].lower() or "invalid" in r.json()["detail"].lower()
