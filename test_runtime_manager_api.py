import time
from importlib import reload

from fastapi.testclient import TestClient


def _boot_app(monkeypatch, tmp_path):
    from cryptography.fernet import Fernet

    monkeypatch.setenv("DASHBOARD_SECRET_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("DASHBOARD_DB_PATH", str(tmp_path / "db.sqlite"))
    monkeypatch.setenv("DASHBOARD_ADMIN_PASSWORD", "pw")

    import dashboard.database
    import dashboard.app

    reload(dashboard.database)
    reload(dashboard.app)
    return dashboard.app


def _login(client):
    r = client.post("/api/login", json={"username": "admin", "password": "pw"})
    assert r.status_code == 200
    return r.json()["csrf_token"]


def test_start_success(monkeypatch, tmp_path):
    app_mod = _boot_app(monkeypatch, tmp_path)
    import dashboard.bot_runtime as br

    async def fake_run(config):
        while True:
            await br.asyncio.sleep(0.05)

    async def noop(self):
        return None

    monkeypatch.setattr(br, "darwin_run", fake_run)
    monkeypatch.setattr(br.RuntimeManager, "_set_hard_leverage", noop)
    monkeypatch.setattr(br.RuntimeManager, "_update_live_snapshot", noop)

    client = TestClient(app_mod.app)
    with client:
        csrf = _login(client)
        r = client.post("/bot/start", json={"mode": "test"}, headers={"x-csrf-token": csrf})
        assert r.status_code == 200
        assert r.json()["ok"] is True
        stop = client.post("/bot/stop", headers={"x-csrf-token": csrf})
        assert stop.status_code == 200


def test_start_missing_credentials_live(monkeypatch, tmp_path):
    app_mod = _boot_app(monkeypatch, tmp_path)

    client = TestClient(app_mod.app)
    with client:
        csrf = _login(client)
        r = client.post("/bot/start", json={"mode": "live"}, headers={"x-csrf-token": csrf})
        assert r.status_code == 400
        assert "missing" in r.json()["detail"].lower() or "required" in r.json()["detail"].lower()


def test_stop_works(monkeypatch, tmp_path):
    app_mod = _boot_app(monkeypatch, tmp_path)
    import dashboard.bot_runtime as br

    async def fake_run(config):
        while True:
            await br.asyncio.sleep(0.05)

    async def noop(self):
        return None

    monkeypatch.setattr(br, "darwin_run", fake_run)
    monkeypatch.setattr(br.RuntimeManager, "_set_hard_leverage", noop)
    monkeypatch.setattr(br.RuntimeManager, "_update_live_snapshot", noop)

    client = TestClient(app_mod.app)
    with client:
        csrf = _login(client)
        client.post("/bot/start", json={"mode": "test"}, headers={"x-csrf-token": csrf})
        r = client.post("/bot/stop", headers={"x-csrf-token": csrf})
        assert r.status_code == 200
        assert r.json()["ok"] is True


def test_status_endpoint_structure(monkeypatch, tmp_path):
    app_mod = _boot_app(monkeypatch, tmp_path)
    client = TestClient(app_mod.app)
    with client:
        _login(client)
        r = client.get("/bot/status")
        assert r.status_code == 200
        payload = r.json()
        required = {"is_running", "current_equity", "wallet_balance", "positions", "drawdown", "last_update", "mode"}
        assert required.issubset(set(payload.keys()))


def test_live_equity_calculation_uses_wallet_plus_upnl(monkeypatch, tmp_path):
    import dashboard.bot_runtime as br
    from dashboard.bot_controller import BotController
    from darwin_agent.monitoring.execution_audit import ExecutionAudit

    class FakeAdapter:
        def __init__(self, *args, **kwargs):
            pass

        async def get_positions(self):
            return []

        async def get_wallet_balance_and_upnl(self):
            return 110.0, -7.0

        async def close(self):
            return None

    runtime = br.RuntimeManager(controller=BotController(), audit=ExecutionAudit(log_dir="logs/audit-runtime-manager"))
    runtime.config.mode = "live"
    runtime.config.exchanges[0].enabled = True
    runtime.config.exchanges[0].api_key = "k"
    runtime.config.exchanges[0].api_secret = "s"

    monkeypatch.setattr(br, "BinanceAdapter", FakeAdapter)

    import asyncio

    asyncio.run(runtime._update_live_snapshot())
    assert runtime.get_runtime_status()["current_equity"] == 103.0
