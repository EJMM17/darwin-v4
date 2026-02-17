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
