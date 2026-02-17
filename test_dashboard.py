"""
Tests for Darwin v4 Dashboard.

Covers: CryptoVault, Database, BotController, App endpoints, WebSocket.
"""

import json
import os
import tempfile
import time

import pytest
from fastapi.testclient import TestClient


# ═══ CryptoVault ═══

class TestCryptoVault:
    def test_encrypt_decrypt(self):
        from cryptography.fernet import Fernet
        key = Fernet.generate_key().decode()
        os.environ["DASHBOARD_SECRET_KEY"] = key
        from dashboard.crypto_vault import CryptoVault
        v = CryptoVault()
        enc = v.encrypt("my-api-key-12345")
        assert enc != "my-api-key-12345"
        dec = v.decrypt(enc)
        assert dec == "my-api-key-12345"

    def test_mask(self):
        os.environ["DASHBOARD_SECRET_KEY"] = "test-key-for-hashing"
        from dashboard.crypto_vault import CryptoVault
        v = CryptoVault()
        assert v.mask("abcd1234efgh5678") == "abcd****5678"
        assert v.mask("short") == "****"

    def test_missing_key_raises(self):
        old = os.environ.pop("DASHBOARD_SECRET_KEY", None)
        try:
            from importlib import reload
            import dashboard.crypto_vault as cv
            reload(cv)
            with pytest.raises(RuntimeError):
                cv.CryptoVault(key="")
        finally:
            if old:
                os.environ["DASHBOARD_SECRET_KEY"] = old


# ═══ Database ═══

class TestDatabase:
    def _make_db(self):
        from dashboard.database import Database
        td = tempfile.mkdtemp()
        return Database(os.path.join(td, "test.db"))

    def test_credential_crud(self):
        db = self._make_db()
        cid = db.save_credential("binance", "enc_key", "enc_secret", testnet=True)
        assert cid > 0

        creds = db.list_credentials()
        assert len(creds) == 1
        assert creds[0]["exchange"] == "binance"

        deleted = db.delete_credential(cid)
        assert deleted
        assert len(db.list_credentials()) == 0

    def test_user_management(self):
        db = self._make_db()
        uid = db.create_user("admin", "hashedpw123")
        assert uid > 0

        user = db.get_user("admin")
        assert user is not None
        assert user["username"] == "admin"
        assert user["password_hash"] == "hashedpw123"

        assert db.get_user("nonexistent") is None

    def test_event_logging(self):
        db = self._make_db()
        db.log_event("admin", "BOT_START", "127.0.0.1", True, {"mode": "paper"})
        events = db.recent_events(10)
        assert len(events) == 1
        assert events[0]["action"] == "BOT_START"

    def test_equity_snapshots(self):
        db = self._make_db()
        db.save_equity_snapshot(850.0, 900.0, 5.5, 3.2, 12.50)
        db.save_equity_snapshot(860.0, 900.0, 4.4, 3.0, 22.50)
        history = db.get_equity_history(10)
        assert len(history) == 2
        assert history[0]["equity"] == 850.0
        assert history[1]["equity"] == 860.0


# ═══ BotController ═══

class TestBotController:
    def test_start_stop(self):
        from dashboard.bot_controller import BotController, BotState
        ctrl = BotController()

        def runner(stop_event, controller):
            controller.update_status(equity=850.0, peak_equity=900.0)
            while not stop_event.is_set():
                time.sleep(0.05)

        ctrl.set_runner(runner)
        result = ctrl.start("paper")
        assert result["ok"]
        time.sleep(0.2)
        assert ctrl.is_running

        s = ctrl.status
        assert s.equity == 850.0

        result = ctrl.stop()
        assert result["ok"]
        assert not ctrl.is_running

    def test_emergency_lock(self):
        from dashboard.bot_controller import BotController, BotState
        ctrl = BotController()

        def runner(stop_event, controller):
            while not stop_event.is_set():
                time.sleep(0.05)

        ctrl.set_runner(runner)
        ctrl.start("paper")
        time.sleep(0.1)

        result = ctrl.emergency_close()
        assert result["ok"]
        assert ctrl.is_locked

        # Cannot start while locked
        result = ctrl.start("paper")
        assert not result["ok"]

        # Unlock
        result = ctrl.unlock()
        assert result["ok"]
        assert not ctrl.is_locked

    def test_double_start(self):
        from dashboard.bot_controller import BotController
        ctrl = BotController()
        ctrl.set_runner(lambda s, c: None)

        result = ctrl.start()
        # The runner returns immediately, so the bot stops
        time.sleep(0.1)
        # Should be able to restart
        result = ctrl.start()

    def test_no_runner(self):
        from dashboard.bot_controller import BotController
        ctrl = BotController()
        result = ctrl.start()
        assert not result["ok"]
        assert "runner" in result["error"].lower()


# ═══ FastAPI App ═══

class TestApp:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        from cryptography.fernet import Fernet
        os.environ["DASHBOARD_SECRET_KEY"] = Fernet.generate_key().decode()
        os.environ["DASHBOARD_DB_PATH"] = str(tmp_path / "test.db")
        os.environ["DASHBOARD_ADMIN_PASSWORD"] = "testpass123"

        # Reimport to pick up new env vars
        from importlib import reload
        import dashboard.database
        import dashboard.app
        reload(dashboard.database)
        reload(dashboard.app)

        # Reinitialize globals
        dashboard.app.db = dashboard.database.Database(str(tmp_path / "test.db"))
        from dashboard.crypto_vault import CryptoVault
        dashboard.app.vault = CryptoVault()
        dashboard.app.sessions.clear()

        if dashboard.app.db.user_count() == 0:
            dashboard.app._create_default_admin()

        self.client = TestClient(dashboard.app.app)

    def _login(self):
        r = self.client.post("/api/login", json={
            "username": "admin", "password": "testpass123"
        })
        assert r.status_code == 200
        d = r.json()
        return d["csrf_token"]

    def test_login_success(self):
        csrf = self._login()
        assert csrf

    def test_login_failure(self):
        r = self.client.post("/api/login", json={
            "username": "admin", "password": "wrong"
        })
        assert r.status_code == 401

    def test_auth_required(self):
        r = self.client.get("/bot/status")
        assert r.status_code == 401

    def test_bot_status(self):
        self._login()
        r = self.client.get("/bot/status")
        assert r.status_code == 200
        d = r.json()
        assert "running" in d
        assert "equity" in d
        assert "drawdown_pct" in d

    def test_credential_crud(self):
        csrf = self._login()

        # Create
        r = self.client.post("/api/credentials",
            json={"exchange": "binance", "api_key": "key123", "secret_key": "sec456"},
            headers={"x-csrf-token": csrf})
        assert r.status_code == 200
        cred_id = r.json()["id"]

        # List
        r = self.client.get("/api/credentials")
        assert r.status_code == 200
        creds = r.json()["credentials"]
        assert len(creds) == 1
        # Verify no decrypted keys in response
        assert "key123" not in json.dumps(creds)
        assert "sec456" not in json.dumps(creds)

        # Delete
        r = self.client.delete("/api/credentials/{}".format(cred_id),
            headers={"x-csrf-token": csrf})
        assert r.status_code == 200

    def test_csrf_protection(self):
        self._login()
        # POST without CSRF should fail
        r = self.client.post("/bot/start", json={"mode": "paper"})
        assert r.status_code == 403

    def test_bot_lifecycle(self):
        csrf = self._login()
        import dashboard.app as dapp

        def runner(stop_event, controller):
            while not stop_event.is_set():
                time.sleep(0.05)

        dapp.controller.set_runner(runner)

        # Start
        r = self.client.post("/bot/start",
            json={"mode": "paper"}, headers={"x-csrf-token": csrf})
        assert r.json()["ok"]
        time.sleep(0.2)

        # Status
        r = self.client.get("/bot/status")
        assert r.json()["running"]

        # Stop
        r = self.client.post("/bot/stop", headers={"x-csrf-token": csrf})
        assert r.json()["ok"]

    def test_emergency_close(self):
        csrf = self._login()
        import dashboard.app as dapp

        def runner(stop_event, controller):
            while not stop_event.is_set():
                time.sleep(0.05)

        dapp.controller.set_runner(runner)
        self.client.post("/bot/start",
            json={"mode": "paper"}, headers={"x-csrf-token": csrf})
        time.sleep(0.1)

        r = self.client.post("/bot/emergency-close",
            headers={"x-csrf-token": csrf})
        assert r.json()["ok"]

        # Check locked
        r = self.client.get("/bot/status")
        assert r.json()["state"] == "emergency_locked"

        # Unlock
        r = self.client.post("/bot/unlock",
            headers={"x-csrf-token": csrf})
        assert r.json()["ok"]

    def test_index_page(self):
        r = self.client.get("/")
        assert r.status_code == 200
        assert "DARWIN" in r.text

    def test_equity_history(self):
        self._login()
        r = self.client.get("/api/equity-history")
        assert r.status_code == 200
        assert "history" in r.json()

    def test_events(self):
        csrf = self._login()
        import dashboard.app as dapp

        def runner(stop_event, controller):
            while not stop_event.is_set():
                time.sleep(0.05)

        dapp.controller.set_runner(runner)
        # Bot start logs to DB
        self.client.post("/bot/start",
            json={"mode": "paper"}, headers={"x-csrf-token": csrf})
        time.sleep(0.1)
        self.client.post("/bot/stop", headers={"x-csrf-token": csrf})

        r = self.client.get("/api/events")
        assert r.status_code == 200
        events = r.json()["events"]
        assert len(events) >= 1
        actions = [e["action"] for e in events]
        assert "BOT_START" in actions


# ═══ DashboardLogger ═══

class TestDashboardLogger:
    def test_writes_jsonl(self):
        from dashboard.dash_logger import DashboardLogger
        td = tempfile.mkdtemp()
        logger = DashboardLogger(log_dir=td)
        logger.log("admin", "TEST_ACTION", "127.0.0.1", True, {"key": "val"})
        logger.close()

        files = [f for f in os.listdir(td) if f.startswith("dashboard-")]
        assert len(files) == 1
        with open(os.path.join(td, files[0])) as f:
            record = json.loads(f.readline())
            assert record["user"] == "admin"
            assert record["action"] == "TEST_ACTION"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_bot_start_live_with_valid_credentials_sets_equity(monkeypatch, tmp_path):
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
        encrypted_api_key=vault.encrypt("good-key"),
        encrypted_secret_key=vault.encrypt("good-secret"),
        testnet=False,
    )

    async def ok_validate(self):
        return None

    monkeypatch.setattr(dashboard.app.BinanceAdapter, "validate_live_credentials", ok_validate)

    runtime = dashboard.app._ensure_runtime()

    def fake_start(mode="live"):
        dashboard.app.controller.update_status(
            equity=125.0,
            peak_equity=125.0,
            drawdown_pct=0.0,
            exposure_by_symbol={"BTCUSDT": 0.01},
            leverage=5.0,
            mode="LIVE",
            uptime_seconds=1.0,
        )
        return True

    monkeypatch.setattr(runtime, "start", fake_start)

    client = TestClient(dashboard.app.app)
    with client:
        r = client.post("/api/login", json={"username": "admin", "password": "pw"})
        csrf = r.json()["csrf_token"]
        r = client.post("/bot/start", json={"mode": "live"}, headers={"x-csrf-token": csrf})
        assert r.status_code == 200
        assert r.json()["ok"] is True

        status = client.get("/bot/status").json()
        assert status["equity"] > 0
        assert status["mode"] == "LIVE"
        assert "BTCUSDT" in status["exposure_by_symbol"]
