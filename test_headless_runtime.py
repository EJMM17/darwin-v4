import asyncio
import logging

import pytest

from darwin_agent.core.engine import DarwinCoreEngine, EngineConfig, EquitySnapshot
from darwin_agent.credentials_loader import CredentialsError, load_runtime_credentials
from darwin_agent.infrastructure.telegram_notifier import TelegramNotifier


def test_start_without_credentials_fails_cleanly(monkeypatch, tmp_path):
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_API_SECRET", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    with pytest.raises(CredentialsError):
        load_runtime_credentials(logging.getLogger("test"), path=tmp_path / "runtime_credentials.json")


def test_start_with_env_credentials_works_in_daemon_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("BINANCE_API_KEY", "k")
    monkeypatch.setenv("BINANCE_API_SECRET", "s")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    creds = load_runtime_credentials(logging.getLogger("test"), path=tmp_path / "runtime_credentials.json")
    assert creds.binance_api_key == "k"
    assert creds.telegram_chat_id == "123"


def test_trade_open_triggers_telegram_message(monkeypatch):
    sent = []

    def fake_send(self, text):
        sent.append(text)

    monkeypatch.setattr(TelegramNotifier, "send", fake_send)
    notifier = TelegramNotifier("token", "chat")
    notifier.notify_trade_opened("BTCUSDT", "BUY", 100.0, 0.2, 1.0, 1000.0)

    assert "🟢 TRADE OPENED" in sent[0]
    assert "Symbol: BTCUSDT" in sent[0]


def test_trade_close_triggers_telegram_message(monkeypatch):
    sent = []

    def fake_send(self, text):
        sent.append(text)

    monkeypatch.setattr(TelegramNotifier, "send", fake_send)
    notifier = TelegramNotifier("token", "chat")
    notifier.notify_trade_closed("BTCUSDT", 105.0, 10.0, 1010.0)

    assert "🔴 TRADE CLOSED" in sent[0]
    assert "New Equity: 1010.0" in sent[0]


def test_invalid_api_key_aborts_before_trading(monkeypatch):
    import darwin_agent.main as dm

    class FakeRuntime:
        async def run_forever(self):
            raise AssertionError("must not enter trading loop")

        def stop(self):
            return None

    class FakeBinance:
        def close(self):
            return None

    class FakeTelegram:
        def notify_error(self, _):
            return None

        def close(self):
            return None

    def bad_startup(*args, **kwargs):
        raise RuntimeError("Invalid API-key")

    monkeypatch.setattr(dm, "_create_runtime", lambda logger: (FakeRuntime(), FakeBinance(), FakeTelegram(), object()))
    monkeypatch.setattr(dm, "startup_validation", bad_startup)

    code = asyncio.run(dm.run_live(logging.getLogger("test")))
    assert code == 1


def test_leverage_forced_to_5x(monkeypatch):
    import darwin_agent.main as dm

    class FakeBinance:
        def validate_startup(self, symbols, leverage):
            self.got = leverage
            return {"wallet_balance": 100.0, "open_positions": [], "leverage_result": {s: leverage == 5 for s in symbols}}

    class FakeTelegram:
        def notify_engine_connected(self, payload):
            self.payload = payload

    b = FakeBinance()
    t = FakeTelegram()
    dm.startup_validation(b, t, leverage=5, symbols=["BTCUSDT"], logger=logging.getLogger("test"))
    assert b.got == 5


def test_equity_uses_wallet_plus_unrealized_pnl():
    engine = DarwinCoreEngine(EngineConfig(risk_percent=1.0, leverage=5))
    snapshot = EquitySnapshot(wallet_balance=200.0, unrealized_pnl=-25.0)
    ctx = engine.evaluate(snapshot, [])
    assert ctx["equity"] == 175.0
    assert ctx["position_size"] == pytest.approx(1.75)
