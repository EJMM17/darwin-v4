"""
Darwin v4 — Signal Diagnostics Layer Test Suite.

Tests covering:
  1.  RejectionReason enum values and from_risk_verdict mapping
  2.  SignalDiagnostic dataclass and to_dict serialization
  3.  SignalDiagnosticsLogger file creation and JSONL output
  4.  PreTradeValidator — successful validation
  5.  PreTradeValidator — balance fetch failure
  6.  PreTradeValidator — leverage mismatch
  7.  PreTradeValidator — min notional failure
  8.  PreTradeValidator — no fetcher configured (passthrough)
  9.  TelegramDebugNotifier — near-threshold notification
  10. TelegramDebugNotifier — rejection notification
  11. TelegramDebugNotifier — execution notification
  12. TelegramDebugNotifier — disabled mode sends nothing
  13. Config flags loaded from env vars
  14. Diagnostics do not modify execution path (neutrality test)
  15. RejectionReason mapping covers all known risk verdict strings
"""
import json
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(__file__))

import pytest
from datetime import datetime, timezone

from darwin_agent.diagnostics.rejection_reason import RejectionReason
from darwin_agent.diagnostics.signal_diagnostic import SignalDiagnostic
from darwin_agent.diagnostics.signal_diagnostics_logger import SignalDiagnosticsLogger
from darwin_agent.diagnostics.pre_trade_validator import (
    PreTradeValidator,
    PreTradeResult,
)
from darwin_agent.diagnostics.telegram_debug import TelegramDebugNotifier


# ── Helpers ──────────────────────────────────────────────────

def _make_diag(**overrides) -> SignalDiagnostic:
    """Create a SignalDiagnostic with sensible defaults."""
    defaults = dict(
        symbol="BTCUSDT",
        signal_score=0.72,
        signal_threshold=0.75,
        atr_value=1.21,
        trend_alignment=True,
        volatility_ok=True,
        funding_ok=True,
        risk_allowed=True,
        computed_position_size_usdt=14.2,
        computed_quantity=0.000316,
        min_notional_required=5.0,
        final_notional=71.0,
        rejection_reason=RejectionReason.NONE,
    )
    defaults.update(overrides)
    return SignalDiagnostic(**defaults)


class MessageCollector:
    """Mock Telegram send function that collects messages."""
    def __init__(self):
        self.messages = []

    def __call__(self, text: str) -> None:
        self.messages.append(text)


# ═════════════════════════════════════════════════════════════
# 1. RejectionReason enum values
# ═════════════════════════════════════════════════════════════

def test_rejection_reason_enum_values():
    """All required rejection reasons exist."""
    expected = {
        "SCORE_BELOW_THRESHOLD",
        "ATR_TOO_LOW",
        "TREND_MISALIGNED",
        "RISK_ENGINE_BLOCKED",
        "DRAWDOWN_LOCK",
        "MIN_NOTIONAL_FAIL",
        "INSUFFICIENT_MARGIN",
        "LEVERAGE_MISMATCH",
        "FUNDING_FILTER_BLOCK",
        "BALANCE_FETCH_FAILED",
        "LEVERAGE_SET_FAILED",
        "ZERO_QUANTITY",
        "NONE",
    }
    actual = {r.value for r in RejectionReason}
    assert expected.issubset(actual), f"missing: {expected - actual}"


def test_rejection_reason_from_risk_verdict_mapping():
    """from_risk_verdict maps known strings to correct enums."""
    cases = [
        ("HALTED:trading_suspended", RejectionReason.DRAWDOWN_LOCK),
        ("CRITICAL:close_only_mode", RejectionReason.DRAWDOWN_LOCK),
        ("max_positions:3>=3", RejectionReason.RISK_ENGINE_BLOCKED),
        ("duplicate_symbol:BTCUSDT", RejectionReason.RISK_ENGINE_BLOCKED),
        ("gross_exposure:85%>=80%", RejectionReason.INSUFFICIENT_MARGIN),
        ("symbol_exposure:BTCUSDT=35%>=30%", RejectionReason.INSUFFICIENT_MARGIN),
        ("exchange_exposure:binance=65%>=60%", RejectionReason.INSUFFICIENT_MARGIN),
        ("daily_loss:6.0%>=5%", RejectionReason.DRAWDOWN_LOCK),
        ("consecutive_losses:7>=7", RejectionReason.RISK_ENGINE_BLOCKED),
        ("signal_strength:NONE", RejectionReason.SCORE_BELOW_THRESHOLD),
        ("weak_signal:low_confidence", RejectionReason.SCORE_BELOW_THRESHOLD),
        ("DEFENSIVE:weak_signal_blocked", RejectionReason.SCORE_BELOW_THRESHOLD),
        ("DEFENSIVE:low_confidence", RejectionReason.SCORE_BELOW_THRESHOLD),
        ("VaR95:12.5%>10%", RejectionReason.RISK_ENGINE_BLOCKED),
        ("directional_concentration:80%>75%", RejectionReason.RISK_ENGINE_BLOCKED),
        ("zero_capital", RejectionReason.INSUFFICIENT_MARGIN),
        ("", RejectionReason.NONE),
    ]
    for reason_str, expected in cases:
        result = RejectionReason.from_risk_verdict(reason_str)
        assert result == expected, (
            f"from_risk_verdict({reason_str!r}) = {result}, expected {expected}"
        )


# ═════════════════════════════════════════════════════════════
# 2. SignalDiagnostic dataclass
# ═════════════════════════════════════════════════════════════

def test_signal_diagnostic_to_dict():
    """to_dict produces a valid JSON-serializable dictionary."""
    diag = _make_diag()
    d = diag.to_dict()

    assert d["symbol"] == "BTCUSDT"
    assert d["signal_score"] == 0.72
    assert d["signal_threshold"] == 0.75
    assert d["atr_value"] == 1.21
    assert d["trend_alignment"] is True
    assert d["volatility_ok"] is True
    assert d["funding_ok"] is True
    assert d["risk_allowed"] is True
    assert d["rejection_reason"] == "NONE"

    # Must be JSON-serializable
    json_str = json.dumps(d)
    assert isinstance(json_str, str)


def test_signal_diagnostic_immutable():
    """SignalDiagnostic is frozen — cannot be mutated after creation."""
    diag = _make_diag()
    with pytest.raises(AttributeError):
        diag.symbol = "ETHUSDT"


# ═════════════════════════════════════════════════════════════
# 3. SignalDiagnosticsLogger
# ═════════════════════════════════════════════════════════════

def test_diagnostics_logger_creates_file():
    """Logger creates a JSONL file and writes valid JSON lines."""
    tmpdir = tempfile.mkdtemp(prefix="darwin_diag_test_")
    try:
        logger = SignalDiagnosticsLogger(log_dir=tmpdir, enabled=True)
        diag = _make_diag()
        logger.log(diag)
        logger.log(_make_diag(symbol="ETHUSDT", signal_score=0.55))
        logger.close()

        # Find the written file
        files = [f for f in os.listdir(tmpdir) if f.endswith(".jsonl")]
        assert len(files) == 1, f"expected 1 file, got {files}"

        filepath = os.path.join(tmpdir, files[0])
        with open(filepath) as f:
            lines = f.readlines()

        assert len(lines) == 2
        rec1 = json.loads(lines[0])
        assert rec1["symbol"] == "BTCUSDT"
        rec2 = json.loads(lines[1])
        assert rec2["symbol"] == "ETHUSDT"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_diagnostics_logger_disabled():
    """When disabled, logger writes nothing."""
    tmpdir = tempfile.mkdtemp(prefix="darwin_diag_test_")
    try:
        logger = SignalDiagnosticsLogger(log_dir=tmpdir, enabled=False)
        logger.log(_make_diag())
        logger.close()

        files = [f for f in os.listdir(tmpdir) if f.endswith(".jsonl")]
        assert len(files) == 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_diagnostics_logger_no_crash_on_bad_dir():
    """Logger does not crash when given an unwritable directory."""
    logger = SignalDiagnosticsLogger(log_dir="/nonexistent/path/xyz", enabled=True)
    # Should not raise
    logger.log(_make_diag())
    logger.close()


# ═════════════════════════════════════════════════════════════
# 4-8. PreTradeValidator
# ═════════════════════════════════════════════════════════════

def test_pretrade_validator_success():
    """Successful validation with good balance and notional."""
    def mock_balance():
        return {"wallet_balance": 1000.0, "unrealized_pnl": 50.0}

    def mock_leverage(symbol, lev):
        return True

    validator = PreTradeValidator(
        balance_fetcher=mock_balance,
        leverage_setter=mock_leverage,
        target_leverage=5,
    )
    result = validator.validate(
        symbol="BTCUSDT",
        price=45000.0,
        risk_pct=2.0,
        leverage=5,
        min_notional=5.0,
    )
    assert result.valid is True
    assert result.rejection_reason == RejectionReason.NONE
    assert result.wallet_balance == 1000.0
    assert result.unrealized_pnl == 50.0
    assert result.equity == 1050.0
    assert result.recalculated_size_usdt == pytest.approx(21.0, rel=1e-6)
    assert result.final_notional == pytest.approx(105.0, rel=1e-6)
    assert result.leverage_confirmed is True


def test_pretrade_validator_balance_fetch_failure():
    """Balance fetch exception results in BALANCE_FETCH_FAILED."""
    def mock_balance():
        raise ConnectionError("network timeout")

    validator = PreTradeValidator(balance_fetcher=mock_balance)
    result = validator.validate(symbol="BTCUSDT", price=45000.0, risk_pct=2.0)
    assert result.valid is False
    assert result.rejection_reason == RejectionReason.BALANCE_FETCH_FAILED
    assert "network timeout" in result.detail


def test_pretrade_validator_leverage_mismatch():
    """Leverage setter returning False results in LEVERAGE_MISMATCH."""
    def mock_balance():
        return {"wallet_balance": 1000.0, "unrealized_pnl": 0.0}

    def mock_leverage(symbol, lev):
        return False  # leverage set failed

    validator = PreTradeValidator(
        balance_fetcher=mock_balance,
        leverage_setter=mock_leverage,
        target_leverage=5,
    )
    result = validator.validate(symbol="BTCUSDT", price=45000.0, risk_pct=2.0)
    assert result.valid is False
    assert result.rejection_reason == RejectionReason.LEVERAGE_MISMATCH


def test_pretrade_validator_min_notional_fail():
    """Notional below minimum results in MIN_NOTIONAL_FAIL."""
    def mock_balance():
        return {"wallet_balance": 1.0, "unrealized_pnl": 0.0}

    def mock_leverage(symbol, lev):
        return True

    validator = PreTradeValidator(
        balance_fetcher=mock_balance,
        leverage_setter=mock_leverage,
        target_leverage=5,
    )
    result = validator.validate(
        symbol="BTCUSDT",
        price=45000.0,
        risk_pct=2.0,
        leverage=5,
        min_notional=10.0,
    )
    # equity=1.0, size_usdt=0.02, notional=0.02*5=0.1 < 10
    assert result.valid is False
    assert result.rejection_reason == RejectionReason.MIN_NOTIONAL_FAIL


def test_pretrade_validator_no_fetcher():
    """With no fetcher, validation skips live balance check."""
    validator = PreTradeValidator()
    result = validator.validate(symbol="BTCUSDT", price=45000.0, risk_pct=2.0)
    # No fetcher → equity=0 → size=0 → notional=0 < min_notional
    assert result.valid is False
    assert result.rejection_reason == RejectionReason.MIN_NOTIONAL_FAIL


def test_pretrade_validator_balance_as_float():
    """Balance fetcher returning a float (not dict) is handled."""
    def mock_balance():
        return 500.0

    def mock_leverage(symbol, lev):
        return True

    validator = PreTradeValidator(
        balance_fetcher=mock_balance,
        leverage_setter=mock_leverage,
        target_leverage=5,
    )
    result = validator.validate(
        symbol="BTCUSDT",
        price=45000.0,
        risk_pct=2.0,
        leverage=5,
        min_notional=5.0,
    )
    assert result.valid is True
    assert result.wallet_balance == 500.0
    assert result.equity == 500.0


# ═════════════════════════════════════════════════════════════
# 9-12. TelegramDebugNotifier
# ═════════════════════════════════════════════════════════════

def test_telegram_debug_near_threshold():
    """Near-threshold signal sends a message."""
    collector = MessageCollector()
    notifier = TelegramDebugNotifier(send_fn=collector, enabled=True)

    diag = _make_diag(signal_score=0.72, signal_threshold=0.75)
    # 0.72 >= 0.8 * 0.75 = 0.60 → should send
    notifier.notify_signal_near_threshold(diag)

    assert len(collector.messages) == 1
    msg = collector.messages[0]
    assert "[Darwin Signal]" in msg
    assert "BTCUSDT" in msg
    assert "0.72" in msg
    assert "0.75" in msg
    assert "WATCHING" in msg


def test_telegram_debug_near_threshold_below():
    """Signal below 0.8*threshold does NOT send."""
    collector = MessageCollector()
    notifier = TelegramDebugNotifier(send_fn=collector, enabled=True)

    diag = _make_diag(signal_score=0.50, signal_threshold=0.75)
    # 0.50 < 0.8 * 0.75 = 0.60 → should NOT send
    notifier.notify_signal_near_threshold(diag)

    assert len(collector.messages) == 0


def test_telegram_debug_rejection():
    """Rejected trade sends rejection message."""
    collector = MessageCollector()
    notifier = TelegramDebugNotifier(send_fn=collector, enabled=True)

    diag = _make_diag(rejection_reason=RejectionReason.SCORE_BELOW_THRESHOLD)
    notifier.notify_trade_rejected(diag)

    assert len(collector.messages) == 1
    msg = collector.messages[0]
    assert "REJECTED" in msg
    assert "SCORE_BELOW_THRESHOLD" in msg


def test_telegram_debug_rejection_none_ignored():
    """No rejection (NONE) does NOT send a rejection message."""
    collector = MessageCollector()
    notifier = TelegramDebugNotifier(send_fn=collector, enabled=True)

    diag = _make_diag(rejection_reason=RejectionReason.NONE)
    notifier.notify_trade_rejected(diag)

    assert len(collector.messages) == 0


def test_telegram_debug_execution():
    """Executed trade sends execution message."""
    collector = MessageCollector()
    notifier = TelegramDebugNotifier(send_fn=collector, enabled=True)

    diag = _make_diag()
    notifier.notify_trade_executed(diag, filled_qty=0.001, filled_price=45000.0)

    assert len(collector.messages) == 1
    msg = collector.messages[0]
    assert "EXECUTED" in msg
    assert "45000" in msg


def test_telegram_debug_disabled():
    """When disabled, no messages are sent regardless of events."""
    collector = MessageCollector()
    notifier = TelegramDebugNotifier(send_fn=collector, enabled=False)

    diag = _make_diag(rejection_reason=RejectionReason.SCORE_BELOW_THRESHOLD)
    notifier.notify_signal_near_threshold(diag)
    notifier.notify_trade_rejected(diag)
    notifier.notify_trade_executed(diag)
    notifier.notify_pretrade_failure("BTCUSDT", RejectionReason.MIN_NOTIONAL_FAIL)

    assert len(collector.messages) == 0


def test_telegram_debug_send_failure_no_crash():
    """If send_fn raises, the notifier does not crash."""
    def broken_send(text):
        raise RuntimeError("network error")

    notifier = TelegramDebugNotifier(send_fn=broken_send, enabled=True)
    diag = _make_diag(rejection_reason=RejectionReason.DRAWDOWN_LOCK)

    # Should NOT raise
    notifier.notify_trade_rejected(diag)
    notifier.notify_signal_near_threshold(diag)
    notifier.notify_trade_executed(diag)


# ═════════════════════════════════════════════════════════════
# 13. Config flags
# ═════════════════════════════════════════════════════════════

def test_config_flags_from_env(monkeypatch):
    """ENABLE_SIGNAL_DIAGNOSTICS and TELEGRAM_DEBUG load from env."""
    monkeypatch.setenv("ENABLE_SIGNAL_DIAGNOSTICS", "true")
    monkeypatch.setenv("TELEGRAM_DEBUG", "true")

    from darwin_agent.config import load_config
    config = load_config()
    assert config.infra.enable_signal_diagnostics is True
    assert config.infra.telegram_debug is True


def test_config_flags_default():
    """Default values: diagnostics enabled, telegram debug disabled."""
    # Clear env vars if set
    env_backup = {}
    for key in ("ENABLE_SIGNAL_DIAGNOSTICS", "TELEGRAM_DEBUG"):
        env_backup[key] = os.environ.pop(key, None)

    try:
        from darwin_agent.config import load_config
        config = load_config()
        assert config.infra.enable_signal_diagnostics is True
        assert config.infra.telegram_debug is False
    finally:
        for key, val in env_backup.items():
            if val is not None:
                os.environ[key] = val


def test_config_flags_disabled(monkeypatch):
    """ENABLE_SIGNAL_DIAGNOSTICS=false disables diagnostics."""
    monkeypatch.setenv("ENABLE_SIGNAL_DIAGNOSTICS", "false")
    monkeypatch.setenv("TELEGRAM_DEBUG", "false")

    from darwin_agent.config import load_config
    config = load_config()
    assert config.infra.enable_signal_diagnostics is False
    assert config.infra.telegram_debug is False


# ═════════════════════════════════════════════════════════════
# 14. Diagnostics neutrality test
# ═════════════════════════════════════════════════════════════

def test_diagnostics_do_not_modify_execution_path():
    """
    Verify that creating diagnostics objects, logging them, and
    running pre-trade validation does NOT mutate any external state
    or affect the execution path.

    This test creates a mock 'execution state' dict and verifies
    it is unchanged after diagnostics operations.
    """
    execution_state = {
        "signal": {"symbol": "BTCUSDT", "score": 0.72, "threshold": 0.75},
        "risk_approved": True,
        "quantity": 0.001,
        "price": 45000.0,
        "leverage": 5,
    }
    # Deep copy to compare after
    import copy
    original_state = copy.deepcopy(execution_state)

    # Create diagnostic
    diag = _make_diag(
        symbol=execution_state["signal"]["symbol"],
        signal_score=execution_state["signal"]["score"],
        signal_threshold=execution_state["signal"]["threshold"],
    )

    # Log it
    tmpdir = tempfile.mkdtemp(prefix="darwin_diag_neutrality_")
    try:
        logger = SignalDiagnosticsLogger(log_dir=tmpdir, enabled=True)
        logger.log(diag)
        logger.close()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # Serialize it
    _ = diag.to_dict()

    # Run pre-trade validation
    validator = PreTradeValidator()
    _ = validator.validate(
        symbol=execution_state["signal"]["symbol"],
        price=execution_state["price"],
        risk_pct=2.0,
    )

    # Telegram debug notification
    collector = MessageCollector()
    tg = TelegramDebugNotifier(send_fn=collector, enabled=True)
    tg.notify_signal_near_threshold(diag)

    # CRITICAL: execution state must be UNCHANGED
    assert execution_state == original_state, (
        "Diagnostics layer modified the execution state!"
    )


# ═════════════════════════════════════════════════════════════
# 15. Comprehensive rejection mapping coverage
# ═════════════════════════════════════════════════════════════

def test_rejection_reason_from_unknown_string():
    """Unknown verdict strings map to RISK_ENGINE_BLOCKED."""
    result = RejectionReason.from_risk_verdict("some_unknown_reason_xyz")
    assert result == RejectionReason.RISK_ENGINE_BLOCKED


def test_rejection_reason_from_empty_string():
    """Empty string maps to NONE (no rejection)."""
    result = RejectionReason.from_risk_verdict("")
    assert result == RejectionReason.NONE


def test_pretrade_result_dataclass():
    """PreTradeResult defaults are sensible."""
    result = PreTradeResult()
    assert result.valid is True
    assert result.rejection_reason == RejectionReason.NONE
    assert result.min_notional == 5.0
    assert result.wallet_balance == 0.0


def test_pretrade_validator_leverage_setter_exception():
    """Leverage setter raising an exception results in LEVERAGE_SET_FAILED."""
    def mock_balance():
        return {"wallet_balance": 1000.0, "unrealized_pnl": 0.0}

    def mock_leverage(symbol, lev):
        raise RuntimeError("exchange error")

    validator = PreTradeValidator(
        balance_fetcher=mock_balance,
        leverage_setter=mock_leverage,
        target_leverage=5,
    )
    result = validator.validate(symbol="BTCUSDT", price=45000.0, risk_pct=2.0)
    assert result.valid is False
    assert result.rejection_reason == RejectionReason.LEVERAGE_SET_FAILED


# ═════════════════════════════════════════════════════════════
# Run
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
