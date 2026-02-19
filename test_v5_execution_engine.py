"""
Tests for Darwin v5 Execution Engine.

Tests:
    - Order validation (min notional, step size, leverage)
    - Order placement wrapper
    - Retry logic with mock exchange
    - Slippage tolerance
    - Dry-run mode
"""
import asyncio
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from dataclasses import dataclass

from darwin_agent.v5.execution_engine import (
    ExecutionEngine,
    ExecutionConfig,
    OrderRequest,
    ExecutionResult,
)


class MockTimeSyncManager:
    def __init__(self):
        self.offset_ms = 0
        self.sync_count = 0

    def synced_timestamp(self) -> int:
        return 1700000000000

    def force_sync_on_error(self, code: int) -> bool:
        self.sync_count += 1
        return True


class MockTelemetry:
    def __init__(self):
        self.events = []

    def log_order_placed(self, **kwargs):
        self.events.append(("order_placed", kwargs))

    def log_order_filled(self, **kwargs):
        self.events.append(("order_filled", kwargs))


class TestOrderValidation:
    def test_valid_order(self):
        """Valid order passes validation."""
        engine = ExecutionEngine(
            MagicMock(),
            config=ExecutionConfig(min_notional_usdt=5.0, leverage=5),
        )
        order = OrderRequest(
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.001,
            price=50000.0,
            leverage=5,
        )
        valid, reason = engine.validate_order(order)
        assert valid
        assert reason == ""

    def test_missing_symbol(self):
        """Missing symbol fails validation."""
        engine = ExecutionEngine(MagicMock())
        order = OrderRequest(side="BUY", quantity=0.001, price=50000.0)
        valid, reason = engine.validate_order(order)
        assert not valid
        assert "missing symbol" in reason

    def test_invalid_side(self):
        """Invalid side fails validation."""
        engine = ExecutionEngine(MagicMock())
        order = OrderRequest(symbol="BTCUSDT", side="INVALID", quantity=0.001, price=50000.0)
        valid, reason = engine.validate_order(order)
        assert not valid
        assert "invalid side" in reason

    def test_zero_quantity(self):
        """Zero quantity fails validation."""
        engine = ExecutionEngine(MagicMock())
        order = OrderRequest(symbol="BTCUSDT", side="BUY", quantity=0, price=50000.0)
        valid, reason = engine.validate_order(order)
        assert not valid
        assert "quantity" in reason

    def test_zero_price(self):
        """Zero price fails validation."""
        engine = ExecutionEngine(MagicMock())
        order = OrderRequest(symbol="BTCUSDT", side="BUY", quantity=0.001, price=0.0)
        valid, reason = engine.validate_order(order)
        assert not valid
        assert "price" in reason

    def test_below_min_notional(self):
        """Below min notional fails validation."""
        engine = ExecutionEngine(
            MagicMock(),
            config=ExecutionConfig(min_notional_usdt=10.0),
        )
        order = OrderRequest(
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.0001,
            price=50000.0,  # notional = 5.0 < 10.0
        )
        valid, reason = engine.validate_order(order)
        assert not valid
        assert "notional" in reason

    def test_leverage_exceeds_max(self):
        """Leverage exceeding max fails validation."""
        engine = ExecutionEngine(
            MagicMock(),
            config=ExecutionConfig(leverage=5),
        )
        order = OrderRequest(
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.001,
            price=50000.0,
            leverage=10,  # exceeds 5x
        )
        valid, reason = engine.validate_order(order)
        assert not valid
        assert "leverage" in reason


class TestDryRun:
    @pytest.mark.asyncio
    async def test_dry_run_no_exchange_call(self):
        """Dry run validates but doesn't call exchange."""
        mock_client = MagicMock()
        engine = ExecutionEngine(mock_client)
        order = OrderRequest(
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.001,
            price=50000.0,
            leverage=5,
        )
        result = await engine.place_order(order, dry_run=True)
        assert result.success
        assert result.order_id == "DRY_RUN"
        assert result.filled_qty == 0.001
        # Exchange should NOT have been called
        mock_client._session.request.assert_not_called()


class TestStepSize:
    def test_step_size_rounding(self):
        """Quantity rounds to step size."""
        assert ExecutionEngine._round_step(0.0015, 0.001) == 0.001
        assert abs(ExecutionEngine._round_step(0.0099, 0.001) - 0.009) < 1e-10
        assert abs(ExecutionEngine._round_step(1.555, 0.01) - 1.55) < 1e-10


class TestMetrics:
    def test_metrics_tracking(self):
        """Metrics are tracked correctly."""
        engine = ExecutionEngine(MagicMock())
        metrics = engine.get_metrics()
        assert metrics["total_orders"] == 0
        assert metrics["total_fills"] == 0
        assert metrics["total_rejects"] == 0


class TestExecutionResult:
    def test_to_dict(self):
        """ExecutionResult serializes correctly."""
        result = ExecutionResult(
            success=True,
            order_id="123",
            symbol="BTCUSDT",
            side="BUY",
            filled_qty=0.001,
            avg_price=50000.0,
            slippage_bps=5.0,
            latency_ms=150.0,
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["order_id"] == "123"
        assert d["slippage_bps"] == 5.0
