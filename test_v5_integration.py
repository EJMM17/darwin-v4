"""
Darwin v5 Integration Tests.

Tests:
    - Full pipeline: market data → regime → signal → sizing → execution
    - Dry-run with mock exchange (no orders placed)
    - TimeSyncManager integration
    - HealthMonitor lifecycle
    - TelemetryReporter event logging
    - Portfolio construction end-to-end
"""
import asyncio
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import List

from darwin_agent.v5.time_sync import TimeSyncManager
from darwin_agent.v5.health_monitor import HealthMonitor
from darwin_agent.v5.telemetry import TelemetryReporter
from darwin_agent.v5.market_data import MarketDataLayer, FeatureSet, _ema, _atr, _adx, _returns, _z_score, _roc
from darwin_agent.v5.regime_detector import RegimeDetector, Regime
from darwin_agent.v5.signal_generator import SignalGenerator
from darwin_agent.v5.position_sizer import PositionSizer, SizerConfig
from darwin_agent.v5.execution_engine import ExecutionEngine, OrderRequest
from darwin_agent.v5.monte_carlo import MonteCarloValidator
from darwin_agent.v5.portfolio_constructor import PortfolioConstructor


# ═══════════════════════════════════════════════════════
# Mock Exchange Client
# ═══════════════════════════════════════════════════════

class MockBinanceClient:
    """Lightweight mock for Binance client."""

    def __init__(self):
        self._base_url = "https://fapi.binance.com"
        self._timeout_s = 10.0
        self._session = MagicMock()
        self._call_count = 0

    def _sign(self, params):
        return "mock_signature"

    def ping_futures(self):
        self._call_count += 1

    def get_wallet_balance(self):
        return 1000.0

    def get_unrealized_pnl(self):
        return 50.0

    def get_open_positions(self):
        return []

    def set_leverage(self, symbol, leverage):
        return True

    def close(self):
        pass


# ═══════════════════════════════════════════════════════
# TimeSyncManager Tests
# ═══════════════════════════════════════════════════════

class TestTimeSyncManager:
    def test_offset_starts_at_zero(self):
        client = MockBinanceClient()
        sync = TimeSyncManager(client)
        assert sync.offset_ms == 0
        assert sync.sync_count == 0

    def test_synced_timestamp_returns_int(self):
        client = MockBinanceClient()
        sync = TimeSyncManager(client, sync_interval_s=9999.0)
        # Set last_sync to prevent auto-sync with mock (which fails)
        sync._last_sync = __import__("time").time()
        ts = sync.synced_timestamp()
        assert isinstance(ts, int)
        assert ts > 0

    def test_diagnostics(self):
        client = MockBinanceClient()
        sync = TimeSyncManager(client)
        diag = sync.get_diagnostics()
        assert "offset_ms" in diag
        assert "sync_count" in diag

    def test_force_sync_on_400(self):
        client = MockBinanceClient()
        sync = TimeSyncManager(client)
        result = sync.force_sync_on_error(400)
        assert result is True

    def test_no_force_sync_on_200(self):
        client = MockBinanceClient()
        sync = TimeSyncManager(client)
        result = sync.force_sync_on_error(200)
        assert result is False


# ═══════════════════════════════════════════════════════
# HealthMonitor Tests
# ═══════════════════════════════════════════════════════

class TestHealthMonitor:
    @pytest.mark.asyncio
    async def test_health_monitor_lifecycle(self):
        """Health monitor starts and stops cleanly."""
        client = MockBinanceClient()
        monitor = HealthMonitor(client, ping_interval_s=0.1)

        task = asyncio.create_task(monitor.run())
        await asyncio.sleep(0.3)  # let it run a few pings
        monitor.stop()
        await task

        assert monitor.status.total_pings > 0
        assert monitor.is_healthy is True

    @pytest.mark.asyncio
    async def test_wait_for_connection(self):
        """wait_for_connection returns True for working client."""
        client = MockBinanceClient()
        monitor = HealthMonitor(client)
        result = await monitor.wait_for_connection(timeout_s=5.0)
        assert result is True

    def test_diagnostics(self):
        client = MockBinanceClient()
        monitor = HealthMonitor(client)
        diag = monitor.get_diagnostics()
        assert "connected" in diag
        assert "uptime_pct" in diag


# ═══════════════════════════════════════════════════════
# TelemetryReporter Tests
# ═══════════════════════════════════════════════════════

class TestTelemetryReporter:
    def test_event_counting(self):
        telemetry = TelemetryReporter(log_dir="/tmp/darwin_test_logs")
        assert telemetry.event_count == 0

        telemetry.log_heartbeat(
            equity=1000.0, wallet_balance=950.0,
            unrealized_pnl=50.0, open_positions=2,
            regime="trending_up",
        )
        assert telemetry.event_count == 1

    def test_signal_logging(self):
        telemetry = TelemetryReporter(log_dir="/tmp/darwin_test_logs")
        telemetry.log_signal_generated(
            symbol="BTCUSDT",
            direction="LONG",
            confidence=0.75,
            threshold=0.6,
            regime="trending_up",
            factors={"momentum": 0.5, "mean_reversion": -0.1},
            position_size_usdt=50.0,
        )
        assert telemetry.event_count == 1

    def test_rejection_logging(self):
        telemetry = TelemetryReporter(log_dir="/tmp/darwin_test_logs")
        telemetry.log_signal_rejected(
            symbol="ETHUSDT",
            reason="confidence below threshold",
            confidence=0.45,
            regime="range_bound",
        )
        assert telemetry.event_count == 1

    def test_monte_carlo_logging(self):
        telemetry = TelemetryReporter(log_dir="/tmp/darwin_test_logs")
        telemetry.log_monte_carlo_result(
            n_simulations=1000,
            mean_pnl=25.0,
            p5_pnl=-10.0,
            p95_pnl=60.0,
            edge_ratio=0.8,
            max_drawdown_p95=15.0,
        )
        assert telemetry.event_count == 1


# ═══════════════════════════════════════════════════════
# Market Data Technical Indicators
# ═══════════════════════════════════════════════════════

class TestTechnicalIndicators:
    def test_ema_convergence(self):
        """EMA converges toward constant value."""
        data = [100.0] * 100
        ema = _ema(data, 20)
        assert abs(ema[-1] - 100.0) < 0.01

    def test_ema_length_matches(self):
        data = [float(i) for i in range(50)]
        ema = _ema(data, 10)
        assert len(ema) == 50

    def test_atr_non_negative(self):
        """ATR is always non-negative."""
        highs = [101.0 + i * 0.1 for i in range(50)]
        lows = [99.0 + i * 0.1 for i in range(50)]
        closes = [100.0 + i * 0.1 for i in range(50)]
        atr = _atr(highs, lows, closes, 14)
        assert all(v >= 0 for v in atr)

    def test_adx_range(self):
        """ADX values are in [0, 100]."""
        import random
        rng = random.Random(42)
        n = 100
        closes = [100.0]
        highs = [101.0]
        lows = [99.0]
        for _ in range(n - 1):
            change = rng.gauss(0, 1)
            closes.append(closes[-1] + change)
            highs.append(closes[-1] + abs(rng.gauss(0, 0.5)))
            lows.append(closes[-1] - abs(rng.gauss(0, 0.5)))

        adx = _adx(highs, lows, closes, 14)
        for v in adx:
            assert 0.0 <= v <= 100.0 or v == 0.0

    def test_returns_length(self):
        """Returns list is one shorter than closes."""
        closes = [100.0, 101.0, 102.0, 101.5, 103.0]
        r = _returns(closes)
        assert len(r) == len(closes) - 1

    def test_z_score_standard_normal(self):
        """Z-score of mean is approximately 0."""
        closes = [100.0] * 20
        z = _z_score(closes, 20)
        assert abs(z) < 0.01

    def test_roc_basic(self):
        """Rate of change for simple sequence."""
        closes = [100.0] * 20 + [110.0]
        roc = _roc(closes, 20)
        assert abs(roc - 0.10) < 0.01


# ═══════════════════════════════════════════════════════
# Portfolio Constructor Tests
# ═══════════════════════════════════════════════════════

class TestPortfolioConstructor:
    def test_equal_weights_initially(self):
        """Initial weights are equal."""
        pc = PortfolioConstructor(["BTCUSDT", "ETHUSDT", "SOLUSDT"])
        weights = pc.get_weights()
        expected = 1.0 / 3.0
        for w in weights.values():
            assert abs(w - expected) < 0.01

    def test_weights_sum_to_one(self):
        """Weights always sum to 1.0."""
        pc = PortfolioConstructor(["BTCUSDT", "ETHUSDT", "SOLUSDT"])
        pc.update_metrics("BTCUSDT", sharpe=1.5, realized_vol=0.02, cumulative_pnl=100.0)
        pc.update_metrics("ETHUSDT", sharpe=0.8, realized_vol=0.04, cumulative_pnl=50.0)
        pc.update_metrics("SOLUSDT", sharpe=0.3, realized_vol=0.06, cumulative_pnl=-20.0)

        snapshot = pc.construct()
        total = sum(snapshot.weights.values())
        assert abs(total - 1.0) < 0.01

    def test_better_symbol_gets_higher_weight(self):
        """Symbol with better metrics gets higher allocation."""
        pc = PortfolioConstructor(["BTCUSDT", "ETHUSDT", "SOLUSDT"])
        pc.update_metrics("BTCUSDT", sharpe=2.0, realized_vol=0.01, cumulative_pnl=200.0, momentum_score=0.5)
        pc.update_metrics("ETHUSDT", sharpe=0.5, realized_vol=0.05, cumulative_pnl=10.0, momentum_score=0.1)
        pc.update_metrics("SOLUSDT", sharpe=0.1, realized_vol=0.08, cumulative_pnl=-50.0, momentum_score=-0.1)

        snapshot = pc.construct()
        assert snapshot.weights["BTCUSDT"] > snapshot.weights["SOLUSDT"]

    def test_weight_constraints_respected(self):
        """Weights respect min/max constraints."""
        from darwin_agent.v5.portfolio_constructor import PortfolioConfig
        config = PortfolioConfig(min_symbol_weight=0.10, max_symbol_weight=0.50)
        pc = PortfolioConstructor(["BTCUSDT", "ETHUSDT", "SOLUSDT"], config)
        snapshot = pc.construct()
        for w in snapshot.weights.values():
            assert w >= 0.09  # small tolerance
            assert w <= 0.51


# ═══════════════════════════════════════════════════════
# Full Pipeline Integration
# ═══════════════════════════════════════════════════════

class TestFullPipeline:
    def test_regime_to_signal_to_sizing(self):
        """Full pipeline: features → regime → signal → sizing."""
        @dataclass
        class MockFeatures:
            symbol: str = "BTCUSDT"
            close: float = 50000.0
            ema_20: float = 49500.0
            ema_50: float = 49000.0
            ema_200: float = 48000.0
            atr_14: float = 500.0
            realized_vol_20: float = 0.03
            realized_vol_60: float = 0.025
            roc_20: float = 0.05
            roc_50: float = 0.08
            roc_100: float = 0.12
            z_score_20: float = 0.5
            z_score_50: float = 0.3
            distance_from_mean_20: float = 0.02
            adx_14: float = 30.0
            returns: List[float] = field(default_factory=lambda: [0.01] * 50)
            closes: List[float] = field(default_factory=lambda: [50000.0] * 200)
            highs: List[float] = field(default_factory=lambda: [50500.0] * 200)
            lows: List[float] = field(default_factory=lambda: [49500.0] * 200)
            volumes: List[float] = field(default_factory=lambda: [1000.0] * 200)

        features = MockFeatures()

        # Step 1: Regime detection
        detector = RegimeDetector()
        regime = detector.detect(features)
        assert regime.regime in [r for r in Regime]

        # Step 2: Signal generation (build history)
        gen = SignalGenerator()
        for _ in range(20):
            signal = gen.generate(features, regime, funding_rate=0.0001)

        assert signal.symbol == "BTCUSDT"
        assert 0.0 <= signal.confidence <= 1.0

        # Step 3: Position sizing
        sizer = PositionSizer(SizerConfig(base_risk_pct=1.0, leverage=5))
        size = sizer.compute(
            equity=1000.0,
            price=features.close,
            realized_vol=features.realized_vol_20,
            drawdown_pct=0.0,
            regime_multiplier=regime.risk_multiplier,
            signal_confidence=signal.confidence,
        )

        assert size.approved
        assert size.position_size_usdt > 0
        assert size.quantity > 0

    @pytest.mark.asyncio
    async def test_dry_run_execution(self):
        """Full pipeline with dry-run execution."""
        client = MockBinanceClient()
        engine = ExecutionEngine(client)
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

    def test_monte_carlo_after_trading(self):
        """Monte Carlo validates after accumulating trades."""
        mc = MonteCarloValidator()
        # Simulated trade PnLs
        pnls = [5.0, -2.0, 8.0, -3.0, 4.0, -1.0, 6.0, -2.5, 3.0, -1.5,
                7.0, -4.0, 5.0, -2.0, 9.0, -3.0, 4.0, -1.0, 6.0, -2.0]

        result = mc.validate(pnls)
        assert result.n_simulations > 0
        assert result.actual_pnl == sum(pnls)
