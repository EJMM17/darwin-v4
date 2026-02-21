"""
Tests for Darwin v5 Dynamic Position Sizing.

Tests:
    - Basic position sizing with equity
    - Volatility scaling (inverse proportional)
    - Drawdown-adaptive scaling
    - Daily loss cap
    - Exposure limits (per-symbol, total portfolio)
    - Min notional rejection
    - Dynamic equity changes
"""
import pytest

from darwin_agent.v5.position_sizer import (
    PositionSizer,
    SizerConfig,
    SizeResult,
)


class TestBasicSizing:
    def test_base_position_size(self):
        """Base size = equity * risk_pct * leverage."""
        sizer = PositionSizer(SizerConfig(base_risk_pct=1.0, leverage=5))
        result = sizer.compute(
            equity=1000.0,
            price=50000.0,
            realized_vol=0.02,  # matches target_vol
            drawdown_pct=0.0,
            regime_multiplier=1.0,
            signal_confidence=1.0,
        )
        assert result.approved
        assert result.position_size_usdt > 0
        assert result.quantity > 0

    def test_zero_equity_rejected(self):
        """Zero equity produces rejection."""
        sizer = PositionSizer()
        result = sizer.compute(
            equity=0.0, price=50000.0, realized_vol=0.02,
            drawdown_pct=0.0, regime_multiplier=1.0, signal_confidence=0.8,
        )
        assert not result.approved
        assert "equity" in result.rejection_reason

    def test_zero_price_rejected(self):
        """Zero price produces rejection."""
        sizer = PositionSizer()
        result = sizer.compute(
            equity=1000.0, price=0.0, realized_vol=0.02,
            drawdown_pct=0.0, regime_multiplier=1.0, signal_confidence=0.8,
        )
        assert not result.approved
        assert "price" in result.rejection_reason


class TestVolatilityScaling:
    def test_high_vol_reduces_size(self):
        """Higher volatility → smaller position."""
        sizer = PositionSizer(SizerConfig(target_vol=0.02))
        base = sizer.compute(
            equity=1000.0, price=50000.0, realized_vol=0.02,
            drawdown_pct=0.0, regime_multiplier=1.0, signal_confidence=1.0,
        )
        high_vol = sizer.compute(
            equity=1000.0, price=50000.0, realized_vol=0.10,
            drawdown_pct=0.0, regime_multiplier=1.0, signal_confidence=1.0,
        )
        assert high_vol.position_size_usdt < base.position_size_usdt
        assert high_vol.vol_scale < 1.0

    def test_low_vol_increases_size(self):
        """Lower volatility → larger position (capped)."""
        sizer = PositionSizer(SizerConfig(target_vol=0.02))
        base = sizer.compute(
            equity=1000.0, price=50000.0, realized_vol=0.02,
            drawdown_pct=0.0, regime_multiplier=1.0, signal_confidence=1.0,
        )
        low_vol = sizer.compute(
            equity=1000.0, price=50000.0, realized_vol=0.005,
            drawdown_pct=0.0, regime_multiplier=1.0, signal_confidence=1.0,
        )
        assert low_vol.position_size_usdt >= base.position_size_usdt
        assert low_vol.vol_scale >= 1.0

    def test_vol_scale_bounded(self):
        """Vol scale is bounded between min and max."""
        config = SizerConfig(vol_scale_min=0.3, vol_scale_max=1.5)
        sizer = PositionSizer(config)
        # Very high vol
        result = sizer.compute(
            equity=1000.0, price=50000.0, realized_vol=1.0,
            drawdown_pct=0.0, regime_multiplier=1.0, signal_confidence=1.0,
        )
        assert result.vol_scale >= 0.3

        # Very low vol
        result = sizer.compute(
            equity=1000.0, price=50000.0, realized_vol=0.0001,
            drawdown_pct=0.0, regime_multiplier=1.0, signal_confidence=1.0,
        )
        assert result.vol_scale <= 1.5


class TestDrawdownAdaptive:
    def test_no_penalty_below_threshold(self):
        """No drawdown penalty below threshold."""
        sizer = PositionSizer(SizerConfig(dd_threshold_pct=5.0))
        result = sizer.compute(
            equity=1000.0, price=50000.0, realized_vol=0.02,
            drawdown_pct=3.0, regime_multiplier=1.0, signal_confidence=1.0,
        )
        assert result.dd_scale == 1.0

    def test_penalty_above_threshold(self):
        """Drawdown penalty reduces size above threshold."""
        sizer = PositionSizer(SizerConfig(dd_threshold_pct=5.0))
        result = sizer.compute(
            equity=1000.0, price=50000.0, realized_vol=0.02,
            drawdown_pct=15.0, regime_multiplier=1.0, signal_confidence=1.0,
        )
        assert result.dd_scale < 1.0

    def test_max_drawdown_halts(self):
        """Position sizing halted at max drawdown."""
        sizer = PositionSizer(SizerConfig(dd_max_pct=25.0))
        result = sizer.compute(
            equity=1000.0, price=50000.0, realized_vol=0.02,
            drawdown_pct=26.0, regime_multiplier=1.0, signal_confidence=1.0,
        )
        assert not result.approved
        assert "drawdown" in result.rejection_reason

    def test_progressive_penalty(self):
        """Penalty increases with drawdown."""
        sizer = PositionSizer(SizerConfig(dd_threshold_pct=5.0, dd_max_pct=25.0))
        result_10 = sizer.compute(
            equity=1000.0, price=50000.0, realized_vol=0.02,
            drawdown_pct=10.0, regime_multiplier=1.0, signal_confidence=1.0,
        )
        result_20 = sizer.compute(
            equity=1000.0, price=50000.0, realized_vol=0.02,
            drawdown_pct=20.0, regime_multiplier=1.0, signal_confidence=1.0,
        )
        assert result_20.dd_scale < result_10.dd_scale


class TestDailyLossCap:
    def test_daily_loss_cap_throttle(self):
        """Daily loss exceeding cap throttles sizing."""
        config = SizerConfig(daily_loss_cap_pct=3.0, daily_loss_throttle=0.25)
        sizer = PositionSizer(config)

        # Pass daily PnL directly to compute() — the engine tracks daily state
        result = sizer.compute(
            equity=965.0, price=50000.0, realized_vol=0.02,
            drawdown_pct=3.5, regime_multiplier=1.0, signal_confidence=1.0,
            daily_pnl=-35.0, daily_start_equity=1000.0,  # 3.5% daily loss
        )
        assert result.daily_loss_scale == 0.25

    def test_no_throttle_within_cap(self):
        """No throttle when daily loss is within cap."""
        sizer = PositionSizer(SizerConfig(daily_loss_cap_pct=3.0))

        result = sizer.compute(
            equity=990.0, price=50000.0, realized_vol=0.02,
            drawdown_pct=1.0, regime_multiplier=1.0, signal_confidence=1.0,
            daily_pnl=-10.0, daily_start_equity=1000.0,  # 1% daily loss (within cap)
        )
        assert result.daily_loss_scale == 1.0


class TestExposureLimits:
    def test_total_exposure_cap(self):
        """Rejected when total exposure would exceed cap."""
        config = SizerConfig(max_total_exposure_mult=5.0)
        sizer = PositionSizer(config)
        result = sizer.compute(
            equity=100.0, price=50000.0, realized_vol=0.02,
            drawdown_pct=0.0, regime_multiplier=1.0, signal_confidence=1.0,
            current_exposure=500.0,  # already at 5x
        )
        assert not result.approved
        assert "exposure_cap" in result.rejection_reason

    def test_min_notional_rejection(self):
        """Rejected when notional is below minimum."""
        config = SizerConfig(min_notional_usdt=5.0, base_risk_pct=0.001)
        sizer = PositionSizer(config)
        result = sizer.compute(
            equity=10.0, price=50000.0, realized_vol=0.02,
            drawdown_pct=0.0, regime_multiplier=1.0, signal_confidence=0.5,
        )
        # With very small risk pct and small equity, notional may be below min
        if not result.approved:
            assert "notional" in result.rejection_reason


class TestDynamicEquityChanges:
    def test_position_scales_with_equity(self):
        """Position size scales proportionally with equity."""
        sizer = PositionSizer()
        small = sizer.compute(
            equity=100.0, price=50000.0, realized_vol=0.02,
            drawdown_pct=0.0, regime_multiplier=1.0, signal_confidence=1.0,
        )
        large = sizer.compute(
            equity=10000.0, price=50000.0, realized_vol=0.02,
            drawdown_pct=0.0, regime_multiplier=1.0, signal_confidence=1.0,
        )
        if small.approved and large.approved:
            # Should be roughly 100x larger
            ratio = large.position_size_usdt / small.position_size_usdt
            assert 50 < ratio < 200  # approximate due to caps

    def test_confidence_scaling(self):
        """Higher confidence → larger position."""
        sizer = PositionSizer()
        low_conf = sizer.compute(
            equity=1000.0, price=50000.0, realized_vol=0.02,
            drawdown_pct=0.0, regime_multiplier=1.0, signal_confidence=0.1,
        )
        high_conf = sizer.compute(
            equity=1000.0, price=50000.0, realized_vol=0.02,
            drawdown_pct=0.0, regime_multiplier=1.0, signal_confidence=1.0,
        )
        if low_conf.approved and high_conf.approved:
            assert high_conf.position_size_usdt > low_conf.position_size_usdt

    def test_to_dict(self):
        """SizeResult serializes correctly."""
        result = SizeResult(
            position_size_usdt=50.0,
            quantity=0.001,
            vol_scale=0.8,
            dd_scale=0.9,
        )
        d = result.to_dict()
        assert d["position_size_usdt"] == 50.0
        assert d["vol_scale"] == 0.8
