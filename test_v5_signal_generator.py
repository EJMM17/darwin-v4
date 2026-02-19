"""
Tests for Darwin v5 Multi-Factor Signal Generator.

Tests:
    - Signal score thresholds and factor outputs
    - Momentum factor computation
    - Mean reversion factor
    - Residual alpha (regression-based)
    - Funding carry factor
    - Factor z-score normalization
    - Regime-based weight adjustment
    - Signal gating by regime
"""
import pytest
from dataclasses import dataclass, field
from typing import List

from darwin_agent.v5.signal_generator import (
    SignalGenerator,
    SignalConfig,
    TradeSignal,
)
from darwin_agent.v5.regime_detector import RegimeState, Regime


@dataclass
class MockFeatureSet:
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


def _mock_regime(regime: Regime = Regime.TRENDING_UP, adx: float = 30.0) -> RegimeState:
    return RegimeState(regime=regime, adx=adx, realized_vol=0.03, confidence=0.7)


class TestSignalGeneration:
    def test_generates_signal_with_all_factors(self):
        """Signal includes all four factors."""
        gen = SignalGenerator()
        features = MockFeatureSet()
        regime = _mock_regime()

        # Generate several signals to build history for z-scoring
        for _ in range(20):
            signal = gen.generate(features, regime, funding_rate=0.0001)

        assert "momentum" in signal.factors
        assert "mean_reversion" in signal.factors
        assert "residual_alpha" in signal.factors
        assert "funding_carry" in signal.factors

    def test_confidence_between_0_and_1(self):
        """Confidence is bounded."""
        gen = SignalGenerator()
        features = MockFeatureSet()
        regime = _mock_regime()

        for _ in range(30):
            signal = gen.generate(features, regime)
        assert 0.0 <= signal.confidence <= 1.0

    def test_direction_is_valid(self):
        """Direction is LONG, SHORT, or empty."""
        gen = SignalGenerator()
        features = MockFeatureSet()
        regime = _mock_regime()

        for _ in range(30):
            signal = gen.generate(features, regime)
        assert signal.direction in ("LONG", "SHORT", "")

    def test_momentum_factor_positive_with_uptrend(self):
        """Positive ROCs produce positive momentum score."""
        gen = SignalGenerator()
        features = MockFeatureSet(roc_20=0.05, roc_50=0.08, roc_100=0.12)
        regime = _mock_regime()
        signal = gen.generate(features, regime)
        assert signal.factors["momentum"] > 0

    def test_momentum_factor_negative_with_downtrend(self):
        """Negative ROCs produce negative momentum score."""
        gen = SignalGenerator()
        features = MockFeatureSet(roc_20=-0.05, roc_50=-0.08, roc_100=-0.12)
        regime = _mock_regime()
        signal = gen.generate(features, regime)
        assert signal.factors["momentum"] < 0

    def test_mean_reversion_inverted(self):
        """High z-score (overbought) produces negative MR score (short signal)."""
        gen = SignalGenerator()
        features = MockFeatureSet(
            z_score_20=2.5,
            z_score_50=2.0,
            distance_from_mean_20=0.05,
        )
        regime = _mock_regime(Regime.RANGE_BOUND)
        signal = gen.generate(features, regime)
        # MR is inverted: high z → negative MR → short opportunity
        assert signal.factors["mean_reversion"] < 0

    def test_funding_carry_contrarian(self):
        """High positive funding → negative carry signal (short pressure)."""
        gen = SignalGenerator()
        features = MockFeatureSet()
        regime = _mock_regime()
        signal = gen.generate(features, regime, funding_rate=0.001)
        assert signal.factors["funding_carry"] < 0  # contrarian

    def test_funding_carry_negative_funding(self):
        """Negative funding → positive carry signal (long pressure)."""
        gen = SignalGenerator()
        features = MockFeatureSet()
        regime = _mock_regime()
        signal = gen.generate(features, regime, funding_rate=-0.001)
        assert signal.factors["funding_carry"] > 0

    def test_residual_alpha_with_btc_features(self):
        """Residual alpha is computed when BTC features provided."""
        import math
        gen = SignalGenerator()
        sym_returns = [0.01 * math.sin(i * 0.3) + 0.005 for i in range(50)]
        btc_returns = [0.005 * math.sin(i * 0.3 + 0.5) + 0.002 for i in range(50)]
        features = MockFeatureSet(symbol="SOLUSDT", returns=sym_returns)
        btc_features = MockFeatureSet(symbol="BTCUSDT", returns=btc_returns)
        regime = _mock_regime()
        signal = gen.generate(features, regime, btc_features)
        # Residual alpha is a float (may be zero for perfectly correlated returns)
        assert isinstance(signal.factors["residual_alpha"], float)

    def test_residual_alpha_zero_without_btc(self):
        """Residual alpha is 0 when no BTC features."""
        gen = SignalGenerator()
        features = MockFeatureSet()
        regime = _mock_regime()
        signal = gen.generate(features, regime, btc_features=None)
        assert signal.factors["residual_alpha"] == 0.0

    def test_signal_threshold_rejection(self):
        """Signal below confidence threshold is rejected."""
        gen = SignalGenerator(SignalConfig(confidence_threshold=0.99))
        features = MockFeatureSet()
        regime = _mock_regime()
        for _ in range(30):
            signal = gen.generate(features, regime)
        # With threshold at 0.99, most signals won't pass
        assert not signal.passed or signal.confidence >= 0.99

    def test_regime_gating_momentum_in_range(self):
        """Momentum signals gated in range-bound regime with low ADX."""
        gen = SignalGenerator()
        features = MockFeatureSet(
            roc_20=0.10,
            roc_50=0.15,
            roc_100=0.20,
            adx_14=15.0,  # low ADX
            z_score_20=0.1,
            z_score_50=0.1,
            distance_from_mean_20=0.001,
        )
        regime = _mock_regime(Regime.RANGE_BOUND, adx=15.0)

        # Build enough history
        for _ in range(30):
            signal = gen.generate(features, regime)

        if signal.rejection_reason:
            assert "momentum" in signal.rejection_reason or "confidence" in signal.rejection_reason

    def test_to_dict_serialization(self):
        """TradeSignal serializes correctly."""
        signal = TradeSignal(
            symbol="BTCUSDT",
            direction="LONG",
            confidence=0.75,
            regime="trending_up",
            factors={"momentum": 0.5, "mean_reversion": -0.2},
            factor_z_scores={"momentum": 1.2, "mean_reversion": -0.5},
            threshold=0.6,
            passed=True,
        )
        d = signal.to_dict()
        assert d["symbol"] == "BTCUSDT"
        assert d["direction"] == "LONG"
        assert d["passed"] is True
        assert "momentum" in d["factors"]

    def test_factor_z_scores_populated(self):
        """Factor z-scores are computed after enough history."""
        gen = SignalGenerator()
        features = MockFeatureSet()
        regime = _mock_regime()

        for _ in range(20):
            signal = gen.generate(features, regime)

        assert len(signal.factor_z_scores) == 4


class TestSignalConfig:
    def test_default_weights_sum_to_one(self):
        config = SignalConfig()
        total = (
            config.momentum_weight
            + config.mean_reversion_weight
            + config.residual_alpha_weight
            + config.funding_carry_weight
        )
        assert abs(total - 1.0) < 1e-6
