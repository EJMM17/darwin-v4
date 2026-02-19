"""
Tests for Darwin v5 Regime Detection.

Tests:
    - Trending regime when ADX is high + price above EMA
    - Range-bound when ADX is low
    - High volatility detection
    - Low volatility detection
    - Regime gating (momentum/mean-reversion)
    - BTC dominance proxy
"""
import pytest
from dataclasses import dataclass, field
from typing import List

from darwin_agent.v5.regime_detector import (
    RegimeDetector,
    RegimeConfig,
    RegimeState,
    Regime,
)


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


class TestRegimeDetection:
    def test_trending_up_detection(self):
        """High ADX + price above EMA50 → TRENDING_UP."""
        detector = RegimeDetector()
        features = MockFeatureSet(
            close=50000.0,
            ema_50=48000.0,  # price above EMA50
            adx_14=35.0,     # high ADX = strong trend
            realized_vol_20=0.03,
        )
        regime = detector.detect(features)
        assert regime.regime == Regime.TRENDING_UP
        assert regime.allows_momentum is True

    def test_trending_down_detection(self):
        """High ADX + price below EMA50 → TRENDING_DOWN."""
        detector = RegimeDetector()
        features = MockFeatureSet(
            close=46000.0,
            ema_50=48000.0,  # price below EMA50
            adx_14=35.0,
            realized_vol_20=0.03,
        )
        regime = detector.detect(features)
        assert regime.regime == Regime.TRENDING_DOWN
        assert regime.allows_momentum is True

    def test_range_bound_detection(self):
        """Low ADX → RANGE_BOUND."""
        detector = RegimeDetector()
        features = MockFeatureSet(
            close=50000.0,
            ema_50=50000.0,
            adx_14=15.0,     # low ADX = no trend
            realized_vol_20=0.025,  # moderate vol
        )
        regime = detector.detect(features)
        assert regime.regime == Regime.RANGE_BOUND
        assert regime.allows_mean_reversion is True
        assert regime.allows_momentum is False

    def test_high_volatility_detection(self):
        """Very high realized vol → HIGH_VOL."""
        detector = RegimeDetector()
        # Feed many low-vol readings first to establish baseline
        low_vol_features = MockFeatureSet(realized_vol_20=0.01, adx_14=20.0)
        for _ in range(50):
            detector.detect(low_vol_features)

        # Then detect with high vol
        high_vol_features = MockFeatureSet(
            realized_vol_20=0.10,  # much higher than baseline
            adx_14=20.0,
        )
        regime = detector.detect(high_vol_features)
        assert regime.regime == Regime.HIGH_VOL

    def test_low_volatility_detection(self):
        """Very low realized vol with low ADX → LOW_VOL."""
        detector = RegimeDetector()
        # Feed many high-vol readings first
        high_vol_features = MockFeatureSet(realized_vol_20=0.05, adx_14=15.0)
        for _ in range(50):
            detector.detect(high_vol_features)

        # Then detect with low vol
        low_vol_features = MockFeatureSet(
            realized_vol_20=0.005,
            adx_14=15.0,
        )
        regime = detector.detect(low_vol_features)
        assert regime.regime == Regime.LOW_VOL
        assert regime.allows_mean_reversion is True

    def test_risk_multiplier_mapping(self):
        """Each regime maps to correct risk multiplier."""
        state = RegimeState(regime=Regime.TRENDING_UP)
        assert state.risk_multiplier == 1.0

        state = RegimeState(regime=Regime.TRENDING_DOWN)
        assert state.risk_multiplier == 0.7

        state = RegimeState(regime=Regime.RANGE_BOUND)
        assert state.risk_multiplier == 0.6

        state = RegimeState(regime=Regime.HIGH_VOL)
        assert state.risk_multiplier == 0.4

        state = RegimeState(regime=Regime.LOW_VOL)
        assert state.risk_multiplier == 0.8

    def test_confidence_computation(self):
        """Confidence is between 0 and 1."""
        detector = RegimeDetector()
        features = MockFeatureSet(adx_14=40.0, realized_vol_20=0.03)
        regime = detector.detect(features)
        assert 0.0 <= regime.confidence <= 1.0

    def test_to_dict(self):
        """RegimeState serializes correctly."""
        state = RegimeState(
            regime=Regime.TRENDING_UP,
            adx=30.5,
            realized_vol=0.035,
            trend_score=0.7,
        )
        d = state.to_dict()
        assert d["regime"] == "trending_up"
        assert d["adx"] == 30.5
        assert d["allows_momentum"] is True

    def test_btc_regime_proxy(self):
        """BTC features are used as macro proxy."""
        detector = RegimeDetector()
        symbol_features = MockFeatureSet(symbol="SOLUSDT", adx_14=30.0)
        btc_features = MockFeatureSet(symbol="BTCUSDT", adx_14=35.0)

        regime = detector.detect(symbol_features, btc_features)
        assert regime.btc_regime in [r for r in Regime]


class TestRegimeConfig:
    def test_default_config(self):
        config = RegimeConfig()
        assert config.adx_trend_threshold == 25.0
        assert config.adx_range_threshold == 20.0
        assert config.vol_high_percentile == 0.75
        assert config.vol_low_percentile == 0.25

    def test_custom_config(self):
        config = RegimeConfig(adx_trend_threshold=30.0, vol_lookback=200)
        detector = RegimeDetector(config)
        features = MockFeatureSet(adx_14=27.0)  # between 25-30
        regime = detector.detect(features)
        # With threshold at 30, ADX 27 should not be trending
        assert regime.regime != Regime.TRENDING_UP
