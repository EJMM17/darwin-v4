"""
Darwin v5 — Enhanced Regime Detection.

Detects market regime using realized volatility, ADX, and BTC dominance.
Regime gates signal execution (e.g., no momentum in range-bound markets
without breakout confirmation).

Regimes:
    - TRENDING_UP: ADX > 25 and price > EMA50
    - TRENDING_DOWN: ADX > 25 and price < EMA50
    - RANGE_BOUND: ADX < 20, low vol
    - HIGH_VOL: Realized vol in top quartile
    - LOW_VOL: Realized vol in bottom quartile

Usage:
    detector = RegimeDetector()
    regime = detector.detect(features, btc_features)
    if regime.allows_momentum:
        # proceed with momentum signal
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("darwin.v5.regime")


class Regime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGE_BOUND = "range_bound"
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"


@dataclass(slots=True)
class RegimeState:
    """Current regime detection result."""
    regime: Regime = Regime.RANGE_BOUND
    adx: float = 0.0
    realized_vol: float = 0.0
    trend_score: float = 0.0  # [-1, +1]: bullish to bearish
    vol_percentile: float = 0.5
    btc_regime: Regime = Regime.RANGE_BOUND
    confidence: float = 0.0

    @property
    def allows_momentum(self) -> bool:
        """Momentum signals allowed in trending or high-vol breakout."""
        return self.regime in (Regime.TRENDING_UP, Regime.TRENDING_DOWN, Regime.HIGH_VOL)

    @property
    def allows_mean_reversion(self) -> bool:
        """Mean reversion allowed in range-bound or low-vol regimes."""
        return self.regime in (Regime.RANGE_BOUND, Regime.LOW_VOL)

    @property
    def risk_multiplier(self) -> float:
        """Risk multiplier based on regime."""
        return _REGIME_RISK_MULT.get(self.regime, 0.5)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime.value,
            "adx": round(self.adx, 2),
            "realized_vol": round(self.realized_vol, 6),
            "trend_score": round(self.trend_score, 4),
            "vol_percentile": round(self.vol_percentile, 4),
            "btc_regime": self.btc_regime.value,
            "confidence": round(self.confidence, 4),
            "allows_momentum": self.allows_momentum,
            "allows_mean_reversion": self.allows_mean_reversion,
            "risk_multiplier": self.risk_multiplier,
        }


_REGIME_RISK_MULT = {
    Regime.TRENDING_UP: 1.0,
    Regime.TRENDING_DOWN: 0.7,
    Regime.RANGE_BOUND: 0.6,
    Regime.HIGH_VOL: 0.4,
    Regime.LOW_VOL: 0.8,
}


@dataclass
class RegimeConfig:
    """Configuration for regime detection."""
    adx_trend_threshold: float = 25.0     # ADX above = trending
    adx_range_threshold: float = 20.0     # ADX below = range-bound
    vol_high_percentile: float = 0.75     # above = high_vol
    vol_low_percentile: float = 0.25      # below = low_vol
    vol_lookback: int = 100               # bars for vol percentile
    btc_weight: float = 0.3              # weight of BTC regime in confidence


class RegimeDetector:
    """
    Detects market regime for a given symbol.

    Uses ADX for trend strength, realized volatility for vol regime,
    and BTC as macro regime proxy.

    Parameters
    ----------
    config : RegimeConfig, optional
        Tunable thresholds.
    """

    def __init__(self, config: RegimeConfig | None = None) -> None:
        self._config = config or RegimeConfig()
        self._vol_history: Dict[str, List[float]] = {}

    def detect(
        self,
        features: Any,
        btc_features: Optional[Any] = None,
    ) -> RegimeState:
        """
        Detect regime for a symbol given its features.

        Parameters
        ----------
        features : FeatureSet
            Computed features for the target symbol.
        btc_features : FeatureSet, optional
            BTC features as macro proxy.

        Returns
        -------
        RegimeState
            Detected regime with metadata.
        """
        cfg = self._config
        symbol = features.symbol

        adx = features.adx_14
        realized_vol = features.realized_vol_20
        close = features.close
        ema_50 = features.ema_50

        # Track vol history for percentile computation
        if symbol not in self._vol_history:
            self._vol_history[symbol] = []
        self._vol_history[symbol].append(realized_vol)
        if len(self._vol_history[symbol]) > cfg.vol_lookback:
            self._vol_history[symbol] = self._vol_history[symbol][-cfg.vol_lookback:]

        # Compute vol percentile
        vol_hist = self._vol_history[symbol]
        vol_percentile = _percentile_rank(vol_hist, realized_vol)

        # Trend score: positive = bullish, negative = bearish
        if ema_50 > 0 and close > 0:
            trend_score = (close - ema_50) / ema_50
            trend_score = max(-1.0, min(1.0, trend_score * 10.0))  # scale and clamp
        else:
            trend_score = 0.0

        # Determine regime
        if vol_percentile >= cfg.vol_high_percentile:
            regime = Regime.HIGH_VOL
        elif vol_percentile <= cfg.vol_low_percentile and adx < cfg.adx_range_threshold:
            regime = Regime.LOW_VOL
        elif adx >= cfg.adx_trend_threshold:
            if trend_score > 0:
                regime = Regime.TRENDING_UP
            else:
                regime = Regime.TRENDING_DOWN
        elif adx < cfg.adx_range_threshold:
            regime = Regime.RANGE_BOUND
        else:
            # Transition zone: ADX between 20-25
            if vol_percentile > 0.5:
                regime = Regime.HIGH_VOL if vol_percentile > cfg.vol_high_percentile else Regime.RANGE_BOUND
            else:
                regime = Regime.LOW_VOL if vol_percentile < cfg.vol_low_percentile else Regime.RANGE_BOUND

        # BTC macro regime
        btc_regime = Regime.RANGE_BOUND
        if btc_features is not None:
            btc_state = self.detect(btc_features)
            btc_regime = btc_state.regime

        # Confidence: how certain we are about the regime
        confidence = self._compute_confidence(adx, vol_percentile, trend_score, cfg)

        return RegimeState(
            regime=regime,
            adx=adx,
            realized_vol=realized_vol,
            trend_score=trend_score,
            vol_percentile=vol_percentile,
            btc_regime=btc_regime,
            confidence=confidence,
        )

    def _compute_confidence(
        self,
        adx: float,
        vol_percentile: float,
        trend_score: float,
        cfg: RegimeConfig,
    ) -> float:
        """
        Compute confidence in the regime detection.

        Higher ADX = more confident in trend regime.
        Extreme vol percentile = more confident in vol regime.
        """
        # ADX contribution: higher ADX = clearer regime
        adx_conf = min(1.0, adx / 50.0)

        # Vol percentile contribution: extremes = clearer
        vol_conf = abs(vol_percentile - 0.5) * 2.0

        # Trend contribution: clear direction = higher confidence
        trend_conf = abs(trend_score)

        # Weighted average
        confidence = 0.4 * adx_conf + 0.3 * vol_conf + 0.3 * trend_conf
        return max(0.0, min(1.0, confidence))


def _percentile_rank(values: List[float], current: float) -> float:
    """Compute the percentile rank of current value in the list."""
    if len(values) < 2:
        return 0.5
    count_below = sum(1 for v in values if v < current)
    return count_below / (len(values) - 1)
