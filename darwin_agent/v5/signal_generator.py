"""
Darwin v5 — Multi-Factor Signal Generator.

Implements four trading factors:
    1. Momentum: standardized returns over 20-100 period ranges
    2. Mean Reversion: distance from mean + z-score significance
    3. Residual Alpha: regression residual vs BTC/ETH market proxy
    4. Funding Carry: funding rate extremes as predictive signal

Factors are standardized to z-scores and combined into a unified
confidence metric.

Usage:
    gen = SignalGenerator()
    signal = gen.generate(features, regime, btc_features, funding_rate)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from darwin_agent.v5.regime_detector import Regime, RegimeState

logger = logging.getLogger("darwin.v5.signal")


@dataclass(slots=True)
class TradeSignal:
    """Output of the signal generator."""
    symbol: str = ""
    direction: str = ""  # "LONG" or "SHORT" or ""
    confidence: float = 0.0
    regime: str = ""
    factors: Dict[str, float] = field(default_factory=dict)
    factor_z_scores: Dict[str, float] = field(default_factory=dict)
    threshold: float = 0.0
    passed: bool = False
    rejection_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "confidence": round(self.confidence, 4),
            "regime": self.regime,
            "factors": {k: round(v, 4) for k, v in self.factors.items()},
            "factor_z_scores": {k: round(v, 4) for k, v in self.factor_z_scores.items()},
            "threshold": round(self.threshold, 4),
            "passed": self.passed,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class SignalConfig:
    """Configuration for signal generation."""
    # Factor weights
    momentum_weight: float = 0.30
    mean_reversion_weight: float = 0.25
    residual_alpha_weight: float = 0.25
    funding_carry_weight: float = 0.20
    # Thresholds
    confidence_threshold: float = 0.60
    momentum_lookbacks: List[int] = field(default_factory=lambda: [20, 50, 100])
    # Mean reversion
    mr_z_score_threshold: float = 1.5  # z-score must exceed this for MR signal
    # Funding
    funding_extreme_threshold: float = 0.0005  # 0.05% is extreme
    # Signal gating
    min_adx_for_momentum: float = 20.0
    max_adx_for_mean_reversion: float = 25.0


class SignalGenerator:
    """
    Multi-factor signal generator.

    Computes individual factor scores, standardizes them to z-scores,
    and combines them into a unified confidence metric gated by regime.

    Parameters
    ----------
    config : SignalConfig, optional
        Tunable parameters.
    """

    def __init__(self, config: SignalConfig | None = None) -> None:
        self._config = config or SignalConfig()
        # Rolling factor history for z-score normalization
        self._factor_history: Dict[str, List[float]] = {
            "momentum": [],
            "mean_reversion": [],
            "residual_alpha": [],
            "funding_carry": [],
        }
        self._max_history = 200

    def generate(
        self,
        features: Any,
        regime: RegimeState,
        btc_features: Optional[Any] = None,
        funding_rate: float = 0.0,
    ) -> TradeSignal:
        """
        Generate a trade signal for a symbol.

        Parameters
        ----------
        features : FeatureSet
            Computed features for the target symbol.
        regime : RegimeState
            Current regime detection result.
        btc_features : FeatureSet, optional
            BTC features for residual alpha computation.
        funding_rate : float
            Current funding rate for the symbol.

        Returns
        -------
        TradeSignal
            Signal with confidence, direction, and factor breakdown.
        """
        cfg = self._config
        symbol = features.symbol

        # Compute individual factors
        momentum = self._compute_momentum(features)
        mean_rev = self._compute_mean_reversion(features)
        residual = self._compute_residual_alpha(features, btc_features)
        funding = self._compute_funding_carry(funding_rate)

        raw_factors = {
            "momentum": momentum,
            "mean_reversion": mean_rev,
            "residual_alpha": residual,
            "funding_carry": funding,
        }

        # Update history and compute z-scores
        z_scores = {}
        for factor_name, value in raw_factors.items():
            self._factor_history[factor_name].append(value)
            if len(self._factor_history[factor_name]) > self._max_history:
                self._factor_history[factor_name] = self._factor_history[factor_name][-self._max_history:]
            z_scores[factor_name] = _standardize(
                value, self._factor_history[factor_name]
            )

        # Apply regime gating to weights
        weights = self._get_regime_weights(regime, cfg)

        # Weighted combination of z-scores
        combined_score = sum(
            weights[k] * z_scores[k] for k in z_scores
        )

        # Determine direction from dominant factor
        direction = self._determine_direction(
            z_scores, weights, regime, features
        )

        # Confidence = absolute combined score, scaled to [0, 1]
        confidence = _sigmoid(abs(combined_score))

        # Check threshold and regime gate
        signal = TradeSignal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            regime=regime.regime.value,
            factors=raw_factors,
            factor_z_scores=z_scores,
            threshold=cfg.confidence_threshold,
        )

        # Gate checks
        rejection = self._check_gates(signal, regime, features, cfg)
        if rejection:
            signal.passed = False
            signal.rejection_reason = rejection
        else:
            signal.passed = confidence >= cfg.confidence_threshold
            if not signal.passed:
                signal.rejection_reason = (
                    f"confidence {confidence:.3f} < threshold {cfg.confidence_threshold:.3f}"
                )

        return signal

    def _compute_momentum(self, features: Any) -> float:
        """
        Momentum factor: standardized returns over multiple lookback periods.

        Positive = bullish momentum, negative = bearish.
        """
        rocs = []
        if features.roc_20 != 0:
            rocs.append(features.roc_20)
        if features.roc_50 != 0:
            rocs.append(features.roc_50)
        if features.roc_100 != 0:
            rocs.append(features.roc_100)

        if not rocs:
            return 0.0

        # Average of standardized ROCs across timeframes
        return sum(rocs) / len(rocs)

    def _compute_mean_reversion(self, features: Any) -> float:
        """
        Mean reversion factor: distance from mean + z-score significance.

        Positive = price above mean (short signal for MR).
        Negative = price below mean (long signal for MR).
        We invert so positive = long opportunity.
        """
        z_20 = features.z_score_20
        z_50 = features.z_score_50
        dist = features.distance_from_mean_20

        # Combine z-scores with distance
        # Invert: extreme low = high long signal
        mr_score = -(0.4 * z_20 + 0.3 * z_50 + 0.3 * dist * 10.0)
        return mr_score

    def _compute_residual_alpha(
        self,
        features: Any,
        btc_features: Optional[Any],
    ) -> float:
        """
        Residual alpha: regression residual of symbol returns vs BTC.

        Positive residual = symbol outperforming BTC (bullish alpha).
        Negative = underperforming.
        """
        if btc_features is None or not features.returns or not btc_features.returns:
            return 0.0

        sym_returns = features.returns
        btc_returns = btc_features.returns

        # Align lengths
        min_len = min(len(sym_returns), len(btc_returns), 50)
        if min_len < 10:
            return 0.0

        y = sym_returns[-min_len:]
        x = btc_returns[-min_len:]

        # Simple OLS: y = alpha + beta * x + residual
        beta, alpha = _ols_regression(x, y)

        # Residual of the most recent observation
        predicted = alpha + beta * x[-1]
        residual = y[-1] - predicted

        # Scale by realized vol of residuals
        residuals = [y[i] - (alpha + beta * x[i]) for i in range(len(x))]
        res_vol = _std(residuals)
        if res_vol > 1e-10:
            return residual / res_vol
        return 0.0

    def _compute_funding_carry(self, funding_rate: float) -> float:
        """
        Funding carry factor: funding rate extremes as predictive signal.

        Extreme positive funding (longs pay shorts) → contrarian short signal.
        Extreme negative funding (shorts pay longs) → contrarian long signal.
        """
        cfg = self._config
        if abs(funding_rate) < cfg.funding_extreme_threshold * 0.1:
            return 0.0  # negligible funding

        # Invert: high positive funding = short opportunity (negative signal)
        return -funding_rate / cfg.funding_extreme_threshold

    def _get_regime_weights(
        self, regime: RegimeState, cfg: SignalConfig
    ) -> Dict[str, float]:
        """Adjust factor weights based on detected regime."""
        w = {
            "momentum": cfg.momentum_weight,
            "mean_reversion": cfg.mean_reversion_weight,
            "residual_alpha": cfg.residual_alpha_weight,
            "funding_carry": cfg.funding_carry_weight,
        }

        if regime.regime in (Regime.TRENDING_UP, Regime.TRENDING_DOWN):
            # Boost momentum, reduce MR
            w["momentum"] *= 1.5
            w["mean_reversion"] *= 0.5
        elif regime.regime == Regime.RANGE_BOUND:
            # Boost MR, reduce momentum
            w["momentum"] *= 0.5
            w["mean_reversion"] *= 1.5
        elif regime.regime == Regime.HIGH_VOL:
            # Boost funding carry (extremes more predictive in high vol)
            w["funding_carry"] *= 1.5
            w["momentum"] *= 0.8
        elif regime.regime == Regime.LOW_VOL:
            # Boost mean reversion
            w["mean_reversion"] *= 1.3
            w["momentum"] *= 0.7

        # Renormalize
        total = sum(w.values())
        if total > 0:
            w = {k: v / total for k, v in w.items()}

        return w

    def _determine_direction(
        self,
        z_scores: Dict[str, float],
        weights: Dict[str, float],
        regime: RegimeState,
        features: Any,
    ) -> str:
        """Determine trade direction from weighted factor scores."""
        combined = sum(weights[k] * z_scores[k] for k in z_scores)

        if abs(combined) < 0.1:
            return ""  # no clear direction

        if combined > 0:
            return "LONG"
        else:
            return "SHORT"

    def _check_gates(
        self,
        signal: TradeSignal,
        regime: RegimeState,
        features: Any,
        cfg: SignalConfig,
    ) -> str:
        """
        Check regime-based gates. Returns rejection reason or empty string.
        """
        # Momentum in range-bound without high confidence is gated
        if (
            signal.direction
            and abs(signal.factor_z_scores.get("momentum", 0)) > abs(signal.factor_z_scores.get("mean_reversion", 0))
            and regime.regime == Regime.RANGE_BOUND
            and features.adx_14 < cfg.min_adx_for_momentum
        ):
            return "momentum_signal_in_range_bound_regime"

        # Mean reversion in strong trend is gated
        if (
            signal.direction
            and abs(signal.factor_z_scores.get("mean_reversion", 0)) > abs(signal.factor_z_scores.get("momentum", 0))
            and regime.regime in (Regime.TRENDING_UP, Regime.TRENDING_DOWN)
            and features.adx_14 > cfg.max_adx_for_mean_reversion
        ):
            return "mean_reversion_in_strong_trend"

        return ""


# ── Helper Functions ──────────────────────────────────────

def _standardize(value: float, history: List[float]) -> float:
    """Standardize a value against its history (z-score)."""
    if len(history) < 10:
        return 0.0
    mu = sum(history) / len(history)
    var = sum((v - mu) ** 2 for v in history) / len(history)
    std = math.sqrt(var)
    if std < 1e-10:
        return 0.0
    return (value - mu) / std


def _sigmoid(x: float) -> float:
    """Sigmoid function scaled to [0, 1]."""
    return 1.0 / (1.0 + math.exp(-x))


def _ols_regression(x: List[float], y: List[float]) -> tuple[float, float]:
    """
    Simple OLS regression: y = alpha + beta * x.

    Returns (beta, alpha).
    """
    n = len(x)
    if n < 2:
        return 0.0, 0.0

    x_mean = sum(x) / n
    y_mean = sum(y) / n

    cov_xy = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n)) / n
    var_x = sum((xi - x_mean) ** 2 for xi in x) / n

    if var_x < 1e-20:
        return 0.0, y_mean

    beta = cov_xy / var_x
    alpha = y_mean - beta * x_mean
    return beta, alpha


def _std(values: List[float]) -> float:
    """Standard deviation."""
    if len(values) < 2:
        return 0.0
    mu = sum(values) / len(values)
    var = sum((v - mu) ** 2 for v in values) / len(values)
    return math.sqrt(var)
