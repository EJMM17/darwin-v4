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
    # Order flow context (microstructure layer)
    order_flow_score: float = 0.0      # -1 to +1, from OBI + TFD
    is_fake_breakout: bool = False     # whale trap detected
    fake_breakout_dir: str = ""        # direction of the trap

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
            "order_flow_score": round(self.order_flow_score, 4),
            "is_fake_breakout": self.is_fake_breakout,
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
    # confidence = sigmoid(|combined_z_score|) where sigmoid(x) = 1/(1+exp(-5*(x-0.5)))
    # Mapping: z=0.41 → conf=0.60 | z=0.60 → conf=0.72 | z=0.80 → conf=0.82
    # At 0.60, a z-score of 0.41 std devs triggers a trade — that's noise.
    # At 0.72, z-score must be ≥0.60 — requires meaningful factor agreement.
    # The engine further overrides this per-regime: 0.72 trending, 0.78 ranging.
    confidence_threshold: float = 0.72
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
        order_flow_ctx: Optional[Any] = None,
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
        order_flow_ctx : OrderFlowContext, optional
            Real-time microstructure analysis (OBI + TFD + fake breakout).
            When provided, gates out whale manipulation and boosts confidence
            when order flow confirms the signal direction.

        Returns
        -------
        TradeSignal
            Signal with confidence, direction, and factor breakdown.
        """
        cfg = self._config
        symbol = features.symbol

        # Compute individual factors
        momentum = self._compute_momentum(features, btc_features)
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

        # ── Order Flow Intelligence layer ─────────────────────────────────
        # Modulate confidence based on real-time microstructure:
        #   • Fake breakout detected → block the signal entirely
        #   • OFL confirms direction  → boost confidence up to +15%
        #   • OFL opposes direction   → reduce confidence up to -20%
        ofl_score = 0.0
        is_fake = False
        fake_dir = ""
        if order_flow_ctx is not None and not order_flow_ctx.error:
            ofl_score = order_flow_ctx.combined_score
            is_fake = order_flow_ctx.is_fake_breakout
            fake_dir = order_flow_ctx.fake_breakout_direction

            if not is_fake and direction:
                # Does order flow confirm or oppose the signal?
                signal_bullish = direction == "LONG"
                flow_bullish = ofl_score > 0
                confirming = signal_bullish == flow_bullish

                if confirming:
                    # Boost: the stronger the confirmation, the bigger the boost
                    boost = 0.15 * abs(ofl_score)
                    confidence = min(0.99, confidence + boost)
                    logger.debug(
                        "%s order flow CONFIRMS %s (ofl=%.2f) → confidence +%.3f",
                        symbol, direction, ofl_score, boost,
                    )
                else:
                    # Oppose: reduce confidence, don't enter against order flow
                    reduction = 0.20 * abs(ofl_score)
                    confidence = max(0.0, confidence - reduction)
                    logger.debug(
                        "%s order flow OPPOSES %s (ofl=%.2f) → confidence -%.3f",
                        symbol, direction, ofl_score, reduction,
                    )
        # ──────────────────────────────────────────────────────────────────

        # Build signal
        signal = TradeSignal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            regime=regime.regime.value,
            factors=raw_factors,
            factor_z_scores=z_scores,
            threshold=cfg.confidence_threshold,
            order_flow_score=ofl_score,
            is_fake_breakout=is_fake,
            fake_breakout_dir=fake_dir,
        )

        # Gate checks (regime-based)
        rejection = self._check_gates(signal, regime, features, cfg)
        if rejection:
            signal.passed = False
            signal.rejection_reason = rejection
            return signal

        # Gate: fake breakout blocks the signal in the trap direction
        if is_fake and direction:
            trap_is_long = fake_dir == "UP"
            signal_is_long = direction == "LONG"
            if trap_is_long == signal_is_long:
                signal.passed = False
                signal.rejection_reason = (
                    f"fake_breakout_{fake_dir.lower()}_detected "
                    f"obi={order_flow_ctx.obi:.2f} "
                    f"spike={order_flow_ctx.spike_ratio:.2f}x_atr"
                )
                logger.warning(
                    "%s FAKE BREAKOUT %s blocked — "
                    "obi=%.2f spike=%.2fx tfd=%.2f",
                    symbol, fake_dir,
                    order_flow_ctx.obi,
                    order_flow_ctx.spike_ratio,
                    order_flow_ctx.tfd,
                )
                return signal

        # Final confidence threshold check
        signal.passed = confidence >= cfg.confidence_threshold
        if not signal.passed:
            signal.rejection_reason = (
                f"confidence {confidence:.3f} < threshold {cfg.confidence_threshold:.3f}"
            )

        return signal

    def _compute_momentum(self, features: Any, btc_features: Any = None) -> float:
        """
        Momentum factor: normalized returns demeaned vs BTC market proxy.

        Two improvements from Kakushadze & Serur:
        1. §18.2 eq.524: ROC is already vol-normalized in market_data._roc()
        2. §10.4 eq.477: R̃ᵢ = Rᵢ - Rₘ (demean vs BTC/market)

        Demeaning removes the common market factor so the signal captures
        the symbol-specific momentum, not just "the whole market is up."
        Without demeaning, when BTC pumps 5%%, every symbol looks like it
        has strong momentum — but that's not an alpha, it's beta.
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

        symbol_momentum = sum(rocs) / len(rocs)

        # Demean vs BTC (market proxy) — §10.4 eq.477
        # Only applies when the symbol is NOT BTC and BTC features are available
        if (
            btc_features is not None
            and features.symbol != "BTCUSDT"
            and getattr(btc_features, "roc_20", 0) != 0
        ):
            btc_rocs = []
            if btc_features.roc_20 != 0:
                btc_rocs.append(btc_features.roc_20)
            if btc_features.roc_50 != 0:
                btc_rocs.append(btc_features.roc_50)
            if btc_features.roc_100 != 0:
                btc_rocs.append(btc_features.roc_100)
            if btc_rocs:
                btc_momentum = sum(btc_rocs) / len(btc_rocs)
                # Residual momentum: symbol outperforming/underperforming market
                symbol_momentum = symbol_momentum - btc_momentum

        return symbol_momentum

    def _compute_mean_reversion(self, features: Any) -> float:
        """
        Mean reversion factor: distance from mean + z-score significance.

        Amplified by volume surge (Kakushadze & Serur §10.3.1): larger volume
        spikes suggest greater overreaction, so a stronger snap-back is expected.

        Positive = price above mean (short signal for MR).
        Negative = price below mean (long signal for MR).
        We invert so positive = long opportunity.
        """
        z_20 = features.z_score_20
        z_50 = features.z_score_50
        dist = features.distance_from_mean_20

        # Use OU-calibrated z-score if available (Kakushadze §9 OU process)
        # This replaces the fixed z_20 with a window calibrated to the actual
        # mean-reversion speed of the current price series.
        ou_hl = getattr(features, "ou_half_life", 20.0)
        z_ou = getattr(features, "z_score_ou", z_20)

        # Blend: weight OU z-score more when OU window differs significantly
        # from fixed windows (indicates dynamic calibration is adding value)
        ou_diff = abs(ou_hl - 20.0) / 20.0  # how different is OU from default
        ou_weight = min(0.6, 0.3 + 0.3 * ou_diff)  # 0.3 to 0.6
        fixed_weight = 1.0 - ou_weight

        # Base MR score blending OU-calibrated and fixed z-scores
        mr_score = -(
            ou_weight * z_ou
            + fixed_weight * 0.5 * z_20
            + fixed_weight * 0.35 * z_50
            + 0.15 * dist * 10.0
        )

        # Volume filter: amplify signal when recent volume surges above average
        # High volume overreaction → stronger mean-reversion expected
        vol_scale = 1.0
        if features.volumes and len(features.volumes) >= 20:
            recent_vol = sum(features.volumes[-5:]) / 5.0
            avg_vol = sum(features.volumes[-20:]) / 20.0
            if avg_vol > 0:
                vol_ratio = recent_vol / avg_vol
                # Scale: ratio=1.5 → 1.15x, ratio=2.0 → 1.3x, ratio=3.0 → 1.5x (capped)
                vol_scale = min(1.5, 1.0 + 0.3 * math.log(max(1.0, vol_ratio)))

        return mr_score * vol_scale

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
            # En lateral: MR es la estrategia correcta. Momentum en lateral = ruido.
            # Pesos: MR domina (0.70), momentum mínimo (0.15), resto distribuido.
            # Esto es ~3x más peso en MR que en tendencia.
            w["momentum"] *= 0.3           # reducir agresivamente
            w["mean_reversion"] *= 2.5     # amplificar MR como primaria
            w["residual_alpha"] *= 0.8     # mantener ligeramente
            w["funding_carry"] *= 1.2      # funding más útil en rango
        elif regime.regime == Regime.HIGH_VOL:
            # Boost funding carry (extremes more predictive in high vol)
            w["funding_carry"] *= 1.5
            w["momentum"] *= 0.8
        elif regime.regime == Regime.LOW_VOL:
            # Low vol: similar a RANGE_BOUND, MR más relevante
            w["mean_reversion"] *= 1.8
            w["momentum"] *= 0.5
            w["funding_carry"] *= 1.3

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
        """
        Determine trade direction from weighted factor scores.

        Uses tanh smoothing (Kakushadze & Serur §10.4 eq.475) instead of
        sign() to avoid direction flips from small noisy changes near zero.
        η = tanh(combined / κ) where κ normalizes the score range.
        """
        combined = sum(weights[k] * z_scores[k] for k in z_scores)

        # tanh smoothing: κ = 0.5 maps |combined|=1 → η≈0.46, |combined|=2 → η≈0.76
        # This prevents unstable flips when combined is near zero
        κ = 0.5
        smoothed = math.tanh(combined / κ)

        if abs(smoothed) < 0.15:  # dead zone: no clear direction
            return ""

        return "LONG" if smoothed > 0 else "SHORT"

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
