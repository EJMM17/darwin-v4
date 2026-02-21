"""
Darwin v5 — Dynamic Leverage Manager.

Adaptive leverage per-trade based on:
    1. Signal confidence: higher confidence → more aggressive leverage
    2. ATR (volatility): higher vol → lower leverage (avoid liquidation)
    3. Regime: trending = higher, range-bound = lower
    4. Win streak / drawdown: scale leverage with recent performance
    5. Account equity protection: hard cap at liquidation-safe distance

The core principle (Thorp / Kelly):
    Optimal leverage ∝ edge / variance
    More edge + less variance → more leverage
    Less edge + more variance → less leverage

For extreme-volatility assets (memecoins like WIF, PIPPIN):
    ATR can be 5-15% per day. At 20x leverage, a 5% adverse move
    = 100% equity loss = liquidation. Dynamic leverage prevents this
    by capping leverage so the worst-case drawdown (2× ATR) never
    exceeds the daily loss limit.

Usage:
    dlm = DynamicLeverageManager(config)
    lev = dlm.compute(
        signal_confidence=0.82,
        atr_pct=3.5,
        regime="trending_up",
        drawdown_pct=2.3,
        win_streak=3,
    )
    # lev = DynamicLeverageResult(leverage=12, reason="...")
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List

logger = logging.getLogger("darwin.v5.dynamic_leverage")


@dataclass
class DynamicLeverageConfig:
    """Tunable parameters for dynamic leverage."""
    # Leverage bounds
    min_leverage: int = 1          # allow 1x for extreme-vol assets (memecoins)
    max_leverage: int = 20         # Binance max for most pairs
    default_leverage: int = 5      # fallback when no signal

    # Confidence scaling: maps [0.6, 1.0] → [min_lev, max_lev]
    confidence_floor: float = 0.60  # below this = min leverage
    confidence_ceiling: float = 0.95  # above this = max leverage

    # ATR-based ceiling: leverage ≤ max_adverse_move_pct / (2 × ATR%)
    # This ensures that a 2-ATR adverse move never exceeds the daily loss limit.
    # Example: daily_loss_limit=5%, ATR=3% → max_lev = 5/(2×3) = 0.83x → floor to min
    # Example: daily_loss_limit=5%, ATR=0.5% → max_lev = 5/(2×0.5) = 5x
    max_adverse_atr_multiplier: float = 2.0  # assume worst case = 2× ATR move
    daily_loss_budget_pct: float = 5.0       # max daily loss to survive

    # Regime multipliers
    regime_multipliers: Dict[str, float] = None  # set in __post_init__

    # Drawdown scaling: reduce leverage in drawdown
    dd_start_pct: float = 3.0      # start reducing at 3% drawdown
    dd_max_pct: float = 15.0       # at 15% drawdown → min leverage
    dd_curve_power: float = 1.5    # curve shape (1=linear, 2=quadratic)

    # Win/loss streak adjustment
    streak_boost_per_win: float = 0.5  # +0.5x per consecutive win (capped)
    streak_max_boost: float = 3.0      # max +3x from streak
    streak_penalty_per_loss: float = 1.0  # -1x per consecutive loss
    streak_max_penalty: float = 5.0    # max -5x from loss streak

    def __post_init__(self):
        if self.regime_multipliers is None:
            self.regime_multipliers = {
                "trending_up": 1.0,      # full leverage in strong trend
                "trending_down": 0.8,    # slightly cautious in downtrend
                "range_bound": 0.5,      # half leverage in chop
                "high_vol": 0.4,         # very cautious in high vol
                "low_vol": 0.7,          # moderate in low vol
            }


@dataclass(slots=True)
class DynamicLeverageResult:
    """Output of leverage computation."""
    leverage: int = 5
    raw_leverage: float = 5.0       # before rounding
    confidence_component: float = 0.0
    atr_ceiling: float = 20.0
    regime_mult: float = 1.0
    dd_mult: float = 1.0
    streak_adj: float = 0.0
    reason: str = ""                # human-readable explanation

    def to_dict(self) -> dict:
        return {
            "leverage": self.leverage,
            "raw_leverage": round(self.raw_leverage, 2),
            "confidence_component": round(self.confidence_component, 2),
            "atr_ceiling": round(self.atr_ceiling, 2),
            "regime_mult": round(self.regime_mult, 2),
            "dd_mult": round(self.dd_mult, 2),
            "streak_adj": round(self.streak_adj, 2),
            "reason": self.reason,
        }


class DynamicLeverageManager:
    """
    Computes optimal leverage for each trade based on multiple risk factors.

    The leverage formula:
        base = lerp(min_lev, max_lev, confidence_scaled)
        atr_cap = daily_loss_budget / (atr_mult × atr_pct)
        regime_adj = base × regime_mult
        dd_adj = regime_adj × dd_mult
        streak_adj = dd_adj + streak_bonus
        final = clamp(min(streak_adj, atr_cap), min_lev, max_lev)

    Parameters
    ----------
    config : DynamicLeverageConfig, optional
        Tunable parameters.
    """

    def __init__(self, config: DynamicLeverageConfig | None = None) -> None:
        self._config = config or DynamicLeverageConfig()
        # Track consecutive wins/losses for streak adjustment
        self._consecutive_wins: int = 0
        self._consecutive_losses: int = 0

    def record_trade_result(self, is_win: bool) -> None:
        """Update win/loss streak tracking."""
        if is_win:
            self._consecutive_wins += 1
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            self._consecutive_wins = 0

    def compute(
        self,
        signal_confidence: float,
        atr_pct: float,
        regime: str,
        drawdown_pct: float = 0.0,
    ) -> DynamicLeverageResult:
        """
        Compute dynamic leverage for a trade.

        Parameters
        ----------
        signal_confidence : float
            Signal confidence [0, 1].
        atr_pct : float
            Current ATR as fraction of price (e.g. 0.03 = 3%).
        regime : str
            Current market regime (e.g. "trending_up", "range_bound").
        drawdown_pct : float
            Current drawdown from peak (0-100).

        Returns
        -------
        DynamicLeverageResult
            Computed leverage with breakdown.
        """
        cfg = self._config
        result = DynamicLeverageResult()

        # 1. Confidence → base leverage (linear interpolation)
        conf_scaled = max(0.0, min(1.0,
            (signal_confidence - cfg.confidence_floor)
            / max(0.01, cfg.confidence_ceiling - cfg.confidence_floor)
        ))
        base_lev = cfg.min_leverage + conf_scaled * (cfg.max_leverage - cfg.min_leverage)
        result.confidence_component = base_lev

        # 2. ATR-based ceiling: prevent liquidation from volatility spike
        #    max_lev = daily_loss_budget / (atr_mult × atr_pct)
        #    This is THE critical safety guard for memecoins.
        if atr_pct > 0.001:
            atr_ceiling = (cfg.daily_loss_budget_pct / 100.0) / (
                cfg.max_adverse_atr_multiplier * atr_pct
            )
            # Floor: never let ATR ceiling below min_leverage
            atr_ceiling = max(float(cfg.min_leverage), atr_ceiling)
        else:
            atr_ceiling = float(cfg.max_leverage)  # near-zero ATR = no restriction

        result.atr_ceiling = atr_ceiling

        # 3. Regime multiplier
        regime_mult = cfg.regime_multipliers.get(regime, 0.5)
        result.regime_mult = regime_mult

        # 4. Drawdown scaling: reduce leverage during drawdown
        dd_mult = self._compute_dd_mult(drawdown_pct)
        result.dd_mult = dd_mult

        # 5. Win/loss streak adjustment
        streak_adj = 0.0
        if self._consecutive_wins > 0:
            streak_adj = min(
                self._consecutive_wins * cfg.streak_boost_per_win,
                cfg.streak_max_boost,
            )
        elif self._consecutive_losses > 0:
            streak_adj = -min(
                self._consecutive_losses * cfg.streak_penalty_per_loss,
                cfg.streak_max_penalty,
            )
        result.streak_adj = streak_adj

        # Combine: base × regime × drawdown + streak, capped by ATR
        raw_lev = base_lev * regime_mult * dd_mult + streak_adj

        # Apply ATR ceiling
        raw_lev = min(raw_lev, atr_ceiling)

        # Clamp to [min, max]
        raw_lev = max(float(cfg.min_leverage), min(float(cfg.max_leverage), raw_lev))

        result.raw_leverage = raw_lev
        result.leverage = max(cfg.min_leverage, min(cfg.max_leverage, int(round(raw_lev))))

        # Build explanation
        parts = [f"conf={signal_confidence:.2f}→base={base_lev:.1f}x"]
        if atr_ceiling < base_lev:
            parts.append(f"atr_cap={atr_ceiling:.1f}x(atr={atr_pct*100:.2f}%)")
        parts.append(f"regime={regime}×{regime_mult:.1f}")
        if dd_mult < 1.0:
            parts.append(f"dd={drawdown_pct:.1f}%×{dd_mult:.2f}")
        if streak_adj != 0:
            parts.append(f"streak={streak_adj:+.1f}")
        parts.append(f"→{result.leverage}x")
        result.reason = " | ".join(parts)

        logger.debug("dynamic_leverage: %s", result.reason)
        return result

    def _compute_dd_mult(self, drawdown_pct: float) -> float:
        """Drawdown → leverage multiplier. Reduces leverage during drawdowns."""
        cfg = self._config
        if drawdown_pct <= cfg.dd_start_pct:
            return 1.0

        dd_range = cfg.dd_max_pct - cfg.dd_start_pct
        if dd_range <= 0:
            return 1.0

        ratio = min(1.0, (drawdown_pct - cfg.dd_start_pct) / dd_range)
        # Smooth curve from 1.0 → 0.15 (never zero, always some leverage)
        penalty = ratio ** cfg.dd_curve_power
        return max(0.15, 1.0 - penalty * 0.85)
