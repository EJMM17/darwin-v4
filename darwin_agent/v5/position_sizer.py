"""
Darwin v5 — Dynamic Position Sizing.

Position sizing based on:
    1. Wallet equity + unrealized PnL
    2. Volatility scaling: size inversely proportional to realized vol
    3. Drawdown-adaptive scaling: reduce risk % when drawdown worsens
    4. Max daily loss cap: throttle new entries when equity drops past threshold
    5. Regime-based risk multiplier

Usage:
    sizer = PositionSizer(config)
    result = sizer.compute(equity, realized_vol, drawdown_pct, regime, ...)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

logger = logging.getLogger("darwin.v5.position_sizer")


@dataclass
class SizerConfig:
    """Configuration for dynamic position sizing."""
    base_risk_pct: float = 1.0          # base risk per trade (1% of equity)
    leverage: int = 5                    # max leverage
    # Volatility scaling
    target_vol: float = 0.02            # target annualized vol (2%)
    vol_scale_min: float = 0.5          # minimum vol scaling factor (floor para capital pequeño)
    vol_scale_max: float = 1.5          # maximum vol scaling factor
    # Drawdown adaptive
    dd_threshold_pct: float = 5.0       # start reducing at 5% drawdown
    dd_max_pct: float = 25.0            # max drawdown before halting
    dd_scale_power: float = 2.0         # quadratic penalty
    # Daily loss cap
    daily_loss_cap_pct: float = 3.0     # max daily loss before throttle
    daily_loss_throttle: float = 0.25   # reduce to 25% of normal size
    # Per-symbol exposure cap
    max_symbol_exposure_pct: float = 20.0  # max 20% of equity per symbol
    # Total portfolio exposure
    max_total_exposure_mult: float = 5.0  # max 5x leverage total
    # Min notional
    min_notional_usdt: float = 5.5      # Binance min es $5, +$0.5 de margen por slippage


@dataclass(slots=True)
class SizeResult:
    """Output of position sizing computation."""
    position_size_usdt: float = 0.0
    quantity: float = 0.0
    risk_pct_used: float = 0.0
    vol_scale: float = 1.0
    dd_scale: float = 1.0
    regime_scale: float = 1.0
    daily_loss_scale: float = 1.0
    notional: float = 0.0
    approved: bool = True
    rejection_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position_size_usdt": round(self.position_size_usdt, 4),
            "quantity": round(self.quantity, 8),
            "risk_pct_used": round(self.risk_pct_used, 4),
            "vol_scale": round(self.vol_scale, 4),
            "dd_scale": round(self.dd_scale, 4),
            "regime_scale": round(self.regime_scale, 4),
            "daily_loss_scale": round(self.daily_loss_scale, 4),
            "notional": round(self.notional, 4),
            "approved": self.approved,
            "rejection_reason": self.rejection_reason,
        }


class PositionSizer:
    """
    Dynamic position sizing engine.

    Computes position size through multiple scaling layers:
        base_size = equity × risk_pct × leverage
        vol_adjusted = base_size × vol_scale
        dd_adjusted = vol_adjusted × dd_scale
        regime_adjusted = dd_adjusted × regime_mult
        daily_loss_adjusted = regime_adjusted × daily_loss_scale

    Parameters
    ----------
    config : SizerConfig, optional
        Tunable parameters.
    """

    def __init__(self, config: SizerConfig | None = None) -> None:
        self._config = config or SizerConfig()
        self._daily_pnl: float = 0.0
        self._daily_start_equity: float = 0.0
        self._current_day: int = -1

    def reset_daily(self, equity: float) -> None:
        """Reset daily P&L tracking (call at start of each day)."""
        self._daily_pnl = 0.0
        self._daily_start_equity = equity

    def record_trade_pnl(self, pnl: float) -> None:
        """Record a closed trade's P&L for daily loss tracking."""
        self._daily_pnl += pnl

    def compute(
        self,
        equity: float,
        price: float,
        realized_vol: float,
        drawdown_pct: float,
        regime_multiplier: float,
        signal_confidence: float,
        current_exposure: float = 0.0,
        step_size: float = 0.0,
    ) -> SizeResult:
        """
        Compute position size for a new trade.

        Parameters
        ----------
        equity : float
            Current equity (wallet + unrealized PnL).
        price : float
            Current price of the asset.
        realized_vol : float
            Recent realized volatility of the asset.
        drawdown_pct : float
            Current drawdown from peak (0-100).
        regime_multiplier : float
            Risk multiplier from regime detection.
        signal_confidence : float
            Signal confidence (0-1), used to scale position.
        current_exposure : float
            Current total portfolio exposure in USDT.
        step_size : float
            Minimum quantity increment for this symbol (e.g. 0.001 for BTC).
            If provided, quantity is rounded to this step and notional is
            re-checked AFTER rounding to avoid validation_failed: rounds to 0.

        Returns
        -------
        SizeResult
            Computed position size with scaling details.
        """
        cfg = self._config
        result = SizeResult()

        if equity <= 0:
            result.approved = False
            result.rejection_reason = "equity <= 0"
            return result

        if price <= 0:
            result.approved = False
            result.rejection_reason = "price <= 0"
            return result

        # 1. Base size
        base_risk_pct = cfg.base_risk_pct / 100.0
        base_size = equity * base_risk_pct * cfg.leverage

        # 2. Volatility scaling: inversely proportional to realized vol
        vol_scale = self._compute_vol_scale(realized_vol)
        result.vol_scale = vol_scale

        # 3. Drawdown-adaptive scaling
        dd_scale = self._compute_dd_scale(drawdown_pct)
        result.dd_scale = dd_scale

        # 4. Regime scaling
        regime_scale = max(0.1, min(1.0, regime_multiplier))
        result.regime_scale = regime_scale

        # 5. Daily loss scaling
        daily_loss_scale = self._compute_daily_loss_scale(equity)
        result.daily_loss_scale = daily_loss_scale

        # 6. Confidence scaling: higher confidence → closer to full size
        confidence_scale = 0.5 + 0.5 * signal_confidence  # [0.5, 1.0]

        # Combined
        position_size = (
            base_size
            * vol_scale
            * dd_scale
            * regime_scale
            * daily_loss_scale
            * confidence_scale
        )

        # Per-symbol exposure cap
        max_symbol = equity * (cfg.max_symbol_exposure_pct / 100.0) * cfg.leverage
        position_size = min(position_size, max_symbol)

        # Total exposure cap
        max_additional = equity * cfg.max_total_exposure_mult - current_exposure
        if max_additional <= 0:
            result.approved = False
            result.rejection_reason = "total_exposure_cap_reached"
            return result
        position_size = min(position_size, max_additional)

        # Halted by drawdown
        if drawdown_pct >= cfg.dd_max_pct:
            result.approved = False
            result.rejection_reason = f"drawdown {drawdown_pct:.1f}% >= halt threshold {cfg.dd_max_pct}%"
            return result

        # Compute quantity
        quantity = position_size / price

        # Apply step_size rounding BEFORE notional check.
        # Bug: validating notional on raw quantity (e.g. 0.000167 BTC = $16 OK),
        # then rounding in execution_engine makes it 0.000 BTC = $0 → Binance error.
        # Fix: round here so the check uses the actual quantity that will be sent.
        import math as _math
        if step_size > 0:
            quantity = _math.floor(quantity / step_size) * step_size

        notional = quantity * price

        # Min notional check (post-rounding)
        if notional < cfg.min_notional_usdt:
            result.approved = False
            result.rejection_reason = (
                f"notional ${notional:.2f} < min ${cfg.min_notional_usdt}"
                + (f" (rounds to 0 with step={step_size})" if step_size > 0 and quantity == 0 else "")
            )
            return result

        result.position_size_usdt = position_size
        result.quantity = quantity
        result.risk_pct_used = (position_size / equity) * 100.0 if equity > 0 else 0.0
        result.notional = notional
        result.approved = True

        return result

    def _compute_vol_scale(self, realized_vol: float) -> float:
        """
        Inversely scale by volatility.

        When vol > target: reduce size.
        When vol < target: increase size (capped).
        """
        cfg = self._config
        if realized_vol <= 0:
            return 1.0

        raw_scale = cfg.target_vol / realized_vol
        return max(cfg.vol_scale_min, min(cfg.vol_scale_max, raw_scale))

    def _compute_dd_scale(self, drawdown_pct: float) -> float:
        """
        Quadratic drawdown penalty.

        No penalty below threshold, increasing penalty above.
        """
        cfg = self._config
        if drawdown_pct <= cfg.dd_threshold_pct:
            return 1.0

        # Normalize to [0, 1] range
        dd_range = cfg.dd_max_pct - cfg.dd_threshold_pct
        if dd_range <= 0:
            return 1.0

        dd_ratio = (drawdown_pct - cfg.dd_threshold_pct) / dd_range
        dd_ratio = min(1.0, dd_ratio)

        # Quadratic penalty
        penalty = dd_ratio ** cfg.dd_scale_power
        return max(0.1, 1.0 - penalty * 0.9)  # floor at 10%

    def _compute_daily_loss_scale(self, equity: float) -> float:
        """Check daily loss cap and return scaling factor."""
        cfg = self._config
        if self._daily_start_equity <= 0:
            return 1.0

        daily_loss_pct = abs(self._daily_pnl) / self._daily_start_equity * 100.0
        if self._daily_pnl < 0 and daily_loss_pct >= cfg.daily_loss_cap_pct:
            logger.warning(
                "daily loss cap reached: %.1f%% loss (cap: %.1f%%)",
                daily_loss_pct,
                cfg.daily_loss_cap_pct,
            )
            return cfg.daily_loss_throttle

        return 1.0
