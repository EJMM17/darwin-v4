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
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict

logger = logging.getLogger("darwin.v5.position_sizer")


@dataclass
class SizerConfig:
    """Configuration for dynamic position sizing."""
    base_risk_pct: float = 1.0          # base risk per trade (1% of equity)
    leverage: int = 5                    # max leverage (may be overridden by dynamic leverage)
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
    # Half-Kelly adaptive sizing
    use_kelly: bool = True              # activar Kelly scaling basado en historial de trades
    kelly_fraction: float = 0.5         # usar half-Kelly (más conservador que Kelly completo)
    kelly_lookback: int = 30            # número de trades históricos para estimar Kelly
    kelly_min: float = 0.5             # escalar mínimo del Kelly (no bajar más del 50%)
    kelly_max: float = 1.5             # escalar máximo del Kelly (no subir más del 150%)
    # ── Micro-capital compounding ──────────────────────────────────
    # For accounts < $500, every cent matters. These settings ensure:
    # 1. Profits are reinvested immediately via equity-based sizing
    # 2. Position sizes are maximized within risk bounds
    # 3. Geometric compounding is achieved by always using current equity
    micro_capital_threshold: float = 500.0  # below this, use aggressive compounding
    micro_min_position_pct: float = 8.0     # min position as % of equity (prevent dust)
    micro_max_positions: int = 2            # max concurrent positions with small capital


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
    notes: str = ""  # metadata adicional (ej. kelly_scale aplicado)

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
            "notes": self.notes,
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
        self._trade_pnls: deque = deque(maxlen=200)


    def _compute_kelly_scale(self, cfg: SizerConfig) -> float:
        """
        Compute half-Kelly position scaling factor from recent trade history.

        Kelly Criterion (Thorp 2006):
            K% = W - (1 - W) / R
        where:
            W = win rate (fraction of profitable trades)
            R = avg_win / avg_loss (reward-to-risk ratio)

        We use fractional Kelly (cfg.kelly_fraction, default 0.5) to reduce
        variance while preserving most of the long-term growth benefit.
        A full-Kelly system has ~2x the volatility of half-Kelly with only
        ~30% more return — not worth it for institutional capital preservation.

        Returns a scaling factor clamped to [kelly_min, kelly_max].
        Returns 1.0 (no adjustment) if insufficient data or poor statistics.
        """
        recent = list(self._trade_pnls)[-cfg.kelly_lookback:]
        wins = [p for p in recent if p > 0]
        losses = [p for p in recent if p < 0]

        if len(wins) < 3 or len(losses) < 3:
            return 1.0  # insufficient sample — no Kelly adjustment

        win_rate = len(wins) / len(recent)
        avg_win = sum(wins) / len(wins)
        avg_loss = abs(sum(losses) / len(losses))

        if avg_loss < 1e-8:
            return 1.0

        reward_to_risk = avg_win / avg_loss
        kelly_full = win_rate - (1.0 - win_rate) / reward_to_risk

        if kelly_full <= 0:
            # Negative Kelly = strategy has no edge right now → scale down
            return cfg.kelly_min

        kelly_fractional = kelly_full * cfg.kelly_fraction

        # Map Kelly fraction to a position scaling factor around 1.0
        # kelly_fractional = 1.0 means "bet 100% of capital per trade" (too large)
        # We normalize so that kelly_fractional = 0.02 (2% risk) → scale = 1.0
        base_risk = max(cfg.base_risk_pct / 100.0, 1e-8)
        scale = kelly_fractional / base_risk

        return max(cfg.kelly_min, min(cfg.kelly_max, scale))

    def record_trade_pnl(self, pnl: float) -> None:
        """Record a closed trade's P&L for Kelly criterion tracking."""
        self._trade_pnls.append(pnl)

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
        risk_pct_override: float = 0.0,
        daily_pnl: float = 0.0,
        daily_start_equity: float = 0.0,
        sl_distance_pct: float = 0.0,
        leverage_override: int = 0,
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
        sl_distance_pct : float
            Actual stop-loss distance as a fraction (e.g., 0.015 for 1.5%).
            When provided, position is sized so that SL hit = exactly
            risk_fraction of equity lost. When 0, falls back to config default.

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

        # 1. Base size — Risk-Based Position Sizing (institutional standard)
        #
        # OLD (broken): base_size = equity * risk_pct * leverage
        #   This conflates NOTIONAL with RISK. $100 * 1% * 5x = $5 notional.
        #   Actual loss at SL = $5 * 1.5% = $0.075 = 0.075% of equity (way too small).
        #
        # NEW (correct): size so that hitting SL loses exactly risk_fraction of equity.
        #   risk_amount = equity * risk_fraction  (e.g., $100 * 2% = $2)
        #   base_size = risk_amount / sl_distance  (e.g., $2 / 1.5% = $133.33 notional)
        #   Cap at max_leverage * equity to prevent margin violation.
        #
        # This means:
        #   - If SL = 1.5% and risk = 2%, notional = equity * 2%/1.5% = 1.33x equity
        #   - If SL = 0.7% and risk = 2%, notional = equity * 2%/0.7% = 2.86x equity
        #   - The position automatically gets SMALLER when SL is wider (high vol)
        #     and LARGER when SL is tighter (low vol) — natural vol targeting.
        # Use dynamic leverage if provided, otherwise fall back to config
        leverage_for_sizing = leverage_override if leverage_override > 0 else cfg.leverage

        base_risk_pct = (risk_pct_override / 100.0) if risk_pct_override > 0 else (cfg.base_risk_pct / 100.0)
        risk_amount = equity * base_risk_pct
        # Use actual SL distance when provided, otherwise conservative fallback.
        # The SL distance comes from the engine (ATR-dynamic per regime).
        # Floor at 0.5% to prevent extreme sizing on very tight SLs.
        if sl_distance_pct > 0:
            sl_for_sizing = max(0.005, sl_distance_pct)
        else:
            # Fallback: use 1.5% (trending default) as conservative estimate
            sl_for_sizing = 0.015
        base_size = min(risk_amount / sl_for_sizing, equity * leverage_for_sizing)

        # 2. Volatility scaling: inversely proportional to realized vol
        vol_scale = self._compute_vol_scale(realized_vol)
        result.vol_scale = vol_scale

        # 3. Drawdown-adaptive scaling
        dd_scale = self._compute_dd_scale(drawdown_pct)
        result.dd_scale = dd_scale

        # 4. Regime scaling
        regime_scale = max(0.1, min(1.0, regime_multiplier))
        result.regime_scale = regime_scale

        # 5. Daily loss scaling (uses engine's single source of truth)
        daily_loss_scale = self._compute_daily_loss_scale(daily_pnl, daily_start_equity)
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
        max_symbol = equity * (cfg.max_symbol_exposure_pct / 100.0) * leverage_for_sizing
        position_size = min(position_size, max_symbol)

        # Total exposure cap
        max_additional = equity * cfg.max_total_exposure_mult - current_exposure
        if max_additional <= 0:
            result.approved = False
            result.rejection_reason = "total_exposure_cap_reached"
            return result

        # 7. Half-Kelly scaling — SINGLE SOURCE OF TRUTH
        #
        # ARCHITECTURE FIX: Kelly is computed ONLY in PortfolioRiskManager
        # and passed here as risk_pct_override. Applying Kelly a second time
        # here would DOUBLE the Kelly adjustment:
        #   - PortfolioRiskManager outputs half-Kelly risk % (e.g., 3.2%)
        #   - That risk % is already used as base_risk_pct via risk_pct_override
        #   - Multiplying by kelly_scale again would push past optimal Kelly
        #   - Past full Kelly = NEGATIVE growth rate = path to ruin
        #
        # The old code applied Kelly twice:
        #   1. risk_pct_override from PortfolioRiskManager._compute_kelly_risk()
        #   2. kelly_scale from PositionSizer._compute_kelly_scale()
        # This has been removed. Kelly lives in ONE place: PortfolioRiskManager.
        #
        # Reference: Thorp (2006) "The Kelly Criterion in Blackjack" — over-betting
        # by even 10% past Kelly optimal reduces geometric growth rate.
        position_size = min(position_size, max_additional)

        # Halted by drawdown
        if drawdown_pct >= cfg.dd_max_pct:
            result.approved = False
            result.rejection_reason = f"drawdown {drawdown_pct:.1f}% >= halt threshold {cfg.dd_max_pct}%"
            return result

        # ── Micro-capital compounding boost ────────────────────────────
        # For accounts < $500, the standard sizing formula produces positions
        # that barely exceed min notional ($5). This means:
        #   - Most of the capital sits idle
        #   - Compounding is negligible (can't compound dust)
        #   - ROI growth is linear, not geometric
        #
        # Fix: ensure the position is at least micro_min_position_pct of equity.
        # This trades a bit more aggressively but is necessary to achieve
        # geometric growth on small accounts. The kill switch and SL still
        # protect the downside.
        if equity < cfg.micro_capital_threshold:
            min_micro_notional = equity * (cfg.micro_min_position_pct / 100.0) * leverage_for_sizing
            if position_size < min_micro_notional:
                position_size = min_micro_notional
                result.notes = f"micro_capital_boost: min_notional raised to ${min_micro_notional:.2f}"

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
            # For micro-capital: try bumping quantity to exactly meet min notional
            # instead of rejecting. With $80 equity, losing a trade to min-notional
            # is worse than a slightly larger position.
            if equity < cfg.micro_capital_threshold and price > 0:
                min_qty = (cfg.min_notional_usdt * 1.05) / price  # +5% margin
                if step_size > 0:
                    min_qty = _math.ceil(min_qty / step_size) * step_size
                min_notional_check = min_qty * price
                # Only bump if we can afford it (within leverage limits)
                max_affordable = equity * leverage_for_sizing
                if min_notional_check <= max_affordable:
                    quantity = min_qty
                    notional = quantity * price
                    result.notes = (
                        f"micro_notional_bump: qty→{quantity:.8f} "
                        f"notional→${notional:.2f} (was below min)"
                    )
                else:
                    result.approved = False
                    result.rejection_reason = (
                        f"notional ${notional:.2f} < min ${cfg.min_notional_usdt} "
                        f"(micro-cap cannot afford min position)"
                    )
                    return result
            else:
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

    def _compute_daily_loss_scale(self, daily_pnl: float, daily_start_equity: float) -> float:
        """Check daily loss cap and return scaling factor."""
        cfg = self._config
        if daily_start_equity <= 0:
            return 1.0

        daily_loss_pct = abs(daily_pnl) / daily_start_equity * 100.0
        if daily_pnl < 0 and daily_loss_pct >= cfg.daily_loss_cap_pct:
            logger.warning(
                "daily loss cap reached: %.1f%% loss (cap: %.1f%%)",
                daily_loss_pct,
                cfg.daily_loss_cap_pct,
            )
            return cfg.daily_loss_throttle

        return 1.0
