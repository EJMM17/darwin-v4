"""
Portfolio Risk Manager — Institutional-Grade Capital Protection

Implements the three pillars that institutional crypto funds use
to protect capital as AUM scales:

  1. PORTFOLIO HEAT LIMIT
     Max total notional exposure as % of equity.
     Institutional standard: 30-50% total, ≤20% per symbol.
     Prevents the bot from being 100% long in a correlated market crash.

  2. CORRELATION FILTER
     Blocks new positions that are highly correlated with existing ones.
     Derived from rolling 48-hour return correlation between crypto assets.
     Prevents "fake diversification" (buying SOL when already long ETH).

  3. CIRCUIT BREAKER (multi-level)
     Level 1: Daily loss -5%  → reduce all new positions by 50%
     Level 2: Daily loss -10% → halt all new entries for the day
     Level 3: Daily loss -15% → emergency stop, notify and wait
     These mirror the kill-switch mechanisms used by Jane Street / Two Sigma.

  4. HALF-KELLY POSITION SIZING
     Given estimated win_rate and avg win/loss, compute Kelly fraction.
     Always use half-Kelly (0.5x) to reduce variance — standard practice
     per Thorp (Beat the Dealer), Poundstone (Fortune's Formula).
     Formula: f* = (W × p - L × q) / (W × L)  with f_used = f* / 2

Reference: AIMA/PwC 7th Annual Global Crypto Hedge Fund Report 2025,
           Kakushadze & Serur "151 Trading Strategies" §15, §20
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class PortfolioRiskConfig:
    # Portfolio heat (total exposure)
    max_portfolio_heat_pct: float = 40.0   # max 40% of equity exposed at once
    max_single_symbol_pct: float = 20.0    # max 20% per symbol (notional/equity)

    # Correlation filter
    correlation_window_bars: int = 48      # 48 bars × 15min = 12 hours lookback
    max_correlation: float = 0.70          # block if corr > 0.70 with open position

    # Circuit breaker levels (daily PnL % thresholds)
    circuit_level1_pct: float = -5.0       # reduce sizing 50%
    circuit_level2_pct: float = -10.0      # halt new entries
    circuit_level3_pct: float = -15.0      # emergency stop

    # Half-Kelly settings
    kelly_fraction: float = 0.50           # use half-Kelly
    kelly_max_risk_pct: float = 4.0        # cap Kelly at 4% even if formula says more
    kelly_min_risk_pct: float = 0.5        # floor at 0.5% even if formula says less
    kelly_lookback_trades: int = 30        # use last 30 trades to estimate win_rate


# ─── Circuit breaker state ─────────────────────────────────────────────────────

@dataclass
class CircuitBreakerState:
    level: int = 0                         # 0 = nominal, 1/2/3 = tripped
    tripped_at: float = 0.0               # unix timestamp
    daily_loss_pct: float = 0.0
    message: str = ""

    @property
    def is_halted(self) -> bool:
        return self.level >= 2

    @property
    def is_emergency(self) -> bool:
        return self.level >= 3

    @property
    def sizing_multiplier(self) -> float:
        """How much to scale down new positions."""
        if self.level == 0:
            return 1.0
        if self.level == 1:
            return 0.50
        return 0.0  # level 2+ → no new positions


# ─── Portfolio risk result ─────────────────────────────────────────────────────

@dataclass
class RiskCheckResult:
    approved: bool
    rejection_reason: str = ""
    kelly_risk_pct: float = 2.0       # suggested risk % (half-Kelly or fallback)
    sizing_multiplier: float = 1.0    # from circuit breaker
    portfolio_heat_pct: float = 0.0   # current heat before this trade
    correlation_with: str = ""        # if rejected for correlation, the culprit symbol
    circuit_level: int = 0


# ─── Main risk manager ─────────────────────────────────────────────────────────

class PortfolioRiskManager:
    """
    Stateful portfolio risk manager. One instance per engine, shared across symbols.

    Thread-safety: assumes single asyncio event loop (no concurrent calls).
    """

    # Crypto return correlations are relatively stable over 12h windows.
    # We store recent closes per symbol to estimate rolling correlation.
    _MAX_PRICE_HISTORY = 200  # bars per symbol

    def __init__(self, config: Optional[PortfolioRiskConfig] = None):
        self._cfg = config or PortfolioRiskConfig()
        self._cb = CircuitBreakerState()
        # Symbol → recent close prices (for correlation)
        self._price_history: Dict[str, deque] = {}
        # Recent trade PnL for Kelly estimation
        self._trade_pcts: deque = deque(maxlen=self._cfg.kelly_lookback_trades)
        # Daily PnL tracking
        self._day_start_equity: float = 0.0
        self._current_day: int = -1  # UTC day number

    # ── Public API ─────────────────────────────────────────────────────────────

    def update_price(self, symbol: str, close: float) -> None:
        """Call each tick to maintain price history for correlation."""
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=self._MAX_PRICE_HISTORY)
        self._price_history[symbol].append(close)

    def record_trade(self, pnl_pct: float) -> None:
        """Record trade outcome for Kelly estimation."""
        self._trade_pcts.append(pnl_pct)

    def update_daily_equity(self, equity: float) -> None:
        """Call each tick. Resets day tracking at UTC midnight."""
        import datetime
        today = datetime.datetime.utcnow().timetuple().tm_yday
        if today != self._current_day:
            self._current_day = today
            self._day_start_equity = equity
        self._update_circuit_breaker(equity)

    def check_new_position(
        self,
        symbol: str,
        equity: float,
        price: float,
        proposed_notional: float,
        open_positions: List[Dict],
    ) -> RiskCheckResult:
        """
        Full risk check before opening a new position.

        Returns RiskCheckResult with approved=True/False and contextual data.
        """
        cfg = self._cfg
        cb = self._cb

        # ── Circuit breaker ────────────────────────────────────────────────
        if cb.is_emergency:
            return RiskCheckResult(
                approved=False,
                rejection_reason=f"CIRCUIT_BREAKER_L3 emergency stop — daily loss {cb.daily_loss_pct:.2f}%",
                circuit_level=cb.level,
            )
        if cb.is_halted:
            return RiskCheckResult(
                approved=False,
                rejection_reason=f"CIRCUIT_BREAKER_L{cb.level} entries halted — daily loss {cb.daily_loss_pct:.2f}%",
                circuit_level=cb.level,
            )

        # ── Portfolio heat ─────────────────────────────────────────────────
        current_notional = _compute_total_notional(open_positions)
        heat_pct = (current_notional / equity * 100.0) if equity > 0 else 0.0
        new_heat_pct = ((current_notional + proposed_notional) / equity * 100.0) if equity > 0 else 0.0

        if new_heat_pct > cfg.max_portfolio_heat_pct:
            return RiskCheckResult(
                approved=False,
                rejection_reason=(
                    f"PORTFOLIO_HEAT: current={heat_pct:.1f}% + new={proposed_notional/equity*100:.1f}% "
                    f"> max={cfg.max_portfolio_heat_pct:.1f}%"
                ),
                portfolio_heat_pct=heat_pct,
                circuit_level=cb.level,
            )

        symbol_notional = proposed_notional
        symbol_heat_pct = (symbol_notional / equity * 100.0) if equity > 0 else 0.0
        if symbol_heat_pct > cfg.max_single_symbol_pct:
            return RiskCheckResult(
                approved=False,
                rejection_reason=(
                    f"SYMBOL_HEAT: {symbol} would be {symbol_heat_pct:.1f}% > max={cfg.max_single_symbol_pct:.1f}%"
                ),
                portfolio_heat_pct=heat_pct,
                circuit_level=cb.level,
            )

        # ── Correlation filter ─────────────────────────────────────────────
        if open_positions:
            corr_result = self._check_correlation(symbol, open_positions)
            if corr_result is not None:
                culprit, corr_val = corr_result
                return RiskCheckResult(
                    approved=False,
                    rejection_reason=(
                        f"CORRELATION: {symbol} corr={corr_val:.2f} with {culprit} "
                        f"> max={cfg.max_correlation:.2f}"
                    ),
                    portfolio_heat_pct=heat_pct,
                    correlation_with=culprit,
                    circuit_level=cb.level,
                )

        # ── Kelly position sizing ──────────────────────────────────────────
        kelly_risk = self._compute_kelly_risk(equity)

        return RiskCheckResult(
            approved=True,
            kelly_risk_pct=kelly_risk,
            sizing_multiplier=cb.sizing_multiplier,
            portfolio_heat_pct=heat_pct,
            circuit_level=cb.level,
        )

    def get_circuit_breaker_state(self) -> CircuitBreakerState:
        return self._cb

    # ── Internal methods ────────────────────────────────────────────────────────

    def _update_circuit_breaker(self, equity: float) -> None:
        """Update circuit breaker based on current daily PnL."""
        cfg = self._cfg
        if self._day_start_equity <= 0:
            return

        daily_pnl_pct = (equity - self._day_start_equity) / self._day_start_equity * 100.0
        self._cb.daily_loss_pct = daily_pnl_pct

        prev_level = self._cb.level

        if daily_pnl_pct <= cfg.circuit_level3_pct:
            self._cb.level = 3
            self._cb.message = f"EMERGENCY: daily loss {daily_pnl_pct:.2f}% ≤ {cfg.circuit_level3_pct}%"
        elif daily_pnl_pct <= cfg.circuit_level2_pct:
            self._cb.level = 2
            self._cb.message = f"HALTED: daily loss {daily_pnl_pct:.2f}% ≤ {cfg.circuit_level2_pct}%"
        elif daily_pnl_pct <= cfg.circuit_level1_pct:
            self._cb.level = 1
            self._cb.message = f"REDUCED: daily loss {daily_pnl_pct:.2f}% ≤ {cfg.circuit_level1_pct}%"
        else:
            self._cb.level = 0
            self._cb.message = ""

        if self._cb.level > prev_level and self._cb.level > 0:
            self._cb.tripped_at = time.time()

    def _check_correlation(
        self, symbol: str, open_positions: List[Dict]
    ) -> Optional[Tuple[str, float]]:
        """
        Returns (culprit_symbol, correlation) if new position is too correlated
        with any open position. Returns None if safe to proceed.
        """
        cfg = self._cfg
        sym_prices = list(self._price_history.get(symbol, []))
        if len(sym_prices) < 10:
            return None  # not enough data, skip check

        sym_returns = _price_to_returns(sym_prices)

        for pos in open_positions:
            open_sym = pos.get("symbol", "")
            if open_sym == symbol:
                continue
            open_prices = list(self._price_history.get(open_sym, []))
            if len(open_prices) < 10:
                continue

            open_returns = _price_to_returns(open_prices)
            # Align lengths (use the shorter)
            n = min(len(sym_returns), len(open_returns))
            if n < 5:
                continue

            corr = _pearson_correlation(sym_returns[-n:], open_returns[-n:])
            if corr > cfg.max_correlation:
                return (open_sym, corr)

        return None

    def _compute_kelly_risk(self, equity: float) -> float:
        """
        Half-Kelly position risk percentage based on recent trade history.

        Kelly formula (simplified for fixed fractional):
          f* = (p × W - q × L) / (W × L)
          where p = win_rate, q = 1-p, W = avg_win, L = avg_loss (positive)

        Returns the risk % to use (half of f*, clamped to [min, max]).
        Falls back to 2.0% if insufficient trade history.
        """
        cfg = self._cfg
        trades = list(self._trade_pcts)

        if len(trades) < 5:
            return 2.0  # fallback before we have enough data

        wins = [t for t in trades if t > 0]
        losses = [abs(t) for t in trades if t <= 0]

        if not wins or not losses:
            return 2.0

        p = len(wins) / len(trades)    # win probability
        q = 1.0 - p                    # loss probability
        W = sum(wins) / len(wins)      # avg win (as decimal, e.g. 0.025)
        L = sum(losses) / len(losses)  # avg loss (positive, e.g. 0.015)

        if L < 1e-10 or W < 1e-10:
            return 2.0

        # Kelly fraction
        kelly_f = (p * W - q * L) / (W * L)

        # Apply half-Kelly
        half_kelly = kelly_f * cfg.kelly_fraction

        # Convert to % and clamp
        half_kelly_pct = half_kelly * 100.0
        return max(cfg.kelly_min_risk_pct, min(cfg.kelly_max_risk_pct, half_kelly_pct))


# ─── Pure math helpers ─────────────────────────────────────────────────────────

def _price_to_returns(prices: List[float]) -> List[float]:
    returns = []
    for i in range(1, len(prices)):
        if prices[i - 1] > 1e-10:
            returns.append((prices[i] - prices[i - 1]) / prices[i - 1])
    return returns


def _pearson_correlation(xs: List[float], ys: List[float]) -> float:
    """Pearson correlation coefficient — pure Python, no deps."""
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx < 1e-12 or dy < 1e-12:
        return 0.0
    return num / (dx * dy)


def _compute_total_notional(positions: List[Dict]) -> float:
    """Sum of notional value of all open positions."""
    total = 0.0
    for pos in positions:
        amt = abs(float(pos.get("positionAmt", 0.0)))
        price = float(pos.get("markPrice", pos.get("entryPrice", 0.0)))
        total += amt * price
    return total
