"""
Performance Analytics — Institutional-Grade Risk Metrics

Computes real-time Sharpe, Sortino, Calmar, and other institutional
metrics used by crypto hedge funds (AIMA/PwC 2024 standards).

Institutional thresholds (XBTO / Breaking Alpha research 2025):
  Sharpe  > 1.0  → acceptable,  > 2.0 → institutional grade
  Sortino > 2.0  → good,        > 3.0 → excellent
  Calmar  > 1.0  → acceptable,  > 2.0 → elite
  Max DD  < 10%  → conservative, < 20% → acceptable for crypto
  Win rate> 50%  → required,    > 55% → good edge

Usage:
    analytics = PerformanceAnalytics()
    analytics.record_equity(timestamp, equity)
    analytics.record_trade(pnl_usdt, pnl_pct)
    report = analytics.get_report()
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class EquityPoint:
    ts: float        # unix timestamp
    equity: float


@dataclass
class TradeRecord:
    ts: float
    pnl_usdt: float
    pnl_pct: float   # as decimal, e.g. 0.02 for +2%


@dataclass
class PerformanceReport:
    # Core returns
    total_return_pct: float      # e.g. 12.5 (%)
    annualized_return_pct: float

    # Risk-adjusted (institutional trinity)
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Drawdown
    max_drawdown_pct: float      # e.g. -8.3 (%)
    current_drawdown_pct: float

    # Trade statistics
    total_trades: int
    win_rate_pct: float          # e.g. 55.0 (%)
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float         # gross_wins / gross_losses
    avg_trade_pct: float
    best_trade_pct: float
    worst_trade_pct: float

    # Risk
    volatility_annualized_pct: float
    downside_vol_annualized_pct: float

    # Meta
    elapsed_days: float
    equity_start: float
    equity_current: float
    sample_size_returns: int     # number of return observations used

    # Institutional grade flags
    @property
    def sharpe_grade(self) -> str:
        if self.sharpe_ratio >= 2.0:
            return "INSTITUTIONAL"
        if self.sharpe_ratio >= 1.0:
            return "ACCEPTABLE"
        return "BELOW_THRESHOLD"

    @property
    def sortino_grade(self) -> str:
        if self.sortino_ratio >= 3.0:
            return "EXCELLENT"
        if self.sortino_ratio >= 2.0:
            return "GOOD"
        return "BELOW_THRESHOLD"

    @property
    def calmar_grade(self) -> str:
        if self.calmar_ratio >= 2.0:
            return "ELITE"
        if self.calmar_ratio >= 1.0:
            return "ACCEPTABLE"
        return "BELOW_THRESHOLD"

    @property
    def overall_grade(self) -> str:
        """Single grade for hedge fund readiness."""
        score = 0
        if self.sharpe_ratio >= 1.0:
            score += 1
        if self.sharpe_ratio >= 2.0:
            score += 1
        if self.sortino_ratio >= 2.0:
            score += 1
        if self.calmar_ratio >= 1.0:
            score += 1
        if self.max_drawdown_pct > -20.0:
            score += 1
        if self.win_rate_pct >= 50.0:
            score += 1
        if self.profit_factor >= 1.5:
            score += 1

        if score >= 6:
            return "HEDGE_FUND_READY"
        if score >= 4:
            return "DEVELOPING"
        return "RETAIL"

    def to_telegram(self) -> str:
        """Formatted message for Telegram daily report."""
        grade_emoji = {
            "HEDGE_FUND_READY": "🏦",
            "DEVELOPING": "📈",
            "RETAIL": "🌱",
        }
        emoji = grade_emoji.get(self.overall_grade, "📊")

        dd_emoji = "🟢" if self.current_drawdown_pct > -5 else ("🟡" if self.current_drawdown_pct > -10 else "🔴")
        sharpe_emoji = "✅" if self.sharpe_ratio >= 1.0 else "⚠️"
        sortino_emoji = "✅" if self.sortino_ratio >= 2.0 else "⚠️"
        calmar_emoji = "✅" if self.calmar_ratio >= 1.0 else "⚠️"

        return (
            f"{emoji} *Darwin v5 — Performance Report*\n"
            f"═══════════════════════════\n"
            f"*Capital:* ${self.equity_current:.2f} "
            f"({'+' if self.total_return_pct >= 0 else ''}{self.total_return_pct:.2f}%)\n"
            f"*Días activo:* {self.elapsed_days:.1f}\n"
            f"\n"
            f"📐 *Risk-Adjusted Metrics*\n"
            f"{sharpe_emoji} Sharpe:  {self.sharpe_ratio:+.3f} [{self.sharpe_grade}]\n"
            f"{sortino_emoji} Sortino: {self.sortino_ratio:+.3f} [{self.sortino_grade}]\n"
            f"{calmar_emoji} Calmar:  {self.calmar_ratio:+.3f} [{self.calmar_grade}]\n"
            f"\n"
            f"📉 *Drawdown*\n"
            f"{dd_emoji} Max DD: {self.max_drawdown_pct:.2f}%\n"
            f"   Current: {self.current_drawdown_pct:.2f}%\n"
            f"\n"
            f"🎯 *Trade Stats* ({self.total_trades} trades)\n"
            f"   Win rate: {self.win_rate_pct:.1f}%\n"
            f"   Profit factor: {self.profit_factor:.2f}x\n"
            f"   Avg win: {self.avg_win_pct:+.2f}% | Avg loss: {self.avg_loss_pct:.2f}%\n"
            f"   Best: {self.best_trade_pct:+.2f}% | Worst: {self.worst_trade_pct:.2f}%\n"
            f"\n"
            f"📊 *Volatility*\n"
            f"   Ann. Vol: {self.volatility_annualized_pct:.1f}%\n"
            f"   Downside Vol: {self.downside_vol_annualized_pct:.1f}%\n"
            f"\n"
            f"🏷️ *Overall Grade: {self.overall_grade}* {emoji}\n"
            f"═══════════════════════════"
        )

    def to_log(self) -> str:
        """Compact single-line for log files."""
        return (
            f"PERF sharpe={self.sharpe_ratio:.3f} sortino={self.sortino_ratio:.3f} "
            f"calmar={self.calmar_ratio:.3f} maxdd={self.max_drawdown_pct:.2f}% "
            f"winrate={self.win_rate_pct:.1f}% pf={self.profit_factor:.2f} "
            f"trades={self.total_trades} grade={self.overall_grade}"
        )


# ─── Core analytics engine ────────────────────────────────────────────────────

class PerformanceAnalytics:
    """
    Computes institutional performance metrics from equity curve and trade history.

    Design principles:
    - Pure Python, no external deps (numpy/pandas not available in prod container)
    - O(1) memory with rolling windows (deque with maxlen)
    - All annualization assumes 365 days (crypto never sleeps)
    - Sortino uses 0% MAR (minimum acceptable return) — standard for crypto
    - Sharpe uses 0% risk-free rate — appropriate for crypto futures

    The metrics match what AIMA/PwC crypto hedge fund reports use:
      Sharpe, Sortino, Calmar (3-year rolling, but we use all available data)
    """

    # Crypto: 365 days × 24h × 12 ticks/hour (5 min) = 105,120 ticks/year
    # We record equity once per tick (5s) → 6,307,200 points/year — too many
    # Instead, we sample equity at heartbeat (60s) → 525,960 points/year
    # For analytics we use hourly snapshots stored in a rolling 365-day window
    HOURS_PER_YEAR = 365 * 24  # 8760

    def __init__(self, max_equity_points: int = 8760 * 7):
        # Equity curve: rolling 7-year window (way more than enough)
        self._equity_history: Deque[EquityPoint] = deque(maxlen=max_equity_points)
        # Trade history: all trades (bounded by max_trades)
        self._trades: Deque[TradeRecord] = deque(maxlen=10_000)
        # Peak equity for drawdown calculation
        self._peak_equity: float = 0.0
        # Start values
        self._start_equity: float = 0.0
        self._start_ts: float = time.time()

    def record_equity(self, equity: float, ts: Optional[float] = None) -> None:
        """Record equity snapshot. Call at heartbeat frequency (60s)."""
        if ts is None:
            ts = time.time()
        if self._start_equity == 0.0 and equity > 0:
            self._start_equity = equity
            self._start_ts = ts
        self._equity_history.append(EquityPoint(ts=ts, equity=equity))
        if equity > self._peak_equity:
            self._peak_equity = equity

    def record_trade(self, pnl_usdt: float, pnl_pct: float,
                     ts: Optional[float] = None) -> None:
        """Record a completed trade."""
        if ts is None:
            ts = time.time()
        self._trades.append(TradeRecord(ts=ts, pnl_usdt=pnl_usdt, pnl_pct=pnl_pct))

    def get_report(self) -> Optional[PerformanceReport]:
        """
        Compute full performance report. Returns None if insufficient data.
        Requires at least 2 equity points and 1 trade to compute anything useful.
        """
        if len(self._equity_history) < 2:
            return None

        eq_points = list(self._equity_history)
        trades = list(self._trades)

        equity_start = self._start_equity or eq_points[0].equity
        equity_current = eq_points[-1].equity
        elapsed_secs = eq_points[-1].ts - self._start_ts
        elapsed_days = max(elapsed_secs / 86400.0, 1.0 / 1440.0)  # min 1 minute

        # ── 1. Total and annualized return ──────────────────────────────────
        total_return = (equity_current - equity_start) / equity_start
        total_return_pct = total_return * 100.0

        # Compound annualized: (1 + r)^(365/days) - 1
        years = elapsed_days / 365.0
        if years > 0 and equity_start > 0:
            ann_return = ((equity_current / equity_start) ** (1.0 / years)) - 1.0
        else:
            ann_return = 0.0
        annualized_return_pct = ann_return * 100.0

        # ── 2. Returns series (hourly) ───────────────────────────────────────
        returns = _compute_returns_series(eq_points)
        n_returns = len(returns)

        # ── 3. Volatility (annualized) ──────────────────────────────────────
        vol_ann = _annualized_vol(returns, elapsed_days)
        downside_vol_ann = _annualized_downside_vol(returns, elapsed_days, mar=0.0)

        # ── 4. Sharpe ratio (risk-free = 0, annualized) ─────────────────────
        # Sharpe = annualized_return / annualized_vol
        if vol_ann > 1e-10:
            sharpe = ann_return / vol_ann
        else:
            sharpe = 0.0

        # ── 5. Sortino ratio (MAR = 0) ──────────────────────────────────────
        if downside_vol_ann > 1e-10:
            sortino = ann_return / downside_vol_ann
        else:
            sortino = ann_return * 10.0 if ann_return > 0 else 0.0

        # ── 6. Maximum drawdown ──────────────────────────────────────────────
        max_dd, current_dd = _compute_drawdowns(eq_points)
        max_dd_pct = max_dd * 100.0
        current_dd_pct = current_dd * 100.0

        # ── 7. Calmar ratio = annualized_return / |max_drawdown| ────────────
        if abs(max_dd) > 1e-10:
            calmar = ann_return / abs(max_dd)
        else:
            calmar = ann_return * 10.0 if ann_return > 0 else 0.0

        # ── 8. Trade statistics ──────────────────────────────────────────────
        total_trades = len(trades)
        wins = [t for t in trades if t.pnl_pct > 0]
        losses = [t for t in trades if t.pnl_pct <= 0]

        win_rate = len(wins) / total_trades * 100.0 if total_trades > 0 else 0.0
        avg_win = (sum(t.pnl_pct for t in wins) / len(wins) * 100.0) if wins else 0.0
        avg_loss = (sum(t.pnl_pct for t in losses) / len(losses) * 100.0) if losses else 0.0
        avg_trade = (sum(t.pnl_pct for t in trades) / total_trades * 100.0) if trades else 0.0

        gross_wins = sum(t.pnl_usdt for t in wins) if wins else 0.0
        gross_losses = abs(sum(t.pnl_usdt for t in losses)) if losses else 0.0
        profit_factor = gross_wins / gross_losses if gross_losses > 1e-10 else (999.0 if gross_wins > 0 else 0.0)

        best_trade = max((t.pnl_pct * 100.0 for t in trades), default=0.0)
        worst_trade = min((t.pnl_pct * 100.0 for t in trades), default=0.0)

        return PerformanceReport(
            total_return_pct=total_return_pct,
            annualized_return_pct=annualized_return_pct,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown_pct=max_dd_pct,
            current_drawdown_pct=current_dd_pct,
            total_trades=total_trades,
            win_rate_pct=win_rate,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            profit_factor=profit_factor,
            avg_trade_pct=avg_trade,
            best_trade_pct=best_trade,
            worst_trade_pct=worst_trade,
            volatility_annualized_pct=vol_ann * 100.0,
            downside_vol_annualized_pct=downside_vol_ann * 100.0,
            elapsed_days=elapsed_days,
            equity_start=equity_start,
            equity_current=equity_current,
            sample_size_returns=n_returns,
        )


# ─── Pure math helpers (no deps) ─────────────────────────────────────────────

def _compute_returns_series(eq_points: List[EquityPoint]) -> List[float]:
    """Period-over-period returns from equity curve."""
    returns = []
    for i in range(1, len(eq_points)):
        prev = eq_points[i - 1].equity
        curr = eq_points[i].equity
        if prev > 1e-10:
            returns.append((curr - prev) / prev)
    return returns


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float], mu: Optional[float] = None) -> float:
    if len(xs) < 2:
        return 0.0
    if mu is None:
        mu = _mean(xs)
    variance = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(variance)


def _annualized_vol(returns: List[float], elapsed_days: float) -> float:
    """
    Annualized volatility from return series.

    We scale by sqrt(observations_per_year). Since equity is recorded at
    heartbeat frequency (60s), there are 365*24*60 = 525,600 obs/year.
    But we use actual elapsed time for the scaling factor.
    """
    if len(returns) < 2:
        return 0.0
    sigma_period = _std(returns)
    # obs_per_year = number of observations per year at current sampling rate
    obs_count = len(returns)
    obs_per_year = obs_count / max(elapsed_days / 365.0, 1e-6)
    return sigma_period * math.sqrt(obs_per_year)


def _annualized_downside_vol(returns: List[float], elapsed_days: float,
                              mar: float = 0.0) -> float:
    """
    Annualized downside deviation (for Sortino ratio).
    Only penalizes returns below MAR (minimum acceptable return).
    """
    if len(returns) < 2:
        return 0.0
    downside = [min(r - mar, 0.0) for r in returns]
    sq_sum = sum(d ** 2 for d in downside)
    downside_variance = sq_sum / max(len(returns) - 1, 1)
    sigma_period = math.sqrt(downside_variance)
    obs_count = len(returns)
    obs_per_year = obs_count / max(elapsed_days / 365.0, 1e-6)
    return sigma_period * math.sqrt(obs_per_year)


def _compute_drawdowns(eq_points: List[EquityPoint]) -> Tuple[float, float]:
    """
    Returns (max_drawdown, current_drawdown) as negative fractions.
    e.g. (-0.083, -0.02) means max DD was -8.3%, currently -2%.
    """
    peak = eq_points[0].equity
    max_dd = 0.0
    for ep in eq_points:
        if ep.equity > peak:
            peak = ep.equity
        if peak > 1e-10:
            dd = (ep.equity - peak) / peak
            if dd < max_dd:
                max_dd = dd

    # Current drawdown from all-time peak
    last_equity = eq_points[-1].equity
    peak_all = max(ep.equity for ep in eq_points)
    current_dd = (last_equity - peak_all) / peak_all if peak_all > 1e-10 else 0.0

    return max_dd, current_dd
