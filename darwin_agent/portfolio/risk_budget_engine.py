"""
Darwin v4 — Portfolio-Level Risk Budget Engine (RBE).

Dynamically scales total system exposure based on portfolio health.
Independent layer — does NOT modify evolution, fitness, GMRT, or PAE.

Architecture:
    Portfolio equity stream
        → rolling drawdown, PF, volatility
        → dd_penalty × vol_penalty + pf_bonus
        → rbe_mult ∈ [0.30, 1.00]
        → final_size = capital × risk_pct × weight × gmrt × rbe_mult

Usage:
    rbe = RiskBudgetEngine()
    rbe.update(equity=105.0, pnl=+2.3)
    mult = rbe.step()  # → 0.85
    final_size = capital * risk_pct * weight * gmrt_mult * mult
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List


# ── Configuration ────────────────────────────────────────────

@dataclass
class RBEConfig:
    """Tunable parameters for the Risk Budget Engine."""
    # Live-calibrated: tighter dd_limit + higher floor keeps system active through drawdowns
    dd_limit: float = 0.32       # max acceptable drawdown (32%) — tighter throttle onset
    target_vol: float = 0.06     # target 4H volatility (wider for crypto regime)
    window: int = 200            # rolling lookback bars
    mult_floor: float = 0.50     # minimum multiplier — prevents stagnation at 0.30
    mult_ceil: float = 1.00      # maximum multiplier
    warmup: int = 30             # bars before RBE activates


# ── Snapshot for diagnostics ─────────────────────────────────

@dataclass
class RBESnapshot:
    """Point-in-time RBE state."""
    bar: int
    equity: float
    peak: float
    drawdown: float
    rolling_pf: float
    rolling_vol: float
    dd_penalty: float
    pf_bonus: float
    vol_penalty: float
    rbe_mult: float


# ── Core engine ──────────────────────────────────────────────

class RiskBudgetEngine:
    """
    Portfolio-level exposure scaling based on health metrics.

    Maintains rolling equity state and computes a single multiplier
    that throttles the entire portfolio's position sizing.

    Three components blend into rbe_mult:
      - dd_penalty:  reduces exposure as drawdown approaches dd_limit
      - vol_penalty: scales down when realized vol exceeds target
      - pf_bonus:    small reward/penalty based on rolling profit factor

    Formula:
        dd_ratio    = current_dd / dd_limit
        dd_penalty  = clamp(1 - dd_ratio, 0.30, 1.00)
        pf_bonus    = clamp((rolling_pf - 1.0) × 0.5, -0.10, +0.10)
        vol_penalty = clamp(target_vol / rolling_vol, 0.70, 1.10)
        rbe_mult    = clamp(dd_penalty × vol_penalty + pf_bonus, 0.30, 1.00)

    Deterministic: pure function of input sequence.
    """

    def __init__(self, config: RBEConfig | None = None) -> None:
        self._cfg = config or RBEConfig()
        self._equity_curve: List[float] = []
        self._pnl_history: List[float] = []
        self._peak: float = 0.0
        self._bar: int = 0
        self._last_mult: float = 1.0
        self._history: List[RBESnapshot] = []

    @property
    def bar(self) -> int:
        return self._bar

    @property
    def last_mult(self) -> float:
        return self._last_mult

    @property
    def history(self) -> List[RBESnapshot]:
        return self._history

    @property
    def peak(self) -> float:
        return self._peak

    @property
    def current_drawdown(self) -> float:
        if self._peak <= 0 or not self._equity_curve:
            return 0.0
        return (self._peak - self._equity_curve[-1]) / self._peak

    # ── Update + step ────────────────────────────────────────

    def update(self, equity: float, pnl: float = 0.0) -> None:
        """
        Feed new portfolio-level data.

        Parameters
        ----------
        equity : float
            Current total portfolio equity.
        pnl : float
            Aggregate P&L from latest bar (0 if no trades closed).
        """
        self._equity_curve.append(equity)
        if equity > self._peak:
            self._peak = equity
        if pnl != 0.0:
            self._pnl_history.append(pnl)

    def step(self) -> float:
        """
        Advance one bar and compute risk multiplier.

        Returns
        -------
        float
            rbe_mult ∈ [0.30, 1.00]
        """
        self._bar += 1
        cfg = self._cfg

        # Warmup: full exposure
        if len(self._equity_curve) < cfg.warmup:
            self._last_mult = cfg.mult_ceil
            self._record_snapshot(
                dd=0.0, pf=1.0, vol=cfg.target_vol,
                dd_pen=1.0, pf_bon=0.0, vol_pen=1.0,
                mult=cfg.mult_ceil,
            )
            return self._last_mult

        # ── Current drawdown ─────────────────────────────
        cur_equity = self._equity_curve[-1]
        dd = (self._peak - cur_equity) / self._peak if self._peak > 0 else 0.0
        dd = max(0.0, dd)

        # ── DD penalty ───────────────────────────────────
        dd_ratio = dd / cfg.dd_limit if cfg.dd_limit > 0 else 0.0
        dd_penalty = self._clamp(1.0 - dd_ratio, 0.30, 1.00)

        # ── Rolling PF ───────────────────────────────────
        recent_pnl = self._pnl_history[-cfg.window:]
        gross_profit = sum(p for p in recent_pnl if p > 0)
        gross_loss = abs(sum(p for p in recent_pnl if p < 0))
        rolling_pf = gross_profit / gross_loss if gross_loss > 0 else 1.5
        pf_bonus = self._clamp((rolling_pf - 1.0) * 0.5, -0.10, +0.10)

        # ── Rolling volatility ───────────────────────────
        eq_window = self._equity_curve[-cfg.window:]
        if len(eq_window) >= 2:
            returns = []
            for i in range(1, len(eq_window)):
                if eq_window[i - 1] > 0:
                    returns.append(
                        (eq_window[i] - eq_window[i - 1]) / eq_window[i - 1]
                    )
            if len(returns) >= 2:
                mu = sum(returns) / len(returns)
                var = sum((r - mu) ** 2 for r in returns) / len(returns)
                rolling_vol = math.sqrt(var)
            else:
                rolling_vol = cfg.target_vol
        else:
            rolling_vol = cfg.target_vol

        # Floor volatility to avoid division by zero
        rolling_vol = max(rolling_vol, 1e-8)
        vol_penalty = self._clamp(
            cfg.target_vol / rolling_vol, 0.70, 1.10
        )

        # ── Composite multiplier ─────────────────────────
        # PF bonus modifies the base before vol scaling so it doesn't
        # bypass volatility risk: raw = (dd_penalty + pf_bonus) * vol_penalty
        raw = (dd_penalty + pf_bonus) * vol_penalty
        rbe_mult = self._clamp(raw, cfg.mult_floor, cfg.mult_ceil)

        self._last_mult = rbe_mult
        self._record_snapshot(
            dd=dd, pf=rolling_pf, vol=rolling_vol,
            dd_pen=dd_penalty, pf_bon=pf_bonus, vol_pen=vol_penalty,
            mult=rbe_mult,
        )
        return rbe_mult

    # ── Helpers ──────────────────────────────────────────────

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    def _record_snapshot(
        self, dd: float, pf: float, vol: float,
        dd_pen: float, pf_bon: float, vol_pen: float, mult: float,
    ) -> None:
        eq = self._equity_curve[-1] if self._equity_curve else 0.0
        self._history.append(RBESnapshot(
            bar=self._bar, equity=round(eq, 4), peak=round(self._peak, 4),
            drawdown=round(dd, 6), rolling_pf=round(pf, 4),
            rolling_vol=round(vol, 6),
            dd_penalty=round(dd_pen, 6), pf_bonus=round(pf_bon, 6),
            vol_penalty=round(vol_pen, 6), rbe_mult=round(mult, 6),
        ))

    # ── Static integration helper ────────────────────────────

    @staticmethod
    def apply_full_stack(
        capital: float,
        risk_pct: float,
        asset_weight: float,
        gmrt_mult: float,
        rbe_mult: float,
    ) -> float:
        """
        Compute final position size through the full risk stack.

        final = capital × risk_pct × asset_weight × gmrt_mult × rbe_mult

        All multipliers are independent and compose multiplicatively.
        """
        return max(0.0, capital * risk_pct * asset_weight * gmrt_mult * rbe_mult)

    # ── Reporting ────────────────────────────────────────────

    def print_summary(self, every_n: int = 50) -> str:
        """Print RBE state history."""
        lines = []
        lines.append("=" * 95)
        lines.append("  RISK BUDGET ENGINE — Portfolio Health Monitor")
        lines.append("=" * 95)

        if not self._history:
            lines.append("  No history yet.")
            return "\n".join(lines)

        lines.append(f"\n  Config: dd_limit={self._cfg.dd_limit:.0%}  "
                     f"target_vol={self._cfg.target_vol:.2%}  "
                     f"window={self._cfg.window}  "
                     f"range=[{self._cfg.mult_floor}, {self._cfg.mult_ceil}]")

        lines.append(f"\n  {'Bar':>5s}  {'Equity':>9s}  {'DD':>6s}  "
                     f"{'DDpen':>6s}  {'PF':>5s}  {'PFbon':>6s}  "
                     f"{'Vol':>7s}  {'Vpen':>5s}  {'Mult':>5s}")
        lines.append("  " + "-" * 72)

        for s in self._history:
            if s.bar % every_n == 0 or s is self._history[-1]:
                lines.append(
                    f"  {s.bar:5d}  ${s.equity:8.2f}  {s.drawdown:5.1%}  "
                    f"{s.dd_penalty:6.3f}  {s.rolling_pf:5.2f}  "
                    f"{s.pf_bonus:+5.3f}  {s.rolling_vol:7.4f}  "
                    f"{s.vol_penalty:5.3f}  {s.rbe_mult:5.3f}"
                )

        # Distribution
        mults = [s.rbe_mult for s in self._history]
        n_full = sum(1 for m in mults if m >= 0.95)
        n_reduced = sum(1 for m in mults if 0.60 <= m < 0.95)
        n_throttled = sum(1 for m in mults if 0.40 <= m < 0.60)
        n_minimal = sum(1 for m in mults if m < 0.40)
        total = len(mults)
        pct = lambda n: n / total * 100 if total > 0 else 0

        lines.append(f"\n  MULTIPLIER DISTRIBUTION ({total} bars):")
        lines.append(f"    Full     [0.95-1.00]: {n_full:5d} ({pct(n_full):5.1f}%)")
        lines.append(f"    Reduced  [0.60-0.95]: {n_reduced:5d} ({pct(n_reduced):5.1f}%)")
        lines.append(f"    Throttle [0.40-0.60]: {n_throttled:5d} ({pct(n_throttled):5.1f}%)")
        lines.append(f"    Minimal  [0.30-0.40]: {n_minimal:5d} ({pct(n_minimal):5.1f}%)")
        lines.append(f"    Avg multiplier: {sum(mults)/len(mults):.3f}")

        out = "\n".join(lines)
        print(out)
        return out
