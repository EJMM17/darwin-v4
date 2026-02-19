"""
Darwin v4 — Portfolio Allocation Engine (PAE).

Dynamically allocates capital between symbols based on rolling
performance metrics. Independent layer — does not modify evolution,
fitness, GMRT, or harness internals.

Architecture:
    Per-symbol equity curves
        → rolling return, maxDD, PF, volatility
        → performance score per symbol
        → EMA-smoothed weights with floor/ceiling constraints
        → capital allocation

Usage:
    pae = PortfolioAllocationEngine(symbols=["BTC","SOL","DOGE"])
    pae.update("BTC", equity=105.0, pnl=+2.3)
    pae.update("SOL", equity=98.0, pnl=-1.5)
    pae.update("DOGE", equity=101.0, pnl=+0.8)
    weights = pae.get_weights()
    # {"BTC": 0.48, "SOL": 0.15, "DOGE": 0.37}
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List


# ── Configuration ────────────────────────────────────────────

@dataclass
class PAEConfig:
    """Tunable parameters for the allocation engine."""
    lookback: int = 200           # rolling window size (bars)
    min_weight: float = 0.10      # floor per symbol
    max_weight: float = 0.60      # ceiling per symbol
    ema_alpha: float = 0.05       # weight smoothing (lower = smoother)
    pf_clamp_lo: float = 0.0      # PF floor
    pf_clamp_hi: float = 3.0      # PF ceiling
    dd_floor: float = 0.01        # min maxDD to avoid div/zero (1%)
    vol_floor: float = 0.001      # min volatility to avoid div/zero
    warmup_bars: int = 50         # bars before dynamic allocation activates


# ── Per-symbol state ─────────────────────────────────────────

@dataclass
class SymbolState:
    """Tracks rolling equity and trade P&L for one symbol."""
    symbol: str
    equity_curve: List[float] = field(default_factory=list)
    pnl_history: List[float] = field(default_factory=list)
    raw_score: float = 1.0
    smoothed_weight: float = 0.0  # initialized at first rebalance
    n_updates: int = 0


# ── Allocation result ────────────────────────────────────────

@dataclass
class AllocationSnapshot:
    """Point-in-time allocation state."""
    bar: int
    weights: Dict[str, float]
    scores: Dict[str, float]
    metrics: Dict[str, Dict[str, float]]  # per-symbol rolling metrics


# ── Core engine ──────────────────────────────────────────────

class PortfolioAllocationEngine:
    """
    Dynamic capital allocation across symbols.

    Computes per-symbol performance scores from rolling metrics
    and converts to constrained, EMA-smoothed portfolio weights.

    Deterministic: no random state, pure function of input sequence.
    """

    def __init__(
        self,
        symbols: List[str],
        config: PAEConfig | None = None,
    ) -> None:
        self._config = config or PAEConfig()
        self._symbols = list(symbols)
        self._n = len(symbols)
        if self._n < 2:
            raise ValueError("Need at least 2 symbols")

        equal_w = 1.0 / self._n
        self._states: Dict[str, SymbolState] = {
            s: SymbolState(symbol=s, smoothed_weight=equal_w)
            for s in symbols
        }
        self._bar = 0
        self._history: List[AllocationSnapshot] = []
        self._initialized = False

    @property
    def symbols(self) -> List[str]:
        return list(self._symbols)

    @property
    def bar(self) -> int:
        return self._bar

    # ── Update interface ─────────────────────────────────────

    def update(self, symbol: str, equity: float, pnl: float = 0.0) -> None:
        """
        Feed new bar data for one symbol.

        Parameters
        ----------
        symbol : str
            Symbol identifier.
        equity : float
            Current equity allocated to this symbol.
        pnl : float
            P&L from the latest closed trade (0 if no trade closed).
        """
        st = self._states[symbol]
        st.equity_curve.append(equity)
        if pnl != 0.0:
            st.pnl_history.append(pnl)
        st.n_updates += 1

    def step(self) -> Dict[str, float]:
        """
        Advance one bar and recompute weights.

        Call after update() for all symbols in this bar.
        Returns current weight dict.
        """
        self._bar += 1
        cfg = self._config

        # Before warmup: equal weight
        min_updates = min(st.n_updates for st in self._states.values())
        if min_updates < cfg.warmup_bars:
            return self.get_weights()

        # Compute per-symbol metrics and scores
        scores = {}
        metrics = {}
        for sym in self._symbols:
            m = self._compute_rolling_metrics(sym)
            metrics[sym] = m
            scores[sym] = self._compute_score(m)
            self._states[sym].raw_score = scores[sym]

        # Convert scores to raw weights
        total_score = sum(scores.values())
        if total_score <= 0:
            raw_weights = {s: 1.0 / self._n for s in self._symbols}
        else:
            raw_weights = {s: scores[s] / total_score for s in self._symbols}

        # Apply constraints (floor, ceiling, renormalize)
        constrained = self._apply_constraints(raw_weights)

        # EMA smooth
        for sym in self._symbols:
            st = self._states[sym]
            if not self._initialized:
                st.smoothed_weight = constrained[sym]
            else:
                st.smoothed_weight = (
                    cfg.ema_alpha * constrained[sym]
                    + (1.0 - cfg.ema_alpha) * st.smoothed_weight
                )

        self._initialized = True

        # Final renormalize after EMA (may drift slightly)
        total_smooth = sum(st.smoothed_weight for st in self._states.values())
        if total_smooth > 0:
            for st in self._states.values():
                st.smoothed_weight /= total_smooth

        # Record snapshot
        snap = AllocationSnapshot(
            bar=self._bar,
            weights={s: round(self._states[s].smoothed_weight, 6) for s in self._symbols},
            scores={s: round(scores.get(s, 0), 6) for s in self._symbols},
            metrics=metrics,
        )
        self._history.append(snap)

        return self.get_weights()

    def get_weights(self) -> Dict[str, float]:
        """Current allocation weights. Always sums to 1.0."""
        return {s: self._states[s].smoothed_weight for s in self._symbols}

    def get_capital_allocation(
        self, total_capital: float,
    ) -> Dict[str, float]:
        """Convert weights to dollar amounts."""
        w = self.get_weights()
        return {s: total_capital * w[s] for s in self._symbols}

    # ── Rolling metrics ──────────────────────────────────────

    def _compute_rolling_metrics(self, symbol: str) -> Dict[str, float]:
        """Compute rolling return, maxDD, PF, volatility for symbol."""
        cfg = self._config
        st = self._states[symbol]
        eq = st.equity_curve
        pnls = st.pnl_history

        # Use last `lookback` bars of equity
        window = eq[-cfg.lookback:] if len(eq) >= cfg.lookback else eq

        if len(window) < 2:
            return {"return": 0.0, "maxdd": cfg.dd_floor,
                    "pf": 1.0, "volatility": cfg.vol_floor}

        # Return
        start_eq = window[0]
        end_eq = window[-1]
        ret = (end_eq - start_eq) / start_eq if start_eq > 0 else 0.0

        # MaxDD in window
        peak = window[0]
        max_dd = 0.0
        for v in window:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        max_dd = max(max_dd, cfg.dd_floor)

        # PF from recent P&L
        recent_pnls = pnls[-cfg.lookback:] if len(pnls) >= cfg.lookback else pnls
        gross_profit = sum(p for p in recent_pnls if p > 0)
        gross_loss = abs(sum(p for p in recent_pnls if p < 0))
        pf = gross_profit / gross_loss if gross_loss > 0 else 1.5
        pf = max(cfg.pf_clamp_lo, min(cfg.pf_clamp_hi, pf))

        # Volatility: std of equity returns
        eq_returns = [
            (window[i] - window[i - 1]) / window[i - 1]
            for i in range(1, len(window))
            if window[i - 1] > 0
        ]
        if len(eq_returns) >= 2:
            mu = sum(eq_returns) / len(eq_returns)
            var = sum((r - mu) ** 2 for r in eq_returns) / len(eq_returns)
            vol = math.sqrt(var)
        else:
            vol = cfg.vol_floor
        vol = max(vol, cfg.vol_floor)

        return {"return": round(ret, 6), "maxdd": round(max_dd, 6),
                "pf": round(pf, 4), "volatility": round(vol, 6)}

    @staticmethod
    def _compute_score(metrics: Dict[str, float]) -> float:
        """
        Performance score: Calmar-adjusted profit factor, volatility-penalized.

            score = max(0, pf × ret / maxDD) / sqrt(vol)

        Using sqrt(vol) instead of vol directly reduces the over-penalization
        of high-volatility crypto assets relative to their actual risk profile.

        Handles edge cases:
        - Negative return → score floored to 0 (gets minimum weight)
        - Zero maxdd → guarded by dd_floor in config (min 1%)
        - Very low volatility → guarded by vol_floor (min 0.001)
        """
        pf = metrics["pf"]
        ret = metrics["return"]
        maxdd = metrics["maxdd"]
        vol = metrics["volatility"]

        if ret <= 0:
            return 0.0
        # Use sqrt(vol) to avoid over-penalizing normally-volatile crypto
        raw = (pf * ret / maxdd) / (vol ** 0.5)
        return max(0.0, raw)

    # ── Constraint engine ────────────────────────────────────

    def _apply_constraints(
        self, raw_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Apply min/max weight constraints with iterative renormalization.

        Algorithm:
        1. Clamp all weights to [min, max]
        2. Redistribute excess to unclamped symbols
        3. Repeat until stable (max 10 iterations)
        """
        cfg = self._config
        weights = dict(raw_weights)
        for _iteration in range(10):
            clamped = {}
            excess = 0.0
            free_count = 0
            for s in self._symbols:
                w = weights[s]
                if w < cfg.min_weight:
                    clamped[s] = cfg.min_weight
                    excess += cfg.min_weight - w
                elif w > cfg.max_weight:
                    clamped[s] = cfg.max_weight
                    excess -= w - cfg.max_weight
                else:
                    clamped[s] = w
                    free_count += 1

            # Redistribute excess among free symbols
            if free_count > 0 and abs(excess) > 1e-10:
                adj = excess / free_count
                for s in self._symbols:
                    if cfg.min_weight < clamped[s] < cfg.max_weight:
                        clamped[s] -= adj

            # Renormalize
            total = sum(clamped.values())
            if total > 0:
                weights = {s: clamped[s] / total for s in self._symbols}
            else:
                weights = {s: 1.0 / self._n for s in self._symbols}

            # Check convergence
            all_valid = all(
                cfg.min_weight - 1e-6 <= weights[s] <= cfg.max_weight + 1e-6
                for s in self._symbols
            )
            if all_valid:
                break

        return weights

    # ── Reporting ────────────────────────────────────────────

    def print_summary(self, every_n: int = 50) -> str:
        """Print allocation history summary."""
        lines = []
        lines.append("=" * 90)
        lines.append("  PORTFOLIO ALLOCATION ENGINE — Dynamic Weights")
        lines.append("=" * 90)

        if not self._history:
            lines.append("  No history yet.")
            return "\n".join(lines)

        # Header
        sym_headers = "  ".join(f"{s:>8s}" for s in self._symbols)
        lines.append(f"\n  {'Bar':>5s}  {sym_headers}  |  {'Best':>8s}")
        lines.append("  " + "-" * (10 + 10 * self._n + 12))

        for snap in self._history:
            if snap.bar % every_n == 0 or snap is self._history[-1]:
                w_str = "  ".join(f"{snap.weights[s]:8.3f}" for s in self._symbols)
                best = max(snap.weights, key=snap.weights.get)
                lines.append(f"  {snap.bar:5d}  {w_str}  |  {best:>8s}")

        # Final weights
        lines.append("")
        lines.append("  FINAL WEIGHTS:")
        for s in self._symbols:
            w = self._states[s].smoothed_weight
            sc = self._states[s].raw_score
            lines.append(f"    {s:12s}: w={w:.4f}  score={sc:.4f}")

        # Weight stats
        all_w = {s: [] for s in self._symbols}
        for snap in self._history:
            for s in self._symbols:
                all_w[s].append(snap.weights[s])

        lines.append("\n  WEIGHT RANGES:")
        for s in self._symbols:
            ws = all_w[s]
            mn = min(ws); mx = max(ws)
            avg = sum(ws) / len(ws)
            lines.append(f"    {s:12s}: min={mn:.3f}  avg={avg:.3f}  max={mx:.3f}")

        out = "\n".join(lines)
        print(out)
        return out
