"""
Darwin v4 — Global Market Regime Throttle (GMRT).

Uses BTC as the macro reference asset to compute a continuous regime
score that modulates position sizing across the entire portfolio.

Standalone module. Does NOT modify evolution, fitness, DDT, or
cluster-aware initialization.

Architecture:
    BTC OHLCV (1H)
        → EMA50, EMA200, ATR14, rolling 30-day max drawdown
        → regime_score ∈ [-1, +1]
        → global_multiplier ∈ [0.2, 1.0]
        → final_size = base_size × symbol_weight × global_multiplier

Usage:
    gmrt = GlobalRegimeThrottle()
    gmrt.load_btc("/path/to/btc_1h.csv")
    gmrt.compute()

    # Per-bar access
    m = gmrt.multiplier_at(bar_index)
    size = base_size * symbol_weight * m

    # Full time series
    gmrt.print_summary(every_n=500)
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from typing import List


# ── Data structures ──────────────────────────────────────────

@dataclass
class RegimeBar:
    """Per-bar regime state."""
    index: int
    price: float
    ema50: float
    ema200: float
    atr14: float
    drawdown_30d: float        # rolling 30-day max drawdown [0, 1]
    trend_score: float         # tanh component [-1, +1]
    vol_score: float           # normalized ATR/price [0, 1]
    dd_score: float            # drawdown component [0, 1]
    regime_score: float        # composite [-1, +1]
    global_multiplier: float   # position sizing factor [0.2, 1.0]


@dataclass
class RegimeResult:
    """Full GMRT output."""
    bars: List[RegimeBar] = field(default_factory=list)
    n_bars: int = 0

    # Distribution counts
    n_full: int = 0       # multiplier = 1.0
    n_cautious: int = 0   # multiplier = 0.75
    n_defensive: int = 0  # multiplier = 0.5
    n_minimal: int = 0    # multiplier = 0.2

    def multiplier_at(self, index: int) -> float:
        """Get global multiplier for bar index. Safe for out-of-range."""
        if 0 <= index < len(self.bars):
            return self.bars[index].global_multiplier
        return 0.5  # default cautious if no data

    def regime_score_at(self, index: int) -> float:
        """Get regime score for bar index."""
        if 0 <= index < len(self.bars):
            return self.bars[index].regime_score
        return 0.0

    def print_summary(self, every_n: int = 500) -> str:
        """Print regime time series and distribution summary."""
        lines = []
        lines.append("=" * 90)
        lines.append("  GLOBAL MARKET REGIME THROTTLE — BTC Reference")
        lines.append("=" * 90)
        lines.append("")
        lines.append("  %6s  %10s  %7s  %7s  %7s  %7s  %5s" % (
            "Bar", "Price", "Trend", "Vol", "DD", "Regime", "Mult"))
        lines.append("  " + "-" * 62)

        for b in self.bars:
            if b.index % every_n == 0 or b.index == len(self.bars) - 1:
                lines.append(
                    "  %6d  $%9.2f  %+7.3f  %7.3f  %7.3f  %+7.3f  %.2f" % (
                        b.index, b.price, b.trend_score, b.vol_score,
                        b.dd_score, b.regime_score, b.global_multiplier))

        lines.append("")
        lines.append("  REGIME DISTRIBUTION (%d bars)" % self.n_bars)
        lines.append("  " + "-" * 40)
        pct = lambda n: n / self.n_bars * 100 if self.n_bars > 0 else 0
        lines.append("  Full      (1.00): %5d bars (%5.1f%%)" % (
            self.n_full, pct(self.n_full)))
        lines.append("  Cautious  (0.75): %5d bars (%5.1f%%)" % (
            self.n_cautious, pct(self.n_cautious)))
        lines.append("  Defensive (0.50): %5d bars (%5.1f%%)" % (
            self.n_defensive, pct(self.n_defensive)))
        lines.append("  Minimal   (0.20): %5d bars (%5.1f%%)" % (
            self.n_minimal, pct(self.n_minimal)))

        avg_mult = (sum(b.global_multiplier for b in self.bars) / self.n_bars
                    if self.n_bars > 0 else 0)
        avg_score = (sum(b.regime_score for b in self.bars) / self.n_bars
                     if self.n_bars > 0 else 0)
        lines.append("")
        lines.append("  Avg regime_score:    %+.3f" % avg_score)
        lines.append("  Avg multiplier:       %.3f" % avg_mult)
        lines.append("  Effective exposure:   %.1f%%" % (avg_mult * 100))

        lines.append("")
        lines.append("=" * 90)
        out = "\n".join(lines)
        print(out)
        return out


# ── Core throttle ────────────────────────────────────────────

class GlobalRegimeThrottle:
    """
    Compute global regime multiplier from BTC OHLCV reference data.

    The regime score blends three components:
      - Trend:    tanh((EMA50 - EMA200) / price × 5)
      - Vol:      normalized ATR14 / price
      - Drawdown: rolling 30-day max drawdown

    Mapping to multiplier:
      score > 0.3   → 1.00 (full risk)
      0.0 – 0.3     → 0.75 (cautious)
      -0.3 – 0.0    → 0.50 (defensive)
      score < -0.3   → 0.20 (minimal)

    Parameters
    ----------
    dd_window : int
        Rolling drawdown lookback in bars. Default 720 (30 days × 24h).
    """

    def __init__(self, dd_window: int = 720) -> None:
        self._dd_window = dd_window
        self._closes: List[float] = []
        self._highs: List[float] = []
        self._lows: List[float] = []
        self._result: RegimeResult | None = None

    def load_btc(self, csv_path: str) -> None:
        """Load BTC OHLCV from CSV."""
        self._closes = []
        self._highs = []
        self._lows = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._closes.append(float(row["close"]))
                self._highs.append(float(row["high"]))
                self._lows.append(float(row["low"]))

    def load_arrays(
        self,
        closes: List[float],
        highs: List[float],
        lows: List[float],
    ) -> None:
        """Load from pre-existing arrays (for integration)."""
        self._closes = list(closes)
        self._highs = list(highs)
        self._lows = list(lows)

    def compute(self) -> RegimeResult:
        """Run full GMRT computation. Returns RegimeResult."""
        n = len(self._closes)
        if n < 250:
            raise ValueError(f"Need ≥250 bars, got {n}")

        # Compute indicators
        ema50 = self._ema(self._closes, 50)
        ema200 = self._ema(self._closes, 200)
        atr14 = self._atr(self._highs, self._lows, self._closes, 14)
        dd30 = self._rolling_drawdown(self._closes, self._dd_window)

        # Normalize vol: compute ATR/price distribution for z-scoring
        vol_ratios = [
            atr14[i] / self._closes[i]
            for i in range(n)
            if self._closes[i] > 0 and atr14[i] > 0
        ]
        vol_mean = sum(vol_ratios) / len(vol_ratios) if vol_ratios else 0.01
        vol_var = (sum((v - vol_mean) ** 2 for v in vol_ratios) / len(vol_ratios)
                   if vol_ratios else 0.001)
        vol_std = math.sqrt(vol_var) if vol_var > 0 else 0.01

        result = RegimeResult(n_bars=n)

        for i in range(n):
            price = self._closes[i]
            e50 = ema50[i]
            e200 = ema200[i]
            atr = atr14[i]
            dd = dd30[i]

            # ── Component scores ──────────────────────────
            # Trend: positive = bullish, negative = bearish
            if price > 0 and e200 > 0:
                trend_score = math.tanh((e50 - e200) / price * 5.0)
            else:
                trend_score = 0.0

            # Vol: z-score normalized, clamped to [0, 1]
            # Higher vol → lower score (riskier)
            if price > 0 and atr > 0:
                vol_raw = atr / price
                vol_z = (vol_raw - vol_mean) / vol_std if vol_std > 0 else 0.0
                # Invert: high vol = low score. Sigmoid maps z to [0,1]
                vol_score = 1.0 / (1.0 + math.exp(vol_z))
            else:
                vol_score = 0.5

            # Drawdown: 0 = no drawdown (good), 1 = deep drawdown (bad)
            # Invert for regime: dd_score = 1 - dd (high = favorable)
            dd_score = max(0.0, min(1.0, dd))

            # ── Composite regime score ────────────────────
            # trend_score ∈ [-1, +1]  → weight 0.5
            # vol_score   ∈ [0, 1]    → weight 0.3, shift to [-1,+1]
            # dd_score    ∈ [0, 1]    → weight 0.4, invert (low dd = good)
            regime_score = (
                0.5 * trend_score
                + 0.3 * (vol_score * 2.0 - 1.0)   # [0,1] → [-1,+1]
                - 0.4 * dd_score                     # higher dd penalizes
            )
            # Clamp to [-1, +1]
            regime_score = max(-1.0, min(1.0, regime_score))

            # ── Map to multiplier ─────────────────────────
            if regime_score > 0.3:
                mult = 1.0
                result.n_full += 1
            elif regime_score > 0.0:
                mult = 0.75
                result.n_cautious += 1
            elif regime_score > -0.3:
                mult = 0.5
                result.n_defensive += 1
            else:
                mult = 0.2
                result.n_minimal += 1

            bar = RegimeBar(
                index=i, price=price, ema50=e50, ema200=e200,
                atr14=atr, drawdown_30d=dd,
                trend_score=round(trend_score, 6),
                vol_score=round(vol_score, 6),
                dd_score=round(dd_score, 6),
                regime_score=round(regime_score, 6),
                global_multiplier=mult,
            )
            result.bars.append(bar)

        self._result = result
        return result

    @property
    def result(self) -> RegimeResult | None:
        """Access last computed result."""
        return self._result

    # ── Position sizing integration ──────────────────────────

    @staticmethod
    def apply_multiplier(
        base_size: float,
        symbol_weight: float,
        global_multiplier: float,
    ) -> float:
        """
        Compute final position size with GMRT modulation.

        final = base_size × symbol_weight × global_multiplier

        Parameters
        ----------
        base_size : float
            Raw position size from agent genes (risk_pct × capital).
        symbol_weight : float
            Per-symbol allocation weight from portfolio (e.g. 0.4 for BTC).
        global_multiplier : float
            GMRT multiplier [0.2, 1.0] from regime_score.

        Returns
        -------
        float
            Final position size, always ≥ 0.
        """
        return max(0.0, base_size * symbol_weight * global_multiplier)

    # ── Technical indicators (zero-dependency) ───────────────

    @staticmethod
    def _ema(data: List[float], period: int) -> List[float]:
        """Exponential moving average. Same length as input."""
        n = len(data)
        out = [0.0] * n
        if n == 0 or period < 1:
            return out
        k = 2.0 / (period + 1)
        out[0] = data[0]
        for i in range(1, n):
            out[i] = data[i] * k + out[i - 1] * (1.0 - k)
        return out

    @staticmethod
    def _atr(
        highs: List[float], lows: List[float],
        closes: List[float], period: int = 14,
    ) -> List[float]:
        """Average True Range."""
        n = len(closes)
        tr = [highs[0] - lows[0]]
        for i in range(1, n):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i - 1])
            lc = abs(lows[i] - closes[i - 1])
            tr.append(max(hl, hc, lc))
        atr = [0.0] * n
        if n >= period:
            atr[period - 1] = sum(tr[:period]) / period
            for i in range(period, n):
                atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        return atr

    @staticmethod
    def _rolling_drawdown(
        closes: List[float], window: int,
    ) -> List[float]:
        """
        Rolling max drawdown over `window` bars.

        Returns list of drawdown values [0, 1] where:
          0 = price at peak (no drawdown)
          1 = 100% drawdown from peak
        """
        n = len(closes)
        dd = [0.0] * n
        for i in range(n):
            start = max(0, i - window + 1)
            peak = max(closes[start:i + 1])
            if peak > 0:
                dd[i] = (peak - closes[i]) / peak
        return dd
