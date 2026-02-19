"""
Darwin v5 — Monte Carlo Validation.

Runs simulated random sequences on factor outputs to produce a
distribution of outcomes and assess strategy risk-adjusted edge.

Evaluates the last N bars of factor scores to determine whether
the signal generator has a statistically significant edge.

Usage:
    mc = MonteCarloValidator()
    result = mc.validate(factor_scores, n_simulations=1000)
    if result.has_edge:
        # strategy shows positive expected value
"""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger("darwin.v5.monte_carlo")


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo validation."""
    n_simulations: int = 1000
    lookback_bars: int = 100
    confidence_level: float = 0.95
    min_edge_ratio: float = 0.1  # minimum edge ratio to be considered significant
    seed: Optional[int] = None


@dataclass(slots=True)
class MonteCarloResult:
    """Output of Monte Carlo validation."""
    n_simulations: int = 0
    actual_pnl: float = 0.0
    mean_random_pnl: float = 0.0
    std_random_pnl: float = 0.0
    p5_pnl: float = 0.0
    p25_pnl: float = 0.0
    p50_pnl: float = 0.0
    p75_pnl: float = 0.0
    p95_pnl: float = 0.0
    edge_ratio: float = 0.0  # (actual - mean_random) / std_random
    max_drawdown_p95: float = 0.0
    win_rate_actual: float = 0.0
    win_rate_random_mean: float = 0.0
    percentile_rank: float = 0.0  # where actual falls in random distribution

    @property
    def has_edge(self) -> bool:
        """Whether the strategy shows statistically significant edge."""
        return self.edge_ratio > 0.1 and self.percentile_rank > 0.5

    def to_dict(self) -> Dict[str, float]:
        return {
            "n_simulations": self.n_simulations,
            "actual_pnl": round(self.actual_pnl, 4),
            "mean_random_pnl": round(self.mean_random_pnl, 4),
            "std_random_pnl": round(self.std_random_pnl, 4),
            "p5_pnl": round(self.p5_pnl, 4),
            "p25_pnl": round(self.p25_pnl, 4),
            "p50_pnl": round(self.p50_pnl, 4),
            "p75_pnl": round(self.p75_pnl, 4),
            "p95_pnl": round(self.p95_pnl, 4),
            "edge_ratio": round(self.edge_ratio, 4),
            "max_drawdown_p95": round(self.max_drawdown_p95, 4),
            "win_rate_actual": round(self.win_rate_actual, 4),
            "win_rate_random_mean": round(self.win_rate_random_mean, 4),
            "percentile_rank": round(self.percentile_rank, 4),
            "has_edge": self.has_edge,
        }


class MonteCarloValidator:
    """
    Monte Carlo validation for signal quality.

    Takes historical factor outputs (signal scores and resulting PnLs)
    and runs random permutations to assess whether the signal generator
    has a statistically significant edge over random.

    Process:
        1. Collect actual signal → PnL mapping
        2. Randomly shuffle the PnL assignments N times
        3. Compare actual cumulative PnL to random distribution
        4. Compute edge ratio and percentile rank

    Parameters
    ----------
    config : MonteCarloConfig, optional
        Tunable parameters.
    """

    def __init__(self, config: MonteCarloConfig | None = None) -> None:
        self._config = config or MonteCarloConfig()
        self._rng = random.Random(self._config.seed)

    def validate(
        self,
        trade_pnls: List[float],
        signal_confidences: Optional[List[float]] = None,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo validation on historical trade results.

        Parameters
        ----------
        trade_pnls : list of float
            Realized PnL for each trade in chronological order.
        signal_confidences : list of float, optional
            Confidence scores for each trade (for weighted analysis).

        Returns
        -------
        MonteCarloResult
            Validation result with edge metrics.
        """
        cfg = self._config

        if len(trade_pnls) < 5:
            logger.warning("insufficient trades for Monte Carlo: %d", len(trade_pnls))
            return MonteCarloResult(n_simulations=0)

        # Use last N trades
        pnls = trade_pnls[-cfg.lookback_bars:]
        n_trades = len(pnls)

        # Actual performance
        actual_cum_pnl = sum(pnls)
        actual_win_rate = sum(1 for p in pnls if p > 0) / n_trades

        # Run random simulations
        random_pnls: List[float] = []
        random_drawdowns: List[float] = []
        random_win_rates: List[float] = []

        for _ in range(cfg.n_simulations):
            shuffled = list(pnls)
            self._rng.shuffle(shuffled)

            cum_pnl = sum(shuffled)
            random_pnls.append(cum_pnl)

            # Max drawdown of shuffled sequence
            dd = self._max_drawdown(shuffled)
            random_drawdowns.append(dd)

            # Win rate (same as actual since we just shuffle)
            wr = sum(1 for p in shuffled if p > 0) / n_trades
            random_win_rates.append(wr)

        # Statistics
        mean_random = sum(random_pnls) / len(random_pnls)
        var_random = sum((p - mean_random) ** 2 for p in random_pnls) / len(random_pnls)
        std_random = math.sqrt(var_random)

        # Edge ratio: how many std devs above random mean
        edge_ratio = 0.0
        if std_random > 1e-10:
            edge_ratio = (actual_cum_pnl - mean_random) / std_random

        # Percentile rank of actual in random distribution
        count_below = sum(1 for p in random_pnls if p < actual_cum_pnl)
        percentile_rank = count_below / len(random_pnls)

        # Percentiles of random distribution
        sorted_pnls = sorted(random_pnls)
        sorted_dds = sorted(random_drawdowns)

        return MonteCarloResult(
            n_simulations=cfg.n_simulations,
            actual_pnl=actual_cum_pnl,
            mean_random_pnl=mean_random,
            std_random_pnl=std_random,
            p5_pnl=_percentile(sorted_pnls, 5),
            p25_pnl=_percentile(sorted_pnls, 25),
            p50_pnl=_percentile(sorted_pnls, 50),
            p75_pnl=_percentile(sorted_pnls, 75),
            p95_pnl=_percentile(sorted_pnls, 95),
            edge_ratio=edge_ratio,
            max_drawdown_p95=_percentile(sorted_dds, 95),
            win_rate_actual=actual_win_rate,
            win_rate_random_mean=sum(random_win_rates) / len(random_win_rates),
            percentile_rank=percentile_rank,
        )

    def validate_confidence_distribution(
        self,
        confidences: List[float],
        pnls: List[float],
    ) -> Dict[str, float]:
        """
        Validate whether higher confidence signals produce better PnL.

        Splits trades into confidence buckets and compares.
        """
        if len(confidences) != len(pnls) or len(confidences) < 10:
            return {}

        # Split into quartiles
        paired = list(zip(confidences, pnls))
        paired.sort(key=lambda x: x[0])

        n = len(paired)
        q1 = paired[:n // 4]
        q4 = paired[3 * n // 4:]

        q1_avg = sum(p for _, p in q1) / len(q1) if q1 else 0
        q4_avg = sum(p for _, p in q4) / len(q4) if q4 else 0

        return {
            "low_confidence_avg_pnl": round(q1_avg, 4),
            "high_confidence_avg_pnl": round(q4_avg, 4),
            "confidence_edge": round(q4_avg - q1_avg, 4),
            "monotonic": q4_avg > q1_avg,
        }

    @staticmethod
    def _max_drawdown(pnls: List[float]) -> float:
        """Compute max drawdown from a PnL sequence."""
        peak = 0.0
        max_dd = 0.0
        cumulative = 0.0

        for pnl in pnls:
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd

        return max_dd


def _percentile(sorted_values: List[float], p: float) -> float:
    """Compute percentile from sorted list."""
    if not sorted_values:
        return 0.0
    k = (p / 100.0) * (len(sorted_values) - 1)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    d = k - f
    return sorted_values[f] + d * (sorted_values[c] - sorted_values[f])
