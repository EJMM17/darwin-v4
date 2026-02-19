"""
Darwin v5 — Portfolio Construction.

Ranks tradable symbols by risk-adjusted alpha, computes relative weights,
and enforces exposure limits.

Usage:
    pc = PortfolioConstructor(symbols, config)
    pc.update_signal("BTCUSDT", signal)
    pc.update_metrics("BTCUSDT", sharpe=1.2, vol=0.03, pnl=100.0)
    weights = pc.construct()
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger("darwin.v5.portfolio")


@dataclass
class PortfolioConfig:
    """Configuration for portfolio construction."""
    max_total_exposure_mult: float = 5.0  # max 5x leverage total
    max_symbol_exposure_pct: float = 25.0  # max 25% of equity per symbol
    min_symbol_weight: float = 0.05       # minimum weight per symbol
    max_symbol_weight: float = 0.50       # maximum weight per symbol
    # Ranking weights
    alpha_weight: float = 0.40
    sharpe_weight: float = 0.30
    vol_penalty_weight: float = 0.20
    momentum_weight: float = 0.10
    # Smoothing
    ema_alpha: float = 0.1


@dataclass(slots=True)
class SymbolRanking:
    """Ranking data for a single symbol."""
    symbol: str = ""
    alpha_score: float = 0.0
    sharpe_ratio: float = 0.0
    realized_vol: float = 0.0
    momentum_score: float = 0.0
    composite_score: float = 0.0
    raw_weight: float = 0.0
    final_weight: float = 0.0


@dataclass(slots=True)
class PortfolioSnapshot:
    """Point-in-time portfolio construction result."""
    rankings: List[SymbolRanking] = field(default_factory=list)
    weights: Dict[str, float] = field(default_factory=dict)
    total_exposure_pct: float = 0.0


class PortfolioConstructor:
    """
    Constructs portfolio weights from signal quality and risk metrics.

    Process:
        1. Compute risk-adjusted alpha score per symbol
        2. Rank symbols by composite score
        3. Convert scores to weights with min/max constraints
        4. Enforce total exposure limits

    Parameters
    ----------
    symbols : list of str
        Tradable symbols.
    config : PortfolioConfig, optional
        Tunable parameters.
    """

    def __init__(
        self,
        symbols: List[str],
        config: PortfolioConfig | None = None,
    ) -> None:
        self._config = config or PortfolioConfig()
        self._symbols = list(symbols)
        self._metrics: Dict[str, Dict[str, float]] = {s: {} for s in symbols}
        self._signal_scores: Dict[str, float] = {s: 0.0 for s in symbols}
        self._smoothed_weights: Dict[str, float] = {
            s: 1.0 / len(symbols) for s in symbols
        }
        self._initialized = False

    def update_signal(self, symbol: str, signal: Any) -> None:
        """Update signal score for a symbol."""
        if symbol in self._signal_scores:
            confidence = getattr(signal, "confidence", 0.0)
            self._signal_scores[symbol] = confidence

    def update_metrics(
        self,
        symbol: str,
        sharpe: float = 0.0,
        realized_vol: float = 0.0,
        cumulative_pnl: float = 0.0,
        momentum_score: float = 0.0,
    ) -> None:
        """Update risk metrics for a symbol."""
        if symbol in self._metrics:
            self._metrics[symbol] = {
                "sharpe": sharpe,
                "realized_vol": realized_vol,
                "cumulative_pnl": cumulative_pnl,
                "momentum_score": momentum_score,
            }

    def construct(self) -> PortfolioSnapshot:
        """
        Construct portfolio weights from current metrics and signals.

        Returns
        -------
        PortfolioSnapshot
            Rankings and final weights.
        """
        cfg = self._config
        rankings: List[SymbolRanking] = []

        for symbol in self._symbols:
            metrics = self._metrics.get(symbol, {})
            signal_score = self._signal_scores.get(symbol, 0.0)

            sharpe = metrics.get("sharpe", 0.0)
            vol = metrics.get("realized_vol", 0.01)
            pnl = metrics.get("cumulative_pnl", 0.0)
            mom = metrics.get("momentum_score", 0.0)

            # Alpha score: signal quality + PnL contribution
            alpha_score = signal_score + max(0, pnl / 100.0)

            # Composite score
            vol_penalty = 1.0 / (1.0 + vol * 10.0)  # lower vol = higher score
            composite = (
                cfg.alpha_weight * alpha_score
                + cfg.sharpe_weight * max(0, sharpe)
                + cfg.vol_penalty_weight * vol_penalty
                + cfg.momentum_weight * max(0, mom)
            )

            rankings.append(SymbolRanking(
                symbol=symbol,
                alpha_score=alpha_score,
                sharpe_ratio=sharpe,
                realized_vol=vol,
                momentum_score=mom,
                composite_score=max(0.0, composite),
            ))

        # Sort by composite score descending
        rankings.sort(key=lambda r: r.composite_score, reverse=True)

        # Convert to raw weights
        total_score = sum(r.composite_score for r in rankings)
        if total_score > 0:
            for r in rankings:
                r.raw_weight = r.composite_score / total_score
        else:
            equal = 1.0 / len(rankings) if rankings else 0
            for r in rankings:
                r.raw_weight = equal

        # Apply constraints
        constrained = self._apply_constraints(
            {r.symbol: r.raw_weight for r in rankings}
        )

        # EMA smoothing
        for r in rankings:
            if self._initialized:
                prev = self._smoothed_weights.get(r.symbol, constrained[r.symbol])
                r.final_weight = cfg.ema_alpha * constrained[r.symbol] + (1 - cfg.ema_alpha) * prev
            else:
                r.final_weight = constrained[r.symbol]

        self._initialized = True

        # Renormalize
        total_w = sum(r.final_weight for r in rankings)
        if total_w > 0:
            for r in rankings:
                r.final_weight /= total_w

        # Update smoothed weights
        for r in rankings:
            self._smoothed_weights[r.symbol] = r.final_weight

        weights = {r.symbol: r.final_weight for r in rankings}

        return PortfolioSnapshot(
            rankings=rankings,
            weights=weights,
            total_exposure_pct=sum(weights.values()) * 100.0,
        )

    def _apply_constraints(
        self, raw_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Apply min/max weight constraints with iterative renormalization."""
        cfg = self._config
        weights = dict(raw_weights)

        for _ in range(10):
            clamped = {}
            for s in self._symbols:
                w = weights.get(s, 0.0)
                clamped[s] = max(cfg.min_symbol_weight, min(cfg.max_symbol_weight, w))

            total = sum(clamped.values())
            if total > 0:
                weights = {s: clamped[s] / total for s in self._symbols}
            else:
                equal = 1.0 / len(self._symbols)
                weights = {s: equal for s in self._symbols}

            all_valid = all(
                cfg.min_symbol_weight - 1e-6 <= weights[s] <= cfg.max_symbol_weight + 1e-6
                for s in self._symbols
            )
            if all_valid:
                break

        return weights

    def get_weights(self) -> Dict[str, float]:
        """Return current portfolio weights."""
        return dict(self._smoothed_weights)
