"""
Darwin v4 — Risk-Aware Fitness Model v4.1.

9-component model. v4.0 → v4.1 rebalance:
  - risk_adjusted_profit boosted 0.20 → 0.35 (profit-dominant)
  - capital_velocity raised 0.10 → 0.12 (reward throughput)
  - tp_shaping raised 0.07 → 0.10 (stronger TP convergence)
  - drawdown_health lowered 0.13 → 0.12 (slightly more risk-tolerant)
  - Remaining components scaled proportionally

COMPONENTS (weights sum to 1.0):
  ┌───────────────────────────┬────────┬────────────────────────────────────────┐
  │ Component                 │ Weight │ What it rewards                        │
  ├───────────────────────────┼────────┼────────────────────────────────────────┤
  │ risk_adjusted_profit      │ 0.35   │ Returns per unit of risk taken         │
  │ sharpe_quality            │ 0.07   │ Risk-adjusted return consistency       │
  │ drawdown_health           │ 0.12   │ Low drawdown (quadratic penalty)       │
  │ consistency               │ 0.08   │ Stable win rate + low trade variance   │
  │ portfolio_harmony         │ 0.08   │ Not worsening systemic risk state      │
  │ diversification_bonus     │ 0.05   │ Low correlation with pool exposure     │
  │ capital_efficiency        │ 0.03   │ PnL per unit of capital allocated      │
  │ capital_velocity          │ 0.12   │ Trade count × win rate (throughput)    │
  │ tp_shaping                │ 0.10   │ Gaussian bonus near optimal TP ~1.8%   │
  └───────────────────────────┴────────┴────────────────────────────────────────┘
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from darwin_agent.interfaces.enums import PortfolioRiskState
from darwin_agent.interfaces.types import PortfolioRiskMetrics, TradeResult


# ── Configuration ────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class FitnessWeights:
    """
    Component weights. Sum MUST equal 1.0.
    Frozen so they can't be mutated post-creation.
    """
    risk_adjusted_profit: float = 0.35
    sharpe_quality: float = 0.07
    drawdown_health: float = 0.12
    consistency: float = 0.08
    portfolio_harmony: float = 0.08
    diversification_bonus: float = 0.05
    capital_efficiency: float = 0.03
    capital_velocity: float = 0.12
    tp_shaping: float = 0.10

    def __post_init__(self):
        total = (
            self.risk_adjusted_profit + self.sharpe_quality +
            self.drawdown_health + self.consistency +
            self.portfolio_harmony + self.diversification_bonus +
            self.capital_efficiency + self.capital_velocity +
            self.tp_shaping
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"FitnessWeights must sum to 1.0, got {total:.6f}")


@dataclass(frozen=True, slots=True)
class FitnessConfig:
    """Tuning knobs for the fitness model."""
    weights: FitnessWeights = field(default_factory=FitnessWeights)

    # risk_adjusted_profit
    profit_cap_multiple: float = 2.0
    profit_floor_multiple: float = -0.5

    # sharpe_quality
    sharpe_excellent: float = 3.0

    # drawdown_health (quadratic)
    drawdown_fatal_pct: float = 40.0

    # consistency
    min_trades_for_consistency: int = 10

    # portfolio_harmony
    state_scores: Dict[PortfolioRiskState, float] = field(default_factory=lambda: {
        PortfolioRiskState.NORMAL: 1.0,
        PortfolioRiskState.DEFENSIVE: 0.65,
        PortfolioRiskState.CRITICAL: 0.30,
        PortfolioRiskState.HALTED: 0.0,
    })

    # capital_efficiency
    roc_excellent: float = 1.0

    # capital_velocity
    velocity_trades_excellent: int = 100
    velocity_winrate_ref: float = 0.55

    # tp_shaping (Gaussian)
    tp_optimal_pct: float = 1.8
    tp_sigma: float = 1.2


# ── Fitness breakdown ────────────────────────────────────────

@dataclass(slots=True)
class FitnessBreakdown:
    """Per-component scores for debugging and dashboard display."""
    risk_adjusted_profit: float = 0.0
    sharpe_quality: float = 0.0
    drawdown_health: float = 0.0
    consistency: float = 0.0
    portfolio_harmony: float = 0.0
    diversification_bonus: float = 0.0
    capital_efficiency: float = 0.0
    capital_velocity: float = 0.0
    tp_shaping: float = 0.0
    final_score: float = 0.0
    portfolio_state: str = "NORMAL"
    penalty_applied: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "risk_adjusted_profit": round(self.risk_adjusted_profit, 4),
            "sharpe_quality": round(self.sharpe_quality, 4),
            "drawdown_health": round(self.drawdown_health, 4),
            "consistency": round(self.consistency, 4),
            "portfolio_harmony": round(self.portfolio_harmony, 4),
            "diversification_bonus": round(self.diversification_bonus, 4),
            "capital_efficiency": round(self.capital_efficiency, 4),
            "capital_velocity": round(self.capital_velocity, 4),
            "tp_shaping": round(self.tp_shaping, 4),
            "final_score": round(self.final_score, 4),
            "portfolio_state": self.portfolio_state,
            "penalty_applied": round(self.penalty_applied, 4),
        }


# ── Numerical utilities ─────────────────────────────────────

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp to [lo, hi]. NaN → lo, ±inf → nearest bound."""
    if math.isnan(x):
        return lo
    if math.isinf(x):
        return hi if x > 0 else lo
    return max(lo, min(hi, x))


def _gaussian(x: float, mu: float, sigma: float) -> float:
    """
    Unnormalized Gaussian: peaks at 1.0 when x == mu,
    decays smoothly. Numerically stable for any input.
    """
    if sigma <= 0:
        return 1.0 if abs(x - mu) < 1e-9 else 0.0
    z = (x - mu) / sigma
    return math.exp(-0.5 * min(z * z, 50.0))


# ── Core fitness model ───────────────────────────────────────

class RiskAwareFitness:
    """
    Stateless fitness calculator. Thread-safe (no mutable state).

    Usage:
        model = RiskAwareFitness()
        score = model.compute(
            realized_pnl=50.0,
            initial_capital=100.0,
            current_capital=150.0,
            sharpe=1.5,
            max_drawdown_pct=8.0,
            win_count=12, loss_count=8,
            total_trades=20,
            pnl_series=[0.5, -0.3, 0.7, ...],
            portfolio_snapshot=risk_engine.get_portfolio_state(),
            agent_exposure={"BTCUSDT": 0.3},
            take_profit_pct=1.8,
        )
    """

    __slots__ = ("_cfg",)

    def __init__(self, config: FitnessConfig | None = None) -> None:
        self._cfg = config or FitnessConfig()

    # ── Main entry point ─────────────────────────────────────

    def compute(
        self,
        *,
        realized_pnl: float,
        initial_capital: float,
        current_capital: float,
        sharpe: float,
        max_drawdown_pct: float,
        win_count: int,
        loss_count: int,
        pnl_series: Sequence[float],
        portfolio_snapshot: PortfolioRiskMetrics | None = None,
        agent_exposure: Dict[str, float] | None = None,
        total_trades: int = 0,
        take_profit_pct: float = 3.0,
    ) -> float:
        """Compute final bounded fitness score in [0, 1]."""
        bd = self.compute_breakdown(
            realized_pnl=realized_pnl,
            initial_capital=initial_capital,
            current_capital=current_capital,
            sharpe=sharpe,
            max_drawdown_pct=max_drawdown_pct,
            win_count=win_count,
            loss_count=loss_count,
            pnl_series=pnl_series,
            portfolio_snapshot=portfolio_snapshot,
            agent_exposure=agent_exposure,
            total_trades=total_trades,
            take_profit_pct=take_profit_pct,
        )
        return bd.final_score

    def compute_breakdown(
        self,
        *,
        realized_pnl: float,
        initial_capital: float,
        current_capital: float,
        sharpe: float,
        max_drawdown_pct: float,
        win_count: int,
        loss_count: int,
        pnl_series: Sequence[float],
        portfolio_snapshot: PortfolioRiskMetrics | None = None,
        agent_exposure: Dict[str, float] | None = None,
        total_trades: int = 0,
        take_profit_pct: float = 3.0,
    ) -> FitnessBreakdown:
        """Full breakdown for debugging / dashboard."""
        w = self._cfg.weights
        trade_count = total_trades or (win_count + loss_count)

        if trade_count == 0:
            return FitnessBreakdown(final_score=0.0)

        cap = max(initial_capital, 0.01)
        win_rate = win_count / trade_count if trade_count > 0 else 0.0

        # ── 1. Risk-adjusted profit ──────────────────────────
        rng = self._cfg.profit_cap_multiple - self._cfg.profit_floor_multiple
        raw_return = realized_pnl / cap
        rap = _clamp((raw_return - self._cfg.profit_floor_multiple) / rng)

        # ── 2. Sharpe quality ────────────────────────────────
        sq = _clamp(max(0, sharpe) / self._cfg.sharpe_excellent)

        # ── 3. Drawdown health (QUADRATIC) ───────────────────
        # (1 - dd/fatal)² — light drawdowns barely hurt, deep
        # drawdowns crushed.
        #   dd=0%→1.00  dd=10%→0.56  dd=20%→0.25  dd=30%→0.06
        linear_dd = _clamp(1.0 - max_drawdown_pct / self._cfg.drawdown_fatal_pct)
        dh = linear_dd * linear_dd

        # ── 4. Consistency (multiplicative) ──────────────────
        cons = self._compute_consistency(win_count, loss_count, pnl_series)

        # ── 5. Portfolio harmony (systemic multiplier) ───────
        ph, state_str, penalty = self._compute_portfolio_harmony(
            realized_pnl, portfolio_snapshot,
        )

        # ── 6. Diversification bonus ─────────────────────────
        div = self._compute_diversification(
            agent_exposure, portfolio_snapshot,
        )

        # ── 7. Capital efficiency ────────────────────────────
        roc = realized_pnl / cap
        ce = _clamp(roc / self._cfg.roc_excellent) if roc > 0 else 0.0

        # ── 8. Capital velocity (NEW) ────────────────────────
        # velocity = saturating_trade_count × win_rate_factor
        #
        # Trade frequency: 0→0.0, 50→0.5, 100→1.0
        tc_norm = _clamp(trade_count / self._cfg.velocity_trades_excellent)
        #
        # Win rate gate: wr<0.35→0.0, wr=0.55→1.0, wr>0.55→up to 1.3
        # An agent that trades 100 times at 30% WR gets velocity=0.
        wr_denom = self._cfg.velocity_winrate_ref - 0.35
        if wr_denom > 0:
            wr_factor = _clamp((win_rate - 0.35) / wr_denom, 0.0, 1.3)
        else:
            wr_factor = 0.0
        cv = _clamp(tc_norm * wr_factor)

        # ── 9. TP shaping (NEW) ─────────────────────────────
        # Gaussian centered at optimal TP%. Smooth preference
        # surface with no hard thresholds.
        #   TP=1.8%→1.00  TP=1.0%→0.68  TP=3.0%→0.58
        #   TP=5.0%→0.07  TP=8.0%→~0.00
        tp_score = _gaussian(
            take_profit_pct,
            self._cfg.tp_optimal_pct,
            self._cfg.tp_sigma,
        )

        # ── Weighted sum ─────────────────────────────────────
        raw_score = (
            w.risk_adjusted_profit * rap +
            w.sharpe_quality * sq +
            w.drawdown_health * dh +
            w.consistency * cons +
            w.portfolio_harmony * ph +
            w.diversification_bonus * div +
            w.capital_efficiency * ce +
            w.capital_velocity * cv +
            w.tp_shaping * tp_score
        )

        final = _clamp(raw_score)

        return FitnessBreakdown(
            risk_adjusted_profit=rap,
            sharpe_quality=sq,
            drawdown_health=dh,
            consistency=cons,
            portfolio_harmony=ph,
            diversification_bonus=div,
            capital_efficiency=ce,
            capital_velocity=cv,
            tp_shaping=tp_score,
            final_score=round(final, 4),
            portfolio_state=state_str,
            penalty_applied=penalty,
        )

    # ── Component implementations ────────────────────────────

    def _compute_consistency(
        self,
        wins: int,
        losses: int,
        pnl_series: Sequence[float],
    ) -> float:
        """
        Rewards agents with stable, predictable returns.

        Two sub-scores blended 50/50:
          a) Win rate quality: logistic curve, 50%→0.5, 65%→~1.0
          b) PnL stability: inverse coefficient of variation
        """
        total = wins + losses
        if total < self._cfg.min_trades_for_consistency:
            return 0.5

        wr = wins / total
        wr_score = _clamp(1.0 / (1.0 + math.exp(-12.0 * (wr - 0.50))))

        if len(pnl_series) >= 5:
            try:
                mean = statistics.mean(pnl_series)
                std = statistics.stdev(pnl_series)
                if abs(mean) > 1e-9:
                    cv = abs(std / mean)
                    stability = _clamp(1.0 / (1.0 + cv))
                else:
                    stability = 0.3
            except (statistics.StatisticsError, ZeroDivisionError):
                stability = 0.3
        else:
            stability = 0.5

        return _clamp(0.5 * wr_score + 0.5 * stability)

    def _compute_portfolio_harmony(
        self,
        agent_pnl: float,
        snapshot: PortfolioRiskMetrics | None,
    ) -> Tuple[float, str, float]:
        """
        Systemic multiplier: scores portfolio integration quality.
        Returns (score, state_name, penalty_amount).
        """
        if snapshot is None:
            return 1.0, "UNKNOWN", 0.0

        state = snapshot.risk_state
        state_str = state.value
        base = self._cfg.state_scores.get(state, 0.5)

        penalty = 0.0
        if state in (PortfolioRiskState.DEFENSIVE, PortfolioRiskState.CRITICAL):
            if agent_pnl < 0 and snapshot.total_equity > 0:
                pool_dd = snapshot.drawdown_pct
                if pool_dd > 0:
                    agent_dd_contribution = abs(agent_pnl) / max(snapshot.total_equity, 1.0)
                    penalty = _clamp(agent_dd_contribution * 10.0, 0.0, 0.3)
        elif state == PortfolioRiskState.HALTED:
            if agent_pnl < 0:
                penalty = 0.3

        score = _clamp(base - penalty)
        return score, state_str, penalty

    def _compute_diversification(
        self,
        agent_exposure: Dict[str, float] | None,
        snapshot: PortfolioRiskMetrics | None,
    ) -> float:
        """
        Rewards agents whose positions diversify the pool.
        """
        if snapshot is None:
            return 0.5

        pool_corr = snapshot.correlation_risk
        corr_score = _clamp(1.0 - pool_corr)

        if agent_exposure is None or not agent_exposure:
            return corr_score

        n_symbols = len(agent_exposure)
        spread_score = _clamp(n_symbols / 3.0)

        pool_exp = snapshot.exposure_by_symbol
        if pool_exp:
            pool_total = sum(abs(v) for v in pool_exp.values())
            if pool_total > 0:
                overlap = 0.0
                for sym, agent_frac in agent_exposure.items():
                    pool_frac = abs(pool_exp.get(sym, 0.0)) / pool_total
                    overlap += abs(agent_frac) * pool_frac
                overlap_score = _clamp(1.0 - overlap)
            else:
                overlap_score = 0.8
        else:
            overlap_score = 0.8

        return _clamp(
            0.35 * corr_score +
            0.30 * spread_score +
            0.35 * overlap_score
        )


# ── Singleton default instance ───────────────────────────────

_DEFAULT_MODEL = RiskAwareFitness()


def compute_fitness(
    *,
    realized_pnl: float,
    initial_capital: float,
    current_capital: float,
    sharpe: float,
    max_drawdown_pct: float,
    win_count: int,
    loss_count: int,
    pnl_series: Sequence[float],
    portfolio_snapshot: PortfolioRiskMetrics | None = None,
    agent_exposure: Dict[str, float] | None = None,
    total_trades: int = 0,
    take_profit_pct: float = 3.0,
) -> float:
    """Convenience function using default config."""
    return _DEFAULT_MODEL.compute(
        realized_pnl=realized_pnl,
        initial_capital=initial_capital,
        current_capital=current_capital,
        sharpe=sharpe,
        max_drawdown_pct=max_drawdown_pct,
        win_count=win_count,
        loss_count=loss_count,
        pnl_series=pnl_series,
        portfolio_snapshot=portfolio_snapshot,
        agent_exposure=agent_exposure,
        total_trades=total_trades,
        take_profit_pct=take_profit_pct,
    )


def compute_fitness_breakdown(
    *,
    realized_pnl: float,
    initial_capital: float,
    current_capital: float,
    sharpe: float,
    max_drawdown_pct: float,
    win_count: int,
    loss_count: int,
    pnl_series: Sequence[float],
    portfolio_snapshot: PortfolioRiskMetrics | None = None,
    agent_exposure: Dict[str, float] | None = None,
    total_trades: int = 0,
    take_profit_pct: float = 3.0,
) -> FitnessBreakdown:
    """Convenience function returning full breakdown."""
    return _DEFAULT_MODEL.compute_breakdown(
        realized_pnl=realized_pnl,
        initial_capital=initial_capital,
        current_capital=current_capital,
        sharpe=sharpe,
        max_drawdown_pct=max_drawdown_pct,
        win_count=win_count,
        loss_count=loss_count,
        pnl_series=pnl_series,
        portfolio_snapshot=portfolio_snapshot,
        agent_exposure=agent_exposure,
        total_trades=total_trades,
        take_profit_pct=take_profit_pct,
    )
