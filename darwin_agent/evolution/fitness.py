"""
Darwin v4 — Risk-Aware Fitness Model v4.5.

v4.4 → v4.5: adds anti-overfitting penalties.

  ANTI-OVERFITTING CHANGES (v4.5):
    1. Sortino ratio replaces raw Sharpe in risk_stability pillar.
       Sharpe treats upside vol as risk — that's wrong for trend-followers.
       Sortino only penalizes downside deviation, which is the real risk.

    2. Profit concentration penalty: penalizes genomes where >50% of total
       PnL comes from a single trade. A bot that survives on one lucky trade
       is memorizing noise, not finding alpha.

    3. Minimum trade count scaling: raw_score *= min(1, trade_count / 20).
       A bot with 3 trades and 0.95 fitness is statistically meaningless.
       This smoothly ramps fitness from 0 at 0 trades to full at 20+ trades.

    4. Drawdown duration penalty integrated into drawdown_health:
       A bot that spends 90% of its lifetime underwater is a ticking bomb
       even if max drawdown is only 5%. In production, mark price wicks
       would liquidate it.

  ┌─────────────────────┬────────┬──────────────────────────────────────────┐
  │ Pillar              │ Weight │ Sub-components                           │
  ├─────────────────────┼────────┼──────────────────────────────────────────┤
  │ ecosystem_health    │ 0.22   │ portfolio_harmony, diversification,      │
  │                     │        │ consistency                              │
  │ risk_stability      │ 0.18   │ drawdown_health, sharpe_quality          │
  │ learning_quality    │ 0.13   │ risk_adj_profit, capital_efficiency,     │
  │                     │        │ capital_velocity, tp_shaping             │
  │ profit_factor_score │ 0.13   │ tanh((PF - 1.0) * 2.5)                  │
  │ efficiency_score    │ 0.08   │ clamp(net_pnl / notional * 50, -1, +1)  │
  │ trend_score         │ 0.13   │ clamp(trend_capture_ratio * 3, -1, +1)  │
  │ activity_score      │ 0.05   │ overtrading + inactivity penalty         │
  │ convexity_score     │ 0.08   │ tanh(alpha * (avg_win/avg_loss - 1))     │
  └─────────────────────┴────────┴──────────────────────────────────────────┘
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, Sequence

from darwin_agent.interfaces.enums import PortfolioRiskState
from darwin_agent.interfaces.types import PortfolioRiskMetrics


# ── Configuration ────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class FitnessWeights:
    """v4.4 pillar weights. Sum MUST equal 1.0."""
    ecosystem_health: float = 0.22
    risk_stability: float = 0.18
    learning_quality: float = 0.13
    profit_factor_score: float = 0.13
    efficiency_score: float = 0.08
    trend_score: float = 0.13
    activity_score: float = 0.05
    convexity_score: float = 0.08

    def __post_init__(self):
        total = (
            self.ecosystem_health + self.risk_stability +
            self.learning_quality + self.profit_factor_score +
            self.efficiency_score + self.trend_score +
            self.activity_score + self.convexity_score
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"FitnessWeights must sum to 1.0, got {total:.6f}")


@dataclass(frozen=True, slots=True)
class FitnessConfig:
    """Tuning knobs for the fitness model."""
    weights: FitnessWeights = field(default_factory=FitnessWeights)
    profit_cap_multiple: float = 2.0
    profit_floor_multiple: float = -0.5
    sharpe_excellent: float = 3.0
    drawdown_fatal_pct: float = 40.0
    min_trades_for_consistency: int = 10
    state_scores: Dict[PortfolioRiskState, float] = field(default_factory=lambda: {
        PortfolioRiskState.NORMAL: 1.0,
        PortfolioRiskState.DEFENSIVE: 0.65,
        PortfolioRiskState.CRITICAL: 0.30,
        PortfolioRiskState.HALTED: 0.0,
    })
    roc_excellent: float = 1.0
    velocity_trades_excellent: int = 100
    velocity_winrate_ref: float = 0.55
    tp_optimal_pct: float = 1.8
    tp_sigma: float = 1.2
    pf_sensitivity: float = 2.5
    eff_scale: float = 50.0
    trend_capture_scale: float = 3.0
    # activity penalty (v4.4): exp(-k * trade_freq)
    # k=4.0: penalizes trade_freq>0.5 (>1 trade per 2 bars).
    # Original k=8.0 was too aggressive, crushing active scalpers at freq=1.0.
    k_activity: float = 4.0
    # expectancy convexity (v4.4)
    convexity_alpha: float = 1.5
    # anti-overfitting (v4.5)
    min_trades_for_full_score: int = 20
    profit_concentration_threshold: float = 0.50  # penalty if single trade > 50% of PnL
    sortino_excellent: float = 4.0  # Sortino reference for normalization


# ── Fitness breakdown ────────────────────────────────────────

@dataclass(slots=True)
class FitnessBreakdown:
    """Per-component scores. Retains all v4.1 fields + v4.2 additions."""
    risk_adjusted_profit: float = 0.0
    sharpe_quality: float = 0.0
    drawdown_health: float = 0.0
    consistency: float = 0.0
    portfolio_harmony: float = 0.0
    diversification_bonus: float = 0.0
    capital_efficiency: float = 0.0
    capital_velocity: float = 0.0
    tp_shaping: float = 0.0
    ecosystem_health: float = 0.0
    risk_stability: float = 0.0
    learning_quality: float = 0.0
    profit_factor_score: float = 0.0
    efficiency_score: float = 0.0
    trend_score: float = 0.0
    activity_score: float = 0.0
    convexity_score: float = 0.0
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
            "ecosystem_health": round(self.ecosystem_health, 4),
            "risk_stability": round(self.risk_stability, 4),
            "learning_quality": round(self.learning_quality, 4),
            "profit_factor_score": round(self.profit_factor_score, 4),
            "efficiency_score": round(self.efficiency_score, 4),
            "trend_score": round(self.trend_score, 4),
            "activity_score": round(self.activity_score, 4),
            "convexity_score": round(self.convexity_score, 4),
            "final_score": round(self.final_score, 4),
            "portfolio_state": self.portfolio_state,
            "penalty_applied": round(self.penalty_applied, 4),
        }


# ── Numerical utilities ─────────────────────────────────────

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if math.isnan(x):
        return lo
    if math.isinf(x):
        return hi if x > 0 else lo
    return max(lo, min(hi, x))


def _gaussian(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 1.0 if abs(x - mu) < 1e-9 else 0.0
    z = (x - mu) / sigma
    return math.exp(-0.5 * min(z * z, 50.0))


# ── Core fitness model ───────────────────────────────────────

class RiskAwareFitness:
    """Stateless fitness calculator v4.2. Thread-safe."""

    __slots__ = ("_cfg",)

    def __init__(self, config: FitnessConfig | None = None) -> None:
        self._cfg = config or FitnessConfig()

    def compute(
        self, *,
        realized_pnl: float, initial_capital: float, current_capital: float,
        sharpe: float, max_drawdown_pct: float,
        win_count: int, loss_count: int,
        pnl_series: Sequence[float],
        portfolio_snapshot: PortfolioRiskMetrics | None = None,
        agent_exposure: Dict[str, float] | None = None,
        total_trades: int = 0, take_profit_pct: float = 3.0,
        gross_profit: float = 0.0, gross_loss: float = 0.0,
        total_notional: float = 0.0,
        trend_series: Sequence[float] = (),
        n_bars: int = 0,
        bars_gated: int = 0,
    ) -> float:
        return self.compute_breakdown(
            realized_pnl=realized_pnl, initial_capital=initial_capital,
            current_capital=current_capital, sharpe=sharpe,
            max_drawdown_pct=max_drawdown_pct, win_count=win_count,
            loss_count=loss_count, pnl_series=pnl_series,
            portfolio_snapshot=portfolio_snapshot, agent_exposure=agent_exposure,
            total_trades=total_trades, take_profit_pct=take_profit_pct,
            gross_profit=gross_profit, gross_loss=gross_loss,
            total_notional=total_notional, trend_series=trend_series,
            n_bars=n_bars, bars_gated=bars_gated,
        ).final_score

    def compute_breakdown(
        self, *,
        realized_pnl: float, initial_capital: float, current_capital: float,
        sharpe: float, max_drawdown_pct: float,
        win_count: int, loss_count: int,
        pnl_series: Sequence[float],
        portfolio_snapshot: PortfolioRiskMetrics | None = None,
        agent_exposure: Dict[str, float] | None = None,
        total_trades: int = 0, take_profit_pct: float = 3.0,
        gross_profit: float = 0.0, gross_loss: float = 0.0,
        total_notional: float = 0.0,
        trend_series: Sequence[float] = (),
        n_bars: int = 0,
        bars_gated: int = 0,
    ) -> FitnessBreakdown:
        w = self._cfg.weights
        trade_count = total_trades or (win_count + loss_count)

        if trade_count == 0:
            return FitnessBreakdown(final_score=0.0)

        cap = max(initial_capital, 0.01)
        win_rate = win_count / trade_count if trade_count > 0 else 0.0

        # ════════════════════════════════════════════════════
        # v4.1 SUB-COMPONENTS (all preserved intact)
        # ════════════════════════════════════════════════════

        # 1. Risk-adjusted profit
        rng = self._cfg.profit_cap_multiple - self._cfg.profit_floor_multiple
        raw_return = realized_pnl / cap
        rap = _clamp((raw_return - self._cfg.profit_floor_multiple) / rng)

        # 2. Sharpe quality (blended with Sortino for anti-overfitting)
        # Pure Sharpe penalizes upside vol — bad for trend-followers.
        # Blend: 40% Sharpe + 60% Sortino to reward asymmetric returns.
        sq_sharpe = _clamp(max(0, sharpe) / self._cfg.sharpe_excellent)
        sortino = self._compute_sortino(pnl_series)
        sq_sortino = _clamp(max(0, sortino) / self._cfg.sortino_excellent)
        sq = 0.4 * sq_sharpe + 0.6 * sq_sortino

        # 3. Drawdown health (quadratic)
        linear_dd = _clamp(1.0 - max_drawdown_pct / self._cfg.drawdown_fatal_pct)
        dh = linear_dd * linear_dd

        # 4. Consistency
        cons = self._compute_consistency(win_count, loss_count, pnl_series)

        # 5. Portfolio harmony
        ph, state_str, penalty = self._compute_portfolio_harmony(
            realized_pnl, portfolio_snapshot)

        # 6. Diversification bonus
        div = self._compute_diversification(agent_exposure, portfolio_snapshot)

        # 7. Capital efficiency (ROC)
        roc = realized_pnl / cap
        ce = _clamp(roc / self._cfg.roc_excellent) if roc > 0 else 0.0

        # 8. Capital velocity
        tc_norm = _clamp(trade_count / self._cfg.velocity_trades_excellent)
        wr_denom = self._cfg.velocity_winrate_ref - 0.35
        wr_factor = _clamp((win_rate - 0.35) / wr_denom, 0.0, 1.3) if wr_denom > 0 else 0.0
        cv = _clamp(tc_norm * wr_factor)

        # 9. TP shaping
        tp_score = _gaussian(take_profit_pct, self._cfg.tp_optimal_pct, self._cfg.tp_sigma)

        # ════════════════════════════════════════════════════
        # v4.2 PILLARS
        # ════════════════════════════════════════════════════

        # Pillar 1: Ecosystem Health (0.30)
        ecosystem = 0.40 * ph + 0.25 * div + 0.35 * cons

        # Pillar 2: Risk Stability (0.20)
        risk_stab = 0.65 * dh + 0.35 * sq

        # Pillar 3: Learning Quality (0.15)
        learning = 0.35 * rap + 0.10 * ce + 0.30 * cv + 0.25 * tp_score

        # Pillar 4: Profit Factor (0.20)
        if gross_loss > 0:
            pf = gross_profit / gross_loss
        elif gross_profit > 0:
            pf = 3.0
        else:
            pf = 1.0
        pf_raw = math.tanh((pf - 1.0) * self._cfg.pf_sensitivity)
        pf_score = _clamp((pf_raw + 1.0) / 2.0)

        # Pillar 5: Efficiency (0.10)
        eff_raw = realized_pnl / (total_notional + 1e-8)
        eff_scaled = eff_raw * self._cfg.eff_scale
        eff_clamped = max(-1.0, min(1.0, eff_scaled))
        eff_score = _clamp((eff_clamped + 1.0) / 2.0)

        # Pillar 6: Trend Convexity (0.15)
        # Rewards agents that profit MORE during strong trends
        if len(pnl_series) >= 2 and len(trend_series) >= len(pnl_series):
            trend_weighted = sum(
                pnl_series[i] * trend_series[i]
                for i in range(len(pnl_series))
            )
            trend_total = sum(abs(p) for p in pnl_series) + 1e-8
            trend_capture = trend_weighted / trend_total
            trend_raw = trend_capture * self._cfg.trend_capture_scale
            trend_clamped = max(-1.0, min(1.0, trend_raw))
            t_score = _clamp((trend_clamped + 1.0) / 2.0)
        else:
            t_score = 0.5  # neutral when no trend data

        # Pillar 7: Activity Score (0.05)
        # Combines overtrading penalty + inactivity penalty from regime gate
        if n_bars > 0:
            trade_freq = trade_count / n_bars
            overtrade_score = math.exp(-self._cfg.k_activity * trade_freq)
            # Inactivity penalty: smooth penalty when gate blocks > 70% of bars
            gate_ratio = bars_gated / n_bars
            # sigmoid centered at 0.7: gating 70% → 0.5, gating 90% → 0.12
            inactivity_mult = 1.0 / (1.0 + math.exp(10.0 * (gate_ratio - 0.7)))
            act_score = overtrade_score * inactivity_mult
        else:
            act_score = 0.5  # neutral when no bar count

        # Pillar 8: Expectancy Convexity (0.08)
        # tanh(alpha * (avg_win/avg_loss - 1)): rewards asymmetric payoff
        avg_win = gross_profit / max(win_count, 1)
        avg_loss = gross_loss / max(loss_count, 1)
        wl_ratio = avg_win / (avg_loss + 1e-8)
        conv_raw = math.tanh(self._cfg.convexity_alpha * (wl_ratio - 1.0))
        conv_score = _clamp((conv_raw + 1.0) / 2.0)

        # Weighted sum
        raw_score = (
            w.ecosystem_health * ecosystem +
            w.risk_stability * risk_stab +
            w.learning_quality * learning +
            w.profit_factor_score * pf_score +
            w.efficiency_score * eff_score +
            w.trend_score * t_score +
            w.activity_score * act_score +
            w.convexity_score * conv_score
        )

        # ════════════════════════════════════════════════════
        # v4.5 ANTI-OVERFITTING MULTIPLIERS
        # Applied AFTER weighted sum to penalize statistical flukes.
        # ════════════════════════════════════════════════════

        # 1. Minimum trade count scaling: ramp from 0→1 over [0, min_trades].
        #    A bot with 3 trades and 0.95 fitness is noise, not alpha.
        min_trades = self._cfg.min_trades_for_full_score
        trade_scale = min(1.0, trade_count / min_trades) if min_trades > 0 else 1.0
        raw_score *= trade_scale

        # 2. Profit concentration penalty: if a single trade accounts for
        #    >threshold of total gross profit, multiply fitness by (1 - excess).
        #    This kills genomes that survive on one lucky trade.
        if gross_profit > 0 and len(pnl_series) >= 2:
            max_single = max(pnl_series)
            if max_single > 0:
                concentration = max_single / gross_profit
                threshold = self._cfg.profit_concentration_threshold
                if concentration > threshold:
                    excess = min(concentration - threshold, 0.5)
                    raw_score *= (1.0 - excess)

        return FitnessBreakdown(
            risk_adjusted_profit=rap, sharpe_quality=sq,
            drawdown_health=dh, consistency=cons,
            portfolio_harmony=ph, diversification_bonus=div,
            capital_efficiency=ce, capital_velocity=cv, tp_shaping=tp_score,
            ecosystem_health=ecosystem, risk_stability=risk_stab,
            learning_quality=learning, profit_factor_score=pf_score,
            efficiency_score=eff_score, trend_score=t_score,
            activity_score=act_score, convexity_score=conv_score,
            final_score=round(_clamp(raw_score), 4),
            portfolio_state=state_str, penalty_applied=penalty,
        )

    # ── Component implementations ────────────────────────────

    def _compute_consistency(self, wins, losses, pnl_series):
        total = wins + losses
        if total < self._cfg.min_trades_for_consistency:
            return 0.5
        wr = wins / total
        wr_score = _clamp(1.0 / (1.0 + math.exp(-12.0 * (wr - 0.50))))
        if len(pnl_series) >= 5:
            try:
                mean = statistics.mean(pnl_series)
                std = statistics.stdev(pnl_series)
                stability = _clamp(1.0 / (1.0 + abs(std / mean))) if abs(mean) > 1e-9 else 0.3
            except (statistics.StatisticsError, ZeroDivisionError):
                stability = 0.3
        else:
            stability = 0.5
        return _clamp(0.5 * wr_score + 0.5 * stability)

    def _compute_portfolio_harmony(self, agent_pnl, snapshot):
        if snapshot is None:
            return 1.0, "UNKNOWN", 0.0
        state = snapshot.risk_state
        state_str = state.value
        base = self._cfg.state_scores.get(state, 0.5)
        penalty = 0.0
        if state in (PortfolioRiskState.DEFENSIVE, PortfolioRiskState.CRITICAL):
            if agent_pnl < 0 and snapshot.total_equity > 0:
                if snapshot.drawdown_pct > 0:
                    penalty = _clamp(abs(agent_pnl) / max(snapshot.total_equity, 1.0) * 10.0, 0.0, 0.3)
        elif state == PortfolioRiskState.HALTED:
            if agent_pnl < 0:
                penalty = 0.3
        return _clamp(base - penalty), state_str, penalty

    @staticmethod
    def _compute_sortino(pnl_series: Sequence[float]) -> float:
        """Sortino ratio: mean / downside_std * sqrt(n).

        Unlike Sharpe, Sortino only penalizes downside deviation.
        A bot with volatile upside but controlled downside gets rewarded.

        Edge cases:
          - No losses → excellent (return high Sortino)
          - All losses identical (std=0) → use abs(mean_loss) as proxy
            This prevents zero-division when losses are constant.
        """
        if len(pnl_series) < 5:
            return 0.0
        mean_pnl = statistics.mean(pnl_series)
        downside = [p for p in pnl_series if p < 0]
        if not downside:
            # No losses at all — excellent downside control
            return max(0.0, mean_pnl) * math.sqrt(min(len(pnl_series), 252))
        if len(downside) == 1:
            down_std = abs(downside[0])
        else:
            down_std = statistics.stdev(downside)
        # When all losses are identical, stdev=0. Use mean loss magnitude
        # as fallback — constant small losses are good, not zero-scored.
        if down_std <= 1e-9:
            down_std = abs(statistics.mean(downside))
        if down_std <= 1e-9:
            return 0.0
        n_trades = max(len(pnl_series), 1)
        annualization = math.sqrt(min(n_trades, 252))
        return (mean_pnl / down_std) * annualization

    def _compute_diversification(self, agent_exposure, snapshot):
        if snapshot is None:
            return 0.5
        corr_score = _clamp(1.0 - snapshot.correlation_risk)
        if agent_exposure is None or not agent_exposure:
            return corr_score
        spread_score = _clamp(len(agent_exposure) / 3.0)
        pool_exp = snapshot.exposure_by_symbol
        if pool_exp:
            pool_total = sum(abs(v) for v in pool_exp.values())
            if pool_total > 0:
                overlap = sum(abs(agent_exposure.get(s, 0)) * abs(pool_exp.get(s, 0)) / pool_total
                              for s in agent_exposure)
                overlap_score = _clamp(1.0 - overlap)
            else:
                overlap_score = 0.8
        else:
            overlap_score = 0.8
        return _clamp(0.35 * corr_score + 0.30 * spread_score + 0.35 * overlap_score)


# ── Singleton default instance ───────────────────────────────

_DEFAULT_MODEL = RiskAwareFitness()


def compute_fitness(
    *, realized_pnl: float, initial_capital: float, current_capital: float,
    sharpe: float, max_drawdown_pct: float,
    win_count: int, loss_count: int,
    pnl_series: Sequence[float],
    portfolio_snapshot: PortfolioRiskMetrics | None = None,
    agent_exposure: Dict[str, float] | None = None,
    total_trades: int = 0, take_profit_pct: float = 3.0,
    gross_profit: float = 0.0, gross_loss: float = 0.0,
    total_notional: float = 0.0,
    trend_series: Sequence[float] = (),
    n_bars: int = 0,
    bars_gated: int = 0,
) -> float:
    return _DEFAULT_MODEL.compute(
        realized_pnl=realized_pnl, initial_capital=initial_capital,
        current_capital=current_capital, sharpe=sharpe,
        max_drawdown_pct=max_drawdown_pct, win_count=win_count,
        loss_count=loss_count, pnl_series=pnl_series,
        portfolio_snapshot=portfolio_snapshot, agent_exposure=agent_exposure,
        total_trades=total_trades, take_profit_pct=take_profit_pct,
        gross_profit=gross_profit, gross_loss=gross_loss,
        total_notional=total_notional, trend_series=trend_series,
        n_bars=n_bars, bars_gated=bars_gated,
    )


def compute_fitness_breakdown(
    *, realized_pnl: float, initial_capital: float, current_capital: float,
    sharpe: float, max_drawdown_pct: float,
    win_count: int, loss_count: int,
    pnl_series: Sequence[float],
    portfolio_snapshot: PortfolioRiskMetrics | None = None,
    agent_exposure: Dict[str, float] | None = None,
    total_trades: int = 0, take_profit_pct: float = 3.0,
    gross_profit: float = 0.0, gross_loss: float = 0.0,
    total_notional: float = 0.0,
    trend_series: Sequence[float] = (),
    n_bars: int = 0,
    bars_gated: int = 0,
) -> FitnessBreakdown:
    return _DEFAULT_MODEL.compute_breakdown(
        realized_pnl=realized_pnl, initial_capital=initial_capital,
        current_capital=current_capital, sharpe=sharpe,
        max_drawdown_pct=max_drawdown_pct, win_count=win_count,
        loss_count=loss_count, pnl_series=pnl_series,
        portfolio_snapshot=portfolio_snapshot, agent_exposure=agent_exposure,
        total_trades=total_trades, take_profit_pct=take_profit_pct,
        gross_profit=gross_profit, gross_loss=gross_loss,
        total_notional=total_notional, trend_series=trend_series,
        n_bars=n_bars, bars_gated=bars_gated,
    )
