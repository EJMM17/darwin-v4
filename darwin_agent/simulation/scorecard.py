"""
Darwin v4 — SimulationScorecard.

Pure analytical evaluation layer.  ZERO trading logic.
Reads diagnostic, risk, and evolution outputs to produce
a structured health report with 5 normalized sub-scores.

SCORES (each 0–10):

  1. RISK STABILITY SCORE
     How well the portfolio maintained controlled risk throughout
     the simulation.  Penalizes high drawdowns, time spent in
     CRITICAL/HALTED states, excessive consecutive losses, and
     high drawdown variance across generations.

  2. EVOLUTION HEALTH SCORE
     How well the evolutionary algorithm is functioning.  Rewards
     genetic diversity, balanced lineages, fitness improvement
     trends, and penalizes stagnation and genetic collapse.

  3. CONCENTRATION RISK SCORE
     How well-distributed capital and exposure are.  Penalizes
     Gini inequality, single-agent dominance, single-symbol
     exposure concentration, and high HHI.

  4. SHOCK RESILIENCE SCORE
     How well the system recovered from adverse events.  Measures
     drawdown recovery speed, capital preservation ratio, worst-
     case generation PnL, and post-shock fitness recovery.

  5. LEARNING QUALITY SCORE
     Whether evolution is producing genuinely better agents over
     time.  Measures fitness trend slope, win-rate improvement,
     Sharpe improvement, and trades-per-generation growth (agents
     learning to actually trade rather than sitting idle).

ECOSYSTEM HEALTH:
     Weighted combination of all 5 scores → 0–10 final grade.
     Default weights:  Risk 0.25, Evolution 0.20, Concentration 0.15,
                       Shock 0.20, Learning 0.20

DESIGN:
     - Every formula is bounded, NaN-safe, and inf-safe
     - Every sub-score clips to [0, 10] after computation
     - All inputs are read-only; no mutation of source data
     - Pure functions — no internal state, fully testable
     - to_dict() produces JSON-serializable report
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


# ═════════════════════════════════════════════════════════════
# Output dataclasses
# ═════════════════════════════════════════════════════════════

@dataclass(slots=True)
class SubScore:
    """One of the 5 sub-scores with breakdown."""
    name: str = ""
    score: float = 0.0            # [0, 10]
    grade: str = ""               # A/B/C/D/F
    components: Dict[str, float] = field(default_factory=dict)
    commentary: str = ""


@dataclass(slots=True)
class ScorecardReport:
    """Complete scorecard output."""
    # Sub-scores
    risk_stability: SubScore = field(default_factory=SubScore)
    evolution_health: SubScore = field(default_factory=SubScore)
    concentration_risk: SubScore = field(default_factory=SubScore)
    shock_resilience: SubScore = field(default_factory=SubScore)
    learning_quality: SubScore = field(default_factory=SubScore)

    # Ecosystem aggregate
    ecosystem_health: float = 0.0   # [0, 10]
    ecosystem_grade: str = ""
    weights_used: Dict[str, float] = field(default_factory=dict)

    # Context
    generations_evaluated: int = 0
    starting_capital: float = 0.0
    final_capital: float = 0.0
    capital_return_pct: float = 0.0
    total_alerts: int = 0
    critical_alerts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        def _sub(s: SubScore) -> Dict:
            return {
                "score": round(s.score, 2),
                "grade": s.grade,
                "components": {k: round(v, 4) for k, v in s.components.items()},
                "commentary": s.commentary,
            }
        return {
            "ecosystem_health": round(self.ecosystem_health, 2),
            "ecosystem_grade": self.ecosystem_grade,
            "weights_used": {k: round(v, 2) for k, v in self.weights_used.items()},
            "scores": {
                "risk_stability": _sub(self.risk_stability),
                "evolution_health": _sub(self.evolution_health),
                "concentration_risk": _sub(self.concentration_risk),
                "shock_resilience": _sub(self.shock_resilience),
                "learning_quality": _sub(self.learning_quality),
            },
            "context": {
                "generations_evaluated": self.generations_evaluated,
                "starting_capital": round(self.starting_capital, 2),
                "final_capital": round(self.final_capital, 2),
                "capital_return_pct": round(self.capital_return_pct, 2),
                "total_alerts": self.total_alerts,
                "critical_alerts": self.critical_alerts,
            },
        }


# ═════════════════════════════════════════════════════════════
# Input types (loose dicts from SimulationResults)
# ═════════════════════════════════════════════════════════════

@dataclass
class GenerationData:
    """
    Flattened view of one generation's data for scorecard input.
    Constructed from SimulationResults.generation_results entries.
    All fields are primitive — no dependency on Darwin internals.
    """
    generation: int = 0
    pool_capital: float = 0.0
    pool_pnl: float = 0.0
    portfolio_state: str = "normal"

    # Snapshot fields
    best_fitness: float = 0.0
    avg_fitness: float = 0.0
    worst_fitness: float = 0.0
    total_trades: int = 0
    pool_win_rate: float = 0.0
    pool_sharpe: float = 0.0
    pool_max_drawdown: float = 0.0
    survivors: int = 0
    eliminated: int = 0
    population_size: int = 0

    # Diagnostics fields
    diversity_score: float = 0.0
    mean_entropy: float = 0.0
    capital_gini: float = 0.0
    capital_herfindahl: float = 0.0
    capital_top1_pct: float = 0.0
    exposure_herfindahl: float = 0.0
    exposure_top1_pct: float = 0.0
    dominant_lineage_share: float = 0.0
    effective_lineages: float = 0.0
    fitness_improving: bool = False
    fitness_stagnating: bool = False
    stagnation_generations: int = 0
    health_score: float = 0.0
    n_alerts: int = 0
    alerts: List[str] = field(default_factory=list)


def extract_generation_data(gen_result: Dict[str, Any]) -> GenerationData:
    """
    Build GenerationData from a SimulationResults.generation_results entry.
    Tolerant of missing keys — defaults to 0/empty.
    """
    snap = gen_result.get("snapshot", {})
    diag = gen_result.get("diagnostics", {})
    div = diag.get("diversity", {})
    conc = diag.get("concentration", {})
    dom = diag.get("dominance", {})
    fit = diag.get("fitness", {})

    return GenerationData(
        generation=gen_result.get("generation", 0),
        pool_capital=gen_result.get("pool_capital", 0.0),
        pool_pnl=gen_result.get("pool_pnl", 0.0),
        portfolio_state=gen_result.get("portfolio_state", "normal"),
        best_fitness=snap.get("best_fitness", 0.0),
        avg_fitness=snap.get("avg_fitness", 0.0),
        worst_fitness=snap.get("worst_fitness", 0.0),
        total_trades=snap.get("total_trades", 0),
        pool_win_rate=snap.get("pool_win_rate", 0.0),
        pool_sharpe=snap.get("pool_sharpe", 0.0),
        pool_max_drawdown=snap.get("pool_max_drawdown", 0.0),
        survivors=snap.get("survivors", 0),
        eliminated=snap.get("eliminated", 0),
        population_size=snap.get("population", 0),
        diversity_score=div.get("overall_diversity_score", 0.0),
        mean_entropy=div.get("mean_entropy", 0.0),
        capital_gini=conc.get("capital_gini", 0.0),
        capital_herfindahl=conc.get("capital_herfindahl", 0.0),
        capital_top1_pct=conc.get("capital_top1_pct", 0.0),
        exposure_herfindahl=conc.get("exposure_herfindahl", 0.0),
        exposure_top1_pct=conc.get("exposure_top1_pct", 0.0),
        dominant_lineage_share=dom.get("dominant_lineage_share", 0.0),
        effective_lineages=dom.get("effective_lineages", 0.0),
        fitness_improving=fit.get("improving", False),
        fitness_stagnating=fit.get("stagnating", False),
        stagnation_generations=fit.get("stagnation_generations", 0),
        health_score=diag.get("health_score", 0.0),
        n_alerts=len(diag.get("alerts", [])),
        alerts=diag.get("alerts", []),
    )


# ═════════════════════════════════════════════════════════════
# Scorecard weights
# ═════════════════════════════════════════════════════════════

@dataclass(slots=True)
class ScorecardWeights:
    """Weights for final ecosystem health. Must sum to 1.0."""
    risk_stability: float = 0.25
    evolution_health: float = 0.20
    concentration_risk: float = 0.15
    shock_resilience: float = 0.20
    learning_quality: float = 0.20

    def validate(self) -> None:
        total = (self.risk_stability + self.evolution_health +
                 self.concentration_risk + self.shock_resilience +
                 self.learning_quality)
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Scorecard weights must sum to 1.0, got {total:.3f}")

    def to_dict(self) -> Dict[str, float]:
        return {
            "risk_stability": self.risk_stability,
            "evolution_health": self.evolution_health,
            "concentration_risk": self.concentration_risk,
            "shock_resilience": self.shock_resilience,
            "learning_quality": self.learning_quality,
        }


# ═════════════════════════════════════════════════════════════
# SimulationScorecard
# ═════════════════════════════════════════════════════════════

class SimulationScorecard:
    """
    Pure analytical evaluation layer.
    No internal state. No trading logic.
    All methods are deterministic pure functions.
    """

    __slots__ = ("_weights",)

    def __init__(self, weights: ScorecardWeights | None = None) -> None:
        self._weights = weights or ScorecardWeights()
        self._weights.validate()

    def evaluate(
        self,
        generations: Sequence[GenerationData],
        starting_capital: float = 0.0,
        final_capital: float = 0.0,
    ) -> ScorecardReport:
        """
        Compute the full scorecard from generation-level data.

        Args:
            generations:     Sequence of GenerationData, one per generation.
            starting_capital: Pool starting capital.
            final_capital:    Pool final capital.

        Returns:
            ScorecardReport with 5 sub-scores + ecosystem health.
        """
        if not generations:
            return ScorecardReport(
                ecosystem_grade="F",
                weights_used=self._weights.to_dict(),
            )

        risk = self._score_risk_stability(generations, starting_capital)
        evol = self._score_evolution_health(generations)
        conc = self._score_concentration_risk(generations)
        shock = self._score_shock_resilience(
            generations, starting_capital, final_capital)
        learn = self._score_learning_quality(generations)

        # Weighted ecosystem health
        w = self._weights
        ecosystem = (
            w.risk_stability * risk.score +
            w.evolution_health * evol.score +
            w.concentration_risk * conc.score +
            w.shock_resilience * shock.score +
            w.learning_quality * learn.score
        )
        ecosystem = _clip10(ecosystem)

        # Collect alerts
        all_alerts = []
        for g in generations:
            all_alerts.extend(g.alerts)
        critical = [a for a in all_alerts
                    if any(kw in a for kw in
                           ("COLLAPSE", "CONCENTRATION", "DOMINANCE",
                            "STAGNATION", "HALTED"))]

        ret_pct = 0.0
        if starting_capital > 0:
            ret_pct = ((final_capital - starting_capital) /
                       starting_capital) * 100

        return ScorecardReport(
            risk_stability=risk,
            evolution_health=evol,
            concentration_risk=conc,
            shock_resilience=shock,
            learning_quality=learn,
            ecosystem_health=ecosystem,
            ecosystem_grade=_grade(ecosystem),
            weights_used=self._weights.to_dict(),
            generations_evaluated=len(generations),
            starting_capital=starting_capital,
            final_capital=final_capital,
            capital_return_pct=ret_pct,
            total_alerts=len(all_alerts),
            critical_alerts=critical[:20],
        )

    # ════════════════════════════════════════════════════════
    # 1. Risk Stability Score (0–10)
    # ════════════════════════════════════════════════════════

    def _score_risk_stability(
        self,
        gens: Sequence[GenerationData],
        starting_capital: float,
    ) -> SubScore:
        """
        Components:
          drawdown_control:  Penalize max drawdown across all gens.
                             0% dd → 10,  25%+ dd → 0
          state_stability:   Fraction of gens in NORMAL state.
                             100% normal → 10,  0% → 0
          loss_control:      Inverse of worst generation loss vs capital.
                             No loss → 10,  loss > 20% capital → 0
          dd_consistency:    Low variance in drawdown across gens → stable.
                             Std(dd) < 1% → 10,  > 10% → 0
        """
        # Max drawdown across all generations
        max_dd = max((g.pool_max_drawdown for g in gens), default=0.0)
        drawdown_control = _clip10(10.0 * (1.0 - min(max_dd / 25.0, 1.0)))

        # State stability: fraction of gens in "normal"
        n = len(gens)
        normal_count = sum(1 for g in gens if g.portfolio_state == "normal")
        state_stability = _clip10(10.0 * normal_count / max(n, 1))

        # Worst single-gen PnL loss relative to starting capital
        cap = max(starting_capital, 1.0)
        worst_pnl = min((g.pool_pnl for g in gens), default=0.0)
        loss_ratio = abs(min(worst_pnl, 0.0)) / cap
        loss_control = _clip10(10.0 * (1.0 - min(loss_ratio / 0.20, 1.0)))

        # Drawdown consistency (low std = good)
        dds = [g.pool_max_drawdown for g in gens]
        dd_std = _safe_std(dds)
        dd_consistency = _clip10(10.0 * (1.0 - min(dd_std / 10.0, 1.0)))

        # Weighted blend
        score = _clip10(
            0.35 * drawdown_control +
            0.25 * state_stability +
            0.20 * loss_control +
            0.20 * dd_consistency
        )

        return SubScore(
            name="Risk Stability",
            score=score,
            grade=_grade(score),
            components={
                "drawdown_control": drawdown_control,
                "state_stability": state_stability,
                "loss_control": loss_control,
                "dd_consistency": dd_consistency,
            },
            commentary=_risk_commentary(score, max_dd, normal_count, n),
        )

    # ════════════════════════════════════════════════════════
    # 2. Evolution Health Score (0–10)
    # ════════════════════════════════════════════════════════

    def _score_evolution_health(
        self, gens: Sequence[GenerationData],
    ) -> SubScore:
        """
        Components:
          diversity:       Mean diversity score across gens.
                           1.0 → 10, 0.0 → 0
          lineage_balance: Mean effective lineages / population.
                           1.0 (all unique) → 10, 0 → 0
          no_stagnation:   Fraction of gens NOT stagnating.
                           100% → 10, 0% → 0
          entropy:         Mean gene entropy (normalized).
                           1.0 → 10, 0 → 0
        """
        n = max(len(gens), 1)

        mean_div = _safe_mean([g.diversity_score for g in gens])
        diversity = _clip10(10.0 * mean_div)

        # Effective lineages normalized by population
        lineage_ratios = []
        for g in gens:
            pop = max(g.population_size, 1)
            ratio = min(g.effective_lineages / pop, 1.0)
            lineage_ratios.append(ratio)
        lineage_balance = _clip10(10.0 * _safe_mean(lineage_ratios))

        # Not stagnating
        non_stag = sum(1 for g in gens if not g.fitness_stagnating)
        no_stagnation = _clip10(10.0 * non_stag / n)

        # Gene entropy
        mean_ent = _safe_mean([g.mean_entropy for g in gens])
        entropy = _clip10(10.0 * mean_ent)

        score = _clip10(
            0.30 * diversity +
            0.25 * lineage_balance +
            0.25 * no_stagnation +
            0.20 * entropy
        )

        return SubScore(
            name="Evolution Health",
            score=score,
            grade=_grade(score),
            components={
                "diversity": diversity,
                "lineage_balance": lineage_balance,
                "no_stagnation": no_stagnation,
                "entropy": entropy,
            },
            commentary=_evolution_commentary(score, mean_div, non_stag, n),
        )

    # ════════════════════════════════════════════════════════
    # 3. Concentration Risk Score (0–10)
    # ════════════════════════════════════════════════════════

    def _score_concentration_risk(
        self, gens: Sequence[GenerationData],
    ) -> SubScore:
        """
        INVERTED: higher score = LESS concentration = BETTER.

        Components:
          capital_equality:  1 - mean Gini.
                             Gini 0 → 10, Gini 1 → 0
          exposure_spread:   1 - mean exposure HHI.
                             HHI 0 → 10, HHI 1 → 0
          no_monopoly:       Fraction of gens where top1 < 50%.
                             100% → 10, 0% → 0
          dominance_safety:  1 - mean dominant lineage share.
                             0% → 10, 100% → 0
        """
        mean_gini = _safe_mean([g.capital_gini for g in gens])
        capital_equality = _clip10(10.0 * (1.0 - _clamp01(mean_gini)))

        mean_exp_hhi = _safe_mean([g.exposure_herfindahl for g in gens])
        exposure_spread = _clip10(10.0 * (1.0 - _clamp01(mean_exp_hhi)))

        n = max(len(gens), 1)
        no_mono = sum(1 for g in gens if g.capital_top1_pct < 50.0)
        no_monopoly = _clip10(10.0 * no_mono / n)

        mean_dom = _safe_mean([g.dominant_lineage_share for g in gens])
        dominance_safety = _clip10(10.0 * (1.0 - _clamp01(mean_dom)))

        score = _clip10(
            0.30 * capital_equality +
            0.25 * exposure_spread +
            0.25 * no_monopoly +
            0.20 * dominance_safety
        )

        return SubScore(
            name="Concentration Risk",
            score=score,
            grade=_grade(score),
            components={
                "capital_equality": capital_equality,
                "exposure_spread": exposure_spread,
                "no_monopoly": no_monopoly,
                "dominance_safety": dominance_safety,
            },
            commentary=_concentration_commentary(score, mean_gini, mean_exp_hhi),
        )

    # ════════════════════════════════════════════════════════
    # 4. Shock Resilience Score (0–10)
    # ════════════════════════════════════════════════════════

    def _score_shock_resilience(
        self,
        gens: Sequence[GenerationData],
        starting_capital: float,
        final_capital: float,
    ) -> SubScore:
        """
        Components:
          capital_preservation: final / starting.
                                100%+ → 10, 0% → 0
          recovery_rate:        How quickly capital recovers after dips.
                                Measured as fraction of gens where capital
                                is above starting.
          worst_gen_survival:   Worst single-gen PnL bounded.
                                No loss → 10, >30% loss → 0
          no_halts:             Fraction of gens NOT in halted state.
                                100% → 10, 0% → 0
        """
        cap = max(starting_capital, 0.01)

        # Capital preservation
        ratio = final_capital / cap
        # Sigmoid mapping: ratio 0→0, 0.5→3, 1.0→7, 1.5→9, 2.0→10
        preservation = _clip10(10.0 * _sigmoid_map(ratio, midpoint=1.0, steepness=3.0))

        # Recovery rate: fraction of gens with capital >= starting
        n = max(len(gens), 1)
        above = sum(1 for g in gens if g.pool_capital >= cap * 0.95)
        recovery_rate = _clip10(10.0 * above / n)

        # Worst single-gen P&L
        worst = min((g.pool_pnl for g in gens), default=0.0)
        worst_loss_pct = abs(min(worst, 0.0)) / cap
        worst_gen = _clip10(10.0 * (1.0 - min(worst_loss_pct / 0.30, 1.0)))

        # No halts
        non_halted = sum(1 for g in gens if g.portfolio_state != "halted")
        no_halts = _clip10(10.0 * non_halted / n)

        score = _clip10(
            0.30 * preservation +
            0.25 * recovery_rate +
            0.25 * worst_gen +
            0.20 * no_halts
        )

        return SubScore(
            name="Shock Resilience",
            score=score,
            grade=_grade(score),
            components={
                "capital_preservation": preservation,
                "recovery_rate": recovery_rate,
                "worst_gen_survival": worst_gen,
                "no_halts": no_halts,
            },
            commentary=_shock_commentary(
                score, ratio, worst_loss_pct, non_halted, n),
        )

    # ════════════════════════════════════════════════════════
    # 5. Learning Quality Score (0–10)
    # ════════════════════════════════════════════════════════

    def _score_learning_quality(
        self, gens: Sequence[GenerationData],
    ) -> SubScore:
        """
        Components:
          fitness_trend:     OLS slope of avg_fitness across gens.
                             Positive slope → higher score.
          win_rate_trend:    OLS slope of pool_win_rate.
                             Improving → higher score.
          sharpe_trend:      OLS slope of pool_sharpe.
                             Improving → higher score.
          trade_activity:    Mean trades per gen. More trades =
                             agents learned to actually participate.
                             0 trades → 0, 20+ per agent → 10
        """
        n = len(gens)

        # Fitness trend
        fitnesses = [g.avg_fitness for g in gens]
        fit_slope = _ols_slope(fitnesses)
        # Normalize: slope of 0.01/gen is very good for [0,1] fitness
        fitness_trend = _clip10(
            5.0 + 500.0 * fit_slope  # center at 5, ±0.01/gen spans full range
        )

        # Win rate trend
        wrs = [g.pool_win_rate for g in gens]
        wr_slope = _ols_slope(wrs)
        win_rate_trend = _clip10(5.0 + 200.0 * wr_slope)

        # Sharpe trend
        sharpes = [g.pool_sharpe for g in gens]
        sh_slope = _ols_slope(sharpes)
        sharpe_trend = _clip10(5.0 + 50.0 * sh_slope)

        # Trade activity (agents actually participating)
        mean_trades = _safe_mean([float(g.total_trades) for g in gens])
        pool_sizes = [max(g.population_size, 1) for g in gens]
        mean_pool = _safe_mean([float(p) for p in pool_sizes])
        trades_per_agent = mean_trades / max(mean_pool, 1.0)
        # 0 → 0, 5 → 5, 10+ → 8, 20+ → 10
        trade_activity = _clip10(10.0 * min(trades_per_agent / 15.0, 1.0))

        score = _clip10(
            0.30 * fitness_trend +
            0.25 * win_rate_trend +
            0.25 * sharpe_trend +
            0.20 * trade_activity
        )

        return SubScore(
            name="Learning Quality",
            score=score,
            grade=_grade(score),
            components={
                "fitness_trend": fitness_trend,
                "win_rate_trend": win_rate_trend,
                "sharpe_trend": sharpe_trend,
                "trade_activity": trade_activity,
            },
            commentary=_learning_commentary(
                score, fit_slope, wr_slope, trades_per_agent),
        )


# ═════════════════════════════════════════════════════════════
# Math helpers — all NaN/inf safe
# ═════════════════════════════════════════════════════════════

def _clip10(x: float) -> float:
    """Clip to [0, 10], NaN → 0."""
    if math.isnan(x) or math.isinf(x):
        return 0.0
    return max(0.0, min(10.0, x))

def _clamp01(x: float) -> float:
    if math.isnan(x) or math.isinf(x):
        return 0.0
    return max(0.0, min(1.0, x))

def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    filtered = [v for v in values if not (math.isnan(v) or math.isinf(v))]
    return sum(filtered) / max(len(filtered), 1)

def _safe_std(values: Sequence[float]) -> float:
    filtered = [v for v in values if not (math.isnan(v) or math.isinf(v))]
    if len(filtered) < 2:
        return 0.0
    try:
        return statistics.stdev(filtered)
    except statistics.StatisticsError:
        return 0.0

def _ols_slope(values: Sequence[float]) -> float:
    """Ordinary least squares slope for equally-spaced x values."""
    n = len(values)
    if n < 2:
        return 0.0
    filtered = [v if not (math.isnan(v) or math.isinf(v)) else 0.0
                for v in values]
    x_mean = (n - 1) / 2.0
    y_mean = sum(filtered) / n
    num = sum((i - x_mean) * (filtered[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den > 0 else 0.0

def _sigmoid_map(x: float, midpoint: float = 1.0, steepness: float = 3.0) -> float:
    """Map x to [0, 1] via sigmoid centered at midpoint."""
    try:
        return 1.0 / (1.0 + math.exp(-steepness * (x - midpoint)))
    except OverflowError:
        return 1.0 if x > midpoint else 0.0

def _grade(score: float) -> str:
    """Convert 0–10 score to letter grade."""
    if score >= 8.5:
        return "A"
    if score >= 7.0:
        return "B"
    if score >= 5.0:
        return "C"
    if score >= 3.0:
        return "D"
    return "F"


# ═════════════════════════════════════════════════════════════
# Commentary generators
# ═════════════════════════════════════════════════════════════

def _risk_commentary(
    score: float, max_dd: float, normal_gens: int, total_gens: int,
) -> str:
    if score >= 8:
        return (f"Excellent risk control. Max drawdown {max_dd:.1f}%, "
                f"{normal_gens}/{total_gens} generations in normal state.")
    if score >= 5:
        return (f"Moderate risk exposure. Max drawdown {max_dd:.1f}%. "
                f"Consider tightening stop-losses or reducing position sizes.")
    return (f"High risk detected. Max drawdown {max_dd:.1f}%, "
            f"only {normal_gens}/{total_gens} generations in normal state. "
            f"Risk parameters need significant adjustment.")

def _evolution_commentary(
    score: float, mean_div: float, non_stag: int, total: int,
) -> str:
    if score >= 8:
        return (f"Evolution functioning well. Mean diversity {mean_div:.2f}, "
                f"{non_stag}/{total} generations actively improving.")
    if score >= 5:
        return (f"Evolution adequate but showing signs of convergence. "
                f"Diversity at {mean_div:.2f}. Consider increasing mutation rate.")
    return (f"Evolution struggling. Diversity at {mean_div:.2f}, "
            f"only {non_stag}/{total} non-stagnating generations. "
            f"Population may need restart with higher mutation/crossover rates.")

def _concentration_commentary(
    score: float, mean_gini: float, mean_exp_hhi: float,
) -> str:
    if score >= 8:
        return (f"Well-distributed capital and exposure. "
                f"Gini {mean_gini:.2f}, exposure HHI {mean_exp_hhi:.2f}.")
    if score >= 5:
        return (f"Some concentration risk. Gini {mean_gini:.2f}, "
                f"exposure HHI {mean_exp_hhi:.2f}. "
                f"Consider adding more trading symbols or rebalancing capital.")
    return (f"Dangerous concentration. Gini {mean_gini:.2f}, "
            f"exposure HHI {mean_exp_hhi:.2f}. "
            f"Single point of failure risk is high.")

def _shock_commentary(
    score: float, ratio: float, worst_loss: float,
    non_halted: int, total: int,
) -> str:
    pct = ratio * 100
    if score >= 8:
        return (f"Strong resilience. Capital at {pct:.0f}% of start, "
                f"{non_halted}/{total} generations without halts.")
    if score >= 5:
        return (f"Moderate resilience. Capital at {pct:.0f}% of start, "
                f"worst gen loss {worst_loss*100:.1f}%. "
                f"Drawdown recovery is adequate but could improve.")
    return (f"Poor shock resilience. Capital at {pct:.0f}% of start, "
            f"worst gen loss {worst_loss*100:.1f}%, "
            f"only {non_halted}/{total} without halts. "
            f"Circuit breakers may need recalibration.")

def _learning_commentary(
    score: float, fit_slope: float, wr_slope: float, tpa: float,
) -> str:
    if score >= 8:
        return (f"Strong learning signal. Fitness trend +{fit_slope:.4f}/gen, "
                f"win rate trend +{wr_slope:.4f}/gen, "
                f"{tpa:.1f} trades/agent/gen.")
    if score >= 5:
        return (f"Some learning detected. Fitness trend {fit_slope:+.4f}/gen, "
                f"{tpa:.1f} trades/agent/gen. "
                f"Evolution is finding better strategies gradually.")
    return (f"Minimal learning. Fitness trend {fit_slope:+.4f}/gen, "
            f"only {tpa:.1f} trades/agent/gen. "
            f"Agents may not be generating enough signals or "
            f"the fitness landscape may be too flat.")
