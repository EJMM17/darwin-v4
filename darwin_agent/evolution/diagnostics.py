"""
Darwin v4 — EvolutionDiagnostics.

Pure analytical module. ZERO modification to trading logic.
Reads from AgentEvalData, DNAData, and GenerationSnapshot to
produce structured diagnostic reports per generation.

WHAT THIS MODULE DETECTS:

  1. GENETIC COLLAPSE — When all agents converge on identical genes,
     the population can no longer explore the fitness landscape. This
     module computes pairwise gene distance and Shannon entropy per gene
     to quantify how much "genetic material" remains for evolution to
     work with.

  2. CAPITAL CONCENTRATION — When one agent holds 80%+ of pool capital,
     killing it (from low fitness) would crash the pool. The Herfindahl
     index quantifies how evenly capital is distributed.

  3. EXPOSURE CONCENTRATION — When all agents trade the same symbol,
     a single market event can wipe the entire pool. This module
     computes per-symbol exposure share and flags dangerous concentration.

  4. EVOLUTIONARY DOMINANCE — When one DNA lineage (parent_id chain)
     dominates the population, diversity is effectively lost even if
     gene values vary slightly. This module tracks lineage dominance.

  5. FITNESS STAGNATION — When average fitness stops improving across
     generations, evolution has either converged (good) or stalled (bad).
     This module tracks generation-over-generation fitness trends and
     detects stagnation.

DESIGN PRINCIPLES:
  - Stateless per-call: each method takes explicit inputs, returns values
  - Only historical state is the fitness history ring buffer
  - All outputs are dataclasses with to_dict() for JSON serialization
  - No dependencies on EvolutionEngine, AgentManager, or any mutable state
  - Safe for concurrent reads (no locks needed — pure functions)
"""
from __future__ import annotations

import math
import statistics
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Sequence, Tuple

from darwin_agent.interfaces.types import AgentEvalData, DNAData


# ═════════════════════════════════════════════════════════════
# Output dataclasses
# ═════════════════════════════════════════════════════════════

@dataclass(slots=True)
class GeneEntropy:
    """Shannon entropy for a single gene across the population."""
    gene_name: str = ""
    entropy: float = 0.0          # 0 = all identical, higher = more diverse
    max_entropy: float = 0.0      # log2(n_bins), the theoretical max
    normalized: float = 0.0       # entropy / max_entropy, bounded [0, 1]
    mean_value: float = 0.0
    std_value: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0
    range_usage_pct: float = 0.0  # how much of [gene_min, gene_max] is used


@dataclass(slots=True)
class DiversityReport:
    """Genetic diversity across the active population."""
    population_size: int = 0
    mean_pairwise_distance: float = 0.0  # avg Euclidean dist between all pairs
    min_pairwise_distance: float = 0.0
    max_pairwise_distance: float = 0.0
    overall_diversity_score: float = 0.0  # [0,1] — 0=clones, 1=maximally spread
    gene_entropies: List[GeneEntropy] = field(default_factory=list)
    mean_entropy: float = 0.0
    lowest_entropy_gene: str = ""        # the gene that's most converged
    alert_collapsed: bool = False        # True if diversity dangerously low


@dataclass(slots=True)
class ConcentrationReport:
    """Capital and exposure concentration across agents."""
    # Capital
    capital_herfindahl: float = 0.0      # HHI: 1/n = perfect equality, 1.0 = monopoly
    capital_gini: float = 0.0            # Gini: 0 = equal, 1 = one agent has all
    capital_top1_pct: float = 0.0        # % of total capital held by richest agent
    capital_top3_pct: float = 0.0
    capital_alert: bool = False          # True if top1 > 60%

    # Exposure
    exposure_herfindahl: float = 0.0     # HHI across symbols
    exposure_top1_symbol: str = ""
    exposure_top1_pct: float = 0.0       # % of total notional in top symbol
    exposure_by_symbol: Dict[str, float] = field(default_factory=dict)
    exposure_alert: bool = False         # True if top1 > 70%


@dataclass(slots=True)
class DominanceReport:
    """Evolutionary lineage dominance detection."""
    unique_lineages: int = 0
    total_agents: int = 0
    dominant_lineage_id: str = ""
    dominant_lineage_share: float = 0.0  # fraction of pop from one parent
    effective_lineages: float = 0.0      # 1/HHI — like "effective number of species"
    alert_dominant: bool = False         # True if one lineage > 50%


@dataclass(slots=True)
class FitnessTrajectory:
    """Cross-generation fitness tracking."""
    generation: int = 0
    current_best: float = 0.0
    current_avg: float = 0.0
    current_worst: float = 0.0
    current_variance: float = 0.0
    current_spread: float = 0.0    # best - worst

    # Trends (compared to previous generations)
    avg_trend_5gen: float = 0.0    # slope of avg fitness over last 5 gens
    best_trend_5gen: float = 0.0
    stagnation_generations: int = 0  # how many gens avg hasn't improved
    improving: bool = False
    stagnating: bool = False       # True if stagnation > 5 gens

    # Historical context
    all_time_best: float = 0.0
    all_time_best_generation: int = 0
    generations_tracked: int = 0


@dataclass(slots=True)
class DiagnosticsReport:
    """Complete diagnostics for one generation."""
    generation: int = 0
    diversity: DiversityReport = field(default_factory=DiversityReport)
    concentration: ConcentrationReport = field(default_factory=ConcentrationReport)
    dominance: DominanceReport = field(default_factory=DominanceReport)
    fitness: FitnessTrajectory = field(default_factory=FitnessTrajectory)

    # Aggregated health
    health_score: float = 0.0      # [0,1] composite of all diagnostics
    alerts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "generation": self.generation,
            "health_score": round(self.health_score, 4),
            "alerts": self.alerts,
            "diversity": {
                "population_size": self.diversity.population_size,
                "overall_diversity_score": round(self.diversity.overall_diversity_score, 4),
                "mean_pairwise_distance": round(self.diversity.mean_pairwise_distance, 4),
                "mean_entropy": round(self.diversity.mean_entropy, 4),
                "lowest_entropy_gene": self.diversity.lowest_entropy_gene,
                "alert_collapsed": self.diversity.alert_collapsed,
                "gene_entropies": [
                    {
                        "gene": ge.gene_name,
                        "entropy": round(ge.entropy, 4),
                        "normalized": round(ge.normalized, 4),
                        "mean": round(ge.mean_value, 4),
                        "std": round(ge.std_value, 4),
                        "range_usage_pct": round(ge.range_usage_pct, 1),
                    }
                    for ge in self.diversity.gene_entropies
                ],
            },
            "concentration": {
                "capital_herfindahl": round(self.concentration.capital_herfindahl, 4),
                "capital_gini": round(self.concentration.capital_gini, 4),
                "capital_top1_pct": round(self.concentration.capital_top1_pct, 1),
                "capital_top3_pct": round(self.concentration.capital_top3_pct, 1),
                "capital_alert": self.concentration.capital_alert,
                "exposure_herfindahl": round(self.concentration.exposure_herfindahl, 4),
                "exposure_top1_symbol": self.concentration.exposure_top1_symbol,
                "exposure_top1_pct": round(self.concentration.exposure_top1_pct, 1),
                "exposure_by_symbol": {
                    k: round(v, 4) for k, v in self.concentration.exposure_by_symbol.items()
                },
                "exposure_alert": self.concentration.exposure_alert,
            },
            "dominance": {
                "unique_lineages": self.dominance.unique_lineages,
                "total_agents": self.dominance.total_agents,
                "dominant_lineage_id": self.dominance.dominant_lineage_id,
                "dominant_lineage_share": round(self.dominance.dominant_lineage_share, 4),
                "effective_lineages": round(self.dominance.effective_lineages, 2),
                "alert_dominant": self.dominance.alert_dominant,
            },
            "fitness": {
                "current_best": round(self.fitness.current_best, 4),
                "current_avg": round(self.fitness.current_avg, 4),
                "current_worst": round(self.fitness.current_worst, 4),
                "current_variance": round(self.fitness.current_variance, 6),
                "current_spread": round(self.fitness.current_spread, 4),
                "avg_trend_5gen": round(self.fitness.avg_trend_5gen, 6),
                "best_trend_5gen": round(self.fitness.best_trend_5gen, 6),
                "stagnation_generations": self.fitness.stagnation_generations,
                "improving": self.fitness.improving,
                "stagnating": self.fitness.stagnating,
                "all_time_best": round(self.fitness.all_time_best, 4),
                "all_time_best_generation": self.fitness.all_time_best_generation,
                "generations_tracked": self.fitness.generations_tracked,
            },
        }


# ═════════════════════════════════════════════════════════════
# EvolutionDiagnostics
# ═════════════════════════════════════════════════════════════

# Thresholds
_DIVERSITY_COLLAPSE_THRESHOLD = 0.15
_CAPITAL_ALERT_TOP1_PCT = 60.0
_EXPOSURE_ALERT_TOP1_PCT = 70.0
_DOMINANCE_ALERT_SHARE = 0.50
_STAGNATION_GENS = 5
_ENTROPY_BINS = 10
_FITNESS_HISTORY_MAXLEN = 50


@dataclass(slots=True)
class _FitnessSnapshot:
    """Internal per-generation fitness record."""
    generation: int
    best: float
    avg: float
    worst: float
    variance: float


class EvolutionDiagnostics:
    """
    Pure analytical engine. Reads data, returns structured reports.
    Holds only a fitness history ring buffer as internal state.
    Thread-safe for concurrent reads (single-writer for record_generation).
    """

    __slots__ = ("_gene_ranges", "_fitness_history",
                 "_all_time_best", "_all_time_best_gen")

    def __init__(
        self,
        gene_ranges: Dict[str, Tuple[float, float]] | None = None,
    ) -> None:
        from darwin_agent.evolution.engine import DEFAULT_GENE_RANGES
        self._gene_ranges = gene_ranges or dict(DEFAULT_GENE_RANGES)
        self._fitness_history: Deque[_FitnessSnapshot] = deque(
            maxlen=_FITNESS_HISTORY_MAXLEN,
        )
        self._all_time_best = 0.0
        self._all_time_best_gen = 0

    # ════════════════════════════════════════════════════════
    # Main entry point
    # ════════════════════════════════════════════════════════

    def diagnose(
        self,
        generation: int,
        agents: Sequence[AgentEvalData],
        dna_pool: Sequence[DNAData] | None = None,
    ) -> DiagnosticsReport:
        """
        Produce a complete diagnostics report for the current generation.

        Args:
            generation: Current generation number.
            agents:     Active AgentEvalData bundles (with metrics, exposure, DNA).
            dna_pool:   Optional explicit DNA pool. If None, extracted from agents.

        Returns:
            DiagnosticsReport with all sub-reports and composite health score.
        """
        if not agents:
            return DiagnosticsReport(generation=generation)

        dna_list = dna_pool or [a.dna for a in agents if a.dna is not None]
        fitnesses = [a.metrics.fitness for a in agents]

        diversity = self._compute_diversity(dna_list)
        concentration = self._compute_concentration(agents)
        dominance = self._compute_dominance(dna_list)
        fitness_traj = self._compute_fitness_trajectory(generation, fitnesses)

        # Record for history tracking
        self._record_generation(generation, fitnesses)

        # Aggregate alerts
        alerts = []
        if diversity.alert_collapsed:
            alerts.append(
                f"GENETIC_COLLAPSE: diversity={diversity.overall_diversity_score:.2f}, "
                f"most converged gene={diversity.lowest_entropy_gene}"
            )
        if concentration.capital_alert:
            alerts.append(
                f"CAPITAL_CONCENTRATION: top1={concentration.capital_top1_pct:.0f}%"
            )
        if concentration.exposure_alert:
            alerts.append(
                f"EXPOSURE_CONCENTRATION: {concentration.exposure_top1_symbol}"
                f"={concentration.exposure_top1_pct:.0f}%"
            )
        if dominance.alert_dominant:
            alerts.append(
                f"LINEAGE_DOMINANCE: {dominance.dominant_lineage_id}"
                f"={dominance.dominant_lineage_share:.0%} of population"
            )
        if fitness_traj.stagnating:
            alerts.append(
                f"FITNESS_STAGNATION: {fitness_traj.stagnation_generations} gens "
                f"without improvement"
            )

        # Composite health: average of 4 sub-scores, each [0,1]
        div_health = diversity.overall_diversity_score
        cap_health = 1.0 - concentration.capital_gini
        dom_health = min(1.0, dominance.effective_lineages / max(len(agents), 1))
        fit_health = 1.0 if fitness_traj.improving else (
            0.5 if not fitness_traj.stagnating else 0.2
        )
        health = _safe_mean([div_health, cap_health, dom_health, fit_health])

        return DiagnosticsReport(
            generation=generation,
            diversity=diversity,
            concentration=concentration,
            dominance=dominance,
            fitness=fitness_traj,
            health_score=round(health, 4),
            alerts=alerts,
        )

    # ════════════════════════════════════════════════════════
    # 1 + 2: Genetic diversity + per-gene entropy
    # ════════════════════════════════════════════════════════

    def _compute_diversity(
        self, dna_list: Sequence[DNAData],
    ) -> DiversityReport:
        n = len(dna_list)
        if n == 0:
            return DiversityReport()

        # Extract gene matrices
        all_genes = sorted(self._gene_ranges.keys())
        matrix = []  # n × g normalized values
        for dna in dna_list:
            row = []
            for gene in all_genes:
                lo, hi = self._gene_ranges.get(gene, (0.0, 1.0))
                raw = dna.genes.get(gene, (lo + hi) / 2)
                rng = hi - lo if hi > lo else 1.0
                row.append((raw - lo) / rng)  # normalize to [0, 1]
            matrix.append(row)

        # Pairwise Euclidean distances (normalized by sqrt(n_genes))
        g = len(all_genes)
        norm_factor = math.sqrt(g) if g > 0 else 1.0
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                d = math.sqrt(sum(
                    (matrix[i][k] - matrix[j][k]) ** 2 for k in range(g)
                )) / norm_factor
                distances.append(d)

        mean_dist = _safe_mean(distances) if distances else 0.0
        min_dist = min(distances) if distances else 0.0
        max_dist = max(distances) if distances else 0.0

        # Per-gene Shannon entropy
        gene_entropies = []
        for gi, gene in enumerate(all_genes):
            values = [matrix[row_i][gi] for row_i in range(n)]
            lo, hi = self._gene_ranges.get(gene, (0.0, 1.0))
            raw_values = [dna.genes.get(gene, (lo + hi) / 2) for dna in dna_list]
            ge = self._gene_entropy(gene, values, raw_values, lo, hi)
            gene_entropies.append(ge)

        mean_ent = _safe_mean([ge.normalized for ge in gene_entropies])
        lowest = min(gene_entropies, key=lambda ge: ge.normalized) if gene_entropies else None

        # Overall diversity score: blend of mean distance and mean entropy
        diversity_score = _clamp01(0.5 * mean_dist + 0.5 * mean_ent)

        return DiversityReport(
            population_size=n,
            mean_pairwise_distance=mean_dist,
            min_pairwise_distance=min_dist,
            max_pairwise_distance=max_dist,
            overall_diversity_score=diversity_score,
            gene_entropies=gene_entropies,
            mean_entropy=mean_ent,
            lowest_entropy_gene=lowest.gene_name if lowest else "",
            alert_collapsed=diversity_score < _DIVERSITY_COLLAPSE_THRESHOLD,
        )

    def _gene_entropy(
        self,
        gene_name: str,
        normalized_values: List[float],
        raw_values: List[float],
        range_lo: float,
        range_hi: float,
    ) -> GeneEntropy:
        """Shannon entropy for one gene using histogram binning."""
        n = len(normalized_values)
        if n <= 1:
            return GeneEntropy(
                gene_name=gene_name,
                mean_value=raw_values[0] if raw_values else 0.0,
                min_value=raw_values[0] if raw_values else 0.0,
                max_value=raw_values[0] if raw_values else 0.0,
            )

        # Bin the normalized [0,1] values
        bins = min(_ENTROPY_BINS, n)
        counts = [0] * bins
        for v in normalized_values:
            idx = min(int(v * bins), bins - 1)
            counts[idx] += 1

        # Shannon entropy: H = -Σ p·log2(p)
        entropy = 0.0
        for c in counts:
            if c > 0:
                p = c / n
                entropy -= p * math.log2(p)

        max_entropy = math.log2(bins)
        normalized = entropy / max_entropy if max_entropy > 0 else 0.0

        # Raw value statistics
        mean_v = _safe_mean(raw_values)
        std_v = _safe_std(raw_values)
        min_v = min(raw_values)
        max_v = max(raw_values)
        rng = range_hi - range_lo if range_hi > range_lo else 1.0
        range_usage = ((max_v - min_v) / rng) * 100.0

        return GeneEntropy(
            gene_name=gene_name,
            entropy=entropy,
            max_entropy=max_entropy,
            normalized=normalized,
            mean_value=mean_v,
            std_value=std_v,
            min_value=min_v,
            max_value=max_v,
            range_usage_pct=range_usage,
        )

    # ════════════════════════════════════════════════════════
    # 3 + 4: Capital and exposure concentration
    # ════════════════════════════════════════════════════════

    def _compute_concentration(
        self, agents: Sequence[AgentEvalData],
    ) -> ConcentrationReport:
        n = len(agents)
        if n == 0:
            return ConcentrationReport()

        # ── Capital concentration ────────────────────────────
        capitals = sorted(
            [max(a.metrics.capital, 0.0) for a in agents], reverse=True,
        )
        total_cap = sum(capitals) or 1.0

        shares = [c / total_cap for c in capitals]
        hhi_cap = sum(s ** 2 for s in shares)
        gini = self._gini_coefficient(capitals)
        top1 = shares[0] * 100 if shares else 0.0
        top3 = sum(shares[:3]) * 100 if len(shares) >= 3 else top1

        # ── Exposure concentration ───────────────────────────
        symbol_totals: Dict[str, float] = {}
        for a in agents:
            for sym, frac in (a.exposure or {}).items():
                symbol_totals[sym] = symbol_totals.get(sym, 0.0) + abs(frac)

        total_exp = sum(symbol_totals.values()) or 1.0
        exp_shares = {s: v / total_exp for s, v in symbol_totals.items()}

        hhi_exp = sum(s ** 2 for s in exp_shares.values())
        top_sym = max(exp_shares, key=exp_shares.get) if exp_shares else ""
        top_exp_pct = exp_shares.get(top_sym, 0.0) * 100

        return ConcentrationReport(
            capital_herfindahl=hhi_cap,
            capital_gini=gini,
            capital_top1_pct=top1,
            capital_top3_pct=top3,
            capital_alert=top1 > _CAPITAL_ALERT_TOP1_PCT,
            exposure_herfindahl=hhi_exp,
            exposure_top1_symbol=top_sym,
            exposure_top1_pct=top_exp_pct,
            exposure_by_symbol=exp_shares,
            exposure_alert=top_exp_pct > _EXPOSURE_ALERT_TOP1_PCT,
        )

    @staticmethod
    def _gini_coefficient(sorted_desc: List[float]) -> float:
        """Gini coefficient from a descending-sorted list of values."""
        n = len(sorted_desc)
        if n <= 1:
            return 0.0
        values = sorted(sorted_desc)  # need ascending for formula
        total = sum(values)
        if total <= 0:
            return 0.0
        cum = 0.0
        weighted_sum = 0.0
        for i, v in enumerate(values):
            cum += v
            weighted_sum += (2 * (i + 1) - n - 1) * v
        return weighted_sum / (n * total)

    # ════════════════════════════════════════════════════════
    # 5: Evolutionary dominance
    # ════════════════════════════════════════════════════════

    def _compute_dominance(
        self, dna_list: Sequence[DNAData],
    ) -> DominanceReport:
        n = len(dna_list)
        if n == 0:
            return DominanceReport()

        # Count lineages by parent_id (None = unique origin)
        lineage_counts: Counter = Counter()
        for dna in dna_list:
            lineage = dna.parent_id or dna.dna_id
            lineage_counts[lineage] += 1

        unique = len(lineage_counts)
        dominant_id, dominant_count = lineage_counts.most_common(1)[0]
        dominant_share = dominant_count / n

        # Effective number of lineages = 1 / HHI
        shares = [c / n for c in lineage_counts.values()]
        hhi = sum(s ** 2 for s in shares)
        effective = 1.0 / hhi if hhi > 0 else n

        return DominanceReport(
            unique_lineages=unique,
            total_agents=n,
            dominant_lineage_id=dominant_id,
            dominant_lineage_share=dominant_share,
            effective_lineages=effective,
            alert_dominant=dominant_share > _DOMINANCE_ALERT_SHARE,
        )

    # ════════════════════════════════════════════════════════
    # 6: Fitness variance and trajectory
    # ════════════════════════════════════════════════════════

    def _compute_fitness_trajectory(
        self,
        generation: int,
        fitnesses: List[float],
    ) -> FitnessTrajectory:
        if not fitnesses:
            return FitnessTrajectory(generation=generation)

        best = max(fitnesses)
        avg = _safe_mean(fitnesses)
        worst = min(fitnesses)
        var = _safe_var(fitnesses)
        spread = best - worst

        # Trends from history
        avg_trend = 0.0
        best_trend = 0.0
        stagnation = 0

        history = list(self._fitness_history)
        if len(history) >= 2:
            # Slope of average fitness over last 5 generations
            recent = history[-min(5, len(history)):]
            if len(recent) >= 2:
                avg_trend = self._linear_slope(
                    [h.avg for h in recent] + [avg])
                best_trend = self._linear_slope(
                    [h.best for h in recent] + [best])

            # Stagnation: count consecutive gens where avg didn't improve
            prev_avg = history[-1].avg if history else 0.0
            if avg <= prev_avg + 1e-6:
                # Check how far back stagnation extends
                stagnation = 1
                for i in range(len(history) - 1, 0, -1):
                    if history[i].avg <= history[i - 1].avg + 1e-6:
                        stagnation += 1
                    else:
                        break

        improving = avg_trend > 1e-4

        # All-time best
        at_best = self._all_time_best
        at_best_gen = self._all_time_best_gen
        if best > at_best:
            at_best = best
            at_best_gen = generation

        return FitnessTrajectory(
            generation=generation,
            current_best=best,
            current_avg=avg,
            current_worst=worst,
            current_variance=var,
            current_spread=spread,
            avg_trend_5gen=avg_trend,
            best_trend_5gen=best_trend,
            stagnation_generations=stagnation,
            improving=improving,
            stagnating=stagnation >= _STAGNATION_GENS,
            all_time_best=at_best,
            all_time_best_generation=at_best_gen,
            generations_tracked=len(history) + 1,
        )

    def _record_generation(
        self, generation: int, fitnesses: List[float],
    ) -> None:
        """Append to fitness history ring buffer."""
        if not fitnesses:
            return
        best = max(fitnesses)
        if best > self._all_time_best:
            self._all_time_best = best
            self._all_time_best_gen = generation

        self._fitness_history.append(_FitnessSnapshot(
            generation=generation,
            best=best,
            avg=_safe_mean(fitnesses),
            worst=min(fitnesses),
            variance=_safe_var(fitnesses),
        ))

    # ════════════════════════════════════════════════════════
    # Dashboard convenience
    # ════════════════════════════════════════════════════════

    def get_fitness_history(self) -> List[Dict]:
        """Return full fitness history for charting."""
        return [
            {
                "generation": h.generation,
                "best": round(h.best, 4),
                "avg": round(h.avg, 4),
                "worst": round(h.worst, 4),
                "variance": round(h.variance, 6),
            }
            for h in self._fitness_history
        ]

    def reset(self) -> None:
        """Clear all history (for testing)."""
        self._fitness_history.clear()
        self._all_time_best = 0.0
        self._all_time_best_gen = 0

    # ════════════════════════════════════════════════════════
    # Static math utilities
    # ════════════════════════════════════════════════════════

    @staticmethod
    def _linear_slope(values: List[float]) -> float:
        """Ordinary least squares slope for equally-spaced x values."""
        n = len(values)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n
        num = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n))
        return num / den if den > 0 else 0.0


# ═════════════════════════════════════════════════════════════
# Module-level math helpers
# ═════════════════════════════════════════════════════════════

def _clamp01(x: float) -> float:
    if math.isnan(x) or math.isinf(x):
        return 0.0
    return max(0.0, min(1.0, x))


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _safe_var(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    try:
        return statistics.variance(values)
    except statistics.StatisticsError:
        return 0.0


def _safe_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    try:
        return statistics.stdev(values)
    except statistics.StatisticsError:
        return 0.0
