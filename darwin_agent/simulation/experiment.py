"""
Darwin v4 — ExperimentRunner.

Batch simulation engine for systematic parameter exploration.
Accepts N config variations × M seeds, runs every combination,
scores each with SimulationScorecard, and exports a comparison
CSV that reveals which evolution parameters produce the best
risk-adjusted results across stochastic market paths.

USAGE:

    from darwin_agent.simulation.experiment import (
        ExperimentRunner, ExperimentConfig, ConfigVariation,
    )
    from darwin_agent.simulation.harness import MonteCarloScenario

    runner = ExperimentRunner(ExperimentConfig(
        scenario=MonteCarloScenario(symbols=["BTCUSDT", "ETHUSDT"]),
        variations=[
            ConfigVariation(config_id="conservative",
                            pool_size=5, mutation_rate=0.05, survival_rate=0.6),
            ConfigVariation(config_id="aggressive",
                            pool_size=10, mutation_rate=0.30, survival_rate=0.3),
            ConfigVariation(config_id="balanced",
                            pool_size=8, mutation_rate=0.15, survival_rate=0.5),
        ],
        seeds=[42, 123, 456, 789, 1024],
        generations=30,
        trades_per_generation=200,
        starting_capital=50.0,
    ))
    results = await runner.run()
    results.export_csv("experiment_results.csv")
    results.export_json("experiment_results.json")
    results.print_leaderboard()

DESIGN:
    - Zero trading logic; delegates entirely to SimulationHarness
    - Deterministic: same variation + same seed = identical output
    - Each (variation, seed) pair is an independent SimulationHarness run
    - Scorecard computed per-run, then aggregated across seeds per variation
    - CSV columns designed for direct pandas/spreadsheet analysis
"""
from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from darwin_agent.simulation.harness import (
    MarketScenario, MonteCarloScenario, SimConfig, SimulationHarness,
    SimulationResults,
)
from darwin_agent.simulation.scorecard import (
    GenerationData, ScorecardWeights, SimulationScorecard,
)
from darwin_agent.risk.portfolio_engine import RiskLimits

logger = logging.getLogger("darwin.experiment")


# ═════════════════════════════════════════════════════════════
# Config variation
# ═════════════════════════════════════════════════════════════

@dataclass
class ConfigVariation:
    """
    One parameter set to test.  Every field is optional — unset
    fields inherit from ExperimentConfig defaults.
    """
    config_id: str = "default"
    # Evolution
    pool_size: Optional[int] = None
    survival_rate: Optional[float] = None
    elitism_count: Optional[int] = None
    mutation_rate: Optional[float] = None
    # Simulation
    generations: Optional[int] = None
    trades_per_generation: Optional[int] = None
    starting_capital: Optional[float] = None
    # Risk
    risk_limits: Optional[RiskLimits] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"config_id": self.config_id}
        for f in ("pool_size", "survival_rate", "elitism_count",
                  "mutation_rate", "generations", "trades_per_generation",
                  "starting_capital"):
            v = getattr(self, f)
            if v is not None:
                d[f] = v
        return d


# ═════════════════════════════════════════════════════════════
# Experiment config
# ═════════════════════════════════════════════════════════════

@dataclass
class ExperimentConfig:
    """Top-level experiment parameters."""
    # Scenario (shared across all variations)
    scenario: MarketScenario = field(
        default_factory=lambda: MonteCarloScenario())

    # Variations to test
    variations: List[ConfigVariation] = field(default_factory=lambda: [
        ConfigVariation(config_id="default"),
    ])

    # Seeds for stochastic runs
    seeds: List[int] = field(default_factory=lambda: [42])

    # Defaults (overridden per-variation where specified)
    generations: int = 20
    pool_size: int = 5
    trades_per_generation: int = 200
    starting_capital: float = 50.0
    survival_rate: float = 0.5
    elitism_count: int = 1
    mutation_rate: float = 0.15
    risk_limits: RiskLimits = field(default_factory=RiskLimits)

    # Scorecard
    scorecard_weights: ScorecardWeights = field(
        default_factory=ScorecardWeights)


# ═════════════════════════════════════════════════════════════
# Per-run result
# ═════════════════════════════════════════════════════════════

@dataclass
class RunResult:
    """Result of one (config_id, seed) pair."""
    config_id: str = ""
    seed: int = 0
    scenario_type: str = ""

    # Simulation outputs
    generations_run: int = 0
    starting_capital: float = 0.0
    final_capital: float = 0.0
    capital_return_pct: float = 0.0
    best_fitness: float = 0.0
    elapsed_ms: float = 0.0

    # Scorecard
    ecosystem_health: float = 0.0
    ecosystem_grade: str = ""
    risk_stability: float = 0.0
    evolution_health: float = 0.0
    concentration_risk: float = 0.0
    shock_resilience: float = 0.0
    learning_quality: float = 0.0

    # Config params used
    config_params: Dict[str, Any] = field(default_factory=dict)

    # Full objects for JSON export
    scorecard_dict: Dict[str, Any] = field(default_factory=dict)

    def to_csv_row(self) -> Dict[str, Any]:
        """Flat dict for CSV export."""
        row = {
            "config_id": self.config_id,
            "seed": self.seed,
            "scenario": self.scenario_type,
            "generations": self.generations_run,
            "starting_capital": round(self.starting_capital, 2),
            "final_capital": round(self.final_capital, 2),
            "return_pct": round(self.capital_return_pct, 2),
            "best_fitness": round(self.best_fitness, 4),
            "elapsed_ms": round(self.elapsed_ms, 0),
            "ecosystem_health": round(self.ecosystem_health, 2),
            "ecosystem_grade": self.ecosystem_grade,
            "risk_stability": round(self.risk_stability, 2),
            "evolution_health": round(self.evolution_health, 2),
            "concentration_risk": round(self.concentration_risk, 2),
            "shock_resilience": round(self.shock_resilience, 2),
            "learning_quality": round(self.learning_quality, 2),
        }
        # Append variation params as columns
        for k, v in self.config_params.items():
            if k != "config_id":
                row[f"param_{k}"] = v
        return row


# ═════════════════════════════════════════════════════════════
# Aggregated variation summary
# ═════════════════════════════════════════════════════════════

@dataclass
class VariationSummary:
    """Aggregated statistics across seeds for one config variation."""
    config_id: str = ""
    n_seeds: int = 0
    config_params: Dict[str, Any] = field(default_factory=dict)

    # Means
    mean_ecosystem: float = 0.0
    mean_return_pct: float = 0.0
    mean_risk: float = 0.0
    mean_evolution: float = 0.0
    mean_concentration: float = 0.0
    mean_shock: float = 0.0
    mean_learning: float = 0.0
    mean_best_fitness: float = 0.0

    # Std devs (cross-seed stability)
    std_ecosystem: float = 0.0
    std_return_pct: float = 0.0

    # Worst/best across seeds
    min_return_pct: float = 0.0
    max_return_pct: float = 0.0
    min_ecosystem: float = 0.0
    max_ecosystem: float = 0.0

    # Grade distribution
    grade_counts: Dict[str, int] = field(default_factory=dict)

    def to_csv_row(self) -> Dict[str, Any]:
        row = {
            "config_id": self.config_id,
            "n_seeds": self.n_seeds,
            "mean_ecosystem": round(self.mean_ecosystem, 2),
            "std_ecosystem": round(self.std_ecosystem, 2),
            "mean_return_pct": round(self.mean_return_pct, 2),
            "std_return_pct": round(self.std_return_pct, 2),
            "min_return_pct": round(self.min_return_pct, 2),
            "max_return_pct": round(self.max_return_pct, 2),
            "mean_risk": round(self.mean_risk, 2),
            "mean_evolution": round(self.mean_evolution, 2),
            "mean_concentration": round(self.mean_concentration, 2),
            "mean_shock": round(self.mean_shock, 2),
            "mean_learning": round(self.mean_learning, 2),
            "mean_best_fitness": round(self.mean_best_fitness, 4),
            "grades": json.dumps(self.grade_counts),
        }
        for k, v in self.config_params.items():
            if k != "config_id":
                row[f"param_{k}"] = v
        return row


# ═════════════════════════════════════════════════════════════
# Experiment results
# ═════════════════════════════════════════════════════════════

@dataclass
class ExperimentResults:
    """Complete experiment output."""
    total_runs: int = 0
    total_elapsed_ms: float = 0.0
    runs: List[RunResult] = field(default_factory=list)
    summaries: List[VariationSummary] = field(default_factory=list)
    leaderboard: List[str] = field(default_factory=list)

    def export_csv(self, path: str) -> str:
        """Export per-run results to CSV."""
        p = Path(path)
        if not self.runs:
            p.write_text("")
            return str(p)

        rows = [r.to_csv_row() for r in self.runs]
        with open(p, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info("CSV exported: %s (%d runs)", p, len(rows))
        return str(p)

    def export_summary_csv(self, path: str) -> str:
        """Export per-variation aggregated summary."""
        p = Path(path)
        if not self.summaries:
            p.write_text("")
            return str(p)

        rows = [s.to_csv_row() for s in self.summaries]
        with open(p, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Summary CSV: %s (%d variations)", p, len(rows))
        return str(p)

    def export_json(self, path: str) -> str:
        """Export full results to JSON."""
        data = {
            "meta": {
                "total_runs": self.total_runs,
                "elapsed_ms": round(self.total_elapsed_ms, 0),
            },
            "leaderboard": self.leaderboard,
            "summaries": [s.to_csv_row() for s in self.summaries],
            "runs": [
                {
                    "config_id": r.config_id,
                    "seed": r.seed,
                    "final_capital": round(r.final_capital, 2),
                    "return_pct": round(r.capital_return_pct, 2),
                    "scorecard": r.scorecard_dict,
                    "config_params": r.config_params,
                }
                for r in self.runs
            ],
        }
        p = Path(path)
        with open(p, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("JSON exported: %s", p)
        return str(p)

    def print_leaderboard(self) -> None:
        """Print ranked variation leaderboard to stdout."""
        if not self.summaries:
            print("  No results.")
            return
        print()
        print("  ┌─────────────────────────────────────────────────────────────────┐")
        print("  │  EXPERIMENT LEADERBOARD                                         │")
        print("  ├──────┬──────────────────┬──────────┬──────────┬────────┬────────┤")
        print("  │ Rank │ Config           │ Health   │ Return   │ Risk   │ Learn  │")
        print("  ├──────┼──────────────────┼──────────┼──────────┼────────┼────────┤")
        for i, s in enumerate(self.summaries):
            print(f"  │ {i+1:>4} │ {s.config_id:<16s} │ "
                  f"{s.mean_ecosystem:>5.1f}/10  │ "
                  f"{s.mean_return_pct:>+7.1f}% │ "
                  f"{s.mean_risk:>5.1f}  │ "
                  f"{s.mean_learning:>5.1f}  │")
        print("  └──────┴──────────────────┴──────────┴──────────┴────────┴────────┘")
        print()


# ═════════════════════════════════════════════════════════════
# ExperimentRunner
# ═════════════════════════════════════════════════════════════

class ExperimentRunner:
    """
    Batch simulation engine.

    For each (variation, seed) pair:
      1. Build SimConfig from ExperimentConfig defaults + variation overrides
      2. Run SimulationHarness
      3. Convert results → GenerationData list
      4. Score with SimulationScorecard
      5. Store RunResult

    Then aggregate across seeds per variation → VariationSummary.
    Rank by mean ecosystem health → leaderboard.
    """

    __slots__ = ("_cfg", "_scorecard")

    def __init__(self, config: ExperimentConfig) -> None:
        self._cfg = config
        self._scorecard = SimulationScorecard(config.scorecard_weights)

    async def run(self) -> ExperimentResults:
        """Execute all (variation × seed) combinations."""
        cfg = self._cfg
        total_combinations = len(cfg.variations) * len(cfg.seeds)

        logger.info(
            "EXPERIMENT START: %d variations × %d seeds = %d runs",
            len(cfg.variations), len(cfg.seeds), total_combinations,
        )

        t0 = time.monotonic()
        all_runs: List[RunResult] = []
        run_idx = 0

        for var in cfg.variations:
            for seed in cfg.seeds:
                run_idx += 1
                logger.info(
                    "  run %d/%d: config=%s seed=%d",
                    run_idx, total_combinations, var.config_id, seed,
                )

                sim_cfg = self._build_sim_config(var, seed)
                harness = SimulationHarness(sim_cfg)
                sim_results = await harness.run()

                # Convert to GenerationData for scorecard
                gen_data = self._extract_gen_data(sim_results)

                # Score
                scorecard = self._scorecard.evaluate(
                    gen_data,
                    starting_capital=sim_results.starting_capital,
                    final_capital=sim_results.final_capital,
                )

                # Build RunResult
                ret_pct = 0.0
                if sim_results.starting_capital > 0:
                    ret_pct = ((sim_results.final_capital -
                                sim_results.starting_capital) /
                               sim_results.starting_capital) * 100

                sc_dict = scorecard.to_dict()

                sc_dict = scorecard.to_dict()
                run_result = RunResult(
                    config_id=var.config_id,
                    seed=seed,
                    scenario_type=sim_results.scenario_type,
                    generations_run=sim_results.generations_run,
                    starting_capital=sim_results.starting_capital,
                    final_capital=sim_results.final_capital,
                    capital_return_pct=ret_pct,
                    best_fitness=sim_results.best_fitness_ever,
                    elapsed_ms=sim_results.total_elapsed_ms,
                    ecosystem_health=scorecard.ecosystem_health,
                    ecosystem_grade=scorecard.ecosystem_grade,
                    risk_stability=scorecard.risk_stability.score,
                    evolution_health=scorecard.evolution_health.score,
                    concentration_risk=scorecard.concentration_risk.score,
                    shock_resilience=scorecard.shock_resilience.score,
                    learning_quality=scorecard.learning_quality.score,
                    config_params=var.to_dict(),
                    scorecard_dict=sc_dict,
                )
                all_runs.append(run_result)

        total_ms = (time.monotonic() - t0) * 1000

        # Aggregate per variation
        summaries = self._aggregate(all_runs, cfg.variations)

        # Rank by mean ecosystem health (descending)
        summaries.sort(key=lambda s: s.mean_ecosystem, reverse=True)
        leaderboard = [
            f"#{i+1} {s.config_id}: health={s.mean_ecosystem:.1f}/10, "
            f"return={s.mean_return_pct:+.1f}%"
            for i, s in enumerate(summaries)
        ]

        logger.info(
            "EXPERIMENT COMPLETE: %d runs in %.0fms",
            total_combinations, total_ms,
        )

        return ExperimentResults(
            total_runs=len(all_runs),
            total_elapsed_ms=total_ms,
            runs=all_runs,
            summaries=summaries,
            leaderboard=leaderboard,
        )

    # ════════════════════════════════════════════════════════
    # Internal helpers
    # ════════════════════════════════════════════════════════

    def _build_sim_config(
        self, var: ConfigVariation, seed: int,
    ) -> SimConfig:
        """Merge experiment defaults with per-variation overrides."""
        cfg = self._cfg
        return SimConfig(
            scenario=cfg.scenario,
            generations=var.generations or cfg.generations,
            pool_size=var.pool_size or cfg.pool_size,
            trades_per_generation=(var.trades_per_generation or
                                   cfg.trades_per_generation),
            starting_capital=var.starting_capital or cfg.starting_capital,
            seed=seed,
            survival_rate=var.survival_rate if var.survival_rate is not None
                          else cfg.survival_rate,
            elitism_count=var.elitism_count if var.elitism_count is not None
                          else cfg.elitism_count,
            mutation_rate=var.mutation_rate if var.mutation_rate is not None
                          else cfg.mutation_rate,
            risk_limits=var.risk_limits or cfg.risk_limits,
        )

    @staticmethod
    def _extract_gen_data(
        sim_results: SimulationResults,
    ) -> List[GenerationData]:
        """Convert SimulationResults → list of GenerationData for scorecard."""
        gen_data: List[GenerationData] = []
        for gr in sim_results.generation_results:
            diag = gr.diagnostics_dict
            div = diag.get("diversity", {})
            conc = diag.get("concentration", {})
            dom = diag.get("dominance", {})
            fit = diag.get("fitness", {})

            gen_data.append(GenerationData(
                generation=gr.generation,
                pool_capital=gr.pool_capital,
                pool_pnl=gr.pool_pnl,
                portfolio_state=gr.portfolio_state,
                best_fitness=gr.snapshot.best_fitness,
                avg_fitness=gr.snapshot.avg_fitness,
                worst_fitness=gr.snapshot.worst_fitness,
                total_trades=gr.snapshot.total_trades,
                pool_win_rate=gr.snapshot.pool_win_rate,
                pool_sharpe=gr.snapshot.pool_sharpe,
                pool_max_drawdown=gr.snapshot.pool_max_drawdown,
                survivors=gr.snapshot.survivors,
                eliminated=gr.snapshot.eliminated,
                population_size=gr.snapshot.population_size,
                diversity_score=div.get("overall_diversity_score", 0),
                mean_entropy=div.get("mean_entropy", 0),
                capital_gini=conc.get("capital_gini", 0),
                capital_herfindahl=conc.get("capital_herfindahl", 0),
                capital_top1_pct=conc.get("capital_top1_pct", 0),
                exposure_herfindahl=conc.get("exposure_herfindahl", 0),
                exposure_top1_pct=conc.get("exposure_top1_pct", 0),
                dominant_lineage_share=dom.get("dominant_lineage_share", 0),
                effective_lineages=dom.get("effective_lineages", 0),
                fitness_improving=fit.get("improving", False),
                fitness_stagnating=fit.get("stagnating", False),
                stagnation_generations=fit.get("stagnation_generations", 0),
                health_score=diag.get("health_score", 0),
                n_alerts=len(diag.get("alerts", [])),
                alerts=diag.get("alerts", []),
            ))
        return gen_data

    @staticmethod
    def _aggregate(
        runs: List[RunResult],
        variations: List[ConfigVariation],
    ) -> List[VariationSummary]:
        """Aggregate RunResults per config_id into VariationSummary."""

        # Group by config_id
        by_config: Dict[str, List[RunResult]] = {}
        for r in runs:
            by_config.setdefault(r.config_id, []).append(r)

        # Build config_id → variation params lookup
        var_lookup = {v.config_id: v.to_dict() for v in variations}

        summaries: List[VariationSummary] = []
        for cid, group in by_config.items():
            n = len(group)
            ecosystems = [r.ecosystem_health for r in group]
            returns = [r.capital_return_pct for r in group]

            grades: Dict[str, int] = {}
            for r in group:
                grades[r.ecosystem_grade] = grades.get(
                    r.ecosystem_grade, 0) + 1

            summaries.append(VariationSummary(
                config_id=cid,
                n_seeds=n,
                config_params=var_lookup.get(cid, {}),
                mean_ecosystem=_mean(ecosystems),
                mean_return_pct=_mean(returns),
                mean_risk=_mean([r.risk_stability for r in group]),
                mean_evolution=_mean([r.evolution_health for r in group]),
                mean_concentration=_mean(
                    [r.concentration_risk for r in group]),
                mean_shock=_mean([r.shock_resilience for r in group]),
                mean_learning=_mean([r.learning_quality for r in group]),
                mean_best_fitness=_mean(
                    [r.best_fitness for r in group]),
                std_ecosystem=_std(ecosystems),
                std_return_pct=_std(returns),
                min_return_pct=min(returns) if returns else 0.0,
                max_return_pct=max(returns) if returns else 0.0,
                min_ecosystem=min(ecosystems) if ecosystems else 0.0,
                max_ecosystem=max(ecosystems) if ecosystems else 0.0,
                grade_counts=grades,
            ))
        return summaries


# ═════════════════════════════════════════════════════════════
# Stats helpers
# ═════════════════════════════════════════════════════════════

def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)

def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    import statistics
    try:
        return statistics.stdev(values)
    except statistics.StatisticsError:
        return 0.0
