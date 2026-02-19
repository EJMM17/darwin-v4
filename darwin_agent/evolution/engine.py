"""
Darwin v4 — EvolutionEngine with RiskAwareFitness integration.

Layer 5 (evolution). Depends only on interfaces/ and evolution/fitness.

INTEGRATION CHANGES FROM LEGACY:
  1. evaluate_generation() now receives List[AgentEvalData] instead of
     List[AgentMetrics]. Each AgentEvalData bundles metrics + pnl_series
     + exposure + initial_capital + optional DNA.

  2. evaluate_generation() now accepts an optional PortfolioRiskMetrics
     snapshot from the risk engine. This is passed to RiskAwareFitness
     to compute portfolio-harmony and diversification penalties.

  3. Fitness is computed INSIDE the engine, not pre-baked in AgentMetrics.
     The agent's .fitness field is OVERWRITTEN with the authoritative
     risk-aware score. This is the single source of truth for ranking.

  4. FitnessBreakdown is stored per agent per generation. Accessible
     via get_fitness_breakdowns() for dashboard display.

  5. GenerationSnapshot.agent_rankings now includes a "fitness_breakdown"
     dict per agent for full transparency.

  6. All ranking, selection, and hall-of-fame use the risk-aware score.

  7. Zero legacy inline fitness formulas. RiskAwareFitness is the only
     scorer in the entire codebase.

Architecture remains clean:
  - EvolutionEngine depends on evolution/fitness (same package)
  - Does NOT depend on PortfolioRiskEngine (receives snapshot via arg)
  - Does NOT depend on DarwinAgent (receives AgentEvalData via arg)
  - Stateless fitness calculation, stateful only for generation counter,
    hall of fame, and pending DNA
"""
from __future__ import annotations

import logging
import random
import statistics
from darwin_agent.determinism import next_deterministic_id
from datetime import datetime, timezone
from typing import Dict, List, Tuple

# EventType reserved for future event bus integration
from darwin_agent.interfaces.types import (
    AgentEvalData, AgentMetrics, DNAData, GenerationSnapshot,
    PortfolioRiskMetrics,
)
from darwin_agent.evolution.fitness import (
    FitnessBreakdown, FitnessConfig, RiskAwareFitness,
)

try:
    from darwin_agent.evolution.archive import EliteArchive
except ImportError:
    EliteArchive = None  # type: ignore

logger = logging.getLogger("darwin.evolution")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

def _new_id() -> str:
    return next_deterministic_id()


# ── Default gene ranges ─────────────────────────────────────

DEFAULT_GENE_RANGES: Dict[str, Tuple[float, float]] = {
    "risk_pct":            (0.5, 8.0),
    "stop_loss_pct":       (0.5, 5.0),
    "take_profit_pct":     (1.0, 10.0),
    "trailing_stop_pct":   (0.3, 3.0),
    "trailing_activation": (0.5, 5.0),
    "cooldown_minutes":    (5.0, 60.0),
    "max_consec_losses":   (3.0, 10.0),
    "momentum_weight":     (0.0, 1.0),
    "mean_rev_weight":     (0.0, 1.0),
    "scalping_weight":     (0.0, 1.0),
    "breakout_weight":     (0.0, 1.0),
    "confidence_threshold":(0.3, 0.9),
    "leverage_aggression": (0.3, 1.0),
    # ── Structural strategy genes ────────────────────────
    # regime_bias: -1=pure mean-revert, 0=neutral, +1=pure trend-follow
    "regime_bias":         (-1.0, 1.0),
    # volatility_threshold: skip entry when simulated vol is below this
    # 0.0=always enter, 1.0=only enter on extreme vol
    "volatility_threshold":(0.0, 1.0),
    # timeframe_bias: categorical encoded as float
    #   [0.0, 0.33) = 1m (fast scalper, short holds)
    #   [0.33, 0.66) = 5m (medium swing)
    #   [0.66, 1.0]  = 15m (patient position)
    "timeframe_bias":      (0.0, 1.0),
    # ── Technical indicator signal weights ───────────────
    "ema_weight":          (0.0, 1.0),
    "rsi_weight":          (0.0, 1.0),
    "atr_weight":          (0.0, 1.0),
    "trend_strength_weight":(0.0, 1.0),
    "volatility_weight":   (0.0, 1.0),
    # ── Macro regime ────────────────────────────────────
    "macro_regime_weight": (-1.0, 1.0),
    # ── Regime gate thresholds ──────────────────────────
    "regime_trend_gate":   (0.005, 0.08),   # |EMA50-EMA200|/price threshold
    "regime_vol_gate":     (0.002, 0.04),   # ATR/price threshold
}

# ── Scored agent (internal) ─────────────────────────────────

class _ScoredAgent:
    """Agent with risk-aware fitness attached. Internal to engine."""
    __slots__ = ("eval_data", "fitness", "breakdown")

    def __init__(
        self,
        eval_data: AgentEvalData,
        fitness: float,
        breakdown: FitnessBreakdown,
    ):
        self.eval_data = eval_data
        self.fitness = fitness
        self.breakdown = breakdown

    @property
    def metrics(self) -> AgentMetrics:
        return self.eval_data.metrics

    @property
    def agent_id(self) -> str:
        return self.eval_data.metrics.agent_id


# ═════════════════════════════════════════════════════════════
# EvolutionEngine
# ═════════════════════════════════════════════════════════════

class EvolutionEngine:
    """
    Implements IEvolutionEngine protocol.

    Lifecycle per generation:
      1. All agents trade for N trades (generation_trade_limit)
      2. evaluate_generation() RE-SCORES agents using RiskAwareFitness
      3. select_survivors() picks top performers
      4. breed() + mutate() creates offspring for next gen
      5. Snapshot persisted to Postgres via IGenerationRepository

    The engine is stateless w.r.t. trading — it only processes
    AgentEvalData and DNAData. The AgentManager orchestrates the
    actual spawn/kill cycle.
    """

    __slots__ = (
        "_generation", "_gene_ranges", "_hall_of_fame",
        "_elitism_count", "_tournament_size",
        "_base_mutation_rate", "_mutation_decay",
        "_crossover_rate", "_survival_rate",
        "_gen_repo", "_dna_repo",
        "_pending_dna", "_fitness_model",
        "_last_breakdowns",
    )

    def __init__(
        self,
        gene_ranges: Dict[str, Tuple[float, float]] | None = None,
        elitism_count: int = 1,
        tournament_size: int = 3,
        base_mutation_rate: float = 0.15,
        mutation_decay: float = 0.995,
        crossover_rate: float = 0.7,
        survival_rate: float = 0.5,
        gen_repo=None,          # IGenerationRepository (optional)
        dna_repo=None,          # IDNARepository (optional)
        fitness_config: FitnessConfig | None = None,
    ) -> None:
        self._generation = 0
        self._gene_ranges = gene_ranges or dict(DEFAULT_GENE_RANGES)
        self._hall_of_fame: List[DNAData] = []
        self._elitism_count = elitism_count
        self._tournament_size = tournament_size
        self._base_mutation_rate = base_mutation_rate
        self._mutation_decay = mutation_decay
        self._crossover_rate = crossover_rate
        self._survival_rate = survival_rate
        self._gen_repo = gen_repo
        self._dna_repo = dna_repo
        self._pending_dna: List[DNAData] = []
        self._fitness_model = RiskAwareFitness(fitness_config)
        # Dashboard access: breakdown per agent for the most recent generation
        self._last_breakdowns: Dict[str, FitnessBreakdown] = {}

    # ════════════════════════════════════════════════════════
    # IEvolutionEngine protocol
    # ════════════════════════════════════════════════════════

    async def flush_pending_dna(self) -> int:
        """Persist any DNA queued by create_next_generation."""
        if not self._dna_repo or not self._pending_dna:
            return 0
        saved = 0
        for dna in self._pending_dna:
            try:
                await self._dna_repo.save_dna(dna)
                saved += 1
            except Exception as exc:
                logger.error("failed to persist DNA %s: %s", dna.dna_id, exc)
        self._pending_dna.clear()
        return saved

    def get_generation(self) -> int:
        return self._generation

    async def evaluate_generation(
        self,
        agents: List[AgentEvalData],
        portfolio_snapshot: PortfolioRiskMetrics | None = None,
    ) -> GenerationSnapshot:
        """
        Re-score all agents using RiskAwareFitness, rank by the
        authoritative risk-aware score, compute generation stats,
        persist snapshot, and advance the generation counter.

        Args:
            agents: List of AgentEvalData bundles. Each carries metrics,
                    pnl_series, exposure, initial_capital, and optional DNA.
            portfolio_snapshot: Current portfolio risk state from
                    PortfolioRiskEngine.get_portfolio_state(). If None,
                    portfolio-harmony and diversification score neutral.

        Returns:
            GenerationSnapshot with fitness_breakdown in each ranking entry.
        """
        if not agents:
            return GenerationSnapshot(generation=self._generation)

        # ── 1. Score every agent using RiskAwareFitness ──────
        scored = self._score_all(agents, portfolio_snapshot)

        # ── 2. Rank by risk-aware fitness (descending) ───────
        ranked = sorted(scored, key=lambda s: s.fitness, reverse=True)
        fitnesses = [s.fitness for s in ranked]

        # ── 3. Store breakdowns for dashboard ────────────────
        self._last_breakdowns = {
            s.agent_id: s.breakdown for s in ranked
        }

        # ── 4. Determine survivors/eliminated ────────────────
        n_survive = max(1, int(len(ranked) * self._survival_rate))
        survivors = ranked[:n_survive]
        eliminated = ranked[n_survive:]
        survivor_ids = frozenset(s.agent_id for s in survivors)

        # ── 5. Build snapshot with enriched rankings ─────────
        total_trades = sum(s.metrics.total_trades for s in ranked)
        total_wins = sum(s.metrics.winning_trades for s in ranked)

        snapshot = GenerationSnapshot(
            generation=self._generation,
            population_size=len(ranked),
            best_fitness=fitnesses[0],
            avg_fitness=statistics.mean(fitnesses),
            worst_fitness=fitnesses[-1],
            best_agent_id=ranked[0].agent_id,
            total_trades=total_trades,
            total_pnl=sum(s.metrics.realized_pnl for s in ranked),
            pool_win_rate=total_wins / max(total_trades, 1),
            pool_sharpe=statistics.mean(
                [s.metrics.sharpe_ratio for s in ranked]
            ) if ranked else 0.0,
            pool_max_drawdown=max(
                s.metrics.max_drawdown_pct for s in ranked
            ) if ranked else 0.0,
            survivors=len(survivors),
            eliminated=len(eliminated),
            agent_rankings=[
                {
                    "rank": i + 1,
                    "agent_id": s.agent_id,
                    "fitness": round(s.fitness, 4),
                    "pnl": round(s.metrics.realized_pnl, 2),
                    "trades": s.metrics.total_trades,
                    "win_rate": round(s.metrics.win_rate * 100, 1),
                    "survived": s.agent_id in survivor_ids,
                    "fitness_breakdown": s.breakdown.to_dict(),
                }
                for i, s in enumerate(ranked)
            ],
            metadata={
                "portfolio_state": (
                    portfolio_snapshot.risk_state.value
                    if portfolio_snapshot else "UNKNOWN"
                ),
                "portfolio_drawdown_pct": (
                    round(portfolio_snapshot.drawdown_pct, 2)
                    if portfolio_snapshot else 0.0
                ),
                "portfolio_correlation_risk": (
                    round(portfolio_snapshot.correlation_risk, 4)
                    if portfolio_snapshot else 0.0
                ),
            },
            started_at=min(s.metrics.timestamp for s in ranked),
            ended_at=_utcnow(),
        )

        # ── 6. Persist snapshot ──────────────────────────────
        if self._gen_repo:
            try:
                await self._gen_repo.save_snapshot(snapshot)
            except Exception as exc:
                logger.error("failed to persist generation snapshot: %s", exc)

        # ── 7. Update hall of fame with risk-aware scores ────
        self._update_hall_of_fame(ranked)

        self._generation += 1
        logger.info(
            "gen %d evaluated: pop=%d best=%.4f avg=%.4f worst=%.4f "
            "survivors=%d portfolio=%s",
            snapshot.generation, snapshot.population_size,
            snapshot.best_fitness, snapshot.avg_fitness,
            snapshot.worst_fitness, snapshot.survivors,
            snapshot.metadata.get("portfolio_state", "?"),
        )

        return snapshot

    def select_survivors(
        self,
        rankings: List[AgentMetrics],
        survival_rate: float = 0.0,
    ) -> List[str]:
        """
        Return agent_ids of survivors via elitism + tournament.

        NOTE: This method uses the .fitness field already on AgentMetrics.
        If called after evaluate_generation(), those fitness values are
        the risk-aware scores (evaluate_generation overwrites them).
        """
        rate = survival_rate or self._survival_rate
        if not rankings:
            return []

        ranked = sorted(rankings, key=lambda a: a.fitness, reverse=True)
        n_survive = max(1, int(len(ranked) * rate))

        # Elites always survive (deterministic — ranked is already sorted)
        survivors = []
        seen = set()
        for i in range(min(self._elitism_count, len(ranked))):
            aid = ranked[i].agent_id
            if aid not in seen:
                survivors.append(aid)
                seen.add(aid)

        # Tournament selection for remaining slots
        while len(survivors) < n_survive and len(survivors) < len(ranked):
            winner = self._tournament_select(ranked)
            if winner.agent_id not in seen:
                survivors.append(winner.agent_id)
                seen.add(winner.agent_id)

        return sorted(survivors)  # deterministic output order

    # ════════════════════════════════════════════════════════
    # Breeding + mutation (unchanged logic, clean interface)
    # ════════════════════════════════════════════════════════

    def breed(self, parent_a: DNAData, parent_b: DNAData) -> DNAData:
        """Two-point crossover between parents."""
        child_genes: Dict[str, float] = {}
        all_keys = sorted(set(parent_a.genes) | set(parent_b.genes))

        if len(all_keys) < 2 or random.random() > self._crossover_rate:
            base = parent_a if parent_a.fitness >= parent_b.fitness else parent_b
            child_genes = dict(base.genes)
        else:
            p1 = random.randint(0, len(all_keys) - 1)
            p2 = random.randint(p1, len(all_keys) - 1)
            for i, key in enumerate(all_keys):
                if p1 <= i <= p2:
                    child_genes[key] = parent_b.genes.get(
                        key, parent_a.genes.get(key, 0.5))
                else:
                    child_genes[key] = parent_a.genes.get(
                        key, parent_b.genes.get(key, 0.5))

        return DNAData(
            genes=child_genes,
            generation=self._generation,
            parent_id=parent_a.dna_id,
            dna_id=_new_id(),
        )

    def mutate(self, dna: DNAData, mutation_rate: float = 0.0) -> DNAData:
        """Gaussian mutation with adaptive rate decay."""
        rate = mutation_rate or (
            self._base_mutation_rate * (self._mutation_decay ** self._generation)
        )
        mutated_genes = dict(dna.genes)

        for gene_name in sorted(mutated_genes.keys()):
            value = mutated_genes[gene_name]
            if random.random() < rate:
                lo, hi = self._gene_ranges.get(gene_name, (0.0, 1.0))
                spread = (hi - lo) * 0.15
                noise = random.gauss(0, spread)
                mutated_genes[gene_name] = max(lo, min(hi, value + noise))

        return DNAData(
            genes=mutated_genes,
            generation=dna.generation,
            parent_id=dna.parent_id,
            dna_id=dna.dna_id,
            fitness=dna.fitness,
            birth_time=dna.birth_time,
        )

    # ════════════════════════════════════════════════════════
    # Seeding + generation creation
    # ════════════════════════════════════════════════════════

    # ── Cluster-aware gene biases ────────────────────────────
    # Narrow ranges for initial DNA based on asset structural cluster.
    # Does NOT override gene limits — clamps to DEFAULT_GENE_RANGES.
    # Bias is stochastic: samples from narrowed sub-range, not fixed.
    CLUSTER_BIASES: Dict[str, Dict[str, Tuple[float, float]]] = {
        "momentum-volatile": {
            "momentum_weight":  (0.6, 1.0),
            "breakout_weight":  (0.5, 1.0),
            "regime_bias":      (0.2, 1.0),
            "trend_strength_weight": (0.4, 1.0),
            "ema_weight":       (0.3, 1.0),
        },
        "mean-reverting": {
            "mean_rev_weight":  (0.6, 1.0),
            "volatility_threshold": (0.0, 0.5),
            "regime_bias":      (-1.0, 0.0),
            "scalping_weight":  (0.3, 1.0),
            "rsi_weight":       (0.4, 1.0),
        },
        "regime-switching": {},   # BTC-like: no bias, full exploration
        "range-bound": {
            "mean_rev_weight":  (0.4, 0.9),
            "scalping_weight":  (0.4, 1.0),
            "regime_bias":      (-0.5, 0.3),
        },
        "trending-stable": {
            "momentum_weight":  (0.5, 1.0),
            "ema_weight":       (0.4, 1.0),
            "trend_strength_weight": (0.3, 0.9),
        },
        "high-liquidity": {},     # no bias
        "low-liquidity": {
            "risk_pct":         (0.5, 4.0),  # smaller positions
        },
        "mixed": {},              # default cluster: no bias
    }

    def create_random_dna(self, cluster_type: str | None = None) -> DNAData:
        """Generate a random genome within gene ranges.

        Parameters
        ----------
        cluster_type : str, optional
            Asset cluster archetype from AssetProfiler (e.g. "momentum-volatile",
            "mean-reverting"). When provided, narrows initial sampling ranges for
            cluster-relevant genes. All genes stay within DEFAULT_GENE_RANGES.
        """
        biases = self.CLUSTER_BIASES.get(cluster_type or "", {})
        genes = {}
        for name, (lo, hi) in self._gene_ranges.items():
            if name in biases:
                b_lo, b_hi = biases[name]
                # Clamp bias to gene limits
                eff_lo = max(lo, b_lo)
                eff_hi = min(hi, b_hi)
                if eff_lo > eff_hi:
                    eff_lo, eff_hi = lo, hi  # fallback to full range
            else:
                eff_lo, eff_hi = lo, hi
            genes[name] = eff_lo + random.random() * (eff_hi - eff_lo)
        return DNAData(
            genes=genes,
            generation=self._generation,
            dna_id=_new_id(),
        )

    def seed_initial_population(
        self,
        pool_size: int,
        archive: "EliteArchive | None" = None,
        rng: "random.Random | None" = None,
        cluster_type: str | None = None,
    ) -> List[DNAData]:
        """
        Create generation-0 population.

        If archive is provided and non-empty:
          - 20% of pool seeded from archive elites (with reduced mutation)
          - 80% random DNA
        If archive is None or empty:
          - 100% random DNA (same as before)

        Deterministic: uses provided rng for archive selection,
        falls back to module-level random for DNA generation.
        """
        if archive is None or archive.size == 0:
            return [self.create_random_dna(cluster_type) for _ in range(pool_size)]

        selection_rng = rng or random.Random(42)

        # Calculate injection count
        n_inject = max(1, int(pool_size * archive.injection_ratio))
        # n_inject may be less than requested if archive has fewer entries

        # Select from archive (with deduplication + injection limits)
        selected = archive.select_for_injection(n_inject, selection_rng)

        # Convert selected entries to DNAData with reduced mutation
        elite_rate = self._base_mutation_rate * archive.mutation_dampen
        pool: List[DNAData] = []
        for entry in selected:
            dna = DNAData(
                genes=dict(entry.genes),
                generation=0,
                parent_id=entry.dna_id,
                dna_id=_new_id(),
                fitness=0.0,  # reset — must earn fitness fresh
            )
            # Apply reduced mutation to prevent identical clones
            dna = self.mutate(dna, mutation_rate=elite_rate)
            pool.append(dna)

        # Fill remaining slots with random DNA
        n_remaining = pool_size - len(pool)
        for _ in range(n_remaining):
            pool.append(self.create_random_dna())

        logger.info(
            "Population seeded: %d from archive (mut=%.3f) + %d random = %d total",
            len(selected), elite_rate, n_remaining, len(pool),
        )
        return pool

    def create_next_generation(
        self,
        survivor_dna: List[DNAData],
        target_size: int,
    ) -> List[DNAData]:
        """
        Produce a full new generation from survivors:
          1. Elites pass through unchanged
          2. Crossover pairs breed offspring
          3. Mutation applied to all non-elites
          4. Fill remaining slots with random DNA
        """
        if not survivor_dna:
            return [self.create_random_dna() for _ in range(target_size)]

        next_gen: List[DNAData] = []

        # Elites (best N pass through)
        elites = sorted(survivor_dna, key=lambda d: d.fitness, reverse=True)
        for e in elites[:self._elitism_count]:
            next_gen.append(DNAData(
                genes=dict(e.genes), generation=self._generation,
                parent_id=e.dna_id, dna_id=_new_id(), fitness=0.0,
            ))

        # Breed offspring
        while len(next_gen) < target_size:
            if len(survivor_dna) >= 2:
                pa = random.choice(survivor_dna)
                pb = random.choice(survivor_dna)
                child = self.breed(pa, pb)
                child = self.mutate(child)
            else:
                child = self.mutate(survivor_dna[0])
            next_gen.append(child)

        # Queue for persistence
        if self._dna_repo:
            for dna in next_gen:
                try:
                    self._pending_dna.append(dna)
                except Exception:
                    pass

        return next_gen[:target_size]

    # ════════════════════════════════════════════════════════
    # Dashboard access
    # ════════════════════════════════════════════════════════

    def get_fitness_breakdowns(self) -> Dict[str, FitnessBreakdown]:
        """
        Return fitness breakdowns from the most recent generation.
        Keys are agent_ids, values are FitnessBreakdown dataclasses.
        Dashboard can call breakdown.to_dict() for JSON serialization.
        """
        return dict(self._last_breakdowns)

    def get_fitness_breakdown(self, agent_id: str) -> FitnessBreakdown | None:
        """Get breakdown for a specific agent, or None."""
        return self._last_breakdowns.get(agent_id)

    def get_hall_of_fame(self) -> List[DNAData]:
        return list(self._hall_of_fame)

    @property
    def fitness_model(self) -> RiskAwareFitness:
        """Expose the fitness model for external callers (e.g. tests)."""
        return self._fitness_model

    # ════════════════════════════════════════════════════════
    # Internal — scoring
    # ════════════════════════════════════════════════════════

    def _score_all(
        self,
        agents: List[AgentEvalData],
        portfolio_snapshot: PortfolioRiskMetrics | None,
    ) -> List[_ScoredAgent]:
        """
        Compute risk-aware fitness for every agent.
        Pure function of inputs (stateless scoring).
        """
        scored: List[_ScoredAgent] = []
        for ed in agents:
            m = ed.metrics
            # Extract take_profit_pct from DNA genes for TP shaping
            tp_pct = 3.0  # default
            if ed.dna and ed.dna.genes:
                tp_pct = ed.dna.genes.get("take_profit_pct", 3.0)
            bd = self._fitness_model.compute_breakdown(
                realized_pnl=m.realized_pnl,
                initial_capital=ed.initial_capital or max(m.capital, 0.01),
                current_capital=m.capital,
                sharpe=m.sharpe_ratio,
                max_drawdown_pct=m.max_drawdown_pct,
                win_count=m.winning_trades,
                loss_count=m.losing_trades,
                pnl_series=ed.pnl_series,
                portfolio_snapshot=portfolio_snapshot,
                agent_exposure=ed.exposure or None,
                total_trades=m.total_trades,
                take_profit_pct=tp_pct,
                gross_profit=ed.gross_profit,
                gross_loss=ed.gross_loss,
                total_notional=ed.total_notional,
                trend_series=ed.trend_series,
                n_bars=ed.n_bars,
                bars_gated=ed.bars_gated,
            )
            # Overwrite the agent's fitness with the authoritative score
            m.fitness = bd.final_score
            scored.append(_ScoredAgent(
                eval_data=ed, fitness=bd.final_score, breakdown=bd,
            ))
        return scored

    # ════════════════════════════════════════════════════════
    # Internal — selection + hall of fame
    # ════════════════════════════════════════════════════════

    def _tournament_select(self, ranked: List[AgentMetrics]) -> AgentMetrics:
        pool = random.sample(ranked, min(self._tournament_size, len(ranked)))
        return max(pool, key=lambda a: a.fitness)

    def _update_hall_of_fame(self, ranked: List[_ScoredAgent]) -> None:
        """Track top 3 from each generation, keep global top 20."""
        for s in ranked[:3]:
            dna = s.eval_data.dna
            self._hall_of_fame.append(DNAData(
                dna_id=dna.dna_id if dna else _new_id(),
                genes=dict(dna.genes) if dna and dna.genes else {},
                generation=self._generation,
                fitness=s.fitness,
            ))
        self._hall_of_fame.sort(key=lambda d: d.fitness, reverse=True)
        self._hall_of_fame = self._hall_of_fame[:20]
