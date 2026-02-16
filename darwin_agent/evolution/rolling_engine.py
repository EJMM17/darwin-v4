"""
Darwin v4 — Rolling Quarterly Re-Evolution Engine (RQRE).

Periodically re-trains genotypes on recent market data to prevent
stale-genome decay in regime shifts.

Architecture:
    Every 90 days:
        1. Slice last 12 months of data as training set
        2. Slice last 30 days as internal validation set
        3. Run B3 × 400gen evolution per symbol (seeded)
        4. Filter candidates: PF ≥ 1.0, MaxDD ≤ 45%, trades ≥ 20
        5. Replace bottom 30% of active pool with top 30% candidates
        6. Keep 70% of existing pool untouched
        7. Log evolution hash for audit trail

Integration:
    rqre = RollingEvolutionEngine(symbols, cluster_types, config)
    # Each bar:
    updated = rqre.step_if_due(bar_index, market_data)
    if updated:
        genomes = rqre.active_genomes  # dict[symbol → genes]

Constraints:
    - Does NOT modify GMRT, PAE, RBE, or fitness logic
    - Deterministic via fixed seed per cycle
    - Zero dependencies on portfolio layers
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ── Configuration ────────────────────────────────────────────

@dataclass
class RQREConfig:
    """Tunable parameters for Rolling Re-Evolution."""
    re_evolution_interval: int = 90     # days between re-evolutions
    training_lookback_days: int = 365   # 12 months training window
    validation_days: int = 30           # internal validation window
    generations: int = 400              # GA generations per run
    seeds: Tuple[int, ...] = (42, 123, 456)  # B3 ensemble
    pool_size: int = 10                 # agents per GA run
    replace_bottom_pct: float = 0.30    # % of pool to replace
    select_top_pct: float = 0.30        # % of candidates to promote
    # Safety thresholds for candidate acceptance
    min_pf: float = 1.0
    max_dd: float = 0.45               # 45%
    min_trades: int = 20
    # Harness params
    starting_capital: float = 100.0
    mutation_rate: float = 0.18
    survival_rate: float = 0.40
    capital_floor_pct: float = 0.45
    maker_fee: float = 0.0002
    taker_fee: float = 0.0005
    slippage: float = 0.0002
    trades_per_generation: int = 50


# ── Evolution cycle log ──────────────────────────────────────

@dataclass
class EvolutionCycleLog:
    """Audit record of one re-evolution cycle."""
    cycle_id: int
    bar_triggered: int
    day_triggered: int
    symbol: str
    candidates_total: int
    candidates_passed: int
    replaced_count: int
    kept_count: int
    seed_hash: str                       # determinism proof
    old_fitness: float                   # avg fitness of replaced
    new_fitness: float                   # avg fitness of replacements
    training_bars: int
    validation_bars: int


# ── Per-symbol genome slot ───────────────────────────────────

@dataclass
class GenomeSlot:
    """One genome in the active pool for a symbol."""
    genes: Dict[str, float]
    fitness: float
    origin_cycle: int                    # which RQRE cycle created it
    locked: bool = False                 # if True, never replaced


# ── Train function type ──────────────────────────────────────
# Async callable: (csv_data, seed, cluster_type) → list[{genes, fitness}]
TrainFn = Callable


# ── Core engine ──────────────────────────────────────────────

class RollingEvolutionEngine:
    """
    Rolling Quarterly Re-Evolution Engine.

    Manages a pool of active genotypes per symbol. Every
    `re_evolution_interval` days, re-trains on recent data and
    selectively replaces underperformers.

    The engine is PASSIVE — it does not run training itself.
    Instead, it:
      1. Tells callers WHEN re-evolution is due
      2. Accepts candidate genotypes from external training
      3. Applies replacement logic deterministically
      4. Logs all decisions for audit

    For simulation, use `execute_cycle()` with a train function.
    For live, use `is_due()` + `submit_candidates()` + `apply_replacement()`.
    """

    def __init__(
        self,
        symbols: List[str],
        cluster_types: Dict[str, str],
        config: RQREConfig | None = None,
        initial_genomes: Dict[str, Dict[str, float]] | None = None,
    ) -> None:
        self._cfg = config or RQREConfig()
        self._symbols = list(symbols)
        self._cluster_types = dict(cluster_types)
        self._cycle_count = 0
        self._last_evolution_day = -self._cfg.re_evolution_interval  # trigger on first check
        self._logs: List[EvolutionCycleLog] = []

        # Active genome pools: symbol → list of GenomeSlot
        self._pools: Dict[str, List[GenomeSlot]] = {}
        for sym in symbols:
            if initial_genomes and sym in initial_genomes:
                self._pools[sym] = [GenomeSlot(
                    genes=dict(initial_genomes[sym]),
                    fitness=0.5,  # unknown initial fitness
                    origin_cycle=0,
                )]
            else:
                self._pools[sym] = []

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def logs(self) -> List[EvolutionCycleLog]:
        return list(self._logs)

    @property
    def active_genomes(self) -> Dict[str, Dict[str, float]]:
        """Best genome per symbol from active pool."""
        result = {}
        for sym in self._symbols:
            pool = self._pools[sym]
            if pool:
                best = max(pool, key=lambda s: s.fitness)
                result[sym] = dict(best.genes)
            else:
                result[sym] = {}
        return result

    def get_pool(self, symbol: str) -> List[GenomeSlot]:
        """Get full pool for a symbol."""
        return list(self._pools.get(symbol, []))

    # ── Due check ────────────────────────────────────────────

    def is_due(self, current_day: int) -> bool:
        """Check if re-evolution should trigger."""
        return (current_day - self._last_evolution_day) >= self._cfg.re_evolution_interval

    def bars_to_days(self, bars: int, bars_per_day: int = 6) -> int:
        """Convert bar count to day count."""
        return bars // bars_per_day

    # ── Step interface (simulation convenience) ──────────────

    def step_if_due(
        self,
        bar_index: int,
        bars_per_day: int = 6,
    ) -> bool:
        """
        Check if re-evolution is due at this bar.

        Returns True if a cycle should be triggered. Caller is
        responsible for running training and calling submit_candidates().

        Does NOT auto-train — keeps engine independent of harness.
        """
        day = bar_index // bars_per_day
        return self.is_due(day)

    # ── Candidate submission + replacement ───────────────────

    def submit_candidates(
        self,
        symbol: str,
        candidates: List[Dict[str, Any]],
        current_day: int,
    ) -> EvolutionCycleLog:
        """
        Submit trained candidates for one symbol and apply replacement.

        Parameters
        ----------
        symbol : str
        candidates : list of dicts with keys: genes, fitness, pf, max_dd, trades
        current_day : int

        Returns
        -------
        EvolutionCycleLog
        """
        cfg = self._cfg
        self._cycle_count += 1
        cycle_id = self._cycle_count

        # ── Filter candidates ────────────────────────────
        passed = []
        for c in candidates:
            if c.get("pf", 0) < cfg.min_pf:
                continue
            if c.get("max_dd", 1.0) > cfg.max_dd:
                continue
            if c.get("trades", 0) < cfg.min_trades:
                continue
            passed.append(c)

        # Sort by fitness descending
        passed.sort(key=lambda x: x.get("fitness", 0), reverse=True)

        # Select top N%
        n_select = max(1, int(len(passed) * cfg.select_top_pct)) if passed else 0
        selected = passed[:n_select]

        # ── Replacement logic ────────────────────────────
        pool = self._pools[symbol]

        # How many to replace
        n_replace = max(1, int(len(pool) * cfg.replace_bottom_pct)) if pool else 0
        n_replace = min(n_replace, len(selected))  # can't replace more than we have

        # Sort pool by fitness ascending (worst first)
        replaceable = [s for s in pool if not s.locked]
        replaceable.sort(key=lambda s: s.fitness)

        # Only replace if new genome is better than worst existing
        replaced_slots = []
        new_slots = []
        for i in range(n_replace):
            if i < len(replaceable) and i < len(selected):
                old = replaceable[i]
                new_candidate = selected[i]
                # Safety: only replace if new fitness > old fitness
                if new_candidate.get("fitness", 0) > old.fitness:
                    replaced_slots.append(old)
                    new_slots.append(GenomeSlot(
                        genes=dict(new_candidate["genes"]),
                        fitness=new_candidate["fitness"],
                        origin_cycle=cycle_id,
                    ))

        # Apply replacements
        for old_slot in replaced_slots:
            pool.remove(old_slot)
        pool.extend(new_slots)

        # If pool was empty, add all selected
        if not pool and selected:
            for c in selected:
                pool.append(GenomeSlot(
                    genes=dict(c["genes"]),
                    fitness=c.get("fitness", 0),
                    origin_cycle=cycle_id,
                ))

        self._pools[symbol] = pool
        self._last_evolution_day = current_day

        # ── Compute hash for determinism proof ───────────
        hash_input = json.dumps(
            [c.get("genes", {}) for c in selected],
            sort_keys=True,
        )
        seed_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

        # ── Log ──────────────────────────────────────────
        old_fit = (
            sum(s.fitness for s in replaced_slots) / len(replaced_slots)
            if replaced_slots else 0.0
        )
        new_fit = (
            sum(s.fitness for s in new_slots) / len(new_slots)
            if new_slots else 0.0
        )

        log = EvolutionCycleLog(
            cycle_id=cycle_id,
            bar_triggered=current_day * 6,  # approximate
            day_triggered=current_day,
            symbol=symbol,
            candidates_total=len(candidates),
            candidates_passed=len(passed),
            replaced_count=len(new_slots),
            kept_count=len(pool) - len(new_slots),
            seed_hash=seed_hash,
            old_fitness=round(old_fit, 4),
            new_fitness=round(new_fit, 4),
            training_bars=0,
            validation_bars=0,
        )
        self._logs.append(log)
        return log

    # ── High-level execute (for simulation) ──────────────────

    async def execute_cycle(
        self,
        current_day: int,
        train_fn: TrainFn,
        market_data: Dict[str, List[Dict]],
        bars_per_day: int = 6,
    ) -> Dict[str, EvolutionCycleLog]:
        """
        Run a full re-evolution cycle for all symbols.

        Parameters
        ----------
        current_day : int
            Current simulation day.
        train_fn : async callable
            (csv_data_dict, seed, cluster_type, config) → list[{genes, fitness, pf, max_dd, trades}]
        market_data : dict
            {symbol: list of OHLCV dicts} — full history available
        bars_per_day : int

        Returns
        -------
        Dict[symbol, EvolutionCycleLog]
        """
        cfg = self._cfg
        logs = {}

        for sym in self._symbols:
            data = market_data.get(sym, [])
            if not data:
                continue

            total_bars = len(data)
            train_bars = cfg.training_lookback_days * bars_per_day
            val_bars = cfg.validation_days * bars_per_day

            # Slice: train = [-12mo-30d : -30d], val = [-30d : now]
            end = total_bars
            val_start = max(0, end - val_bars)
            train_start = max(0, val_start - train_bars)

            train_slice = data[train_start:val_start]
            val_slice = data[val_start:end]

            if len(train_slice) < 200 or len(val_slice) < 30:
                continue

            # Run B3 ensemble training
            all_candidates = []
            ct = self._cluster_types.get(sym)
            for si, seed in enumerate(cfg.seeds):
                cycle_seed = seed + self._cycle_count * 1000 + si
                results = await train_fn(
                    train_data=train_slice,
                    val_data=val_slice,
                    seed=cycle_seed,
                    cluster_type=ct,
                    config=cfg,
                )
                all_candidates.extend(results)

            log = self.submit_candidates(sym, all_candidates, current_day)
            log.training_bars = len(train_slice)
            log.validation_bars = len(val_slice)
            logs[sym] = log

        return logs

    # ── Reporting ────────────────────────────────────────────

    def print_summary(self) -> str:
        lines = []
        lines.append("=" * 80)
        lines.append("  ROLLING QUARTERLY RE-EVOLUTION — Summary")
        lines.append("=" * 80)
        lines.append(f"\n  Total cycles: {self._cycle_count}")
        lines.append(f"  Interval: every {self._cfg.re_evolution_interval} days")

        for log in self._logs:
            lines.append(
                f"\n  Cycle {log.cycle_id} | Day {log.day_triggered} | {log.symbol}"
                f" | candidates={log.candidates_total} passed={log.candidates_passed}"
                f" | replaced={log.replaced_count} kept={log.kept_count}"
                f" | old_fit={log.old_fitness:.4f} new_fit={log.new_fitness:.4f}"
                f" | hash={log.seed_hash}"
            )

        for sym in self._symbols:
            pool = self._pools[sym]
            if pool:
                fits = [s.fitness for s in pool]
                lines.append(f"\n  {sym} pool: {len(pool)} genomes, "
                             f"fitness=[{min(fits):.4f}, {max(fits):.4f}]")

        out = "\n".join(lines)
        print(out)
        return out
