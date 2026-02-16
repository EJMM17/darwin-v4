"""
Darwin v4 — EliteArchive.

Maintains a ranked archive of top-performing genotypes across
completed simulation runs.  On new run start, seeds a fraction
of the initial population from the archive with reduced mutation,
keeping the majority random to prevent archive domination.

USAGE:

    from darwin_agent.evolution.archive import EliteArchive

    # Create or load
    archive = EliteArchive(max_size=50)
    archive.load("archive.json")

    # Seed initial population (inside EvolutionEngine)
    dna_pool = engine.seed_initial_population(
        pool_size=10, archive=archive)

    # After simulation completes, deposit results
    archive.deposit(dna_list, returns={dna_id: return_pct, ...})
    archive.save("archive.json")

DESIGN:
    - Standalone: zero imports from RiskEngine, Fitness, or Harness
    - Deterministic: seeding uses caller-provided RNG
    - Deduplication: gene-distance threshold prevents clones
    - Configurable: injection ratio, mutation dampening, max archive size
    - JSON-serializable for persistence across sessions
"""
from __future__ import annotations

import json
import logging
import math
import random as _random_module
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("darwin.archive")


# ═════════════════════════════════════════════════════════════
# Archive entry
# ═════════════════════════════════════════════════════════════

@dataclass
class ArchiveEntry:
    """One stored genotype with performance metadata."""
    dna_id: str = ""
    genes: Dict[str, float] = field(default_factory=dict)
    fitness: float = 0.0
    return_pct: float = 0.0
    source_seed: int = 0
    source_generation: int = 0
    injection_count: int = 0  # how many times this entry was injected

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dna_id": self.dna_id,
            "genes": dict(self.genes),
            "fitness": round(self.fitness, 6),
            "return_pct": round(self.return_pct, 4),
            "source_seed": self.source_seed,
            "source_generation": self.source_generation,
            "injection_count": self.injection_count,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ArchiveEntry":
        return ArchiveEntry(
            dna_id=d.get("dna_id", ""),
            genes=d.get("genes", {}),
            fitness=d.get("fitness", 0.0),
            return_pct=d.get("return_pct", 0.0),
            source_seed=d.get("source_seed", 0),
            source_generation=d.get("source_generation", 0),
            injection_count=d.get("injection_count", 0),
        )


# ═════════════════════════════════════════════════════════════
# Gene distance (deduplication)
# ═════════════════════════════════════════════════════════════

def gene_distance(
    a: Dict[str, float],
    b: Dict[str, float],
    gene_ranges: Dict[str, Tuple[float, float]],
) -> float:
    """
    Normalized Euclidean distance between two genomes.
    Each gene is scaled to [0,1] by its range before distance calc.
    Returns value in [0, 1]: 0 = identical, 1 = maximally different.
    """
    all_keys = sorted(set(a) | set(b))
    if not all_keys:
        return 0.0

    sum_sq = 0.0
    for key in all_keys:
        va = a.get(key, 0.0)
        vb = b.get(key, 0.0)
        lo, hi = gene_ranges.get(key, (0.0, 1.0))
        span = hi - lo if hi > lo else 1.0
        diff = (va - vb) / span
        sum_sq += diff * diff

    return math.sqrt(sum_sq / len(all_keys))


# ═════════════════════════════════════════════════════════════
# EliteArchive
# ═════════════════════════════════════════════════════════════

class EliteArchive:
    """
    Cross-run genotype memory.

    Maintains a ranked list of top-performing genotypes.
    Provides seeding for new simulation runs with deduplication
    and injection-count tracking to prevent archive domination.

    Parameters:
        max_size:           Maximum entries in archive (default 50)
        dedupe_threshold:   Minimum gene distance to consider genotypes
                            different (default 0.05). Below this,
                            new entries are merged with existing ones.
        injection_ratio:    Fraction of initial population to seed from
                            archive (default 0.2 = 20%).
        mutation_dampen:    Mutation rate multiplier for archived elites
                            (default 0.3 = 30% of base rate).
        max_injections:     Max times a single entry can be injected
                            before it's skipped (default 3).
        gene_ranges:        Gene range dict for distance normalization.
                            If None, uses DEFAULT_GENE_RANGES.
    """

    __slots__ = (
        "_entries", "_max_size", "_dedupe_threshold",
        "_injection_ratio", "_mutation_dampen", "_max_injections",
        "_gene_ranges",
    )

    def __init__(
        self,
        max_size: int = 50,
        dedupe_threshold: float = 0.05,
        injection_ratio: float = 0.2,
        mutation_dampen: float = 0.3,
        max_injections: int = 3,
        gene_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        self._entries: List[ArchiveEntry] = []
        self._max_size = max_size
        self._dedupe_threshold = dedupe_threshold
        self._injection_ratio = injection_ratio
        self._mutation_dampen = mutation_dampen
        self._max_injections = max_injections

        if gene_ranges is None:
            from darwin_agent.evolution.engine import DEFAULT_GENE_RANGES
            self._gene_ranges = dict(DEFAULT_GENE_RANGES)
        else:
            self._gene_ranges = dict(gene_ranges)

    # ════════════════════════════════════════════════════════
    # Properties
    # ════════════════════════════════════════════════════════

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> List[ArchiveEntry]:
        return list(self._entries)

    @property
    def injection_ratio(self) -> float:
        return self._injection_ratio

    @property
    def mutation_dampen(self) -> float:
        return self._mutation_dampen

    # ════════════════════════════════════════════════════════
    # Deposit: add genotypes after a completed run
    # ════════════════════════════════════════════════════════

    def deposit(
        self,
        genotypes: List[Dict[str, Any]],
        seed: int = 0,
    ) -> int:
        """
        Deposit genotypes from a completed simulation run.

        Each genotype dict should have:
            genes: Dict[str, float]
            fitness: float
            return_pct: float  (capital return percentage)
            generation: int (optional)
            dna_id: str (optional)

        Returns number of new entries actually added.
        """
        added = 0
        for g in genotypes:
            genes = g.get("genes", {})
            if not genes:
                continue

            fitness = g.get("fitness", 0.0)
            return_pct = g.get("return_pct", 0.0)
            dna_id = g.get("dna_id", "")
            generation = g.get("generation", 0)

            # Skip very low fitness
            if fitness <= 0:
                continue

            # Deduplication: check distance to all existing entries
            is_dupe = False
            for existing in self._entries:
                dist = gene_distance(genes, existing.genes, self._gene_ranges)
                if dist < self._dedupe_threshold:
                    # Merge: keep the better-performing version
                    if fitness > existing.fitness:
                        existing.genes = dict(genes)
                        existing.fitness = fitness
                        existing.return_pct = return_pct
                        existing.dna_id = dna_id
                        existing.source_seed = seed
                        existing.source_generation = generation
                    is_dupe = True
                    break

            if not is_dupe:
                self._entries.append(ArchiveEntry(
                    dna_id=dna_id,
                    genes=dict(genes),
                    fitness=fitness,
                    return_pct=return_pct,
                    source_seed=seed,
                    source_generation=generation,
                    injection_count=0,
                ))
                added += 1

        # Sort by fitness descending and trim to max_size
        self._entries.sort(key=lambda e: e.fitness, reverse=True)
        self._entries = self._entries[:self._max_size]

        logger.info(
            "Archive deposit: %d candidates → %d new (total %d/%d)",
            len(genotypes), added, len(self._entries), self._max_size,
        )
        return added

    # ════════════════════════════════════════════════════════
    # Select: pick entries for injection into a new run
    # ════════════════════════════════════════════════════════

    def select_for_injection(
        self,
        n_inject: int,
        rng: _random_module.Random,
    ) -> List[ArchiveEntry]:
        """
        Select entries for injection into a new population.

        Selection strategy:
            1. Filter out entries that exceeded max_injections
            2. From eligible entries, pick using fitness-proportional
               selection (roulette wheel) to maintain diversity
            3. Ensure no two selected entries are too similar
            4. Increment injection_count on selected entries

        Returns list of selected ArchiveEntry (may be shorter than
        n_inject if not enough eligible entries exist).
        """
        if not self._entries or n_inject <= 0:
            return []

        # 1. Filter eligible (not over-injected)
        eligible = [
            e for e in self._entries
            if e.injection_count < self._max_injections
        ]
        if not eligible:
            return []

        # 2. Fitness-proportional selection
        min_fit = min(e.fitness for e in eligible)
        weights = [max(e.fitness - min_fit + 0.01, 0.01) for e in eligible]
        total_w = sum(weights)
        probs = [w / total_w for w in weights]

        selected: List[ArchiveEntry] = []
        attempts = 0
        max_attempts = n_inject * 10

        while len(selected) < n_inject and attempts < max_attempts:
            attempts += 1

            # Roulette wheel selection
            r = rng.random()
            cumulative = 0.0
            pick = eligible[-1]
            for e, p in zip(eligible, probs):
                cumulative += p
                if r <= cumulative:
                    pick = e
                    break

            # 3. Check distance to already-selected entries
            too_close = False
            for s in selected:
                dist = gene_distance(
                    pick.genes, s.genes, self._gene_ranges)
                if dist < self._dedupe_threshold:
                    too_close = True
                    break

            if not too_close:
                selected.append(pick)

        # 4. Increment injection count
        for e in selected:
            e.injection_count += 1

        logger.info(
            "Archive injection: %d/%d selected from %d eligible",
            len(selected), n_inject, len(eligible),
        )
        return selected

    # ════════════════════════════════════════════════════════
    # Persistence: JSON save/load
    # ════════════════════════════════════════════════════════

    def save(self, path: str) -> str:
        """Save archive to JSON file."""
        data = {
            "version": 1,
            "max_size": self._max_size,
            "dedupe_threshold": self._dedupe_threshold,
            "injection_ratio": self._injection_ratio,
            "mutation_dampen": self._mutation_dampen,
            "max_injections": self._max_injections,
            "entries": [e.to_dict() for e in self._entries],
        }
        p = Path(path)
        with open(p, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Archive saved: %s (%d entries)", p, len(self._entries))
        return str(p)

    def load(self, path: str) -> int:
        """
        Load archive from JSON file.
        Returns number of entries loaded.
        Silently returns 0 if file doesn't exist.
        """
        p = Path(path)
        if not p.exists():
            logger.info("Archive file not found: %s (starting empty)", p)
            return 0

        with open(p) as f:
            data = json.load(f)

        self._entries = [
            ArchiveEntry.from_dict(e)
            for e in data.get("entries", [])
        ]
        # Sort to ensure consistency
        self._entries.sort(key=lambda e: e.fitness, reverse=True)
        self._entries = self._entries[:self._max_size]

        logger.info("Archive loaded: %s (%d entries)", p, len(self._entries))
        return len(self._entries)

    # ════════════════════════════════════════════════════════
    # Inspection
    # ════════════════════════════════════════════════════════

    def top(self, n: int = 5) -> List[ArchiveEntry]:
        """Return top N entries by fitness."""
        return self._entries[:n]

    def clear(self) -> None:
        """Remove all entries."""
        self._entries.clear()

    def summary(self) -> Dict[str, Any]:
        """Return archive summary stats."""
        if not self._entries:
            return {
                "size": 0, "max_size": self._max_size,
                "best_fitness": 0.0, "best_return": 0.0,
                "mean_fitness": 0.0, "mean_return": 0.0,
            }
        fitnesses = [e.fitness for e in self._entries]
        returns = [e.return_pct for e in self._entries]
        return {
            "size": len(self._entries),
            "max_size": self._max_size,
            "best_fitness": max(fitnesses),
            "best_return": max(returns),
            "mean_fitness": sum(fitnesses) / len(fitnesses),
            "mean_return": sum(returns) / len(returns),
            "unique_seeds": len({e.source_seed for e in self._entries}),
            "total_injections": sum(e.injection_count for e in self._entries),
        }
