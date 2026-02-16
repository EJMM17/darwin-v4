"""
Darwin v4 — EliteArchive Tests.

Covers all 7 requirements:
  1. Top N genotypes across runs
  2. Store genes + fitness + return
  3. Seed 20% from archive, 80% random, reduced mutation
  4. Prevent archive domination (dedup + injection limits)
  5. Deterministic behavior
  6. No RiskEngine or Fitness modification
  7. Clean EvolutionEngine integration
"""
import asyncio
import json
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

from darwin_agent.evolution.archive import (
    EliteArchive, ArchiveEntry, gene_distance,
)
from darwin_agent.evolution.engine import EvolutionEngine, DEFAULT_GENE_RANGES
from darwin_agent.interfaces.types import DNAData
from darwin_agent.simulation.harness import (
    SimulationHarness, SimConfig, MonteCarloScenario,
)

PASS = FAIL = 0

def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ✅ {name}")
    else:
        FAIL += 1; print(f"  ❌ {name}: {detail}")


def make_genotype(fitness, return_pct, gene_seed=0):
    """Helper: make a genotype dict with deterministic genes."""
    rng = random.Random(gene_seed)
    genes = {}
    for name, (lo, hi) in sorted(DEFAULT_GENE_RANGES.items()):
        genes[name] = lo + rng.random() * (hi - lo)
    return {
        "genes": genes,
        "fitness": fitness,
        "return_pct": return_pct,
        "generation": 100,
        "dna_id": f"test-{gene_seed}",
    }


# ═════════════════════════════════════════════════════════════
# 1. Top N genotypes across runs
# ═════════════════════════════════════════════════════════════

def test_1_top_n():
    print("\n🧪 1. Top N genotypes across runs")
    print("─" * 50)

    archive = EliteArchive(max_size=10)

    # Deposit from "run 1"
    entries_1 = [make_genotype(0.5 + i * 0.01, 10.0 + i, gene_seed=i)
                 for i in range(8)]
    added_1 = archive.deposit(entries_1, seed=42)
    check("Run 1: entries added", added_1 > 0)
    check("Archive size <= max_size", archive.size <= 10)

    # Deposit from "run 2"
    entries_2 = [make_genotype(0.6 + i * 0.01, 20.0 + i, gene_seed=100 + i)
                 for i in range(8)]
    added_2 = archive.deposit(entries_2, seed=123)
    check("Run 2: entries added", added_2 > 0)
    check("Archive capped at max_size", archive.size <= 10)

    # Best entries should be from run 2 (higher fitness)
    top = archive.top(3)
    check("Top entries are highest fitness",
          all(t.fitness >= 0.6 for t in top))

    # Entries from both seeds present
    seeds = {e.source_seed for e in archive.entries}
    check("Archive has entries from multiple runs",
          len(seeds) >= 2,
          f"seeds={seeds}")
    print()


# ═════════════════════════════════════════════════════════════
# 2. Store genes + fitness + return
# ═════════════════════════════════════════════════════════════

def test_2_storage():
    print("🧪 2. Store genes + fitness + return")
    print("─" * 50)

    archive = EliteArchive(max_size=5)
    g = make_genotype(0.75, 42.5, gene_seed=77)
    archive.deposit([g], seed=99)

    entry = archive.top(1)[0]
    check("Genes stored", len(entry.genes) == len(DEFAULT_GENE_RANGES))
    check("Fitness stored", entry.fitness == 0.75)
    check("Return stored", entry.return_pct == 42.5)
    check("Source seed stored", entry.source_seed == 99)
    check("DNA ID stored", entry.dna_id == "test-77")

    # to_dict / from_dict roundtrip
    d = entry.to_dict()
    restored = ArchiveEntry.from_dict(d)
    check("Roundtrip: genes match",
          restored.genes == entry.genes)
    check("Roundtrip: fitness match",
          restored.fitness == entry.fitness)
    check("Roundtrip: return match",
          restored.return_pct == entry.return_pct)
    print()


# ═════════════════════════════════════════════════════════════
# 3. Seed 20% from archive, reduced mutation, 80% random
# ═════════════════════════════════════════════════════════════

def test_3_seeding():
    print("🧪 3. Seeding ratio and reduced mutation")
    print("─" * 50)

    archive = EliteArchive(max_size=20, injection_ratio=0.2)

    # Fill archive with diverse entries
    for i in range(20):
        archive.deposit([make_genotype(0.5 + i * 0.02, 10.0 + i * 5,
                                       gene_seed=i * 37)], seed=i)

    check("Archive populated", archive.size >= 10)

    # Seed a pool of 10
    engine = EvolutionEngine()
    random.seed(42)
    pool = engine.seed_initial_population(10, archive=archive,
                                          rng=random.Random(42))

    check("Pool size = 10", len(pool) == 10)

    # At most 20% should come from archive (= 2 of 10)
    # We can detect archive-sourced entries by parent_id being set
    from_archive = [d for d in pool if d.parent_id is not None]
    from_random = [d for d in pool if d.parent_id is None]
    check(f"~20% from archive ({len(from_archive)}/10)",
          1 <= len(from_archive) <= 3,
          f"got {len(from_archive)}")
    check(f"~80% random ({len(from_random)}/10)",
          len(from_random) >= 7,
          f"got {len(from_random)}")

    # Archive entries should have been mutated (not identical to source)
    if from_archive:
        orig = archive.top(1)[0]
        injected = from_archive[0]
        # At least one gene should differ due to mutation
        diffs = sum(1 for g in orig.genes
                    if abs(orig.genes[g] - injected.genes.get(g, 0)) > 0.001)
        check("Elite DNA was mutated (not cloned)",
              diffs > 0,
              f"only {diffs} genes differ")

    # Fitness should be reset to 0
    for d in pool:
        check("Fitness reset to 0", d.fitness == 0.0)
        break  # just check one
    print()


# ═════════════════════════════════════════════════════════════
# 4. Prevent archive domination
# ═════════════════════════════════════════════════════════════

def test_4_domination():
    print("🧪 4. Prevent archive domination")
    print("─" * 50)

    # Deduplication: similar genotypes should merge
    archive = EliteArchive(max_size=10, dedupe_threshold=0.05)
    g1 = make_genotype(0.8, 50.0, gene_seed=1)
    g2 = dict(g1)  # identical genes
    g2["fitness"] = 0.7
    g2["dna_id"] = "clone"

    archive.deposit([g1], seed=1)
    archive.deposit([g2], seed=2)
    check("Duplicate merged (same genes)", archive.size == 1)
    check("Higher fitness kept", archive.top(1)[0].fitness == 0.8)

    # Injection count limit
    archive2 = EliteArchive(max_size=5, max_injections=2)
    for i in range(5):
        archive2.deposit([make_genotype(0.5 + i * 0.1, 20.0,
                                        gene_seed=i * 100)], seed=i)

    rng = random.Random(42)
    sel1 = archive2.select_for_injection(3, rng)
    sel2 = archive2.select_for_injection(3, random.Random(43))
    sel3 = archive2.select_for_injection(3, random.Random(44))

    # After 2 injections each, entries should be exhausted
    total_inj = sum(e.injection_count for e in archive2.entries)
    check("Injection counts tracked", total_inj > 0)

    # Gene distance function
    a = {"x": 0.0, "y": 0.0}
    b = {"x": 1.0, "y": 1.0}
    ranges = {"x": (0.0, 1.0), "y": (0.0, 1.0)}
    dist = gene_distance(a, b, ranges)
    check("Gene distance: opposite corners = 1.0",
          abs(dist - 1.0) < 0.01,
          f"got {dist}")

    dist_same = gene_distance(a, a, ranges)
    check("Gene distance: identical = 0.0",
          dist_same == 0.0)
    print()


# ═════════════════════════════════════════════════════════════
# 5. Deterministic behavior
# ═════════════════════════════════════════════════════════════

def test_5_deterministic():
    print("🧪 5. Deterministic behavior")
    print("─" * 50)

    archive = EliteArchive(max_size=10)
    for i in range(10):
        archive.deposit([make_genotype(0.5 + i * 0.05, 10.0,
                                       gene_seed=i * 50)], seed=i)

    engine = EvolutionEngine()

    # Two seeds with same RNG → same result
    random.seed(42)
    pool1 = engine.seed_initial_population(10, archive, random.Random(99))

    random.seed(42)
    pool2 = engine.seed_initial_population(10, archive, random.Random(99))

    genes1 = [d.genes for d in pool1]
    genes2 = [d.genes for d in pool2]

    match = True
    for g1, g2 in zip(genes1, genes2):
        for key in g1:
            if abs(g1.get(key, 0) - g2.get(key, 0)) > 0.0001:
                match = False
                break

    check("Same RNG → identical populations", match)

    # Different RNG → different result
    random.seed(42)
    pool3 = engine.seed_initial_population(10, archive, random.Random(777))

    diff = False
    for d1, d3 in zip(pool1, pool3):
        for key in d1.genes:
            if abs(d1.genes.get(key, 0) - d3.genes.get(key, 0)) > 0.001:
                diff = True
                break
        if diff:
            break
    check("Different RNG → different populations", diff)
    print()


# ═════════════════════════════════════════════════════════════
# 6. No RiskEngine or Fitness modification
# ═════════════════════════════════════════════════════════════

def test_6_no_risk_fitness():
    print("🧪 6. No RiskEngine or Fitness changes")
    print("─" * 50)

    source = open("darwin_agent/evolution/archive.py").read()
    check("No RiskEngine import", "PortfolioRiskEngine" not in source)
    check("No RiskLimits import", "RiskLimits" not in source)
    check("No RiskAwareFitness import", "RiskAwareFitness" not in source)
    check("No FitnessConfig import", "FitnessConfig" not in source)
    check("No risk_state reference", "risk_state" not in source)
    print()


# ═════════════════════════════════════════════════════════════
# 7. EvolutionEngine integration
# ═════════════════════════════════════════════════════════════

async def test_7_integration():
    print("🧪 7. EvolutionEngine + SimulationHarness integration")
    print("─" * 50)

    archive = EliteArchive(max_size=20)

    # Run 1: no archive (cold start)
    cfg1 = SimConfig(
        scenario=MonteCarloScenario(symbols=["BTCUSDT"]),
        generations=20, pool_size=5,
        trades_per_generation=100, seed=42,
        starting_capital=100.0,
        elite_archive=archive,
    )
    r1 = await SimulationHarness(cfg1).run()
    check("Run 1 completes", r1.generations_run == 20)
    check("Archive populated after run 1", archive.size > 0,
          f"size={archive.size}")

    size_after_r1 = archive.size

    # Run 2: with archive seeding
    cfg2 = SimConfig(
        scenario=MonteCarloScenario(symbols=["BTCUSDT"]),
        generations=20, pool_size=5,
        trades_per_generation=100, seed=123,
        starting_capital=100.0,
        elite_archive=archive,
    )
    r2 = await SimulationHarness(cfg2).run()
    check("Run 2 completes (with archive seeding)", r2.generations_run == 20)
    check("Archive grew after run 2", archive.size >= size_after_r1,
          f"before={size_after_r1}, after={archive.size}")

    # Run 3: different seed, same archive
    cfg3 = SimConfig(
        scenario=MonteCarloScenario(symbols=["BTCUSDT"]),
        generations=20, pool_size=5,
        trades_per_generation=100, seed=456,
        starting_capital=100.0,
        elite_archive=archive,
    )
    r3 = await SimulationHarness(cfg3).run()
    check("Run 3 completes", r3.generations_run == 20)

    # Archive summary
    summary = archive.summary()
    check("Summary has size", summary["size"] > 0)
    check("Summary tracks seeds",
          summary["unique_seeds"] >= 1)

    # Persistence: save and reload
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    archive.save(path)
    check("Archive saved", os.path.exists(path))

    archive2 = EliteArchive(max_size=20)
    loaded = archive2.load(path)
    check("Archive loaded", loaded == archive.size)
    check("Loaded entries match",
          archive2.top(1)[0].fitness == archive.top(1)[0].fitness)

    os.unlink(path)

    # No archive → pure random (backward compatible)
    cfg_none = SimConfig(
        scenario=MonteCarloScenario(symbols=["BTCUSDT"]),
        generations=5, pool_size=3,
        trades_per_generation=50, seed=42,
        starting_capital=100.0,
        elite_archive=None,
    )
    r_none = await SimulationHarness(cfg_none).run()
    check("No archive → still works", r_none.generations_run == 5)
    print()


# ═════════════════════════════════════════════════════════════
# 8. Edge cases
# ═════════════════════════════════════════════════════════════

def test_8_edge_cases():
    print("🧪 8. Edge cases")
    print("─" * 50)

    # Empty archive seeding
    archive = EliteArchive(max_size=5)
    engine = EvolutionEngine()
    random.seed(42)
    pool = engine.seed_initial_population(5, archive=archive)
    check("Empty archive → all random", len(pool) == 5)
    check("All have None parent_id",
          all(d.parent_id is None for d in pool))

    # Zero-fitness entries rejected
    archive2 = EliteArchive(max_size=5)
    added = archive2.deposit([{
        "genes": {"x": 0.5}, "fitness": 0.0, "return_pct": 0.0
    }])
    check("Zero fitness rejected", added == 0)

    # Load from nonexistent file
    archive3 = EliteArchive()
    loaded = archive3.load("/nonexistent/path.json")
    check("Missing file → 0 loaded", loaded == 0)

    # Clear
    archive4 = EliteArchive(max_size=5)
    archive4.deposit([make_genotype(0.5, 10.0, 1)])
    archive4.clear()
    check("Clear empties archive", archive4.size == 0)
    print()


# ═════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════

async def main():
    global PASS, FAIL
    print("═" * 60)
    print("  🧪 ELITE ARCHIVE TESTS")
    print("═" * 60)

    for t in [test_1_top_n, test_2_storage, test_3_seeding,
              test_4_domination, test_5_deterministic,
              test_6_no_risk_fitness, test_8_edge_cases]:
        try:
            t()
        except Exception as exc:
            FAIL += 1
            print(f"  ❌ CRASHED: {t.__name__}: {exc}")
            import traceback; traceback.print_exc()

    for t in [test_7_integration]:
        try:
            await t()
        except Exception as exc:
            FAIL += 1
            print(f"  ❌ CRASHED: {t.__name__}: {exc}")
            import traceback; traceback.print_exc()

    print("═" * 60)
    if FAIL == 0:
        print(f"  ✅ ALL {PASS} ARCHIVE TESTS PASSED")
    else:
        print(f"  ❌ {FAIL} FAILED, {PASS} passed")
    print("═" * 60)
    return FAIL == 0


if __name__ == "__main__":
    ok = asyncio.run(main())
    sys.exit(0 if ok else 1)
