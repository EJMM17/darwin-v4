"""
Darwin v4 — ExperimentRunner Tests.

Covers all 8 requirements:
  1. Accept config variations
  2. Accept seed list
  3. Batch simulations
  4. Structured results (config_id, seed, scenario, scorecard)
  5. Summary CSV export
  6. No trading logic
  7. Deterministic per seed
  8. Production-grade (edge cases, error handling)
"""
import asyncio
import csv
import inspect
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

from darwin_agent.simulation.experiment import (
    ExperimentRunner, ExperimentConfig, ConfigVariation,
    ExperimentResults, RunResult, VariationSummary,
)
from darwin_agent.simulation.harness import MonteCarloScenario

PASS = FAIL = 0

def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ✅ {name}")
    else:
        FAIL += 1; print(f"  ❌ {name}: {detail}")


# ═════════════════════════════════════════════════════════════
# 1. Accept config variations
# ═════════════════════════════════════════════════════════════

async def test_1_variations():
    print("\n🧪 1. Config variations")
    print("─" * 50)

    runner = ExperimentRunner(ExperimentConfig(
        scenario=MonteCarloScenario(symbols=["BTCUSDT"]),
        variations=[
            ConfigVariation(config_id="low_mut", mutation_rate=0.05),
            ConfigVariation(config_id="high_mut", mutation_rate=0.30),
        ],
        seeds=[42],
        generations=3,
        trades_per_generation=50,
    ))
    results = await runner.run()

    check("2 variations produced 2 runs", results.total_runs == 2)

    ids = {r.config_id for r in results.runs}
    check("Both config_ids present", ids == {"low_mut", "high_mut"})

    # Verify config params stored
    low = next(r for r in results.runs if r.config_id == "low_mut")
    high = next(r for r in results.runs if r.config_id == "high_mut")
    check("low_mut params stored",
          low.config_params.get("mutation_rate") == 0.05)
    check("high_mut params stored",
          high.config_params.get("mutation_rate") == 0.30)

    # Variations with different pool sizes
    runner2 = ExperimentRunner(ExperimentConfig(
        scenario=MonteCarloScenario(symbols=["BTCUSDT"]),
        variations=[
            ConfigVariation(config_id="small", pool_size=3),
            ConfigVariation(config_id="large", pool_size=10),
        ],
        seeds=[42],
        generations=3,
        trades_per_generation=50,
    ))
    r2 = await runner2.run()
    check("Pool size variations work", r2.total_runs == 2)
    print()


# ═════════════════════════════════════════════════════════════
# 2. Accept seed list
# ═════════════════════════════════════════════════════════════

async def test_2_seeds():
    print("🧪 2. Seed list")
    print("─" * 50)

    seeds = [42, 123, 456]
    runner = ExperimentRunner(ExperimentConfig(
        scenario=MonteCarloScenario(symbols=["BTCUSDT"]),
        variations=[ConfigVariation(config_id="test")],
        seeds=seeds,
        generations=3,
        trades_per_generation=50,
    ))
    results = await runner.run()

    check("3 seeds × 1 variation = 3 runs", results.total_runs == 3)

    run_seeds = [r.seed for r in results.runs]
    check("All seeds present", set(run_seeds) == set(seeds))

    # Different seeds should produce different results
    capitals = [r.final_capital for r in results.runs]
    check("Different seeds → different capitals",
          len(set(round(c, 2) for c in capitals)) > 1,
          f"capitals={[round(c, 2) for c in capitals]}")
    print()


# ═════════════════════════════════════════════════════════════
# 3. Batch simulations
# ═════════════════════════════════════════════════════════════

async def test_3_batch():
    print("🧪 3. Batch execution")
    print("─" * 50)

    runner = ExperimentRunner(ExperimentConfig(
        scenario=MonteCarloScenario(symbols=["BTCUSDT"]),
        variations=[
            ConfigVariation(config_id="A", mutation_rate=0.05),
            ConfigVariation(config_id="B", mutation_rate=0.15),
            ConfigVariation(config_id="C", mutation_rate=0.30),
        ],
        seeds=[42, 99],
        generations=3,
        trades_per_generation=50,
    ))
    results = await runner.run()

    check("3 configs × 2 seeds = 6 runs", results.total_runs == 6)
    check("Elapsed time tracked", results.total_elapsed_ms > 0)

    # All combinations present
    combos = {(r.config_id, r.seed) for r in results.runs}
    expected = {(c, s) for c in ["A", "B", "C"] for s in [42, 99]}
    check("All (config, seed) combinations present",
          combos == expected,
          f"missing: {expected - combos}")

    # Summaries aggregated
    check("3 variation summaries", len(results.summaries) == 3)
    for s in results.summaries:
        if s.n_seeds != 2:
            check(f"Summary {s.config_id}: 2 seeds", False,
                  f"got {s.n_seeds}")
            break
    else:
        check("All summaries have 2 seeds", True)

    # Leaderboard populated
    check("Leaderboard has 3 entries", len(results.leaderboard) == 3)
    check("Leaderboard ranked by health (desc)",
          results.summaries[0].mean_ecosystem >=
          results.summaries[-1].mean_ecosystem)
    print()


# ═════════════════════════════════════════════════════════════
# 4. Structured results
# ═════════════════════════════════════════════════════════════

async def test_4_structured():
    print("🧪 4. Structured results")
    print("─" * 50)

    runner = ExperimentRunner(ExperimentConfig(
        scenario=MonteCarloScenario(symbols=["BTCUSDT"]),
        variations=[ConfigVariation(config_id="test", mutation_rate=0.10)],
        seeds=[42],
        generations=5,
        trades_per_generation=80,
    ))
    results = await runner.run()
    run = results.runs[0]

    # RunResult fields
    check("config_id stored", run.config_id == "test")
    check("seed stored", run.seed == 42)
    check("scenario_type stored", run.scenario_type == "MonteCarloScenario")
    check("generations_run stored", run.generations_run == 5)
    check("starting_capital stored", run.starting_capital > 0)
    check("final_capital stored", isinstance(run.final_capital, float))
    check("capital_return_pct stored", isinstance(run.capital_return_pct, float))
    check("best_fitness stored", isinstance(run.best_fitness, float))
    check("elapsed_ms stored", run.elapsed_ms > 0)

    # Scorecard fields
    check("ecosystem_health in [0, 10]",
          0 <= run.ecosystem_health <= 10)
    check("ecosystem_grade valid",
          run.ecosystem_grade in {"A", "B", "C", "D", "F"})
    check("risk_stability in [0, 10]",
          0 <= run.risk_stability <= 10)
    check("evolution_health in [0, 10]",
          0 <= run.evolution_health <= 10)
    check("concentration_risk in [0, 10]",
          0 <= run.concentration_risk <= 10)
    check("shock_resilience in [0, 10]",
          0 <= run.shock_resilience <= 10)
    check("learning_quality in [0, 10]",
          0 <= run.learning_quality <= 10)

    # Full scorecard dict
    check("scorecard_dict has scores",
          "scores" in run.scorecard_dict)
    check("scorecard_dict has context",
          "context" in run.scorecard_dict)

    # Config params
    check("config_params has mutation_rate",
          run.config_params.get("mutation_rate") == 0.10)

    # VariationSummary
    summary = results.summaries[0]
    check("Summary has mean_ecosystem",
          0 <= summary.mean_ecosystem <= 10)
    check("Summary has grade_counts",
          isinstance(summary.grade_counts, dict))
    check("Summary has std_ecosystem",
          isinstance(summary.std_ecosystem, float))
    print()


# ═════════════════════════════════════════════════════════════
# 5. CSV/JSON export
# ═════════════════════════════════════════════════════════════

async def test_5_export():
    print("🧪 5. Export CSV/JSON")
    print("─" * 50)

    runner = ExperimentRunner(ExperimentConfig(
        scenario=MonteCarloScenario(symbols=["BTCUSDT"]),
        variations=[
            ConfigVariation(config_id="A", mutation_rate=0.05),
            ConfigVariation(config_id="B", mutation_rate=0.20),
        ],
        seeds=[42, 99],
        generations=3,
        trades_per_generation=50,
    ))
    results = await runner.run()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Per-run CSV
        csv_path = os.path.join(tmpdir, "runs.csv")
        results.export_csv(csv_path)
        check("Per-run CSV created", os.path.exists(csv_path))

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        check("CSV has 4 rows (2×2)", len(rows) == 4)
        check("CSV has config_id column", "config_id" in rows[0])
        check("CSV has seed column", "seed" in rows[0])
        check("CSV has ecosystem_health column",
              "ecosystem_health" in rows[0])
        check("CSV has return_pct column", "return_pct" in rows[0])
        check("CSV has param_mutation_rate column",
              "param_mutation_rate" in rows[0])

        # Summary CSV
        sum_path = os.path.join(tmpdir, "summary.csv")
        results.export_summary_csv(sum_path)
        check("Summary CSV created", os.path.exists(sum_path))

        with open(sum_path) as f:
            reader = csv.DictReader(f)
            sum_rows = list(reader)
        check("Summary has 2 rows (2 variations)", len(sum_rows) == 2)
        check("Summary has mean_ecosystem", "mean_ecosystem" in sum_rows[0])
        check("Summary has std_ecosystem", "std_ecosystem" in sum_rows[0])
        check("Summary has grades", "grades" in sum_rows[0])

        # JSON
        json_path = os.path.join(tmpdir, "results.json")
        results.export_json(json_path)
        check("JSON created", os.path.exists(json_path))

        with open(json_path) as f:
            data = json.load(f)
        check("JSON has meta", "meta" in data)
        check("JSON has leaderboard", "leaderboard" in data)
        check("JSON has summaries", "summaries" in data)
        check("JSON has runs", "runs" in data)
        check("JSON runs count = 4", len(data["runs"]) == 4)
        check("JSON each run has scorecard",
              all("scorecard" in r for r in data["runs"]))
    print()


# ═════════════════════════════════════════════════════════════
# 6. No trading logic
# ═════════════════════════════════════════════════════════════

def test_6_no_trading():
    print("🧪 6. No trading logic")
    print("─" * 50)

    source = open("darwin_agent/simulation/experiment.py").read()

    check("No BybitAdapter", "BybitAdapter" not in source)
    check("No ExchangeRouter", "ExchangeRouter" not in source)
    check("No aiohttp", "aiohttp" not in source)
    check("No OrderRequest", "OrderRequest" not in source)
    check("No Position class", "class Position" not in source)
    print()


# ═════════════════════════════════════════════════════════════
# 7. Deterministic per seed
# ═════════════════════════════════════════════════════════════

async def test_7_deterministic():
    print("🧪 7. Deterministic per seed")
    print("─" * 50)

    cfg = ExperimentConfig(
        scenario=MonteCarloScenario(symbols=["BTCUSDT"]),
        variations=[ConfigVariation(config_id="det_test")],
        seeds=[42],
        generations=5,
        trades_per_generation=80,
    )

    r1 = await ExperimentRunner(cfg).run()
    r2 = await ExperimentRunner(cfg).run()

    run1 = r1.runs[0]
    run2 = r2.runs[0]

    check("Same seed → same final capital",
          abs(run1.final_capital - run2.final_capital) < 0.01,
          f"r1={run1.final_capital:.2f}, r2={run2.final_capital:.2f}")
    check("Same seed → same ecosystem health",
          abs(run1.ecosystem_health - run2.ecosystem_health) < 0.01,
          f"r1={run1.ecosystem_health:.2f}, r2={run2.ecosystem_health:.2f}")
    check("Same seed → same best fitness",
          abs(run1.best_fitness - run2.best_fitness) < 0.001)
    check("Same seed → same risk score",
          abs(run1.risk_stability - run2.risk_stability) < 0.01)
    check("Same seed → same return pct",
          abs(run1.capital_return_pct - run2.capital_return_pct) < 0.01)
    print()


# ═════════════════════════════════════════════════════════════
# 8. Edge cases
# ═════════════════════════════════════════════════════════════

async def test_8_edge_cases():
    print("🧪 8. Edge cases")
    print("─" * 50)

    # Single variation, single seed
    r1 = await ExperimentRunner(ExperimentConfig(
        variations=[ConfigVariation(config_id="solo")],
        seeds=[1],
        generations=2,
        trades_per_generation=20,
    )).run()
    check("Single run completes", r1.total_runs == 1)
    check("Single summary", len(r1.summaries) == 1)

    # Many seeds, one variation (statistical test)
    r2 = await ExperimentRunner(ExperimentConfig(
        scenario=MonteCarloScenario(symbols=["BTCUSDT"]),
        variations=[ConfigVariation(config_id="multi_seed")],
        seeds=list(range(5)),
        generations=3,
        trades_per_generation=30,
    )).run()
    check("5 seeds → 5 runs", r2.total_runs == 5)
    summary = r2.summaries[0]
    check("Summary aggregates 5 seeds", summary.n_seeds == 5)
    check("Min return <= max return",
          summary.min_return_pct <= summary.max_return_pct)
    check("Min ecosystem <= max ecosystem",
          summary.min_ecosystem <= summary.max_ecosystem)

    # ConfigVariation with all overrides
    full_var = ConfigVariation(
        config_id="full",
        pool_size=3,
        survival_rate=0.4,
        elitism_count=2,
        mutation_rate=0.25,
        generations=2,
        trades_per_generation=20,
        starting_capital=200.0,
    )
    r3 = await ExperimentRunner(ExperimentConfig(
        variations=[full_var], seeds=[42],
    )).run()
    check("Full override variation works", r3.total_runs == 1)
    check("Starting capital overridden",
          r3.runs[0].starting_capital == 200.0)

    # Print leaderboard (no crash)
    results = await ExperimentRunner(ExperimentConfig(
        variations=[
            ConfigVariation(config_id="X"),
            ConfigVariation(config_id="Y"),
        ],
        seeds=[42],
        generations=2,
        trades_per_generation=20,
    )).run()
    results.print_leaderboard()
    check("print_leaderboard() works", True)
    print()


# ═════════════════════════════════════════════════════════════
# 9. Aggregation correctness
# ═════════════════════════════════════════════════════════════

async def test_9_aggregation():
    print("🧪 9. Aggregation correctness")
    print("─" * 50)

    runner = ExperimentRunner(ExperimentConfig(
        scenario=MonteCarloScenario(symbols=["BTCUSDT"]),
        variations=[ConfigVariation(config_id="agg_test")],
        seeds=[42, 99, 123],
        generations=3,
        trades_per_generation=50,
    ))
    results = await runner.run()

    summary = results.summaries[0]
    runs = results.runs

    # Manual mean check
    manual_mean_eco = sum(r.ecosystem_health for r in runs) / len(runs)
    check("Mean ecosystem matches manual calc",
          abs(summary.mean_ecosystem - manual_mean_eco) < 0.01,
          f"summary={summary.mean_ecosystem:.2f}, manual={manual_mean_eco:.2f}")

    manual_mean_ret = sum(r.capital_return_pct for r in runs) / len(runs)
    check("Mean return matches manual calc",
          abs(summary.mean_return_pct - manual_mean_ret) < 0.01)

    # Grade counts
    total_grades = sum(summary.grade_counts.values())
    check("Grade counts sum to n_seeds",
          total_grades == 3,
          f"got {total_grades}")
    print()


# ═════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════

async def main():
    global PASS, FAIL

    print("═" * 60)
    print("  🧪 EXPERIMENT RUNNER TESTS")
    print("═" * 60)

    sync_tests = [test_6_no_trading]
    async_tests = [
        test_1_variations, test_2_seeds, test_3_batch,
        test_4_structured, test_5_export, test_7_deterministic,
        test_8_edge_cases, test_9_aggregation,
    ]

    for t in sync_tests:
        try:
            t()
        except Exception as exc:
            FAIL += 1
            print(f"  ❌ CRASHED: {t.__name__}: {exc}")
            import traceback; traceback.print_exc()

    for t in async_tests:
        try:
            await t()
        except Exception as exc:
            FAIL += 1
            print(f"  ❌ CRASHED: {t.__name__}: {exc}")
            import traceback; traceback.print_exc()

    print("═" * 60)
    if FAIL == 0:
        print(f"  ✅ ALL {PASS} EXPERIMENT TESTS PASSED")
    else:
        print(f"  ❌ {FAIL} FAILED, {PASS} passed")
    print("═" * 60)
    return FAIL == 0


if __name__ == "__main__":
    ok = asyncio.run(main())
    sys.exit(0 if ok else 1)
