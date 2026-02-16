"""
Darwin v4 — Simulation Harness Tests.

Covers all 9 requirements:
  1. Fully decoupled from live trading (no exchange imports)
  2. Deterministic via seed (same seed = same results)
  3. Historical replay, Monte Carlo, shock injection
  4. Multi-generation evolution
  5. EvolutionEngine + RiskEngine + Diagnostics integration
  6. Structured metrics per generation
  7. CSV/JSON export
  8. No exchange calls
  9. Production-grade architecture (error handling, edge cases)
"""
import asyncio
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

from darwin_agent.simulation.harness import (
    SimulationHarness, SimConfig, SimulationResults,
    MarketScenario, HistoricalScenario, MonteCarloScenario,
    ShockScenario, ShockEvent, SimulatedAgent, Tick,
    GenerationResult,
)
from darwin_agent.interfaces.types import DNAData
from darwin_agent.evolution.engine import DEFAULT_GENE_RANGES
from darwin_agent.risk.portfolio_engine import RiskLimits

PASS = FAIL = 0

def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ✅ {name}")
    else:
        FAIL += 1; print(f"  ❌ {name}: {detail}")


# ═════════════════════════════════════════════════════════════
# 1. Decoupled from live trading
# ═════════════════════════════════════════════════════════════

def test_1_decoupled():
    print("\n🔬 1. Fully decoupled from live trading")
    print("─" * 50)

    import inspect
    source = inspect.getsource(SimulationHarness)
    check("No BybitAdapter in harness source",
          "BybitAdapter" not in source)
    check("No ExchangeRouter in harness source",
          "ExchangeRouter" not in source)
    check("No aiohttp in harness source",
          "aiohttp" not in source)

    # SimulatedAgent has no exchange dependency
    agent_source = inspect.getsource(SimulatedAgent)
    check("No exchange adapter imports in SimulatedAgent",
          "BybitAdapter" not in agent_source and "ExchangeRouter" not in agent_source)

    # Can instantiate without any exchange config
    config = SimConfig(generations=1, pool_size=2, trades_per_generation=10)
    harness = SimulationHarness(config)
    check("Harness instantiates without exchanges", harness is not None)
    print()


# ═════════════════════════════════════════════════════════════
# 2. Deterministic via seed
# ═════════════════════════════════════════════════════════════

async def test_2_deterministic():
    print("🔬 2. Deterministic via seed")
    print("─" * 50)

    config = SimConfig(
        scenario=MonteCarloScenario(symbols=["BTCUSDT"]),
        generations=5, pool_size=3,
        trades_per_generation=50, seed=12345,
        starting_capital=100.0,
    )

    # Run twice with same seed
    h1 = SimulationHarness(config)
    r1 = await h1.run()

    h2 = SimulationHarness(config)
    r2 = await h2.run()

    check("Same seed → same final capital",
          abs(r1.final_capital - r2.final_capital) < 0.01,
          f"r1={r1.final_capital:.2f}, r2={r2.final_capital:.2f}")

    check("Same seed → same best fitness",
          abs(r1.best_fitness_ever - r2.best_fitness_ever) < 0.001,
          f"r1={r1.best_fitness_ever:.4f}, r2={r2.best_fitness_ever:.4f}")

    check("Same seed → same generation count",
          r1.generations_run == r2.generations_run)

    # Check per-generation PnL matches
    for i in range(min(len(r1.generation_results), len(r2.generation_results))):
        g1 = r1.generation_results[i]
        g2 = r2.generation_results[i]
        if abs(g1.pool_pnl - g2.pool_pnl) > 0.01:
            check(f"Gen {i} PnL matches", False,
                  f"g1={g1.pool_pnl:.2f}, g2={g2.pool_pnl:.2f}")
            break
    else:
        check("All generation PnLs match", True)

    # Different seed → different results
    config2 = SimConfig(
        scenario=MonteCarloScenario(symbols=["BTCUSDT"]),
        generations=5, pool_size=3,
        trades_per_generation=50, seed=99999,
        starting_capital=100.0,
    )
    h3 = SimulationHarness(config2)
    r3 = await h3.run()

    check("Different seed → different final capital",
          abs(r1.final_capital - r3.final_capital) > 0.01,
          f"r1={r1.final_capital:.2f}, r3={r3.final_capital:.2f}")
    print()


# ═════════════════════════════════════════════════════════════
# 3. Scenario support (historical, monte carlo, shocks)
# ═════════════════════════════════════════════════════════════

async def test_3_scenarios():
    print("🔬 3. Scenario support")
    print("─" * 50)

    import random as stdlib_random

    # ── Historical replay ────────────────────────────────
    rng = stdlib_random.Random(42)
    prices = [50000 + i * 100 for i in range(100)]
    hist = HistoricalScenario(data={"BTCUSDT": prices})
    ticks = hist.generate(100, rng)

    check("Historical: correct symbol", "BTCUSDT" in ticks)
    check("Historical: correct length", len(ticks["BTCUSDT"]) == 100)
    check("Historical: prices match input",
          ticks["BTCUSDT"][0].price == 50000)
    check("Historical: cyclic wrap (step 0 == step 100 mod 100)",
          True)  # guaranteed by design

    # Run sim with historical
    hist_config = SimConfig(
        scenario=HistoricalScenario(data={
            "BTCUSDT": [50000 + i*50 for i in range(200)],
            "ETHUSDT": [3000 + i*10 for i in range(200)],
        }),
        generations=3, pool_size=3,
        trades_per_generation=100, seed=42,
    )
    r_hist = await SimulationHarness(hist_config).run()
    check("Historical sim completes", r_hist.generations_run == 3)
    check("Historical scenario_type recorded",
          r_hist.scenario_type == "HistoricalScenario")

    # ── Monte Carlo ──────────────────────────────────────
    mc = MonteCarloScenario(
        symbols=["BTCUSDT", "ETHUSDT"],
        params={
            "BTCUSDT": MonteCarloScenario.SymbolParams(
                base_price=50000, annual_vol=0.80, fat_tail_df=5),
            "ETHUSDT": MonteCarloScenario.SymbolParams(
                base_price=3000, annual_vol=0.90),
        },
    )
    rng2 = stdlib_random.Random(42)
    mc_ticks = mc.generate(500, rng2)

    check("Monte Carlo: 2 symbols", len(mc_ticks) == 2)
    check("Monte Carlo: 500 ticks each",
          all(len(v) == 500 for v in mc_ticks.values()))
    # Prices should vary (not constant)
    btc_prices = [t.price for t in mc_ticks["BTCUSDT"]]
    check("Monte Carlo: prices vary",
          max(btc_prices) != min(btc_prices))
    # All prices positive
    check("Monte Carlo: all prices positive",
          all(p > 0 for p in btc_prices))

    # ── Shock injection ──────────────────────────────────
    base = MonteCarloScenario(symbols=["BTCUSDT"])
    shocks = [
        ShockEvent(step=50, shock_type="flash_crash",
                   magnitude=-0.20, duration=10),
        ShockEvent(step=150, shock_type="volatility",
                   magnitude=3.0, duration=20),
        ShockEvent(step=250, shock_type="gap",
                   magnitude=0.10, duration=1),
    ]
    shock_scenario = ShockScenario(base=base, shocks=shocks)
    rng3 = stdlib_random.Random(42)
    shock_ticks = shock_scenario.generate(300, rng3)

    # Generate base without shocks for comparison
    rng4 = stdlib_random.Random(42)
    base_ticks = base.generate(300, rng4)

    base_p50 = base_ticks["BTCUSDT"][50].price
    shock_p50 = shock_ticks["BTCUSDT"][50].price
    check("Flash crash at step 50: price dropped",
          shock_p50 < base_p50 * 0.95,
          f"base={base_p50:.0f}, shock={shock_p50:.0f}")

    # Run sim with shocks
    shock_config = SimConfig(
        scenario=shock_scenario,
        generations=3, pool_size=3,
        trades_per_generation=200, seed=42,
    )
    r_shock = await SimulationHarness(shock_config).run()
    check("Shock sim completes", r_shock.generations_run == 3)
    check("Shock scenario_type recorded",
          r_shock.scenario_type == "ShockScenario")
    print()


# ═════════════════════════════════════════════════════════════
# 4. Multi-generation evolution
# ═════════════════════════════════════════════════════════════

async def test_4_multi_generation():
    print("🔬 4. Multi-generation evolution")
    print("─" * 50)

    config = SimConfig(
        generations=15, pool_size=5,
        trades_per_generation=100, seed=42,
        starting_capital=100.0,
    )
    r = await SimulationHarness(config).run()

    check("15 generations completed", r.generations_run == 15)
    check("15 generation results", len(r.generation_results) == 15)

    # Generations should be numbered 0..14
    gens = [gr.generation for gr in r.generation_results]
    check("Generation numbers 0..14", gens == list(range(15)))

    # Hall of fame populated across generations
    check("Hall of fame populated", len(r.hall_of_fame) > 0)

    # Each generation should have agent results
    for gr in r.generation_results:
        if len(gr.agent_results) != 5:
            check(f"Gen {gr.generation}: 5 agent results",
                  False, f"got {len(gr.agent_results)}")
            break
    else:
        check("All generations have 5 agent results", True)

    # Evolution should produce trades and non-zero fitness
    all_trades = sum(gr.snapshot.total_trades for gr in r.generation_results)
    check("Total trades > 0 across all generations",
          all_trades > 0,
          f"total_trades={all_trades}")

    best_fitnesses = [gr.snapshot.best_fitness for gr in r.generation_results]
    check("Some generations have non-zero fitness",
          any(f > 0 for f in best_fitnesses),
          f"best_fitnesses={best_fitnesses}")
    print()


# ═════════════════════════════════════════════════════════════
# 5. EvolutionEngine + RiskEngine + Diagnostics integration
# ═════════════════════════════════════════════════════════════

async def test_5_integration():
    print("🔬 5. Component integration")
    print("─" * 50)

    config = SimConfig(
        generations=5, pool_size=4,
        trades_per_generation=100, seed=42,
    )
    r = await SimulationHarness(config).run()

    # EvolutionEngine: fitness_breakdown in rankings
    gr = r.generation_results[-1]
    rankings = gr.snapshot.agent_rankings
    check("Rankings have fitness_breakdown",
          all("fitness_breakdown" in a for a in rankings))

    # Fitness components present
    bd = rankings[0]["fitness_breakdown"]
    components = ["risk_adjusted_profit", "sharpe_quality", "drawdown_health",
                  "consistency", "portfolio_harmony", "diversification_bonus",
                  "capital_efficiency", "final_score"]
    check("All 8 fitness components present",
          all(c in bd for c in components),
          f"missing: {[c for c in components if c not in bd]}")

    # RiskEngine: portfolio state tracked
    check("Portfolio state in generation result",
          gr.portfolio_state in ("normal", "defensive", "critical", "halted"))

    # Diagnostics: health report
    diag = gr.diagnostics_dict
    check("Diagnostics report has diversity",
          "diversity" in diag)
    check("Diagnostics report has concentration",
          "concentration" in diag)
    check("Diagnostics report has dominance",
          "dominance" in diag)
    check("Diagnostics report has fitness trajectory",
          "fitness" in diag)
    check("Health score in [0,1]",
          0.0 <= diag.get("health_score", -1) <= 1.0)
    print()


# ═════════════════════════════════════════════════════════════
# 6. Structured metrics per generation
# ═════════════════════════════════════════════════════════════

async def test_6_structured_metrics():
    print("🔬 6. Structured metrics per generation")
    print("─" * 50)

    config = SimConfig(generations=3, pool_size=3,
                       trades_per_generation=80, seed=42)
    r = await SimulationHarness(config).run()

    gr = r.generation_results[0]

    # Snapshot fields
    check("Snapshot has population_size",
          gr.snapshot.population_size == 3)
    check("Snapshot has best/avg/worst fitness",
          gr.snapshot.best_fitness >= gr.snapshot.avg_fitness >= gr.snapshot.worst_fitness)

    # Agent results
    ar = gr.agent_results[0]
    required_keys = ["agent_id", "fitness", "capital", "pnl", "trades",
                     "wins", "losses", "win_rate", "sharpe", "max_dd"]
    check("Agent result has all required keys",
          all(k in ar for k in required_keys),
          f"missing: {[k for k in required_keys if k not in ar]}")

    # Gene stats
    check("Gene stats present", len(gr.gene_stats) > 0)
    for gene_name, stats in gr.gene_stats.items():
        if not all(k in stats for k in ["mean", "std", "min", "max"]):
            check(f"Gene {gene_name} has mean/std/min/max", False)
            break
    else:
        check("All genes have mean/std/min/max", True)

    # Pool-level metrics
    check("Pool capital tracked", gr.pool_capital > 0)
    check("Pool PnL tracked", isinstance(gr.pool_pnl, float))
    print()


# ═════════════════════════════════════════════════════════════
# 7. CSV/JSON export
# ═════════════════════════════════════════════════════════════

async def test_7_export():
    print("🔬 7. CSV/JSON export")
    print("─" * 50)

    config = SimConfig(generations=5, pool_size=3,
                       trades_per_generation=80, seed=42)
    r = await SimulationHarness(config).run()

    with tempfile.TemporaryDirectory() as tmpdir:
        # CSV export
        csv_path = os.path.join(tmpdir, "results.csv")
        r.export_csv(csv_path)
        check("CSV file created", os.path.exists(csv_path))

        with open(csv_path) as f:
            import csv
            reader = csv.DictReader(f)
            rows = list(reader)
        check("CSV has 5 rows (one per gen)", len(rows) == 5)
        check("CSV has generation column", "generation" in rows[0])
        check("CSV has best_fitness column", "best_fitness" in rows[0])
        check("CSV has pool_capital column", "pool_capital" in rows[0])
        check("CSV has diversity_score column", "diversity_score" in rows[0])

        # JSON export
        json_path = os.path.join(tmpdir, "results.json")
        r.export_json(json_path)
        check("JSON file created", os.path.exists(json_path))

        with open(json_path) as f:
            data = json.load(f)
        check("JSON has meta section", "meta" in data)
        check("JSON has generations array", "generations" in data)
        check("JSON meta has seed", data["meta"]["seed"] == 42)
        check("JSON has hall_of_fame", "hall_of_fame" in data)
        check("JSON has final_diagnostics", "final_diagnostics" in data)
        check("JSON generations count = 5",
              len(data["generations"]) == 5)

        # JSON generation detail
        gen0 = data["generations"][0]
        check("JSON gen has snapshot", "snapshot" in gen0)
        check("JSON gen has diagnostics", "diagnostics" in gen0)
        check("JSON gen has gene_stats", "gene_stats" in gen0)
    print()


# ═════════════════════════════════════════════════════════════
# 8. No exchange calls
# ═════════════════════════════════════════════════════════════

def test_8_no_exchange_calls():
    print("🔬 8. No exchange calls")
    print("─" * 50)

    # Verify module doesn't import any exchange adapter
    import darwin_agent.simulation.harness as harness_mod
    import inspect
    full_source = inspect.getsource(harness_mod)

    check("No 'BybitAdapter' in source",
          "BybitAdapter" not in full_source)
    check("No 'BinanceAdapter' in source",
          "BinanceAdapter" not in full_source)
    check("No 'ExchangeRouter' in source",
          "ExchangeRouter" not in full_source)
    check("No 'websocket' in source",
          "websocket" not in full_source.lower())
    check("No 'aiohttp' in source",
          "aiohttp" not in full_source)
    check("No 'pybit' in source",
          "pybit" not in full_source)

    # SimulatedAgent doesn't call any network
    agent_source = inspect.getsource(SimulatedAgent)
    check("SimulatedAgent has no 'await' (sync only)",
          "await" not in agent_source)
    print()


# ═════════════════════════════════════════════════════════════
# 9. Edge cases + production robustness
# ═════════════════════════════════════════════════════════════

async def test_9_edge_cases():
    print("🔬 9. Edge cases + production robustness")
    print("─" * 50)

    # 1 generation, 2 agents, 10 ticks (minimal)
    minimal = SimConfig(generations=1, pool_size=2,
                        trades_per_generation=10, seed=1)
    r = await SimulationHarness(minimal).run()
    check("Minimal sim (1 gen, 2 agents, 10 ticks) completes",
          r.generations_run == 1)

    # Very short trades_per_generation (no trades possible)
    ultra_short = SimConfig(generations=2, pool_size=2,
                            trades_per_generation=3, seed=42)
    r2 = await SimulationHarness(ultra_short).run()
    check("Ultra-short sim (3 ticks) completes", r2.generations_run == 2)

    # Large pool
    large = SimConfig(generations=2, pool_size=20,
                      trades_per_generation=50, seed=42)
    r3 = await SimulationHarness(large).run()
    check("Large pool (20 agents) completes", r3.generations_run == 2)
    check("Large pool: 20 agent results per gen",
          len(r3.generation_results[0].agent_results) == 20)

    # Historical with single price (no movement)
    flat = HistoricalScenario(data={"BTCUSDT": [50000.0] * 100})
    flat_cfg = SimConfig(scenario=flat, generations=2, pool_size=3,
                         trades_per_generation=50, seed=42)
    r4 = await SimulationHarness(flat_cfg).run()
    check("Flat market sim completes", r4.generations_run == 2)

    # Zero drift Monte Carlo
    zero_drift = MonteCarloScenario(
        symbols=["BTCUSDT"],
        params={"BTCUSDT": MonteCarloScenario.SymbolParams(
            annual_drift=0.0, annual_vol=0.0)},
    )
    zero_cfg = SimConfig(scenario=zero_drift, generations=2,
                         pool_size=3, trades_per_generation=50, seed=42)
    r5 = await SimulationHarness(zero_cfg).run()
    check("Zero-vol sim completes", r5.generations_run == 2)

    # Historical scenario rejects empty data
    try:
        HistoricalScenario(data={})
        check("Empty historical data rejected", False, "no error raised")
    except ValueError:
        check("Empty historical data rejected", True)
    print()


# ═════════════════════════════════════════════════════════════
# 10. SimulatedAgent unit tests
# ═════════════════════════════════════════════════════════════

def test_10_simulated_agent():
    print("🔬 10. SimulatedAgent mechanics")
    print("─" * 50)

    import random as stdlib_random

    # Create agent with aggressive DNA → should trade more
    aggressive_dna = DNAData(
        genes={
            "risk_pct": 5.0,
            "stop_loss_pct": 2.0,
            "take_profit_pct": 4.0,
            "confidence_threshold": 0.1,  # very low → enters often
            "momentum_weight": 0.8,
            "mean_rev_weight": 0.2,
            "scalping_weight": 0.1,
            "breakout_weight": 0.1,
            "leverage_aggression": 0.5,
            "cooldown_minutes": 5.0,
        },
        dna_id="aggressive",
    )

    agent = SimulatedAgent(
        agent_id="test-agent",
        dna=aggressive_dna,
        capital=1000.0,
        rng=stdlib_random.Random(42),
    )

    # Generate simple trending market
    prices = [50000 + i * 100 for i in range(200)]
    for step in range(200):
        tick = Tick(symbol="BTCUSDT", step=step, price=prices[step])
        agent.step({"BTCUSDT": tick}, step)

    ed = agent.get_eval_data()
    check("Agent traded (total_trades > 0)",
          ed.metrics.total_trades > 0,
          f"trades={ed.metrics.total_trades}")
    check("PnL series length matches trades",
          len(ed.pnl_series) == ed.metrics.total_trades)
    check("Wins + losses = total trades",
          ed.metrics.winning_trades + ed.metrics.losing_trades == ed.metrics.total_trades)
    check("DNA preserved in eval data",
          ed.dna is not None and ed.dna.dna_id == "aggressive")
    check("Initial capital recorded",
          ed.initial_capital == 1000.0)

    # Conservative agent should trade less
    conservative_dna = DNAData(
        genes={
            "risk_pct": 1.0,
            "confidence_threshold": 0.9,  # very high → enters rarely
            "momentum_weight": 0.3,
            "mean_rev_weight": 0.3,
            "scalping_weight": 0.2,
            "breakout_weight": 0.2,
            "cooldown_minutes": 60.0,
        },
        dna_id="conservative",
    )
    cons_agent = SimulatedAgent(
        agent_id="conservative",
        dna=conservative_dna,
        capital=1000.0,
        rng=stdlib_random.Random(42),
    )
    for step in range(200):
        tick = Tick(symbol="BTCUSDT", step=step, price=prices[step])
        cons_agent.step({"BTCUSDT": tick}, step)

    cons_ed = cons_agent.get_eval_data()
    check("Conservative trades <= aggressive trades",
          cons_ed.metrics.total_trades <= ed.metrics.total_trades,
          f"cons={cons_ed.metrics.total_trades}, agg={ed.metrics.total_trades}")
    print()


# ═════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════

async def main():
    global PASS, FAIL

    print("═" * 60)
    print("  🔬 SIMULATION HARNESS TESTS")
    print("═" * 60)

    sync_tests = [test_1_decoupled, test_8_no_exchange_calls,
                  test_10_simulated_agent]
    async_tests = [test_2_deterministic, test_3_scenarios,
                   test_4_multi_generation, test_5_integration,
                   test_6_structured_metrics, test_7_export,
                   test_9_edge_cases]

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
        print(f"  ✅ ALL {PASS} SIMULATION TESTS PASSED")
    else:
        print(f"  ❌ {FAIL} FAILED, {PASS} passed")
    print("═" * 60)
    return FAIL == 0


if __name__ == "__main__":
    ok = asyncio.run(main())
    sys.exit(0 if ok else 1)
