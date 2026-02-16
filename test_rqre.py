"""Unit tests for Rolling Quarterly Re-Evolution Engine."""
import sys, asyncio, json, hashlib
sys.path.insert(0, ".")
from darwin_agent.evolution.rolling_engine import (
    RollingEvolutionEngine, RQREConfig, GenomeSlot, EvolutionCycleLog,
)

passed = 0
failed = 0

def test(name, condition):
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  ❌ {name}")


# ═══ 1. Initialization ══════════════════════════════════════

initial = {"BTC": {"risk_pct": 5.0, "momentum_weight": 0.8},
           "SOL": {"risk_pct": 3.0, "momentum_weight": 0.6}}
rqre = RollingEvolutionEngine(
    symbols=["BTC", "SOL"],
    cluster_types={"BTC": "regime-switching", "SOL": "momentum-volatile"},
    initial_genomes=initial,
)
test("init_cycle_zero", rqre.cycle_count == 0)
test("init_has_genomes", "BTC" in rqre.active_genomes)
test("init_genes_match", rqre.active_genomes["BTC"]["risk_pct"] == 5.0)
test("init_pool_size", len(rqre.get_pool("BTC")) == 1)


# ═══ 2. Due check — triggers every 90 days ══════════════════

rqre2 = RollingEvolutionEngine(
    symbols=["A"], cluster_types={"A": "none"},
    config=RQREConfig(re_evolution_interval=90),
)
test("due_at_day_0", rqre2.is_due(0))       # initial trigger (last=-90)
test("still_due_day_1", rqre2.is_due(1))    # no submit yet, still due
# Simulate submit at day 0 — this resets the counter
rqre2.submit_candidates("A", [], 0)
test("not_due_after_submit", not rqre2.is_due(1))  # now reset
test("not_due_day_30", not rqre2.is_due(30))
test("not_due_day_89", not rqre2.is_due(89))
test("due_day_90", rqre2.is_due(90))
test("due_day_180", rqre2.is_due(180))


# ═══ 3. Candidate filtering — PF, DD, trades ════════════════

rqre3 = RollingEvolutionEngine(
    symbols=["X"], cluster_types={"X": "none"},
    config=RQREConfig(min_pf=1.0, max_dd=0.45, min_trades=20),
    initial_genomes={"X": {"gene_a": 0.5}},
)
# Set initial pool with known fitness for replacement test
rqre3._pools["X"] = [
    GenomeSlot(genes={"gene_a": 0.1}, fitness=0.1, origin_cycle=0),
    GenomeSlot(genes={"gene_a": 0.2}, fitness=0.2, origin_cycle=0),
    GenomeSlot(genes={"gene_a": 0.3}, fitness=0.3, origin_cycle=0),
]

candidates = [
    {"genes": {"gene_a": 0.9}, "fitness": 0.9, "pf": 1.5, "max_dd": 0.30, "trades": 50},  # PASS
    {"genes": {"gene_a": 0.8}, "fitness": 0.8, "pf": 0.8, "max_dd": 0.20, "trades": 40},  # FAIL: PF
    {"genes": {"gene_a": 0.7}, "fitness": 0.7, "pf": 1.2, "max_dd": 0.50, "trades": 30},  # FAIL: DD
    {"genes": {"gene_a": 0.6}, "fitness": 0.6, "pf": 1.1, "max_dd": 0.10, "trades": 10},  # FAIL: trades
    {"genes": {"gene_a": 0.85}, "fitness": 0.85, "pf": 1.3, "max_dd": 0.25, "trades": 35}, # PASS
]

log = rqre3.submit_candidates("X", candidates, current_day=90)
test("filter_total", log.candidates_total == 5)
test("filter_passed", log.candidates_passed == 2)
test("cycle_incremented", rqre3.cycle_count == 1)
test("log_has_hash", len(log.seed_hash) == 16)


# ═══ 4. Replacement logic — exactly 30% ═════════════════════

rqre4 = RollingEvolutionEngine(
    symbols=["Y"], cluster_types={"Y": "none"},
    config=RQREConfig(replace_bottom_pct=0.30, select_top_pct=0.30),
)
# Pool of 10 with ascending fitness
pool_10 = [
    GenomeSlot(genes={"g": i/10}, fitness=i/10, origin_cycle=0)
    for i in range(1, 11)
]
rqre4._pools["Y"] = list(pool_10)

# Submit 10 candidates, all passing filters
big_candidates = [
    {"genes": {"g": 0.9+i*0.01}, "fitness": 0.95+i*0.001,
     "pf": 1.5, "max_dd": 0.20, "trades": 50}
    for i in range(10)
]
log4 = rqre4.submit_candidates("Y", big_candidates, current_day=90)

# 30% of 10 = 3 replaced, 30% of 10 candidates = 3 selected
test("replace_count_30pct", log4.replaced_count == 3)
test("kept_count_70pct", log4.kept_count == 10 - 3 + 3 - 3)  # 7 kept + 3 new - 3 replaced
# Pool should still be 10
test("pool_size_stable", len(rqre4.get_pool("Y")) == 10)
# Bottom 3 (fitness 0.1, 0.2, 0.3) should be gone
pool_fits = sorted(s.fitness for s in rqre4.get_pool("Y"))
test("worst_replaced", pool_fits[0] >= 0.4)  # old 0.1,0.2,0.3 gone


# ═══ 5. Safety: don't replace if new < old ══════════════════

rqre5 = RollingEvolutionEngine(
    symbols=["Z"], cluster_types={"Z": "none"},
)
rqre5._pools["Z"] = [
    GenomeSlot(genes={"g": 0.9}, fitness=0.9, origin_cycle=0),
]
# Candidate with lower fitness
weak = [{"genes": {"g": 0.1}, "fitness": 0.5, "pf": 1.2, "max_dd": 0.30, "trades": 30}]
log5 = rqre5.submit_candidates("Z", weak, current_day=90)
test("no_downgrade", log5.replaced_count == 0)
test("original_preserved", rqre5.get_pool("Z")[0].fitness == 0.9)


# ═══ 6. Determinism — same input → same output ══════════════

def run_replacement():
    r = RollingEvolutionEngine(
        symbols=["D"], cluster_types={"D": "none"},
        config=RQREConfig(replace_bottom_pct=0.50),
    )
    r._pools["D"] = [
        GenomeSlot(genes={"a": 0.1}, fitness=0.1, origin_cycle=0),
        GenomeSlot(genes={"a": 0.5}, fitness=0.5, origin_cycle=0),
    ]
    cands = [
        {"genes": {"a": 0.8}, "fitness": 0.8, "pf": 1.5, "max_dd": 0.20, "trades": 50},
        {"genes": {"a": 0.9}, "fitness": 0.9, "pf": 1.8, "max_dd": 0.15, "trades": 60},
    ]
    log = r.submit_candidates("D", cands, 90)
    return log.seed_hash, r.active_genomes["D"], log.replaced_count

h1, g1, r1 = run_replacement()
h2, g2, r2 = run_replacement()
test("determinism_hash", h1 == h2)
test("determinism_genes", g1 == g2)
test("determinism_replaced", r1 == r2)


# ═══ 7. Locked genomes cannot be replaced ════════════════════

rqre7 = RollingEvolutionEngine(
    symbols=["L"], cluster_types={"L": "none"},
)
rqre7._pools["L"] = [
    GenomeSlot(genes={"g": 0.1}, fitness=0.1, origin_cycle=0, locked=True),
    GenomeSlot(genes={"g": 0.2}, fitness=0.2, origin_cycle=0),
]
cands7 = [
    {"genes": {"g": 0.95}, "fitness": 0.95, "pf": 2.0, "max_dd": 0.10, "trades": 100},
]
log7 = rqre7.submit_candidates("L", cands7, 90)
# Only unlocked slot (fitness=0.2) should be replaced
pool_L = rqre7.get_pool("L")
locked_still = [s for s in pool_L if s.locked]
test("locked_preserved", len(locked_still) == 1)
test("locked_fitness_unchanged", locked_still[0].fitness == 0.1)


# ═══ 8. Empty pool + candidates ══════════════════════════════

rqre8 = RollingEvolutionEngine(
    symbols=["E"], cluster_types={"E": "none"},
)
test("empty_pool_initially", len(rqre8.get_pool("E")) == 0)
test("empty_genome_fallback", rqre8.active_genomes["E"] == {})

cands8 = [
    {"genes": {"x": 1.0}, "fitness": 0.7, "pf": 1.2, "max_dd": 0.30, "trades": 25},
    {"genes": {"x": 2.0}, "fitness": 0.8, "pf": 1.5, "max_dd": 0.20, "trades": 40},
]
log8 = rqre8.submit_candidates("E", cands8, 90)
test("empty_pool_populated", len(rqre8.get_pool("E")) >= 1)
test("best_selected", rqre8.active_genomes["E"]["x"] == 2.0)


# ═══ 9. bars_to_days conversion ══════════════════════════════

rqre9 = RollingEvolutionEngine(["X"], {"X": "none"})
test("bars_to_days_6", rqre9.bars_to_days(540, 6) == 90)
test("bars_to_days_24", rqre9.bars_to_days(2160, 24) == 90)


# ═══ 10. step_if_due ═════════════════════════════════════════

rqre10 = RollingEvolutionEngine(
    symbols=["S"], cluster_types={"S": "none"},
    config=RQREConfig(re_evolution_interval=90),
)
test("step_due_bar_0", rqre10.step_if_due(0, bars_per_day=6))
rqre10.submit_candidates("S", [], 0)
test("step_not_due_bar_100", not rqre10.step_if_due(100, bars_per_day=6))
test("step_due_bar_540", rqre10.step_if_due(540, bars_per_day=6))


# ═══ 11. Multiple symbols independent ════════════════════════

rqre11 = RollingEvolutionEngine(
    symbols=["A", "B"], cluster_types={"A": "none", "B": "none"},
    initial_genomes={"A": {"g": 1.0}, "B": {"g": 2.0}},
)
# Submit only for A
rqre11.submit_candidates("A", [
    {"genes": {"g": 5.0}, "fitness": 0.9, "pf": 1.5, "max_dd": 0.20, "trades": 50}
], 90)
# B should be unchanged
test("b_unchanged", rqre11.active_genomes["B"]["g"] == 2.0)
test("a_updated", rqre11.active_genomes["A"]["g"] == 5.0)


# ═══ 12. Cycle log accumulates ═══════════════════════════════

rqre12 = RollingEvolutionEngine(
    symbols=["M"], cluster_types={"M": "none"},
    initial_genomes={"M": {"g": 0.1}},
)
rqre12.submit_candidates("M", [], 90)
rqre12.submit_candidates("M", [], 180)
rqre12.submit_candidates("M", [], 270)
test("log_count", len(rqre12.logs) == 3)
test("cycle_count_3", rqre12.cycle_count == 3)
test("log_days", [l.day_triggered for l in rqre12.logs] == [90, 180, 270])


# ═══ 13. All candidates filtered → no replacement ═══════════

rqre13 = RollingEvolutionEngine(
    symbols=["F"], cluster_types={"F": "none"},
    config=RQREConfig(min_pf=2.0),  # very strict
    initial_genomes={"F": {"g": 0.5}},
)
bad_cands = [
    {"genes": {"g": 0.9}, "fitness": 0.9, "pf": 1.5, "max_dd": 0.20, "trades": 50},
]
log13 = rqre13.submit_candidates("F", bad_cands, 90)
test("all_filtered_no_replace", log13.replaced_count == 0)
test("all_filtered_kept", log13.candidates_passed == 0)


# ═══ 14. async execute_cycle (mock) ══════════════════════════

async def mock_train(train_data, val_data, seed, cluster_type, config):
    """Mock training function that returns deterministic candidates."""
    return [
        {"genes": {"g": seed * 0.001}, "fitness": 0.7 + seed * 0.0001,
         "pf": 1.3, "max_dd": 0.25, "trades": 40},
    ]

async def test_execute():
    global passed, failed
    rqre14 = RollingEvolutionEngine(
        symbols=["T"], cluster_types={"T": "test"},
        config=RQREConfig(re_evolution_interval=90, seeds=(42, 123, 456)),
        initial_genomes={"T": {"g": 0.01}},
    )
    market = {"T": [{"close": str(100 + i)} for i in range(3000)]}
    logs = await rqre14.execute_cycle(90, mock_train, market, bars_per_day=6)
    if "T" in logs:
        passed += 1
    else:
        failed += 1
        print("  ❌ execute_cycle_returns_log")

    if rqre14.cycle_count >= 1:
        passed += 1
    else:
        failed += 1
        print("  ❌ execute_cycle_increments")

asyncio.run(test_execute())


# ═══ Results ═════════════════════════════════════════════════

total = passed + failed
if failed == 0:
    print(f"  ✅ ALL {total} RQRE TESTS PASSED")
else:
    print(f"  ❌ {failed}/{total} FAILED")
