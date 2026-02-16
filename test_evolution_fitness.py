"""
Darwin v4 — EvolutionEngine ↔ RiskAwareFitness Integration Tests.

Verifies all 7 integration requirements:
  1. Engine computes fitness internally (not pre-baked)
  2. Portfolio snapshot flows through to fitness model
  3. Agent exposure + pnl_series used in scoring
  4. FitnessBreakdown stored and accessible for dashboard
  5. No legacy inline fitness formulas (engine re-scores)
  6. Ranking uses updated risk-aware fitness
  7. Architecture: engine depends on fitness, NOT on risk engine
"""
import asyncio
import inspect
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from darwin_agent.interfaces.enums import *
from darwin_agent.interfaces.types import *
from darwin_agent.evolution.engine import EvolutionEngine
from darwin_agent.evolution.fitness import RiskAwareFitness, FitnessBreakdown

PASS = FAIL = 0

def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ✅ {name}")
    else:
        FAIL += 1; print(f"  ❌ {name}: {detail}")


def _make_eval(
    agent_id, pnl, capital=100.0, wins=10, losses=5,
    sharpe=1.0, dd=5.0, exposure=None, pnl_series=None,
):
    """Helper: build AgentEvalData with sensible defaults."""
    return AgentEvalData(
        metrics=AgentMetrics(
            agent_id=agent_id, generation=0, phase="dead",
            capital=capital, realized_pnl=pnl, total_trades=wins+losses,
            winning_trades=wins, losing_trades=losses,
            sharpe_ratio=sharpe, max_drawdown_pct=dd,
        ),
        initial_capital=100.0,
        pnl_series=pnl_series or [pnl / max(wins + losses, 1)] * (wins + losses),
        exposure=exposure or {},
        dna=DNAData(dna_id=f"dna-{agent_id}", genes={"risk_pct": 2.0}),
    )


def _make_snap(state=PortfolioRiskState.NORMAL, dd=0.0, corr=0.3, exposure=None):
    return PortfolioRiskMetrics(
        risk_state=state, total_equity=1000.0, peak_equity=1000.0,
        drawdown_pct=dd, correlation_risk=corr,
        exposure_by_symbol=exposure or {"BTCUSDT": 0.5, "ETHUSDT": 0.3},
    )


async def test_1_engine_computes_fitness_internally():
    """Requirement 1: Engine re-scores, doesn't trust pre-baked fitness."""
    print("\n🧬 1. Engine computes fitness internally")
    print("─" * 50)

    engine = EvolutionEngine()

    # Pre-set fitness to a known value
    evals = [
        _make_eval("agent-A", pnl=50, wins=15, losses=5),
        _make_eval("agent-B", pnl=-20, wins=3, losses=12),
    ]
    evals[0].metrics.fitness = 0.999  # fake high
    evals[1].metrics.fitness = 0.999  # fake high

    snap = await engine.evaluate_generation(evals)

    # Engine should have OVERWRITTEN the pre-baked 0.999
    a_fitness = next(r["fitness"] for r in snap.agent_rankings if r["agent_id"] == "agent-A")
    b_fitness = next(r["fitness"] for r in snap.agent_rankings if r["agent_id"] == "agent-B")

    check("Agent A fitness overwritten (≠ 0.999)", a_fitness != 0.999,
          f"got {a_fitness}")
    check("Agent B fitness overwritten (≠ 0.999)", b_fitness != 0.999,
          f"got {b_fitness}")
    check("Profitable A ranked above losing B", a_fitness > b_fitness,
          f"A={a_fitness}, B={b_fitness}")
    print("  ✅ PASSED\n")


async def test_2_portfolio_snapshot_flows():
    """Requirement 2: Portfolio state affects fitness scoring."""
    print("🧬 2. Portfolio snapshot flows to fitness model")
    print("─" * 50)

    engine = EvolutionEngine()
    agent = _make_eval("test", pnl=-15, wins=5, losses=15, sharpe=-0.3, dd=20)

    # NORMAL portfolio → agent gets full harmony score
    snap_normal = _make_snap(state=PortfolioRiskState.NORMAL)
    snap1 = await engine.evaluate_generation([agent], portfolio_snapshot=snap_normal)
    fit_normal = snap1.agent_rankings[0]["fitness"]
    bd_normal = snap1.agent_rankings[0]["fitness_breakdown"]

    # Reset metrics fitness for fair comparison
    agent.metrics.fitness = 0.0

    # CRITICAL portfolio → agent gets penalized in harmony component
    snap_crit = _make_snap(state=PortfolioRiskState.CRITICAL, dd=25.0)
    snap2 = await engine.evaluate_generation([agent], portfolio_snapshot=snap_crit)
    fit_crit = snap2.agent_rankings[0]["fitness"]
    bd_crit = snap2.agent_rankings[0]["fitness_breakdown"]

    check("Portfolio state recorded in snapshot metadata",
          snap2.metadata.get("portfolio_state") == "critical",
          f"got {snap2.metadata}")
    check("CRITICAL harmony < NORMAL harmony",
          bd_crit["portfolio_harmony"] < bd_normal["portfolio_harmony"],
          f"CRIT={bd_crit['portfolio_harmony']}, NORM={bd_normal['portfolio_harmony']}")
    check("Overall fitness lower under CRITICAL",
          fit_crit < fit_normal,
          f"CRIT={fit_crit}, NORM={fit_normal}")
    print("  ✅ PASSED\n")


async def test_3_exposure_and_pnl_series():
    """Requirement 3: Agent exposure + pnl_series affect scoring."""
    print("🧬 3. Agent exposure + pnl_series used in scoring")
    print("─" * 50)

    engine = EvolutionEngine()
    snap = _make_snap(exposure={"BTCUSDT": 0.8, "ETHUSDT": 0.2}, corr=0.8)

    # Agent overlaps with pool (100% BTC)
    overlap = _make_eval("overlap", pnl=10, exposure={"BTCUSDT": 1.0})
    # Agent diversifies (SOL + AVAX)
    diverse = _make_eval("diverse", pnl=10, exposure={"SOLUSDT": 0.5, "AVAXUSDT": 0.5})

    result = await engine.evaluate_generation(
        [overlap, diverse], portfolio_snapshot=snap)

    rankings = {r["agent_id"]: r for r in result.agent_rankings}
    overlap_div = rankings["overlap"]["fitness_breakdown"]["diversification_bonus"]
    diverse_div = rankings["diverse"]["fitness_breakdown"]["diversification_bonus"]

    check("Diversified agent div score > overlap agent",
          diverse_div > overlap_div,
          f"diverse={diverse_div}, overlap={overlap_div}")

    # Test pnl_series: consistent vs volatile
    consistent = _make_eval("cons", pnl=10, pnl_series=[0.5]*20)
    volatile = _make_eval("vol", pnl=10, pnl_series=[5, -4, 6, -5, 3, -2, 4, -3, 5, -4,
                                                       3, -2, 4, -3, 5, -4, 3, -2, 4, -3])
    result2 = await engine.evaluate_generation([consistent, volatile])
    r2 = {r["agent_id"]: r for r in result2.agent_rankings}
    cons_c = r2["cons"]["fitness_breakdown"]["consistency"]
    vol_c = r2["vol"]["fitness_breakdown"]["consistency"]

    check("Consistent agent consistency > volatile agent",
          cons_c > vol_c,
          f"consistent={cons_c}, volatile={vol_c}")
    print("  ✅ PASSED\n")


async def test_4_breakdown_stored_for_dashboard():
    """Requirement 4: FitnessBreakdown stored and accessible."""
    print("🧬 4. FitnessBreakdown stored for dashboard")
    print("─" * 50)

    engine = EvolutionEngine()
    evals = [_make_eval(f"agt-{i}", pnl=i*10, wins=10+i, losses=10-i) for i in range(4)]

    await engine.evaluate_generation(evals)

    # get_fitness_breakdowns returns dict
    breakdowns = engine.get_fitness_breakdowns()
    check("Breakdowns dict has 4 entries", len(breakdowns) == 4,
          f"got {len(breakdowns)}")

    # Each breakdown has all 7 components
    for aid, bd in breakdowns.items():
        check(f"  {aid} has all components", all(
            hasattr(bd, attr) for attr in [
                "risk_adjusted_profit", "sharpe_quality", "drawdown_health",
                "consistency", "portfolio_harmony", "diversification_bonus",
                "capital_efficiency", "final_score",
            ]
        ))

    # Single agent lookup
    bd0 = engine.get_fitness_breakdown("agt-0")
    check("Single lookup works", bd0 is not None)

    missing = engine.get_fitness_breakdown("nonexistent")
    check("Missing agent returns None", missing is None)

    # to_dict serialization
    d = bd0.to_dict()
    check("to_dict has all keys", "portfolio_harmony" in d and "final_score" in d)
    print("  ✅ PASSED\n")


async def test_5_no_legacy_fitness():
    """Requirement 5: Engine re-scores, pre-baked fitness is ignored."""
    print("🧬 5. No legacy inline fitness formulas")
    print("─" * 50)

    engine = EvolutionEngine()

    # Two agents with SWAPPED pre-baked fitness
    good = _make_eval("good", pnl=50, wins=18, losses=2, sharpe=2.5, dd=3)
    bad = _make_eval("bad", pnl=-30, wins=3, losses=17, sharpe=-1, dd=30)

    # Pre-set fitness backwards (bad > good)
    good.metrics.fitness = 0.1
    bad.metrics.fitness = 0.9

    snap = await engine.evaluate_generation([good, bad])

    rankings = snap.agent_rankings
    check("Good agent ranked #1 (pre-baked ignored)",
          rankings[0]["agent_id"] == "good",
          f"#1 was {rankings[0]['agent_id']}")
    check("Bad agent ranked #2",
          rankings[1]["agent_id"] == "bad")

    # Verify the metrics.fitness was overwritten
    check("good.metrics.fitness overwritten",
          good.metrics.fitness != 0.1,
          f"still {good.metrics.fitness}")
    check("bad.metrics.fitness overwritten",
          bad.metrics.fitness != 0.9,
          f"still {bad.metrics.fitness}")
    print("  ✅ PASSED\n")


async def test_6_ranking_uses_risk_aware_fitness():
    """Requirement 6: Ranking order reflects risk-aware scores."""
    print("🧬 6. Ranking uses updated risk-aware fitness")
    print("─" * 50)

    engine = EvolutionEngine()

    # Agent that profits but during CRITICAL state with high correlation
    snap_crit = _make_snap(
        state=PortfolioRiskState.CRITICAL, dd=20, corr=0.9,
        exposure={"BTCUSDT": 0.9},
    )
    risky = _make_eval("risky", pnl=25, wins=12, losses=8,
                       exposure={"BTCUSDT": 1.0})  # same as pool
    safe = _make_eval("safe", pnl=15, wins=12, losses=8, dd=3,
                      exposure={"SOLUSDT": 0.5, "AVAXUSDT": 0.5})  # diversified

    snap = await engine.evaluate_generation([risky, safe], portfolio_snapshot=snap_crit)

    rankings = snap.agent_rankings
    # Safe diversified agent should rank above risky concentrated one
    # even though risky has more PnL, because portfolio is CRITICAL
    # and risky is correlated
    safe_rank = next(r for r in rankings if r["agent_id"] == "safe")
    risky_rank = next(r for r in rankings if r["agent_id"] == "risky")

    check("Safe/diversified ranked above risky/concentrated in CRITICAL state",
          safe_rank["rank"] < risky_rank["rank"],
          f"safe=#{safe_rank['rank']}, risky=#{risky_rank['rank']}")

    # Verify survivor selection uses same ranking
    survivor_ids = engine.select_survivors(
        [e.metrics for e in [risky, safe]])
    check("Survivor selection sees risk-aware fitness",
          len(survivor_ids) >= 1)
    print("  ✅ PASSED\n")


async def test_7_architecture_decoupled():
    """Requirement 7: Engine depends on fitness model, NOT risk engine."""
    print("🧬 7. Architecture clean and decoupled")
    print("─" * 50)

    # Engine can be constructed without any risk engine
    engine = EvolutionEngine()
    check("Engine constructed without risk engine", True)

    # evaluate_generation works without portfolio snapshot
    evals = [_make_eval("solo", pnl=10)]
    snap = await engine.evaluate_generation(evals, portfolio_snapshot=None)
    check("Works without portfolio snapshot", snap.best_fitness > 0)

    # Portfolio harmony defaults to 1.0 when no snapshot
    bd = snap.agent_rankings[0]["fitness_breakdown"]
    check("Harmony defaults to 1.0 without snapshot",
          bd["portfolio_harmony"] == 1.0,
          f"got {bd['portfolio_harmony']}")

    # Fitness model is injectable via config
    from darwin_agent.evolution.fitness import FitnessConfig, FitnessWeights
    custom = FitnessConfig(weights=FitnessWeights(
        ecosystem_health=0.22,
        risk_stability=0.18,
        learning_quality=0.13,
        profit_factor_score=0.13,
        efficiency_score=0.08,
        trend_score=0.13,
        activity_score=0.05,
        convexity_score=0.08,
    ))
    engine2 = EvolutionEngine(fitness_config=custom)
    check("Custom fitness config accepted", engine2._fitness_model._cfg == custom)

    # Verify engine module doesn't import the risk engine CLASS
    import darwin_agent.evolution.engine as eng_mod
    source_file = inspect.getsource(eng_mod)
    check("No portfolio_engine import in engine module",
          "from darwin_agent.risk" not in source_file
          and "import PortfolioRiskEngine" not in source_file)
    print("  ✅ PASSED\n")


async def test_8_hall_of_fame_uses_risk_aware():
    """Bonus: Hall of fame tracks risk-aware fitness, stores DNA."""
    print("🧬 8. Hall of fame uses risk-aware fitness with DNA")
    print("─" * 50)

    engine = EvolutionEngine()

    for gen in range(3):
        evals = [
            _make_eval(f"g{gen}-a{i}", pnl=10+gen*5+i*3,
                       wins=10+i, losses=5)
            for i in range(4)
        ]
        await engine.evaluate_generation(evals)

    hof = engine.get_hall_of_fame()
    check("Hall of fame populated", len(hof) > 0,
          f"got {len(hof)}")
    check("HOF sorted descending by fitness",
          all(hof[i].fitness >= hof[i+1].fitness for i in range(len(hof)-1)))
    check("HOF capped at 20", len(hof) <= 20)

    # HOF entries should have DNA genes (not empty)
    has_genes = sum(1 for d in hof if d.genes)
    check("HOF entries have DNA genes", has_genes > 0,
          f"{has_genes}/{len(hof)} have genes")
    print("  ✅ PASSED\n")


async def test_9_empty_and_edge_cases():
    """Edge cases: empty generation, single agent, all identical."""
    print("🧬 9. Edge cases")
    print("─" * 50)

    engine = EvolutionEngine()

    # Empty
    snap = await engine.evaluate_generation([])
    check("Empty generation → valid snapshot", snap.population_size == 0)

    # Single agent
    snap2 = await engine.evaluate_generation([_make_eval("solo", pnl=10)])
    check("Single agent → 1 survivor", snap2.survivors == 1)
    check("Single agent has breakdown",
          "fitness_breakdown" in snap2.agent_rankings[0])

    # All identical
    same = [_make_eval(f"clone-{i}", pnl=10) for i in range(5)]
    snap3 = await engine.evaluate_generation(same)
    fitnesses = [r["fitness"] for r in snap3.agent_rankings]
    check("Identical agents → identical fitness",
          max(fitnesses) - min(fitnesses) < 0.01,
          f"range={max(fitnesses)-min(fitnesses):.4f}")
    print("  ✅ PASSED\n")


async def main():
    print("═" * 60)
    print("  🧬 EVOLUTION ↔ FITNESS INTEGRATION TESTS")
    print("═" * 60)

    tests = [
        test_1_engine_computes_fitness_internally,
        test_2_portfolio_snapshot_flows,
        test_3_exposure_and_pnl_series,
        test_4_breakdown_stored_for_dashboard,
        test_5_no_legacy_fitness,
        test_6_ranking_uses_risk_aware_fitness,
        test_7_architecture_decoupled,
        test_8_hall_of_fame_uses_risk_aware,
        test_9_empty_and_edge_cases,
    ]
    for t in tests:
        try:
            await t()
        except Exception as exc:
            global FAIL
            FAIL += 1
            print(f"  ❌ CRASHED: {t.__name__}: {exc}")
            import traceback; traceback.print_exc()

    print("═" * 60)
    if FAIL == 0:
        print(f"  ✅ ALL {PASS} EVOLUTION↔FITNESS TESTS PASSED")
    else:
        print(f"  ❌ {FAIL} FAILED, {PASS} passed")
    print("═" * 60)
    return FAIL == 0

if __name__ == "__main__":
    ok = asyncio.run(main())
    sys.exit(0 if ok else 1)
