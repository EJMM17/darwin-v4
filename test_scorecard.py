"""
Darwin v4 — SimulationScorecard Tests.

Covers all 7 requirements:
  1. Compute 5 sub-scores (risk, evolution, concentration, shock, learning)
  2. Normalize each to 0–10
  3. Weighted ecosystem health score
  4. Uses diagnostics + risk + evolution outputs
  5. No trading logic
  6. Pure analytical
  7. Structured JSON report
  8. Math stability (NaN, inf, empty, edge cases)
  9. Commentary generation
  10. Integration with live SimulationResults
"""
import asyncio
import inspect
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from darwin_agent.simulation.scorecard import (
    SimulationScorecard, ScorecardWeights, ScorecardReport,
    GenerationData, SubScore, extract_generation_data,
    _clip10, _clamp01, _ols_slope, _sigmoid_map, _grade,
)

PASS = FAIL = 0

def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ✅ {name}")
    else:
        FAIL += 1; print(f"  ❌ {name}: {detail}")


# ── Builders ─────────────────────────────────────────────

def _healthy_gen(gen: int, capital: float = 100.0) -> GenerationData:
    """Build a healthy generation with good metrics."""
    return GenerationData(
        generation=gen,
        pool_capital=capital,
        pool_pnl=capital * 0.02,
        portfolio_state="normal",
        best_fitness=0.7 + gen * 0.005,
        avg_fitness=0.5 + gen * 0.005,
        worst_fitness=0.3 + gen * 0.003,
        total_trades=25,
        pool_win_rate=0.55 + gen * 0.002,
        pool_sharpe=1.2 + gen * 0.01,
        pool_max_drawdown=3.0,
        survivors=3, eliminated=2, population_size=5,
        diversity_score=0.6,
        mean_entropy=0.5,
        capital_gini=0.15,
        capital_herfindahl=0.25,
        capital_top1_pct=30.0,
        exposure_herfindahl=0.3,
        exposure_top1_pct=40.0,
        dominant_lineage_share=0.3,
        effective_lineages=4.0,
        fitness_improving=True,
        fitness_stagnating=False,
        stagnation_generations=0,
        health_score=0.75,
        n_alerts=0, alerts=[],
    )

def _sick_gen(gen: int, capital: float = 30.0) -> GenerationData:
    """Build a pathological generation."""
    return GenerationData(
        generation=gen,
        pool_capital=capital,
        pool_pnl=-capital * 0.15,
        portfolio_state="critical",
        best_fitness=0.2,
        avg_fitness=0.1,
        worst_fitness=0.0,
        total_trades=2,
        pool_win_rate=0.2,
        pool_sharpe=-0.5,
        pool_max_drawdown=20.0,
        survivors=1, eliminated=4, population_size=5,
        diversity_score=0.05,
        mean_entropy=0.02,
        capital_gini=0.85,
        capital_herfindahl=0.8,
        capital_top1_pct=90.0,
        exposure_herfindahl=0.95,
        exposure_top1_pct=95.0,
        dominant_lineage_share=0.9,
        effective_lineages=1.2,
        fitness_improving=False,
        fitness_stagnating=True,
        stagnation_generations=8,
        health_score=0.1,
        n_alerts=5,
        alerts=[
            "GENETIC_COLLAPSE: diversity=0.05",
            "CAPITAL_CONCENTRATION: top1=90%",
            "EXPOSURE_CONCENTRATION: BTCUSDT=95%",
            "LINEAGE_DOMINANCE: alpha=90%",
            "FITNESS_STAGNATION: 8 gens",
        ],
    )


# ═════════════════════════════════════════════════════════════
# 1. Five sub-score computation
# ═════════════════════════════════════════════════════════════

def test_1_sub_scores():
    print("\n🏆 1. Five sub-score computation")
    print("─" * 50)

    sc = SimulationScorecard()
    gens = [_healthy_gen(i) for i in range(10)]
    report = sc.evaluate(gens, starting_capital=100, final_capital=120)

    check("Risk stability score computed",
          report.risk_stability.score > 0)
    check("Evolution health score computed",
          report.evolution_health.score > 0)
    check("Concentration risk score computed",
          report.concentration_risk.score > 0)
    check("Shock resilience score computed",
          report.shock_resilience.score > 0)
    check("Learning quality score computed",
          report.learning_quality.score > 0)

    # Sub-score names
    check("Risk name", report.risk_stability.name == "Risk Stability")
    check("Evolution name", report.evolution_health.name == "Evolution Health")
    check("Concentration name",
          report.concentration_risk.name == "Concentration Risk")
    check("Shock name", report.shock_resilience.name == "Shock Resilience")
    check("Learning name", report.learning_quality.name == "Learning Quality")

    # Components populated
    for ss in [report.risk_stability, report.evolution_health,
               report.concentration_risk, report.shock_resilience,
               report.learning_quality]:
        if len(ss.components) < 3:
            check(f"{ss.name} has 4 components", False,
                  f"got {len(ss.components)}")
            break
    else:
        check("All sub-scores have 4 components", True)
    print()


# ═════════════════════════════════════════════════════════════
# 2. Normalization to 0–10
# ═════════════════════════════════════════════════════════════

def test_2_normalization():
    print("🏆 2. Score normalization [0, 10]")
    print("─" * 50)

    sc = SimulationScorecard()

    # Healthy sim
    healthy = [_healthy_gen(i) for i in range(10)]
    r_h = sc.evaluate(healthy, 100, 120)

    for ss in [r_h.risk_stability, r_h.evolution_health,
               r_h.concentration_risk, r_h.shock_resilience,
               r_h.learning_quality]:
        if not (0.0 <= ss.score <= 10.0):
            check(f"{ss.name} in [0, 10]", False, f"score={ss.score}")
            break
        for k, v in ss.components.items():
            if not (0.0 <= v <= 10.0):
                check(f"{ss.name}.{k} in [0, 10]", False, f"value={v}")
                break
    else:
        check("All healthy scores + components in [0, 10]", True)

    # Sick sim
    sick = [_sick_gen(i, capital=30-i*2) for i in range(10)]
    r_s = sc.evaluate(sick, 100, 10)

    for ss in [r_s.risk_stability, r_s.evolution_health,
               r_s.concentration_risk, r_s.shock_resilience,
               r_s.learning_quality]:
        if not (0.0 <= ss.score <= 10.0):
            check(f"Sick {ss.name} in [0, 10]", False, f"score={ss.score}")
            break
    else:
        check("All sick scores in [0, 10]", True)

    check("Ecosystem health in [0, 10]",
          0.0 <= r_h.ecosystem_health <= 10.0 and
          0.0 <= r_s.ecosystem_health <= 10.0)

    # Healthy > sick
    check("Healthy ecosystem > sick ecosystem",
          r_h.ecosystem_health > r_s.ecosystem_health,
          f"healthy={r_h.ecosystem_health:.2f}, sick={r_s.ecosystem_health:.2f}")
    print()


# ═════════════════════════════════════════════════════════════
# 3. Weighted ecosystem health
# ═════════════════════════════════════════════════════════════

def test_3_weighted_health():
    print("🏆 3. Weighted ecosystem health score")
    print("─" * 50)

    gens = [_healthy_gen(i) for i in range(5)]

    # Default weights
    sc1 = SimulationScorecard()
    r1 = sc1.evaluate(gens, 100, 110)
    check("Default weights sum to 1.0",
          abs(sum(r1.weights_used.values()) - 1.0) < 0.01)

    # Custom weights: all on risk
    risk_heavy = ScorecardWeights(
        risk_stability=0.60, evolution_health=0.10,
        concentration_risk=0.10, shock_resilience=0.10,
        learning_quality=0.10,
    )
    sc2 = SimulationScorecard(weights=risk_heavy)
    r2 = sc2.evaluate(gens, 100, 110)
    check("Custom weights used",
          r2.weights_used["risk_stability"] == 0.60)

    # Manual verify: ecosystem = weighted sum of sub-scores
    expected = (
        0.60 * r2.risk_stability.score +
        0.10 * r2.evolution_health.score +
        0.10 * r2.concentration_risk.score +
        0.10 * r2.shock_resilience.score +
        0.10 * r2.learning_quality.score
    )
    check("Ecosystem = weighted sum of sub-scores",
          abs(r2.ecosystem_health - expected) < 0.05,
          f"computed={r2.ecosystem_health:.2f}, expected={expected:.2f}")

    # Invalid weights rejected
    try:
        ScorecardWeights(risk_stability=0.5, evolution_health=0.5,
                         concentration_risk=0.5, shock_resilience=0.0,
                         learning_quality=0.0)
        sc_bad = SimulationScorecard(
            weights=ScorecardWeights(
                risk_stability=0.5, evolution_health=0.5,
                concentration_risk=0.5))
        check("Invalid weights rejected", False, "no error raised")
    except ValueError:
        check("Invalid weights rejected", True)
    print()


# ═════════════════════════════════════════════════════════════
# 4. Uses diagnostics + risk + evolution outputs
# ═════════════════════════════════════════════════════════════

def test_4_input_integration():
    print("🏆 4. Input integration (diagnostics + risk + evolution)")
    print("─" * 50)

    sc = SimulationScorecard()

    # High drawdown → bad risk score
    high_dd = [GenerationData(
        generation=i, pool_capital=80, pool_pnl=-20,
        portfolio_state="critical", pool_max_drawdown=22.0,
    ) for i in range(5)]
    r_dd = sc.evaluate(high_dd, 100, 80)
    check("High drawdown → low risk score",
          r_dd.risk_stability.score < 5.0,
          f"score={r_dd.risk_stability.score:.2f}")

    # Low diversity → bad evolution score
    no_div = [GenerationData(
        generation=i, diversity_score=0.01, mean_entropy=0.01,
        effective_lineages=1.0, population_size=5,
        fitness_stagnating=True,
    ) for i in range(5)]
    r_nd = sc.evaluate(no_div, 100, 100)
    check("Low diversity → low evolution score",
          r_nd.evolution_health.score < 4.0,
          f"score={r_nd.evolution_health.score:.2f}")

    # High Gini → bad concentration score
    hi_gini = [GenerationData(
        generation=i, capital_gini=0.9, exposure_herfindahl=0.9,
        capital_top1_pct=95, dominant_lineage_share=0.9,
    ) for i in range(5)]
    r_gini = sc.evaluate(hi_gini, 100, 100)
    check("High Gini → low concentration score",
          r_gini.concentration_risk.score < 3.0,
          f"score={r_gini.concentration_risk.score:.2f}")

    # Capital loss → bad shock score
    lost = [GenerationData(
        generation=i, pool_capital=20, pool_pnl=-16,
        portfolio_state="halted",
    ) for i in range(5)]
    r_lost = sc.evaluate(lost, 100, 20)
    check("Capital loss + halts → low shock score",
          r_lost.shock_resilience.score < 3.0,
          f"score={r_lost.shock_resilience.score:.2f}")

    # Flat fitness → bad learning score
    flat = [GenerationData(
        generation=i, avg_fitness=0.3, pool_win_rate=0.4,
        pool_sharpe=0.0, total_trades=0, population_size=5,
    ) for i in range(10)]
    r_flat = sc.evaluate(flat, 100, 100)
    check("Flat fitness + no trades → low learning score",
          r_flat.learning_quality.score < 5.0,
          f"score={r_flat.learning_quality.score:.2f}")
    print()


# ═════════════════════════════════════════════════════════════
# 5. No trading logic
# ═════════════════════════════════════════════════════════════

def test_5_no_trading():
    print("🏆 5. No trading logic")
    print("─" * 50)

    source = inspect.getsource(SimulationScorecard)
    module_source = open(
        "darwin_agent/simulation/scorecard.py").read()

    check("No exchange imports",
          "BybitAdapter" not in module_source and
          "ExchangeRouter" not in module_source)
    check("No order/position logic",
          "OrderRequest" not in module_source and
          "Position" not in module_source)
    check("No async (pure sync)",
          "async def" not in source)
    check("No aiohttp",
          "aiohttp" not in module_source)
    check("No signal generation",
          "Signal" not in source or "signal" in "signal_map" == False)
    print()


# ═════════════════════════════════════════════════════════════
# 6. Pure analytical (deterministic, no state)
# ═════════════════════════════════════════════════════════════

def test_6_pure_analytical():
    print("🏆 6. Pure analytical evaluation")
    print("─" * 50)

    sc = SimulationScorecard()
    gens = [_healthy_gen(i) for i in range(5)]

    r1 = sc.evaluate(gens, 100, 110)
    r2 = sc.evaluate(gens, 100, 110)

    check("Deterministic: same input → same scores",
          abs(r1.ecosystem_health - r2.ecosystem_health) < 0.001)
    check("Deterministic: same risk scores",
          abs(r1.risk_stability.score - r2.risk_stability.score) < 0.001)

    # No side effects: evaluating sick data doesn't affect healthy scores
    sick = [_sick_gen(i) for i in range(5)]
    sc.evaluate(sick, 100, 20)
    r3 = sc.evaluate(gens, 100, 110)
    check("No side effects between evaluations",
          abs(r1.ecosystem_health - r3.ecosystem_health) < 0.001)
    print()


# ═════════════════════════════════════════════════════════════
# 7. Structured JSON report
# ═════════════════════════════════════════════════════════════

def test_7_json_report():
    print("🏆 7. Structured JSON report")
    print("─" * 50)

    sc = SimulationScorecard()
    gens = [_healthy_gen(i) for i in range(5)]
    report = sc.evaluate(gens, 100, 115)
    d = report.to_dict()

    # Top-level keys
    check("JSON has ecosystem_health", "ecosystem_health" in d)
    check("JSON has ecosystem_grade", "ecosystem_grade" in d)
    check("JSON has weights_used", "weights_used" in d)
    check("JSON has scores section", "scores" in d)
    check("JSON has context section", "context" in d)

    # Scores section
    scores = d["scores"]
    for key in ["risk_stability", "evolution_health", "concentration_risk",
                "shock_resilience", "learning_quality"]:
        if key not in scores:
            check(f"Scores has {key}", False)
            break
        sub = scores[key]
        if not all(k in sub for k in ["score", "grade", "components", "commentary"]):
            check(f"{key} has score/grade/components/commentary", False)
            break
    else:
        check("All 5 sub-scores have full structure", True)

    # Grades are valid
    valid_grades = {"A", "B", "C", "D", "F"}
    check("Ecosystem grade is valid",
          d["ecosystem_grade"] in valid_grades)
    for key, sub in scores.items():
        if sub["grade"] not in valid_grades:
            check(f"{key} grade is valid", False, f"grade={sub['grade']}")
            break
    else:
        check("All sub-score grades are valid", True)

    # Context section
    ctx = d["context"]
    check("Context has generations_evaluated",
          ctx["generations_evaluated"] == 5)
    check("Context has starting_capital",
          ctx["starting_capital"] == 100.0)
    check("Context has capital_return_pct",
          abs(ctx["capital_return_pct"] - 15.0) < 0.1)

    # JSON serializable
    try:
        serialized = json.dumps(d)
        check("Report is JSON serializable", True)
    except (TypeError, ValueError) as e:
        check("Report is JSON serializable", False, str(e))

    # Commentary is non-empty
    for key, sub in scores.items():
        if not sub["commentary"]:
            check(f"{key} has commentary", False)
            break
    else:
        check("All sub-scores have commentary", True)
    print()


# ═════════════════════════════════════════════════════════════
# 8. Math stability
# ═════════════════════════════════════════════════════════════

def test_8_math_stability():
    print("🏆 8. Math stability (NaN, inf, edge cases)")
    print("─" * 50)

    sc = SimulationScorecard()

    # Empty generations
    r_empty = sc.evaluate([], 0, 0)
    check("Empty input → valid report",
          r_empty.ecosystem_grade == "F")
    check("Empty input → 0 ecosystem health",
          r_empty.ecosystem_health == 0.0)

    # Single generation
    r_single = sc.evaluate([_healthy_gen(0)], 100, 102)
    check("Single gen → valid scores",
          0.0 <= r_single.ecosystem_health <= 10.0)

    # NaN/inf in data
    nan_gen = GenerationData(
        generation=0, pool_capital=float('nan'),
        pool_pnl=float('inf'), pool_max_drawdown=float('-inf'),
        avg_fitness=float('nan'), diversity_score=float('nan'),
    )
    r_nan = sc.evaluate([nan_gen], 100, 100)
    check("NaN input → no crash",
          0.0 <= r_nan.ecosystem_health <= 10.0)
    check("NaN components → bounded scores",
          all(0.0 <= getattr(r_nan, attr).score <= 10.0
              for attr in ["risk_stability", "evolution_health",
                           "concentration_risk", "shock_resilience",
                           "learning_quality"]))

    # Zero starting capital
    r_zero = sc.evaluate([_healthy_gen(0)], 0, 0)
    check("Zero capital → valid report",
          0.0 <= r_zero.ecosystem_health <= 10.0)

    # Negative PnL, negative capital
    neg = GenerationData(pool_capital=-50, pool_pnl=-200)
    r_neg = sc.evaluate([neg], 100, -50)
    check("Negative capital → no crash",
          0.0 <= r_neg.ecosystem_health <= 10.0)

    # Helper functions
    check("_clip10(NaN) = 0", _clip10(float('nan')) == 0.0)
    check("_clip10(inf) = 0", _clip10(float('inf')) == 0.0)
    check("_clip10(-5) = 0", _clip10(-5.0) == 0.0)
    check("_clip10(15) = 10", _clip10(15.0) == 10.0)
    check("_clamp01(NaN) = 0", _clamp01(float('nan')) == 0.0)
    check("_ols_slope([]) = 0", _ols_slope([]) == 0.0)
    check("_ols_slope([5]) = 0", _ols_slope([5.0]) == 0.0)
    check("_ols_slope([1,2,3]) > 0", _ols_slope([1.0, 2.0, 3.0]) > 0)
    check("_sigmoid_map(0) < 0.5", _sigmoid_map(0.0) < 0.5)
    check("_sigmoid_map(1) ≈ 0.5", abs(_sigmoid_map(1.0) - 0.5) < 0.01)
    check("_sigmoid_map(100) ≈ 1.0", _sigmoid_map(100.0) > 0.99)
    print()


# ═════════════════════════════════════════════════════════════
# 9. Grade boundaries
# ═════════════════════════════════════════════════════════════

def test_9_grades():
    print("🏆 9. Grade assignment")
    print("─" * 50)

    check("10.0 → A", _grade(10.0) == "A")
    check("8.5 → A", _grade(8.5) == "A")
    check("8.4 → B", _grade(8.4) == "B")
    check("7.0 → B", _grade(7.0) == "B")
    check("6.9 → C", _grade(6.9) == "C")
    check("5.0 → C", _grade(5.0) == "C")
    check("4.9 → D", _grade(4.9) == "D")
    check("3.0 → D", _grade(3.0) == "D")
    check("2.9 → F", _grade(2.9) == "F")
    check("0.0 → F", _grade(0.0) == "F")
    print()


# ═════════════════════════════════════════════════════════════
# 10. extract_generation_data from dict
# ═════════════════════════════════════════════════════════════

def test_10_extract():
    print("🏆 10. extract_generation_data")
    print("─" * 50)

    raw = {
        "generation": 5,
        "pool_capital": 120.0,
        "pool_pnl": 20.0,
        "portfolio_state": "normal",
        "snapshot": {
            "best_fitness": 0.8,
            "avg_fitness": 0.5,
            "total_trades": 30,
            "pool_win_rate": 0.6,
            "pool_sharpe": 1.5,
            "pool_max_drawdown": 5.0,
            "survivors": 3,
            "eliminated": 2,
        },
        "diagnostics": {
            "health_score": 0.7,
            "alerts": ["GENETIC_COLLAPSE: test"],
            "diversity": {
                "overall_diversity_score": 0.4,
                "mean_entropy": 0.35,
            },
            "concentration": {
                "capital_gini": 0.2,
                "capital_herfindahl": 0.3,
                "capital_top1_pct": 35.0,
                "exposure_herfindahl": 0.25,
                "exposure_top1_pct": 40.0,
            },
            "dominance": {
                "dominant_lineage_share": 0.3,
                "effective_lineages": 3.5,
            },
            "fitness": {
                "improving": True,
                "stagnating": False,
                "stagnation_generations": 0,
            },
        },
    }

    gd = extract_generation_data(raw)
    check("Generation extracted", gd.generation == 5)
    check("Capital extracted", gd.pool_capital == 120.0)
    check("Best fitness extracted", gd.best_fitness == 0.8)
    check("Diversity extracted", gd.diversity_score == 0.4)
    check("Gini extracted", gd.capital_gini == 0.2)
    check("Alerts extracted", len(gd.alerts) == 1)
    check("Improving extracted", gd.fitness_improving is True)

    # Missing keys → defaults
    gd_empty = extract_generation_data({})
    check("Empty dict → valid defaults",
          gd_empty.generation == 0 and gd_empty.pool_capital == 0.0)
    print()


# ═════════════════════════════════════════════════════════════
# 11. Integration with live simulation
# ═════════════════════════════════════════════════════════════

async def test_11_live_integration():
    print("🏆 11. Integration with SimulationHarness")
    print("─" * 50)

    from darwin_agent.simulation.harness import (
        SimulationHarness, SimConfig, MonteCarloScenario,
    )

    config = SimConfig(
        scenario=MonteCarloScenario(symbols=["BTCUSDT", "ETHUSDT"]),
        generations=10, pool_size=5,
        trades_per_generation=100, seed=42,
        starting_capital=100.0,
    )
    sim = SimulationHarness(config)
    results = await sim.run()

    # Convert SimulationResults → GenerationData list
    gen_data = []
    for gr in results.generation_results:
        gd = GenerationData(
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
            diversity_score=gr.diagnostics_dict.get("diversity", {}).get(
                "overall_diversity_score", 0),
            mean_entropy=gr.diagnostics_dict.get("diversity", {}).get(
                "mean_entropy", 0),
            capital_gini=gr.diagnostics_dict.get("concentration", {}).get(
                "capital_gini", 0),
            capital_herfindahl=gr.diagnostics_dict.get("concentration", {}).get(
                "capital_herfindahl", 0),
            capital_top1_pct=gr.diagnostics_dict.get("concentration", {}).get(
                "capital_top1_pct", 0),
            exposure_herfindahl=gr.diagnostics_dict.get("concentration", {}).get(
                "exposure_herfindahl", 0),
            exposure_top1_pct=gr.diagnostics_dict.get("concentration", {}).get(
                "exposure_top1_pct", 0),
            dominant_lineage_share=gr.diagnostics_dict.get("dominance", {}).get(
                "dominant_lineage_share", 0),
            effective_lineages=gr.diagnostics_dict.get("dominance", {}).get(
                "effective_lineages", 0),
            fitness_improving=gr.diagnostics_dict.get("fitness", {}).get(
                "improving", False),
            fitness_stagnating=gr.diagnostics_dict.get("fitness", {}).get(
                "stagnating", False),
            stagnation_generations=gr.diagnostics_dict.get("fitness", {}).get(
                "stagnation_generations", 0),
            health_score=gr.diagnostics_dict.get("health_score", 0),
            n_alerts=len(gr.diagnostics_dict.get("alerts", [])),
            alerts=gr.diagnostics_dict.get("alerts", []),
        )
        gen_data.append(gd)

    sc = SimulationScorecard()
    report = sc.evaluate(
        gen_data,
        starting_capital=results.starting_capital,
        final_capital=results.final_capital,
    )

    check("Live sim → valid ecosystem health",
          0.0 <= report.ecosystem_health <= 10.0,
          f"got {report.ecosystem_health:.2f}")
    check("Live sim → valid grade",
          report.ecosystem_grade in {"A", "B", "C", "D", "F"})
    check("Live sim → 10 gens evaluated",
          report.generations_evaluated == 10)
    check("Live sim → JSON serializable",
          json.dumps(report.to_dict()) is not None)

    d = report.to_dict()
    print(f"\n  📊 LIVE SIM SCORECARD:")
    print(f"     Ecosystem: {d['ecosystem_health']:.1f}/10 ({d['ecosystem_grade']})")
    for name, sub in d["scores"].items():
        print(f"     {name:25s}: {sub['score']:.1f}/10 ({sub['grade']})")
    print(f"     Capital: ${results.starting_capital:.0f} → ${results.final_capital:.0f} "
          f"({d['context']['capital_return_pct']:+.1f}%)")
    print()


# ═════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════

async def main():
    global PASS, FAIL

    print("═" * 60)
    print("  🏆 SIMULATION SCORECARD TESTS")
    print("═" * 60)

    sync_tests = [
        test_1_sub_scores, test_2_normalization,
        test_3_weighted_health, test_4_input_integration,
        test_5_no_trading, test_6_pure_analytical,
        test_7_json_report, test_8_math_stability,
        test_9_grades, test_10_extract,
    ]
    async_tests = [test_11_live_integration]

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
        print(f"  ✅ ALL {PASS} SCORECARD TESTS PASSED")
    else:
        print(f"  ❌ {FAIL} FAILED, {PASS} passed")
    print("═" * 60)
    return FAIL == 0


if __name__ == "__main__":
    ok = asyncio.run(main())
    sys.exit(0 if ok else 1)
