"""
Darwin v4 — EvolutionDiagnostics Tests.

Tests all 7 requirements + integration:
  1. Genetic diversity score
  2. Per-gene entropy
  3. Capital concentration index
  4. Exposure concentration by symbol
  5. Evolutionary dominance detection
  6. Fitness variance across generations
  7. Structured diagnostics report
  8. Edge cases (empty, single, clones)
  9. Alert generation
  10. Health score composite
  11. Multi-generation trajectory
  12. Serialization (to_dict)
"""
import asyncio, sys, os, math
sys.path.insert(0, os.path.dirname(__file__))

from darwin_agent.interfaces.enums import *
from darwin_agent.interfaces.types import *
from darwin_agent.evolution.diagnostics import (
    EvolutionDiagnostics, DiagnosticsReport,
    _DIVERSITY_COLLAPSE_THRESHOLD,
    _CAPITAL_ALERT_TOP1_PCT,
    _EXPOSURE_ALERT_TOP1_PCT,
    _DOMINANCE_ALERT_SHARE,
    _STAGNATION_GENS,
)

PASS = FAIL = 0

def check(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ✅ {name}")
    else: FAIL += 1; print(f"  ❌ {name}: {detail}")


def _make_dna(genes, parent_id=None, dna_id=None):
    return DNAData(
        genes=genes, generation=0,
        parent_id=parent_id,
        dna_id=dna_id or f"dna-{id(genes) % 10000}",
    )

def _make_eval(agent_id, capital=100.0, pnl=10.0, exposure=None,
               dna=None, wins=10, losses=5, fitness=0.5):
    return AgentEvalData(
        metrics=AgentMetrics(
            agent_id=agent_id, capital=capital, realized_pnl=pnl,
            total_trades=wins+losses, winning_trades=wins, losing_trades=losses,
            fitness=fitness,
        ),
        initial_capital=100.0,
        exposure=exposure or {},
        dna=dna,
    )


def run_tests():
    global PASS, FAIL
    diag = EvolutionDiagnostics()

    # ════════════════════════════════════════════════════════
    # 1. Genetic diversity score
    # ════════════════════════════════════════════════════════
    print("\n🧬 1. Genetic diversity score")
    print("─" * 50)

    # Diverse population: all genes spread across ranges
    diverse_dna = [
        _make_dna({"risk_pct": 1.0, "stop_loss_pct": 1.0, "momentum_weight": 0.1,
                   "ema_weight": 0.1, "rsi_weight": 0.9, "atr_weight": 0.2}, dna_id="d1"),
        _make_dna({"risk_pct": 4.0, "stop_loss_pct": 3.0, "momentum_weight": 0.5,
                   "ema_weight": 0.5, "rsi_weight": 0.3, "atr_weight": 0.7}, dna_id="d2"),
        _make_dna({"risk_pct": 7.0, "stop_loss_pct": 4.5, "momentum_weight": 0.9,
                   "ema_weight": 0.9, "rsi_weight": 0.1, "atr_weight": 0.4}, dna_id="d3"),
        _make_dna({"risk_pct": 2.5, "stop_loss_pct": 2.0, "momentum_weight": 0.3,
                   "ema_weight": 0.3, "rsi_weight": 0.7, "atr_weight": 0.9}, dna_id="d4"),
        _make_dna({"risk_pct": 6.0, "stop_loss_pct": 3.5, "momentum_weight": 0.7,
                   "ema_weight": 0.7, "rsi_weight": 0.5, "atr_weight": 0.1}, dna_id="d5"),
    ]
    agents_diverse = [_make_eval(f"a{i}", dna=d) for i, d in enumerate(diverse_dna)]
    report_diverse = diag.diagnose(0, agents_diverse, diverse_dna)

    check("Diverse pop diversity > 0 (has some variation)",
          report_diverse.diversity.overall_diversity_score > 0.1,
          f"got {report_diverse.diversity.overall_diversity_score:.4f}")
    check("No collapse alert", not report_diverse.diversity.alert_collapsed)

    # Clone population: all identical
    diag.reset()
    clone_genes = {"risk_pct": 3.0, "stop_loss_pct": 2.0, "momentum_weight": 0.5}
    clone_dna = [_make_dna(dict(clone_genes), dna_id=f"c{i}") for i in range(5)]
    agents_clone = [_make_eval(f"c{i}", dna=d) for i, d in enumerate(clone_dna)]
    report_clone = diag.diagnose(0, agents_clone, clone_dna)

    check("Clone pop diversity ≈ 0",
          report_clone.diversity.overall_diversity_score < 0.05,
          f"got {report_clone.diversity.overall_diversity_score:.4f}")
    check("Clone pop triggers collapse alert",
          report_clone.diversity.alert_collapsed)
    check("Mean pairwise distance ≈ 0",
          report_clone.diversity.mean_pairwise_distance < 0.01,
          f"got {report_clone.diversity.mean_pairwise_distance:.4f}")

    # Relative check: diverse > clones
    check("Diverse pop diversity >> clone pop diversity",
          report_diverse.diversity.overall_diversity_score >
          report_clone.diversity.overall_diversity_score + 0.1,
          f"diverse={report_diverse.diversity.overall_diversity_score:.4f}, "
          f"clone={report_clone.diversity.overall_diversity_score:.4f}")
    print("  ✅ PASSED\n")

    # ════════════════════════════════════════════════════════
    # 2. Per-gene entropy
    # ════════════════════════════════════════════════════════
    print("🧬 2. Per-gene entropy")
    print("─" * 50)

    # Check that diverse_dna has higher entropy than clone_dna
    diverse_ent = report_diverse.diversity.gene_entropies
    clone_ent = report_clone.diversity.gene_entropies

    # Find risk_pct entropy in both
    div_risk = next((g for g in diverse_ent if g.gene_name == "risk_pct"), None)
    cln_risk = next((g for g in clone_ent if g.gene_name == "risk_pct"), None)

    check("risk_pct entropy: diverse > clone",
          div_risk and cln_risk and div_risk.entropy > cln_risk.entropy,
          f"diverse={div_risk.entropy:.4f}, clone={cln_risk.entropy:.4f}" if div_risk and cln_risk else "missing")

    check("Normalized entropy in [0, 1]",
          all(0.0 <= g.normalized <= 1.0 for g in diverse_ent))

    check("Range usage > 0 for diverse pop",
          div_risk and div_risk.range_usage_pct > 10,
          f"got {div_risk.range_usage_pct:.1f}%" if div_risk else "missing")

    check("Lowest entropy gene identified",
          report_diverse.diversity.lowest_entropy_gene != "")

    check("Mean entropy computed",
          report_diverse.diversity.mean_entropy > 0)
    print("  ✅ PASSED\n")

    # ════════════════════════════════════════════════════════
    # 3. Capital concentration index
    # ════════════════════════════════════════════════════════
    print("🧬 3. Capital concentration index")
    print("─" * 50)

    diag.reset()
    # Equal capital
    equal_agents = [_make_eval(f"eq{i}", capital=100.0, dna=_make_dna({}, dna_id=f"eq{i}")) for i in range(5)]
    report_eq = diag.diagnose(0, equal_agents)

    check("Equal capital: HHI ≈ 1/n = 0.2",
          abs(report_eq.concentration.capital_herfindahl - 0.2) < 0.01,
          f"got {report_eq.concentration.capital_herfindahl:.4f}")
    check("Equal capital: Gini ≈ 0",
          report_eq.concentration.capital_gini < 0.05,
          f"got {report_eq.concentration.capital_gini:.4f}")
    check("Equal capital: top1 = 20%",
          abs(report_eq.concentration.capital_top1_pct - 20.0) < 1.0,
          f"got {report_eq.concentration.capital_top1_pct:.1f}%")
    check("Equal capital: no alert", not report_eq.concentration.capital_alert)

    # Monopoly capital
    diag.reset()
    mono_agents = [
        _make_eval("rich", capital=900.0, dna=_make_dna({}, dna_id="rich")),
        _make_eval("poor1", capital=50.0, dna=_make_dna({}, dna_id="poor1")),
        _make_eval("poor2", capital=50.0, dna=_make_dna({}, dna_id="poor2")),
    ]
    report_mono = diag.diagnose(0, mono_agents)

    check("Monopoly: HHI > 0.8",
          report_mono.concentration.capital_herfindahl > 0.8,
          f"got {report_mono.concentration.capital_herfindahl:.4f}")
    check("Monopoly: top1 = 90%",
          report_mono.concentration.capital_top1_pct > 85,
          f"got {report_mono.concentration.capital_top1_pct:.1f}%")
    check("Monopoly: capital alert triggered",
          report_mono.concentration.capital_alert)
    print("  ✅ PASSED\n")

    # ════════════════════════════════════════════════════════
    # 4. Exposure concentration by symbol
    # ════════════════════════════════════════════════════════
    print("🧬 4. Exposure concentration by symbol")
    print("─" * 50)

    diag.reset()
    # All agents trade BTC only
    btc_agents = [
        _make_eval(f"btc{i}", exposure={"BTCUSDT": 1.0},
                   dna=_make_dna({}, dna_id=f"btc{i}"))
        for i in range(5)
    ]
    report_btc = diag.diagnose(0, btc_agents)

    check("All-BTC: exposure HHI = 1.0",
          abs(report_btc.concentration.exposure_herfindahl - 1.0) < 0.01,
          f"got {report_btc.concentration.exposure_herfindahl:.4f}")
    check("All-BTC: top symbol = BTCUSDT",
          report_btc.concentration.exposure_top1_symbol == "BTCUSDT")
    check("All-BTC: top1 = 100%",
          report_btc.concentration.exposure_top1_pct > 99)
    check("All-BTC: exposure alert triggered",
          report_btc.concentration.exposure_alert)

    # Diversified exposure
    diag.reset()
    div_agents = [
        _make_eval("a0", exposure={"BTCUSDT": 0.3, "ETHUSDT": 0.3, "SOLUSDT": 0.4},
                   dna=_make_dna({}, dna_id="a0")),
        _make_eval("a1", exposure={"AVAXUSDT": 0.5, "DOTUSDT": 0.5},
                   dna=_make_dna({}, dna_id="a1")),
    ]
    report_div = diag.diagnose(0, div_agents)

    check("Diversified: exposure HHI < 0.4",
          report_div.concentration.exposure_herfindahl < 0.4,
          f"got {report_div.concentration.exposure_herfindahl:.4f}")
    check("Diversified: no exposure alert",
          not report_div.concentration.exposure_alert)
    print("  ✅ PASSED\n")

    # ════════════════════════════════════════════════════════
    # 5. Evolutionary dominance
    # ════════════════════════════════════════════════════════
    print("🧬 5. Evolutionary dominance detection")
    print("─" * 50)

    diag.reset()
    # 4 out of 5 from same parent
    dom_dna = [
        _make_dna({}, parent_id="alpha", dna_id="d0"),
        _make_dna({}, parent_id="alpha", dna_id="d1"),
        _make_dna({}, parent_id="alpha", dna_id="d2"),
        _make_dna({}, parent_id="alpha", dna_id="d3"),
        _make_dna({}, parent_id="beta", dna_id="d4"),
    ]
    dom_agents = [_make_eval(f"d{i}", dna=d) for i, d in enumerate(dom_dna)]
    report_dom = diag.diagnose(0, dom_agents, dom_dna)

    check("Dominant lineage = alpha",
          report_dom.dominance.dominant_lineage_id == "alpha")
    check("Dominant share = 80%",
          abs(report_dom.dominance.dominant_lineage_share - 0.8) < 0.01,
          f"got {report_dom.dominance.dominant_lineage_share:.2f}")
    check("Unique lineages = 2", report_dom.dominance.unique_lineages == 2)
    check("Dominance alert triggered", report_dom.dominance.alert_dominant)
    check("Effective lineages < 2",
          report_dom.dominance.effective_lineages < 2.0,
          f"got {report_dom.dominance.effective_lineages:.2f}")

    # Balanced lineages
    diag.reset()
    bal_dna = [_make_dna({}, parent_id=f"parent-{i}", dna_id=f"b{i}") for i in range(5)]
    bal_agents = [_make_eval(f"b{i}", dna=d) for i, d in enumerate(bal_dna)]
    report_bal = diag.diagnose(0, bal_agents, bal_dna)

    check("Balanced: no dominance alert", not report_bal.dominance.alert_dominant)
    check("Balanced: effective lineages = 5",
          abs(report_bal.dominance.effective_lineages - 5.0) < 0.1,
          f"got {report_bal.dominance.effective_lineages:.2f}")
    print("  ✅ PASSED\n")

    # ════════════════════════════════════════════════════════
    # 6. Fitness variance across generations
    # ════════════════════════════════════════════════════════
    print("🧬 6. Fitness variance tracking")
    print("─" * 50)

    diag.reset()
    # Simulate improving generations
    for gen in range(8):
        agents = [
            _make_eval(f"g{gen}-{i}",
                       fitness=0.3 + gen * 0.05 + i * 0.02,
                       dna=_make_dna({}, dna_id=f"g{gen}-{i}"))
            for i in range(5)
        ]
        report = diag.diagnose(gen, agents)

    check("Improving: avg_trend > 0",
          report.fitness.avg_trend_5gen > 0,
          f"got {report.fitness.avg_trend_5gen:.6f}")
    check("Improving flag set", report.fitness.improving)
    check("Not stagnating", not report.fitness.stagnating)
    check("All-time best tracked",
          report.fitness.all_time_best > 0)
    check("Variance computed",
          report.fitness.current_variance > 0)
    check("Spread = best - worst",
          abs(report.fitness.current_spread -
              (report.fitness.current_best - report.fitness.current_worst)) < 1e-6)

    # Simulate stagnation
    diag.reset()
    for gen in range(10):
        agents = [
            _make_eval(f"s{gen}-{i}", fitness=0.5,
                       dna=_make_dna({}, dna_id=f"s{gen}-{i}"))
            for i in range(5)
        ]
        report = diag.diagnose(gen, agents)

    check("Stagnation detected after 10 flat gens",
          report.fitness.stagnating,
          f"stagnation_gens={report.fitness.stagnation_generations}")
    check(f"Stagnation count >= {_STAGNATION_GENS}",
          report.fitness.stagnation_generations >= _STAGNATION_GENS)
    print("  ✅ PASSED\n")

    # ════════════════════════════════════════════════════════
    # 7. Structured diagnostics report
    # ════════════════════════════════════════════════════════
    print("🧬 7. Structured report + to_dict()")
    print("─" * 50)

    diag.reset()
    agents = [
        _make_eval(f"r{i}", capital=50+i*50, fitness=0.3+i*0.1,
                   exposure={"BTCUSDT": 0.5, "ETHUSDT": 0.5},
                   dna=_make_dna({"risk_pct": 1+i, "momentum_weight": i*0.2},
                                 parent_id="root", dna_id=f"r{i}"))
        for i in range(5)
    ]
    report = diag.diagnose(0, agents)

    check("Report has generation", report.generation == 0)
    check("Report has health_score in [0,1]",
          0.0 <= report.health_score <= 1.0,
          f"got {report.health_score}")

    d = report.to_dict()
    check("to_dict has all top-level keys",
          all(k in d for k in ["generation", "health_score", "alerts",
                               "diversity", "concentration", "dominance", "fitness"]))
    check("to_dict diversity has gene_entropies",
          "gene_entropies" in d["diversity"])
    check("to_dict concentration has exposure_by_symbol",
          "exposure_by_symbol" in d["concentration"])
    check("to_dict fitness has all_time_best",
          "all_time_best" in d["fitness"])
    check("to_dict is JSON-safe (no objects)",
          all(isinstance(v, (int, float, str, bool, list, dict, type(None)))
              for v in d.values()))
    print("  ✅ PASSED\n")

    # ════════════════════════════════════════════════════════
    # 8. Edge cases
    # ════════════════════════════════════════════════════════
    print("🧬 8. Edge cases")
    print("─" * 50)

    diag.reset()

    # Empty
    empty = diag.diagnose(0, [])
    check("Empty: report valid", empty.generation == 0)
    check("Empty: diversity = 0", empty.diversity.overall_diversity_score == 0)

    # Single agent
    single = diag.diagnose(0, [_make_eval("solo", dna=_make_dna({}, dna_id="solo"))])
    check("Single: diversity = 0", single.diversity.overall_diversity_score == 0)
    check("Single: HHI = 1.0 (monopoly)",
          abs(single.concentration.capital_herfindahl - 1.0) < 0.01,
          f"got {single.concentration.capital_herfindahl:.4f}")
    check("Single: effective lineages = 1",
          abs(single.dominance.effective_lineages - 1.0) < 0.01)

    # Agents with no exposure
    no_exp = [_make_eval(f"ne{i}", exposure={},
                         dna=_make_dna({}, dna_id=f"ne{i}")) for i in range(3)]
    report_ne = diag.diagnose(0, no_exp)
    check("No exposure: HHI = 0", report_ne.concentration.exposure_herfindahl == 0)
    check("No exposure: no alert", not report_ne.concentration.exposure_alert)

    # Agents with zero capital
    zero_cap = [_make_eval(f"zc{i}", capital=0.0,
                           dna=_make_dna({}, dna_id=f"zc{i}")) for i in range(3)]
    report_zc = diag.diagnose(0, zero_cap)
    check("Zero capital: no crash", report_zc.concentration.capital_herfindahl >= 0)
    print("  ✅ PASSED\n")

    # ════════════════════════════════════════════════════════
    # 9. Alert generation
    # ════════════════════════════════════════════════════════
    print("🧬 9. Alert generation")
    print("─" * 50)

    diag.reset()
    # Build a pathological population: clones, monopoly, single lineage, all BTC
    pathological = [
        _make_eval("p0", capital=950, exposure={"BTCUSDT": 1.0},
                   dna=_make_dna({"risk_pct": 3.0}, parent_id="boss", dna_id="p0")),
        _make_eval("p1", capital=25, exposure={"BTCUSDT": 1.0},
                   dna=_make_dna({"risk_pct": 3.0}, parent_id="boss", dna_id="p1")),
        _make_eval("p2", capital=25, exposure={"BTCUSDT": 1.0},
                   dna=_make_dna({"risk_pct": 3.0}, parent_id="boss", dna_id="p2")),
    ]
    # Run enough gens to trigger stagnation
    for gen in range(8):
        report = diag.diagnose(gen, pathological,
                               [a.dna for a in pathological])

    alerts = report.alerts
    check("GENETIC_COLLAPSE alert present",
          any("GENETIC_COLLAPSE" in a for a in alerts),
          f"alerts: {alerts}")
    check("CAPITAL_CONCENTRATION alert present",
          any("CAPITAL_CONCENTRATION" in a for a in alerts),
          f"alerts: {alerts}")
    check("EXPOSURE_CONCENTRATION alert present",
          any("EXPOSURE_CONCENTRATION" in a for a in alerts),
          f"alerts: {alerts}")
    check("LINEAGE_DOMINANCE alert present",
          any("LINEAGE_DOMINANCE" in a for a in alerts),
          f"alerts: {alerts}")
    check("FITNESS_STAGNATION alert present",
          any("FITNESS_STAGNATION" in a for a in alerts),
          f"alerts: {alerts}")
    check("Health score very low", report.health_score < 0.3,
          f"got {report.health_score}")
    print("  ✅ PASSED\n")

    # ════════════════════════════════════════════════════════
    # 10. Fitness history for dashboard
    # ════════════════════════════════════════════════════════
    print("🧬 10. Fitness history")
    print("─" * 50)

    diag.reset()
    for gen in range(5):
        agents = [_make_eval(f"h{gen}-{i}", fitness=0.3+gen*0.1,
                             dna=_make_dna({}, dna_id=f"h{gen}-{i}"))
                  for i in range(3)]
        diag.diagnose(gen, agents)

    history = diag.get_fitness_history()
    check("History has 5 entries", len(history) == 5)
    check("Each entry has gen/best/avg/worst/variance",
          all(all(k in h for k in ["generation", "best", "avg", "worst", "variance"])
              for h in history))
    check("Generations ascending",
          all(history[i]["generation"] < history[i+1]["generation"]
              for i in range(len(history)-1)))
    print("  ✅ PASSED\n")

    # ════════════════════════════════════════════════════════
    # Summary
    # ════════════════════════════════════════════════════════
    print("═" * 60)
    if FAIL == 0:
        print(f"  ✅ ALL {PASS} DIAGNOSTICS TESTS PASSED")
    else:
        print(f"  ❌ {FAIL} FAILED, {PASS} passed")
    print("═" * 60)
    return FAIL == 0


if __name__ == "__main__":
    ok = run_tests()
    sys.exit(0 if ok else 1)
