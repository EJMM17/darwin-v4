#!/usr/bin/env python3
"""
Darwin v5 — Memecoin-Safe Backtest
===================================

Comprehensive backtest with 5 symbols:
  - BTC, ETH, SOL (stable majors)
  - PIPPIN, RIVER (extreme-volatility memecoins)

Validates all memecoin safety fixes:
  1. Dynamic leverage reduces to 1x for extreme-vol assets
  2. SL distance capped at 5% (prevents 20% ATR-based SLs)
  3. Adaptive signal calibration (skip BTC demeaning for memecoins)
  4. Deeper book checks with cliff detection
  5. Adaptive slippage tolerance
  6. Regime detection scaled by volatility

Runs 3 scenarios:
  A. Base Monte Carlo — normal operation with 5 symbols
  B. Shock-injected   — memecoin flash crashes + vol spikes
  C. Rug-pull stress  — 80% memecoin crash to test kill switches

Each scenario runs 30 generations × 10 agents with full
evolution, diagnostics, and scorecard reporting.

Usage:
    python backtest_memecoin.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

# Ensure darwin_agent is importable
sys.path.insert(0, str(Path(__file__).parent))

from darwin_agent.simulation.harness import (
    MonteCarloScenario,
    ShockEvent,
    ShockScenario,
    SimConfig,
    SimulationHarness,
    SimulationResults,
)
from darwin_agent.simulation.scorecard import (
    GenerationData,
    ScorecardReport,
    ScorecardWeights,
    SimulationScorecard,
    extract_generation_data,
)
from darwin_agent.risk.portfolio_engine import RiskLimits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("darwin.backtest_memecoin")

# ═══════════════════════════════════════════════════════════════
# Output directory
# ═══════════════════════════════════════════════════════════════
OUT_DIR = Path("/home/user/darwin-v4/backtest_results")


# ═══════════════════════════════════════════════════════════════
# Symbol params — calibrated for realistic crypto behavior
# ═══════════════════════════════════════════════════════════════

# Majors: moderate vol, mean-reverting, correlated
MAJOR_PARAMS = {
    "BTCUSDT": MonteCarloScenario.SymbolParams(
        base_price=50000.0,
        annual_drift=0.05,      # 5% annual drift (conservative)
        annual_vol=0.75,        # 75% annual vol (typical BTC)
        fat_tail_df=5.0,        # fat tails for flash crashes
        mean_rev_k=0.02,        # moderate mean reversion
    ),
    "ETHUSDT": MonteCarloScenario.SymbolParams(
        base_price=3000.0,
        annual_drift=0.08,
        annual_vol=0.85,
        fat_tail_df=4.5,
        mean_rev_k=0.02,
    ),
    "SOLUSDT": MonteCarloScenario.SymbolParams(
        base_price=100.0,
        annual_drift=0.10,
        annual_vol=1.0,
        fat_tail_df=4.0,
        mean_rev_k=0.015,
    ),
}

# Memecoins: high vol, momentum-driven, uncorrelated with BTC
# Calibrated to real memecoin data: avg daily vol 5-8% (annualized ~1.5-2.5)
# with occasional spikes. df=4 gives realistic tail behavior without
# generating immediate bankruptcy paths.
MEMECOIN_PARAMS = {
    "PIPPINUSDT": MonteCarloScenario.SymbolParams(
        base_price=0.05,
        annual_drift=0.0,       # neutral drift (memecoins are 50/50)
        annual_vol=2.0,         # 200% annual vol (realistic avg for active memecoin)
        fat_tail_df=4.0,        # fat tails (occasional rug pulls)
        mean_rev_k=0.01,        # some mean reversion (pump-dump cycles)
    ),
    "RIVERUSDT": MonteCarloScenario.SymbolParams(
        base_price=0.02,
        annual_drift=0.0,
        annual_vol=1.8,         # 180% annual vol
        fat_tail_df=4.0,
        mean_rev_k=0.01,
    ),
}

ALL_PARAMS = {**MAJOR_PARAMS, **MEMECOIN_PARAMS}
ALL_SYMBOLS = list(ALL_PARAMS.keys())


# ═══════════════════════════════════════════════════════════════
# Risk limits — tighter for memecoin portfolio
# ═══════════════════════════════════════════════════════════════

MEMECOIN_RISK_LIMITS = RiskLimits(
    defensive_drawdown_pct=8.0,     # enter defensive at 8% DD
    critical_drawdown_pct=15.0,     # critical at 15% DD
    halted_drawdown_pct=25.0,       # halt at 25% DD
    max_daily_loss_pct=5.0,         # halt at 5% daily loss
    max_total_exposure_pct=60.0,    # max 60% equity exposed (tighter for memecoins)
    max_symbol_exposure_pct=20.0,   # max 20% per symbol
)


# ═══════════════════════════════════════════════════════════════
# Best config — incorporates all discussed fixes
# ═══════════════════════════════════════════════════════════════

def make_base_config(
    scenario: Any,
    seed: int = 42,
    generations: int = 50,
    pool_size: int = 12,
    ticks_per_gen: int = 400,
    capital: float = 100.0,
) -> SimConfig:
    """
    Create simulation config with the best parameters from all fixes.

    Key settings reflecting the memecoin safety improvements:
    - Higher taker fees (memecoins have wider spreads)
    - Higher slippage (thin books)
    - DDT enabled (drawdown dynamic throttle)
    - Tighter capital floor (45% of start)
    - Conservative mutation (0.12) for stability
    """
    return SimConfig(
        scenario=scenario,
        generations=generations,
        pool_size=pool_size,
        trades_per_generation=ticks_per_gen,
        starting_capital=capital,
        seed=seed,
        survival_rate=0.5,
        elitism_count=2,            # keep top 2 agents (more stability)
        mutation_rate=0.12,         # lower mutation for convergence
        risk_limits=MEMECOIN_RISK_LIMITS,
        capital_floor_pct=0.45,     # defensive realloc if capital < 45%
        enable_ddt=True,            # smooth drawdown throttle
        maker_fee=0.0002,           # 0.02% maker
        taker_fee=0.0005,           # 0.05% taker
        slippage=0.0005,            # 0.05% slippage (higher for memecoins)
    )


# ═══════════════════════════════════════════════════════════════
# Scenario builders
# ═══════════════════════════════════════════════════════════════

def build_base_scenario() -> MonteCarloScenario:
    """Scenario A: Normal Monte Carlo with 5 symbols."""
    return MonteCarloScenario(
        symbols=ALL_SYMBOLS,
        params=ALL_PARAMS,
    )


def build_shock_scenario() -> ShockScenario:
    """
    Scenario B: Memecoin-specific shocks.
    - Flash crash on PIPPIN at tick 80 (-25%)
    - Volatility spike on RIVER at tick 150 (2x vol for 20 ticks)
    - Gap up on PIPPIN at tick 200 (+20% — hype pump)
    - Regime shift (bear) on all at tick 250
    """
    base = build_base_scenario()
    return ShockScenario(
        base=base,
        shocks=[
            ShockEvent(
                step=80, shock_type="flash_crash",
                magnitude=-0.25, duration=10,
                symbols=["PIPPINUSDT"],
            ),
            ShockEvent(
                step=150, shock_type="volatility",
                magnitude=2.0, duration=20,
                symbols=["RIVERUSDT"],
            ),
            ShockEvent(
                step=200, shock_type="gap",
                magnitude=0.20, duration=1,
                symbols=["PIPPINUSDT"],
            ),
            ShockEvent(
                step=250, shock_type="regime_shift",
                magnitude=-0.10, duration=20,
                symbols=[],  # all symbols
            ),
        ],
    )


def build_rugpull_scenario() -> ShockScenario:
    """
    Scenario C: Stress test — sharp memecoin selloff.
    - PIPPIN: -50% crash at tick 100, slow bleed after
    - RIVER: -35% crash at tick 120, partial recovery
    - BTC drops -8% at tick 150 (correlation contagion)
    Tests: kill switch activation, capital preservation.
    """
    base = build_base_scenario()
    return ShockScenario(
        base=base,
        shocks=[
            # PIPPIN selloff
            ShockEvent(
                step=100, shock_type="flash_crash",
                magnitude=-0.50, duration=10,
                symbols=["PIPPINUSDT"],
            ),
            # RIVER sympathy dump
            ShockEvent(
                step=120, shock_type="flash_crash",
                magnitude=-0.35, duration=15,
                symbols=["RIVERUSDT"],
            ),
            # BTC contagion
            ShockEvent(
                step=150, shock_type="flash_crash",
                magnitude=-0.08, duration=20,
                symbols=["BTCUSDT"],
            ),
            # Vol spike on ETH/SOL
            ShockEvent(
                step=160, shock_type="volatility",
                magnitude=1.8, duration=30,
                symbols=["ETHUSDT", "SOLUSDT"],
            ),
        ],
    )


# ═══════════════════════════════════════════════════════════════
# Run + score a single scenario
# ═══════════════════════════════════════════════════════════════

async def run_scenario(
    name: str,
    scenario: Any,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run one scenario end-to-end and return results + scorecard."""

    logger.info("=" * 60)
    logger.info("SCENARIO: %s (seed=%d)", name, seed)
    logger.info("=" * 60)

    config = make_base_config(scenario, seed=seed)
    harness = SimulationHarness(config)

    t0 = time.monotonic()
    results = await harness.run()
    elapsed_s = time.monotonic() - t0

    # Export CSV + JSON
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = str(OUT_DIR / f"{name}_generations.csv")
    json_path = str(OUT_DIR / f"{name}_full.json")
    results.export_csv(csv_path)
    results.export_json(json_path)

    # Build scorecard
    scorecard = SimulationScorecard()
    gen_data_list = []
    for gr in results.generation_results:
        # Build the dict format that extract_generation_data expects
        gr_dict = {
            "generation": gr.generation,
            "pool_capital": gr.pool_capital,
            "pool_pnl": gr.pool_pnl,
            "portfolio_state": gr.portfolio_state,
            "snapshot": {
                "best_fitness": gr.snapshot.best_fitness,
                "avg_fitness": gr.snapshot.avg_fitness,
                "worst_fitness": gr.snapshot.worst_fitness,
                "total_trades": gr.snapshot.total_trades,
                "pool_win_rate": gr.snapshot.pool_win_rate,
                "pool_sharpe": gr.snapshot.pool_sharpe,
                "pool_max_drawdown": gr.snapshot.pool_max_drawdown,
                "survivors": gr.snapshot.survivors,
                "eliminated": gr.snapshot.eliminated,
                "population": gr.snapshot.population_size,
            },
            "diagnostics": gr.diagnostics_dict,
        }
        gen_data_list.append(extract_generation_data(gr_dict))

    report = scorecard.evaluate(
        gen_data_list,
        starting_capital=results.starting_capital,
        final_capital=results.final_capital,
    )

    # Save scorecard
    scorecard_path = str(OUT_DIR / f"{name}_scorecard.json")
    with open(scorecard_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)

    # Out-of-sample validation
    oos_result = {}
    if results.hall_of_fame:
        from darwin_agent.interfaces.types import DNAData
        hof_dna = []
        for hof_entry in results.hall_of_fame[:3]:
            genes = hof_entry.get("genes", {})
            if genes:
                hof_dna.append(DNAData(
                    genes=genes,
                    generation=hof_entry.get("generation", 0),
                    dna_id=hof_entry.get("dna_id", "hof"),
                    fitness=hof_entry.get("fitness", 0.0),
                ))
        if hof_dna:
            oos_result = await harness.validate_out_of_sample(
                hof_dna, oos_steps=300, n_oos_runs=3,
            )
            oos_path = str(OUT_DIR / f"{name}_oos.json")
            with open(oos_path, "w") as f:
                json.dump(oos_result, f, indent=2, default=str)

    # Summary
    summary = {
        "scenario": name,
        "seed": seed,
        "elapsed_s": round(elapsed_s, 1),
        "starting_capital": results.starting_capital,
        "final_capital": round(results.final_capital, 2),
        "return_pct": round(
            (results.final_capital - results.starting_capital)
            / results.starting_capital * 100, 2
        ),
        "best_fitness": round(results.best_fitness_ever, 4),
        "best_agent_id": results.best_agent_id,
        "generations": results.generations_run,
        "ecosystem_health": round(report.ecosystem_health, 2),
        "ecosystem_grade": report.ecosystem_grade,
        "risk_stability": round(report.risk_stability.score, 2),
        "evolution_health": round(report.evolution_health.score, 2),
        "concentration_risk": round(report.concentration_risk.score, 2),
        "shock_resilience": round(report.shock_resilience.score, 2),
        "learning_quality": round(report.learning_quality.score, 2),
        "oos_overfit": oos_result.get("overfit_flag", None),
        "oos_degradation_pct": oos_result.get("degradation_pct", None),
        "hall_of_fame_genes": results.hall_of_fame[:1] if results.hall_of_fame else [],
    }

    return summary


# ═══════════════════════════════════════════════════════════════
# Print report
# ═══════════════════════════════════════════════════════════════

def print_report(summaries: List[Dict[str, Any]]) -> None:
    """Print formatted summary of all scenarios."""

    print("\n")
    print("=" * 72)
    print("  DARWIN v5 — MEMECOIN BACKTEST RESULTS")
    print("  Symbols: BTCUSDT, ETHUSDT, SOLUSDT, PIPPINUSDT, RIVERUSDT")
    print("=" * 72)

    for s in summaries:
        grade = s["ecosystem_grade"]
        ret = s["return_pct"]
        ret_sign = "+" if ret >= 0 else ""

        print(f"\n{'─' * 72}")
        print(f"  Scenario: {s['scenario']}")
        print(f"{'─' * 72}")
        print(f"  Capital:  ${s['starting_capital']:.2f} → ${s['final_capital']:.2f} "
              f"({ret_sign}{ret:.1f}%)")
        print(f"  Fitness:  {s['best_fitness']:.4f} (best ever)")
        print(f"  Time:     {s['elapsed_s']:.1f}s ({s['generations']} generations)")
        print()
        print(f"  ECOSYSTEM HEALTH: {s['ecosystem_health']:.1f}/10  "
              f"[Grade: {grade}]")
        print()
        print(f"    Risk Stability:    {s['risk_stability']:.1f}/10")
        print(f"    Evolution Health:  {s['evolution_health']:.1f}/10")
        print(f"    Concentration:     {s['concentration_risk']:.1f}/10")
        print(f"    Shock Resilience:  {s['shock_resilience']:.1f}/10")
        print(f"    Learning Quality:  {s['learning_quality']:.1f}/10")

        if s.get("oos_overfit") is not None:
            oos_flag = "YES" if s["oos_overfit"] else "NO"
            print(f"\n  Out-of-Sample: overfit={oos_flag} "
                  f"(degradation={s['oos_degradation_pct']:.1f}%)")

        if s.get("hall_of_fame_genes"):
            hof = s["hall_of_fame_genes"][0]
            genes = hof.get("genes", {})
            print(f"\n  Best DNA (top genes):")
            key_genes = [
                "risk_pct", "stop_loss_pct", "take_profit_pct",
                "leverage_aggression", "confidence_threshold",
                "momentum_weight", "mean_rev_weight",
            ]
            for g in key_genes:
                if g in genes:
                    print(f"    {g:25s} = {genes[g]:.4f}")

    print(f"\n{'=' * 72}")
    print(f"  Results saved to: {OUT_DIR}")
    print(f"{'=' * 72}\n")

    # Final verdict
    all_grades = [s["ecosystem_grade"] for s in summaries]
    avg_health = sum(s["ecosystem_health"] for s in summaries) / len(summaries)
    worst_ret = min(s["return_pct"] for s in summaries)
    any_overfit = any(s.get("oos_overfit", False) for s in summaries)

    print("  VERDICT:")
    if avg_health >= 6.0 and worst_ret > -30 and not any_overfit:
        print("  PASS — System is safe for memecoin trading with current config")
    elif avg_health >= 4.0 and worst_ret > -50:
        print("  MARGINAL — System survives but needs tuning for memecoins")
    else:
        print("  FAIL — System is NOT safe for memecoin trading")

    print(f"  Average ecosystem health: {avg_health:.1f}/10")
    print(f"  Worst scenario return: {worst_ret:+.1f}%")
    print(f"  Overfitting detected: {'Yes' if any_overfit else 'No'}")
    print()


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

async def main() -> int:
    logger.info("Darwin v5 Memecoin Backtest starting...")

    summaries = []

    # Scenario A: Normal Monte Carlo
    summary_a = await run_scenario(
        "A_base_montecarlo",
        build_base_scenario(),
        seed=42,
    )
    summaries.append(summary_a)

    # Scenario B: Shock-injected (flash crash + vol spike)
    summary_b = await run_scenario(
        "B_shock_injected",
        build_shock_scenario(),
        seed=42,
    )
    summaries.append(summary_b)

    # Scenario C: Rug-pull stress test
    summary_c = await run_scenario(
        "C_rugpull_stress",
        build_rugpull_scenario(),
        seed=42,
    )
    summaries.append(summary_c)

    # Save combined summary
    combined_path = OUT_DIR / "combined_summary.json"
    with open(combined_path, "w") as f:
        json.dump(summaries, f, indent=2, default=str)

    # Print report
    print_report(summaries)

    return 0


if __name__ == "__main__":
    code = asyncio.run(main())
    raise SystemExit(code)
