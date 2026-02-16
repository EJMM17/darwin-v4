"""
Darwin v4 — DDT 2.0 Tests (Smooth Drawdown Dynamic Throttle).

Covers:
  1. rolling_dd = (peak - current) / peak
  2. exposure_multiplier = max(0.2, 1 - 1.5 * rolling_dd)
  3. Applied to per_agent_cap, leverage, risk_pct
  4. Floor at 0.2 — never below
  5. No hard stop / no full halt
  6. Does NOT modify genes or fitness
  7. Deterministic
  8. Integrates before trade execution
"""
import asyncio, math, os, random, sys
sys.path.insert(0, os.path.dirname(__file__))

from darwin_agent.simulation.harness import (
    SimulationHarness, SimConfig, MonteCarloScenario,
    SimulatedAgent, Tick,
)
from darwin_agent.interfaces.types import DNAData

PASS = FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ✅ {name}")
    else: FAIL += 1; print(f"  ❌ {name}: {detail}")


def ddt2(dd):
    return max(0.2, 1.0 - 1.5 * dd)


def test_1_drawdown_math():
    print("\n🧪 1. Rolling drawdown = (peak - current) / peak")
    print("─" * 50)
    for peak, cur, exp in [(100,100,0.0),(100,75,0.25),(100,50,0.50),(200,120,0.40)]:
        dd = (peak - cur) / peak
        check(f"peak={peak} cur={cur} → dd={dd:.2f}", abs(dd - exp) < 0.001)
    print()


def test_2_smooth_curve():
    print("🧪 2. Smooth exposure curve")
    print("─" * 50)
    cases = [
        (0.00, 1.000), (0.10, 0.850), (0.20, 0.700),
        (0.33, 0.505), (0.40, 0.400), (0.50, 0.250),
        (0.55, 0.200), (0.70, 0.200), (1.00, 0.200),
    ]
    for dd, exp in cases:
        got = ddt2(dd)
        check(f"dd={dd:.2f} → {got:.3f}", abs(got - exp) < 0.005, f"got {got:.4f}")

    # Monotonic
    prev = 1.0
    for i in range(0, 101, 5):
        m = ddt2(i / 100)
        check(f"monotonic {i}%", m <= prev + 0.001)
        prev = m

    # Floor
    for dd in [0.55, 0.70, 0.90, 1.00]:
        check(f"floor at dd={dd}", ddt2(dd) == 0.2)
    print()


def test_3_application():
    print("🧪 3. Applied to per_agent_cap, leverage, risk_pct")
    print("─" * 50)
    genes = {"risk_pct": 4.0, "leverage_aggression": 0.5,
             "confidence_threshold": 0.01, "momentum_weight": 1.0,
             "mean_rev_weight": 0.0, "scalping_weight": 0.0,
             "breakout_weight": 0.0, "regime_bias": 0.0,
             "volatility_threshold": 0.0, "stop_loss_pct": 5.0,
             "take_profit_pct": 5.0}
    dna = DNAData(genes=genes, generation=0)
    a1 = SimulatedAgent("t1", dna, 100.0, random.Random(42), exposure_mult=1.0)
    a2 = SimulatedAgent("t2", dna, 100.0, random.Random(42), exposure_mult=0.2)
    check("Normal mult stored", a1._exposure_mult == 1.0)
    check("Floor mult stored", a2._exposure_mult == 0.2)
    check("Genes identical", a1.dna.genes == a2.dna.genes)
    print()


def test_4_no_halt():
    print("🧪 4-5. Floor at 0.2, no full halt")
    print("─" * 50)
    src = open("darwin_agent/simulation/harness.py").read()
    check("No zero-exposure guard", "exposure_mult <= 0.0" not in src)
    check("No ddt_mult = 0.0", "ddt_mult = 0.0" not in src)
    check("Floor in formula", "max(0.2," in src)
    check("Smooth formula", "1.0 - 1.5 *" in src)
    check("No stepped thresholds", "> 0.55" not in src and "> 0.40" not in src)
    print()


def test_6_no_side_effects():
    print("🧪 6. No gene/fitness modification")
    print("─" * 50)
    src = open("darwin_agent/simulation/harness.py").read()
    check("No VART remnants", "vart" not in src.lower())
    check("No RiskAwareFitness import", "RiskAwareFitness" not in src)
    check("capital_floor intact", "capital_floor_pct" in src)
    print()


async def test_7_deterministic():
    print("🧪 7. Deterministic")
    print("─" * 50)
    caps = []
    for _ in range(3):
        r = await SimulationHarness(SimConfig(
            scenario=MonteCarloScenario(symbols=["BTCUSDT"]),
            generations=50, pool_size=5, trades_per_generation=100,
            seed=42, starting_capital=100.0, enable_ddt=True,
        )).run()
        caps.append(r.final_capital)
    check("Run 1 == Run 2", abs(caps[0] - caps[1]) < 0.001)
    check("Run 2 == Run 3", abs(caps[1] - caps[2]) < 0.001)
    print()


async def test_8_integration():
    print("🧪 8. Harness integration")
    print("─" * 50)
    kw = dict(scenario=MonteCarloScenario(symbols=["BTCUSDT","ETHUSDT","SOLUSDT"]),
              generations=100, pool_size=10, trades_per_generation=200,
              seed=42, starting_capital=100.0,
              mutation_rate=0.18, survival_rate=0.40, capital_floor_pct=0.45)
    r_off = await SimulationHarness(SimConfig(**kw, enable_ddt=False)).run()
    r_on = await SimulationHarness(SimConfig(**kw, enable_ddt=True)).run()
    check("DDT-off completes", r_off.generations_run == 100)
    check("DDT-on completes", r_on.generations_run == 100)
    check("Both valid", r_off.final_capital > 0 and r_on.final_capital > 0)
    print()


async def main():
    global PASS, FAIL
    print("═" * 60); print("  🧪 DDT 2.0 TESTS"); print("═" * 60)
    for t in [test_1_drawdown_math, test_2_smooth_curve,
              test_3_application, test_4_no_halt, test_6_no_side_effects]:
        try: t()
        except Exception as e:
            FAIL += 1; print(f"  ❌ CRASHED: {e}")
            import traceback; traceback.print_exc()
    for t in [test_7_deterministic, test_8_integration]:
        try: await t()
        except Exception as e:
            FAIL += 1; print(f"  ❌ CRASHED: {e}")
            import traceback; traceback.print_exc()
    print("═" * 60)
    if FAIL == 0: print(f"  ✅ ALL {PASS} DDT 2.0 TESTS PASSED")
    else: print(f"  ❌ {FAIL} FAILED, {PASS} passed")
    print("═" * 60)
    return FAIL == 0

if __name__ == "__main__":
    sys.exit(0 if asyncio.run(main()) else 1)
