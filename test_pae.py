"""Unit tests for Portfolio Allocation Engine."""
import sys, math
sys.path.insert(0, ".")
from darwin_agent.portfolio.allocation_engine import (
    PortfolioAllocationEngine, PAEConfig, SymbolState,
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

# ── 1. Initialization ────────────────────────────────────────

pae = PortfolioAllocationEngine(["BTC", "SOL", "DOGE"])
w = pae.get_weights()
test("init_equal_weights", abs(w["BTC"] - 1/3) < 1e-6)
test("init_sum_to_one", abs(sum(w.values()) - 1.0) < 1e-6)
test("init_bar_zero", pae.bar == 0)

# ── 2. Warmup: equal weights during warmup ──────────────────

pae2 = PortfolioAllocationEngine(["A", "B"], config=PAEConfig(warmup_bars=10))
for i in range(5):
    pae2.update("A", equity=100 + i * 5, pnl=2.0)
    pae2.update("B", equity=100 - i * 2, pnl=-1.0)
    w = pae2.step()
test("warmup_equal", abs(w["A"] - 0.5) < 1e-6)
test("warmup_equal_b", abs(w["B"] - 0.5) < 1e-6)

# ── 3. After warmup: dynamic allocation ─────────────────────

pae3 = PortfolioAllocationEngine(
    ["WIN", "LOSE"],
    config=PAEConfig(warmup_bars=5, lookback=20, ema_alpha=1.0),
)
# Feed 30 bars: WIN goes up, LOSE goes down
for i in range(30):
    pae3.update("WIN", equity=100 + i * 2.0, pnl=1.0 if i % 3 == 0 else 0)
    pae3.update("LOSE", equity=100 - i * 1.5, pnl=-0.5 if i % 3 == 0 else 0)
    pae3.step()

w3 = pae3.get_weights()
test("winner_gets_more", w3["WIN"] > w3["LOSE"])
test("sum_to_one_dynamic", abs(sum(w3.values()) - 1.0) < 1e-6)
test("min_floor_respected", w3["LOSE"] >= 0.10 - 1e-6)
test("max_ceil_respected", w3["WIN"] <= 0.60 + 1e-6)

# ── 4. Constraint enforcement ────────────────────────────────

pae4 = PortfolioAllocationEngine(
    ["A", "B", "C"],
    config=PAEConfig(warmup_bars=3, lookback=10, ema_alpha=1.0,
                     min_weight=0.15, max_weight=0.50),
)
# A dominates, B/C flat
for i in range(20):
    pae4.update("A", equity=100 + i * 10, pnl=5.0)
    pae4.update("B", equity=100.0, pnl=0.0)
    pae4.update("C", equity=100 - i * 0.1, pnl=-0.05)
    pae4.step()

w4 = pae4.get_weights()
test("max_weight_clamped", w4["A"] <= 0.50 + 1e-6)
test("min_weight_b", w4["B"] >= 0.15 - 1e-6)
test("min_weight_c", w4["C"] >= 0.15 - 1e-6)
test("constraint_sum", abs(sum(w4.values()) - 1.0) < 1e-6)

# ── 5. Determinism ───────────────────────────────────────────

def run_determinism():
    results = []
    for _ in range(2):
        p = PortfolioAllocationEngine(
            ["X", "Y"],
            config=PAEConfig(warmup_bars=3, lookback=10, ema_alpha=0.1),
        )
        for i in range(20):
            p.update("X", equity=100 + i * 1.5, pnl=0.3)
            p.update("Y", equity=100 - i * 0.5, pnl=-0.1)
            p.step()
        results.append(p.get_weights())
    return results

d = run_determinism()
test("determinism_x", abs(d[0]["X"] - d[1]["X"]) < 1e-12)
test("determinism_y", abs(d[0]["Y"] - d[1]["Y"]) < 1e-12)

# ── 6. Capital allocation ───────────────────────────────────

pae6 = PortfolioAllocationEngine(["A", "B"])
alloc = pae6.get_capital_allocation(1000.0)
test("capital_sum", abs(alloc["A"] + alloc["B"] - 1000.0) < 1e-6)
test("capital_equal", abs(alloc["A"] - 500.0) < 1e-6)

# ── 7. EMA smoothing ────────────────────────────────────────

pae7 = PortfolioAllocationEngine(
    ["FAST", "SLOW"],
    config=PAEConfig(warmup_bars=3, lookback=10, ema_alpha=0.02),
)
for i in range(50):
    # FAST wins big early, then reverses
    if i < 25:
        pae7.update("FAST", equity=100 + i * 3, pnl=1.5)
        pae7.update("SLOW", equity=100 + i * 0.5, pnl=0.2)
    else:
        pae7.update("FAST", equity=175 - (i - 25) * 2, pnl=-1.0)
        pae7.update("SLOW", equity=112.5 + (i - 25) * 2, pnl=1.0)
    pae7.step()

# With low alpha (0.02), weights should be smooth, not flip instantly
w7 = pae7.get_weights()
# FAST had recent losses but EMA should still retain some historical allocation
test("ema_smooth_not_extreme", w7["FAST"] > 0.15)
test("ema_sum", abs(sum(w7.values()) - 1.0) < 1e-6)

# ── 8. Edge: all symbols equal ───────────────────────────────

pae8 = PortfolioAllocationEngine(
    ["A", "B", "C"],
    config=PAEConfig(warmup_bars=3, lookback=10, ema_alpha=1.0),
)
for i in range(20):
    for s in ["A", "B", "C"]:
        pae8.update(s, equity=100.0 + i * 0.1, pnl=0.05)
    pae8.step()

w8 = pae8.get_weights()
test("equal_perf_equal_w", abs(w8["A"] - w8["B"]) < 0.05)
test("equal_perf_equal_w2", abs(w8["B"] - w8["C"]) < 0.05)

# ── 9. Score formula edge cases ──────────────────────────────

from darwin_agent.portfolio.allocation_engine import PortfolioAllocationEngine as PAE
# Negative return → score = 0
m = {"return": -0.05, "maxdd": 0.10, "pf": 1.2, "volatility": 0.01}
test("neg_return_zero_score", PAE._compute_score(m) == 0.0)

# Zero volatility floored
m2 = {"return": 0.10, "maxdd": 0.05, "pf": 2.0, "volatility": 0.001}
s2 = PAE._compute_score(m2)
test("score_positive", s2 > 0)

# ── 10. Min 2 symbols requirement ────────────────────────────

try:
    PortfolioAllocationEngine(["ONLY_ONE"])
    test("min_symbols_error", False)
except ValueError:
    test("min_symbols_error", True)

# ── Results ──────────────────────────────────────────────────

total = passed + failed
if failed == 0:
    print(f"  ✅ ALL {total} PAE TESTS PASSED")
else:
    print(f"  ❌ {failed}/{total} FAILED")
