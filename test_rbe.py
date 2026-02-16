"""Unit tests for Risk Budget Engine."""
import sys, math
sys.path.insert(0, ".")
from darwin_agent.portfolio.risk_budget_engine import (
    RiskBudgetEngine, RBEConfig, RBESnapshot,
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

# ── 1. Default init ──────────────────────────────────────────

rbe = RiskBudgetEngine()
test("init_bar_zero", rbe.bar == 0)
test("init_mult_one", rbe.last_mult == 1.0)
test("init_peak_zero", rbe.peak == 0.0)
test("init_dd_zero", rbe.current_drawdown == 0.0)

# ── 2. Warmup: full exposure ────────────────────────────────

rbe2 = RiskBudgetEngine(RBEConfig(warmup=10))
for i in range(8):
    rbe2.update(equity=100.0 + i, pnl=0.5)
    m = rbe2.step()
test("warmup_full", m == 1.0)

# ── 3. No drawdown → mult ≈ 1.0 ─────────────────────────────

rbe3 = RiskBudgetEngine(RBEConfig(warmup=5, window=50))
for i in range(60):
    rbe3.update(equity=100.0 + i * 0.5, pnl=0.3)
    m = rbe3.step()
test("no_dd_high_mult", m >= 0.95)

# ── 4. Heavy drawdown → mult drops ──────────────────────────

rbe4 = RiskBudgetEngine(RBEConfig(warmup=5, window=50, dd_limit=0.35))
# Build up, then crash
for i in range(10):
    rbe4.update(equity=100 + i * 5, pnl=2.0)
    rbe4.step()
# Peak = 150. Now crash to 100 (33% DD, near dd_limit)
for i in range(50):
    rbe4.update(equity=150 - i * 1.0, pnl=-0.5)
    m = rbe4.step()
test("drawdown_reduces_mult", m < 0.80)
# At 100 equity, DD = (150-100)/150 = 33.3%
test("drawdown_correct", abs(rbe4.current_drawdown - (50.0/150.0)) < 0.01)

# ── 5. Extreme drawdown → floor ─────────────────────────────

rbe5 = RiskBudgetEngine(RBEConfig(warmup=5, dd_limit=0.20, mult_floor=0.30))
for i in range(10):
    rbe5.update(equity=200.0, pnl=1.0)
    rbe5.step()
# 50% drawdown (way past dd_limit)
for i in range(40):
    rbe5.update(equity=100.0, pnl=-2.0)
    m = rbe5.step()
test("extreme_dd_floor", m <= 0.40)

# ── 6. High volatility → vol_penalty kicks in ───────────────

rbe6 = RiskBudgetEngine(RBEConfig(warmup=5, target_vol=0.01, window=30))
for i in range(10):
    rbe6.update(equity=100.0, pnl=0)
    rbe6.step()
# Inject high vol: oscillate wildly
for i in range(40):
    eq = 100 + (10 if i % 2 == 0 else -10)
    rbe6.update(equity=eq, pnl=0.5 if i % 2 == 0 else -0.5)
    m = rbe6.step()
# High vol with target_vol=0.01 should reduce mult
test("high_vol_reduces", m < 0.95)

# ── 7. Good PF → pf_bonus positive ──────────────────────────

rbe7 = RiskBudgetEngine(RBEConfig(warmup=5, window=50))
for i in range(60):
    rbe7.update(equity=100 + i * 0.2, pnl=1.0)  # all wins
    rbe7.step()
snap = rbe7.history[-1]
test("good_pf_positive_bonus", snap.pf_bonus > 0.0)

# ── 8. Bad PF → pf_bonus negative ───────────────────────────

rbe8 = RiskBudgetEngine(RBEConfig(warmup=5, window=50))
for i in range(10):
    rbe8.update(equity=100.0, pnl=0.1)
    rbe8.step()
for i in range(50):
    rbe8.update(equity=100 - i * 0.1, pnl=-0.5)  # mostly losses
    rbe8.step()
snap8 = rbe8.history[-1]
test("bad_pf_negative_bonus", snap8.pf_bonus < 0.0)

# ── 9. Multiplier always in bounds ───────────────────────────

rbe9 = RiskBudgetEngine(RBEConfig(warmup=3))
for i in range(100):
    eq = 100 + 50 * math.sin(i * 0.3)  # oscillate
    rbe9.update(equity=max(10, eq), pnl=math.sin(i * 0.5))
    m = rbe9.step()
all_in_bounds = all(0.30 - 1e-6 <= s.rbe_mult <= 1.00 + 1e-6 for s in rbe9.history)
test("bounds_always_respected", all_in_bounds)

# ── 10. Determinism ──────────────────────────────────────────

def run_det():
    r = RiskBudgetEngine(RBEConfig(warmup=5, window=20))
    for i in range(40):
        r.update(equity=100 + i * 0.5 - (i % 7) * 0.3, pnl=0.2 if i % 3 else -0.1)
        r.step()
    return r.last_mult, [s.rbe_mult for s in r.history]

m1, h1 = run_det()
m2, h2 = run_det()
test("determinism_mult", abs(m1 - m2) < 1e-12)
test("determinism_history", all(abs(a - b) < 1e-12 for a, b in zip(h1, h2)))

# ── 11. apply_full_stack static ──────────────────────────────

size = RiskBudgetEngine.apply_full_stack(
    capital=1000, risk_pct=0.02, asset_weight=0.4,
    gmrt_mult=0.75, rbe_mult=0.80,
)
expected = 1000 * 0.02 * 0.4 * 0.75 * 0.80
test("full_stack_calc", abs(size - expected) < 1e-10)
test("full_stack_nonneg", RiskBudgetEngine.apply_full_stack(0, 0.02, 0.4, 0.75, 0.8) == 0.0)

# ── 12. Peak tracking ───────────────────────────────────────

rbe12 = RiskBudgetEngine()
rbe12.update(equity=100); rbe12.step()
rbe12.update(equity=120); rbe12.step()
rbe12.update(equity=110); rbe12.step()
test("peak_tracks_max", rbe12.peak == 120.0)
test("dd_after_decline", abs(rbe12.current_drawdown - 10/120) < 1e-6)

# ── 13. Recovery restores multiplier ────────────────────────

rbe13 = RiskBudgetEngine(RBEConfig(warmup=5, window=20, dd_limit=0.30))
# Build up
for i in range(10):
    rbe13.update(equity=100 + i * 5, pnl=2.0)
    rbe13.step()
# Crash
for i in range(10):
    rbe13.update(equity=150 - i * 3, pnl=-1.5)
    rbe13.step()
m_crash = rbe13.last_mult
# Recover
for i in range(30):
    rbe13.update(equity=120 + i * 2, pnl=1.0)
    rbe13.step()
m_recov = rbe13.last_mult
test("recovery_restores", m_recov > m_crash)

# ── 14. Snapshot fields populated ────────────────────────────

rbe14 = RiskBudgetEngine(RBEConfig(warmup=3))
for i in range(10):
    rbe14.update(equity=100 + i, pnl=0.5)
    rbe14.step()
s = rbe14.history[-1]
test("snap_has_equity", s.equity > 0)
test("snap_has_peak", s.peak > 0)
test("snap_has_bar", s.bar == 10)
test("snap_has_mult", 0.3 <= s.rbe_mult <= 1.0)

# ── 15. Empty PNL → default PF ──────────────────────────────

rbe15 = RiskBudgetEngine(RBEConfig(warmup=3))
for i in range(10):
    rbe15.update(equity=100 + i * 0.1, pnl=0)  # no trades
    rbe15.step()
test("no_pnl_default_pf", rbe15.history[-1].rolling_pf == 1.5)

# ── Results ──────────────────────────────────────────────────

total = passed + failed
if failed == 0:
    print(f"  ✅ ALL {total} RBE TESTS PASSED")
else:
    print(f"  ❌ {failed}/{total} FAILED")
