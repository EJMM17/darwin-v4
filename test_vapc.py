"""
Unit tests for Volatility-Adjusted Position Cap (VAPC) v2.

max_bar_risk = 0.20 (per-symbol, not portfolio-min).

Covers:
    1. Normal volatility → no cap (5x unaffected)
    2. High volatility → leverage reduced < 3x
    3. Edge cases (zero ATR, zero price, epsilon)
    4. Determinism
    5. Per-symbol isolation (each symbol gets own cap)
    6. Config overrides
    7. Boundary conditions
"""

import pytest
from darwin_agent.portfolio.vapc import (
    VAPCConfig,
    compute_vapc_leverage,
    compute_effective_leverage,
)

DEFAULT = VAPCConfig()  # max_bar_risk=0.20


# ── 1. Normal volatility — 5x NOT capped ──

class TestNormalVolNotCapped:
    """With max_bar_risk=0.20, normal crypto vol (~2-3%) must NOT reduce leverage."""

    def test_btc_normal(self):
        # BTC $60k, ATR $1200, bar_vol=0.02 → cap=0.20/0.02=10.0 → clamp 5.0
        assert compute_vapc_leverage(1200.0, 60000.0) == 5.0

    def test_sol_normal(self):
        # SOL $150, ATR $4.5, bar_vol=0.03 → cap=0.20/0.03=6.67 → clamp 5.0
        assert compute_vapc_leverage(4.5, 150.0) == 5.0

    def test_doge_normal(self):
        # DOGE $0.10, ATR $0.003, bar_vol=0.03 → cap=6.67 → clamp 5.0
        assert compute_vapc_leverage(0.003, 0.10) == 5.0

    def test_btc_moderate_vol(self):
        # BTC $60k, ATR $2400, bar_vol=0.04 → cap=0.20/0.04=5.0 → exactly 5.0
        assert compute_vapc_leverage(2400.0, 60000.0) == 5.0

    def test_effective_normal_dynamic_binds(self):
        # Dynamic=4.3, VAPC=5.0 → dynamic binds, no reduction
        eff = compute_effective_leverage(4.3, 1200.0, 60000.0)
        assert eff == 4.3

    def test_effective_normal_full_5x(self):
        # Dynamic=5.0, VAPC=5.0 → full 5x
        eff = compute_effective_leverage(5.0, 1200.0, 60000.0)
        assert eff == 5.0


# ── 2. High volatility — leverage reduced below 3x ──

class TestHighVolReduced:
    """Crash-level vol must push leverage well below 3x."""

    def test_crash_vol_10pct(self):
        # bar_vol=0.10 → cap=0.20/0.10=2.0
        cap = compute_vapc_leverage(5000.0, 50000.0)
        assert abs(cap - 2.0) < 0.01

    def test_crash_vol_15pct(self):
        # bar_vol=0.15 → cap=0.20/0.15=1.33
        cap = compute_vapc_leverage(9000.0, 60000.0)
        assert abs(cap - 1.333) < 0.01

    def test_extreme_vol_clamp(self):
        # bar_vol=0.25 → cap=0.80 → clamp to 1.0
        cap = compute_vapc_leverage(15000.0, 60000.0)
        assert cap == 1.0

    def test_effective_crash(self):
        # Dynamic=4.3, VAPC=2.0 → VAPC binds
        eff = compute_effective_leverage(4.3, 5000.0, 50000.0)
        assert abs(eff - 2.0) < 0.01

    def test_stress_below_3x(self):
        # bar_vol=0.08 → cap=2.5 < 3.0
        cap = compute_vapc_leverage(4800.0, 60000.0)
        assert cap < 3.0
        assert abs(cap - 2.5) < 0.01


# ── 3. Edge cases ──

class TestEdgeCases:
    def test_zero_atr(self):
        assert compute_vapc_leverage(0.0, 60000.0) == 5.0

    def test_negative_atr(self):
        assert compute_vapc_leverage(-100.0, 60000.0) == 5.0

    def test_zero_price(self):
        assert compute_vapc_leverage(1200.0, 0.0) == 5.0

    def test_negative_price(self):
        assert compute_vapc_leverage(1200.0, -1.0) == 5.0

    def test_both_zero(self):
        assert compute_vapc_leverage(0.0, 0.0) == 5.0

    def test_epsilon_atr(self):
        assert compute_vapc_leverage(1e-15, 60000.0) == 5.0

    def test_effective_with_zero(self):
        eff = compute_effective_leverage(4.0, 0.0, 60000.0)
        assert eff == 4.0


# ── 4. Determinism ──

class TestDeterminism:
    def test_repeated_vapc(self):
        results = {compute_vapc_leverage(3000.0, 60000.0) for _ in range(1000)}
        assert len(results) == 1

    def test_repeated_effective(self):
        results = {compute_effective_leverage(3.5, 3000.0, 60000.0) for _ in range(1000)}
        assert len(results) == 1


# ── 5. Per-symbol isolation ──

class TestPerSymbolIsolation:
    """Each symbol gets its own VAPC cap — no cross-asset min."""

    def test_btc_calm_sol_stressed(self):
        btc_cap = compute_vapc_leverage(1200.0, 60000.0)   # bar_vol=0.02 → 5.0
        sol_cap = compute_vapc_leverage(15.0, 150.0)        # bar_vol=0.10 → 2.0
        assert btc_cap == 5.0
        assert abs(sol_cap - 2.0) < 0.01

    def test_three_assets_independent(self):
        caps = {
            "BTC": compute_vapc_leverage(1200.0, 60000.0),   # 0.02 → 5.0
            "SOL": compute_vapc_leverage(9.0, 150.0),         # 0.06 → 3.33
            "DOGE": compute_vapc_leverage(0.012, 0.10),       # 0.12 → 1.67
        }
        assert caps["BTC"] == 5.0
        assert abs(caps["SOL"] - 3.333) < 0.01
        assert abs(caps["DOGE"] - 1.667) < 0.01

    def test_all_stressed_equally(self):
        caps = [
            compute_vapc_leverage(6000.0, 60000.0),
            compute_vapc_leverage(15.0, 150.0),
            compute_vapc_leverage(0.01, 0.10),
        ]
        assert all(abs(c - 2.0) < 0.01 for c in caps)


# ── 6. Config overrides ──

class TestConfigOverrides:
    def test_tighter_risk(self):
        cfg = VAPCConfig(max_bar_risk=0.10)
        assert compute_vapc_leverage(1200.0, 60000.0, cfg) == 5.0

    def test_wider_risk(self):
        cfg = VAPCConfig(max_bar_risk=0.30)
        cap = compute_vapc_leverage(5000.0, 50000.0, cfg)
        assert abs(cap - 3.0) < 0.01

    def test_lower_max_leverage(self):
        cfg = VAPCConfig(max_leverage=3.0)
        assert compute_vapc_leverage(1200.0, 60000.0, cfg) == 3.0


# ── 7. Boundary: VAPC activates at bar_vol > 0.20/5.0 = 0.04 ──

class TestActivationBoundary:
    def test_at_boundary(self):
        cap = compute_vapc_leverage(2400.0, 60000.0)  # bar_vol=0.04
        assert cap == 5.0

    def test_just_above_boundary(self):
        atr = 0.041 * 60000.0
        cap = compute_vapc_leverage(atr, 60000.0)
        assert cap < 5.0 and cap > 4.8

    def test_just_below_boundary(self):
        atr = 0.039 * 60000.0
        cap = compute_vapc_leverage(atr, 60000.0)
        assert cap == 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
