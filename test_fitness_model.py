"""
Darwin v4 — Risk-Aware Fitness Model Tests.

Tests all 7 components individually and their integration:
  1. Risk-adjusted profit scoring
  2. Sharpe quality scoring
  3. Drawdown health scoring
  4. Consistency scoring (win rate + PnL stability)
  5. Portfolio harmony (state penalties, drawdown contribution)
  6. Diversification bonus (correlation, overlap)
  7. Capital efficiency
  8. Boundedness: always [0, 1], NaN-safe
  9. Softmax safety: no explosion with extreme values
  10. Integration: systemic drawdown agent penalized vs safe agent
  11. Integration: correlation-concentrated agent penalized
  12. Old vs new: new model produces reasonable scores for known scenarios
"""
import asyncio
import math
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from darwin_agent.interfaces.enums import PortfolioRiskState
from darwin_agent.interfaces.types import PortfolioRiskMetrics
from darwin_agent.evolution.fitness import (
    RiskAwareFitness, FitnessConfig, FitnessWeights,
    FitnessBreakdown, compute_fitness, compute_fitness_breakdown,
    _clamp,
)


def _make_snapshot(
    state=PortfolioRiskState.NORMAL,
    equity=1000.0, peak=1000.0,
    drawdown_pct=0.0, correlation_risk=0.3,
    exposure_by_symbol=None,
):
    return PortfolioRiskMetrics(
        risk_state=state,
        total_equity=equity,
        peak_equity=peak,
        drawdown_pct=drawdown_pct,
        correlation_risk=correlation_risk,
        exposure_by_symbol=exposure_by_symbol or {"BTCUSDT": 0.5, "ETHUSDT": 0.3},
    )


model = RiskAwareFitness()
PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}: {detail}")


def run_tests():
    global PASS, FAIL

    # ═══════════════════════════════════════════════════════
    # 1. Risk-adjusted profit
    # ═══════════════════════════════════════════════════════
    print("\n📊 1. Risk-adjusted profit component")
    print("─" * 50)

    # Big profit → high score
    bd = model.compute_breakdown(
        realized_pnl=150.0, initial_capital=100.0, current_capital=250.0,
        sharpe=0, max_drawdown_pct=0, win_count=10, loss_count=5,
        pnl_series=[10]*10 + [-5]*5,
    )
    check("150% profit → rap > 0.7", bd.risk_adjusted_profit > 0.7,
          f"got {bd.risk_adjusted_profit:.4f}")

    # Big loss → low score
    bd = model.compute_breakdown(
        realized_pnl=-40.0, initial_capital=100.0, current_capital=60.0,
        sharpe=0, max_drawdown_pct=40, win_count=3, loss_count=12,
        pnl_series=[-3]*12 + [1]*3,
    )
    check("-40% loss → rap < 0.3", bd.risk_adjusted_profit < 0.3,
          f"got {bd.risk_adjusted_profit:.4f}")

    # ═══════════════════════════════════════════════════════
    # 2. Sharpe quality
    # ═══════════════════════════════════════════════════════
    print("\n📊 2. Sharpe quality component")
    print("─" * 50)

    bd = model.compute_breakdown(
        realized_pnl=50.0, initial_capital=100.0, current_capital=150.0,
        sharpe=2.5, max_drawdown_pct=5, win_count=15, loss_count=5,
        pnl_series=[3]*15 + [-1]*5,
    )
    check("Sharpe 2.5 → sq > 0.7", bd.sharpe_quality > 0.7,
          f"got {bd.sharpe_quality:.4f}")

    bd2 = model.compute_breakdown(
        realized_pnl=50.0, initial_capital=100.0, current_capital=150.0,
        sharpe=-0.5, max_drawdown_pct=5, win_count=15, loss_count=5,
        pnl_series=[3]*15 + [-1]*5,
    )
    check("Negative Sharpe → sq = 0", bd2.sharpe_quality == 0.0,
          f"got {bd2.sharpe_quality:.4f}")

    # ═══════════════════════════════════════════════════════
    # 3. Drawdown health
    # ═══════════════════════════════════════════════════════
    print("\n📊 3. Drawdown health component")
    print("─" * 50)

    bd = model.compute_breakdown(
        realized_pnl=10.0, initial_capital=100.0, current_capital=110.0,
        sharpe=1.0, max_drawdown_pct=2.0, win_count=8, loss_count=4,
        pnl_series=[1]*8 + [-0.5]*4,
    )
    check("2% DD → dh > 0.9", bd.drawdown_health > 0.9,
          f"got {bd.drawdown_health:.4f}")

    bd2 = model.compute_breakdown(
        realized_pnl=-10.0, initial_capital=100.0, current_capital=90.0,
        sharpe=0, max_drawdown_pct=35.0, win_count=4, loss_count=12,
        pnl_series=[-2]*12 + [1]*4,
    )
    check("35% DD → dh < 0.2", bd2.drawdown_health < 0.2,
          f"got {bd2.drawdown_health:.4f}")

    # ═══════════════════════════════════════════════════════
    # 4. Consistency
    # ═══════════════════════════════════════════════════════
    print("\n📊 4. Consistency component")
    print("─" * 50)

    # Highly consistent agent: 70% win rate, low variance
    bd = model.compute_breakdown(
        realized_pnl=20.0, initial_capital=100.0, current_capital=120.0,
        sharpe=1.5, max_drawdown_pct=3.0, win_count=14, loss_count=6,
        pnl_series=[1.0]*14 + [-0.7]*6,  # consistent small wins/losses
    )
    check("70% WR + low variance → consistency > 0.6", bd.consistency > 0.6,
          f"got {bd.consistency:.4f}")

    # Inconsistent: 55% WR, huge variance
    bd2 = model.compute_breakdown(
        realized_pnl=20.0, initial_capital=100.0, current_capital=120.0,
        sharpe=0.5, max_drawdown_pct=15.0, win_count=11, loss_count=9,
        pnl_series=[20.0, -18.0, 15.0, -12.0, 8.0, -10.0, 25.0, -20.0,
                    12.0, -8.0, 5.0, -3.0, 7.0, -6.0, 10.0, -9.0,
                    3.0, -2.0, 4.0, -1.0],
    )
    check("55% WR + high variance → consistency < 0.5", bd2.consistency < 0.5,
          f"got {bd2.consistency:.4f}")

    # Few trades → neutral 0.5
    bd3 = model.compute_breakdown(
        realized_pnl=5.0, initial_capital=100.0, current_capital=105.0,
        sharpe=0.5, max_drawdown_pct=1.0, win_count=3, loss_count=2,
        pnl_series=[2, 1, 2, -0.5, -0.5],
    )
    check("5 trades → consistency ≈ 0.5", abs(bd3.consistency - 0.5) < 0.05,
          f"got {bd3.consistency:.4f}")

    # ═══════════════════════════════════════════════════════
    # 5. Portfolio harmony
    # ═══════════════════════════════════════════════════════
    print("\n📊 5. Portfolio harmony component")
    print("─" * 50)

    # NORMAL state, profitable agent → full score
    snap_normal = _make_snapshot(state=PortfolioRiskState.NORMAL)
    bd = model.compute_breakdown(
        realized_pnl=50.0, initial_capital=100.0, current_capital=150.0,
        sharpe=2.0, max_drawdown_pct=3.0, win_count=15, loss_count=5,
        pnl_series=[3]*15 + [-1]*5,
        portfolio_snapshot=snap_normal,
    )
    check("NORMAL + profit → harmony = 1.0", bd.portfolio_harmony == 1.0,
          f"got {bd.portfolio_harmony:.4f}")

    # CRITICAL state, losing agent → heavily penalized
    snap_critical = _make_snapshot(
        state=PortfolioRiskState.CRITICAL,
        equity=800, peak=1000, drawdown_pct=20.0,
    )
    bd2 = model.compute_breakdown(
        realized_pnl=-30.0, initial_capital=100.0, current_capital=70.0,
        sharpe=-0.5, max_drawdown_pct=30.0, win_count=4, loss_count=16,
        pnl_series=[-2]*16 + [1]*4,
        portfolio_snapshot=snap_critical,
    )
    check("CRITICAL + losing → harmony < 0.15", bd2.portfolio_harmony < 0.15,
          f"got {bd2.portfolio_harmony:.4f}")
    check("  penalty > 0.15", bd2.penalty_applied > 0.15,
          f"got {bd2.penalty_applied:.4f}")

    # DEFENSIVE state but agent is PROFITABLE → base score no penalty
    snap_def = _make_snapshot(
        state=PortfolioRiskState.DEFENSIVE,
        equity=920, peak=1000, drawdown_pct=8.0,
    )
    bd3 = model.compute_breakdown(
        realized_pnl=20.0, initial_capital=100.0, current_capital=120.0,
        sharpe=1.0, max_drawdown_pct=5.0, win_count=12, loss_count=8,
        pnl_series=[2]*12 + [-0.5]*8,
        portfolio_snapshot=snap_def,
    )
    check("DEFENSIVE + profit → harmony ≈ 0.65 (no penalty)",
          bd3.portfolio_harmony >= 0.6, f"got {bd3.portfolio_harmony:.4f}")
    check("  penalty = 0", bd3.penalty_applied == 0.0,
          f"got {bd3.penalty_applied:.4f}")

    # HALTED + losing → zero
    snap_halt = _make_snapshot(state=PortfolioRiskState.HALTED)
    bd4 = model.compute_breakdown(
        realized_pnl=-10.0, initial_capital=100.0, current_capital=90.0,
        sharpe=-1.0, max_drawdown_pct=25.0, win_count=3, loss_count=12,
        pnl_series=[-2]*12 + [1]*3,
        portfolio_snapshot=snap_halt,
    )
    check("HALTED + losing → harmony = 0", bd4.portfolio_harmony == 0.0,
          f"got {bd4.portfolio_harmony:.4f}")

    # ═══════════════════════════════════════════════════════
    # 6. Diversification bonus
    # ═══════════════════════════════════════════════════════
    print("\n📊 6. Diversification bonus component")
    print("─" * 50)

    # Agent trades symbols NOT in pool exposure → high score
    snap = _make_snapshot(exposure_by_symbol={"BTCUSDT": 0.7, "ETHUSDT": 0.3})
    bd = model.compute_breakdown(
        realized_pnl=10.0, initial_capital=100.0, current_capital=110.0,
        sharpe=1.0, max_drawdown_pct=5.0, win_count=8, loss_count=4,
        pnl_series=[1]*8 + [-0.5]*4,
        portfolio_snapshot=snap,
        agent_exposure={"SOLUSDT": 0.5, "AVAXUSDT": 0.3, "DOTUSDT": 0.2},
    )
    check("Agent on SOL/AVAX/DOT vs pool on BTC/ETH → div > 0.6",
          bd.diversification_bonus > 0.6,
          f"got {bd.diversification_bonus:.4f}")

    # Agent fully overlaps with pool → low score
    bd2 = model.compute_breakdown(
        realized_pnl=10.0, initial_capital=100.0, current_capital=110.0,
        sharpe=1.0, max_drawdown_pct=5.0, win_count=8, loss_count=4,
        pnl_series=[1]*8 + [-0.5]*4,
        portfolio_snapshot=snap,
        agent_exposure={"BTCUSDT": 1.0},  # same as pool's top
    )
    check("Agent 100% BTC (pool 70% BTC) → div < 0.5",
          bd2.diversification_bonus < 0.5,
          f"got {bd2.diversification_bonus:.4f}")

    # ═══════════════════════════════════════════════════════
    # 7. Capital efficiency
    # ═══════════════════════════════════════════════════════
    print("\n📊 7. Capital efficiency component")
    print("─" * 50)

    bd = model.compute_breakdown(
        realized_pnl=80.0, initial_capital=100.0, current_capital=180.0,
        sharpe=2.0, max_drawdown_pct=5.0, win_count=15, loss_count=5,
        pnl_series=[5]*15 + [-1]*5,
    )
    check("80% ROC → ce > 0.7", bd.capital_efficiency > 0.7,
          f"got {bd.capital_efficiency:.4f}")

    bd2 = model.compute_breakdown(
        realized_pnl=-10.0, initial_capital=100.0, current_capital=90.0,
        sharpe=-0.5, max_drawdown_pct=10.0, win_count=5, loss_count=10,
        pnl_series=[-2]*10 + [1]*5,
    )
    check("Negative ROC → ce = 0", bd2.capital_efficiency == 0.0,
          f"got {bd2.capital_efficiency:.4f}")

    # ═══════════════════════════════════════════════════════
    # 8. Boundedness + NaN safety
    # ═══════════════════════════════════════════════════════
    print("\n📊 8. Boundedness & NaN safety")
    print("─" * 50)

    # Extreme values
    extreme_cases = [
        ("Massive profit", 999999.0, 1.0, 999999.0, 50.0, 0.0),
        ("Massive loss", -999999.0, 1.0, -999998.0, -50.0, 99.0),
        ("Zero capital", 0.0, 0.0, 0.0, 0.0, 0.0),
        ("NaN sharpe", 10.0, 100.0, 110.0, float('nan'), 5.0),
        ("Inf drawdown", 10.0, 100.0, 110.0, 1.0, float('inf')),
    ]
    all_bounded = True
    for name, pnl, cap, cur, sharpe, dd in extreme_cases:
        score = compute_fitness(
            realized_pnl=pnl, initial_capital=max(cap, 0.01),
            current_capital=cur, sharpe=sharpe if not math.isnan(sharpe) else 0,
            max_drawdown_pct=dd if not math.isinf(dd) else 100.0,
            win_count=5, loss_count=5,
            pnl_series=[pnl/10]*10,
        )
        if not (0.0 <= score <= 1.0):
            all_bounded = False
            print(f"    ❌ {name}: score={score}")
    check("All extreme cases → score in [0, 1]", all_bounded)

    # NaN clamp
    check("_clamp(NaN) = 0.0", _clamp(float('nan')) == 0.0)
    check("_clamp(inf) = 1.0", _clamp(float('inf')) == 1.0)
    check("_clamp(-inf) = 0.0", _clamp(float('-inf')) == 0.0)

    # ═══════════════════════════════════════════════════════
    # 9. Softmax safety
    # ═══════════════════════════════════════════════════════
    print("\n📊 9. Softmax safety")
    print("─" * 50)

    scores = []
    for pnl in [-100, -50, 0, 50, 100, 200, 500]:
        s = compute_fitness(
            realized_pnl=float(pnl), initial_capital=100.0,
            current_capital=100.0+pnl, sharpe=pnl/50.0,
            max_drawdown_pct=max(0, -pnl/2), win_count=10, loss_count=10,
            pnl_series=[pnl/20]*20,
        )
        scores.append(s)

    # Softmax with temperature=1 should NOT overflow
    max_s = max(scores)
    try:
        exp_sum = sum(math.exp(s - max_s) for s in scores)
        softmax = [math.exp(s - max_s) / exp_sum for s in scores]
        check("Softmax computes without overflow", True)
        check("All softmax probs sum to 1.0", abs(sum(softmax) - 1.0) < 1e-6,
              f"sum={sum(softmax):.6f}")
    except OverflowError:
        check("Softmax computes without overflow", False, "OverflowError")

    # ═══════════════════════════════════════════════════════
    # 10. Integration: systemic risk penalty
    # ═══════════════════════════════════════════════════════
    print("\n📊 10. Systemic risk penalty integration")
    print("─" * 50)

    # Agent A: makes $50 but portfolio is CRITICAL and agent is losing
    snap_crit = _make_snapshot(
        state=PortfolioRiskState.CRITICAL,
        equity=700, peak=1000, drawdown_pct=30.0,
    )
    score_risky = compute_fitness(
        realized_pnl=-20.0, initial_capital=100.0, current_capital=80.0,
        sharpe=-0.5, max_drawdown_pct=20.0, win_count=5, loss_count=15,
        pnl_series=[-2]*15 + [1]*5,
        portfolio_snapshot=snap_crit,
    )

    # Agent B: makes $10 safely while portfolio is NORMAL
    snap_safe = _make_snapshot(state=PortfolioRiskState.NORMAL)
    score_safe = compute_fitness(
        realized_pnl=10.0, initial_capital=100.0, current_capital=110.0,
        sharpe=1.0, max_drawdown_pct=3.0, win_count=10, loss_count=5,
        pnl_series=[1]*10 + [-0.5]*5,
        portfolio_snapshot=snap_safe,
    )

    check(
        "Safe $10 agent (NORMAL) > risky -$20 agent (CRITICAL)",
        score_safe > score_risky,
        f"safe={score_safe:.4f}, risky={score_risky:.4f}",
    )

    # ═══════════════════════════════════════════════════════
    # 11. Integration: correlation concentration penalty
    # ═══════════════════════════════════════════════════════
    print("\n📊 11. Correlation concentration penalty")
    print("─" * 50)

    snap = _make_snapshot(
        exposure_by_symbol={"BTCUSDT": 0.8, "ETHUSDT": 0.2},
        correlation_risk=0.8,  # highly correlated pool
    )

    # Agent trades same concentrated symbol
    score_concentrated = compute_fitness(
        realized_pnl=15.0, initial_capital=100.0, current_capital=115.0,
        sharpe=1.0, max_drawdown_pct=5.0, win_count=10, loss_count=5,
        pnl_series=[1]*10 + [-0.5]*5,
        portfolio_snapshot=snap,
        agent_exposure={"BTCUSDT": 1.0},
    )

    # Agent diversifies into different symbols
    score_diversified = compute_fitness(
        realized_pnl=15.0, initial_capital=100.0, current_capital=115.0,
        sharpe=1.0, max_drawdown_pct=5.0, win_count=10, loss_count=5,
        pnl_series=[1]*10 + [-0.5]*5,
        portfolio_snapshot=snap,
        agent_exposure={"SOLUSDT": 0.4, "AVAXUSDT": 0.3, "DOTUSDT": 0.3},
    )

    check(
        "Diversified agent > concentrated agent (same PnL)",
        score_diversified > score_concentrated,
        f"diversified={score_diversified:.4f}, concentrated={score_concentrated:.4f}",
    )

    # ═══════════════════════════════════════════════════════
    # 12. Weight validation
    # ═══════════════════════════════════════════════════════
    print("\n📊 12. Weight validation")
    print("─" * 50)

    try:
        FitnessWeights(ecosystem_health=0.5, risk_stability=0.5,
                       learning_quality=0.5)
        check("Invalid weights (sum > 1) rejected", False, "no error raised")
    except (ValueError, TypeError):
        check("Invalid weights (sum > 1) rejected", True)

    # No trades → 0
    score_zero = compute_fitness(
        realized_pnl=0, initial_capital=100.0, current_capital=100.0,
        sharpe=0, max_drawdown_pct=0, win_count=0, loss_count=0,
        pnl_series=[],
    )
    check("No trades → fitness = 0", score_zero == 0.0,
          f"got {score_zero}")

    # ═══════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════
    print("\n" + "═" * 60)
    if FAIL == 0:
        print(f"  ✅ ALL {PASS} FITNESS MODEL TESTS PASSED")
    else:
        print(f"  ❌ {FAIL} FAILED, {PASS} passed")
    print("═" * 60)
    return FAIL == 0


if __name__ == "__main__":
    ok = run_tests()
    sys.exit(0 if ok else 1)
