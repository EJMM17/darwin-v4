"""
Darwin v4 — Audit Fix Regression Tests.

Tests each of the 6 findings from the PortfolioRiskEngine audit:

  FIX 1: Async locking on all mutation paths
  FIX 2: Exchange-scoped exposure validation
  FIX 3: No equity mutation inside approve_order
  FIX 4: HALTED recovery validates ALL constraints
  FIX 5: Thread-safe atomic exposure recalculation
  FIX 6: Concurrent approval sequence counter
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from datetime import datetime, timezone
from darwin_agent.interfaces.enums import *
from darwin_agent.interfaces.types import *
from darwin_agent.risk.portfolio_engine import PortfolioRiskEngine, RiskLimits

def _utcnow():
    return datetime.now(timezone.utc)

PARAMS = PhaseParams(phase=GrowthPhase.BOOTSTRAP, risk_pct=2.0, max_positions=3, leverage=10)

def sig(symbol="BTCUSDT", side=OrderSide.BUY,
        strength=SignalStrength.STRONG, confidence=0.85,
        exchange_id=None):
    meta = {}
    if exchange_id is not None:
        meta["exchange_id"] = exchange_id
    return Signal(strategy=StrategyID.MOMENTUM, symbol=symbol, side=side,
                  strength=strength, confidence=confidence, metadata=meta)

def pos(symbol="BTCUSDT", side=OrderSide.BUY, size=0.1, price=100.0,
        leverage=10, exchange=ExchangeID.BYBIT):
    return Position(symbol=symbol, side=side, size=size, entry_price=price,
                    leverage=leverage, exchange_id=exchange)

def trade(pnl):
    return TradeResult(realized_pnl=pnl, closed_at=_utcnow())


# ═══════════════════════════════════════════════════════════
# FIX 1: Async locking
# ═══════════════════════════════════════════════════════════

async def test_fix1_async_locking():
    print("\n🔒 FIX 1: Async locking on all mutation paths")
    print("─" * 55)

    engine = PortfolioRiskEngine()

    # Verify async variants exist and acquire the lock
    assert hasattr(engine, 'update_equity_async')
    assert hasattr(engine, 'update_after_trade_async')
    assert hasattr(engine, 'update_positions_async')
    assert hasattr(engine, 'approve_order_async')
    assert hasattr(engine, 'reset_async')
    print("  ✅ All 5 async variants exist")

    # Verify lock is actually acquired (concurrent mutations serialise)
    call_order = []

    async def slow_update():
        async with engine._lock:
            call_order.append("slow_start")
            await asyncio.sleep(0.05)
            call_order.append("slow_end")

    async def fast_update():
        await asyncio.sleep(0.01)  # Ensure slow starts first
        await engine.update_equity_async(500.0)
        call_order.append("fast_done")

    await asyncio.gather(slow_update(), fast_update())

    # fast_done must come AFTER slow_end because lock serialises
    assert call_order.index("slow_end") < call_order.index("fast_done"), \
        f"Lock not serialising: {call_order}"
    print(f"  ✅ Lock serialises concurrent mutations: {call_order}")

    # Verify async approve works
    engine.update_equity(1000.0)
    v = await engine.approve_order_async(sig(), [], 1000.0, PARAMS)
    assert v.approved
    print("  ✅ approve_order_async works under lock")
    print("  ✅ PASSED\n")


# ═══════════════════════════════════════════════════════════
# FIX 2: Exchange-scoped exposure
# ═══════════════════════════════════════════════════════════

async def test_fix2_exchange_scoped_exposure():
    print("🎯 FIX 2: Exchange-scoped exposure validation")
    print("─" * 55)

    limits = RiskLimits(max_exchange_exposure_pct=50.0)
    engine = PortfolioRiskEngine(limits=limits)
    engine.update_equity(1000.0)

    # Bybit has 55% exposure, Binance has 5%
    positions = [
        pos("SYM1", size=0.5, price=100, leverage=10, exchange=ExchangeID.BYBIT),
        pos("SYM2", side=OrderSide.SELL, size=0.05, price=100, leverage=10, exchange=ExchangeID.BYBIT),
        pos("SYM3", side=OrderSide.SELL, size=0.05, price=100, leverage=10, exchange=ExchangeID.BINANCE),
    ]
    engine.update_positions(positions)

    state = engine.get_portfolio_state()
    print(f"  Bybit exposure:  ${state.exposure_by_exchange.get('bybit', 0):.0f} "
          f"({state.exposure_by_exchange.get('bybit', 0)/10:.0f}%)")
    print(f"  Binance exposure: ${state.exposure_by_exchange.get('binance', 0):.0f} "
          f"({state.exposure_by_exchange.get('binance', 0)/10:.0f}%)")

    # Signal scoped to Binance → should PASS (Binance only at 5%)
    v = engine.approve_order(
        sig("SYM4", exchange_id=ExchangeID.BINANCE),
        positions, 1000.0,
        PhaseParams(phase=GrowthPhase.BOOTSTRAP, risk_pct=2.0,
                    max_positions=5, leverage=10),
    )
    assert v.approved, f"Binance-scoped signal should pass: {v.reason}"
    print(f"  ✅ Binance-scoped signal APPROVED (Binance at 5%)")

    # Signal scoped to Bybit → should FAIL (Bybit at 55%)
    v = engine.approve_order(
        sig("SYM4", exchange_id=ExchangeID.BYBIT),
        positions, 1000.0,
        PhaseParams(phase=GrowthPhase.BOOTSTRAP, risk_pct=2.0,
                    max_positions=5, leverage=10),
    )
    assert not v.approved
    assert "exchange_exposure" in v.reason and "bybit" in v.reason
    print(f"  ✅ Bybit-scoped signal BLOCKED: {v.reason}")

    # Signal with NO exchange → falls back to checking ALL (conservative)
    v = engine.approve_order(
        sig("SYM4"),  # no exchange_id in metadata
        positions, 1000.0,
        PhaseParams(phase=GrowthPhase.BOOTSTRAP, risk_pct=2.0,
                    max_positions=5, leverage=10),
    )
    assert not v.approved
    assert "exchange_exposure" in v.reason
    print(f"  ✅ Unscoped signal BLOCKED (conservative fallback): {v.reason}")
    print("  ✅ PASSED\n")


# ═══════════════════════════════════════════════════════════
# FIX 3: No equity mutation in approve_order
# ═══════════════════════════════════════════════════════════

async def test_fix3_no_equity_mutation_in_approve():
    print("🔍 FIX 3: approve_order is pure (no equity mutation)")
    print("─" * 55)

    engine = PortfolioRiskEngine()
    engine.update_equity(500.0)

    peak_before = engine._equity.peak
    current_before = engine._equity.current

    # Call approve with a HIGHER capital value
    # Old bug: this would set peak to 9999
    engine.approve_order(sig(), [], 9999.0, PARAMS)

    assert engine._equity.peak == peak_before, \
        f"Peak changed! {peak_before} → {engine._equity.peak}"
    assert engine._equity.current == current_before, \
        f"Current changed! {current_before} → {engine._equity.current}"
    print(f"  Peak before: ${peak_before:.2f}, after: ${engine._equity.peak:.2f}")
    print(f"  ✅ approve_order did NOT mutate equity")

    # Call approve with a LOWER capital value
    # Old bug: this would shift drawdown
    engine.approve_order(sig(), [], 100.0, PARAMS)

    assert engine._equity.peak == peak_before
    assert engine._equity.current == current_before
    print(f"  ✅ approve_order with low capital also did NOT mutate")
    print("  ✅ PASSED\n")


# ═══════════════════════════════════════════════════════════
# FIX 4: HALTED recovery validates ALL constraints
# ═══════════════════════════════════════════════════════════

async def test_fix4_halted_recovery_constraints():
    print("🛡️  FIX 4: HALTED recovery validates ALL constraints")
    print("─" * 55)

    limits = RiskLimits(
        defensive_drawdown_pct=5.0,
        critical_drawdown_pct=15.0,
        halted_drawdown_pct=20.0,
        critical_recovery_pct=12.0,
        halt_duration_minutes=0.0,   # instant expiry
        max_consecutive_losses=3,
        max_daily_loss_pct=5.0,
        max_total_exposure_pct=60.0,
    )
    engine = PortfolioRiskEngine(limits=limits)

    # Drive into HALTED
    engine.update_equity(1000.0)
    engine.update_equity(780.0)  # 22% dd → HALTED
    assert engine.state == PortfolioRiskState.HALTED
    print(f"  State: HALTED (dd=22%)")

    # Scenario A: Drawdown recovers BUT consecutive losses still maxed
    for _ in range(5):
        engine.update_after_trade(trade(-1.0))
    assert engine._stats.consecutive_losses >= 3

    engine.update_equity(900.0)  # dd=10% < critical_recovery(12%)
    # Should NOT recover because consecutive losses = 5 >= 3
    assert engine.state == PortfolioRiskState.HALTED, \
        f"Should stay HALTED with {engine._stats.consecutive_losses} consec losses"
    print(f"  ✅ Recovery BLOCKED by consecutive_losses={engine._stats.consecutive_losses}")

    # Reset consecutive losses with a win
    engine.update_after_trade(trade(5.0))
    assert engine._stats.consecutive_losses == 0

    # Now equity update should trigger recovery
    engine.update_equity(900.0)
    assert engine.state == PortfolioRiskState.DEFENSIVE, \
        f"Should recover to DEFENSIVE, got {engine.state.value}"
    print(f"  ✅ Recovery ALLOWED after consecutive losses reset")

    # Scenario B: Test exposure blocking recovery
    engine2 = PortfolioRiskEngine(limits=limits)
    engine2.update_equity(1000.0)
    engine2.update_equity(780.0)  # HALTED
    assert engine2.state == PortfolioRiskState.HALTED

    # Inject high exposure
    heavy = [pos("SYM1", size=1.0, price=100, leverage=10)]
    engine2.update_positions(heavy)  # 1000 notional = 100% of 1000 > 60%

    engine2.update_equity(900.0)  # dd=10% is fine...
    # ...but exposure is too high
    assert engine2.state == PortfolioRiskState.HALTED, \
        "Should stay HALTED with high exposure"
    print(f"  ✅ Recovery BLOCKED by gross_exposure > 60%")

    # Clear exposure → should recover
    engine2.update_positions([])
    engine2.update_equity(900.0)
    assert engine2.state == PortfolioRiskState.DEFENSIVE
    print(f"  ✅ Recovery ALLOWED after exposure cleared")
    print("  ✅ PASSED\n")


# ═══════════════════════════════════════════════════════════
# FIX 5: Thread-safe atomic exposure recalculation
# ═══════════════════════════════════════════════════════════

async def test_fix5_atomic_exposure_swap():
    print("⚛️  FIX 5: Atomic exposure recalculation")
    print("─" * 55)

    engine = PortfolioRiskEngine()
    engine.update_equity(10000.0)

    # Simulate concurrent exposure updates and reads
    positions_a = [
        pos("BTC", size=1.0, price=100, exchange=ExchangeID.BYBIT),
        pos("ETH", size=2.0, price=50, exchange=ExchangeID.BINANCE),
    ]
    positions_b = [
        pos("SOL", size=5.0, price=20, exchange=ExchangeID.BINANCE),
    ]

    read_results = []

    async def writer():
        for i in range(50):
            data = positions_a if i % 2 == 0 else positions_b
            await engine.update_positions_async(data)
            await asyncio.sleep(0)

    async def reader():
        for _ in range(50):
            s = engine.get_portfolio_state()
            # Each read must see a consistent snapshot:
            # gross must equal sum of all exchange exposures
            ex_total = sum(s.exposure_by_exchange.values())
            sym_total = sum(s.exposure_by_symbol.values())

            # Allow tiny float precision tolerance
            assert abs(s.total_exposure - ex_total) < 0.01, \
                f"Torn read: gross={s.total_exposure} != ex_sum={ex_total}"
            assert abs(s.total_exposure - sym_total) < 0.01, \
                f"Torn read: gross={s.total_exposure} != sym_sum={sym_total}"

            read_results.append(s.total_exposure)
            await asyncio.sleep(0)

    await asyncio.gather(writer(), reader())

    assert len(read_results) == 50
    print(f"  ✅ 50 concurrent reads, 50 concurrent writes, zero torn reads")
    print(f"  ✅ Exposure values seen: {sorted(set(int(x) for x in read_results))}")
    print("  ✅ PASSED\n")


# ═══════════════════════════════════════════════════════════
# FIX 6: Approval sequence counter
# ═══════════════════════════════════════════════════════════

async def test_fix6_approval_sequence():
    print("🔢 FIX 6: Approval sequence counter for race detection")
    print("─" * 55)

    engine = PortfolioRiskEngine()
    engine.update_equity(1000.0)

    assert engine.approval_sequence == 0
    print(f"  Initial sequence: {engine.approval_sequence}")

    # Each approve_order increments the counter
    for i in range(5):
        engine.approve_order(sig(), [], 1000.0, PARAMS)

    assert engine.approval_sequence == 5
    print(f"  After 5 approvals: {engine.approval_sequence}")

    # Counter is in state history records
    engine.update_equity(900.0)  # trigger DEFENSIVE
    history = engine.get_state_history()
    if history:
        assert "approval_seq" in history[-1], "approval_seq missing from history"
        print(f"  ✅ approval_seq in state history: {history[-1]['approval_seq']}")

    # Reset clears it
    engine.reset()
    assert engine.approval_sequence == 0
    print(f"  ✅ Reset clears sequence to 0")
    print("  ✅ PASSED\n")


# ═══════════════════════════════════════════════════════════
# Backward compat: run existing test suites
# ═══════════════════════════════════════════════════════════

async def test_backward_compat():
    print("🔄 BACKWARD COMPAT: Verify IRiskGate bridge still works")
    print("─" * 55)

    engine = PortfolioRiskEngine()
    engine.update_equity(1000.0)

    # approve() delegates to approve_order()
    v = engine.approve(sig(), [], 1000.0, PARAMS)
    assert v.approved
    print("  ✅ approve() works")

    # record_trade_result() delegates to update_after_trade()
    engine.record_trade_result(trade(5.0))
    assert engine._stats.total_trades == 1
    print("  ✅ record_trade_result() works")

    # get_sharpe_ratio() works
    sr = engine.get_sharpe_ratio()
    assert isinstance(sr, float)
    print(f"  ✅ get_sharpe_ratio() = {sr:.3f}")

    # get_max_drawdown_pct() works
    dd = engine.get_max_drawdown_pct()
    assert isinstance(dd, float)
    print(f"  ✅ get_max_drawdown_pct() = {dd:.3f}")

    # get_metrics() returns PortfolioRiskMetrics
    m = engine.get_metrics()
    assert hasattr(m, 'risk_state')
    assert hasattr(m, 'to_dict')
    d = m.to_dict()
    assert isinstance(d, dict)
    print(f"  ✅ get_metrics().to_dict() = {len(d)} keys")

    # reset() works
    engine.reset()
    assert engine._stats.total_trades == 0
    print("  ✅ reset() works")
    print("  ✅ PASSED\n")


# ─────────────────────────────────────────────────────────────

async def main():
    print("═" * 60)
    print("  🔬 DARWIN v4 — Audit Fix Regression Tests")
    print("═" * 60)

    tests = [
        test_fix1_async_locking,
        test_fix2_exchange_scoped_exposure,
        test_fix3_no_equity_mutation_in_approve,
        test_fix4_halted_recovery_constraints,
        test_fix5_atomic_exposure_swap,
        test_fix6_approval_sequence,
        test_backward_compat,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            await t()
            passed += 1
        except Exception as exc:
            failed += 1
            print(f"  ❌ FAILED: {exc}")
            import traceback
            traceback.print_exc()

    print("═" * 60)
    if failed == 0:
        print(f"  ✅ ALL {passed} AUDIT TESTS PASSED")
    else:
        print(f"  ❌ {failed} FAILED, {passed} passed")
    print("═" * 60)
    return failed == 0

if __name__ == "__main__":
    ok = asyncio.run(main())
    sys.exit(0 if ok else 1)
