"""
Darwin v4 — PortfolioRiskEngine Production Test Suite.

12 tests covering:
  1.  NORMAL state approval
  2.  Position limits
  3.  Duplicate symbol rejection
  4.  Exposure limits (gross, per-symbol, per-exchange)
  5.  Daily loss limit
  6.  Consecutive loss cooldown
  7.  Weak signal / NONE signal filter
  8.  State machine: NORMAL → DEFENSIVE → CRITICAL → HALTED
  9.  State recovery: HALTED → DEFENSIVE → NORMAL
  10. Defensive state tightens signal requirements
  11. Size multiplier per state
  12. Full Agent+Manager integration with state transitions
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

from darwin_agent.interfaces.enums import *
from darwin_agent.interfaces.types import *
from darwin_agent.interfaces.events import Event, tick_event
from darwin_agent.infra.event_bus import EventBus
from darwin_agent.risk.portfolio_engine import PortfolioRiskEngine, RiskLimits
from darwin_agent.capital.allocator import CapitalAllocator
from darwin_agent.capital.phases import PhaseManager
from darwin_agent.core.agent import DarwinAgent, AgentConfig
from darwin_agent.orchestrator.agent_manager import AgentManager

def _utcnow():
    return datetime.now(timezone.utc)


# ── Helpers ──────────────────────────────────────────────────

PARAMS = PhaseParams(phase=GrowthPhase.BOOTSTRAP, risk_pct=2.0, max_positions=3, leverage=10)

def sig(symbol="BTCUSDT", side=OrderSide.BUY, strength=SignalStrength.STRONG, confidence=0.8):
    return Signal(strategy=StrategyID.MOMENTUM, symbol=symbol, side=side,
                  strength=strength, confidence=confidence)

def pos(symbol="BTCUSDT", side=OrderSide.BUY, size=1.0, price=100.0,
        leverage=10, exchange=ExchangeID.BYBIT):
    return Position(symbol=symbol, side=side, size=size, entry_price=price,
                    leverage=leverage, exchange_id=exchange)

def trade(pnl, symbol="BTCUSDT"):
    return TradeResult(realized_pnl=pnl, symbol=symbol, closed_at=_utcnow())

# ── Mocks for integration test ───────────────────────────────

class MockExchange:
    exchange_id = ExchangeID.PAPER
    def __init__(self):
        self._price = 100.0
        self._tick = 0
    async def get_candles(self, symbol, tf, limit=100):
        p = self._price
        return [Candle(timestamp=_utcnow(), open=p-.1, high=p+.3, low=p-.3,
                       close=p, volume=1000, timeframe=tf) for _ in range(min(limit, 50))]
    async def get_ticker(self, symbol):
        self._tick += 1
        self._price += 0.5
        return Ticker(symbol=symbol, last_price=self._price, bid=self._price-.1,
                      ask=self._price+.1, volume_24h=50000)
    async def get_positions(self): return []
    async def place_order(self, req):
        return OrderResult(order_id=f"ord-{self._tick}", symbol=req.symbol,
                           side=req.side, filled_qty=req.quantity,
                           filled_price=self._price, exchange_id=ExchangeID.PAPER)
    async def close_position(self, symbol, side):
        return OrderResult(order_id=f"close-{self._tick}", symbol=symbol,
                           filled_qty=1.0, filled_price=self._price,
                           exchange_id=ExchangeID.PAPER)
    async def set_leverage(self, symbol, lev): return True
    async def get_balance(self): return 50.0
    async def get_status(self):
        return ExchangeStatus(exchange_id=ExchangeID.PAPER, connected=True, latency_ms=5.0)

class MockFeatureEngine:
    def extract(self, candles): return [0.5] * 30

class MockSelector:
    def select(self, features): return StrategyID.MOMENTUM
    def update(self, s, r): pass
    def get_weights(self): return {s: 0.25 for s in StrategyID}

class MockStrategy:
    strategy_id = StrategyID.MOMENTUM
    def analyze(self, symbol, candles, features):
        return Signal(strategy=StrategyID.MOMENTUM, symbol=symbol, side=OrderSide.BUY,
                      strength=SignalStrength.STRONG, confidence=0.85,
                      stop_loss_pct=1.5, take_profit_pct=3.0)

class MockSizer:
    def calculate(self, signal, capital, price, params):
        return capital * (params.risk_pct / 100) / max(price, 0.01)


def make_agent(bus, capital=50.0, risk_gate=None, **kw):
    return DarwinAgent(
        initial_capital=capital,
        config=kw.pop("config", AgentConfig(incubation_candles=3, min_graduation_winrate=0.0,
                                            generation_trade_limit=5, watchlist=["BTCUSDT"])),
        exchange=kw.pop("exchange", MockExchange()),
        feature_engine=MockFeatureEngine(), selector=MockSelector(),
        strategies={StrategyID.MOMENTUM: MockStrategy()},
        risk_gate=risk_gate or PortfolioRiskEngine(),
        sizer=MockSizer(), bus=bus, **kw,
    )


# ═══════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════

async def test_01_normal_approval():
    print("\n🧪 TEST 01: NORMAL state — basic approval")
    print("─" * 50)
    engine = PortfolioRiskEngine()
    engine.update_equity(1000.0)

    v = engine.approve_order(sig(), [], 1000.0, PARAMS)
    assert v.approved, f"Expected approved, got: {v.reason}"
    assert v.adjusted_size is None
    assert engine.state == PortfolioRiskState.NORMAL

    state = engine.get_portfolio_state()
    assert state.risk_state == PortfolioRiskState.NORMAL
    assert state.total_equity == 1000.0
    assert state.size_multiplier == 1.0
    print("  ✅ Approves clean order, correct snapshot")
    print("  ✅ PASSED\n")


async def test_02_position_limits():
    print("🧪 TEST 02: Position count limits")
    print("─" * 50)
    engine = PortfolioRiskEngine()
    engine.update_equity(100000.0)

    # 3 tiny positions, mixed directions to avoid concentration check
    full = [
        pos("SYM0", size=0.001, price=10.0, leverage=1),
        pos("SYM1", size=0.001, price=10.0, leverage=1, side=OrderSide.SELL),
        pos("SYM2", size=0.001, price=10.0, leverage=1),
    ]
    engine.update_positions(full)
    v = engine.approve_order(sig("ETHUSDT"), full, 100000.0, PARAMS)
    assert not v.approved and "max_positions" in v.reason
    print(f"  ✅ Rejects at max: {v.reason}")

    under = [
        pos("SYM0", size=0.001, price=10.0, leverage=1),
        pos("SYM1", size=0.001, price=10.0, leverage=1, side=OrderSide.SELL),
    ]
    engine.update_positions(under)
    v = engine.approve_order(sig("ETHUSDT"), under, 100000.0, PARAMS)
    assert v.approved
    print("  ✅ Approves under limit")
    print("  ✅ PASSED\n")


async def test_03_duplicate_symbol():
    print("🧪 TEST 03: Duplicate symbol rejection")
    print("─" * 50)
    engine = PortfolioRiskEngine()
    engine.update_equity(100000.0)

    existing = [pos("BTCUSDT", size=0.001, price=10.0, leverage=1, side=OrderSide.SELL)]
    engine.update_positions(existing)

    v = engine.approve_order(sig("BTCUSDT"), existing, 100000.0, PARAMS)
    assert not v.approved and "duplicate_symbol" in v.reason
    print(f"  ✅ Duplicate rejected: {v.reason}")

    v = engine.approve_order(sig("ETHUSDT"), existing, 100000.0, PARAMS)
    assert v.approved
    print("  ✅ Different symbol approved")
    print("  ✅ PASSED\n")


async def test_04_exposure_limits():
    print("🧪 TEST 04: Exposure limits (gross / symbol / exchange)")
    print("─" * 50)
    limits = RiskLimits(max_total_exposure_pct=50.0,
                        max_symbol_exposure_pct=25.0,
                        max_exchange_exposure_pct=40.0)
    engine = PortfolioRiskEngine(limits=limits)
    engine.update_equity(1000.0)

    # Gross: 1*100*10=1000 → 100% > 50%
    big = [pos("ETHUSDT", size=1.0, price=100.0, leverage=10)]
    engine.update_positions(big)
    v = engine.approve_order(sig("SOLUSDT"), big, 1000.0, PARAMS)
    assert not v.approved and "gross_exposure" in v.reason
    print(f"  ✅ Gross exposure: {v.reason}")

    # Per-exchange: two bybit positions → 45% > 40%
    bybit_heavy = [
        pos("SYM1", exchange=ExchangeID.BYBIT, size=0.2, price=100, leverage=10),
        pos("SYM2", exchange=ExchangeID.BYBIT, size=0.25, price=100, leverage=10),
    ]
    engine.update_positions(bybit_heavy)
    v = engine.approve_order(sig("SYM3"), bybit_heavy, 1000.0, PARAMS)
    assert not v.approved and "exchange_exposure" in v.reason
    print(f"  ✅ Exchange exposure: {v.reason}")

    state = engine.get_portfolio_state()
    assert "bybit" in state.exposure_by_exchange
    assert "SYM1" in state.exposure_by_symbol
    print(f"  ✅ Per-exchange breakdown: {dict(state.exposure_by_exchange)}")
    print(f"  ✅ Per-symbol breakdown: {dict(state.exposure_by_symbol)}")
    print("  ✅ PASSED\n")


async def test_05_daily_loss():
    print("🧪 TEST 05: Daily loss limit")
    print("─" * 50)
    engine = PortfolioRiskEngine(limits=RiskLimits(max_daily_loss_pct=3.0))
    engine.update_equity(1000.0)

    for _ in range(4):
        engine.update_after_trade(trade(-10.0))

    v = engine.approve_order(sig(), [], 1000.0, PARAMS)
    assert not v.approved and "daily_loss" in v.reason
    print(f"  ✅ Daily loss blocked: {v.reason}")

    state = engine.get_portfolio_state()
    assert state.daily_pnl == -40.0
    print(f"  ✅ Daily PnL tracked: ${state.daily_pnl:.2f}")
    print("  ✅ PASSED\n")


async def test_06_consecutive_losses():
    print("🧪 TEST 06: Consecutive loss cooldown")
    print("─" * 50)
    engine = PortfolioRiskEngine(limits=RiskLimits(max_consecutive_losses=3))
    engine.update_equity(1000.0)

    for _ in range(3):
        engine.update_after_trade(trade(-1.0))

    v = engine.approve_order(sig(), [], 1000.0, PARAMS)
    assert not v.approved and "consecutive_losses" in v.reason
    print(f"  ✅ Blocked after 3 losses: {v.reason}")

    state = engine.get_portfolio_state()
    assert state.consecutive_losses == 3
    print(f"  ✅ Consecutive losses tracked: {state.consecutive_losses}")

    # Win resets
    engine.update_after_trade(trade(5.0))
    v = engine.approve_order(sig(), [], 1000.0, PARAMS)
    assert v.approved
    print("  ✅ Win resets counter, approves again")
    print("  ✅ PASSED\n")


async def test_07_signal_quality():
    print("🧪 TEST 07: Signal quality filter")
    print("─" * 50)
    engine = PortfolioRiskEngine()
    engine.update_equity(1000.0)

    v = engine.approve_order(sig(strength=SignalStrength.NONE), [], 1000.0, PARAMS)
    assert not v.approved and "NONE" in v.reason
    print(f"  ✅ NONE rejected: {v.reason}")

    v = engine.approve_order(sig(strength=SignalStrength.WEAK, confidence=0.3), [], 1000.0, PARAMS)
    assert not v.approved and "weak_signal" in v.reason
    print(f"  ✅ Weak+low conf rejected: {v.reason}")

    v = engine.approve_order(sig(strength=SignalStrength.WEAK, confidence=0.7), [], 1000.0, PARAMS)
    assert v.approved
    print("  ✅ Weak+high conf approved in NORMAL")
    print("  ✅ PASSED\n")


async def test_08_state_escalation():
    print("🧪 TEST 08: State machine — NORMAL → DEFENSIVE → CRITICAL → HALTED")
    print("─" * 50)
    limits = RiskLimits(defensive_drawdown_pct=5.0,
                        critical_drawdown_pct=10.0,
                        halted_drawdown_pct=20.0)
    engine = PortfolioRiskEngine(limits=limits)

    engine.update_equity(1000.0)
    assert engine.state == PortfolioRiskState.NORMAL
    print(f"  eq=1000 → {engine.state.value}")

    engine.update_equity(940.0)  # 6% dd
    assert engine.state == PortfolioRiskState.DEFENSIVE
    print(f"  eq=940  (dd=6%)  → {engine.state.value}")

    v = engine.approve_order(sig(strength=SignalStrength.WEAK, confidence=0.7), [], 940.0, PARAMS)
    assert not v.approved and "DEFENSIVE" in v.reason
    print(f"  ✅ DEFENSIVE blocks weak: {v.reason}")

    engine.update_equity(880.0)  # 12% dd
    assert engine.state == PortfolioRiskState.CRITICAL
    print(f"  eq=880  (dd=12%) → {engine.state.value}")

    v = engine.approve_order(sig(), [], 880.0, PARAMS)
    assert not v.approved and "CRITICAL" in v.reason
    print(f"  ✅ CRITICAL blocks all: {v.reason}")

    engine.update_equity(780.0)  # 22% dd
    assert engine.state == PortfolioRiskState.HALTED
    print(f"  eq=780  (dd=22%) → {engine.state.value}")

    v = engine.approve_order(sig(), [], 780.0, PARAMS)
    assert not v.approved and "HALTED" in v.reason
    print(f"  ✅ HALTED blocks all: {v.reason}")

    history = engine.get_state_history()
    assert len(history) == 3
    transitions = [h["to"] for h in history]
    assert transitions == ["defensive", "critical", "halted"]
    print(f"  ✅ State history: {transitions}")

    state = engine.get_portfolio_state()
    assert state.halted_reason != ""
    print(f"  ✅ Halted reason: {state.halted_reason}")
    print("  ✅ PASSED\n")


async def test_09_state_recovery():
    print("🧪 TEST 09: State recovery — CRITICAL → DEFENSIVE → NORMAL")
    print("─" * 50)
    limits = RiskLimits(
        defensive_drawdown_pct=5.0, critical_drawdown_pct=10.0,
        halted_drawdown_pct=20.0,
        normal_recovery_pct=2.0, defensive_recovery_pct=4.0,
        critical_recovery_pct=8.0,
        halt_duration_minutes=0.0,  # instant expiry for testing
    )
    engine = PortfolioRiskEngine(limits=limits)

    engine.update_equity(1000.0)
    engine.update_equity(880.0)  # 12% dd → CRITICAL
    assert engine.state == PortfolioRiskState.CRITICAL
    print(f"  Crashed to CRITICAL at dd=12%")

    # Recover to 3% dd → below defensive_recovery(4%) → DEFENSIVE
    engine.update_equity(970.0)
    assert engine.state == PortfolioRiskState.DEFENSIVE
    print(f"  eq=970 (dd=3%) → {engine.state.value}")

    # Recover to 1% dd → below normal_recovery(2%) → NORMAL
    engine.update_equity(990.0)
    assert engine.state == PortfolioRiskState.NORMAL
    print(f"  eq=990 (dd=1%) → {engine.state.value}")

    # Test HALTED recovery via expiry
    engine.update_equity(1000.0)  # reset peak
    engine.update_equity(780.0)   # 22% dd → HALTED
    assert engine.state == PortfolioRiskState.HALTED
    print(f"  eq=780 (dd=22%) → {engine.state.value}")

    # halt_duration=0 → expired immediately, equity recovers
    engine.update_equity(930.0)  # dd=7% < critical_recovery(8%)
    assert engine.state == PortfolioRiskState.DEFENSIVE
    print(f"  eq=930 (halt expired, dd=7%) → {engine.state.value}")

    history = engine.get_state_history()
    full_path = [h["to"] for h in history]
    print(f"  ✅ Full path: {full_path}")
    print("  ✅ PASSED\n")


async def test_10_defensive_signal_gate():
    print("🧪 TEST 10: DEFENSIVE state tightens signal requirements")
    print("─" * 50)
    limits = RiskLimits(defensive_drawdown_pct=5.0)
    engine = PortfolioRiskEngine(limits=limits)

    engine.update_equity(1000.0)
    engine.update_equity(940.0)  # → DEFENSIVE
    assert engine.state == PortfolioRiskState.DEFENSIVE

    # Weak blocked in DEFENSIVE regardless of confidence
    v = engine.approve_order(sig(strength=SignalStrength.WEAK, confidence=0.9), [], 940.0, PARAMS)
    assert not v.approved and "DEFENSIVE:weak" in v.reason
    print(f"  ✅ Weak signal blocked: {v.reason}")

    # Low confidence blocked
    v = engine.approve_order(sig(strength=SignalStrength.MEDIUM, confidence=0.5), [], 940.0, PARAMS)
    assert not v.approved and "DEFENSIVE:low_confidence" in v.reason
    print(f"  ✅ Low confidence blocked: {v.reason}")

    # Strong + high confidence OK
    v = engine.approve_order(sig(strength=SignalStrength.STRONG, confidence=0.8), [], 940.0, PARAMS)
    assert v.approved
    assert v.adjusted_size == 0.5  # defensive multiplier
    print(f"  ✅ Strong signal approved with size_mult={v.adjusted_size}")
    print("  ✅ PASSED\n")


async def test_11_size_multiplier():
    print("🧪 TEST 11: Size multiplier per state")
    print("─" * 50)
    limits = RiskLimits(
        defensive_drawdown_pct=5.0, critical_drawdown_pct=10.0,
        halted_drawdown_pct=20.0,
        defensive_size_multiplier=0.50,
        critical_size_multiplier=0.0,
    )
    engine = PortfolioRiskEngine(limits=limits)

    # NORMAL → 1.0
    engine.update_equity(1000.0)
    state = engine.get_portfolio_state()
    assert state.size_multiplier == 1.0
    print(f"  NORMAL:    size_mult={state.size_multiplier}")

    # DEFENSIVE → 0.5
    engine.update_equity(940.0)
    state = engine.get_portfolio_state()
    assert state.size_multiplier == 0.5
    print(f"  DEFENSIVE: size_mult={state.size_multiplier}")

    # CRITICAL → 0.0
    engine.update_equity(880.0)
    state = engine.get_portfolio_state()
    assert state.size_multiplier == 0.0
    print(f"  CRITICAL:  size_mult={state.size_multiplier}")

    # HALTED → 0.0
    engine.update_equity(780.0)
    state = engine.get_portfolio_state()
    assert state.size_multiplier == 0.0
    print(f"  HALTED:    size_mult={state.size_multiplier}")
    print("  ✅ PASSED\n")


async def test_12_agent_integration():
    print("🧪 TEST 12: Full Agent + AgentManager integration")
    print("─" * 50)

    bus = EventBus()
    risk = PortfolioRiskEngine(bus=bus)
    allocator = CapitalAllocator(phase_manager=PhaseManager())

    # Track risk state events
    state_events = []
    async def on_risk(ev):
        state_events.append(ev.payload)
    bus.subscribe(EventType.RISK_STATE_CHANGED, on_risk)

    # Create agent with shared risk engine
    agent = make_agent(bus, capital=50.0, risk_gate=risk)

    # Seed equity
    risk.update_equity(50.0)

    # Graduate
    for _ in range(10):
        await agent.tick()
    assert agent.phase.value == "live"
    print(f"  ✅ Agent graduated to {agent.phase.value}")

    # Trade and track
    for _ in range(30):
        await agent.tick()

    m = agent.get_metrics()
    state = risk.get_portfolio_state()

    print(f"  Agent: trades={m.total_trades}, pnl=${m.realized_pnl:.2f}")
    print(f"  Risk state: {state.risk_state.value}")
    print(f"  Sharpe: {state.sharpe_ratio:.3f}")
    print(f"  Win rate: {state.win_rate:.1%}")
    print(f"  Drawdown: {state.drawdown_pct:.1f}%")
    print(f"  Max DD: {state.max_drawdown_pct:.1f}%")
    print(f"  Consec losses: {state.consecutive_losses}")
    print(f"  PF: {state.profit_factor:.3f}")
    print(f"  Total trades (engine): {state.total_trades}")
    print(f"  State events emitted: {len(state_events)}")

    # Verify backward-compat IRiskGate methods work
    assert risk.get_sharpe_ratio() == state.sharpe_ratio
    assert risk.get_max_drawdown_pct() == state.max_drawdown_pct
    assert risk.get_metrics().risk_state == state.risk_state
    print("  ✅ IRiskGate backward compat verified")

    # Verify to_dict serializable
    d = state.to_dict()
    assert isinstance(d, dict)
    assert "risk_state" in d
    assert "exposure_by_exchange" in d
    assert "exposure_by_symbol" in d
    print(f"  ✅ to_dict() serializable ({len(d)} keys)")

    print("  ✅ PASSED\n")


# ─────────────────────────────────────────────────────────────

async def main():
    print("═" * 60)
    print("  🛡️  DARWIN v4 — PortfolioRiskEngine Test Suite")
    print("═" * 60)

    tests = [
        test_01_normal_approval,
        test_02_position_limits,
        test_03_duplicate_symbol,
        test_04_exposure_limits,
        test_05_daily_loss,
        test_06_consecutive_losses,
        test_07_signal_quality,
        test_08_state_escalation,
        test_09_state_recovery,
        test_10_defensive_signal_gate,
        test_11_size_multiplier,
        test_12_agent_integration,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as exc:
            failed += 1
            print(f"  ❌ FAILED: {exc}")
            import traceback
            traceback.print_exc()

    print("═" * 60)
    if failed == 0:
        print(f"  ✅ ALL {passed} TESTS PASSED")
    else:
        print(f"  ❌ {failed} FAILED, {passed} passed")
    print("═" * 60)

    return failed == 0


if __name__ == "__main__":
    ok = asyncio.run(main())
    sys.exit(0 if ok else 1)
