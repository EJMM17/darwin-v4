"""
Darwin v4 — Architecture Hardening Regression Tests.
Tests all 6 audit areas across the full architecture.
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from collections import deque, OrderedDict
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

from darwin_agent.interfaces.enums import *
from darwin_agent.interfaces.events import Event, EventType
from darwin_agent.interfaces.types import *
from darwin_agent.infra.event_bus import EventBus, MAX_LISTENERS_PER_TYPE
from darwin_agent.orchestrator.agent_manager import AgentManager, PoolCheckpoint
from darwin_agent.exchanges.router import ExchangeRouter
from darwin_agent.core.agent import DarwinAgent, AgentConfig
from darwin_agent.evolution.engine import EvolutionEngine
from darwin_agent.risk.portfolio_engine import PortfolioRiskEngine, RiskLimits

def _utcnow():
    return datetime.now(timezone.utc)

def _make_test_agent(risk_gate):
    bus = EventBus()
    exchange = MagicMock()
    features = MagicMock()
    features.extract = MagicMock(return_value=[0.5] * 10)
    selector = MagicMock()
    selector.select = MagicMock(return_value=StrategyID.MOMENTUM)
    selector.update = MagicMock()
    strategy = MagicMock()
    strategy.analyze = MagicMock(return_value=None)
    sizer = MagicMock()
    sizer.calculate = MagicMock(return_value=0.01)
    return DarwinAgent(
        agent_id="test-agent", generation=0, initial_capital=1000.0,
        exchange=exchange, feature_engine=features, selector=selector,
        strategies={StrategyID.MOMENTUM: strategy},
        risk_gate=risk_gate, sizer=sizer, bus=bus,
    )


# ═══════════════════════════════════════════════════════════
# 1. CONCURRENCY SAFETY
# ═══════════════════════════════════════════════════════════

async def test_eventbus_snapshot_dispatch():
    """H-EB2: Handlers subscribing during dispatch don't corrupt iteration."""
    print("\n🔒 1a: EventBus snapshot dispatch")
    print("─" * 50)
    bus = EventBus()
    calls = []
    async def handler_a(e):
        calls.append("A")
        bus.subscribe(EventType.TICK, handler_c)
    async def handler_b(e): calls.append("B")
    async def handler_c(e): calls.append("C")

    bus.subscribe(EventType.TICK, handler_a)
    bus.subscribe(EventType.TICK, handler_b)
    await bus.emit(Event(event_type=EventType.TICK, source="t"))
    assert calls == ["A", "B"], f"Got {calls}"
    print(f"  ✅ First emit: {calls} (C not yet active)")
    calls.clear()
    await bus.emit(Event(event_type=EventType.TICK, source="t"))
    assert "C" in calls
    print(f"  ✅ Second emit: C now active")
    print("  ✅ PASSED\n")

async def test_eventbus_listener_cap():
    """H-EB3: Cap prevents OOM."""
    print("🔒 1b: EventBus listener cap")
    print("─" * 50)
    bus = EventBus()
    for i in range(MAX_LISTENERS_PER_TYPE + 5):
        async def noop(e): pass
        bus.subscribe(EventType.TICK, noop)
    assert len(bus._listeners[EventType.TICK]) == MAX_LISTENERS_PER_TYPE
    print(f"  ✅ Capped at {MAX_LISTENERS_PER_TYPE}")
    print("  ✅ PASSED\n")

async def test_eventbus_shutdown():
    """H-EB4: Events after clear() are no-ops."""
    print("🔒 1c: EventBus shutdown")
    print("─" * 50)
    bus = EventBus()
    fired = []
    async def h(e): fired.append(1)
    bus.subscribe(EventType.TICK, h)
    bus.clear()
    await bus.emit(Event(event_type=EventType.TICK, source="t"))
    assert len(fired) == 0
    print(f"  ✅ No dispatch after clear()")
    print("  ✅ PASSED\n")

async def test_eventbus_bounded_history():
    """H-EB1: deque(maxlen=500)."""
    print("🔒 1d: EventBus bounded history")
    print("─" * 50)
    bus = EventBus()
    for i in range(1000):
        await bus.emit(Event(event_type=EventType.TICK, source=f"t{i}"))
    assert len(bus._history) <= 500
    assert isinstance(bus._history, deque)
    print(f"  ✅ After 1000 emits: history={len(bus._history)}")
    print("  ✅ PASSED\n")

async def test_tick_overlap_guard():
    """H-AM1: Second tick skips if first still running."""
    print("🔒 1e: AgentManager tick overlap guard")
    print("─" * 50)
    bus = EventBus()
    alloc = MagicMock()
    alloc.get_phase_params = MagicMock(return_value=PhaseParams(
        phase=GrowthPhase.BOOTSTRAP, risk_pct=2.0, max_positions=2, leverage=10,
    ))
    alloc.allocate = MagicMock(return_value={})
    tick_count = []
    def factory(aid, gen, cap, dna):
        a = MagicMock()
        a.is_alive = True; a.phase = AgentPhase.LIVE
        a.capital = cap; a.agent_id = aid
        async def slow(): tick_count.append(1); await asyncio.sleep(0.1)
        a.tick = slow
        a.get_metrics = MagicMock(return_value=AgentMetrics(agent_id=aid, generation=gen, phase="live", capital=cap))
        return a
    mgr = AgentManager(agent_factory=factory, allocator=alloc, bus=bus, total_capital=100.0)
    await mgr.start()
    ev = Event(event_type=EventType.TICK, source="t")
    await asyncio.gather(mgr._on_tick(ev), mgr._on_tick(ev))
    assert len(tick_count) == 1, f"Got {len(tick_count)} ticks"
    print(f"  ✅ Only 1 tick ran (overlap skipped)")
    print("  ✅ PASSED\n")


# ═══════════════════════════════════════════════════════════
# 2. PERSISTENCE CONSISTENCY
# ═══════════════════════════════════════════════════════════

async def test_checkpoint_save_load():
    """H-AM3: Pool state survives restart."""
    print("💾 2a: Checkpoint save/load")
    print("─" * 50)
    store = {}
    class Cache:
        async def get_json(self, k): return store.get(k)
        async def set_json(self, k, v, ttl_seconds=300): store[k] = v
    cache = Cache()
    bus = EventBus()
    alloc = MagicMock()
    alloc.get_phase_params = MagicMock(return_value=PhaseParams(
        phase=GrowthPhase.BOOTSTRAP, risk_pct=2.0, max_positions=2, leverage=10,
    ))
    mgr1 = AgentManager(agent_factory=MagicMock(), allocator=alloc, bus=bus,
                         total_capital=500.0, cache=cache)
    mgr1._generation_counter = 7; mgr1._tick_counter = 42
    await mgr1._save_checkpoint()
    assert "pool:checkpoint" in store
    print(f"  ✅ Saved: {store['pool:checkpoint']}")

    mgr2 = AgentManager(agent_factory=MagicMock(), allocator=alloc, bus=bus,
                         total_capital=50.0, cache=cache)
    await mgr2._load_checkpoint()
    assert mgr2._total_capital == 500.0
    assert mgr2._generation_counter == 7
    print(f"  ✅ Restored: capital=${mgr2._total_capital}, gen={mgr2._generation_counter}")
    print("  ✅ PASSED\n")


# ═══════════════════════════════════════════════════════════
# 3. SAFE RESTART RECOVERY
# ═══════════════════════════════════════════════════════════

async def test_checkpoint_no_downgrade():
    """H-AM3: Checkpoint never downgrades capital."""
    print("🔄 3a: Checkpoint no downgrade")
    print("─" * 50)
    store = {"pool:checkpoint": {"total_capital": 10.0, "generation_counter": 2, "tick_counter": 5}}
    class Cache:
        async def get_json(self, k): return store.get(k)
        async def set_json(self, k, v, ttl_seconds=300): store[k] = v
    bus = EventBus()
    alloc = MagicMock()
    alloc.get_phase_params = MagicMock(return_value=PhaseParams(
        phase=GrowthPhase.BOOTSTRAP, risk_pct=2.0, max_positions=2, leverage=10,
    ))
    mgr = AgentManager(agent_factory=MagicMock(), allocator=alloc, bus=bus,
                        total_capital=100.0, cache=Cache())
    await mgr._load_checkpoint()
    assert mgr._total_capital == 100.0, f"Downgraded to {mgr._total_capital}"
    assert mgr._generation_counter == 2
    print(f"  ✅ Capital stayed $100 (not downgraded to $10), gen=2")
    print("  ✅ PASSED\n")


# ═══════════════════════════════════════════════════════════
# 4. MEMORY LEAK PREVENTION
# ═══════════════════════════════════════════════════════════

async def test_agent_bounded_trades():
    """H-AG1: closed_trades deque(maxlen=500)."""
    print("📦 4a: Agent bounded trades")
    print("─" * 50)
    risk = PortfolioRiskEngine(); risk.update_equity(1000.0)
    agent = _make_test_agent(risk)
    agent._set_phase(AgentPhase.LIVE)
    for i in range(600):
        agent._on_trade_closed(TradeResult(realized_pnl=0.1, closed_at=_utcnow(),
                                           symbol=f"S{i}", side=OrderSide.BUY))
    assert len(agent._closed_trades) == 500
    assert isinstance(agent._closed_trades, deque)
    print(f"  ✅ After 600 trades: {len(agent._closed_trades)} (maxlen=500)")
    print("  ✅ PASSED\n")

async def test_router_bounded_symbol_map():
    """H-ER3: symbol_map caps at 500."""
    print("📦 4b: Router bounded symbol map")
    print("─" * 50)
    router = ExchangeRouter(adapters={ExchangeID.BYBIT: MagicMock()}, primary=ExchangeID.BYBIT)
    for i in range(600):
        router._record_symbol(f"SYM{i}", ExchangeID.BYBIT)
    assert len(router._symbol_map) == 500
    assert "SYM0" not in router._symbol_map
    assert "SYM599" in router._symbol_map
    print(f"  ✅ After 600 symbols: {len(router._symbol_map)} (evicted oldest)")
    print("  ✅ PASSED\n")

async def test_dead_history_bounded():
    """H-AM2: deque(maxlen=200)."""
    print("📦 4c: AgentManager bounded dead history")
    print("─" * 50)
    mgr = AgentManager(agent_factory=MagicMock(), allocator=MagicMock(),
                        bus=EventBus(), total_capital=100.0)
    for i in range(300):
        mgr._dead_history.append(AgentMetrics(agent_id=f"d{i}"))
    assert len(mgr._dead_history) == 200
    assert isinstance(mgr._dead_history, deque)
    print(f"  ✅ After 300 deaths: {len(mgr._dead_history)} (maxlen=200)")
    print("  ✅ PASSED\n")


# ═══════════════════════════════════════════════════════════
# 5. DEADLOCK PREVENTION
# ═══════════════════════════════════════════════════════════

async def test_router_timeout():
    """H-ER2: Hung exchange call times out."""
    print("⏰ 5a: Router call timeout")
    print("─" * 50)
    async def hang(*a, **k): await asyncio.sleep(100)
    adapter = MagicMock(); adapter.get_ticker = hang
    router = ExchangeRouter(adapters={ExchangeID.BYBIT: adapter},
                            primary=ExchangeID.BYBIT, call_timeout=0.1)
    router._statuses[ExchangeID.BYBIT] = ExchangeStatus(exchange_id=ExchangeID.BYBIT, connected=True)
    router._status_timestamps[ExchangeID.BYBIT] = _utcnow()
    try:
        await router.get_ticker("BTCUSDT")
        assert False, "Should timeout"
    except asyncio.TimeoutError:
        pass
    print(f"  ✅ Timed out after 100ms")
    print("  ✅ PASSED\n")


# ═══════════════════════════════════════════════════════════
# 6. RACE CONDITIONS
# ═══════════════════════════════════════════════════════════

async def test_agent_die_once():
    """H-AG2: _die called twice only executes once."""
    print("🏁 6a: Agent die-once guard")
    print("─" * 50)
    risk = PortfolioRiskEngine(); risk.update_equity(1000.0)
    agent = _make_test_agent(risk)
    agent._set_phase(AgentPhase.LIVE)
    die_events = []
    orig = agent._bus.emit
    async def track(event):
        if event.event_type == EventType.AGENT_DIED: die_events.append(1)
        await orig(event)
    agent._bus.emit = track
    await asyncio.gather(agent._die("r1"), agent._die("r2"))
    assert len(die_events) == 1, f"Got {len(die_events)} die events"
    assert agent.phase == AgentPhase.DEAD
    print(f"  ✅ Only 1 DIED event (not 2)")
    print("  ✅ PASSED\n")

async def test_closing_symbols_guard():
    """H-AG4: Can't open on symbol being closed."""
    print("🏁 6b: Closing symbols guard")
    print("─" * 50)
    risk = PortfolioRiskEngine(); risk.update_equity(1000.0)
    agent = _make_test_agent(risk)
    agent._set_phase(AgentPhase.LIVE)
    agent._closing_symbols.add("BTCUSDT")

    # Verify the guard exists: BTCUSDT is in watchlist but should be skipped.
    # We can verify by checking that _tick_live doesn't try to get a signal
    # for BTCUSDT. Since _gen_signal calls exchange.get_candles, we track that.
    candle_calls = []
    original_get_candles = agent._exchange.get_candles
    async def track_candles(symbol, *a, **k):
        candle_calls.append(symbol)
        return []  # empty = skip
    agent._exchange.get_candles = track_candles

    await agent._tick_live()
    assert "BTCUSDT" not in candle_calls, f"BTCUSDT scanned despite closing: {candle_calls}"
    print(f"  ✅ BTCUSDT skipped (in closing set, scanned: {candle_calls})")
    print("  ✅ PASSED\n")

async def test_router_close_failover():
    """H-ER1: close_position fails over."""
    print("🏁 6c: Router close failover")
    print("─" * 50)
    async def fail(*a, **k): raise ConnectionError("down")
    async def ok(*a, **k): return OrderResult(success=True, filled_price=100.0, filled_qty=1.0)
    bybit = MagicMock(); bybit.close_position = fail
    binance = MagicMock(); binance.close_position = ok
    router = ExchangeRouter(
        adapters={ExchangeID.BYBIT: bybit, ExchangeID.BINANCE: binance},
        primary=ExchangeID.BYBIT,
    )
    router._symbol_map["BTCUSDT"] = ExchangeID.BYBIT
    result = await router.close_position("BTCUSDT", OrderSide.BUY)
    assert result.success
    print(f"  ✅ Failed on Bybit, succeeded on Binance")
    print("  ✅ PASSED\n")

async def test_evolution_pending_dna():
    """H-EV1: DNA uses pending queue, not fire-and-forget."""
    print("🏁 6d: Evolution pending DNA")
    print("─" * 50)
    engine = EvolutionEngine()
    survivors = [engine.create_random_dna() for _ in range(3)]
    for s in survivors: s.fitness = 0.5
    saved = []
    class Repo:
        async def save_dna(self, dna): saved.append(dna.dna_id)
    engine._dna_repo = Repo()
    engine.create_next_generation(survivors, target_size=5)
    assert len(engine._pending_dna) == 5
    count = await engine.flush_pending_dna()
    assert count == 5 and len(saved) == 5
    print(f"  ✅ 5 DNA queued then flushed")
    print("  ✅ PASSED\n")

async def test_capital_floor():
    """H-AM4: Spawn capital floored at 0.01."""
    print("🏁 6e: Capital floor")
    print("─" * 50)
    mgr = AgentManager(agent_factory=MagicMock(), allocator=MagicMock(),
                        bus=EventBus(), total_capital=0.0)
    cap = mgr._calc_spawn_capital()
    assert cap >= 0.01, f"Got {cap}"
    print(f"  ✅ With $0 pool: spawn capital=${cap}")
    print("  ✅ PASSED\n")


# ─────────────────────────────────────────────────
async def main():
    print("═" * 60)
    print("  🏗️  DARWIN v4 — Architecture Hardening Tests")
    print("═" * 60)
    tests = [
        test_eventbus_snapshot_dispatch,
        test_eventbus_listener_cap,
        test_eventbus_shutdown,
        test_eventbus_bounded_history,
        test_tick_overlap_guard,
        test_checkpoint_save_load,
        test_checkpoint_no_downgrade,
        test_agent_bounded_trades,
        test_router_bounded_symbol_map,
        test_dead_history_bounded,
        test_router_timeout,
        test_agent_die_once,
        test_closing_symbols_guard,
        test_router_close_failover,
        test_evolution_pending_dna,
        test_capital_floor,
    ]
    passed = failed = 0
    for t in tests:
        try:
            await t()
            passed += 1
        except Exception as exc:
            failed += 1
            print(f"  ❌ FAILED: {t.__name__}: {exc}")
            import traceback; traceback.print_exc()
    print("═" * 60)
    if failed == 0:
        print(f"  ✅ ALL {passed} ARCHITECTURE HARDENING TESTS PASSED")
    else:
        print(f"  ❌ {failed} FAILED, {passed} passed")
    print("═" * 60)
    return failed == 0

if __name__ == "__main__":
    ok = asyncio.run(main())
    sys.exit(0 if ok else 1)
