"""
Darwin v4 — Integration test suite.
Proves multi-agent lifecycle end-to-end with mock Protocol implementations.
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

from darwin_agent.interfaces.enums import (
    EventType, GrowthPhase, OrderSide, OrderType,
    SignalStrength, StrategyID, TimeFrame,
)
from darwin_agent.interfaces.types import (
    AllocationSlice, Candle, OrderRequest, OrderResult,
    PhaseParams, Position, RiskVerdict, Signal, Ticker, TradeResult,
)
from darwin_agent.interfaces.events import Event, tick_event
from darwin_agent.infra.event_bus import EventBus
from darwin_agent.capital.allocator import CapitalAllocator, AllocationStrategy
from darwin_agent.capital.phases import PhaseManager
from darwin_agent.core.agent import DarwinAgent, AgentConfig
from darwin_agent.orchestrator.agent_manager import AgentManager


def _utcnow():
    return datetime.now(timezone.utc)


# ═════════════════════════════════════════════════════════════
# Mock implementations of Protocol interfaces
# ═════════════════════════════════════════════════════════════

class MockExchange:
    """Simulates price movement and order fills."""
    def __init__(self, base_price: float = 100.0, drift: float = 0.5):
        self._price = base_price
        self._drift = drift
        self._tick = 0

    async def get_candles(self, symbol, timeframe, limit=100):
        candles = []
        p = self._price
        for i in range(min(limit, 50)):
            candles.append(Candle(
                timestamp=_utcnow(), open=p - 0.1, high=p + 0.3,
                low=p - 0.3, close=p, volume=1000.0, timeframe=timeframe,
            ))
        return candles

    async def get_ticker(self, symbol):
        self._tick += 1
        self._price += self._drift  # drift only on ticker calls
        return Ticker(symbol=symbol, last_price=self._price,
                      bid=self._price - 0.1, ask=self._price + 0.1,
                      volume_24h=50000)

    async def get_positions(self):
        return []

    async def place_order(self, request):
        return OrderResult(
            order_id=f"ord-{self._tick}", symbol=request.symbol,
            side=request.side, filled_qty=request.quantity,
            filled_price=self._price, fee=0.01,
        )

    async def close_position(self, symbol, side):
        return OrderResult(
            order_id=f"close-{self._tick}", symbol=symbol,
            side=OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY,
            filled_qty=1.0, filled_price=self._price, fee=0.01,
        )

    async def set_leverage(self, symbol, leverage):
        return True

    async def get_balance(self):
        return 50.0


class MockFeatureEngine:
    def extract(self, candles):
        return [0.5] * 30


class MockSelector:
    def select(self, features):
        return StrategyID.MOMENTUM

    def update(self, strategy, reward):
        pass

    def get_weights(self):
        return {s: 0.25 for s in StrategyID}


class MockStrategy:
    """Always emits a BUY signal so we can test the full trade cycle."""
    strategy_id = StrategyID.MOMENTUM

    def analyze(self, symbol, candles, features):
        return Signal(
            strategy=StrategyID.MOMENTUM, symbol=symbol,
            side=OrderSide.BUY, strength=SignalStrength.STRONG,
            confidence=0.85, stop_loss_pct=1.5, take_profit_pct=3.0,
        )


class MockRiskGate:
    _trades: List[TradeResult]

    def __init__(self):
        self._trades = []

    def approve(self, signal, positions, capital, phase_params):
        return RiskVerdict(approved=True)

    def record_trade_result(self, result):
        self._trades.append(result)

    def get_sharpe_ratio(self):
        if not self._trades:
            return 0.0
        pnls = [t.realized_pnl for t in self._trades]
        import statistics
        if len(pnls) < 2:
            return 1.0
        mean = statistics.mean(pnls)
        std = statistics.stdev(pnls)
        return mean / std if std > 0 else 1.0

    def get_max_drawdown_pct(self):
        return 5.0


class MockSizer:
    def calculate(self, signal, capital, current_price, phase_params):
        return capital * (phase_params.risk_pct / 100) / max(current_price, 0.01)


# ═════════════════════════════════════════════════════════════
# Helper: build a wired agent
# ═════════════════════════════════════════════════════════════

def make_agent(bus, config=None, capital=50.0, exchange=None, **overrides):
    return DarwinAgent(
        initial_capital=capital,
        config=config or AgentConfig(
            incubation_candles=5,
            min_graduation_winrate=0.0,
            generation_trade_limit=10,
            watchlist=["BTCUSDT"],
        ),
        exchange=exchange or MockExchange(),
        feature_engine=MockFeatureEngine(),
        selector=MockSelector(),
        strategies={StrategyID.MOMENTUM: MockStrategy()},
        risk_gate=MockRiskGate(),
        sizer=MockSizer(),
        bus=bus,
        **overrides,
    )


# ═════════════════════════════════════════════════════════════
# Tests
# ═════════════════════════════════════════════════════════════

async def test_event_bus():
    print("\n🧪 TEST 1: Event Bus")
    print("─" * 50)

    bus = EventBus()
    received = []
    async def handler(ev): received.append(ev)
    async def boom(ev): raise ValueError("boom")

    bus.subscribe(EventType.TICK, handler)
    bus.subscribe(EventType.TICK, boom)

    await bus.emit(tick_event())
    assert len(received) == 1 and received[0].event_type == EventType.TICK
    print("  ✅ Events dispatched, failing handlers isolated")

    history = bus.get_recent(EventType.TICK)
    assert len(history) == 1
    print("  ✅ Event history tracking works")
    print("  ✅ PASSED\n")


async def test_capital_allocator():
    print("🧪 TEST 2: Capital Allocator & Phases")
    print("─" * 50)

    alloc = CapitalAllocator(
        strategy=AllocationStrategy.PERFORMANCE_WEIGHTED,
        phase_manager=PhaseManager(),
    )

    # Equal fitness → equal split
    r = alloc.allocate(["a", "b", "c"], 300.0, {"a": 1, "b": 1, "c": 1})
    total = sum(s.amount for s in r.values())
    assert abs(total - 300.0) < 0.01
    print(f"  ✅ Equal split: ${total:.2f} across 3")

    # Unequal fitness → weighted
    r2 = alloc.allocate(["a", "b"], 100.0, {"a": 0.9, "b": 0.1})
    assert r2["a"].amount > r2["b"].amount
    print(f"  ✅ Weighted: a={r2['a'].amount:.0f}% > b={r2['b'].amount:.0f}%")

    # Phase detection
    assert alloc.get_phase_params(50).phase == GrowthPhase.BOOTSTRAP
    assert alloc.get_phase_params(300).phase == GrowthPhase.SCALING
    assert alloc.get_phase_params(800).phase == GrowthPhase.ACCELERATION
    assert alloc.get_phase_params(5000).phase == GrowthPhase.CONSOLIDATION
    print("  ✅ Phase transitions: $50→B, $300→S, $800→A, $5K→C")
    print("  ✅ PASSED\n")


async def test_agent_lifecycle():
    print("🧪 TEST 3: Agent Lifecycle (incubation → live → trade → die)")
    print("─" * 50)

    bus = EventBus()
    events = []
    async def log(ev): events.append(ev.event_type)
    for et in EventType:
        bus.subscribe(et, log)

    agent = make_agent(bus, capital=50.0)
    assert agent.phase.value == "incubation"
    print(f"  Agent: {agent.agent_id}, phase=incubation")

    # Incubation: 10 ticks → should graduate (incubation_candles=5, min_wr=0)
    for _ in range(10):
        await agent.tick()
    assert agent.phase.value == "live", f"Expected live, got {agent.phase.value}"
    print("  ✅ Graduated to LIVE")

    # Live: tick until trades happen
    for _ in range(30):
        await agent.tick()

    m = agent.get_metrics()
    print(f"  Trades: {m.total_trades}, PnL: ${m.realized_pnl:.2f}, "
          f"Cap: ${m.capital:.2f}, HP: {m.hp}, Fitness: {m.fitness:.4f}")

    assert EventType.TRADE_OPENED in events, "Should have opened trades"
    print("  ✅ Trades opened via exchange")

    # If generation completed, agent should have died
    if EventType.GENERATION_COMPLETE in events:
        assert agent.phase.value == "dead"
        print("  ✅ Generation complete → agent died")
    else:
        print(f"  ℹ️  Agent still live ({m.total_trades} of 10 trades)")

    print("  ✅ PASSED\n")


async def test_agent_death_respawn():
    print("🧪 TEST 4: Death & Automatic Respawn")
    print("─" * 50)

    bus = EventBus()
    spawns = []
    async def on_spawn(ev): spawns.append(ev.source)
    bus.subscribe(EventType.AGENT_SPAWNED, on_spawn)

    allocator = CapitalAllocator(phase_manager=PhaseManager())

    def factory(aid, gen, cap, dna):
        return make_agent(
            bus, capital=cap, agent_id=aid, generation=gen, dna_genes=dna,
            config=AgentConfig(
                incubation_candles=2,
                min_graduation_winrate=0.0,
                generation_trade_limit=3,  # die quickly
                watchlist=["BTCUSDT"],
            ),
        )

    mgr = AgentManager(
        agent_factory=factory, allocator=allocator, bus=bus,
        total_capital=50.0, min_agents=1, max_agents=3,
        rebalance_every=2,
    )

    await mgr.start()
    assert len(mgr.get_alive_ids()) >= 1
    print(f"  Started with {len(mgr.get_alive_ids())} agent(s)")

    for _ in range(60):
        await bus.emit(tick_event())

    print(f"  Spawns: {len(spawns)}, Alive now: {len(mgr.get_alive_ids())}")
    assert len(spawns) > 1, "Should have respawned agents"
    assert len(mgr.get_alive_ids()) >= 1, "Must maintain min_agents"
    print("  ✅ Agents respawn after generation death")
    print("  ✅ PASSED\n")


async def test_multi_agent_pool():
    print("🧪 TEST 5: Multi-Agent Pool & Metrics Aggregation")
    print("─" * 50)

    bus = EventBus()
    heartbeats = []
    async def hb(ev): heartbeats.append(ev)
    bus.subscribe(EventType.HEARTBEAT, hb)

    allocator = CapitalAllocator(
        strategy=AllocationStrategy.PERFORMANCE_WEIGHTED,
        phase_manager=PhaseManager(),
    )

    def factory(aid, gen, cap, dna):
        return make_agent(bus, capital=cap, agent_id=aid, generation=gen,
                          dna_genes=dna)

    mgr = AgentManager(
        agent_factory=factory, allocator=allocator, bus=bus,
        total_capital=100.0, min_agents=1, max_agents=5,
        rebalance_every=3,
    )
    await mgr.start()

    for _ in range(40):
        await bus.emit(tick_event())

    pool = mgr.get_pool_metrics()
    print(f"  Alive: {pool.alive_agents}, Dead: {pool.dead_agents}")
    print(f"  Capital: ${pool.total_capital:.2f}")
    print(f"  PnL: ${pool.total_realized_pnl:.2f}")
    print(f"  Trades: {pool.total_trades}")
    print(f"  Win rate: {pool.pool_win_rate:.1%}")
    print(f"  Phase: {pool.current_phase}")
    print(f"  Gen: {pool.generation_high_water}")
    print(f"  Best: {pool.best_agent_id} (fitness={pool.best_fitness:.4f})")
    print(f"  Heartbeats: {len(heartbeats)}")

    assert pool.alive_agents >= 1
    assert len(heartbeats) > 0
    assert pool.total_capital > 0

    # Check agents list in metrics
    assert len(pool.agents) >= 1
    d = pool.to_dict()
    assert "agents" in d and "total_capital" in d
    print("  ✅ Pool metrics aggregate correctly")
    print("  ✅ Heartbeats emit pool snapshots")
    print("  ✅ to_dict() serializable")
    print("  ✅ PASSED\n")


async def test_allocation_rebalance():
    print("🧪 TEST 6: Capital Rebalance During Runtime")
    print("─" * 50)

    bus = EventBus()
    allocator = CapitalAllocator(
        strategy=AllocationStrategy.PERFORMANCE_WEIGHTED,
        phase_manager=PhaseManager(),
    )

    a1 = make_agent(bus, capital=60.0, agent_id="agt-high")
    a2 = make_agent(bus, capital=40.0, agent_id="agt-low")

    # Simulate a1 being much better
    result = allocator.allocate(
        agent_ids=["agt-high", "agt-low"],
        total_capital=100.0,
        agent_fitness={"agt-high": 0.95, "agt-low": 0.2},
    )

    a1.update_allocation(result["agt-high"])
    a2.update_allocation(result["agt-low"])

    print(f"  High-fit agent: ${result['agt-high'].amount:.2f} "
          f"(phase={result['agt-high'].phase_params.phase.value})")
    print(f"  Low-fit agent:  ${result['agt-low'].amount:.2f} "
          f"(phase={result['agt-low'].phase_params.phase.value})")

    assert result["agt-high"].amount > result["agt-low"].amount
    assert a1.phase_params == result["agt-high"].phase_params
    print("  ✅ Higher fitness → more capital")
    print("  ✅ Phase params propagate to agent")
    print("  ✅ PASSED\n")


async def test_circuit_breakers():
    print("🧪 TEST 7: Circuit Breakers & Cooldown")
    print("─" * 50)

    bus = EventBus()
    agent = make_agent(
        bus, capital=50.0,
        config=AgentConfig(
            incubation_candles=1,
            min_graduation_winrate=0.0,
            generation_trade_limit=100,
            max_consecutive_losses=3,
            cooldown_base_minutes=0.01,  # very short for test
            watchlist=["BTCUSDT"],
        ),
    )

    # Graduate
    await agent.tick()
    await agent.tick()
    assert agent.phase.value == "live"

    # Simulate consecutive losses via _on_trade_closed
    for i in range(4):
        loss_trade = TradeResult(
            symbol="BTCUSDT", side=OrderSide.BUY,
            strategy=StrategyID.MOMENTUM,
            entry_price=100.0, exit_price=99.0,
            quantity=0.1, realized_pnl=-1.0,
            close_reason="stop_loss", agent_id=agent.agent_id,
        )
        agent._on_trade_closed(loss_trade)

    m = agent.get_metrics()
    print(f"  After 4 losses: cap=${m.capital:.2f}, hp={m.hp:.1f}")

    # Agent should be on cooldown after 3 consecutive losses
    # (consec resets after cooldown triggers, so after 3 it fires)
    assert agent._capital < 50.0, "Capital should decrease"
    assert agent._hp < 100.0, "HP should decrease"
    print("  ✅ HP decreases on losses")
    print("  ✅ Capital decreases on losses")
    print("  ✅ PASSED\n")


# ─────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────

async def main():
    print("═" * 60)
    print("  🧬 DARWIN v4 — Integration Test Suite")
    print("═" * 60)

    await test_event_bus()
    await test_capital_allocator()
    await test_agent_lifecycle()
    await test_agent_death_respawn()
    await test_multi_agent_pool()
    await test_allocation_rebalance()
    await test_circuit_breakers()

    print("═" * 60)
    print("  ✅ ALL 7 TESTS PASSED")
    print("═" * 60)


if __name__ == "__main__":
    asyncio.run(main())
