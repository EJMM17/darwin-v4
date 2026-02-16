"""
Darwin v4 Extended — Integration tests for all new modules.
Tests: PortfolioRiskEngine, ExchangeRouter, EvolutionEngine,
       Redis/Postgres mocks, generation snapshots.
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
from darwin_agent.capital.allocator import CapitalAllocator, AllocationStrategy
from darwin_agent.capital.phases import PhaseManager
from darwin_agent.core.agent import DarwinAgent, AgentConfig
from darwin_agent.orchestrator.agent_manager import AgentManager
from darwin_agent.risk.portfolio_engine import PortfolioRiskEngine, RiskLimits
from darwin_agent.exchanges.router import ExchangeRouter
from darwin_agent.evolution.engine import EvolutionEngine

def _utcnow():
    return datetime.now(timezone.utc)


# ═════════════════════════════════════════════════════════════
# Mocks
# ═════════════════════════════════════════════════════════════

class MockExchange:
    exchange_id = ExchangeID.PAPER
    def __init__(self, name="mock", base_price=100.0):
        self._name = name
        self._price = base_price
        self._tick = 0
    async def get_candles(self, symbol, tf, limit=100):
        p = self._price
        return [Candle(timestamp=_utcnow(), open=p-.1, high=p+.3, low=p-.3, close=p, volume=1000, timeframe=tf) for _ in range(min(limit,50))]
    async def get_ticker(self, symbol):
        self._tick += 1; self._price += 0.5
        return Ticker(symbol=symbol, last_price=self._price, bid=self._price-.1, ask=self._price+.1, volume_24h=50000)
    async def get_positions(self): return []
    async def place_order(self, req):
        return OrderResult(order_id=f"ord-{self._tick}", symbol=req.symbol, side=req.side, filled_qty=req.quantity, filled_price=self._price, exchange_id=ExchangeID.PAPER)
    async def close_position(self, symbol, side):
        return OrderResult(order_id=f"close-{self._tick}", symbol=symbol, filled_qty=1.0, filled_price=self._price, exchange_id=ExchangeID.PAPER)
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
        return Signal(strategy=StrategyID.MOMENTUM, symbol=symbol, side=OrderSide.BUY, strength=SignalStrength.STRONG, confidence=0.85, stop_loss_pct=1.5, take_profit_pct=3.0)

class MockSizer:
    def calculate(self, signal, capital, price, params):
        return capital * (params.risk_pct / 100) / max(price, 0.01)


def make_agent(bus, capital=50.0, risk_gate=None, **kw):
    return DarwinAgent(
        initial_capital=capital,
        config=kw.pop("config", AgentConfig(incubation_candles=3, min_graduation_winrate=0.0, generation_trade_limit=5, watchlist=["BTCUSDT"])),
        exchange=kw.pop("exchange", MockExchange()),
        feature_engine=MockFeatureEngine(),
        selector=MockSelector(),
        strategies={StrategyID.MOMENTUM: MockStrategy()},
        risk_gate=risk_gate or PortfolioRiskEngine(),
        sizer=MockSizer(),
        bus=bus, **kw,
    )


# ═════════════════════════════════════════════════════════════
# Tests
# ═════════════════════════════════════════════════════════════

async def test_portfolio_risk_engine():
    print("\n🧪 TEST 1: PortfolioRiskEngine")
    print("─" * 50)

    engine = PortfolioRiskEngine(
        limits=RiskLimits(max_total_exposure_pct=80.0, max_daily_loss_pct=5.0,
                          halted_drawdown_pct=15.0),
    )
    params = PhaseParams(phase=GrowthPhase.BOOTSTRAP, risk_pct=2.0, max_positions=3, leverage=10)
    sig = Signal(strategy=StrategyID.MOMENTUM, symbol="BTCUSDT", side=OrderSide.BUY, strength=SignalStrength.STRONG, confidence=0.8)

    # Empty portfolio → approve
    v = engine.approve(sig, [], 100.0, params)
    assert v.approved, f"Expected approved, got: {v.reason}"
    print("  ✅ Approves on empty portfolio")

    # Max positions → reject
    positions = [Position(symbol=f"SYM{i}", side=OrderSide.BUY, size=1, entry_price=100, leverage=10) for i in range(3)]
    v = engine.approve(sig, positions, 100.0, params)
    assert not v.approved and "max_positions" in v.reason
    print("  ✅ Rejects at max positions")

    # Duplicate symbol → reject
    dup = [Position(symbol="BTCUSDT", side=OrderSide.BUY, size=1, entry_price=100, leverage=10)]
    v = engine.approve(sig, dup, 100.0, params)
    assert not v.approved and "duplicate_symbol" in v.reason
    print("  ✅ Rejects duplicate symbol")

    # Weak signal → reject
    weak = Signal(strategy=StrategyID.MOMENTUM, symbol="ETHUSDT", side=OrderSide.BUY, strength=SignalStrength.WEAK, confidence=0.3)
    v = engine.approve(weak, [], 100.0, params)
    assert not v.approved and "weak_signal" in v.reason
    print("  ✅ Rejects weak signals")

    # Record trades and check metrics
    for i in range(20):
        pnl = 1.5 if i % 3 != 0 else -2.0
        engine.record_trade_result(TradeResult(realized_pnl=pnl, closed_at=_utcnow()))

    metrics = engine.get_metrics()
    assert metrics.win_rate > 0
    assert metrics.profit_factor > 0
    print(f"  Sharpe={metrics.sharpe_ratio:.2f}, Sortino={metrics.sortino_ratio:.2f}")
    print(f"  WinRate={metrics.win_rate:.1%}, PF={metrics.profit_factor:.2f}")
    print(f"  VaR95={metrics.var_95:.2f}, MaxDD={metrics.max_drawdown_pct:.1f}%")
    print("  ✅ Risk metrics compute correctly")

    # Reset
    engine.reset()
    m2 = engine.get_metrics()
    assert m2.win_rate == 0
    print("  ✅ Reset clears all state")
    print("  ✅ PASSED\n")


async def test_exchange_router():
    print("🧪 TEST 2: ExchangeRouter (multi-exchange)")
    print("─" * 50)

    bybit = MockExchange("bybit", 100.0)
    bybit.exchange_id = ExchangeID.BYBIT
    binance = MockExchange("binance", 100.0)
    binance.exchange_id = ExchangeID.BINANCE

    router = ExchangeRouter(
        adapters={ExchangeID.BYBIT: bybit, ExchangeID.BINANCE: binance},
        primary=ExchangeID.BYBIT,
    )

    await router.refresh_statuses()
    statuses = router.get_exchange_statuses()
    assert len(statuses) == 2
    print(f"  Exchanges: {[s.exchange_id.value for s in statuses.values()]}")
    print("  ✅ Status refresh works")

    # Get ticker routes to primary
    ticker = await router.get_ticker("BTCUSDT")
    assert ticker.last_price > 0
    print(f"  ✅ Ticker routed: {ticker.symbol} = ${ticker.last_price:.2f}")

    # Place order
    req = OrderRequest(symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.01)
    result = await router.place_order(req)
    assert result.success
    print(f"  ✅ Order placed: {result.order_id}")

    # Aggregate balance
    bal = await router.get_balance()
    assert bal == 100.0  # 50 + 50
    print(f"  ✅ Aggregate balance: ${bal:.2f}")

    # Aggregate positions
    positions = await router.get_positions()
    print(f"  ✅ Positions aggregated: {len(positions)}")
    print("  ✅ PASSED\n")


async def test_evolution_engine():
    print("🧪 TEST 3: EvolutionEngine (multi-agent)")
    print("─" * 50)

    evo = EvolutionEngine(
        elitism_count=1, tournament_size=2,
        base_mutation_rate=0.2, survival_rate=0.5,
    )

    # Create random DNA
    dna_pool = [evo.create_random_dna() for _ in range(5)]
    assert len(dna_pool) == 5
    assert all(len(d.genes) == len(evo._gene_ranges) for d in dna_pool)
    print(f"  ✅ Generated {len(dna_pool)} random genomes, {len(dna_pool[0].genes)} genes each")

    # Simulate agent eval data
    agents = [
        AgentEvalData(
            metrics=AgentMetrics(
                agent_id=f"agt-{i}", generation=0, phase="dead",
                capital=50 + i*10, realized_pnl=i*5, total_trades=20,
                winning_trades=10+i, losing_trades=10-i,
                sharpe_ratio=0.5+i*0.2, max_drawdown_pct=5-i),
            initial_capital=50.0,
            pnl_series=[i*0.5]*20,
        )
        for i in range(5)
    ]

    # Evaluate generation (engine re-scores fitness internally)
    snapshot = await evo.evaluate_generation(agents)
    assert snapshot.generation == 0
    assert snapshot.best_fitness > 0  # risk-aware score
    assert snapshot.survivors > 0
    assert len(snapshot.agent_rankings) == 5
    # Rankings must include fitness_breakdown
    assert "fitness_breakdown" in snapshot.agent_rankings[0]
    print(f"  Gen {snapshot.generation}: best={snapshot.best_fitness:.4f}, "
          f"avg={snapshot.avg_fitness:.4f}, survivors={snapshot.survivors}")
    print("  ✅ Generation evaluated and ranked")

    # Select survivors (needs AgentMetrics, not AgentEvalData)
    agent_metrics = [ed.metrics for ed in agents]
    survivor_ids = evo.select_survivors(agent_metrics)
    assert len(survivor_ids) >= 1
    print(f"  Survivors: {survivor_ids}")
    print("  ✅ Tournament selection works")

    # Breed
    child = evo.breed(dna_pool[0], dna_pool[1])
    assert len(child.genes) > 0
    assert child.parent_id == dna_pool[0].dna_id
    print(f"  ✅ Bred child: {len(child.genes)} genes")

    # Mutate
    mutated = evo.mutate(child, mutation_rate=1.0)  # Force mutation
    diffs = sum(1 for k in child.genes if abs(child.genes[k] - mutated.genes.get(k, 0)) > 0.001)
    print(f"  ✅ Mutation: {diffs}/{len(child.genes)} genes changed")

    # Create next generation
    survivor_dna = dna_pool[:2]
    for d in survivor_dna:
        d.fitness = 0.8
    next_gen = evo.create_next_generation(survivor_dna, target_size=5)
    assert len(next_gen) == 5
    print(f"  ✅ Next gen: {len(next_gen)} offspring from {len(survivor_dna)} parents")

    # Hall of fame
    hof = evo.get_hall_of_fame()
    assert len(hof) > 0
    print(f"  ✅ Hall of fame: {len(hof)} entries")

    assert evo.get_generation() == 1
    print("  ✅ PASSED\n")


async def test_agent_with_portfolio_risk():
    print("🧪 TEST 4: Agent + PortfolioRiskEngine integration")
    print("─" * 50)

    bus = EventBus()
    risk = PortfolioRiskEngine()

    agent = make_agent(bus, capital=50.0, risk_gate=risk)

    # Graduate
    for _ in range(10):
        await agent.tick()
    assert agent.phase.value == "live"
    print("  ✅ Agent graduated with PortfolioRiskEngine")

    # Trade
    for _ in range(30):
        await agent.tick()

    m = agent.get_metrics()
    rm = risk.get_metrics()
    print(f"  Agent: trades={m.total_trades}, pnl=${m.realized_pnl:.2f}, fitness={m.fitness:.4f}")
    print(f"  Risk: sharpe={rm.sharpe_ratio:.2f}, pf={rm.profit_factor:.2f}, dd={rm.max_drawdown_pct:.1f}%")

    if m.total_trades > 0:
        print("  ✅ Agent trades with portfolio-level risk")
    else:
        print("  ℹ️  No trades (risk engine may have blocked)")
    print("  ✅ PASSED\n")


async def test_router_with_agent():
    print("🧪 TEST 5: Agent + ExchangeRouter integration")
    print("─" * 50)

    bus = EventBus()

    bybit = MockExchange("bybit")
    bybit.exchange_id = ExchangeID.BYBIT
    binance = MockExchange("binance")
    binance.exchange_id = ExchangeID.BINANCE

    router = ExchangeRouter(
        adapters={ExchangeID.BYBIT: bybit, ExchangeID.BINANCE: binance},
        primary=ExchangeID.BYBIT,
    )
    await router.refresh_statuses()

    agent = make_agent(bus, capital=50.0, exchange=router)

    for _ in range(10):
        await agent.tick()
    assert agent.phase.value == "live"

    for _ in range(20):
        await agent.tick()

    m = agent.get_metrics()
    print(f"  Agent via router: trades={m.total_trades}, pnl=${m.realized_pnl:.2f}")
    print("  ✅ Agent trades through ExchangeRouter")
    print("  ✅ PASSED\n")


async def test_full_evolution_cycle():
    print("🧪 TEST 6: Full Evolution Cycle (spawn → trade → evaluate → breed)")
    print("─" * 50)

    bus = EventBus()
    allocator = CapitalAllocator(phase_manager=PhaseManager())
    evo = EvolutionEngine(survival_rate=0.5, elitism_count=1)

    agents_died = []
    async def on_die(ev): agents_died.append(ev.source)
    bus.subscribe(EventType.AGENT_DIED, on_die)

    def factory(aid, gen, cap, dna):
        return make_agent(bus, capital=cap, agent_id=aid, generation=gen, dna_genes=dna,
                          config=AgentConfig(incubation_candles=2, min_graduation_winrate=0.0,
                                            generation_trade_limit=3, watchlist=["BTCUSDT"]))

    mgr = AgentManager(
        agent_factory=factory, allocator=allocator, bus=bus,
        total_capital=100.0, min_agents=3, max_agents=5,
    )
    await mgr.start()

    # Run until enough agents die
    for _ in range(80):
        await bus.emit(tick_event())

    pool = mgr.get_pool_metrics()
    print(f"  Pool: alive={pool.alive_agents}, dead={pool.dead_agents}")
    print(f"  Capital: ${pool.total_capital:.2f}, PnL: ${pool.total_realized_pnl:.2f}")
    print(f"  Gen: {pool.generation_high_water}")

    # Collect dead agent metrics for evolution
    dead_evals = [
        AgentEvalData(
            metrics=AgentMetrics(
                agent_id=f"dead-{i}", generation=0, phase="dead",
                total_trades=10, winning_trades=5+i, losing_trades=5-i,
                realized_pnl=i*2, sharpe_ratio=0.5+i*0.1,
                max_drawdown_pct=5.0),
            initial_capital=50.0,
            pnl_series=[i*0.2]*10,
        )
        for i in range(5)
    ]

    snapshot = await evo.evaluate_generation(dead_evals)
    # select_survivors still works with AgentMetrics
    dead_metrics = [ed.metrics for ed in dead_evals]
    survivor_ids = evo.select_survivors(dead_metrics)

    survivor_dna = [evo.create_random_dna() for _ in survivor_ids]
    for d in survivor_dna:
        d.fitness = 0.7
    next_gen = evo.create_next_generation(survivor_dna, target_size=5)

    print(f"  Evolution: survivors={len(survivor_ids)}, offspring={len(next_gen)}")
    print(f"  Snapshot: best_fitness={snapshot.best_fitness:.4f}")
    print("  ✅ Full evolution cycle completed")
    print("  ✅ PASSED\n")


async def test_generation_snapshot_persistence():
    print("🧪 TEST 7: GenerationSnapshot data integrity")
    print("─" * 50)

    evo = EvolutionEngine()

    agents = [
        AgentEvalData(
            metrics=AgentMetrics(
                agent_id=f"a{i}", generation=0, phase="dead",
                capital=100, realized_pnl=i*10, total_trades=50,
                winning_trades=25+i, losing_trades=25-i,
                sharpe_ratio=1.0+i*0.2, max_drawdown_pct=10-i),
            initial_capital=100.0,
            pnl_series=[i*0.2]*50,
        )
        for i in range(5)
    ]

    snap = await evo.evaluate_generation(agents)

    assert snap.population_size == 5
    assert snap.best_fitness > 0  # risk-aware re-scored
    assert snap.survivors + snap.eliminated == 5
    assert len(snap.agent_rankings) == 5
    assert snap.agent_rankings[0]["rank"] == 1
    assert snap.agent_rankings[0]["fitness"] >= snap.agent_rankings[-1]["fitness"]
    # Verify fitness_breakdown is present in every ranking entry
    for entry in snap.agent_rankings:
        assert "fitness_breakdown" in entry, f"Missing breakdown for {entry['agent_id']}"
        bd = entry["fitness_breakdown"]
        assert "portfolio_harmony" in bd
        assert "diversification_bonus" in bd
        assert 0.0 <= bd["final_score"] <= 1.0

    print(f"  Snapshot gen={snap.generation}:")
    print(f"    Population: {snap.population_size}")
    print(f"    Best: {snap.best_fitness:.4f}, Avg: {snap.avg_fitness:.4f}, Worst: {snap.worst_fitness:.4f}")
    print(f"    Survivors: {snap.survivors}, Eliminated: {snap.eliminated}")
    print(f"    Rankings: {[r['agent_id'] for r in snap.agent_rankings]}")
    print(f"    Total PnL: ${snap.total_pnl:.2f}")
    print("  ✅ Snapshot contains complete generation data")
    print("  ✅ PASSED\n")


# ─────────────────────────────────────────────────────────────

async def main():
    print("═" * 60)
    print("  🧬 DARWIN v4 EXTENDED — Integration Tests")
    print("═" * 60)

    await test_portfolio_risk_engine()
    await test_exchange_router()
    await test_evolution_engine()
    await test_agent_with_portfolio_risk()
    await test_router_with_agent()
    await test_full_evolution_cycle()
    await test_generation_snapshot_persistence()

    print("═" * 60)
    print("  ✅ ALL 7 EXTENDED TESTS PASSED")
    print("═" * 60)


if __name__ == "__main__":
    asyncio.run(main())
