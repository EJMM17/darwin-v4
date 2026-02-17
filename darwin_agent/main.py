"""
Darwin v4 — Application entry point.

Boot sequence:
  1. Load config (YAML + env vars)
  2. Initialize infrastructure (Redis, Postgres, EventBus)
  3. Wire components (exchanges, risk, evolution, allocator)
  4. Build agent factory
  5. Start AgentManager pool
  6. Start dashboard (optional)
  7. Run tick loop until SIGINT/SIGTERM
  8. Graceful shutdown

Usage:
    python -m darwin_agent.main                     # defaults (testnet)
    python -m darwin_agent.main --config config.yaml
    python -m darwin_agent.main --mode live          # DANGEROUS
    python -m darwin_agent.main --mode test --capital 50
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid
import threading
from datetime import datetime, timezone

# Internal
from darwin_agent.config import DarwinConfig, load_config
from darwin_agent.interfaces.enums import (
    AgentPhase, EventType, ExchangeID, GrowthPhase, StrategyID,
)
from darwin_agent.interfaces.events import Event, tick_event
from darwin_agent.interfaces.types import (
    AgentEvalData, AgentMetrics, DNAData, PhaseParams,
)
from darwin_agent.infra.event_bus import EventBus
from darwin_agent.capital.allocator import CapitalAllocator
from darwin_agent.capital.phases import PhaseManager
from darwin_agent.core.agent import AgentConfig, DarwinAgent
from darwin_agent.exchanges.router import ExchangeRouter
from darwin_agent.exchanges.binance import BinanceAdapter
from darwin_agent.evolution.engine import EvolutionEngine
from darwin_agent.evolution.diagnostics import EvolutionDiagnostics
from darwin_agent.risk.portfolio_engine import PortfolioRiskEngine, RiskLimits
from darwin_agent.orchestrator.agent_manager import AgentManager


logger = logging.getLogger("darwin.main")

BANNER = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   🧬  D A R W I N   A G E N T   v 4 . 0  🧬             ║
║                                                           ║
║   Multi-Agent Evolutionary Trading System                 ║
║   "Survive. Adapt. Evolve."                               ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
"""


def _utcnow():
    return datetime.now(timezone.utc)


def setup_logging(level: str, log_file: str | None = None):
    fmt = "%(asctime)s │ %(levelname)-5s │ %(name)-20s │ %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=handlers,
    )


# ═════════════════════════════════════════════════════════════
# Component factory
# ═════════════════════════════════════════════════════════════

def build_exchange_router(config: DarwinConfig) -> ExchangeRouter:
    """Build ExchangeRouter from config."""
    adapters = {}
    primary = ExchangeID.BINANCE

    for ex_cfg in config.exchanges:
        if not ex_cfg.enabled:
            continue
        eid = ExchangeID(ex_cfg.exchange_id)
        if eid != ExchangeID.BINANCE:
            logger.warning("ignoring non-binance exchange config: %s", eid.value)
            continue

        if not ex_cfg.api_key or not ex_cfg.api_secret:
            raise RuntimeError("Binance live credentials missing: configure API key and secret in dashboard")

        adapter = BinanceAdapter(
            api_key=ex_cfg.api_key,
            api_secret=ex_cfg.api_secret,
            testnet=bool(ex_cfg.testnet),
        )
        adapters[eid] = adapter

    if not adapters:
        raise RuntimeError("No exchange adapters enabled in config")

    return ExchangeRouter(adapters=adapters, primary=primary)


def build_risk_engine(config: DarwinConfig) -> PortfolioRiskEngine:
    """Build PortfolioRiskEngine from config."""
    limits = RiskLimits(
        defensive_drawdown_pct=config.risk.defensive_drawdown_pct,
        critical_drawdown_pct=config.risk.critical_drawdown_pct,
        halted_drawdown_pct=config.risk.halted_drawdown_pct,
        max_consecutive_losses=config.risk.max_consecutive_losses,
        max_total_exposure_pct=config.risk.max_total_exposure_pct,
    )
    engine = PortfolioRiskEngine(limits=limits)
    engine.update_equity(config.starting_capital)
    return engine


def build_allocator(config: DarwinConfig) -> CapitalAllocator:
    """Build CapitalAllocator from config."""
    phase_overrides = {
        GrowthPhase.BOOTSTRAP: {
            "max_capital": config.capital.bootstrap_target,
            "risk_pct": config.capital.bootstrap_risk_pct,
        },
        GrowthPhase.SCALING: {
            "max_capital": config.capital.scaling_target,
            "risk_pct": config.capital.scaling_risk_pct,
        },
        GrowthPhase.ACCELERATION: {
            "max_capital": config.capital.acceleration_target,
            "risk_pct": config.capital.acceleration_risk_pct,
        },
        GrowthPhase.CONSOLIDATION: {
            "risk_pct": config.capital.consolidation_risk_pct,
        },
    }

    return CapitalAllocator(
        phase_manager=PhaseManager(overrides=phase_overrides),
    )


def build_evolution(config: DarwinConfig) -> EvolutionEngine:
    """Build EvolutionEngine from config."""
    return EvolutionEngine(
        survival_rate=config.evolution.survival_rate,
        elitism_count=config.evolution.elitism_count,
        tournament_size=config.evolution.tournament_size,
        base_mutation_rate=config.evolution.mutation_rate,
        mutation_decay=config.evolution.mutation_decay,
        crossover_rate=config.evolution.crossover_rate,
    )


def build_agent_factory(
    router: ExchangeRouter,
    risk_engine: PortfolioRiskEngine,
    bus: EventBus,
    config: DarwinConfig,
):
    """
    Return a factory function that creates DarwinAgent instances.
    The AgentManager calls this to spawn new agents.
    """
    from darwin_agent.core.agent import DarwinAgent, AgentConfig
    # Minimal strategy/feature/selector/sizer stubs for now
    # In production, these would be real ML models
    from unittest.mock import MagicMock

    def factory(agent_id: str, generation: int, capital: float, dna: DNAData | None):
        features = MagicMock()
        features.extract = MagicMock(return_value=[0.5] * 10)
        selector = MagicMock()
        selector.select = MagicMock(return_value=StrategyID.MOMENTUM)
        selector.update = MagicMock()

        strategy = MagicMock()
        strategy.analyze = MagicMock(return_value=None)

        sizer = MagicMock()
        sizer.calculate = MagicMock(side_effect=lambda *a, **k: compute_position_size_from_equity(risk_engine.get_portfolio_state().total_equity, config.risk.max_position_pct))

        agent = DarwinAgent(
            agent_id=agent_id,
            generation=generation,
            initial_capital=capital,
            exchange=router,
            feature_engine=features,
            selector=selector,
            strategies={StrategyID.MOMENTUM: strategy},
            risk_gate=risk_engine,
            sizer=sizer,
            bus=bus,
        )
        return agent

    return factory


def compute_position_size_from_equity(equity: float, risk_percent: float) -> float:
    """Compute notional position fraction from real-time equity and risk percentage."""
    eq = max(float(equity), 0.0)
    rp = max(float(risk_percent), 0.0)
    if eq <= 0 or rp <= 0:
        return 0.0
    return min(1.0, rp / 100.0)

# ═════════════════════════════════════════════════════════════
# Main run loop
# ═════════════════════════════════════════════════════════════

async def run(config: DarwinConfig, stop_event: threading.Event | None = None):
    """Main application loop."""
    runtime_capital = float(config.starting_capital)
    print(BANNER)
    print(f"  Mode:     {config.mode.upper()}")
    print(f"  Capital:  ${runtime_capital:.2f}")
    print(f"  Pool:     {config.evolution.pool_size} agents")
    print(f"  Tick:     {config.infra.tick_interval}s")
    exchanges = [e.exchange_id for e in config.exchanges if e.enabled]
    print(f"  Exchange: {', '.join(exchanges)}")
    testnet = all(e.testnet for e in config.exchanges if e.enabled)
    print(f"  Testnet:  {'YES' if testnet else '⚠️  LIVE TRADING ⚠️'}")
    print()

    # Validate
    errors = config.validate()
    if errors:
        for e in errors:
            logger.error("CONFIG ERROR: %s", e)
        raise RuntimeError("Invalid configuration: " + "; ".join(errors))

    # ── Wire components ──────────────────────────────────────
    bus = EventBus()
    router = build_exchange_router(config)
    risk_engine = build_risk_engine(config)
    allocator = build_allocator(config)
    evolution = build_evolution(config)
    diagnostics = EvolutionDiagnostics()
    agent_factory = build_agent_factory(router, risk_engine, bus, config)

    # ── Connect exchanges ────────────────────────────────────
    logger.info("connecting exchanges...")
    try:
        await router.refresh_statuses()
        statuses = router.get_exchange_statuses()
        if not statuses:
            raise RuntimeError("no exchange statuses available")

        connected = {
            eid: status
            for eid, status in statuses.items()
            if status.connected
        }
        for eid, status in statuses.items():
            logger.info("  %s: %s", eid.value,
                        "CONNECTED" if status.connected else "FAILED")
        if not connected:
            raise RuntimeError("all exchange connections failed")

        if config.is_live:
            live_ex = next((ex for ex in config.exchanges if ex.enabled), None)
            if live_ex is None:
                raise RuntimeError("no enabled exchange for live balance sync")
            balance_adapter = BinanceAdapter(
                api_key=live_ex.api_key,
                api_secret=live_ex.api_secret,
                testnet=bool(live_ex.testnet),
            )
            try:
                for symbol in live_ex.symbols:
                    await balance_adapter.set_leverage(symbol, 5)
                    logger.info("live leverage override set: symbol=%s leverage=5", symbol)
                wallet_balance, unrealized_pnl = await balance_adapter.get_wallet_balance_and_upnl()
                runtime_capital = wallet_balance + unrealized_pnl
                logger.info(
                    "live equity synced from Binance: wallet=%.8f upnl=%.8f equity=%.8f",
                    wallet_balance,
                    unrealized_pnl,
                    runtime_capital,
                )
            finally:
                await balance_adapter.close()
        else:
            runtime_capital = await router.get_balance()
            logger.info("exchange connected and balance verified: %.8f USDT", runtime_capital)
        risk_engine.update_equity(runtime_capital)
    except Exception as exc:
        logger.error("exchange connection failed: %s", exc)
        logger.error("cannot proceed without active exchange connection")
        raise RuntimeError(f"Exchange connection failed: {exc}")

    # ── Build AgentManager ───────────────────────────────────
    manager = AgentManager(
        agent_factory=agent_factory,
        allocator=allocator,
        bus=bus,
        total_capital=runtime_capital,
    )
    await manager.start()
    logger.info("agent pool started: %d agents", config.evolution.pool_size)

    # ── Graceful shutdown ────────────────────────────────────
    def should_stop() -> bool:
        return bool(stop_event and stop_event.is_set())

    # ── Tick loop ────────────────────────────────────────────
    tick_count = 0
    logger.info("entering tick loop (interval=%.1fs)", config.infra.tick_interval)

    try:
        while not should_stop():
            tick_count += 1

            # Emit tick event
            await bus.emit(tick_event())

            # Periodic logging
            if tick_count % 10 == 0:
                pool = manager.get_pool_metrics()
                logger.info(
                    "tick %d │ alive=%d dead=%d │ capital=$%.2f │ "
                    "pnl=$%.2f │ gen=%d",
                    tick_count, pool.alive_agents, pool.dead_agents,
                    pool.total_capital, pool.total_realized_pnl,
                    pool.generation_high_water,
                )

                # Run diagnostics periodically
                if tick_count % 50 == 0:
                    # Collect eval data for diagnostics
                    live_agents = manager.get_live_agents()
                    if live_agents:
                        eval_data = [
                            AgentEvalData(
                                metrics=a.get_metrics(),
                                initial_capital=getattr(a, '_initial_capital', 100.0),
                                pnl_series=[t.realized_pnl for t in getattr(a, '_closed_trades', [])],
                                exposure={
                                    sym: pos.size * pos.current_price / max(a._capital, 0.01)
                                    for sym, pos in getattr(a, '_positions', {}).items()
                                },
                                dna=getattr(a, '_dna', None),
                            )
                            for a in live_agents
                        ]
                        diag_report = diagnostics.diagnose(
                            pool.generation_high_water, eval_data)
                        if diag_report.alerts:
                            for alert in diag_report.alerts:
                                logger.warning("DIAG ALERT: %s", alert)
                        logger.info(
                            "diagnostics │ health=%.2f │ diversity=%.2f │ "
                            "cap_gini=%.2f",
                            diag_report.health_score,
                            diag_report.diversity.overall_diversity_score,
                            diag_report.concentration.capital_gini,
                        )

            # Wait for next tick
            interval_remaining = float(config.infra.tick_interval)
            while interval_remaining > 0 and not should_stop():
                sleep_for = min(0.25, interval_remaining)
                await asyncio.sleep(sleep_for)
                interval_remaining -= sleep_for

    except Exception as exc:
        logger.error("fatal error in tick loop: %s", exc, exc_info=True)
    finally:
        # ── Shutdown sequence ────────────────────────────────
        logger.info("shutting down...")

        try:
            await manager.stop()
            logger.info("agent pool stopped")
        except Exception as exc:
            logger.error("error stopping pool: %s", exc)

        try:
            for adapter in getattr(router, "_adapters", {}).values():
                close_fn = getattr(adapter, "close", None)
                if close_fn is not None:
                    await close_fn()
            logger.info("exchanges disconnected")
        except Exception as exc:
            logger.error("error disconnecting exchanges: %s", exc)

        bus.clear()
        logger.info("event bus cleared")

        # Final stats
        pool = manager.get_pool_metrics()
        logger.info(
            "FINAL │ capital=$%.2f │ pnl=$%.2f │ generations=%d │ ticks=%d",
            pool.total_capital, pool.total_realized_pnl,
            pool.generation_high_water, tick_count,
        )
        print("\n  🧬 Darwin Agent shutdown complete.\n")


# ═════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Darwin Agent v4.0")
    parser.add_argument("--config", "-c", default="config.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--mode", "-m", choices=["test", "live"],
                        default=None, help="Trading mode (overrides config)")
    parser.add_argument("--capital", type=float, default=None,
                        help="Starting capital (overrides config)")
    parser.add_argument("--tick", type=float, default=None,
                        help="Tick interval in seconds")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.mode:
        config.mode = args.mode
    if args.capital:
        config.capital.starting_capital = args.capital
    if args.tick:
        config.infra.tick_interval = args.tick

    setup_logging(config.infra.log_level, config.infra.log_file)

    asyncio.run(run(config))


if __name__ == "__main__":
    main()
