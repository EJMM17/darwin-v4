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
import signal
import sys
import uuid
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
from darwin_agent.exchanges.bybit import BybitAdapter
from darwin_agent.exchanges.binance import BinanceAdapter
from darwin_agent.evolution.engine import EvolutionEngine
from darwin_agent.evolution.diagnostics import EvolutionDiagnostics
from darwin_agent.risk.portfolio_engine import PortfolioRiskEngine, RiskLimits
from darwin_agent.orchestrator.agent_manager import AgentManager


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
    primary = None

    for ex_cfg in config.exchanges:
        if not ex_cfg.enabled:
            continue
        eid = ExchangeID(ex_cfg.exchange_id)
        if eid == ExchangeID.BYBIT:
            adapter = BybitAdapter(
                api_key=ex_cfg.api_key,
                api_secret=ex_cfg.api_secret,
                testnet=ex_cfg.testnet,
            )
        elif eid == ExchangeID.BINANCE:
            adapter = BinanceAdapter(
                api_key=ex_cfg.api_key,
                api_secret=ex_cfg.api_secret,
                testnet=ex_cfg.testnet,
            )
        else:
            raise RuntimeError(f"Unsupported exchange configured: {eid.value}")
        adapters[eid] = adapter
        if primary is None:
            primary = eid

    if not adapters:
        raise RuntimeError("No exchange adapters enabled in config")

    return ExchangeRouter(adapters=adapters, primary=primary)


async def validate_exchange_symbol_sync(router: ExchangeRouter, config: DarwinConfig) -> None:
    """Validate configured symbols exist on Binance futures before trading."""
    logger = logging.getLogger("darwin.main")
    for ex_cfg in config.exchanges:
        if not ex_cfg.enabled or ex_cfg.exchange_id.lower() != ExchangeID.BINANCE.value:
            continue
        adapter = router._adapters.get(ExchangeID.BINANCE)
        if adapter is None:
            raise RuntimeError("Binance adapter missing while binance exchange is enabled")
        if not hasattr(adapter, "validate_symbols"):
            logger.warning("binance adapter does not support symbol validation")
            continue
        missing = await adapter.validate_symbols(ex_cfg.symbols)
        if missing:
            msg = (
                "Configured Binance symbols are not tradable/valid: "
                + ", ".join(missing)
            )
            if config.is_live:
                raise RuntimeError(msg)
            logger.warning(msg)
        else:
            logger.info("binance symbols validated: %s", ", ".join(ex_cfg.symbols))



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
    return CapitalAllocator(
        starting_capital=config.starting_capital,
        phase_manager=PhaseManager(
            bootstrap_target=config.capital.bootstrap_target,
            scaling_target=config.capital.scaling_target,
            acceleration_target=config.capital.acceleration_target,
        ),
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
        sizer.calculate = MagicMock(return_value=0.01)

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


# ═════════════════════════════════════════════════════════════
# Main run loop
# ═════════════════════════════════════════════════════════════

async def run(config: DarwinConfig):
    """Main application loop."""
    logger = logging.getLogger("darwin.main")

    print(BANNER)
    print(f"  Mode:     {config.mode.upper()}")
    print(f"  Capital:  ${config.starting_capital:.2f}")
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
        sys.exit(1)

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
        await router.connect_all()
        statuses = await router.get_all_statuses()
        for eid, status in statuses.items():
            logger.info("  %s: %s", eid.value,
                        "CONNECTED" if status.connected else "FAILED")

        await validate_exchange_symbol_sync(router, config)
    except Exception as exc:
        logger.error("exchange connection failed: %s", exc)
        if config.is_live:
            logger.error("cannot proceed in live mode without exchanges")
            sys.exit(1)
        logger.warning("continuing in test mode without exchange connection")

    # ── Build AgentManager ───────────────────────────────────
    manager = AgentManager(
        agent_factory=agent_factory,
        allocator=allocator,
        bus=bus,
        total_capital=config.starting_capital,
    )
    await manager.start()
    logger.info("agent pool started: %d agents", config.evolution.pool_size)

    # ── Graceful shutdown ────────────────────────────────────
    shutdown_event = asyncio.Event()

    def handle_signal(sig, frame):
        logger.info("received signal %s, shutting down...", sig)
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # ── Tick loop ────────────────────────────────────────────
    tick_count = 0
    logger.info("entering tick loop (interval=%.1fs)", config.infra.tick_interval)

    try:
        while not shutdown_event.is_set():
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
            try:
                await asyncio.wait_for(
                    shutdown_event.wait(),
                    timeout=config.infra.tick_interval,
                )
            except asyncio.TimeoutError:
                pass  # normal — just means tick interval elapsed

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
            await router.disconnect_all()
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
