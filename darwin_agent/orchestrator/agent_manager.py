"""
Darwin v4 — Agent Manager — Production Hardened.

Layer 7 (orchestrator). Manages agent lifecycle, capital, metrics.

AUDIT FINDINGS:
  H-AM1  _on_tick has NO guard against overlapping ticks. If tick N takes
         longer than heartbeat interval, tick N+1 fires concurrently.
         FIX: _tick_lock. Second tick skips if first is still running.

  H-AM2  _dead_history is List trimmed with list[-200:] — O(n) copy.
         FIX: deque(maxlen=200).

  H-AM3  No persistence. Restart loses _total_capital, _generation_counter.
         FIX: save_checkpoint / load_checkpoint via injected cache.

  H-AM4  _calc_spawn_capital can return 0 or negative.
         FIX: Floor at 0.01, log invariant violations.

  H-AM5  stop() hangs if exchange calls hang during kill.
         FIX: asyncio.wait_for with timeout on each kill.

  H-AM6  _ensure_minimum loops forever if factory throws.
         FIX: Cap spawn attempts per tick.
"""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Deque, Dict, List, Optional

from darwin_agent.interfaces.enums import AgentPhase, EventType
from darwin_agent.interfaces.events import Event, agent_spawned_event, heartbeat_event
from darwin_agent.interfaces.types import AgentMetrics

from darwin_agent.core.agent import DarwinAgent

logger = logging.getLogger("darwin.manager")

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

AgentFactory = Callable[
    [str, int, float, Dict],
    DarwinAgent,
]

_MAX_SPAWN_ATTEMPTS_PER_TICK = 3
_KILL_TIMEOUT_SECONDS = 10.0


@dataclass(slots=True)
class PoolMetrics:
    total_agents: int = 0
    alive_agents: int = 0
    dead_agents: int = 0
    total_capital: float = 0.0
    total_realized_pnl: float = 0.0
    total_unrealized_pnl: float = 0.0
    total_trades: int = 0
    pool_win_rate: float = 0.0
    pool_sharpe: float = 0.0
    pool_max_drawdown: float = 0.0
    best_fitness: float = 0.0
    best_agent_id: str = ""
    current_phase: str = "bootstrap"
    generation_high_water: int = 0
    agents: List[AgentMetrics] = field(default_factory=list)
    timestamp: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict:
        return {
            "total_agents": self.total_agents,
            "alive_agents": self.alive_agents,
            "dead_agents": self.dead_agents,
            "total_capital": round(self.total_capital, 2),
            "total_realized_pnl": round(self.total_realized_pnl, 2),
            "total_unrealized_pnl": round(self.total_unrealized_pnl, 2),
            "total_trades": self.total_trades,
            "pool_win_rate": round(self.pool_win_rate, 2),
            "pool_sharpe": round(self.pool_sharpe, 2),
            "pool_max_drawdown": round(self.pool_max_drawdown, 2),
            "best_fitness": round(self.best_fitness, 4),
            "best_agent_id": self.best_agent_id,
            "current_phase": self.current_phase,
            "generation_high_water": self.generation_high_water,
            "agents": [
                {
                    "agent_id": a.agent_id, "generation": a.generation,
                    "phase": a.phase, "capital": round(a.capital, 2),
                    "realized_pnl": round(a.realized_pnl, 2),
                    "win_rate": round(a.win_rate * 100, 1),
                    "fitness": round(a.fitness, 4),
                    "hp": round(a.hp, 1),
                    "total_trades": a.total_trades,
                    "open_positions": a.open_positions,
                }
                for a in self.agents
            ],
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(slots=True)
class PoolCheckpoint:
    """Minimal state needed to survive a restart."""
    total_capital: float
    generation_counter: int
    tick_counter: int
    timestamp: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict:
        return {
            "total_capital": self.total_capital,
            "generation_counter": self.generation_counter,
            "tick_counter": self.tick_counter,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PoolCheckpoint":
        return cls(
            total_capital=d["total_capital"],
            generation_counter=d["generation_counter"],
            tick_counter=d.get("tick_counter", 0),
        )


class AgentManager:
    __slots__ = (
        "_agents", "_dead_history", "_factory", "_allocator", "_bus",
        "_max_agents", "_min_agents", "_total_capital",
        "_generation_counter", "_kill_fitness", "_spawn_fitness",
        "_started", "_rebalance_every", "_tick_counter",
        "_tick_lock", "_cache", "_initial_capital",
    )

    def __init__(
        self,
        *,
        agent_factory: AgentFactory,
        allocator,
        bus,
        total_capital: float = 50.0,
        max_agents: int = 5,
        min_agents: int = 1,
        kill_fitness: float = 0.0,
        spawn_fitness: float = 0.6,
        rebalance_every: int = 5,
        cache=None,
    ) -> None:
        self._agents: Dict[str, DarwinAgent] = {}
        self._dead_history: Deque[AgentMetrics] = deque(maxlen=200)
        self._factory = agent_factory
        self._allocator = allocator
        self._bus = bus
        self._total_capital = total_capital
        self._initial_capital = total_capital
        self._max_agents = max_agents
        self._min_agents = min_agents
        self._generation_counter = 0
        self._kill_fitness = kill_fitness
        self._spawn_fitness = spawn_fitness
        self._started = False
        self._rebalance_every = rebalance_every
        self._tick_counter = 0
        self._tick_lock = asyncio.Lock()
        self._cache = cache

    # ═══════════════════════════════════════════════════════
    # Lifecycle
    # ═══════════════════════════════════════════════════════

    async def start(self) -> None:
        self._bus.subscribe(EventType.TICK, self._on_tick)
        self._bus.subscribe(EventType.AGENT_DIED, self._on_agent_died)
        self._bus.subscribe(EventType.SPAWN_AGENT_REQUESTED, self._on_spawn_requested)
        self._bus.subscribe(EventType.GENERATION_COMPLETE, self._on_generation_complete)

        # H-AM3: restore from checkpoint
        await self._load_checkpoint()

        await self._spawn_agent(capital=self._total_capital)
        self._started = True
        logger.info(
            "AgentManager started — %d agent(s), $%.2f pool",
            len(self._agents), self._total_capital,
        )

    async def stop(self) -> None:
        logger.info("AgentManager stopping — killing %d agents", len(self._agents))
        for agent in list(self._agents.values()):
            if agent.is_alive:
                try:
                    # H-AM5: timeout prevents infinite hang
                    await asyncio.wait_for(
                        agent.kill("manager_shutdown"),
                        timeout=_KILL_TIMEOUT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "agent %s kill timed out after %.0fs",
                        agent.agent_id, _KILL_TIMEOUT_SECONDS,
                    )
                except Exception as exc:
                    logger.error("agent %s kill failed: %s", agent.agent_id, exc)

        await self._save_checkpoint()
        self._agents.clear()
        self._started = False

    # ═══════════════════════════════════════════════════════
    # Event handlers
    # ═══════════════════════════════════════════════════════

    async def _on_tick(self, event: Event) -> None:
        # H-AM1: skip if previous tick still running
        if self._tick_lock.locked():
            logger.debug("tick skipped — previous tick still running")
            return

        async with self._tick_lock:
            self._tick_counter += 1

            alive = [a for a in self._agents.values() if a.is_alive]
            if alive:
                results = await asyncio.gather(
                    *(a.tick() for a in alive),
                    return_exceptions=True,
                )
                for i, r in enumerate(results):
                    if isinstance(r, Exception):
                        logger.error(
                            "agent %s tick raised: %s",
                            alive[i].agent_id, r, exc_info=r,
                        )

            self._prune_dead()

            if self._tick_counter % self._rebalance_every == 0:
                self._rebalance()

            await self._ensure_minimum()
            await self._cull_unfit()

            pool = self.get_pool_metrics()
            await self._bus.emit(heartbeat_event(pool.to_dict()))

            # H-AM3: periodic checkpoint
            if self._tick_counter % 10 == 0:
                await self._save_checkpoint()

    async def _on_agent_died(self, event: Event) -> None:
        agent_id = event.source
        agent = self._agents.get(agent_id)
        if agent:
            self._dead_history.append(agent.get_metrics())
        logger.info(
            "agent died: %s cause=%s gen=%d pnl=$%.2f",
            agent_id,
            event.payload.get("cause", "?"),
            event.payload.get("generation", 0),
            event.payload.get("realized_pnl", 0),
        )

    async def _on_spawn_requested(self, event: Event) -> None:
        if len(self._alive_ids) >= self._max_agents:
            logger.debug("spawn rejected — pool at max (%d)", self._max_agents)
            return
        dna = event.payload.get("dna", {})
        capital = self._calc_spawn_capital()
        await self._spawn_agent(capital=capital, dna_genes=dna)

    async def _on_generation_complete(self, event: Event) -> None:
        gen = event.payload.get("generation", 0)
        self._generation_counter = max(self._generation_counter, gen + 1)

    # ═══════════════════════════════════════════════════════
    # Spawn / Kill / Prune
    # ═══════════════════════════════════════════════════════

    async def _spawn_agent(
        self,
        capital: float,
        dna_genes: Dict | None = None,
    ) -> Optional[DarwinAgent]:
        # H-AM4: capital floor
        capital = max(0.01, capital)
        gen = self._generation_counter
        agent_id = f"agt-{gen:03d}-{_utcnow().strftime('%H%M%S')}"

        try:
            agent = self._factory(agent_id, gen, capital, dna_genes or {})
        except Exception as exc:
            logger.error("agent factory failed for %s: %s", agent_id, exc)
            return None

        self._agents[agent_id] = agent
        await self._bus.emit(agent_spawned_event(agent_id, gen, capital))
        logger.info("spawned %s gen=%d capital=$%.2f", agent_id, gen, capital)
        return agent

    def _prune_dead(self) -> None:
        dead = [aid for aid, a in self._agents.items() if a.phase == AgentPhase.DEAD]
        for aid in dead:
            agent = self._agents.pop(aid)
            reclaimed = max(0.0, agent.capital)
            self._total_capital += reclaimed
            logger.debug("pruned %s, reclaimed $%.2f", aid, reclaimed)

        # H-AM4: capital conservation invariant check
        alive_capital = sum(a.capital for a in self._agents.values() if a.is_alive)
        total = alive_capital + self._total_capital
        if total < self._initial_capital * 0.01:
            logger.warning(
                "CAPITAL INVARIANT: pool=$%.2f (alive=$%.2f + free=$%.2f) < 1%% of initial=$%.2f",
                total, alive_capital, self._total_capital, self._initial_capital,
            )

    async def _ensure_minimum(self) -> None:
        # H-AM6: cap spawn attempts to prevent infinite loop
        attempts = 0
        while len(self._alive_ids) < self._min_agents:
            if attempts >= _MAX_SPAWN_ATTEMPTS_PER_TICK:
                logger.error(
                    "failed to meet min_agents=%d after %d spawn attempts",
                    self._min_agents, attempts,
                )
                break
            capital = self._calc_spawn_capital()
            result = await self._spawn_agent(capital=capital)
            if result is None:
                break  # factory failed, don't retry
            attempts += 1

    async def _cull_unfit(self) -> None:
        alive = self._alive_ids
        if len(alive) <= self._min_agents:
            return
        for aid in alive:
            agent = self._agents.get(aid)
            if not agent:
                continue
            m = agent.get_metrics()
            if m.total_trades >= 10 and m.fitness <= self._kill_fitness:
                logger.info("culling unfit agent %s (fitness=%.4f)", aid, m.fitness)
                try:
                    await asyncio.wait_for(
                        agent.kill("fitness_too_low"),
                        timeout=_KILL_TIMEOUT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    logger.warning("cull kill timed out for %s", aid)
                except Exception as exc:
                    logger.error("cull kill failed for %s: %s", aid, exc)
                if len(self._alive_ids) <= self._min_agents:
                    break

    # ═══════════════════════════════════════════════════════
    # Capital management
    # ═══════════════════════════════════════════════════════

    def _rebalance(self) -> None:
        alive = {aid: a for aid, a in self._agents.items() if a.is_alive}
        if not alive:
            return

        fitness_map = {aid: a.get_metrics().fitness for aid, a in alive.items()}
        pool_capital = self._total_capital + sum(a.capital for a in alive.values())

        allocations = self._allocator.allocate(
            agent_ids=list(alive.keys()),
            total_capital=pool_capital,
            agent_fitness=fitness_map,
        )

        for aid, alloc in allocations.items():
            agent = alive.get(aid)
            if agent:
                agent.update_allocation(alloc)

    def _calc_spawn_capital(self) -> float:
        alive = {aid: a for aid, a in self._agents.items() if a.is_alive}
        if not alive:
            return max(0.01, self._total_capital)  # H-AM4: floor

        in_use = sum(a.capital for a in alive.values())
        free = max(0.0, self._total_capital)

        if free > 0:
            return max(0.01, min(free, self._total_capital / (len(alive) + 1)))
        return max(0.01, (in_use + self._total_capital) / (len(alive) + 1))

    # ═══════════════════════════════════════════════════════
    # Checkpoint persistence (H-AM3)
    # ═══════════════════════════════════════════════════════

    async def _save_checkpoint(self) -> None:
        if self._cache is None:
            return
        try:
            cp = PoolCheckpoint(
                total_capital=self._total_capital,
                generation_counter=self._generation_counter,
                tick_counter=self._tick_counter,
            )
            await self._cache.set_json("pool:checkpoint", cp.to_dict(), ttl_seconds=86400)
        except Exception as exc:
            logger.warning("checkpoint save failed: %s", exc)

    async def _load_checkpoint(self) -> None:
        if self._cache is None:
            return
        try:
            data = await self._cache.get_json("pool:checkpoint")
            if data is None:
                logger.info("no checkpoint found, starting fresh")
                return
            cp = PoolCheckpoint.from_dict(data)
            # Only restore if checkpoint capital > current (don't go backwards)
            if cp.total_capital > self._total_capital:
                logger.info(
                    "restoring checkpoint: capital $%.2f → $%.2f, gen %d → %d",
                    self._total_capital, cp.total_capital,
                    self._generation_counter, cp.generation_counter,
                )
                self._total_capital = cp.total_capital
                self._initial_capital = max(self._initial_capital, cp.total_capital)
            self._generation_counter = max(
                self._generation_counter, cp.generation_counter,
            )
            self._tick_counter = cp.tick_counter
        except Exception as exc:
            logger.warning("checkpoint load failed, starting fresh: %s", exc)

    # ═══════════════════════════════════════════════════════
    # Queries
    # ═══════════════════════════════════════════════════════

    @property
    def _alive_ids(self) -> List[str]:
        return [aid for aid, a in self._agents.items() if a.is_alive]

    def get_agent(self, agent_id: str) -> Optional[DarwinAgent]:
        return self._agents.get(agent_id)

    def get_pool_metrics(self) -> PoolMetrics:
        all_m = [a.get_metrics() for a in self._agents.values()]
        alive_m = [m for m in all_m if m.is_alive]
        dead_count = len(all_m) - len(alive_m) + len(self._dead_history)

        total_cap = sum(m.capital for m in alive_m)
        total_rpnl = sum(m.realized_pnl for m in alive_m)
        total_upnl = sum(m.unrealized_pnl for m in alive_m)
        total_trades = sum(m.total_trades for m in alive_m)
        total_wins = sum(m.winning_trades for m in alive_m)

        wr = total_wins / max(total_trades, 1)

        sharpe_num = sum(m.sharpe_ratio * m.total_trades for m in alive_m)
        sharpe = sharpe_num / max(total_trades, 1)

        max_dd = max((m.max_drawdown_pct for m in alive_m), default=0.0)
        best = max(alive_m, key=lambda m: m.fitness) if alive_m else None

        phase_params = self._allocator.get_phase_params(total_cap)

        return PoolMetrics(
            total_agents=len(all_m),
            alive_agents=len(alive_m),
            dead_agents=dead_count,
            total_capital=total_cap + self._total_capital,
            total_realized_pnl=total_rpnl,
            total_unrealized_pnl=total_upnl,
            total_trades=total_trades,
            pool_win_rate=wr,
            pool_sharpe=sharpe,
            pool_max_drawdown=max_dd,
            best_fitness=best.fitness if best else 0.0,
            best_agent_id=best.agent_id if best else "",
            current_phase=phase_params.phase.value,
            generation_high_water=self._generation_counter,
            agents=all_m,
        )

    def get_alive_ids(self) -> List[str]:
        return list(self._alive_ids)

    def get_all_agents(self) -> Dict[str, DarwinAgent]:
        return dict(self._agents)
