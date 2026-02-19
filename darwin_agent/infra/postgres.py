"""
Darwin v4 — PostgreSQL persistence layer.

Layer 1 (infra). Implements ITradeRepository, IAgentRepository,
IGenerationRepository, IDNARepository.
Uses asyncpg for production async Postgres access.
Repository pattern: domain objects in, domain objects out.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import List, Optional

from darwin_agent.interfaces.enums import ExchangeID, OrderSide, StrategyID
from darwin_agent.interfaces.types import (
    AgentMetrics, DNAData, GenerationSnapshot, TradeResult,
)

logger = logging.getLogger("darwin.postgres")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── SQL Schema ───────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    trade_id        TEXT PRIMARY KEY,
    agent_id        TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,
    strategy        TEXT NOT NULL,
    entry_price     DOUBLE PRECISION NOT NULL,
    exit_price      DOUBLE PRECISION NOT NULL,
    quantity        DOUBLE PRECISION NOT NULL,
    realized_pnl    DOUBLE PRECISION NOT NULL,
    fee             DOUBLE PRECISION DEFAULT 0,
    leverage        INTEGER DEFAULT 1,
    exchange_id     TEXT DEFAULT 'paper',
    close_reason    TEXT DEFAULT '',
    opened_at       TIMESTAMPTZ NOT NULL,
    closed_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_trades_agent ON trades(agent_id);
CREATE INDEX IF NOT EXISTS idx_trades_closed ON trades(closed_at);

CREATE TABLE IF NOT EXISTS agent_metrics (
    id              BIGSERIAL PRIMARY KEY,
    agent_id        TEXT NOT NULL,
    generation      INTEGER NOT NULL,
    phase           TEXT NOT NULL,
    capital         DOUBLE PRECISION NOT NULL,
    realized_pnl    DOUBLE PRECISION NOT NULL,
    unrealized_pnl  DOUBLE PRECISION DEFAULT 0,
    total_trades    INTEGER DEFAULT 0,
    winning_trades  INTEGER DEFAULT 0,
    losing_trades   INTEGER DEFAULT 0,
    open_positions  INTEGER DEFAULT 0,
    sharpe_ratio    DOUBLE PRECISION DEFAULT 0,
    max_drawdown_pct DOUBLE PRECISION DEFAULT 0,
    fitness         DOUBLE PRECISION DEFAULT 0,
    hp              DOUBLE PRECISION DEFAULT 100,
    consecutive_losses INTEGER DEFAULT 0,
    uptime_seconds  DOUBLE PRECISION DEFAULT 0,
    recorded_at     TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_agent_metrics_agent ON agent_metrics(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_metrics_gen ON agent_metrics(generation);

CREATE TABLE IF NOT EXISTS agent_deaths (
    id              BIGSERIAL PRIMARY KEY,
    agent_id        TEXT NOT NULL,
    generation      INTEGER NOT NULL,
    cause           TEXT NOT NULL,
    final_capital   DOUBLE PRECISION NOT NULL,
    realized_pnl    DOUBLE PRECISION NOT NULL,
    total_trades    INTEGER DEFAULT 0,
    win_rate        DOUBLE PRECISION DEFAULT 0,
    fitness         DOUBLE PRECISION DEFAULT 0,
    died_at         TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_deaths_gen ON agent_deaths(generation);

CREATE TABLE IF NOT EXISTS generation_snapshots (
    generation      INTEGER PRIMARY KEY,
    population_size INTEGER NOT NULL,
    best_fitness    DOUBLE PRECISION NOT NULL,
    avg_fitness     DOUBLE PRECISION NOT NULL,
    worst_fitness   DOUBLE PRECISION NOT NULL,
    best_dna_id     TEXT DEFAULT '',
    best_agent_id   TEXT DEFAULT '',
    total_trades    INTEGER DEFAULT 0,
    total_pnl       DOUBLE PRECISION DEFAULT 0,
    pool_win_rate   DOUBLE PRECISION DEFAULT 0,
    pool_sharpe     DOUBLE PRECISION DEFAULT 0,
    pool_max_drawdown DOUBLE PRECISION DEFAULT 0,
    survivors       INTEGER DEFAULT 0,
    eliminated      INTEGER DEFAULT 0,
    mutated         INTEGER DEFAULT 0,
    agent_rankings  JSONB DEFAULT '[]',
    dna_pool        JSONB DEFAULT '[]',
    started_at      TIMESTAMPTZ NOT NULL,
    ended_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata        JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS dna_pool (
    dna_id          TEXT PRIMARY KEY,
    generation      INTEGER NOT NULL,
    parent_id       TEXT,
    genes           JSONB NOT NULL,
    fitness         DOUBLE PRECISION DEFAULT 0,
    birth_time      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_dna_gen ON dna_pool(generation);
CREATE INDEX IF NOT EXISTS idx_dna_fitness ON dna_pool(fitness DESC);
"""


class PostgresConnection:
    """
    Manages an asyncpg connection pool.
    Injected into all repository classes.
    """

    __slots__ = ("_dsn", "_pool")

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool = None

    async def connect(self) -> None:
        import asyncpg
        self._pool = await asyncpg.create_pool(
            self._dsn, min_size=2, max_size=10,
            command_timeout=30,
        )
        logger.info("postgres pool connected (min=2, max=10)")

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            logger.info("postgres pool closed")

    async def init_schema(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(SCHEMA_SQL)
        logger.info("database schema initialized")

    @property
    def pool(self):
        if self._pool is None:
            raise RuntimeError("PostgresConnection not connected. Call connect() first.")
        return self._pool

    async def ensure_connected(self) -> None:
        """H-PG1: Health check + reconnect if pool is dead."""
        if self._pool is None:
            await self.connect()
            return
        try:
            async with self._pool.acquire() as conn:
                await conn.execute("SELECT 1")
        except Exception as exc:
            logger.warning("postgres health check failed: %s — reconnecting", exc)
            try:
                await self._pool.close()
            except Exception:
                pass
            self._pool = None
            await self.connect()


# ── Repository implementations ───────────────────────────────

class TradeRepository:
    """Implements ITradeRepository."""

    __slots__ = ("_pg",)

    def __init__(self, pg: PostgresConnection) -> None:
        self._pg = pg

    async def save_trade(self, trade: TradeResult) -> None:
        await self._pg.pool.execute(
            """INSERT INTO trades (trade_id, agent_id, symbol, side, strategy,
               entry_price, exit_price, quantity, realized_pnl, fee, leverage,
               exchange_id, close_reason, opened_at, closed_at)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15)
               ON CONFLICT (trade_id) DO NOTHING""",
            trade.trade_id, trade.agent_id, trade.symbol,
            trade.side.value, trade.strategy.value,
            trade.entry_price, trade.exit_price, trade.quantity,
            trade.realized_pnl, trade.fee, trade.leverage,
            trade.exchange_id.value, trade.close_reason,
            trade.opened_at, trade.closed_at,
        )

    async def get_trades(self, agent_id: str, limit: int = 100) -> List[TradeResult]:
        rows = await self._pg.pool.fetch(
            "SELECT * FROM trades WHERE agent_id=$1 ORDER BY closed_at DESC LIMIT $2",
            agent_id, limit,
        )
        return [self._row_to_trade(r) for r in rows]

    async def get_all_trades(self, limit: int = 500) -> List[TradeResult]:
        rows = await self._pg.pool.fetch(
            "SELECT * FROM trades ORDER BY closed_at DESC LIMIT $1", limit,
        )
        return [self._row_to_trade(r) for r in rows]

    async def get_trades_since(self, since: datetime, limit: int = 500) -> List[TradeResult]:
        rows = await self._pg.pool.fetch(
            "SELECT * FROM trades WHERE closed_at >= $1 ORDER BY closed_at DESC LIMIT $2",
            since, limit,
        )
        return [self._row_to_trade(r) for r in rows]

    @staticmethod
    def _row_to_trade(row) -> TradeResult:
        return TradeResult(
            trade_id=row["trade_id"], symbol=row["symbol"],
            side=OrderSide(row["side"]),
            strategy=StrategyID(row["strategy"]),
            entry_price=row["entry_price"], exit_price=row["exit_price"],
            quantity=row["quantity"], realized_pnl=row["realized_pnl"],
            fee=row["fee"], leverage=row["leverage"],
            opened_at=row["opened_at"], closed_at=row["closed_at"],
            close_reason=row["close_reason"], agent_id=row["agent_id"],
            exchange_id=ExchangeID(row["exchange_id"]),
        )


class AgentRepository:
    """Implements IAgentRepository."""

    __slots__ = ("_pg",)

    def __init__(self, pg: PostgresConnection) -> None:
        self._pg = pg

    async def save_agent_metrics(self, m: AgentMetrics) -> None:
        await self._pg.pool.execute(
            """INSERT INTO agent_metrics (agent_id, generation, phase, capital,
               realized_pnl, unrealized_pnl, total_trades, winning_trades,
               losing_trades, open_positions, sharpe_ratio, max_drawdown_pct,
               fitness, hp, consecutive_losses, uptime_seconds)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16)""",
            m.agent_id, m.generation, m.phase, m.capital,
            m.realized_pnl, m.unrealized_pnl, m.total_trades,
            m.winning_trades, m.losing_trades, m.open_positions,
            m.sharpe_ratio, m.max_drawdown_pct, m.fitness, m.hp,
            m.consecutive_losses, m.uptime_seconds,
        )

    async def save_agent_death(self, agent_id: str, generation: int, cause: str, m: AgentMetrics) -> None:
        await self._pg.pool.execute(
            """INSERT INTO agent_deaths (agent_id, generation, cause, final_capital,
               realized_pnl, total_trades, win_rate, fitness)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8)""",
            agent_id, generation, cause, m.capital,
            m.realized_pnl, m.total_trades, m.win_rate, m.fitness,
        )

    async def get_agent_history(self, agent_id: str) -> List[AgentMetrics]:
        rows = await self._pg.pool.fetch(
            "SELECT * FROM agent_metrics WHERE agent_id=$1 ORDER BY recorded_at", agent_id,
        )
        return [self._row_to_metrics(r) for r in rows]

    async def get_generation_agents(self, generation: int) -> List[AgentMetrics]:
        rows = await self._pg.pool.fetch(
            """SELECT DISTINCT ON (agent_id) * FROM agent_metrics
               WHERE generation=$1 ORDER BY agent_id, recorded_at DESC""",
            generation,
        )
        return [self._row_to_metrics(r) for r in rows]

    @staticmethod
    def _row_to_metrics(row) -> AgentMetrics:
        return AgentMetrics(
            agent_id=row["agent_id"], generation=row["generation"],
            phase=row["phase"], capital=row["capital"],
            realized_pnl=row["realized_pnl"],
            unrealized_pnl=row["unrealized_pnl"],
            total_trades=row["total_trades"],
            winning_trades=row["winning_trades"],
            losing_trades=row["losing_trades"],
            open_positions=row["open_positions"],
            sharpe_ratio=row["sharpe_ratio"],
            max_drawdown_pct=row["max_drawdown_pct"],
            fitness=row["fitness"], hp=row["hp"],
            consecutive_losses=row["consecutive_losses"],
            uptime_seconds=row["uptime_seconds"],
            timestamp=row["recorded_at"],
        )


class GenerationRepository:
    """Implements IGenerationRepository."""

    __slots__ = ("_pg",)

    def __init__(self, pg: PostgresConnection) -> None:
        self._pg = pg

    async def save_snapshot(self, s: GenerationSnapshot) -> None:
        await self._pg.pool.execute(
            """INSERT INTO generation_snapshots (generation, population_size,
               best_fitness, avg_fitness, worst_fitness, best_dna_id,
               best_agent_id, total_trades, total_pnl, pool_win_rate,
               pool_sharpe, pool_max_drawdown, survivors, eliminated,
               mutated, agent_rankings, dna_pool, started_at, ended_at, metadata)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20)
               ON CONFLICT (generation) DO UPDATE SET
               best_fitness=EXCLUDED.best_fitness, avg_fitness=EXCLUDED.avg_fitness,
               ended_at=EXCLUDED.ended_at, agent_rankings=EXCLUDED.agent_rankings,
               dna_pool=EXCLUDED.dna_pool, metadata=EXCLUDED.metadata""",
            s.generation, s.population_size,
            s.best_fitness, s.avg_fitness, s.worst_fitness,
            s.best_dna_id, s.best_agent_id,
            s.total_trades, s.total_pnl, s.pool_win_rate,
            s.pool_sharpe, s.pool_max_drawdown,
            s.survivors, s.eliminated, s.mutated,
            json.dumps(s.agent_rankings), json.dumps(s.dna_pool),
            s.started_at, s.ended_at, json.dumps(s.metadata),
        )

    async def get_snapshot(self, generation: int) -> Optional[GenerationSnapshot]:
        row = await self._pg.pool.fetchrow(
            "SELECT * FROM generation_snapshots WHERE generation=$1", generation,
        )
        return self._row_to_snapshot(row) if row else None

    async def get_all_snapshots(self, limit: int = 100) -> List[GenerationSnapshot]:
        rows = await self._pg.pool.fetch(
            "SELECT * FROM generation_snapshots ORDER BY generation DESC LIMIT $1", limit,
        )
        return [self._row_to_snapshot(r) for r in rows]

    async def get_latest_snapshot(self) -> Optional[GenerationSnapshot]:
        row = await self._pg.pool.fetchrow(
            "SELECT * FROM generation_snapshots ORDER BY generation DESC LIMIT 1",
        )
        return self._row_to_snapshot(row) if row else None

    @staticmethod
    def _row_to_snapshot(row) -> GenerationSnapshot:
        return GenerationSnapshot(
            generation=row["generation"],
            population_size=row["population_size"],
            best_fitness=row["best_fitness"],
            avg_fitness=row["avg_fitness"],
            worst_fitness=row["worst_fitness"],
            best_dna_id=row["best_dna_id"],
            best_agent_id=row["best_agent_id"],
            total_trades=row["total_trades"],
            total_pnl=row["total_pnl"],
            pool_win_rate=row["pool_win_rate"],
            pool_sharpe=row["pool_sharpe"],
            pool_max_drawdown=row["pool_max_drawdown"],
            survivors=row["survivors"],
            eliminated=row["eliminated"],
            mutated=row["mutated"],
            agent_rankings=json.loads(row["agent_rankings"]) if row["agent_rankings"] else [],
            dna_pool=json.loads(row["dna_pool"]) if row["dna_pool"] else [],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )


class DNARepository:
    """Implements IDNARepository."""

    __slots__ = ("_pg",)

    def __init__(self, pg: PostgresConnection) -> None:
        self._pg = pg

    async def save_dna(self, dna: DNAData) -> None:
        await self._pg.pool.execute(
            """INSERT INTO dna_pool (dna_id, generation, parent_id, genes, fitness, birth_time)
               VALUES ($1,$2,$3,$4,$5,$6)
               ON CONFLICT (dna_id) DO UPDATE SET fitness=EXCLUDED.fitness""",
            dna.dna_id, dna.generation, dna.parent_id,
            json.dumps(dna.genes), dna.fitness, dna.birth_time,
        )

    async def get_dna(self, dna_id: str) -> Optional[DNAData]:
        row = await self._pg.pool.fetchrow(
            "SELECT * FROM dna_pool WHERE dna_id=$1", dna_id,
        )
        return self._row_to_dna(row) if row else None

    async def get_hall_of_fame(self, limit: int = 10) -> List[DNAData]:
        rows = await self._pg.pool.fetch(
            "SELECT * FROM dna_pool ORDER BY fitness DESC LIMIT $1", limit,
        )
        return [self._row_to_dna(r) for r in rows]

    async def get_generation_dna(self, generation: int) -> List[DNAData]:
        rows = await self._pg.pool.fetch(
            "SELECT * FROM dna_pool WHERE generation=$1 ORDER BY fitness DESC", generation,
        )
        return [self._row_to_dna(r) for r in rows]

    @staticmethod
    def _row_to_dna(row) -> DNAData:
        return DNAData(
            dna_id=row["dna_id"], generation=row["generation"],
            parent_id=row["parent_id"],
            genes=json.loads(row["genes"]) if isinstance(row["genes"], str) else row["genes"],
            fitness=row["fitness"], birth_time=row["birth_time"],
        )
