"""
Darwin v4 — Redis real-time cache layer.

Layer 1 (infra). Implements ICache protocol.
Used for: agent state snapshots, ticker cache, pool metrics,
WebSocket pub/sub broadcast, rate limiting.
"""
from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

from darwin_agent.interfaces.types import (
    AgentMetrics, PortfolioRiskMetrics, Ticker,
)

logger = logging.getLogger("darwin.redis")


class RedisCache:
    """
    Async Redis cache implementing ICache protocol.
    Uses redis.asyncio for production async Redis access.
    
    Key namespace design:
        darwin:ticker:{symbol}           → Ticker JSON (TTL 5s)
        darwin:agent:{agent_id}          → AgentMetrics JSON (TTL 30s)
        darwin:pool:metrics              → PoolMetrics JSON (TTL 10s)
        darwin:pool:risk                 → PortfolioRiskMetrics JSON (TTL 10s)
        darwin:positions:{agent_id}      → positions JSON (TTL 30s)
        darwin:generation:current        → current generation number
        darwin:leaderboard               → sorted set by fitness
        darwin:lock:{resource}           → distributed lock
    """

    __slots__ = ("_url", "_client", "_prefix")

    def __init__(self, url: str = "redis://localhost:6379", prefix: str = "darwin") -> None:
        self._url = url
        self._client = None
        self._prefix = prefix

    async def connect(self) -> None:
        import redis.asyncio as aioredis
        self._client = aioredis.from_url(
            self._url, decode_responses=True,
            max_connections=20,
        )
        await self._client.ping()
        logger.info("redis connected: %s", self._url)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            logger.info("redis closed")

    def _key(self, key: str) -> str:
        return f"{self._prefix}:{key}"

    # ── ICache protocol ──────────────────────────────────────

    async def get(self, key: str) -> Optional[str]:
        return await self._client.get(self._key(key))

    async def set(self, key: str, value: str, ttl_seconds: int = 300) -> None:
        await self._client.set(self._key(key), value, ex=ttl_seconds)

    async def delete(self, key: str) -> None:
        await self._client.delete(self._key(key))

    async def get_json(self, key: str) -> Optional[Dict]:
        raw = await self.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    async def set_json(self, key: str, value: Dict, ttl_seconds: int = 300) -> None:
        await self.set(key, json.dumps(value, default=str), ttl_seconds)

    async def publish(self, channel: str, message: str) -> None:
        await self._client.publish(self._key(channel), message)

    async def exists(self, key: str) -> bool:
        return bool(await self._client.exists(self._key(key)))

    # ── Domain-specific cache methods ────────────────────────

    async def cache_ticker(self, ticker: Ticker) -> None:
        data = {
            "symbol": ticker.symbol, "last_price": ticker.last_price,
            "bid": ticker.bid, "ask": ticker.ask,
            "volume_24h": ticker.volume_24h,
            "timestamp": ticker.timestamp.isoformat(),
        }
        await self.set_json(f"ticker:{ticker.symbol}", data, ttl_seconds=5)

    async def get_cached_ticker(self, symbol: str) -> Optional[Dict]:
        return await self.get_json(f"ticker:{symbol}")

    async def cache_agent_metrics(self, m: AgentMetrics) -> None:
        data = {
            "agent_id": m.agent_id, "generation": m.generation,
            "phase": m.phase, "capital": m.capital,
            "realized_pnl": m.realized_pnl, "fitness": m.fitness,
            "hp": m.hp, "total_trades": m.total_trades,
            "win_rate": m.win_rate, "open_positions": m.open_positions,
        }
        await self.set_json(f"agent:{m.agent_id}", data, ttl_seconds=30)

    async def cache_pool_metrics(self, metrics_dict: Dict) -> None:
        await self.set_json("pool:metrics", metrics_dict, ttl_seconds=10)

    async def cache_risk_metrics(self, m: PortfolioRiskMetrics) -> None:
        data = {
            "total_exposure": m.total_exposure,
            "net_exposure": m.net_exposure,
            "sharpe_ratio": m.sharpe_ratio,
            "max_drawdown_pct": m.max_drawdown_pct,
            "current_drawdown_pct": m.current_drawdown_pct,
            "var_95": m.var_95, "var_99": m.var_99,
            "win_rate": m.win_rate,
            "profit_factor": m.profit_factor,
        }
        await self.set_json("pool:risk", data, ttl_seconds=10)

    async def update_leaderboard(self, agent_id: str, fitness: float) -> None:
        await self._client.zadd(self._key("leaderboard"), {agent_id: fitness})

    async def get_leaderboard(self, limit: int = 10) -> List[Dict]:
        results = await self._client.zrevrange(
            self._key("leaderboard"), 0, limit - 1, withscores=True,
        )
        return [{"agent_id": aid, "fitness": score} for aid, score in results]

    async def remove_from_leaderboard(self, agent_id: str) -> None:
        await self._client.zrem(self._key("leaderboard"), agent_id)

    async def set_current_generation(self, gen: int) -> None:
        await self.set("generation:current", str(gen), ttl_seconds=86400)

    async def get_current_generation(self) -> int:
        val = await self.get("generation:current")
        return int(val) if val else 0

    async def publish_event(self, event_type: str, data: Dict) -> None:
        msg = json.dumps({"type": event_type, "data": data}, default=str)
        await self.publish("events", msg)
