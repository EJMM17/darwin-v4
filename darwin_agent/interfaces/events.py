"""
Darwin v4 — Event definitions.
Layer 0 (interfaces). Depends only on interfaces.enums + interfaces.types.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict

from darwin_agent.interfaces.enums import EventType
from darwin_agent.interfaces.types import AgentMetrics, TradeResult


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class Event:
    event_type: EventType
    timestamp: datetime = field(default_factory=_utcnow)
    source: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)


# ── Factory functions ────────────────────────────────────────

def trade_opened_event(
    agent_id: str, symbol: str, side: str, size: float,
    entry_price: float, leverage: int, strategy: str,
) -> Event:
    return Event(
        event_type=EventType.TRADE_OPENED, source=agent_id,
        payload={"symbol": symbol, "side": side, "size": size,
                 "entry_price": entry_price, "leverage": leverage,
                 "strategy": strategy},
    )


def trade_closed_event(agent_id: str, trade: TradeResult) -> Event:
    return Event(
        event_type=EventType.TRADE_CLOSED, source=agent_id,
        payload={"trade_id": trade.trade_id, "symbol": trade.symbol,
                 "side": trade.side.value, "strategy": trade.strategy.value,
                 "entry_price": trade.entry_price, "exit_price": trade.exit_price,
                 "realized_pnl": trade.realized_pnl, "pnl_pct": trade.pnl_pct,
                 "fee": trade.fee, "close_reason": trade.close_reason,
                 "leverage": trade.leverage},
    )


def agent_spawned_event(agent_id: str, generation: int, capital: float) -> Event:
    return Event(
        event_type=EventType.AGENT_SPAWNED, source=agent_id,
        payload={"generation": generation, "capital": capital},
    )


def agent_died_event(
    agent_id: str, generation: int, cause: str, metrics: AgentMetrics,
) -> Event:
    return Event(
        event_type=EventType.AGENT_DIED, source=agent_id,
        payload={"generation": generation, "cause": cause,
                 "total_trades": metrics.total_trades,
                 "win_rate": metrics.win_rate,
                 "realized_pnl": metrics.realized_pnl,
                 "fitness": metrics.fitness},
    )


def generation_complete_event(
    agent_id: str, generation: int, metrics: AgentMetrics,
) -> Event:
    return Event(
        event_type=EventType.GENERATION_COMPLETE, source=agent_id,
        payload={"generation": generation,
                 "total_trades": metrics.total_trades,
                 "win_rate": metrics.win_rate,
                 "realized_pnl": metrics.realized_pnl,
                 "fitness": metrics.fitness},
    )


def spawn_agent_requested_event(parent_id: str, dna_payload: Dict) -> Event:
    return Event(
        event_type=EventType.SPAWN_AGENT_REQUESTED, source=parent_id,
        payload={"dna": dna_payload},
    )


def tick_event() -> Event:
    return Event(event_type=EventType.TICK, source="scheduler")


def heartbeat_event(pool_metrics: Dict) -> Event:
    return Event(
        event_type=EventType.HEARTBEAT, source="scheduler",
        payload=pool_metrics,
    )


def risk_state_changed_event(
    old_state: str, new_state: str, reason: str, metrics: Dict,
) -> Event:
    return Event(
        event_type=EventType.RISK_STATE_CHANGED, source="portfolio_risk_engine",
        payload={"old_state": old_state, "new_state": new_state,
                 "reason": reason, "metrics": metrics},
    )
