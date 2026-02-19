"""
Darwin v4 — Shared value types.
Layer 0. Depends only on interfaces.enums.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from darwin_agent.interfaces.enums import (
    ExchangeID, GrowthPhase, OrderSide, OrderType, PortfolioRiskState,
    SignalStrength, StrategyID, TimeFrame,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

def _new_id() -> str:
    return uuid.uuid4().hex[:12]


# ── Market data ──────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: TimeFrame = TimeFrame.M15


@dataclass(slots=True)
class Ticker:
    symbol: str
    last_price: float
    bid: float
    ask: float
    volume_24h: float
    timestamp: datetime = field(default_factory=_utcnow)


# ── Orders ───────────────────────────────────────────────────

@dataclass(slots=True)
class OrderRequest:
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    leverage: int = 1
    reduce_only: bool = False
    agent_id: str = ""
    exchange_id: Optional[ExchangeID] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OrderResult:
    order_id: str = ""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    filled_qty: float = 0.0
    filled_price: float = 0.0
    fee: float = 0.0
    success: bool = True
    error: str = ""
    exchange_id: ExchangeID = ExchangeID.PAPER
    timestamp: datetime = field(default_factory=_utcnow)


# ── Positions ────────────────────────────────────────────────

@dataclass(slots=True)
class Position:
    symbol: str
    side: OrderSide
    size: float
    entry_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    leverage: int = 1
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop_price: Optional[float] = None
    opened_at: datetime = field(default_factory=_utcnow)
    agent_id: str = ""
    exchange_id: ExchangeID = ExchangeID.PAPER


# ── Trade results ────────────────────────────────────────────

@dataclass(slots=True)
class TradeResult:
    trade_id: str = field(default_factory=_new_id)
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    strategy: StrategyID = StrategyID.MOMENTUM
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: float = 0.0
    realized_pnl: float = 0.0
    fee: float = 0.0
    leverage: int = 1
    opened_at: datetime = field(default_factory=_utcnow)
    closed_at: datetime = field(default_factory=_utcnow)
    close_reason: str = ""
    agent_id: str = ""
    exchange_id: ExchangeID = ExchangeID.PAPER

    @property
    def pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        r = (self.exit_price - self.entry_price) / self.entry_price
        if self.side == OrderSide.SELL:
            r = -r
        return r * self.leverage * 100


# ── Signals ──────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Signal:
    strategy: StrategyID
    symbol: str
    side: OrderSide
    strength: SignalStrength
    confidence: float = 0.0
    stop_loss_pct: float = 1.5
    take_profit_pct: float = 3.0
    timeframe: TimeFrame = TimeFrame.M15
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Risk ─────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class RiskVerdict:
    approved: bool
    reason: str = ""
    adjusted_size: Optional[float] = None


@dataclass(slots=True)
class PortfolioRiskMetrics:
    """Snapshot of portfolio-level risk computed by the PortfolioRiskEngine."""
    risk_state: PortfolioRiskState = PortfolioRiskState.NORMAL
    total_equity: float = 0.0
    peak_equity: float = 0.0
    total_exposure: float = 0.0
    net_exposure: float = 0.0
    exposure_by_exchange: Dict[str, float] = field(default_factory=dict)
    exposure_by_symbol: Dict[str, float] = field(default_factory=dict)
    drawdown_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    correlation_risk: float = 0.0
    var_95: float = 0.0
    var_99: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_positions: int = 0
    consecutive_losses: int = 0
    daily_pnl: float = 0.0
    total_realized_pnl: float = 0.0
    total_trades: int = 0
    size_multiplier: float = 1.0
    halted_reason: str = ""
    timestamp: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> Dict:
        return {
            "risk_state": self.risk_state.value,
            "total_equity": round(self.total_equity, 2),
            "peak_equity": round(self.peak_equity, 2),
            "drawdown_pct": round(self.drawdown_pct, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "total_exposure": round(self.total_exposure, 2),
            "net_exposure": round(self.net_exposure, 2),
            "exposure_by_exchange": self.exposure_by_exchange,
            "exposure_by_symbol": self.exposure_by_symbol,
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "sortino_ratio": round(self.sortino_ratio, 3),
            "var_95": round(self.var_95, 2),
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 3),
            "consecutive_losses": self.consecutive_losses,
            "total_positions": self.total_positions,
            "total_trades": self.total_trades,
            "daily_pnl": round(self.daily_pnl, 2),
            "total_realized_pnl": round(self.total_realized_pnl, 2),
            "size_multiplier": self.size_multiplier,
            "halted_reason": self.halted_reason,
        }


# ── Capital ──────────────────────────────────────────────────

@dataclass(slots=True)
class BalanceSnapshot:
    agent_id: str = ""
    capital: float = 0.0
    allocated: float = 0.0
    available: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    timestamp: datetime = field(default_factory=_utcnow)


@dataclass(frozen=True, slots=True)
class PhaseParams:
    phase: GrowthPhase
    risk_pct: float
    max_positions: int
    leverage: int


@dataclass(frozen=True, slots=True)
class AllocationSlice:
    agent_id: str
    amount: float
    phase_params: PhaseParams


# ── Agent metrics ────────────────────────────────────────────

@dataclass(slots=True)
class AgentMetrics:
    agent_id: str = ""
    generation: int = 0
    phase: str = "incubation"
    capital: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    open_positions: int = 0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    fitness: float = 0.0
    hp: float = 100.0
    current_strategy: str = ""
    consecutive_losses: int = 0
    uptime_seconds: float = 0.0
    timestamp: datetime = field(default_factory=_utcnow)

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def is_alive(self) -> bool:
        return self.phase not in ("dead", "dying")


@dataclass(slots=True)
class AgentEvalData:
    """
    Complete evaluation packet for a single agent.

    Carries everything RiskAwareFitness needs to compute a
    portfolio-aware fitness score. The EvolutionEngine receives
    a list of these instead of raw AgentMetrics.

    metrics:          Standard agent metrics (trades, PnL, Sharpe, etc.)
    initial_capital:  Starting capital for ROC calculations.
    pnl_series:       Ordered list of per-trade realized PnL.
    exposure:         Symbol → notional fraction of agent capital.
    dna:              Agent's DNA for breeding if it survives.
    """
    metrics: AgentMetrics
    initial_capital: float = 0.0
    pnl_series: List[float] = field(default_factory=list)
    exposure: Dict[str, float] = field(default_factory=dict)
    dna: Optional["DNAData"] = None
    # v4.2 fields (backward-compatible defaults)
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    total_notional: float = 0.0
    # v4.3 field
    trend_series: List[float] = field(default_factory=list)
    # v4.4 field
    n_bars: int = 0
    bars_gated: int = 0


# ── DNA (evolution) ──────────────────────────────────────────

@dataclass(slots=True)
class DNAData:
    """Serializable gene data for evolution."""
    genes: Dict[str, float] = field(default_factory=dict)
    generation: int = 0
    parent_id: Optional[str] = None
    dna_id: str = field(default_factory=_new_id)
    fitness: float = 0.0
    birth_time: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> Dict:
        return {
            "genes": dict(self.genes),
            "generation": self.generation,
            "parent_id": self.parent_id,
            "dna_id": self.dna_id,
            "fitness": self.fitness,
            "birth_time": self.birth_time.isoformat(),
        }


# ── Generation snapshot (evolution persistence) ──────────────

@dataclass(slots=True)
class GenerationSnapshot:
    """Complete record of a generation for Postgres persistence."""
    generation: int = 0
    population_size: int = 0
    best_fitness: float = 0.0
    avg_fitness: float = 0.0
    worst_fitness: float = 0.0
    best_dna_id: str = ""
    best_agent_id: str = ""
    total_trades: int = 0
    total_pnl: float = 0.0
    pool_win_rate: float = 0.0
    pool_sharpe: float = 0.0
    pool_max_drawdown: float = 0.0
    survivors: int = 0
    eliminated: int = 0
    mutated: int = 0
    agent_rankings: List[Dict[str, Any]] = field(default_factory=list)
    dna_pool: List[Dict[str, Any]] = field(default_factory=list)
    started_at: datetime = field(default_factory=_utcnow)
    ended_at: datetime = field(default_factory=_utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Exchange routing ─────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ExchangeStatus:
    exchange_id: ExchangeID
    connected: bool
    latency_ms: float = 0.0
    rate_limit_remaining: int = 100
    last_error: str = ""
    timestamp: datetime = field(default_factory=_utcnow)
