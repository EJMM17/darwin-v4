"""
Darwin v4 — Enumerations.
Layer 0 (interfaces). Zero dependencies.
"""
from enum import Enum, auto


class AgentPhase(Enum):
    INCUBATION = "incubation"
    LIVE       = "live"
    DYING      = "dying"
    DEAD       = "dead"


class GrowthPhase(Enum):
    BOOTSTRAP      = "bootstrap"
    SCALING        = "scaling"
    ACCELERATION   = "acceleration"
    CONSOLIDATION  = "consolidation"


class OrderSide(Enum):
    BUY  = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT  = "LIMIT"


class TimeFrame(Enum):
    M1  = "1"
    M5  = "5"
    M15 = "15"
    M30 = "30"
    H1  = "60"
    H4  = "240"
    D1  = "D"


class StrategyID(Enum):
    MOMENTUM       = "momentum"
    MEAN_REVERSION = "mean_reversion"
    SCALPING       = "scalping"
    BREAKOUT       = "breakout"


class MarketRegime(Enum):
    TRENDING_UP     = auto()
    TRENDING_DOWN   = auto()
    RANGING         = auto()
    HIGH_VOLATILITY = auto()
    LOW_VOLATILITY  = auto()


class SignalStrength(Enum):
    STRONG = "strong"
    MEDIUM = "medium"
    WEAK   = "weak"
    NONE   = "none"


class ExchangeID(Enum):
    BYBIT   = "bybit"
    BINANCE = "binance"
    PAPER   = "paper"


class EventType(Enum):
    AGENT_SPAWNED           = auto()
    AGENT_DIED              = auto()
    AGENT_PHASE_CHANGED     = auto()
    TRADE_OPENED            = auto()
    TRADE_CLOSED            = auto()
    POSITION_UPDATED        = auto()
    GENERATION_COMPLETE     = auto()
    SPAWN_AGENT_REQUESTED   = auto()
    PHASE_CHANGED           = auto()
    CAPITAL_REBALANCED      = auto()
    TICK                    = auto()
    HEARTBEAT               = auto()
    CIRCUIT_BREAKER_TRIPPED = auto()
    COOLDOWN_STARTED        = auto()
    COOLDOWN_ENDED          = auto()
    GENERATION_SNAPSHOT     = auto()
    RISK_ALERT              = auto()
    RISK_STATE_CHANGED      = auto()
    EXCHANGE_ERROR          = auto()
    EVOLUTION_RANKED        = auto()


class PortfolioRiskState(Enum):
    """Portfolio-level risk state machine."""
    NORMAL     = "normal"       # All systems go, full allocation
    DEFENSIVE  = "defensive"    # Elevated risk, reduce position sizes
    CRITICAL   = "critical"     # Severe drawdown, close-only mode
    HALTED     = "halted"       # Trading fully suspended
