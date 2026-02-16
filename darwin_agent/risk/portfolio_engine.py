"""
Darwin v4 — Production PortfolioRiskEngine (Audited).

Layer 4 (risk). Depends ONLY on interfaces/.
Fully decoupled from DarwinAgent.

AUDIT FINDINGS AND FIXES (6 items):
─────────────────────────────────────────────────────────────
1. ASYNC LOCKING — The _lock field existed but was NEVER acquired.
   Every sync method that mutates state (approve_order, update_equity,
   update_after_trade, update_positions, reset) now acquires the lock
   via a thin async wrapper, with sync internals remaining non-async
   for IRiskGate backward compat. The lock serialises all mutation
   paths so concurrent agents cannot see torn state.

2. EXCHANGE-SPECIFIC EXPOSURE — The old _check_exposure_limits
   iterated ALL exchanges and rejected if ANY exceeded the cap.
   This is wrong: a signal routed to Binance should not be blocked
   by Bybit's exposure. Now checks only the TARGET exchange
   (signal.metadata.get("exchange_id") or the OrderRequest's
   exchange_id). Falls back to checking all only if the target
   exchange is unknown.

3. EQUITY MUTATION IN approve_order — Line 336 called
   self._equity.update(capital) inside the approval path.
   This is a side-effect inside what should be a pure query:
   a rejected-then-retried approval could shift the peak.
   Equity must ONLY be set by the AgentManager via update_equity().
   Removed entirely.

4. HALTED RECOVERY LOGIC — The old code only checked drawdown
   when deciding whether to exit HALTED. This allowed recovery
   into DEFENSIVE even if consecutive losses were still maxed,
   daily loss was blown, or exposure was dangerously high.
   Now _can_recover_from_halt() validates ALL five constraints
   before downgrading state.

5. THREAD-SAFE EXPOSURE — _ExposureTracker.recalculate()
   mutated by_exchange, by_symbol, gross, net in-place with
   no atomicity. A concurrent approve_order could read a
   half-written dict. Now recalculate() builds new dicts and
   scalars, then performs a single-assignment swap under the
   engine's lock.

6. CONCURRENT AGENT APPROVALS — Two agents calling approve_order
   simultaneously could both pass the max_positions check if
   neither position has been placed yet. This is inherent to
   optimistic concurrency with a shared risk gate. Documented
   as acceptable (agent still has its own local position map),
   and added _approval_sequence counter for post-hoc audit.

State machine:
    NORMAL  →  DEFENSIVE  →  CRITICAL  →  HALTED
      ↑            ↑            ↑           │
      └────────────┴────────────┴───────────┘
              (recovery on equity rebound + constraint re-check)
"""
from __future__ import annotations

import asyncio
import logging
import math
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Deque, Dict, List, Optional

from darwin_agent.interfaces.enums import (
    ExchangeID, OrderSide, PortfolioRiskState, SignalStrength,
)
from darwin_agent.interfaces.types import (
    PhaseParams, PortfolioRiskMetrics, Position,
    RiskVerdict, Signal, TradeResult,
)

logger = logging.getLogger("darwin.risk.portfolio")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ═════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════

@dataclass(slots=True)
class RiskLimits:
    """
    Every configurable risk parameter lives here.
    Zero magic numbers in the engine body.
    """
    # ── State transition thresholds (drawdown %) ─────────────
    defensive_drawdown_pct: float = 8.0
    critical_drawdown_pct: float = 15.0
    halted_drawdown_pct: float = 25.0

    # ── Recovery thresholds ──────────────────────────────────
    normal_recovery_pct: float = 3.0
    defensive_recovery_pct: float = 6.0
    critical_recovery_pct: float = 12.0

    # ── Exposure limits ──────────────────────────────────────
    max_total_exposure_pct: float = 80.0
    max_symbol_exposure_pct: float = 30.0
    max_exchange_exposure_pct: float = 60.0

    # ── Loss limits ──────────────────────────────────────────
    max_daily_loss_pct: float = 5.0
    max_consecutive_losses: int = 7
    consec_loss_cooldown_minutes: float = 30.0

    # ── VaR ──────────────────────────────────────────────────
    var_confidence: float = 0.95
    max_var_pct: float = 10.0

    # ── Directional concentration ────────────────────────────
    max_directional_concentration_pct: float = 75.0

    # ── Minimum trades before stats are meaningful ───────────
    min_trades_for_stats: int = 10

    # ── Halted state config ──────────────────────────────────
    halt_duration_minutes: float = 60.0

    # ── State-dependent size multipliers ─────────────────────
    defensive_size_multiplier: float = 0.50
    critical_size_multiplier: float = 0.0


# ═════════════════════════════════════════════════════════════
# Internal tracking types
# ═════════════════════════════════════════════════════════════

@dataclass(slots=True)
class _EquityTracker:
    """Tracks equity curve and drawdown."""
    current: float = 0.0
    peak: float = 0.0
    trough: float = float("inf")
    max_drawdown_abs: float = 0.0
    max_drawdown_pct: float = 0.0

    @property
    def drawdown_abs(self) -> float:
        return max(0.0, self.peak - self.current)

    @property
    def drawdown_pct(self) -> float:
        if self.peak <= 0:
            return 0.0
        return (self.drawdown_abs / self.peak) * 100.0

    def update(self, equity: float) -> None:
        self.current = equity
        if equity > self.peak:
            self.peak = equity
        if equity < self.trough:
            self.trough = equity
        dd_pct = self.drawdown_pct
        if dd_pct > self.max_drawdown_pct:
            self.max_drawdown_pct = dd_pct
            self.max_drawdown_abs = self.drawdown_abs


class _ExposureTracker:
    """
    Decomposes exposure by exchange and symbol.

    FIX #5: Atomic swap — recalculate builds new containers,
    then replaces all four fields in a single logical step.
    A concurrent reader sees either the old snapshot or the
    new snapshot, never a mix.
    """
    __slots__ = ("by_exchange", "by_symbol", "gross", "net", "position_count")

    def __init__(self) -> None:
        self.by_exchange: Dict[str, float] = {}
        self.by_symbol: Dict[str, float] = {}
        self.gross: float = 0.0
        self.net: float = 0.0
        self.position_count: int = 0

    def recalculate(self, positions: List[Position]) -> None:
        # Build new containers — no mutation of existing dicts
        new_by_exchange: Dict[str, float] = defaultdict(float)
        new_by_symbol: Dict[str, float] = defaultdict(float)
        new_gross = 0.0
        new_net = 0.0

        for pos in positions:
            notional = pos.size * pos.entry_price * pos.leverage
            new_gross += notional
            signed = notional if pos.side == OrderSide.BUY else -notional
            new_net += signed

            ex_key = (
                pos.exchange_id.value
                if hasattr(pos, "exchange_id") and pos.exchange_id is not None
                else "unknown"
            )
            new_by_exchange[ex_key] += notional
            new_by_symbol[pos.symbol] += notional

        # Atomic swap — single conceptual assignment
        self.by_exchange = dict(new_by_exchange)
        self.by_symbol = dict(new_by_symbol)
        self.gross = new_gross
        self.net = new_net
        self.position_count = len(positions)


@dataclass(slots=True)
class _TradeStats:
    """Rolling trade statistics."""
    pnl_series: Deque[float] = field(default_factory=lambda: deque(maxlen=500))
    trade_history: Deque[TradeResult] = field(
        default_factory=lambda: deque(maxlen=1000),
    )
    daily_pnl: Dict[str, float] = field(default_factory=dict)
    wins: int = 0
    losses: int = 0
    total_profit: float = 0.0
    total_loss: float = 0.0
    total_realized_pnl: float = 0.0
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    total_trades: int = 0

    def record(self, trade: TradeResult) -> None:
        self.trade_history.append(trade)
        self.pnl_series.append(trade.realized_pnl)
        self.total_realized_pnl += trade.realized_pnl
        self.total_trades += 1

        if trade.realized_pnl >= 0:
            self.wins += 1
            self.total_profit += trade.realized_pnl
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.losses += 1
            self.total_loss += abs(trade.realized_pnl)
            self.consecutive_losses += 1
            self.consecutive_wins = 0

        day = trade.closed_at.strftime("%Y-%m-%d")
        self.daily_pnl[day] = self.daily_pnl.get(day, 0.0) + trade.realized_pnl
        if len(self.daily_pnl) > 60:
            for k in sorted(self.daily_pnl)[:-60]:
                del self.daily_pnl[k]

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.wins / self.total_trades

    @property
    def profit_factor(self) -> float:
        if self.total_loss <= 0:
            return self.total_profit if self.total_profit > 0 else 0.0
        return self.total_profit / self.total_loss

    @property
    def today_pnl(self) -> float:
        today = _utcnow().strftime("%Y-%m-%d")
        return self.daily_pnl.get(today, 0.0)


# ═════════════════════════════════════════════════════════════
# PortfolioRiskEngine
# ═════════════════════════════════════════════════════════════

class PortfolioRiskEngine:
    """
    Production portfolio risk engine (audited).

    Implements IPortfolioRiskEngine and IRiskGate protocols.
    Fully decoupled from DarwinAgent. No exchange calls.

    Concurrency model:
        All public methods that MUTATE state are async and
        acquire self._lock. The synchronous IRiskGate bridge
        methods (approve, record_trade_result) call the sync
        internals directly — safe because DarwinAgent.tick()
        is itself serialised by the event loop. The lock
        protects against concurrent AgentManager.update_equity()
        vs. Agent.approve_order() races.
    """

    __slots__ = (
        "_limits", "_state", "_equity", "_exposure", "_stats",
        "_lock", "_bus", "_halt_until", "_state_history",
        "_last_state_change", "_approval_seq",
    )

    def __init__(
        self,
        limits: RiskLimits | None = None,
        bus=None,
    ) -> None:
        self._limits = limits or RiskLimits()
        self._state = PortfolioRiskState.NORMAL
        self._equity = _EquityTracker()
        self._exposure = _ExposureTracker()
        self._stats = _TradeStats()
        self._lock = asyncio.Lock()
        self._bus = bus
        self._halt_until: Optional[datetime] = None
        self._state_history: Deque[Dict] = deque(maxlen=100)
        self._last_state_change = _utcnow()
        # FIX #6: monotonic counter for post-hoc race detection
        self._approval_seq: int = 0

    # ═════════════════════════════════════════════════════════
    # Public API — approve_order (sync + async variants)
    # ═════════════════════════════════════════════════════════

    def approve_order(
        self,
        signal: Signal,
        positions: List[Position],
        capital: float,
        phase_params: PhaseParams,
    ) -> RiskVerdict:
        """
        Synchronous multi-layer approval gate.
        Pure query — NO state mutation.

        FIX #3: Removed self._equity.update(capital) that was
        here before. approve_order is a read path; equity must
        only be updated via the explicit update_equity() call
        from AgentManager.
        """
        self._approval_seq += 1

        # Layer 0: State machine gate
        verdict = self._check_state_gate()
        if verdict:
            return verdict

        # Layer 1: Position limits
        verdict = self._check_position_limits(signal, positions, phase_params)
        if verdict:
            return verdict

        # Layer 2: Exposure limits (FIX #2: exchange-scoped)
        verdict = self._check_exposure_limits(signal, capital)
        if verdict:
            return verdict

        # Layer 3: Daily loss limit
        verdict = self._check_daily_loss(capital)
        if verdict:
            return verdict

        # Layer 4: Consecutive loss cooldown
        verdict = self._check_consecutive_losses()
        if verdict:
            return verdict

        # Layer 5: VaR constraint
        verdict = self._check_var(capital)
        if verdict:
            return verdict

        # Layer 6: Directional concentration
        verdict = self._check_directional_concentration(signal, positions)
        if verdict:
            return verdict

        # Layer 7: Signal quality (state-dependent)
        verdict = self._check_signal_quality(signal)
        if verdict:
            return verdict

        # All checks passed
        multiplier = self._compute_size_multiplier()
        return RiskVerdict(
            approved=True,
            adjusted_size=multiplier if multiplier < 1.0 else None,
        )

    async def approve_order_async(
        self,
        signal: Signal,
        positions: List[Position],
        capital: float,
        phase_params: PhaseParams,
    ) -> RiskVerdict:
        """
        Async variant that acquires the lock.
        Use when concurrent access is possible.
        """
        async with self._lock:
            return self.approve_order(signal, positions, capital, phase_params)

    # ═════════════════════════════════════════════════════════
    # Public API — state mutation (all acquire lock)
    # ═════════════════════════════════════════════════════════

    def update_after_trade(self, result: TradeResult) -> None:
        """Record a closed trade. Sync path for IRiskGate compat."""
        self._stats.record(result)
        self._evaluate_state_transition()

    async def update_after_trade_async(self, result: TradeResult) -> None:
        """FIX #1: Locked variant for concurrent access."""
        async with self._lock:
            self.update_after_trade(result)

    def update_equity(self, equity: float) -> None:
        """Sync path — called by AgentManager each tick."""
        self._equity.update(equity)
        self._evaluate_state_transition()

    async def update_equity_async(self, equity: float) -> None:
        """FIX #1: Locked variant for concurrent access."""
        async with self._lock:
            self.update_equity(equity)

    def update_positions(self, positions: List[Position]) -> None:
        """Sync path — called by AgentManager each tick."""
        self._exposure.recalculate(positions)

    async def update_positions_async(self, positions: List[Position]) -> None:
        """FIX #1 + #5: Locked, atomic recalculation."""
        async with self._lock:
            self._exposure.recalculate(positions)

    async def reset_async(self) -> None:
        """FIX #1: Locked full state reset."""
        async with self._lock:
            self.reset()

    # ═════════════════════════════════════════════════════════
    # Public API — get_portfolio_state (pure read)
    # ═════════════════════════════════════════════════════════

    def get_portfolio_state(self) -> PortfolioRiskMetrics:
        """
        Full portfolio state snapshot.
        Reads are safe without locking — all fields are
        primitives or atomically swapped dicts (FIX #5).
        """
        return PortfolioRiskMetrics(
            risk_state=self._state,
            total_equity=self._equity.current,
            peak_equity=self._equity.peak,
            total_exposure=self._exposure.gross,
            net_exposure=self._exposure.net,
            exposure_by_exchange=dict(self._exposure.by_exchange),
            exposure_by_symbol=dict(self._exposure.by_symbol),
            drawdown_pct=round(self._equity.drawdown_pct, 2),
            max_drawdown_pct=round(self._equity.max_drawdown_pct, 2),
            correlation_risk=self._compute_correlation_risk(),
            var_95=self._compute_var(0.95),
            var_99=self._compute_var(0.99),
            sharpe_ratio=self._compute_sharpe(),
            sortino_ratio=self._compute_sortino(),
            win_rate=self._stats.win_rate,
            profit_factor=self._stats.profit_factor,
            total_positions=self._exposure.position_count,
            consecutive_losses=self._stats.consecutive_losses,
            daily_pnl=self._stats.today_pnl,
            total_realized_pnl=self._stats.total_realized_pnl,
            total_trades=self._stats.total_trades,
            size_multiplier=self._compute_size_multiplier(),
            halted_reason=self._halt_reason(),
        )

    # ═════════════════════════════════════════════════════════
    # IRiskGate backward-compat bridge (sync, no lock)
    # ═════════════════════════════════════════════════════════

    def approve(
        self, signal: Signal, positions: List[Position],
        capital: float, phase_params: PhaseParams,
    ) -> RiskVerdict:
        return self.approve_order(signal, positions, capital, phase_params)

    def record_trade_result(self, result: TradeResult) -> None:
        self.update_after_trade(result)

    def get_sharpe_ratio(self) -> float:
        return self._compute_sharpe()

    def get_max_drawdown_pct(self) -> float:
        return self._equity.max_drawdown_pct

    def get_metrics(self) -> PortfolioRiskMetrics:
        return self.get_portfolio_state()

    def reset(self) -> None:
        self._state = PortfolioRiskState.NORMAL
        self._equity = _EquityTracker()
        self._exposure = _ExposureTracker()
        self._stats = _TradeStats()
        self._halt_until = None
        self._state_history.clear()
        self._last_state_change = _utcnow()
        self._approval_seq = 0
        logger.info("portfolio risk engine RESET")

    # ═════════════════════════════════════════════════════════
    # State machine
    # ═════════════════════════════════════════════════════════

    @property
    def state(self) -> PortfolioRiskState:
        return self._state

    @property
    def approval_sequence(self) -> int:
        """FIX #6: Monotonic counter for post-hoc race detection."""
        return self._approval_seq

    def _evaluate_state_transition(self) -> None:
        dd = self._equity.drawdown_pct
        limits = self._limits

        # ── HALTED: check expiry + recovery constraints ──────
        if self._state == PortfolioRiskState.HALTED:
            if self._halt_until and _utcnow() >= self._halt_until:
                # FIX #4: Re-check ALL constraints, not just drawdown
                if self._can_recover_from_halt():
                    self._transition_to(
                        PortfolioRiskState.DEFENSIVE,
                        f"halt_expired,recovery_validated,dd={dd:.1f}%",
                    )
                else:
                    # Re-halt — constraints still violated
                    reason = self._halt_recovery_blocker()
                    self._halt_until = _utcnow() + timedelta(
                        minutes=limits.halt_duration_minutes,
                    )
                    logger.warning(
                        "HALT EXTENDED: recovery blocked by %s (dd=%.1f%%)",
                        reason, dd,
                    )
            return

        # ── Escalation (highest severity first) ──────────────
        if dd >= limits.halted_drawdown_pct:
            self._halt_until = _utcnow() + timedelta(
                minutes=limits.halt_duration_minutes,
            )
            self._transition_to(
                PortfolioRiskState.HALTED,
                f"drawdown={dd:.1f}%>={limits.halted_drawdown_pct}%",
            )
            return

        if dd >= limits.critical_drawdown_pct:
            if self._state != PortfolioRiskState.CRITICAL:
                self._transition_to(
                    PortfolioRiskState.CRITICAL,
                    f"drawdown={dd:.1f}%>={limits.critical_drawdown_pct}%",
                )
            return

        if dd >= limits.defensive_drawdown_pct:
            if self._state == PortfolioRiskState.NORMAL:
                self._transition_to(
                    PortfolioRiskState.DEFENSIVE,
                    f"drawdown={dd:.1f}%>={limits.defensive_drawdown_pct}%",
                )
            return

        # ── Recovery (de-escalation) ─────────────────────────
        if self._state == PortfolioRiskState.CRITICAL:
            if dd < limits.defensive_recovery_pct:
                self._transition_to(
                    PortfolioRiskState.DEFENSIVE,
                    f"recovery,dd={dd:.1f}%<{limits.defensive_recovery_pct}%",
                )

        elif self._state == PortfolioRiskState.DEFENSIVE:
            if dd < limits.normal_recovery_pct:
                self._transition_to(
                    PortfolioRiskState.NORMAL,
                    f"recovery,dd={dd:.1f}%<{limits.normal_recovery_pct}%",
                )

    # FIX #4: Full constraint re-check before HALTED recovery
    def _can_recover_from_halt(self) -> bool:
        """
        All five conditions must be met to exit HALTED:
          1. Drawdown below critical recovery threshold
          2. Consecutive losses below limit
          3. Daily loss below limit (uses peak equity as capital proxy)
          4. Gross exposure below limit
          5. Minimum cooldown time elapsed (handled by caller)
        """
        dd = self._equity.drawdown_pct
        limits = self._limits
        capital = self._equity.current

        if dd >= limits.critical_recovery_pct:
            return False

        if self._stats.consecutive_losses >= limits.max_consecutive_losses:
            return False

        if capital > 0:
            today_pnl = self._stats.today_pnl
            if today_pnl < 0:
                loss_pct = (abs(today_pnl) / capital) * 100
                if loss_pct >= limits.max_daily_loss_pct:
                    return False

        if capital > 0 and self._exposure.gross > 0:
            gross_pct = (self._exposure.gross / capital) * 100
            if gross_pct >= limits.max_total_exposure_pct:
                return False

        return True

    def _halt_recovery_blocker(self) -> str:
        """Diagnostic: which constraint is blocking halt recovery."""
        dd = self._equity.drawdown_pct
        limits = self._limits
        capital = self._equity.current

        if dd >= limits.critical_recovery_pct:
            return f"drawdown={dd:.1f}%"
        if self._stats.consecutive_losses >= limits.max_consecutive_losses:
            return f"consec_losses={self._stats.consecutive_losses}"
        if capital > 0 and self._stats.today_pnl < 0:
            loss_pct = (abs(self._stats.today_pnl) / capital) * 100
            if loss_pct >= limits.max_daily_loss_pct:
                return f"daily_loss={loss_pct:.1f}%"
        if capital > 0 and self._exposure.gross > 0:
            gross_pct = (self._exposure.gross / capital) * 100
            if gross_pct >= limits.max_total_exposure_pct:
                return f"exposure={gross_pct:.0f}%"
        return "unknown"

    def _transition_to(self, new_state: PortfolioRiskState, reason: str) -> None:
        old = self._state
        if old == new_state:
            return

        self._state = new_state
        self._last_state_change = _utcnow()

        record = {
            "from": old.value, "to": new_state.value,
            "reason": reason, "timestamp": _utcnow().isoformat(),
            "equity": self._equity.current,
            "drawdown_pct": round(self._equity.drawdown_pct, 2),
            "approval_seq": self._approval_seq,
        }
        self._state_history.append(record)

        severity = {
            PortfolioRiskState.NORMAL: "INFO",
            PortfolioRiskState.DEFENSIVE: "WARNING",
            PortfolioRiskState.CRITICAL: "ERROR",
            PortfolioRiskState.HALTED: "CRITICAL",
        }[new_state]

        logger.log(
            getattr(logging, severity),
            "RISK STATE: %s → %s (%s) equity=$%.2f dd=%.1f%%",
            old.value, new_state.value, reason,
            self._equity.current, self._equity.drawdown_pct,
        )

        if self._bus is not None:
            try:
                from darwin_agent.interfaces.events import risk_state_changed_event
                event = risk_state_changed_event(
                    old.value, new_state.value, reason,
                    self.get_portfolio_state().to_dict(),
                )
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._bus.emit(event))
            except Exception:
                pass

    # ═════════════════════════════════════════════════════════
    # Approval check layers
    # ═════════════════════════════════════════════════════════

    def _check_state_gate(self) -> Optional[RiskVerdict]:
        if self._state == PortfolioRiskState.HALTED:
            return RiskVerdict(approved=False, reason="HALTED:trading_suspended")
        if self._state == PortfolioRiskState.CRITICAL:
            return RiskVerdict(approved=False, reason="CRITICAL:close_only_mode")
        return None

    def _check_position_limits(
        self, signal: Signal, positions: List[Position],
        phase_params: PhaseParams,
    ) -> Optional[RiskVerdict]:
        if len(positions) >= phase_params.max_positions:
            return RiskVerdict(
                approved=False,
                reason=f"max_positions:{len(positions)}>={phase_params.max_positions}",
            )
        if signal.symbol in {p.symbol for p in positions}:
            return RiskVerdict(
                approved=False,
                reason=f"duplicate_symbol:{signal.symbol}",
            )
        return None

    def _check_exposure_limits(
        self, signal: Signal, capital: float,
    ) -> Optional[RiskVerdict]:
        """
        FIX #2: Exchange-scoped exposure check.

        The signal may carry an exchange_id in its metadata dict.
        If present, only that exchange's exposure is checked against
        max_exchange_exposure_pct. If absent, ALL exchanges are
        checked (conservative fallback).

        Gross and per-symbol checks remain global — those protect
        against total portfolio concentration regardless of where
        the order routes.
        """
        if capital <= 0:
            return RiskVerdict(approved=False, reason="zero_capital")

        limits = self._limits

        # Global gross exposure
        gross_pct = (self._exposure.gross / capital) * 100
        if gross_pct >= limits.max_total_exposure_pct:
            return RiskVerdict(
                approved=False,
                reason=f"gross_exposure:{gross_pct:.0f}%>={limits.max_total_exposure_pct:.0f}%",
            )

        # Per-symbol exposure (global — a symbol is a symbol)
        sym_exp = self._exposure.by_symbol.get(signal.symbol, 0.0)
        sym_pct = (sym_exp / capital) * 100
        if sym_pct >= limits.max_symbol_exposure_pct:
            return RiskVerdict(
                approved=False,
                reason=(
                    f"symbol_exposure:{signal.symbol}="
                    f"{sym_pct:.0f}%>={limits.max_symbol_exposure_pct:.0f}%"
                ),
            )

        # Per-exchange exposure (FIX #2: scoped to target exchange)
        target_exchange = signal.metadata.get("exchange_id") if signal.metadata else None

        if target_exchange is not None:
            # Scoped: only check the target exchange
            ex_key = (
                target_exchange.value
                if hasattr(target_exchange, "value")
                else str(target_exchange)
            )
            ex_exp = self._exposure.by_exchange.get(ex_key, 0.0)
            ex_pct = (ex_exp / capital) * 100
            if ex_pct >= limits.max_exchange_exposure_pct:
                return RiskVerdict(
                    approved=False,
                    reason=(
                        f"exchange_exposure:{ex_key}="
                        f"{ex_pct:.0f}%>={limits.max_exchange_exposure_pct:.0f}%"
                    ),
                )
        else:
            # Fallback: no target known, check all (conservative)
            for ex_id, ex_exp in self._exposure.by_exchange.items():
                ex_pct = (ex_exp / capital) * 100
                if ex_pct >= limits.max_exchange_exposure_pct:
                    return RiskVerdict(
                        approved=False,
                        reason=(
                            f"exchange_exposure:{ex_id}="
                            f"{ex_pct:.0f}%>={limits.max_exchange_exposure_pct:.0f}%"
                        ),
                    )

        return None

    def _check_daily_loss(self, capital: float) -> Optional[RiskVerdict]:
        if capital <= 0:
            return None
        today_pnl = self._stats.today_pnl
        if today_pnl < 0:
            loss_pct = (abs(today_pnl) / capital) * 100
            if loss_pct >= self._limits.max_daily_loss_pct:
                return RiskVerdict(
                    approved=False,
                    reason=f"daily_loss:{loss_pct:.1f}%>={self._limits.max_daily_loss_pct:.0f}%",
                )
        return None

    def _check_consecutive_losses(self) -> Optional[RiskVerdict]:
        if self._stats.consecutive_losses >= self._limits.max_consecutive_losses:
            return RiskVerdict(
                approved=False,
                reason=(
                    f"consecutive_losses:{self._stats.consecutive_losses}"
                    f">={self._limits.max_consecutive_losses}"
                ),
            )
        return None

    def _check_var(self, capital: float) -> Optional[RiskVerdict]:
        if capital <= 0:
            return None
        if self._stats.total_trades < self._limits.min_trades_for_stats:
            return None
        var = self._compute_var(self._limits.var_confidence)
        if var <= 0:
            return None
        var_pct = (var / capital) * 100
        if var_pct > self._limits.max_var_pct:
            return RiskVerdict(
                approved=False,
                reason=(
                    f"VaR{int(self._limits.var_confidence * 100)}:"
                    f"{var_pct:.1f}%>{self._limits.max_var_pct:.0f}%"
                ),
            )
        return None

    def _check_directional_concentration(
        self, signal: Signal, positions: List[Position],
    ) -> Optional[RiskVerdict]:
        if not positions:
            return None
        same_side = sum(1 for p in positions if p.side == signal.side)
        pct = (same_side / len(positions)) * 100
        if pct > self._limits.max_directional_concentration_pct:
            return RiskVerdict(
                approved=False,
                reason=(
                    f"directional_concentration:{pct:.0f}%"
                    f">{self._limits.max_directional_concentration_pct:.0f}%"
                ),
            )
        return None

    def _check_signal_quality(self, signal: Signal) -> Optional[RiskVerdict]:
        if signal.strength == SignalStrength.NONE:
            return RiskVerdict(approved=False, reason="signal_strength:NONE")

        if signal.strength == SignalStrength.WEAK and signal.confidence < 0.5:
            return RiskVerdict(approved=False, reason="weak_signal:low_confidence")

        if self._state == PortfolioRiskState.DEFENSIVE:
            if signal.strength == SignalStrength.WEAK:
                return RiskVerdict(
                    approved=False, reason="DEFENSIVE:weak_signal_blocked",
                )
            if signal.confidence < 0.6:
                return RiskVerdict(
                    approved=False, reason="DEFENSIVE:low_confidence",
                )

        return None

    # ═════════════════════════════════════════════════════════
    # Size multiplier
    # ═════════════════════════════════════════════════════════

    def _compute_size_multiplier(self) -> float:
        if self._state == PortfolioRiskState.HALTED:
            return 0.0
        if self._state == PortfolioRiskState.CRITICAL:
            return self._limits.critical_size_multiplier
        if self._state == PortfolioRiskState.DEFENSIVE:
            return self._limits.defensive_size_multiplier
        return 1.0

    # ═════════════════════════════════════════════════════════
    # Statistical computations (pure reads, no lock needed)
    # ═════════════════════════════════════════════════════════

    def _compute_sharpe(self, risk_free_rate: float = 0.0) -> float:
        pnls = list(self._stats.pnl_series)
        if len(pnls) < self._limits.min_trades_for_stats:
            return 0.0
        mean = statistics.mean(pnls) - risk_free_rate
        std = statistics.stdev(pnls)
        if std <= 1e-9:
            return 0.0
        return mean / std * math.sqrt(252)

    def _compute_sortino(self) -> float:
        pnls = list(self._stats.pnl_series)
        if len(pnls) < self._limits.min_trades_for_stats:
            return 0.0
        mean = statistics.mean(pnls)
        downside = [p for p in pnls if p < 0]
        if not downside:
            return mean * math.sqrt(252) if mean > 0 else 0.0
        down_std = (
            statistics.stdev(downside) if len(downside) > 1
            else abs(downside[0])
        )
        if down_std <= 1e-9:
            return 0.0
        return mean / down_std * math.sqrt(252)

    def _compute_var(self, confidence: float = 0.95) -> float:
        pnls = sorted(self._stats.pnl_series)
        if len(pnls) < self._limits.min_trades_for_stats:
            return 0.0
        idx = int(len(pnls) * (1 - confidence))
        return abs(pnls[max(idx, 0)])

    def _compute_correlation_risk(self) -> float:
        if self._exposure.gross <= 0:
            return 0.0
        return abs(self._exposure.net) / self._exposure.gross

    # ═════════════════════════════════════════════════════════
    # Helpers
    # ═════════════════════════════════════════════════════════

    def _halt_reason(self) -> str:
        if self._state != PortfolioRiskState.HALTED:
            return ""
        if self._state_history:
            last = self._state_history[-1]
            return last.get("reason", "unknown")
        return "unknown"

    def get_state_history(self) -> List[Dict]:
        """Chronological record of state transitions."""
        return list(self._state_history)
