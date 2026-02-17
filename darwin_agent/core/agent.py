"""
Darwin v4 — DarwinAgent — Production Hardened.
Layer 6 (core). Depends ONLY on interfaces/.
All concrete deps injected via constructor.

AUDIT FINDINGS:
  H-AG1  _closed_trades is an unbounded List[TradeResult]. After thousands
         of trades it grows indefinitely — pure memory leak.
         FIX: deque(maxlen=500).

  H-AG2  _die can be called multiple times — from tick (hp depleted),
         from kill(), from _complete_generation(). Each call iterates
         _positions and emits AGENT_DIED event. Double-die emits
         duplicate events and double-reclaims capital.
         FIX: Guard with _dying flag checked at entry.

  H-AG3  _manage_positions modifies _positions while iterating via
         `closed.append + pop`. This is safe in CPython but fragile.
         Also: if _close_position raises, the symbol stays in _positions
         but the position state is corrupted.
         FIX: Collect close targets, then process sequentially with
         error handling that ensures cleanup.

  H-AG4  No guard against re-opening a position on a symbol that is
         currently being closed. If _close_position awaits the exchange
         and another tick runs before the await completes, the symbol
         is still in _positions but its state is mid-close.
         FIX: _closing_symbols set prevents re-entry.

  H-AG5  get_metrics calls self._risk_gate.get_sharpe_ratio() which
         may be expensive. If called frequently (every heartbeat for
         every agent), this compounds.
         FIX: Cache the value per-tick. Acceptable staleness: 1 tick.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Deque, Dict, List, Optional, Set

from darwin_agent.interfaces.enums import (
    AgentPhase, EventType, GrowthPhase, OrderSide, OrderType,
    SignalStrength, StrategyID, TimeFrame,
)
from darwin_agent.interfaces.types import (
    AgentMetrics, AllocationSlice, Candle, OrderRequest, OrderResult,
    PhaseParams, Position, RiskVerdict, Signal, TradeResult,
)
from darwin_agent.interfaces.events import (
    Event, agent_died_event, agent_spawned_event,
    generation_complete_event, trade_closed_event, trade_opened_event,
)
from darwin_agent.interfaces.protocols import (
    IEventBus, IExchangeAdapter, IFeatureEngine,
    IPositionSizer, IRiskGate, IStrategy, IStrategySelector,
)
from darwin_agent.evolution.fitness import RiskAwareFitness, compute_fitness

logger = logging.getLogger("darwin.agent")

MAX_LEVERAGE = 5

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class AgentConfig:
    watchlist: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    scan_timeframes: List[TimeFrame] = field(default_factory=lambda: [TimeFrame.M15])
    incubation_candles: int = 100
    min_graduation_winrate: float = 0.50
    generation_trade_limit: int = 50
    max_consecutive_losses: int = 5
    cooldown_base_minutes: float = 15.0
    default_stop_loss_pct: float = 1.5
    default_take_profit_pct: float = 3.0
    trailing_stop_pct: float = 1.0
    trailing_activation_pct: float = 1.5


class DarwinAgent:
    """
    Autonomous trading agent. ALL deps injected.
    Production hardened with bounded collections, die-once guard,
    position leak protection, and double-close prevention.
    """

    __slots__ = (
        "agent_id", "generation", "dna_genes",
        "_exchange", "_features", "_selector", "_strategies",
        "_risk_gate", "_sizer", "_bus", "_cfg",
        "_phase", "_born_at", "_died_at", "_death_cause",
        "_initial_capital", "_capital", "_allocation",
        "_positions", "_closed_trades", "_realized_pnl", "_peak_capital",
        "_cycle", "_gen_trades", "_inc_candles", "_inc_wins",
        "_wins", "_losses", "_consec_losses", "_cooldown_until",
        "_hp", "_max_hp", "_last_features",
        "_dying", "_closing_symbols", "_cached_sharpe",
    )

    def __init__(
        self, *,
        agent_id: str | None = None,
        generation: int = 0,
        dna_genes: Dict[str, float] | None = None,
        initial_capital: float = 50.0,
        config: AgentConfig | None = None,
        exchange: IExchangeAdapter,
        feature_engine: IFeatureEngine,
        selector: IStrategySelector,
        strategies: Dict[StrategyID, IStrategy],
        risk_gate: IRiskGate,
        sizer: IPositionSizer,
        bus: IEventBus,
    ) -> None:
        self.agent_id = agent_id or f"agent-{uuid.uuid4().hex[:8]}"
        self.generation = generation
        self.dna_genes = dna_genes or {}
        self._exchange = exchange
        self._features = feature_engine
        self._selector = selector
        self._strategies = strategies
        self._risk_gate = risk_gate
        self._sizer = sizer
        self._bus = bus
        self._cfg = config or AgentConfig()
        self._phase = AgentPhase.INCUBATION
        self._born_at = _utcnow()
        self._died_at: Optional[datetime] = None
        self._death_cause = ""
        self._initial_capital = initial_capital
        self._capital = initial_capital
        self._peak_capital = initial_capital
        self._allocation: Optional[AllocationSlice] = None
        self._positions: Dict[str, Position] = {}
        # H-AG1: bounded trade history
        self._closed_trades: Deque[TradeResult] = deque(maxlen=500)
        self._realized_pnl = 0.0
        self._cycle = 0
        self._gen_trades = 0
        self._inc_candles = 0
        self._inc_wins = 0
        self._wins = 0
        self._losses = 0
        self._consec_losses = 0
        self._cooldown_until: Optional[datetime] = None
        self._hp = 100.0
        self._max_hp = 100.0
        self._last_features: List[float] = []
        # H-AG2: die-once guard
        self._dying = False
        # H-AG4: double-close prevention
        self._closing_symbols: Set[str] = set()
        # H-AG5: cached sharpe
        self._cached_sharpe: float = 0.0

    # ── Properties ───────────────────────────────────────────

    @property
    def phase(self) -> AgentPhase:
        return self._phase

    @property
    def is_alive(self) -> bool:
        return self._phase not in (AgentPhase.DEAD, AgentPhase.DYING)

    @property
    def capital(self) -> float:
        return self._capital

    @property
    def hp(self) -> float:
        return self._hp

    @property
    def phase_params(self) -> PhaseParams:
        if self._allocation:
            return self._allocation.phase_params
        return PhaseParams(phase=GrowthPhase.BOOTSTRAP, risk_pct=2.0,
                           max_positions=2, leverage=10)

    # ── Main tick ────────────────────────────────────────────

    async def tick(self) -> None:
        if not self.is_alive:
            return
        self._cycle += 1
        try:
            if self._hp <= 0 or self._capital <= self._initial_capital * 0.1:
                await self._die("hp_depleted" if self._hp <= 0 else "capital_depleted")
                return
            if self._gen_trades >= self._cfg.generation_trade_limit:
                await self._complete_generation()
                return
            await self._manage_positions()
            if self._is_on_cooldown():
                return
            if self._phase == AgentPhase.INCUBATION:
                await self._tick_incubation()
            elif self._phase == AgentPhase.LIVE:
                await self._tick_live()
            # H-AG5: refresh cached sharpe once per tick
            self._cached_sharpe = self._risk_gate.get_sharpe_ratio()
        except Exception as exc:
            logger.error("agent %s tick error: %s", self.agent_id, exc, exc_info=True)
            self._damage(2.0)

    # ── Incubation ───────────────────────────────────────────

    async def _tick_incubation(self) -> None:
        for symbol in self._cfg.watchlist:
            for tf in self._cfg.scan_timeframes:
                try:
                    candles = await self._exchange.get_candles(symbol, tf, limit=100)
                    if len(candles) < 30:
                        continue
                    self._last_features = self._features.extract(candles)
                    self._inc_candles += 1
                    sid = self._selector.select(self._last_features)
                    strat = self._strategies.get(sid)
                    if strat:
                        sig = strat.analyze(symbol, candles, self._last_features)
                        if sig and sig.confidence > 0.6:
                            self._inc_wins += 1
                except Exception:
                    pass

        if self._inc_candles >= self._cfg.incubation_candles:
            wr = self._inc_wins / max(self._inc_candles, 1)
            if wr >= self._cfg.min_graduation_winrate:
                self._set_phase(AgentPhase.LIVE)
            else:
                await self._die("incubation_failed")

    # ── Live trading ─────────────────────────────────────────

    async def _tick_live(self) -> None:
        params = self.phase_params
        for symbol in self._cfg.watchlist:
            if symbol in self._positions:
                continue
            # H-AG4: skip if currently closing on this symbol
            if symbol in self._closing_symbols:
                continue
            if len(self._positions) >= params.max_positions:
                break
            for tf in self._cfg.scan_timeframes:
                sig = await self._gen_signal(symbol, tf)
                if sig is None or sig.strength == SignalStrength.NONE:
                    continue
                verdict = self._risk_gate.approve(
                    sig, list(self._positions.values()), self._capital, params)
                if not verdict.approved:
                    continue
                try:
                    ticker = await self._exchange.get_ticker(symbol)
                    price = ticker.last_price
                except Exception:
                    continue
                qty = self._sizer.calculate(sig, self._capital, price, params)
                if qty <= 0:
                    continue
                if verdict.adjusted_size is not None:
                    qty = min(qty, verdict.adjusted_size)
                await self._open_trade(symbol, sig, qty, price, params)
                break

    async def _gen_signal(self, symbol: str, tf: TimeFrame) -> Optional[Signal]:
        try:
            candles = await self._exchange.get_candles(symbol, tf, limit=100)
            if len(candles) < 30:
                return None
            self._last_features = self._features.extract(candles)
            sid = self._selector.select(self._last_features)
            strat = self._strategies.get(sid)
            return strat.analyze(symbol, candles, self._last_features) if strat else None
        except Exception:
            return None

    # ── Trade execution ──────────────────────────────────────

    async def _open_trade(self, symbol, sig, qty, price, params):
        effective_leverage = min(int(params.leverage), MAX_LEVERAGE)
        try:
            await self._exchange.set_leverage(symbol, effective_leverage)
        except Exception:
            pass
        sl = price * (1 - sig.stop_loss_pct / 100) if sig.side == OrderSide.BUY else \
             price * (1 + sig.stop_loss_pct / 100)
        tp = price * (1 + sig.take_profit_pct / 100) if sig.side == OrderSide.BUY else \
             price * (1 - sig.take_profit_pct / 100)

        result = await self._exchange.place_order(OrderRequest(
            symbol=symbol, side=sig.side, order_type=OrderType.MARKET,
            quantity=qty, stop_loss=round(sl, 2), take_profit=round(tp, 2),
            leverage=effective_leverage, agent_id=self.agent_id,
            metadata={"strategy": sig.strategy.value},
        ))
        if not result.success:
            self._damage(1.0)
            return

        self._positions[symbol] = Position(
            symbol=symbol, side=sig.side, size=result.filled_qty,
            entry_price=result.filled_price, current_price=result.filled_price,
            leverage=effective_leverage, stop_loss=round(sl, 2),
            take_profit=round(tp, 2), agent_id=self.agent_id,
        )
        self._gen_trades += 1

        await self._bus.emit(trade_opened_event(
            self.agent_id, symbol, sig.side.value, result.filled_qty,
            result.filled_price, effective_leverage, sig.strategy.value,
        ))

    # ── Position management ──────────────────────────────────

    async def _manage_positions(self) -> None:
        # H-AG3: collect close targets first, process sequentially
        to_close: List[tuple] = []  # (symbol, pos, reason)

        for symbol, pos in self._positions.items():
            # H-AG4: skip if already being closed
            if symbol in self._closing_symbols:
                continue
            try:
                ticker = await self._exchange.get_ticker(symbol)
                pos.current_price = ticker.last_price
            except Exception:
                continue
            pnl_pct = self._pnl_pct(pos)
            pos.unrealized_pnl = self._unrealized(pos)
            reason = ""
            liq = -90.0 / max(pos.leverage, 1)
            if pnl_pct <= liq:
                reason = "liquidation_protection"
            elif pos.stop_loss and (
                (pos.side == OrderSide.BUY and pos.current_price <= pos.stop_loss) or
                (pos.side == OrderSide.SELL and pos.current_price >= pos.stop_loss)):
                reason = "stop_loss"
            elif pos.take_profit and (
                (pos.side == OrderSide.BUY and pos.current_price >= pos.take_profit) or
                (pos.side == OrderSide.SELL and pos.current_price <= pos.take_profit)):
                reason = "take_profit"
            if not reason and pnl_pct >= self._cfg.trailing_activation_pct:
                trail = pos.current_price * (1 - self._cfg.trailing_stop_pct / 100) \
                    if pos.side == OrderSide.BUY else \
                    pos.current_price * (1 + self._cfg.trailing_stop_pct / 100)
                if pos.trailing_stop_price is None:
                    pos.trailing_stop_price = trail
                else:
                    if pos.side == OrderSide.BUY:
                        pos.trailing_stop_price = max(pos.trailing_stop_price, trail)
                        if pos.current_price <= pos.trailing_stop_price:
                            reason = "trailing_stop"
                    else:
                        pos.trailing_stop_price = min(pos.trailing_stop_price, trail)
                        if pos.current_price >= pos.trailing_stop_price:
                            reason = "trailing_stop"
            if reason:
                to_close.append((symbol, pos, reason))

        # H-AG3 + H-AG4: close sequentially with guard
        for symbol, pos, reason in to_close:
            self._closing_symbols.add(symbol)
            try:
                await self._close_position(pos, reason)
            except Exception as exc:
                logger.error(
                    "agent %s close_position %s failed: %s",
                    self.agent_id, symbol, exc,
                )
            finally:
                self._positions.pop(symbol, None)
                self._closing_symbols.discard(symbol)

    async def _close_position(self, pos, reason):
        try:
            result = await self._exchange.close_position(pos.symbol, pos.side)
            exit_price = result.filled_price if result.success else pos.current_price
            fee = result.fee
        except Exception:
            exit_price = pos.current_price
            fee = 0.0
        realized = self._realized(pos, exit_price)
        trade = TradeResult(
            symbol=pos.symbol, side=pos.side, strategy=StrategyID.MOMENTUM,
            entry_price=pos.entry_price, exit_price=exit_price,
            quantity=pos.size, realized_pnl=realized, fee=fee,
            leverage=pos.leverage, opened_at=pos.opened_at,
            closed_at=_utcnow(), close_reason=reason, agent_id=self.agent_id,
        )
        self._on_trade_closed(trade)
        await self._bus.emit(trade_closed_event(self.agent_id, trade))

    # ── Trade close handler ──────────────────────────────────

    def _on_trade_closed(self, trade: TradeResult) -> None:
        pnl = trade.realized_pnl
        self._realized_pnl += pnl
        self._capital += pnl
        self._peak_capital = max(self._peak_capital, self._capital)
        self._closed_trades.append(trade)  # H-AG1: deque auto-bounds
        if pnl >= 0:
            self._wins += 1
            self._consec_losses = 0
            self._hp = min(self._max_hp, self._hp + min(pnl * 0.5, 5.0))
        else:
            self._losses += 1
            self._consec_losses += 1
            self._damage(min(abs(pnl) * 1.5, 15.0))
        if self._consec_losses >= self._cfg.max_consecutive_losses:
            esc = 1 + (self._consec_losses - self._cfg.max_consecutive_losses) * 0.5
            self._cooldown_until = _utcnow() + timedelta(
                minutes=self._cfg.cooldown_base_minutes * esc)
            self._consec_losses = 0
        self._selector.update(trade.strategy, 1.0 if pnl > 0 else -1.0)
        self._risk_gate.record_trade_result(trade)

    # ── Lifecycle ────────────────────────────────────────────

    def _set_phase(self, p: AgentPhase):
        self._phase = p

    async def _die(self, cause: str):
        # H-AG2: die-once guard
        if self._dying:
            return
        self._dying = True

        self._death_cause = cause
        self._set_phase(AgentPhase.DYING)
        for symbol, pos in list(self._positions.items()):
            try:
                await self._close_position(pos, f"dying:{cause}")
            except Exception:
                pass
        self._positions.clear()
        self._closing_symbols.clear()
        self._died_at = _utcnow()
        self._set_phase(AgentPhase.DEAD)
        await self._bus.emit(agent_died_event(
            self.agent_id, self.generation, cause, self.get_metrics()))

    async def _complete_generation(self):
        await self._bus.emit(generation_complete_event(
            self.agent_id, self.generation, self.get_metrics()))
        await self._die("generation_complete")

    async def kill(self, reason: str = "external_kill"):
        if self.is_alive:
            await self._die(reason)

    def update_allocation(self, alloc: AllocationSlice):
        self._allocation = alloc

    # ── Metrics ──────────────────────────────────────────────

    def get_metrics(self) -> AgentMetrics:
        total = self._wins + self._losses
        unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        dd = ((self._peak_capital - self._capital) / max(self._peak_capital, 0.01)) * 100 \
            if self._peak_capital > 0 else 0.0
        return AgentMetrics(
            agent_id=self.agent_id, generation=self.generation,
            phase=self._phase.value, capital=round(self._capital, 2),
            realized_pnl=round(self._realized_pnl, 2),
            unrealized_pnl=round(unrealized, 2),
            total_trades=total, winning_trades=self._wins,
            losing_trades=self._losses,
            open_positions=len(self._positions),
            # H-AG5: use cached sharpe
            sharpe_ratio=self._cached_sharpe,
            max_drawdown_pct=round(dd, 2),
            fitness=self._fitness(total, dd), hp=round(self._hp, 1),
            consecutive_losses=self._consec_losses,
            uptime_seconds=(_utcnow() - self._born_at).total_seconds(),
        )

    # ── Helpers ──────────────────────────────────────────────

    @staticmethod
    def _pnl_pct(pos):
        if pos.entry_price == 0: return 0.0
        r = (pos.current_price - pos.entry_price) / pos.entry_price
        if pos.side == OrderSide.SELL: r = -r
        return r * pos.leverage * 100

    @staticmethod
    def _unrealized(pos):
        d = pos.current_price - pos.entry_price
        if pos.side == OrderSide.SELL: d = -d
        return d * pos.size

    @staticmethod
    def _realized(pos, exit_price):
        d = exit_price - pos.entry_price
        if pos.side == OrderSide.SELL: d = -d
        return d * pos.size

    def _damage(self, amount):
        self._hp = max(0.0, self._hp - amount)

    def _is_on_cooldown(self):
        if not self._cooldown_until: return False
        if _utcnow() < self._cooldown_until: return True
        self._cooldown_until = None
        return False

    def _fitness(self, total_trades, drawdown):
        """Risk-aware fitness: integrates portfolio state, consistency,
        diversification, and capital efficiency."""
        if total_trades == 0:
            return 0.0

        # Build PnL series from closed trades (deque → list)
        pnl_series = [t.realized_pnl for t in self._closed_trades]

        # Get portfolio snapshot from risk gate (if it supports it)
        portfolio_snapshot = None
        try:
            portfolio_snapshot = self._risk_gate.get_metrics()
        except Exception:
            pass

        # Agent's own exposure by symbol
        agent_exposure = {}
        if self._positions and self._capital > 0:
            for sym, pos in self._positions.items():
                notional = pos.size * pos.current_price
                agent_exposure[sym] = notional / max(self._capital, 0.01)

        score = compute_fitness(
            realized_pnl=self._realized_pnl,
            initial_capital=self._initial_capital,
            current_capital=self._capital,
            sharpe=self._cached_sharpe,
            max_drawdown_pct=drawdown,
            win_count=self._wins,
            loss_count=self._losses,
            pnl_series=pnl_series,
            portfolio_snapshot=portfolio_snapshot,
            agent_exposure=agent_exposure,
        )
        return round(score, 4)
