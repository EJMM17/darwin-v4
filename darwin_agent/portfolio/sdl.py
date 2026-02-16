"""
Darwin v4 — Symbol Drawdown Lock (SDL).

Independent protection layer that locks out new entries for a symbol
when its per-symbol equity drawdown exceeds a threshold. Does NOT
modify RBE, GMRT, PAE, RQRE, CB, or leverage logic.

Architecture:
    Per-symbol equity tracking → peak/current → drawdown check
    If dd >= limit → lock new entries for N hours
    On unlock → reset peak to current equity

Behavior during lock:
    - New entries: BLOCKED
    - Closing positions: ALLOWED
    - Stops: ALLOWED
    - RBE/PAE/GMRT: UNAFFECTED
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, List


@dataclass(frozen=True)
class SDLConfig:
    """Tunable parameters for Symbol Drawdown Lock."""
    symbol_dd_limit: float = 0.40     # 40% peak-to-trough triggers lock
    symbol_lock_hours: int = 24       # lock duration in hours


@dataclass
class SymbolState:
    """Per-symbol tracking state for SDL."""
    peak_equity: float
    current_equity: float
    locked_until_bar: Optional[int] = None

    @property
    def drawdown(self) -> float:
        """Current drawdown as fraction [0, 1]."""
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.current_equity) / self.peak_equity)

    @property
    def is_locked(self) -> bool:
        """Whether this symbol is currently locked (bar check done externally)."""
        return self.locked_until_bar is not None


@dataclass
class SDLEvent:
    """Log entry for SDL triggers and releases."""
    event_type: str   # "TRIGGER" or "RELEASE"
    symbol: str
    bar: int
    drawdown: float = 0.0
    peak: float = 0.0
    current: float = 0.0


class SymbolDrawdownLock:
    """
    Symbol Drawdown Lock engine.

    Usage:
        sdl = SymbolDrawdownLock(["BTC", "SOL", "DOGE"], starting_equity=266.67)
        sdl.update("BTC", equity=250.0, current_bar=100)
        if sdl.allows_new_entry("BTC", current_bar=100):
            # place trade
    """

    def __init__(
        self,
        symbols: List[str],
        starting_equity: float,
        config: SDLConfig = SDLConfig(),
        bars_per_hour: float = 0.25,  # 4H bars → 0.25 bars/hour
    ):
        self.config = config
        self.bars_per_hour = bars_per_hour
        self.lock_bars = int(config.symbol_lock_hours * bars_per_hour)
        self.states: Dict[str, SymbolState] = {
            sym: SymbolState(
                peak_equity=starting_equity,
                current_equity=starting_equity,
            )
            for sym in symbols
        }
        self.events: List[SDLEvent] = []

    def update(self, symbol: str, equity: float, current_bar: int) -> None:
        """
        Update symbol equity and check for lock trigger/release.

        Call AFTER computing symbol PnL each bar.
        """
        state = self.states[symbol]
        state.current_equity = equity

        # Update peak
        if equity > state.peak_equity:
            state.peak_equity = equity

        # Check unlock first
        if state.locked_until_bar is not None and current_bar >= state.locked_until_bar:
            self.events.append(SDLEvent(
                event_type="RELEASE",
                symbol=symbol,
                bar=current_bar,
                drawdown=state.drawdown,
                peak=state.peak_equity,
                current=state.current_equity,
            ))
            state.locked_until_bar = None
            # Reset peak to current on unlock
            state.peak_equity = state.current_equity

        # Check lock trigger (only if not already locked)
        if state.locked_until_bar is None and state.drawdown >= self.config.symbol_dd_limit:
            state.locked_until_bar = current_bar + self.lock_bars
            self.events.append(SDLEvent(
                event_type="TRIGGER",
                symbol=symbol,
                bar=current_bar,
                drawdown=state.drawdown,
                peak=state.peak_equity,
                current=state.current_equity,
            ))

    def allows_new_entry(self, symbol: str, current_bar: int) -> bool:
        """
        Check if new entries are allowed for this symbol.

        Returns True if NOT locked, False if locked.
        Closing positions and stops are always allowed (handled externally).
        """
        state = self.states[symbol]
        if state.locked_until_bar is not None and current_bar < state.locked_until_bar:
            return False
        return True

    def is_locked(self, symbol: str, current_bar: int) -> bool:
        """Convenience: True if symbol is currently locked."""
        return not self.allows_new_entry(symbol, current_bar)

    def get_state(self, symbol: str) -> SymbolState:
        """Get current state for a symbol."""
        return self.states[symbol]

    def get_events(self, symbol: Optional[str] = None) -> List[SDLEvent]:
        """Get all events, optionally filtered by symbol."""
        if symbol is None:
            return list(self.events)
        return [e for e in self.events if e.symbol == symbol]

    def lock_count(self) -> int:
        """Number of currently locked symbols."""
        return sum(1 for s in self.states.values() if s.locked_until_bar is not None)
