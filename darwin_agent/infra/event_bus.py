"""
Darwin v4 — EventBus (Layer 1) — Production Hardened.

AUDIT FINDINGS:
  H-EB1  _history is a plain list sliced with `list[-500:]` every emit.
         Creates a NEW list every cycle — O(n) copy + GC churn.
         FIX: deque(maxlen=500). Zero-copy, O(1) append, auto-bounded.

  H-EB2  emit() calls asyncio.gather on live _listeners dict. If a handler
         subscribes/unsubscribes during dispatch, dict mutates mid-iteration.
         FIX: Snapshot listener lists at emit entry.

  H-EB3  No cap on listener growth — subscribe() in a loop can OOM.
         FIX: MAX_LISTENERS_PER_TYPE=50, warning at 80%.

  H-EB4  No graceful shutdown — listeners referencing destroyed objects
         cause cascading errors post-stop.
         FIX: clear() + _stopped flag. Emit after stop is a no-op.
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from typing import Callable, Awaitable, Deque, Dict, List

from darwin_agent.interfaces.events import Event, EventType

Listener = Callable[[Event], Awaitable[None]]
logger = logging.getLogger("darwin.event_bus")

MAX_LISTENERS_PER_TYPE = 50
MAX_HISTORY = 500


class EventBus:
    """
    Central async event dispatcher — production hardened.
    Thread-safe within a single asyncio event loop.
    Bounded memory. Re-entrant safe. Graceful shutdown.
    """

    def __init__(self) -> None:
        self._listeners: Dict[EventType, List[Listener]] = defaultdict(list)
        self._global_listeners: List[Listener] = []
        self._history: Deque[Event] = deque(maxlen=MAX_HISTORY)
        self._event_count: int = 0
        self._error_count: int = 0
        self._stopped: bool = False

    def subscribe(self, event_type: EventType, listener: Listener) -> None:
        bucket = self._listeners[event_type]
        if len(bucket) >= MAX_LISTENERS_PER_TYPE:
            logger.error(
                "listener cap reached for %s (%d), rejected",
                event_type.name, MAX_LISTENERS_PER_TYPE,
            )
            return
        if len(bucket) >= int(MAX_LISTENERS_PER_TYPE * 0.8):
            logger.warning(
                "listener count for %s at %d/%d",
                event_type.name, len(bucket), MAX_LISTENERS_PER_TYPE,
            )
        bucket.append(listener)
        logger.debug("subscribed %s to %s", listener.__qualname__, event_type.name)

    def subscribe_all(self, listener: Listener) -> None:
        if len(self._global_listeners) >= MAX_LISTENERS_PER_TYPE:
            logger.error("global listener cap reached, rejected")
            return
        self._global_listeners.append(listener)

    def unsubscribe(self, event_type: EventType, listener: Listener) -> None:
        try:
            self._listeners[event_type].remove(listener)
        except ValueError:
            pass

    async def emit(self, event: Event) -> None:
        if self._stopped:
            return
        self._event_count += 1
        self._history.append(event)

        # H-EB2: snapshot so mutations during dispatch are safe
        typed = list(self._listeners.get(event.event_type, []))
        global_ = list(self._global_listeners)
        targets = typed + global_
        if not targets:
            return
        await asyncio.gather(*(self._safe_call(ln, event) for ln in targets))

    async def _safe_call(self, listener: Listener, event: Event) -> None:
        try:
            await listener(event)
        except Exception as exc:
            self._error_count += 1
            logger.error(
                "listener %s failed on %s: %s",
                listener.__qualname__, event.event_type.name, exc,
                exc_info=True,
            )

    def clear(self) -> None:
        """Remove all listeners and stop accepting events."""
        self._stopped = True
        self._listeners.clear()
        self._global_listeners.clear()
        logger.info("event bus cleared and stopped")

    def resume(self) -> None:
        self._stopped = False

    @property
    def stopped(self) -> bool:
        return self._stopped

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "total_events": self._event_count,
            "errors": self._error_count,
            "listeners": sum(len(v) for v in self._listeners.values()) + len(self._global_listeners),
            "stopped": int(self._stopped),
        }

    def get_recent(self, event_type: EventType | None = None, limit: int = 20) -> List[Event]:
        if event_type is None:
            return list(self._history)[-limit:]
        return [e for e in self._history if e.event_type == event_type][-limit:]

    @property
    def subscriber_count(self) -> Dict[str, int]:
        return {et.name: len(ls) for et, ls in self._listeners.items() if ls}
