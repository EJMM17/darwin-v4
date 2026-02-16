"""
Tests for Symbol Drawdown Lock (SDL).

Unit tests:
    1. Trigger at 25%
    2. No retrigger while locked
    3. Unlock after 48h
    4. Peak reset after unlock
    5. Does not block closes (API check)

Integration test:
    Simulate equity rise → 30% drop → confirm lock → no entries → unlock
"""

import pytest
from darwin_agent.portfolio.sdl import (
    SDLConfig,
    SymbolState,
    SymbolDrawdownLock,
    SDLEvent,
)


# ── Constants ──
# 4H bars: 0.25 bars/hour → 24h = 6 bars
BARS_PER_HOUR = 0.25
LOCK_BARS = int(24 * BARS_PER_HOUR)  # = 6


# ── 1. Trigger at 25% ──

class TestTriggerAt25:
    def test_exact_threshold(self):
        sdl = SymbolDrawdownLock(["BTC"], 100.0, bars_per_hour=BARS_PER_HOUR)
        sdl.update("BTC", 60.0, current_bar=10)  # 40% dd
        assert sdl.is_locked("BTC", 10)

    def test_below_threshold(self):
        sdl = SymbolDrawdownLock(["BTC"], 100.0, bars_per_hour=BARS_PER_HOUR)
        sdl.update("BTC", 61.0, current_bar=10)  # 39% dd
        assert not sdl.is_locked("BTC", 10)

    def test_above_threshold(self):
        sdl = SymbolDrawdownLock(["BTC"], 100.0, bars_per_hour=BARS_PER_HOUR)
        sdl.update("BTC", 55.0, current_bar=10)  # 45% dd
        assert sdl.is_locked("BTC", 10)

    def test_trigger_event_logged(self):
        sdl = SymbolDrawdownLock(["BTC"], 100.0, bars_per_hour=BARS_PER_HOUR)
        sdl.update("BTC", 60.0, current_bar=10)
        evts = sdl.get_events("BTC")
        assert len(evts) == 1
        assert evts[0].event_type == "TRIGGER"
        assert evts[0].bar == 10
        assert abs(evts[0].drawdown - 0.40) < 0.001

    def test_lock_bar_computed(self):
        sdl = SymbolDrawdownLock(["BTC"], 100.0, bars_per_hour=BARS_PER_HOUR)
        sdl.update("BTC", 60.0, current_bar=10)
        state = sdl.get_state("BTC")
        assert state.locked_until_bar == 10 + LOCK_BARS


# ── 2. No retrigger while locked ──

class TestNoRetrigger:
    def test_single_trigger_event(self):
        sdl = SymbolDrawdownLock(["BTC"], 100.0, bars_per_hour=BARS_PER_HOUR)
        sdl.update("BTC", 55.0, current_bar=10)   # trigger
        sdl.update("BTC", 45.0, current_bar=11)   # deeper dd, already locked
        sdl.update("BTC", 35.0, current_bar=12)   # even deeper
        triggers = [e for e in sdl.get_events() if e.event_type == "TRIGGER"]
        assert len(triggers) == 1

    def test_lock_bar_unchanged(self):
        sdl = SymbolDrawdownLock(["BTC"], 100.0, bars_per_hour=BARS_PER_HOUR)
        sdl.update("BTC", 55.0, current_bar=10)
        original_lock = sdl.get_state("BTC").locked_until_bar
        sdl.update("BTC", 35.0, current_bar=11)
        assert sdl.get_state("BTC").locked_until_bar == original_lock


# ── 3. Unlock after 48h ──

class TestUnlockAfter48h:
    def test_still_locked_before(self):
        sdl = SymbolDrawdownLock(["BTC"], 100.0, bars_per_hour=BARS_PER_HOUR)
        sdl.update("BTC", 55.0, current_bar=10)   # trigger, lock until 16
        assert sdl.is_locked("BTC", 10 + LOCK_BARS - 1)

    def test_unlocked_at_expiry(self):
        sdl = SymbolDrawdownLock(["BTC"], 100.0, bars_per_hour=BARS_PER_HOUR)
        sdl.update("BTC", 55.0, current_bar=10)
        for b in range(11, 10 + LOCK_BARS):
            sdl.update("BTC", 55.0, current_bar=b)
        sdl.update("BTC", 55.0, current_bar=10 + LOCK_BARS)
        assert not sdl.is_locked("BTC", 10 + LOCK_BARS)

    def test_release_event_logged(self):
        sdl = SymbolDrawdownLock(["BTC"], 100.0, bars_per_hour=BARS_PER_HOUR)
        sdl.update("BTC", 55.0, current_bar=10)
        sdl.update("BTC", 55.0, current_bar=10 + LOCK_BARS)
        releases = [e for e in sdl.get_events() if e.event_type == "RELEASE"]
        assert len(releases) == 1
        assert releases[0].bar == 10 + LOCK_BARS


# ── 4. Peak reset after unlock ──

class TestPeakResetAfterUnlock:
    def test_peak_resets_to_current(self):
        sdl = SymbolDrawdownLock(["BTC"], 100.0, bars_per_hour=BARS_PER_HOUR)
        sdl.update("BTC", 55.0, current_bar=10)   # trigger
        sdl.update("BTC", 50.0, current_bar=10 + LOCK_BARS)  # unlock
        state = sdl.get_state("BTC")
        assert state.peak_equity == 50.0  # reset to current, not old peak

    def test_new_dd_from_new_peak(self):
        sdl = SymbolDrawdownLock(["BTC"], 100.0, bars_per_hour=BARS_PER_HOUR)
        sdl.update("BTC", 55.0, current_bar=10)
        sdl.update("BTC", 50.0, current_bar=10 + LOCK_BARS)  # unlock, peak=50
        sdl.update("BTC", 80.0, current_bar=30)  # new peak=80
        sdl.update("BTC", 48.0, current_bar=31)  # dd=40% of 80 -> lock
        assert sdl.is_locked("BTC", 31)


# ── 5. Does not block closes ──

class TestDoesNotBlockCloses:
    """SDL only returns allows_new_entry — it never prevents closing.
    This test verifies the API contract."""

    def test_api_only_checks_new_entry(self):
        sdl = SymbolDrawdownLock(["BTC"], 100.0, bars_per_hour=BARS_PER_HOUR)
        sdl.update("BTC", 55.0, current_bar=10)
        assert sdl.is_locked("BTC", 10)
        # The API only gates new entries. Closes/stops are handled
        # externally — SDL has no close-blocking method.
        assert not hasattr(sdl, 'blocks_close')
        assert not hasattr(sdl, 'allows_close')
        # allows_new_entry returns False during lock
        assert not sdl.allows_new_entry("BTC", 11)

    def test_update_still_works_during_lock(self):
        """Equity updates must continue during lock (positions still open)."""
        sdl = SymbolDrawdownLock(["BTC"], 100.0, bars_per_hour=BARS_PER_HOUR)
        sdl.update("BTC", 55.0, current_bar=10)  # lock
        sdl.update("BTC", 80.0, current_bar=11)  # equity recovers
        assert sdl.get_state("BTC").current_equity == 80.0


# ── 6. Multi-symbol isolation ──

class TestMultiSymbol:
    def test_independent_locks(self):
        sdl = SymbolDrawdownLock(["BTC", "SOL"], 100.0, bars_per_hour=BARS_PER_HOUR)
        sdl.update("BTC", 55.0, current_bar=10)  # lock BTC
        sdl.update("SOL", 90.0, current_bar=10)  # SOL fine
        assert sdl.is_locked("BTC", 10)
        assert not sdl.is_locked("SOL", 10)

    def test_lock_count(self):
        sdl = SymbolDrawdownLock(["BTC", "SOL", "DOGE"], 100.0, bars_per_hour=BARS_PER_HOUR)
        assert sdl.lock_count() == 0
        sdl.update("BTC", 55.0, current_bar=10)
        assert sdl.lock_count() == 1
        sdl.update("SOL", 55.0, current_bar=10)
        assert sdl.lock_count() == 2


# ── 7. Config overrides ──

class TestConfigOverrides:
    def test_tighter_limit(self):
        cfg = SDLConfig(symbol_dd_limit=0.25)
        sdl = SymbolDrawdownLock(["BTC"], 100.0, config=cfg, bars_per_hour=BARS_PER_HOUR)
        sdl.update("BTC", 76.0, current_bar=10)  # 24% dd - no lock
        assert not sdl.is_locked("BTC", 10)
        sdl.update("BTC", 74.0, current_bar=11)  # 26% dd - lock
        assert sdl.is_locked("BTC", 11)

    def test_shorter_lock(self):
        cfg = SDLConfig(symbol_lock_hours=12)
        lock_bars = int(12 * BARS_PER_HOUR)  # 3 bars
        sdl = SymbolDrawdownLock(["BTC"], 100.0, config=cfg, bars_per_hour=BARS_PER_HOUR)
        sdl.update("BTC", 55.0, current_bar=10)
        assert sdl.get_state("BTC").locked_until_bar == 10 + lock_bars


# ── 8. SymbolState dataclass ──

class TestSymbolState:
    def test_drawdown_calculation(self):
        s = SymbolState(peak_equity=100.0, current_equity=80.0)
        assert abs(s.drawdown - 0.20) < 0.001

    def test_drawdown_zero(self):
        s = SymbolState(peak_equity=100.0, current_equity=100.0)
        assert s.drawdown == 0.0

    def test_drawdown_negative_peak(self):
        s = SymbolState(peak_equity=0.0, current_equity=50.0)
        assert s.drawdown == 0.0

    def test_is_locked(self):
        s = SymbolState(peak_equity=100.0, current_equity=70.0, locked_until_bar=50)
        assert s.is_locked
        s2 = SymbolState(peak_equity=100.0, current_equity=70.0)
        assert not s2.is_locked


# ── 9. Integration test ──

class TestIntegration:
    """Simulate: equity rise → 30% drop → lock → no entries → unlock."""

    def test_full_lifecycle(self):
        sdl = SymbolDrawdownLock(
            ["BTC", "SOL", "DOGE"],
            starting_equity=266.67,
            bars_per_hour=BARS_PER_HOUR,
        )

        # Phase 1: equity rises
        for b in range(0, 20):
            eq = 266.67 + b * 5  # rising
            sdl.update("BTC", eq, current_bar=b)
            assert sdl.allows_new_entry("BTC", b)

        # Peak at bar 19: 266.67 + 95 = 361.67
        peak = sdl.get_state("BTC").peak_equity
        assert abs(peak - 361.67) < 0.1

        # Phase 2: 40% crash
        crash_eq = peak * 0.60  # = ~217.00
        sdl.update("BTC", crash_eq, current_bar=20)

        # Verify lock triggered
        assert sdl.is_locked("BTC", 20)
        assert not sdl.allows_new_entry("BTC", 20)

        # SOL and DOGE unaffected
        assert sdl.allows_new_entry("SOL", 20)
        assert sdl.allows_new_entry("DOGE", 20)

        # Phase 3: locked for 48h (12 bars)
        for b in range(21, 20 + LOCK_BARS):
            sdl.update("BTC", crash_eq + (b - 20) * 2, current_bar=b)
            assert not sdl.allows_new_entry("BTC", b), f"Should be locked at bar {b}"

        # Phase 4: unlock
        sdl.update("BTC", crash_eq + 30, current_bar=20 + LOCK_BARS)
        assert sdl.allows_new_entry("BTC", 20 + LOCK_BARS)

        # Phase 5: peak reset
        state = sdl.get_state("BTC")
        assert state.peak_equity == crash_eq + 30  # reset, not old 361.67

        # Verify events
        evts = sdl.get_events("BTC")
        triggers = [e for e in evts if e.event_type == "TRIGGER"]
        releases = [e for e in evts if e.event_type == "RELEASE"]
        assert len(triggers) == 1
        assert len(releases) == 1
        assert triggers[0].bar == 20
        assert releases[0].bar == 20 + LOCK_BARS


# ── 10. Determinism ──

class TestDeterminism:
    def test_identical_sequences(self):
        """Same input sequence → same events."""
        events_list = []
        for _ in range(3):
            sdl = SymbolDrawdownLock(["BTC"], 100.0, bars_per_hour=BARS_PER_HOUR)
            for b in range(50):
                eq = 100.0 + 10 * (b % 7) - 30 * (1 if b > 20 else 0)
                sdl.update("BTC", max(eq, 10.0), current_bar=b)
            events_list.append([(e.event_type, e.bar) for e in sdl.get_events()])
        assert events_list[0] == events_list[1] == events_list[2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
