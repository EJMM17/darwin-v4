# Code Review: PR #21 "8 Critical Bug Fixes"

**Reviewer verdict: REQUEST CHANGES**

## Critical Issues

### 1. threading.Lock wraps HTTP requests (market_data.py:113)

The lock covers the entire `fetch_candles` method including the network call to Binance.
Three symbols calling `compute_features` via `asyncio.to_thread` now serialize completely.
If one request takes 10s timeout, the other two wait 10s.

**Fix:** Lock only cache dict access. Do HTTP calls outside the lock. Double-check locking pattern.

Also: `clear_cache()` at line 256 doesn't acquire the lock — thread safety is incomplete.

### 2. Split-brain daily PnL

Two competing definitions used interchangeably:

| Location | Source | Measures |
|----------|--------|----------|
| Circuit breaker (engine:443) | `equity - daily_start_equity` | Realized + unrealized |
| Position sizer (engine:1003) | `self._state.daily_pnl` | Realized only |
| Kill switch (engine:576) | `(daily_start - equity) / daily_start` | Realized + unrealized |

**Edge case:** Down 4.5% unrealized, zero closed trades. Circuit breaker sees danger.
Position sizer receives `daily_pnl=0.0` and approves full-sized trades.

### 3. `reset_daily()` is a no-op

Method body is `pass`. Still called every day. Comment says "kept for Kelly context"
but there's no Kelly context remaining in PositionSizer's daily scope. Dead code.

### 4. PnL leverage removal is unverified

Changed from `pnl_pct * entry_price * close_qty * leverage` to `pnl_pct * entry_price * close_qty`.
Assumes Binance `positionAmt` is full notional quantity. No test validates this across
isolated vs cross margin modes.

### 5. Circuit breaker has 5-minute blind spot

Only checks every `heartbeat_interval_ticks * 5 = 60 ticks = 5 minutes`.
Flash crashes can move 15%+ in 90 seconds. The separate `_check_kill_switches`
runs every tick but with different thresholds — two overlapping systems, neither complete.

### 6. Midnight rollover resets all daily protection

Open position bleeding for 23 hours at -4.9% gets a fresh baseline at UTC midnight.
Can lose another 5% the next day. Repeated daily = 35% weekly loss with "working" kill switch.

### 7. Restart wipes risk state

`initialize()` resets `daily_start_equity` and `daily_pnl`. Crash at -4% and restart = counter at zero.
No persistence layer (file, Redis, database) preserves risk state across restarts.

### 8. `last_trade_tick` hides execution failures

Recording tick before checking `result.success` means failed orders trigger cooldown.
Broken execution (bad API key, insufficient margin, delisted symbol) is silently throttled
instead of escalated.

### 9. `import math as _math` inside method body (position_sizer.py:289)

Inline import inside `compute()` called on every trade. Should be a top-level import.

### 10. Zero test coverage for changed code paths

No tests for:
- `_compute_daily_loss_scale` new signature
- Circuit breaker integration in engine
- Lock behavior under concurrency
- PnL calculation with/without leverage
- Daily reset / midnight rollover

## Capital Viability Analysis

- $80 capital at 5x leverage = $400 effective
- 1% risk per trade = $0.80 risk
- Binance min notional = $5, after scaling factors most trades are rejected
- Taker fees (0.04%) on $5 trade = $0.004 round trip
- Ranging TP target 1.2% on $5 = $0.06 profit; fees = 6.7% of profit
- Institutional-grade risk infra managing sub-$10 positions
