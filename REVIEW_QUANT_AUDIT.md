# Darwin v5 — Institutional Quant Audit

**Auditor:** Lead Quant Architect / HFT Execution Specialist
**Date:** 2026-02-21
**Verdict:** This bot will bleed money slowly via fees, get liquidated on a fast wick, and has architecture bugs that apply Kelly Criterion TWICE. Below is a line-by-line destruction.

---

## 1. ALPHA & THE MONEY MAKER EDGE

### 1.1 Your R:R is 2:1, NOT 8:1

You say you want a 1:8 Risk/Reward machine. Let me show you what your code actually does:

```
# engine.py:63-64
stop_loss_pct: float = 1.5      # SL = 1.5%
take_profit_pct: float = 3.0    # TP = 3.0%
```

**R:R = 3.0 / 1.5 = 2:1.** That's it. In ranging markets it's even worse:

```
# engine.py:87-88
sl_pct_ranging:  float = 0.7    # SL = 0.7%
tp_pct_ranging:  float = 1.2    # TP = 1.2%
```

**R:R = 1.2 / 0.7 = 1.71:1.** You're risking almost as much as you're trying to make in the regime where you need the MOST edge (range-bound = noise = lower win rate).

To achieve 1:8 R:R you need SL=0.5% and TP=4.0%, or SL=1.0% and TP=8.0%. But 8% TP on a 5-second tick scalper? That's not a scalp, that's a swing trade. Your architecture doesn't know what it wants to be.

### 1.2 Every Single Order Pays Taker Fees

```python
# execution_engine.py:53
order_type: str = "MARKET"  # <-- this is the default for EVERY order
```

At Binance Futures VIP0:
- **Taker fee:** 0.0400% per side
- **Maker rebate:** -0.0200% per side (you GET PAID)

Your round-trip cost per trade:
- **Current (taker/taker):** 0.04% + 0.04% = **0.08%** per round trip
- **Optimal (maker/maker):** -0.02% + (-0.02%) = **you earn 0.04%** per round trip

**Delta: 0.12% per trade you're leaving on the table.**

With your ranging SL of 0.7% and TP of 1.2%:
- Fees eat **11.4% of your SL** (0.08/0.7) and **6.7% of your TP** (0.08/1.2)
- If you used limit orders: fees would CONTRIBUTE to your profit

At 20 trades/day * 30 days: 600 trades/month * 0.12% = **72% annualized drag.** That's your alpha evaporating into Binance's pocket.

### 1.3 Signal Edge: Statistical Assessment

The multi-factor signal generator (`signal_generator.py`) has reasonable academic foundations:
- Momentum demeaned vs BTC (Kakushadze eq.477) -- correct
- OU-calibrated mean reversion -- good idea, correctly clamped
- Funding carry as contrarian signal -- reasonable
- Order flow intelligence (OBI + TFD) -- this is the only real-time alpha

**BUT:** The z-score normalization uses a rolling window of only 200 bars (`_max_history = 200`). At 15-min candles, that's 50 hours. The z-score distribution is non-stationary over 50 hours in crypto. Your factor scores are getting normalized against a distribution that shifts faster than it converges.

**The Monte Carlo validator (`monte_carlo.py:143-157`) shuffles PnLs** -- this tests path dependence, NOT signal quality. Since you're shuffling the same PnL vector, the cumulative sum is identical. The `edge_ratio` will always be ~0. This "validation" validates nothing.

### 1.4 Confidence Threshold is Too Loose

```python
# signal_generator.py:71
confidence_threshold: float = 0.60
```

`_sigmoid(abs(combined_score))` maps |combined| = 0.41 to confidence = 0.60. A combined z-score of 0.41 standard deviations above zero is NOISE. You're trading on noise.

---

## 2. POSITION SIZING (KELLY CRITERION DOUBLE-APPLICATION BUG)

### 2.1 CRITICAL BUG: Two Kelly Implementations That Stack

This is the most dangerous bug in your codebase. You have TWO independent Kelly calculations that BOTH modify position size:

**Kelly #1 — PortfolioRiskManager** (`portfolio_risk.py:302-341`):
```python
def _compute_kelly_risk(self, equity: float) -> float:
    kelly_f = (p * W - q * L) / (W * L)
    half_kelly = kelly_f * cfg.kelly_fraction  # 0.5
    half_kelly_pct = half_kelly * 100.0
    return max(0.5, min(4.0, half_kelly_pct))  # returns RISK PERCENTAGE
```

**Kelly #2 — PositionSizer** (`position_sizer.py:108-155`):
```python
def _compute_kelly_scale(self, cfg: SizerConfig) -> float:
    kelly_fractional = kelly_full * cfg.kelly_fraction  # 0.5
    scale = kelly_fractional / base_risk  # returns SCALING FACTOR
    return max(0.5, min(1.5, scale))
```

**How they stack in engine.py:1102-1103 + position_sizer.py:264-266:**
```python
# engine.py:1102-1103
kelly_risk_pct = risk_check.kelly_risk_pct  # Kelly #1 output (e.g., 3.2%)
effective_risk_pct = kelly_risk_pct * risk_check.sizing_multiplier

# Then passed to position_sizer.compute() as risk_pct_override
# Inside position_sizer.compute():264-266:
if cfg.use_kelly and len(self._trade_pnls) >= cfg.kelly_lookback:
    kelly_scale = self._compute_kelly_scale(cfg)  # Kelly #2 (e.g., 1.3x)
    position_size *= kelly_scale  # DOUBLE KELLY!
```

**Result:** If Kelly #1 says risk 3.2% and Kelly #2 gives a 1.3x scale factor, you're effectively risking 3.2% * 1.3 = **4.16%** per trade when half-Kelly says 3.2%. Over-betting by 30% means you're past the Kelly peak and into the region where growth rate DECREASES. Past full Kelly, you go to ruin.

### 2.2 Kelly Formula Variant Mismatch

Kelly #1 uses: `f* = (p*W - q*L) / (W*L)` — this is the generalized Kelly for variable payoffs.

Kelly #2 uses: `f* = W - (1-W) / R` — this is the simplified Kelly for fixed-odds betting (where R = avg_win/avg_loss).

These are NOT equivalent when wins and losses have different distributions. Using both on the same trade data with different formulas is mathematically incoherent.

### 2.3 Position Size Formula

```python
# position_sizer.py:219
base_size = equity * base_risk_pct * leverage
```

With equity=$100, risk=1%, leverage=5x: `base_size = $100 * 0.01 * 5 = $5 notional`.

This is not how institutional sizing works. The correct formula is:
```
position_size = (equity * risk_fraction) / (SL_distance_pct / leverage)
```
Which sizes the position so that if SL is hit, you lose exactly risk_fraction of equity.

Your formula conflates notional exposure with risk. $5 notional at 5x leverage with 1.5% SL means your actual risk is $5 * 1.5% = $0.075 = 0.075% of equity. That's WAY below optimal. Then Kelly scales it back up... which is why the double-Kelly exists as a hack to compensate.

---

## 3. HIGH LEVERAGE & WICKS (MARK PRICE VS LAST PRICE)

### 3.1 CRITICAL: Client-Side SL Uses Last Price, Server-Side Uses Mark Price

```python
# binance_client.py:101-112
def get_current_price(self, symbol: str) -> float:
    response = self._session.get(
        f"{self._base_url}/fapi/v1/ticker/price",  # <-- LAST TRADE PRICE
        params={"symbol": symbol},
    )
    return float(response.json().get("price", 0.0))
```

```python
# binance_client.py:194
"workingType": "MARK_PRICE",  # <-- server-side SL triggers on MARK PRICE
```

```python
# engine.py:699-701
current_price = await asyncio.to_thread(
    self._binance.get_current_price, symbol  # <-- returns LAST price
)
```

```python
# engine.py:710-714
pnl_pct = (
    (current_price - entry_price) / entry_price  # <-- calculated with LAST price
    if is_long
    else (entry_price - current_price) / entry_price
)
```

**The attack vector during a wick:**

1. Mark Price = $95,000 (fair value)
2. Last Price spikes to $93,500 (wick - 1.6% below entry)
3. Your bot reads Last Price = $93,500 → triggers client-side SL
4. Bot sends MARKET SELL at the bottom of the wick → fills at $93,400 (slippage)
5. Mark Price never moved below $94,800 → server-side SL never triggered
6. Price recovers to $95,200 within 3 seconds
7. **You just got stopped out by a wick that was never real on Mark Price**

OR the opposite scenario:
1. Mark Price drops to $93,800 (real move)
2. Last Price stays at $94,900 (delayed/stale)
3. Server-side SL triggers on Mark Price → closes your position
4. Bot still thinks position is open (Last Price didn't hit SL)
5. Bot tries to close again on next tick → `reduceOnly` fails because position is already closed
6. **Orphaned order attempt, wasted API weight**

### 3.2 Liquidation Price Calculation

With 5x leverage, maintenance margin ≈ 0.4% (for most USDT-M pairs). Liquidation distance ≈ `(1/leverage - maintenance_margin) = (20% - 0.4%) = 19.6%` from entry.

Your SL at 1.5% gives a buffer of `19.6% - 1.5% = 18.1%` before liquidation. This is safe in isolation, BUT:

- With 3 correlated positions at 20% notional each = 60% total exposure
- A 3% market gap (common in crypto) wipes all 3 SLs simultaneously
- If all 3 SLs try to close at once: slippage stacks, Binance rate limits kick in
- The emergency_close_position retry with `0.5 * (2^attempt)` backoff maxes at 8 seconds
- In a flash crash, 8 seconds is an eternity

### 3.3 No Margin Check Before Entry

Nowhere in the codebase do you call `/fapi/v2/account` to check `availableBalance` vs `initialMargin` BEFORE placing an entry order. You check notional and portfolio heat, but not actual available margin. Binance can reject your order with `-2019 Margin is insufficient` even when your heat check passes because of unrealized losses on other positions reducing available balance.

---

## 4. API RATE LIMITS, BANS & WEBSOCKETS

### 4.1 Zero WebSocket Usage

**There is no WebSocket connection in this entire codebase.** Everything is REST polling every 5 seconds. This is catastrophic for a system that calls itself HFT:

- **5-second price latency**: You're trading on data that's already 5 seconds old. In crypto, a 5-second-old price is ancient history.
- **No real-time fill notifications**: After placing an order, you wait 300ms * 3 = 900ms polling for fill status (`execution_engine.py:443-470`), then don't check again until the next 5s tick.
- **No account update stream**: If a server-side SL triggers between ticks, you don't know until the next `get_open_positions()` call.

### 4.2 Rate Limit Math (Per Tick, 3 Symbols)

| API Call | Weight | Count/Tick | Total |
|----------|--------|------------|-------|
| get_account_snapshot (/fapi/v2/account) | 5 | 1 | 5 |
| get_open_positions (/fapi/v2/positionRisk) | 5 | 1 | 5 |
| fetch_candles (/fapi/v1/klines) | 5 | 3 | 15 |
| fetch_funding_rate (/fapi/v1/fundingRate) | 1 | 3 | 3 |
| order_flow depth (/fapi/v1/depth) | 5 | 3 | 15 |
| order_flow aggTrades (/fapi/v1/aggTrades) | 1 | 3 | 3 |
| **Total per tick** | | | **46** |
| **Per minute (12 ticks)** | | | **552** |

Binance limit: **2400 weight/minute** for default accounts.

Current usage: 552/2400 = **23%**. Looks safe, BUT:

- This doesn't count orders (weight 1 each), fill checks (weight 1), SL/TP placements (weight 1), VISA audits (weight 5 per get_open_orders)
- Adding more symbols: 10 symbols → 1840/min = **77%** — dangerously close
- `_signed_request()` at `binance_client.py:54-62` has **NO rate limit handling**. Only `_signed_request_with_retry()` handles 429. Many critical calls go through the unprotected path.
- **No IP-level 418 handling.** Binance escalates from 429 to HTTP 418 (IP ban) after repeated violations. Your code doesn't even check for 418.

### 4.3 Orphan Order Risk

If the bot crashes after placing an entry order but BEFORE placing the VISA server-side SL:

```python
# engine.py:1157-1158 — order placed here
result = await self._execution_engine.place_order(order, dry_run=cfg.dry_run)

# engine.py:1170-1178 — SL placed here (AFTER fill)
if self._visa and not self._config.dry_run and result.avg_price > 0:
    await self._visa_place_after_fill(...)
```

If the process dies between line 1158 and 1178: **open position with no protection.** The `_visa_sync_existing_positions()` on restart partially mitigates this, but there's a window of vulnerability.

### 4.4 Thread Safety in MarketDataLayer

```python
# market_data.py:124-126
with self._lock:
    session = self._client._session  # read reference under lock
    base_url = self._client._base_url
    timeout = self._client._timeout_s
# Then use session OUTSIDE lock → requests.Session is NOT thread-safe
response = session.get(...)
```

`requests.Session` sharing across threads without the lock held during the request means connection pooling corruption under load. With `asyncio.to_thread()` calling `compute_features()` for 3 symbols concurrently, this is a real race condition.

---

## 5. INSTITUTIONAL-GRADE FIXES

The following critical fixes address the most dangerous issues found in this audit. Each fix targets a specific vulnerability that would cause capital loss in production.
