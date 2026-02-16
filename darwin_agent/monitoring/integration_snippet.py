"""
Darwin v4 — Execution Audit Integration Snippet.

Shows exactly how to wire ExecutionAudit into the live trading loop.
Copy the relevant sections into your main trading loop.

This file is documentation, not a runnable script.
"""

# ═══════════════════════════════════════════════════════
# 1. INITIALIZATION (at process start)
# ═══════════════════════════════════════════════════════

from darwin_agent.monitoring.execution_audit import (
    ExecutionAudit,
    AlertThresholds,
)

# Create audit instance (one per process)
audit = ExecutionAudit(
    log_dir="logs/audit",
    thresholds=AlertThresholds(
        max_slippage_pct=0.5,
        max_fill_latency_ms=3000.0,
        max_reject_rate_pct=2.0,
        max_leverage_overshoot=0.0,
        min_liq_distance_pct=15.0,
        max_drift_pct=10.0,
        max_funding_rate=0.001,
    ),
)


# ═══════════════════════════════════════════════════════
# 2. ORDER EXECUTION (wrap around exchange calls)
# ═══════════════════════════════════════════════════════

async def execute_order_with_audit(
    audit, exchange, symbol, side, qty, current_price, leverage,
    signal_strength, bar_index,
):
    """
    Example: wrapping an exchange order call with audit tracking.
    This does NOT modify strategy logic. It only observes.
    """
    import time

    # Record intent BEFORE placing order
    intent = audit.record_intent(
        symbol=symbol,
        side=side,
        intended_qty=qty,
        intended_price=current_price,
        leverage=leverage,
        signal_strength=signal_strength,
        bar_index=bar_index,
    )

    # Place order on exchange (existing logic, unchanged)
    t0 = time.monotonic()
    try:
        result = await exchange.place_order(
            symbol=symbol,
            side=side,
            order_type="MARKET",
            quantity=qty,
        )
        latency_ms = (time.monotonic() - t0) * 1000.0

        # Record fill AFTER order completes
        audit.record_fill(
            intent=intent,
            filled_qty=result.get("filled_qty", 0.0),
            avg_fill_price=result.get("avg_price", 0.0),
            fee=result.get("fee", 0.0),
            order_id=result.get("order_id", ""),
            latency_ms=latency_ms,
        )

    except Exception as e:
        latency_ms = (time.monotonic() - t0) * 1000.0
        # Record rejection
        audit.record_fill(
            intent=intent,
            filled_qty=0.0,
            avg_fill_price=0.0,
            fee=0.0,
            order_id="",
            latency_ms=latency_ms,
        )

    return result


# ═══════════════════════════════════════════════════════
# 3. PER-BAR RISK CHECK (add to end of each bar loop)
# ═══════════════════════════════════════════════════════

def per_bar_audit(
    audit, bar_index, equity, peak, drawdown_pct,
    rbe_mult, gmrt_mult, effective_leverage, lev_cap,
    positions, exchange_positions, mark_prices,
    liquidation_prices, margin_ratios, funding_rates,
    simulated_sizes,
):
    """
    Call at the end of each bar processing.
    exchange_positions, mark_prices, etc. come from the exchange API.
    simulated_sizes come from the portfolio engine.
    """
    # Log risk state
    audit.check_risk_state(
        equity=equity,
        peak=peak,
        drawdown_pct=drawdown_pct,
        rbe_mult=rbe_mult,
        gmrt_mult=gmrt_mult,
        effective_leverage=effective_leverage,
        configured_lev_cap=lev_cap,
        positions=positions,
        bar_index=bar_index,
    )

    # Check exposure drift per symbol
    for sym in sorted(simulated_sizes.keys()):
        actual = exchange_positions.get(sym, 0.0)
        simulated = simulated_sizes[sym]
        audit.check_exposure_drift(
            symbol=sym,
            simulated_size=simulated,
            actual_size=actual,
            mark_price=mark_prices.get(sym, 0.0),
            liquidation_price=liquidation_prices.get(sym, 0.0),
            margin_ratio=margin_ratios.get(sym, 0.0),
            funding_rate=funding_rates.get(sym, 0.0),
            bar_index=bar_index,
        )


# ═══════════════════════════════════════════════════════
# 4. CIRCUIT BREAKER VALIDATION
# ═══════════════════════════════════════════════════════

async def cb_with_audit(
    audit, exchange, bar_index, equity_before, equity_after,
    loss_pct, positions_closed, cooldown_bars,
):
    """
    After CB closes all positions, verify with exchange.
    """
    # Query exchange for remaining open positions
    remaining = await exchange.get_open_positions()
    orphan_symbols = [p.symbol for p in remaining if abs(p.quantity) > 0.0001]

    clean = audit.validate_circuit_breaker(
        trigger_bar=bar_index,
        equity_before=equity_before,
        equity_after=equity_after,
        loss_pct=loss_pct,
        positions_closed=positions_closed,
        actual_open_positions=orphan_symbols,
        cooldown_bars=cooldown_bars,
    )

    if not clean:
        # Emergency: force-close orphan positions
        for pos in remaining:
            if abs(pos.quantity) > 0.0001:
                await exchange.close_position(pos)


# ═══════════════════════════════════════════════════════
# 5. PERIODIC METRICS (every N bars or on schedule)
# ═══════════════════════════════════════════════════════

def periodic_metrics(audit, bar_index):
    """Call every ~6 bars (1 day) or on a schedule."""
    metrics = audit.get_metrics()

    # Print summary to stdout
    audit.print_summary()

    # Optionally expose for Prometheus scraping:
    # for name, value in sorted(metrics.items()):
    #     prometheus_gauge(name, value)

    return metrics


# ═══════════════════════════════════════════════════════
# 6. SHUTDOWN
# ═══════════════════════════════════════════════════════

def shutdown(audit):
    """Call on graceful shutdown."""
    audit.print_summary()
    audit.close()
