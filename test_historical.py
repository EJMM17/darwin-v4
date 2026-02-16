"""
Darwin v4 — HistoricalMarketScenario + Fee/Slippage Tests.

Covers:
  1.  CSV loading (auto-detect header, timestamps)
  2.  Step-by-step close-to-close replay
  3.  Timeframe resampling (1m→1h, passthrough)
  4.  Maker/taker fee model + slippage
  5.  EvolutionEngine preserved
  6.  DDT 2.0 preserved
  7.  capital_floor preserved
  8.  Determinism (no MC randomness in prices)
  9.  Multi-symbol alignment
  10. Integration with harness
"""
import asyncio, math, os, random, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))

from darwin_agent.simulation.harness import (
    SimulationHarness, SimConfig, MonteCarloScenario,
    HistoricalMarketScenario, SimulatedAgent, Tick,
)
from darwin_agent.interfaces.types import DNAData

PASS = FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ✅ {name}")
    else: FAIL += 1; print(f"  ❌ {name}: {detail}")


def _make_csv(path, rows, header="timestamp,open,high,low,close,volume"):
    with open(path, "w") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")


def _make_bars(n=500, base=50000, vol=0.01, start_ts=1700000000, interval=3600):
    """Generate synthetic OHLCV bars."""
    bars = []
    p = base
    for i in range(n):
        o = p
        h = o * (1 + vol * 0.5)
        l = o * (1 - vol * 0.5)
        c = o * (1 + (i % 7 - 3) * vol * 0.1)  # deterministic drift
        v = 100 + i * 0.1
        bars.append((start_ts + i * interval, round(o, 2), round(h, 2),
                     round(l, 2), round(c, 2), round(v, 2)))
        p = c
    return bars


# ═══════════════════════════════════════════════════════════
# 1. CSV loading
# ═══════════════════════════════════════════════════════════

def test_1_csv_loading():
    print("\n🧪 1. CSV loading")
    print("─" * 50)
    with tempfile.TemporaryDirectory() as td:
        # Numeric timestamp
        rows = _make_bars(100)
        path = os.path.join(td, "btc.csv")
        _make_csv(path, rows)
        s = HistoricalMarketScenario(csv_paths={"BTCUSDT": path})
        check("Loaded 100 bars", s.total_bars == 100)
        check("Symbol list", s.get_symbols() == ["BTCUSDT"])

        # ISO date timestamp
        iso_rows = [(f"2024-01-{(i//24)+1:02d} {i%24:02d}:00:00",
                     50000+i, 50100+i, 49900+i, 50050+i, 100+i)
                    for i in range(48)]
        path2 = os.path.join(td, "eth.csv")
        _make_csv(path2, iso_rows)
        s2 = HistoricalMarketScenario(csv_paths={"ETHUSDT": path2})
        check("ISO dates loaded", s2.total_bars == 48)

        # Millisecond timestamps
        ms_rows = [(1700000000000 + i * 3600000,
                    3000+i, 3010+i, 2990+i, 3005+i, 50+i)
                   for i in range(30)]
        path3 = os.path.join(td, "sol.csv")
        _make_csv(path3, ms_rows)
        s3 = HistoricalMarketScenario(csv_paths={"SOLUSDT": path3})
        check("Millisecond timestamps", s3.total_bars == 30)
    print()


# ═══════════════════════════════════════════════════════════
# 2. Close-to-close replay
# ═══════════════════════════════════════════════════════════

def test_2_close_replay():
    print("🧪 2. Close-to-close replay")
    print("─" * 50)
    bars = [HistoricalMarketScenario.OHLCVBar(
        timestamp=1700000000 + i * 3600,
        open=50000 + i * 10, high=50100 + i * 10,
        low=49900 + i * 10, close=50050 + i * 10,
        volume=100.0 + i,
    ) for i in range(200)]
    s = HistoricalMarketScenario(csv_data={"BTCUSDT": bars})
    rng = random.Random(42)
    ticks = s.generate(50, rng)
    check("50 ticks generated", len(ticks["BTCUSDT"]) == 50)
    # Prices should be close values from bars (offset determined by rng)
    for t in ticks["BTCUSDT"]:
        check(f"Step {t.step} uses close price",
              any(abs(t.price - b.close) < 0.01 for b in bars))
        if t.step > 2:
            break  # spot-check first few
    # Volume from actual data, not random
    check("Volume from data", ticks["BTCUSDT"][0].volume > 0)
    print()


# ═══════════════════════════════════════════════════════════
# 3. Timeframe resampling
# ═══════════════════════════════════════════════════════════

def test_3_timeframe():
    print("🧪 3. Timeframe resampling")
    print("─" * 50)
    # 1m bars → 1h should downsample 60:1
    bars_1m = [HistoricalMarketScenario.OHLCVBar(
        timestamp=1700000000 + i * 60,
        open=50000 + i, high=50100 + i, low=49900 + i,
        close=50050 + i, volume=10.0,
    ) for i in range(600)]  # 10 hours of 1m data
    s = HistoricalMarketScenario(csv_data={"BTCUSDT": bars_1m}, timeframe="1h")
    check("1m→1h resampled", s.total_bars == 10, f"got {s.total_bars}")

    # Same-resolution passthrough
    bars_1h = [HistoricalMarketScenario.OHLCVBar(
        timestamp=1700000000 + i * 3600,
        open=50000, high=50100, low=49900, close=50050, volume=100.0,
    ) for i in range(100)]
    s2 = HistoricalMarketScenario(csv_data={"BTCUSDT": bars_1h}, timeframe="1h")
    check("1h→1h passthrough", s2.total_bars == 100)
    print()


# ═══════════════════════════════════════════════════════════
# 4. Fee model + slippage
# ═══════════════════════════════════════════════════════════

def test_4_fees():
    print("🧪 4. Maker/taker fee + slippage")
    print("─" * 50)
    genes = {"risk_pct": 4.0, "leverage_aggression": 0.5,
             "confidence_threshold": 0.01, "momentum_weight": 1.0,
             "mean_rev_weight": 0.0, "scalping_weight": 0.0,
             "breakout_weight": 0.0, "regime_bias": 0.0,
             "volatility_threshold": 0.0, "stop_loss_pct": 5.0,
             "take_profit_pct": 5.0, "timeframe_bias": 0.5}
    dna = DNAData(genes=genes, generation=0)

    # Agent with zero fees
    a1 = SimulatedAgent("t1", dna, 1000.0, random.Random(42), fee_rate=0.0)
    # Agent with fees (taker 0.05% + slippage 0.03% = 0.08% per side)
    a2 = SimulatedAgent("t2", dna, 1000.0, random.Random(42), fee_rate=0.0008)

    # Run both through identical trending prices
    prices = [50000 + i * 50 for i in range(200)]
    ticks = {"BTCUSDT": [Tick("BTCUSDT", i, p, 100.0) for i, p in enumerate(prices)]}
    for step in range(200):
        st = {s: ts[step] for s, ts in ticks.items()}
        a1.step(st, step)
        a2.step(st, step)

    e1 = a1.get_eval_data(); e2 = a2.get_eval_data()
    check("Fee agent has slot", hasattr(a2, "_fee_rate"))
    check("SimConfig has maker_fee", hasattr(SimConfig, "maker_fee"))
    check("SimConfig has taker_fee", hasattr(SimConfig, "taker_fee"))
    check("SimConfig has slippage", hasattr(SimConfig, "slippage"))
    check("Default slippage = 0.0003", SimConfig().slippage == 0.0003)
    check("Default taker = 0.0005", SimConfig().taker_fee == 0.0005)
    # If both traded, fee agent should have less profit
    if e1.metrics.total_trades > 0 and e2.metrics.total_trades > 0:
        check("Fee agent less profit", e2.metrics.realized_pnl < e1.metrics.realized_pnl,
              f"no-fee={e1.metrics.realized_pnl:.2f} fee={e2.metrics.realized_pnl:.2f}")
    print()


# ═══════════════════════════════════════════════════════════
# 5-7. Preserved: Evolution, DDT 2.0, capital_floor
# ═══════════════════════════════════════════════════════════

def test_5_preserved():
    print("🧪 5-7. Evolution + DDT 2.0 + capital_floor preserved")
    print("─" * 50)
    src = open("darwin_agent/simulation/harness.py").read()
    check("EvolutionEngine import", "EvolutionEngine" in src)
    check("DDT 2.0 smooth formula", "1.0 - 1.5 *" in src)
    check("enable_ddt field", "enable_ddt" in src)
    check("capital_floor_pct", "capital_floor_pct" in src)
    check("floor_abs computed", "floor_abs" in src)
    check("ddt_peak tracker", "ddt_peak" in src)
    print()


# ═══════════════════════════════════════════════════════════
# 8. Determinism
# ═══════════════════════════════════════════════════════════

def test_8_determinism():
    print("🧪 8. Determinism (no MC randomness)")
    print("─" * 50)
    bars = [HistoricalMarketScenario.OHLCVBar(
        timestamp=1700000000 + i * 3600,
        open=50000 + i, high=50100, low=49900,
        close=50050 + i * 5, volume=100.0,
    ) for i in range(300)]
    s = HistoricalMarketScenario(csv_data={"BTCUSDT": bars})

    # Same seed → same ticks
    t1 = s.generate(100, random.Random(42))
    t2 = s.generate(100, random.Random(42))
    prices1 = [t.price for t in t1["BTCUSDT"]]
    prices2 = [t.price for t in t2["BTCUSDT"]]
    check("Same seed same prices", prices1 == prices2)

    # Different seed → different offset
    t3 = s.generate(100, random.Random(99))
    prices3 = [t.price for t in t3["BTCUSDT"]]
    check("Diff seed diff offset", prices1 != prices3)

    # Prices are EXACT close values (no random noise)
    close_set = {b.close for b in bars}
    all_from_data = all(t.price in close_set for t in t1["BTCUSDT"])
    check("All prices from data", all_from_data)
    print()


# ═══════════════════════════════════════════════════════════
# 9. Multi-symbol alignment
# ═══════════════════════════════════════════════════════════

def test_9_multi_symbol():
    print("🧪 9. Multi-symbol alignment")
    print("─" * 50)
    btc = [HistoricalMarketScenario.OHLCVBar(
        timestamp=1700000000+i*3600, open=50000, high=50100,
        low=49900, close=50050+i, volume=100.0,
    ) for i in range(200)]
    eth = [HistoricalMarketScenario.OHLCVBar(
        timestamp=1700000000+i*3600, open=3000, high=3010,
        low=2990, close=3005+i, volume=50.0,
    ) for i in range(150)]  # shorter

    s = HistoricalMarketScenario(csv_data={"BTCUSDT": btc, "ETHUSDT": eth})
    check("Aligned to shortest", s.total_bars == 150)
    check("Both symbols present", s.get_symbols() == ["BTCUSDT", "ETHUSDT"])

    ticks = s.generate(50, random.Random(42))
    check("BTC has 50 ticks", len(ticks["BTCUSDT"]) == 50)
    check("ETH has 50 ticks", len(ticks["ETHUSDT"]) == 50)
    print()


# ═══════════════════════════════════════════════════════════
# 10. Harness integration
# ═══════════════════════════════════════════════════════════

async def test_10_harness():
    print("🧪 10. Harness integration")
    print("─" * 50)
    bars = {
        "BTCUSDT": [HistoricalMarketScenario.OHLCVBar(
            timestamp=1700000000+i*3600, open=50000+i*10, high=50100+i*10,
            low=49900+i*10, close=50050+i*10, volume=100+i,
        ) for i in range(2000)],
        "ETHUSDT": [HistoricalMarketScenario.OHLCVBar(
            timestamp=1700000000+i*3600, open=3000+i*5, high=3010+i*5,
            low=2990+i*5, close=3005+i*5, volume=50+i,
        ) for i in range(2000)],
    }
    s = HistoricalMarketScenario(csv_data=bars)

    # Run with fees + DDT
    r = await SimulationHarness(SimConfig(
        scenario=s, generations=50, pool_size=5,
        trades_per_generation=100, starting_capital=100.0,
        seed=42, mutation_rate=0.18, survival_rate=0.40,
        capital_floor_pct=0.45, enable_ddt=True,
        taker_fee=0.0005, slippage=0.0003,
    )).run()
    check("Harness completes", r.generations_run == 50)
    check("Capital valid", r.final_capital > 0)

    # Run without fees — same seed
    r2 = await SimulationHarness(SimConfig(
        scenario=s, generations=50, pool_size=5,
        trades_per_generation=100, starting_capital=100.0,
        seed=42, mutation_rate=0.18, survival_rate=0.40,
        capital_floor_pct=0.45, enable_ddt=True,
        maker_fee=0.0, taker_fee=0.0, slippage=0.0,
    )).run()
    check("No-fee run completes", r2.generations_run == 50)
    # Fee run should produce different (typically lower) capital
    check("Fees affect outcome",
          abs(r.final_capital - r2.final_capital) > 0.01,
          f"fee={r.final_capital:.2f} nofee={r2.final_capital:.2f}")
    print()


# ═══════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════

async def main():
    global PASS, FAIL
    print("═" * 60)
    print("  🧪 HISTORICAL MARKET SCENARIO + FEE TESTS")
    print("═" * 60)
    for t in [test_1_csv_loading, test_2_close_replay, test_3_timeframe,
              test_4_fees, test_5_preserved, test_8_determinism,
              test_9_multi_symbol]:
        try: t()
        except Exception as e:
            FAIL += 1; print(f"  ❌ CRASHED: {e}")
            import traceback; traceback.print_exc()
    for t in [test_10_harness]:
        try: await t()
        except Exception as e:
            FAIL += 1; print(f"  ❌ CRASHED: {e}")
            import traceback; traceback.print_exc()
    print("═" * 60)
    if FAIL == 0: print(f"  ✅ ALL {PASS} TESTS PASSED")
    else: print(f"  ❌ {FAIL} FAILED, {PASS} passed")
    print("═" * 60)
    return FAIL == 0

if __name__ == "__main__":
    sys.exit(0 if asyncio.run(main()) else 1)
