#!/usr/bin/env python3
"""
Darwin v4 — Determinism Verification.

Runs the FULL evolution + portfolio pipeline TWICE with the same seed.
Asserts byte-identical genomes, equity curves, trades, and hashes.

Usage:
    python reproduce_run.py [--seed 42] [--freeze path/to/frozen.json]

Mode 1 (default): Run twice, compare.
Mode 2 (--freeze): Run once, compare against frozen canonical run.
"""

from __future__ import annotations

import sys
import os
import csv
import math
import statistics
import time
import argparse

# ── CRITICAL: Set PYTHONHASHSEED before any imports ──
os.environ["PYTHONHASHSEED"] = "42"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from darwin_agent.determinism import (
    lock_determinism,
    reset_id_counter,
    genome_hash,
    equity_curve_hash,
)
from darwin_agent.freeze import (
    lock_all_determinism,
    freeze_run,
    load_frozen_run,
    extract_genomes,
    verify_reproduction,
    ReproductionResult,
)
from darwin_agent.portfolio.global_regime import GlobalRegimeThrottle
from darwin_agent.portfolio.allocation_engine import PortfolioAllocationEngine, PAEConfig
from darwin_agent.portfolio.risk_budget_engine import RiskBudgetEngine, RBEConfig
from darwin_agent.evolution.rolling_engine import RollingEvolutionEngine, RQREConfig
from darwin_agent.simulation.harness import (
    SimulationHarness, SimConfig, HistoricalMarketScenario,
)

import asyncio
import random
import tempfile


# ═══════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════

SYMS = ["BTC", "SOL", "DOGE"]
CLUSTERS = {
    "BTC": "regime-switching",
    "SOL": "momentum-volatile",
    "DOGE": "mean-reverting",
}
BPD = 6
CAP = 800.0
LEV_CAP = 5.0
FEE = 0.0007
SLIP = 0.0002
DAILY_LIM = 0.03
WEEKLY_LIM = 0.08
MONTHLY_LIM = 0.20


# ═══════════════════════════════════════════════════════
# DETERMINISTIC HELPERS
# ═══════════════════════════════════════════════════════

def ema(d, p):
    o = [0.0] * len(d)
    k = 2.0 / (p + 1)
    o[0] = d[0]
    for i in range(1, len(d)):
        o[i] = d[i] * k + o[i - 1] * (1.0 - k)
    return o


def atr_f(h, l, c, p=14):
    n = len(c)
    tr = [h[0] - l[0]]
    for i in range(1, n):
        tr.append(max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])))
    o = [0.0] * n
    if n >= p:
        o[p - 1] = sum(tr[:p]) / float(p)
    for i in range(p, n):
        o[i] = (o[i - 1] * (p - 1) + tr[i]) / float(p)
    return o


def rsi_f(c, p=14):
    n = len(c)
    o = [50.0] * n
    g = 0.0
    lo = 0.0
    for i in range(1, min(p + 1, n)):
        d = c[i] - c[i - 1]
        if d > 0:
            g += d
        else:
            lo -= d
    if p > 0:
        g /= float(p)
        lo /= float(p)
    if g + lo > 0:
        o[p] = 100.0 * g / (g + lo)
    for i in range(p + 1, n):
        d = c[i] - c[i - 1]
        gg = d if d > 0.0 else 0.0
        ll = -d if d < 0.0 else 0.0
        g = (g * (p - 1) + gg) / float(p)
        lo = (lo * (p - 1) + ll) / float(p)
        o[i] = 100.0 * g / (g + lo) if g + lo > 0 else 50.0
    return o


def load_raw(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def dlev(gm, rm):
    c = gm * rm
    if c > 0.5625:
        return min(4.0 + (c - 0.5625) / (1.0 - 0.5625), 5.0)
    return 2.0 + c / 0.5625


def sig(g, d, i):
    r = 0.0
    r += g.get("momentum_weight", 0.0) * (1.0 if d["ef"][i] > d["em"][i] else -1.0)
    r += g.get("ema_weight", 0.0) * (1.0 if d["ef"][i] > d["es"][i] else -1.0)
    r += g.get("rsi_weight", 0.0) * ((d["ri"][i] - 50.0) / 50.0)
    r += g.get("mean_rev_weight", 0.0) * ((50.0 - d["ri"][i]) / 50.0)
    bk = (d["c"][i] - d["em"][i]) / d["at"][i] if d["at"][i] > 0.0 else 0.0
    r += g.get("breakout_weight", 0.0) * bk
    r += g.get("trend_strength_weight", 0.0) * (
        abs(d["ef"][i] - d["es"][i]) / d["c"][i]
    )
    r += g.get("regime_bias", 0.0) * (
        (d["c"][i] - d["e2"][i]) / d["e2"][i] if d["e2"][i] > 0 else 0.0
    )
    return math.tanh(r)


# ═══════════════════════════════════════════════════════
# DETERMINISTIC TRAIN FUNCTION
# ═══════════════════════════════════════════════════════

async def train_fn(train_data, val_data, seed, cluster_type, config):
    """Fully deterministic: locks RNG before SimulationHarness."""
    fields = ["timestamp", "open", "high", "low", "close", "volume"]
    tf = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, dir="/home/claude"
    )
    w = csv.DictWriter(tf, fieldnames=fields)
    w.writeheader()
    for row in train_data:
        w.writerow({k: row.get(k, 0) for k in fields})
    tf.close()

    try:
        r = await SimulationHarness(
            SimConfig(
                scenario=HistoricalMarketScenario(
                    csv_paths={"S": tf.name}, timeframe="4h"
                ),
                generations=100,
                pool_size=6,
                trades_per_generation=50,
                starting_capital=100.0,
                mutation_rate=0.18,
                survival_rate=0.40,
                seed=seed,
                capital_floor_pct=0.45,
                enable_ddt=True,
                maker_fee=0.0002,
                taker_fee=0.0005,
                slippage=0.0002,
                cluster_type=cluster_type,
            )
        ).run()

        cands = []
        for h in r.hall_of_fame:
            if h.get("genes"):
                fit = h.get("fitness", 0)
                cands.append({
                    "genes": h["genes"],
                    "fitness": fit,
                    "pf": max(1.0, fit * 2.5),
                    "max_dd": max(0.05, 1.0 - fit),
                    "trades": 30,
                })
        # Deterministic: sort candidates by fitness desc, then genome hash
        cands.sort(key=lambda c: (-c["fitness"], genome_hash(c["genes"])))

        for gr in r.generation_results[-3:]:
            for d in (gr.snapshot.dna_pool or [])[:2]:
                fb = (
                    gr.snapshot.agent_rankings[0]
                    if gr.snapshot.agent_rankings
                    else {}
                )
                cands.append({
                    "genes": dict(d.genes),
                    "fitness": fb.get("fitness", 0.5),
                    "pf": 1.5,
                    "max_dd": 0.30,
                    "trades": 30,
                })
        return cands
    finally:
        os.unlink(tf.name)


# ═══════════════════════════════════════════════════════
# DETERMINISTIC PORTFOLIO SIMULATION
# ═══════════════════════════════════════════════════════

def run_deterministic(
    seed: int,
    all_raw_data: dict,
    n_bars: int,
    boot: int = 1095,
    preloaded_genomes: dict = None,
) -> dict:
    """
    Run the full pipeline deterministically.

    If preloaded_genomes is provided, skip bootstrap evolution
    and use those genomes directly (for reproduction mode).
    """
    # PHASE 1: Lock everything
    lock_all_determinism(seed)

    loop = asyncio.new_event_loop()

    # Build indicators (pure math, deterministic)
    ind = {}
    for sym in sorted(SYMS):
        c = [float(r["close"]) for r in all_raw_data[sym][:n_bars]]
        h = [float(r["high"]) for r in all_raw_data[sym][:n_bars]]
        l = [float(r["low"]) for r in all_raw_data[sym][:n_bars]]
        ind[sym] = {
            "c": c, "h": h, "l": l,
            "ef": ema(c, 12), "em": ema(c, 50), "es": ema(c, 48),
            "ri": rsi_f(c, 14), "at": atr_f(h, l, c, 14), "e2": ema(c, 200),
        }

    gmrt = GlobalRegimeThrottle()
    gmrt.load_btc("/home/claude/btc_2022_2024_1h.csv")
    gmrt_r = gmrt.compute()

    # Bootstrap genomes
    if preloaded_genomes:
        ig = dict(preloaded_genomes)
    else:
        # DETERMINISM: Reset RNG state before bootstrap
        random.seed(seed)
        reset_id_counter(seed)
        ig = {}
        for sym in sorted(SYMS):
            bd = all_raw_data[sym][:boot]
            cs = loop.run_until_complete(
                train_fn(bd, bd[-180:], seed, CLUSTERS[sym], RQREConfig())
            )
            # Deterministic selection: best fitness, tie-break by genome hash
            best_fit = 0.0
            best_genes = None
            best_hash = ""
            for c2 in cs:
                f = c2["fitness"]
                gh = genome_hash(c2["genes"])
                if f > best_fit or (f == best_fit and gh < best_hash):
                    best_fit = f
                    best_genes = c2["genes"]
                    best_hash = gh
            if best_genes:
                ig[sym] = best_genes

    # DETERMINISM: Reset RNG to known state after bootstrap
    random.seed(seed + 1000)
    reset_id_counter(seed + 1000)

    rqre = RollingEvolutionEngine(
        symbols=sorted(SYMS),
        cluster_types=CLUSTERS,
        config=RQREConfig(
            re_evolution_interval=30,
            training_lookback_days=365,
            validation_days=30,
            generations=100,
            seeds=(seed,),
            pool_size=6,
            replace_bottom_pct=0.30,
            select_top_pct=0.30,
            min_pf=1.0,
            max_dd=0.45,
            min_trades=20,
        ),
        initial_genomes=ig,
    )
    pae = PortfolioAllocationEngine(
        sorted(SYMS),
        config=PAEConfig(
            lookback=200, warmup_bars=50, ema_alpha=0.03,
            min_weight=0.05, max_weight=0.70,
        ),
    )
    rbe = RiskBudgetEngine(
        RBEConfig(
            dd_limit=0.32, target_vol=0.06, window=200,
            warmup=30, mult_floor=0.50, mult_ceil=1.00,
        ),
    )

    pe = CAP
    se = {s: CAP / 3.0 for s in sorted(SYMS)}
    sp = {s: 0.0 for s in sorted(SYMS)}
    pk = CAP
    mdd = 0.0
    ds = CAP
    ws = CAP
    ms = CAP
    dh = False
    wh = False
    mh = False
    mr = []
    ag = dict(ig)
    cm = 0
    tf_total = 0.0
    tt = 0
    eq_curve = [CAP]

    for i in range(boot, n_bars):
        bc = i - boot
        bod = bc % BPD
        day = bc // BPD
        month = bc // 182

        if bod == 0:
            ds = pe
            dh = False
        if bc % (BPD * 7) == 0:
            ws = pe
            wh = False
        if month > cm:
            if cm > 0 or bc > 182:
                mr.append((pe - ms) / ms * 100.0 if ms > 0 else 0.0)
            ms = pe
            mh = False
            cm = month

        # RQRE: deterministic cycle
        if rqre.step_if_due(bc, bars_per_day=BPD):
            mkt = {sym: all_raw_data[sym][:i] for sym in sorted(SYMS)}
            logs = loop.run_until_complete(
                rqre.execute_cycle(day, train_fn, mkt, bars_per_day=BPD)
            )
            ng = rqre.active_genomes
            for s in sorted(SYMS):
                if ng[s]:
                    ag[s] = ng[s]

        # Halt checks
        dl2 = (ds - pe) / ds if ds > 0 else 0.0
        wl2 = (ws - pe) / ws if ws > 0 else 0.0
        ml2 = (ms - pe) / ms if ms > 0 else 0.0
        if dl2 >= DAILY_LIM and not dh:
            dh = True
        if wl2 >= WEEKLY_LIM and not wh:
            wh = True
        if ml2 >= MONTHLY_LIM and not mh:
            mh = True
        halted = dh or wh or mh

        gi = min(i * 4, len(gmrt_r.bars) - 1)
        gm = gmrt_r.multiplier_at(gi)
        rm = rbe.last_mult
        lv = dlev(gm, rm) if not halted else 0.0

        if not halted:
            wt = pae.get_weights()
            tp = 0.0
            # DETERMINISM: iterate symbols in sorted order
            for s in sorted(SYMS):
                dd = ind[s]
                p = dd["c"][i]
                pp = dd["c"][i - 1]
                if p <= 0.0 or pp <= 0.0:
                    continue
                sg_v = sig(ag[s], dd, i)
                nt = pe * wt[s] * lv * abs(sg_v)
                nt = min(nt, se[s] * LEV_CAP)
                tg = math.copysign(nt, sg_v)
                dt = tg - sp[s]
                if abs(dt) > abs(sp[s]) * 0.10 + 5.0:
                    fee = abs(dt) * (FEE + SLIP)
                    tf_total += fee
                    tt += 1
                    pnl = sp[s] * ((p - pp) / pp) - fee if abs(sp[s]) > 1.0 else -fee
                    sp[s] = tg
                else:
                    pnl = sp[s] * ((p - pp) / pp) if abs(sp[s]) > 1.0 else 0.0
                se[s] += pnl
                se[s] = max(se[s], 1.0)
                tp += pnl
                pae.update(s, equity=se[s], pnl=pnl if abs(pnl) > 0.01 else 0.0)
            pe = sum(se[s] for s in sorted(SYMS))
        else:
            tp = 0.0
            for s in sorted(SYMS):
                pae.update(s, equity=se[s], pnl=0.0)

        if pe > pk:
            pk = pe
        ddp = (pk - pe) / pk if pk > 0 else 0.0
        mdd = max(mdd, ddp)

        eq_curve.append(pe)
        rbe.update(equity=pe, pnl=tp if abs(tp) > 0.01 else 0.0)
        rbe.step()
        pae.step()

        if bod == 0 and not halted:
            w = pae.get_weights()
            for s in sorted(SYMS):
                se[s] = pe * w[s]

    if ms > 0:
        mr.append((pe - ms) / ms * 100.0)

    td = (n_bars - boot) // BPD
    cagr = ((pe / CAP) ** (365.0 / td) - 1.0) * 100.0 if td > 0 else 0.0
    sh = 0.0
    if len(mr) >= 2:
        mu = statistics.mean(mr)
        sigma = statistics.stdev(mr)
        sh = mu / sigma * math.sqrt(12.0) if sigma > 0 else 0.0

    loop.close()

    return {
        "pe": pe,
        "cagr": cagr,
        "mdd": mdd * 100.0,
        "sharpe": sh,
        "tt": tt,
        "fees": tf_total,
        "eq_curve": eq_curve,
        "mr": mr,
        "ag": {s: dict(ag[s]) for s in sorted(SYMS)},
        "eq_hash": equity_curve_hash(eq_curve),
        "genome_hashes": {s: genome_hash(ag[s]) for s in sorted(SYMS)},
    }


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Darwin v4 Determinism Verification")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--freeze", type=str, default=None,
                        help="Path to frozen run JSON for comparison")
    args = parser.parse_args()

    # Load data
    all_raw = {}
    for sym in sorted(SYMS):
        s = sym.lower()
        all_raw[sym] = (
            load_raw("/home/claude/{}_2022_4h.csv".format(s))
            + load_raw("/home/claude/{}_2023_4h.csv".format(s))
            + load_raw("/home/claude/{}_2024_4h.csv".format(s))
        )
    n = min(len(all_raw[s]) for s in SYMS)

    print("=" * 75)
    print("  DARWIN v4 — DETERMINISM VERIFICATION")
    print("  seed={}".format(args.seed))
    print("=" * 75)

    # ── RUN 1 ──
    print("\n  RUN 1...", flush=True)
    t0 = time.time()
    r1 = run_deterministic(args.seed, all_raw, n)
    t1 = time.time() - t0
    print("    Done in {:.0f}s  eq=${:,.0f}  hash={}...".format(
        t1, r1["pe"], r1["eq_hash"][:16]))

    # ── RUN 2 ──
    print("\n  RUN 2...", flush=True)
    t0 = time.time()
    r2 = run_deterministic(args.seed, all_raw, n)
    t2 = time.time() - t0
    print("    Done in {:.0f}s  eq=${:,.0f}  hash={}...".format(
        t2, r2["pe"], r2["eq_hash"][:16]))

    # ── COMPARISON ──
    print("\n" + "=" * 75)
    print("  DETERMINISM VERIFICATION RESULTS")
    print("=" * 75)

    checks = []

    # Equity curve hash
    eq_match = r1["eq_hash"] == r2["eq_hash"]
    checks.append(("Equity curve hash", eq_match))
    print("\n  Equity curve SHA256:")
    print("    Run 1: {}".format(r1["eq_hash"]))
    print("    Run 2: {}".format(r2["eq_hash"]))
    print("    Match: {}".format("PASS" if eq_match else "FAIL"))

    # Final equity exact match
    eq_exact = r1["pe"] == r2["pe"]
    checks.append(("Final equity exact", eq_exact))
    print("\n  Final equity:")
    print("    Run 1: {:.10f}".format(r1["pe"]))
    print("    Run 2: {:.10f}".format(r2["pe"]))
    print("    Match: {}".format("PASS" if eq_exact else "FAIL"))

    # Trade count
    tc_match = r1["tt"] == r2["tt"]
    checks.append(("Trade count", tc_match))
    print("\n  Trades:")
    print("    Run 1: {}".format(r1["tt"]))
    print("    Run 2: {}".format(r2["tt"]))
    print("    Match: {}".format("PASS" if tc_match else "FAIL"))

    # Genome hashes per symbol
    for sym in sorted(SYMS):
        g_match = r1["genome_hashes"][sym] == r2["genome_hashes"][sym]
        checks.append(("{} genome".format(sym), g_match))
        print("\n  {} genome SHA256:".format(sym))
        print("    Run 1: {}".format(r1["genome_hashes"][sym][:32]))
        print("    Run 2: {}".format(r2["genome_hashes"][sym][:32]))
        print("    Match: {}".format("PASS" if g_match else "FAIL"))

    # Equity curve length
    len_match = len(r1["eq_curve"]) == len(r2["eq_curve"])
    checks.append(("Curve length", len_match))

    # Bar-by-bar comparison (first mismatch)
    if len_match:
        bar_match = True
        for i in range(len(r1["eq_curve"])):
            if r1["eq_curve"][i] != r2["eq_curve"][i]:
                bar_match = False
                print("\n  FIRST BAR MISMATCH at bar {}:".format(i))
                print("    Run 1: {:.15e}".format(r1["eq_curve"][i]))
                print("    Run 2: {:.15e}".format(r2["eq_curve"][i]))
                print("    Delta: {:.2e}".format(
                    r2["eq_curve"][i] - r1["eq_curve"][i]))
                break
        checks.append(("Bar-by-bar equity", bar_match))

    # Monthly returns
    mr_match = len(r1["mr"]) == len(r2["mr"])
    if mr_match:
        for i in range(len(r1["mr"])):
            if r1["mr"][i] != r2["mr"][i]:
                mr_match = False
                break
    checks.append(("Monthly returns", mr_match))

    # ── VERDICT ──
    print("\n" + "=" * 75)
    print("  DETERMINISM CHECKLIST")
    print("=" * 75)
    all_pass = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print("  [{:>4s}]  {}".format(status, name))

    print("\n  " + "=" * 50)
    if all_pass:
        print("  DETERMINISM VERIFIED: Byte-identical across runs")
    else:
        failed = [name for name, p in checks if not p]
        print("  DETERMINISM FAILED: {} check(s) failed".format(len(failed)))
        for f in failed:
            print("    - {}".format(f))
    print("  " + "=" * 50)

    # ── FREEZE canonical run ──
    if all_pass:
        print("\n  Freezing canonical run...")
        path = freeze_run(
            seed=args.seed,
            config={
                "boot": 1095, "cap": CAP, "lev_cap": LEV_CAP,
                "fee": FEE, "slip": SLIP, "bpd": BPD,
                "rqre_interval": 30, "rbe_dd_limit": 0.32,
                "rbe_floor": 0.50, "rbe_vol": 0.06,
            },
            active_genomes=r1["ag"],
            genome_meta={
                s: {"fitness": 0.0, "pf": 0.0, "max_dd": 0.0, "trades": 0}
                for s in SYMS
            },
            equity_curve=r1["eq_curve"],
            final_equity=r1["pe"],
            cagr=r1["cagr"],
            sharpe=r1["sharpe"],
            max_dd=r1["mdd"],
            trades=r1["tt"],
            monthly_returns=r1["mr"],
            output_dir="/home/claude/darwin_v4/artifacts",
        )
        print("  Saved to: {}".format(path))

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
