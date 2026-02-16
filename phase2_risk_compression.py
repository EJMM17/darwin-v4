#!/usr/bin/env python3
"""
Darwin v4 — Phase 2: Risk Compression A/B/C Experiment.

Deterministic. Frozen genomes. No architecture changes.

A) BASELINE:      flat 5x, RBE dd_limit=0.32
B) LOWER LEV:     lev * 0.75, RBE dd_limit=0.32
C) LOWER + EARLY: lev * 0.75, RBE dd_limit=0.28
"""

import sys, os
os.environ["PYTHONHASHSEED"] = "42"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import csv, math, statistics, json, time, random

from darwin_agent.determinism import (
    lock_determinism, reset_id_counter, genome_hash, equity_curve_hash,
)
from darwin_agent.freeze import lock_all_determinism
from darwin_agent.portfolio.global_regime import GlobalRegimeThrottle
from darwin_agent.portfolio.allocation_engine import PortfolioAllocationEngine, PAEConfig
from darwin_agent.portfolio.risk_budget_engine import RiskBudgetEngine, RBEConfig

SYMS = ["BTC", "DOGE", "SOL"]  # sorted
SEED = 42
BPD = 6
CAP = 800.0
LEV_CAP = 5.0
FEE = 0.0007
SLIP = 0.0002
DAILY_LIM = 0.03
WEEKLY_LIM = 0.08
MONTHLY_LIM = 0.20
BOOT = 1095

def ema(d, p):
    o = [0.0]*len(d); k = 2.0/(p+1); o[0] = d[0]
    for i in range(1, len(d)): o[i] = d[i]*k + o[i-1]*(1.0-k)
    return o

def atr_f(h, l, c, p=14):
    n = len(c); tr = [h[0]-l[0]]
    for i in range(1,n): tr.append(max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])))
    o = [0.0]*n
    if n >= p: o[p-1] = sum(tr[:p])/float(p)
    for i in range(p,n): o[i] = (o[i-1]*(p-1)+tr[i])/float(p)
    return o

def rsi_f(c, p=14):
    n = len(c); o = [50.0]*n; g = 0.0; lo = 0.0
    for i in range(1, min(p+1,n)):
        d = c[i]-c[i-1]
        if d > 0: g += d
        else: lo -= d
    if p > 0: g /= float(p); lo /= float(p)
    if g+lo > 0: o[p] = 100.0*g/(g+lo)
    for i in range(p+1, n):
        d = c[i]-c[i-1]; gg = d if d > 0 else 0.0; ll = -d if d < 0 else 0.0
        g = (g*(p-1)+gg)/float(p); lo = (lo*(p-1)+ll)/float(p)
        o[i] = 100.0*g/(g+lo) if g+lo > 0 else 50.0
    return o

def load_raw(p):
    with open(p) as f: return list(csv.DictReader(f))

def dlev(gm, rm):
    c = gm * rm
    if c > 0.5625: return min(4.0 + (c-0.5625)/(1.0-0.5625), 5.0)
    return 2.0 + c/0.5625

def sig(g, d, i):
    r = 0.0
    r += g.get("momentum_weight",0.0)*(1.0 if d["ef"][i]>d["em"][i] else -1.0)
    r += g.get("ema_weight",0.0)*(1.0 if d["ef"][i]>d["es"][i] else -1.0)
    r += g.get("rsi_weight",0.0)*((d["ri"][i]-50.0)/50.0)
    r += g.get("mean_rev_weight",0.0)*((50.0-d["ri"][i])/50.0)
    bk = (d["c"][i]-d["em"][i])/d["at"][i] if d["at"][i]>0 else 0.0
    r += g.get("breakout_weight",0.0)*bk
    r += g.get("trend_strength_weight",0.0)*(abs(d["ef"][i]-d["es"][i])/d["c"][i])
    r += g.get("regime_bias",0.0)*((d["c"][i]-d["e2"][i])/d["e2"][i] if d["e2"][i]>0 else 0.0)
    return math.tanh(r)


def run_config(label, genomes, ind, gmrt_r, n_bars, lev_mult, rbe_dd_limit):
    """Run a single configuration. Fully deterministic."""
    # Reset RNG to known state for portfolio sim
    lock_all_determinism(SEED)
    random.seed(SEED + 1000)
    reset_id_counter(SEED + 1000)

    pae = PortfolioAllocationEngine(
        sorted(SYMS),
        config=PAEConfig(lookback=200, warmup_bars=50, ema_alpha=0.03,
                         min_weight=0.05, max_weight=0.70),
    )
    rbe = RiskBudgetEngine(
        RBEConfig(dd_limit=rbe_dd_limit, target_vol=0.06, window=200,
                  warmup=30, mult_floor=0.50, mult_ceil=1.00),
    )

    pe = CAP
    se = {s: CAP/3.0 for s in sorted(SYMS)}
    sp = {s: 0.0 for s in sorted(SYMS)}
    pk = CAP; mdd = 0.0
    ds = CAP; ws = CAP; ms = CAP
    dh = False; wh = False; mh = False
    mr = []; cm = 0; tf_total = 0.0; tt = 0
    eq_curve = [CAP]
    lev_samples = []
    ag = {s: dict(genomes[s]) for s in sorted(SYMS)}

    for i in range(BOOT, n_bars):
        bc = i - BOOT; bod = bc % BPD; month = bc // 182

        if bod == 0: ds = pe; dh = False
        if bc % (BPD*7) == 0: ws = pe; wh = False
        if month > cm:
            if cm > 0 or bc > 182:
                mr.append((pe - ms)/ms*100.0 if ms > 0 else 0.0)
            ms = pe; mh = False; cm = month

        dl2 = (ds-pe)/ds if ds > 0 else 0.0
        wl2 = (ws-pe)/ws if ws > 0 else 0.0
        ml2 = (ms-pe)/ms if ms > 0 else 0.0
        if dl2 >= DAILY_LIM and not dh: dh = True
        if wl2 >= WEEKLY_LIM and not wh: wh = True
        if ml2 >= MONTHLY_LIM and not mh: mh = True
        halted = dh or wh or mh

        gi = min(i*4, len(gmrt_r.bars)-1)
        gm = gmrt_r.multiplier_at(gi)
        rm = rbe.last_mult
        lv = dlev(gm, rm) * lev_mult if not halted else 0.0
        lv = min(lv, LEV_CAP)  # hard cap still 5x
        if not halted:
            lev_samples.append(lv)

        if not halted:
            wt = pae.get_weights(); tp = 0.0
            for s in sorted(SYMS):
                dd = ind[s]; p = dd["c"][i]; pp = dd["c"][i-1]
                if p <= 0.0 or pp <= 0.0: continue
                sg_v = sig(ag[s], dd, i)
                nt = pe * wt[s] * lv * abs(sg_v)
                nt = min(nt, se[s] * LEV_CAP)
                tg = math.copysign(nt, sg_v)
                dt = tg - sp[s]
                if abs(dt) > abs(sp[s])*0.10 + 5.0:
                    fee = abs(dt)*(FEE+SLIP); tf_total += fee; tt += 1
                    pnl = sp[s]*((p-pp)/pp) - fee if abs(sp[s]) > 1.0 else -fee
                    sp[s] = tg
                else:
                    pnl = sp[s]*((p-pp)/pp) if abs(sp[s]) > 1.0 else 0.0
                se[s] += pnl; se[s] = max(se[s], 1.0); tp += pnl
                pae.update(s, equity=se[s], pnl=pnl if abs(pnl) > 0.01 else 0.0)
            pe = sum(se[s] for s in sorted(SYMS))
        else:
            tp = 0.0
            for s in sorted(SYMS):
                pae.update(s, equity=se[s], pnl=0.0)

        if pe > pk: pk = pe
        ddp = (pk-pe)/pk if pk > 0 else 0.0
        mdd = max(mdd, ddp)
        eq_curve.append(pe)
        rbe.update(equity=pe, pnl=tp if abs(tp) > 0.01 else 0.0)
        rbe.step(); pae.step()
        if bod == 0 and not halted:
            w = pae.get_weights()
            for s in sorted(SYMS): se[s] = pe * w[s]

    if ms > 0: mr.append((pe-ms)/ms*100.0)

    td = (n_bars - BOOT) // BPD
    cagr = ((pe/CAP)**(365.0/td) - 1.0)*100.0 if td > 0 else 0.0
    sh = 0.0
    if len(mr) >= 2:
        mu = statistics.mean(mr); sigma = statistics.stdev(mr)
        sh = mu/sigma*math.sqrt(12.0) if sigma > 0 else 0.0

    # Worst 30-day rolling return (approx: worst consecutive ~5 months = 5 entries)
    # Actually 30 days = ~1 month, so worst single monthly return
    worst_30d = min(mr) if mr else 0.0

    avg_lev = statistics.mean(lev_samples) if lev_samples else 0.0

    genome_shas = {s: genome_hash(ag[s]) for s in sorted(SYMS)}
    eq_sha = equity_curve_hash(eq_curve)

    return {
        "label": label,
        "pe": pe, "cagr": cagr, "mdd": mdd*100.0, "sharpe": sh,
        "worst_30d": worst_30d, "tt": tt, "avg_lev": avg_lev,
        "eq_hash": eq_sha, "genome_shas": genome_shas,
        "eq_curve": eq_curve, "mr": mr, "fees": tf_total,
        "lev_mult": lev_mult, "rbe_dd": rbe_dd_limit,
    }


def main():
    # Load frozen genomes
    frozen_path = "/home/claude/darwin_v4/artifacts/frozen_run_20260216_030323.json"
    with open(frozen_path) as f:
        frozen = json.load(f)
    genomes = {g["symbol"]: g["genes"] for g in frozen["genomes"]}
    frozen_eq_hash = frozen["equity_curve_sha256"]
    frozen_pool_hash = frozen["pool_sha256"]

    # Verify genome identity
    for g in frozen["genomes"]:
        actual = genome_hash(genomes[g["symbol"]])
        assert actual == g["genome_sha256"], \
            "Genome mismatch for {}: {} != {}".format(g["symbol"], actual, g["genome_sha256"])
    print("Genome identity verified against frozen artifact.", flush=True)

    # Load market data
    all_raw = {}
    for sym in sorted(SYMS):
        s = sym.lower()
        all_raw[sym] = (
            load_raw("/home/claude/{}_2022_4h.csv".format(s))
            + load_raw("/home/claude/{}_2023_4h.csv".format(s))
            + load_raw("/home/claude/{}_2024_4h.csv".format(s))
        )
    n = min(len(all_raw[s]) for s in SYMS)

    # Build indicators (deterministic, pure math)
    ind = {}
    for sym in sorted(SYMS):
        c = [float(r["close"]) for r in all_raw[sym][:n]]
        h = [float(r["high"]) for r in all_raw[sym][:n]]
        l = [float(r["low"]) for r in all_raw[sym][:n]]
        ind[sym] = {
            "c": c, "h": h, "l": l,
            "ef": ema(c,12), "em": ema(c,50), "es": ema(c,48),
            "ri": rsi_f(c,14), "at": atr_f(h,l,c,14), "e2": ema(c,200),
        }

    gmrt = GlobalRegimeThrottle()
    gmrt.load_btc("/home/claude/btc_2022_2024_1h.csv")
    gmrt_r = gmrt.compute()

    # ═══ RUN A/B/C ═══
    configs = [
        ("A_BASELINE",    1.00, 0.32),
        ("B_LOW_LEV",     0.75, 0.32),
        ("C_LOW_LEV_RBE", 0.75, 0.28),
    ]

    results = []
    for label, lm, rbe_dd in configs:
        print("\nRunning {}  (lev_mult={}, rbe_dd={})...".format(label, lm, rbe_dd), flush=True)
        t0 = time.time()
        r = run_config(label, genomes, ind, gmrt_r, n, lm, rbe_dd)
        elapsed = time.time() - t0
        results.append(r)
        print("  Done in {:.0f}s  eq=${:,.0f}  hash={}...".format(
            elapsed, r["pe"], r["eq_hash"][:16]))

    A, B, C = results

    # ═══ VERIFY A matches frozen baseline ═══
    a_matches = A["eq_hash"] == frozen_eq_hash
    print("\n" + "="*80)
    print("  BASELINE VERIFICATION")
    print("="*80)
    print("  Config A eq_hash: {}".format(A["eq_hash"][:32]))
    print("  Frozen   eq_hash: {}".format(frozen_eq_hash[:32]))
    print("  Match: {}".format("PASS" if a_matches else "FAIL"))
    if not a_matches:
        print("  WARNING: Config A does not match frozen baseline!")
        print("  A equity: ${:,.2f}  Frozen equity: ${:,.2f}".format(
            A["pe"], frozen["final_equity"]))

    # Verify genome hashes match across all configs
    print("\n  Genome hash consistency:")
    for s in sorted(SYMS):
        a_g = A["genome_shas"][s]
        b_g = B["genome_shas"][s]
        c_g = C["genome_shas"][s]
        match = a_g == b_g == c_g
        print("    {}: {} {}".format(s, a_g[:24], "PASS" if match else "FAIL"))

    # ═══ COMPARISON TABLE ═══
    print("\n" + "="*80)
    print("  PHASE 2: RISK COMPRESSION — A/B/C RESULTS")
    print("="*80)
    print()
    hdr = "  {:<22s} {:>14s} {:>14s} {:>14s}".format(
        "Metric", "A (baseline)", "B (low lev)", "C (low+RBE)")
    print(hdr)
    print("  " + "-"*64)

    def fmt_eq(v): return "${:,.0f}".format(v)
    def fmt_pct(v): return "{:.1f}%".format(v)
    def fmt_sh(v): return "{:.2f}".format(v)
    def fmt_lev(v): return "{:.2f}x".format(v)
    def fmt_delta(v): return "{:+.2f}".format(v)
    def fmt_deltap(v): return "{:+.1f}pp".format(v)

    rows = [
        ("Final Equity",  fmt_eq(A["pe"]),  fmt_eq(B["pe"]),  fmt_eq(C["pe"])),
        ("CAGR",          fmt_pct(A["cagr"]), fmt_pct(B["cagr"]), fmt_pct(C["cagr"])),
        ("Sharpe",        fmt_sh(A["sharpe"]), fmt_sh(B["sharpe"]), fmt_sh(C["sharpe"])),
        ("MaxDD",         fmt_pct(A["mdd"]),  fmt_pct(B["mdd"]),  fmt_pct(C["mdd"])),
        ("Worst 30d",     fmt_pct(A["worst_30d"]), fmt_pct(B["worst_30d"]), fmt_pct(C["worst_30d"])),
        ("Avg Leverage",  fmt_lev(A["avg_lev"]), fmt_lev(B["avg_lev"]), fmt_lev(C["avg_lev"])),
        ("Trades",        str(A["tt"]),      str(B["tt"]),      str(C["tt"])),
        ("Fees",          fmt_eq(A["fees"]), fmt_eq(B["fees"]), fmt_eq(C["fees"])),
    ]
    for name, va, vb, vc in rows:
        print("  {:<22s} {:>14s} {:>14s} {:>14s}".format(name, va, vb, vc))

    print("  " + "-"*64)
    print("  {:<22s} {:>14s} {:>14s} {:>14s}".format(
        "dSharpe vs A", "—",
        fmt_delta(B["sharpe"]-A["sharpe"]),
        fmt_delta(C["sharpe"]-A["sharpe"])))
    print("  {:<22s} {:>14s} {:>14s} {:>14s}".format(
        "dMaxDD vs A", "—",
        fmt_deltap(B["mdd"]-A["mdd"]),
        fmt_deltap(C["mdd"]-A["mdd"])))
    print("  {:<22s} {:>14s} {:>14s} {:>14s}".format(
        "dCAGR vs A", "—",
        fmt_deltap(B["cagr"]-A["cagr"]),
        fmt_deltap(C["cagr"]-A["cagr"])))

    # ═══ ACCEPTANCE CRITERIA ═══
    print("\n" + "="*80)
    print("  ACCEPTANCE CRITERIA")
    print("="*80)

    best = None
    for r in results:
        lbl = r["label"]
        s1 = r["sharpe"] >= 1.05
        s2 = r["mdd"] <= A["mdd"] - 10.0  # MaxDD reduced by >=10pp
        s3 = r["cagr"] >= 25.0
        score = sum([s1, s2, s3])

        print("\n  {} (lev_mult={}, rbe_dd={}):".format(lbl, r["lev_mult"], r["rbe_dd"]))
        print("    Sharpe >= 1.05:       {:.2f}  {}".format(r["sharpe"], "PASS" if s1 else "FAIL"))
        print("    MaxDD reduced >=10pp: {:.1f}% (need <={:.1f}%)  {}".format(
            r["mdd"], A["mdd"]-10.0, "PASS" if s2 else "FAIL"))
        print("    CAGR >= 25%:          {:.1f}%  {}".format(r["cagr"], "PASS" if s3 else "FAIL"))
        print("    Score: {}/3".format(score))

        if s1 and s3 and (best is None or r["sharpe"] > best["sharpe"]):
            best = r

    # ═══ VERDICT ═══
    print("\n" + "="*80)
    if best is not None:
        print("  VERDICT: Selected Config = {}".format(best["label"]))
        print("  Sharpe {:.2f}, MaxDD {:.1f}%, CAGR {:.1f}%".format(
            best["sharpe"], best["mdd"], best["cagr"]))
    else:
        # Check if any config improved Sharpe meaningfully
        best_sh = max(results, key=lambda r: r["sharpe"])
        if best_sh["sharpe"] > A["sharpe"] and best_sh["cagr"] >= 25.0:
            print("  VERDICT: Partial improvement found — {}".format(best_sh["label"]))
            print("  Sharpe {:.2f} (target 1.05), MaxDD {:.1f}%, CAGR {:.1f}%".format(
                best_sh["sharpe"], best_sh["mdd"], best_sh["cagr"]))
            print("  Primary objective (Sharpe >= 1.05) NOT MET.")
        else:
            print("  VERDICT: Risk compression FAILED.")
            print("  No configuration meets acceptance criteria.")
    print("="*80)

    # Monthly returns detail
    print("\n  Monthly Returns Detail:")
    print("  {:<8s} {:>10s} {:>10s} {:>10s}".format("Month", "A", "B", "C"))
    for i in range(max(len(A["mr"]), len(B["mr"]), len(C["mr"]))):
        va = "{:+.1f}%".format(A["mr"][i]) if i < len(A["mr"]) else "—"
        vb = "{:+.1f}%".format(B["mr"][i]) if i < len(B["mr"]) else "—"
        vc = "{:+.1f}%".format(C["mr"][i]) if i < len(C["mr"]) else "—"
        print("  {:<8s} {:>10s} {:>10s} {:>10s}".format("M{}".format(i+1), va, vb, vc))

    print()


if __name__ == "__main__":
    main()
