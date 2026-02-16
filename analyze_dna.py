"""DNA Analysis: B3 × 400 gens × capital_floor=0.45."""
import asyncio, sys, statistics, random
sys.path.insert(0, ".")

from darwin_agent.simulation.harness import (
    SimulationHarness, SimConfig, SimulatedAgent, MonteCarloScenario, Tick,
)
from darwin_agent.evolution.engine import EvolutionEngine, DEFAULT_GENE_RANGES
from darwin_agent.evolution.diagnostics import EvolutionDiagnostics
from darwin_agent.risk.portfolio_engine import PortfolioRiskEngine
from darwin_agent.interfaces.types import PortfolioRiskState

SEEDS = [42, 123, 456, 789, 1024]
ALL_GENES = sorted(DEFAULT_GENE_RANGES.keys())
FOCUS = [
    "regime_bias", "volatility_threshold", "timeframe_bias",
    "risk_pct", "momentum_weight", "mean_rev_weight",
    "confidence_threshold", "leverage_aggression",
    "stop_loss_pct", "take_profit_pct", "breakout_weight", "scalping_weight",
]


async def run_seed(seed):
    cfg = SimConfig(
        scenario=MonteCarloScenario(symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"]),
        generations=400, pool_size=10, trades_per_generation=200,
        starting_capital=100.0, mutation_rate=0.18, survival_rate=0.40,
        seed=seed, capital_floor_pct=0.45,
    )
    random.seed(cfg.seed)
    rng = random.Random(cfg.seed)
    evo = EvolutionEngine(survival_rate=0.40, elitism_count=1, base_mutation_rate=0.18)
    risk = PortfolioRiskEngine(limits=cfg.risk_limits)
    diag = EvolutionDiagnostics()
    scenario = cfg.scenario

    tc = cfg.starting_capital
    risk.update_equity(tc)
    pool = [evo.create_random_dna() for _ in range(cfg.pool_size)]
    am = 1.0
    final_data = []

    for gen in range(cfg.generations):
        gr = random.Random(rng.randint(0, 2**32))
        pd = scenario.generate(cfg.trades_per_generation, gr)
        pac = (tc * am) / max(cfg.pool_size, 1)

        agents = []
        for i, dna in enumerate(pool[:cfg.pool_size]):
            agents.append(SimulatedAgent(
                f"s{seed}-g{gen}-{i}", dna, pac,
                random.Random(gr.randint(0, 2**32))))

        for step in range(cfg.trades_per_generation):
            st = {sym: ticks[step] for sym, ticks in pd.items()}
            for a in agents:
                a.step(st, step)

        ed = [a.get_eval_data() for a in agents]
        pc = sum(e.metrics.capital for e in ed)
        risk.update_equity(pc)
        ps = risk.get_portfolio_state()

        # Capital floor
        fa = cfg.capital_floor_pct * cfg.starting_capital
        if cfg.capital_floor_pct > 0 and pc < fa:
            am = 0.5
            ps.risk_state = PortfolioRiskState.CRITICAL
            se = sorted(ed, key=lambda e: e.metrics.fitness)
            nr = max(1, int(len(se) * 0.4))
            ri = {e.metrics.agent_id for e in se[:nr]}
            rr = random.Random(gr.randint(0, 2**32))
            for i, e in enumerate(ed):
                if e.metrics.agent_id in ri and i < len(pool):
                    os = random.getstate()
                    random.seed(rr.randint(0, 2**32))
                    pool[i] = evo.create_random_dna()
                    random.setstate(os)
        else:
            am = 1.0

        snap = await evo.evaluate_generation(ed, portfolio_snapshot=ps)
        diag.diagnose(gen, ed, pool)

        # Capture final gen
        if gen == cfg.generations - 1:
            for i, e in enumerate(ed):
                genes = dict(pool[i].genes) if i < len(pool) else {}
                final_data.append({
                    "id": e.metrics.agent_id,
                    "fitness": e.metrics.fitness,
                    "capital": e.metrics.capital,
                    "pnl": e.metrics.realized_pnl,
                    "trades": e.metrics.total_trades,
                    "wr": e.metrics.win_rate,
                    "genes": genes,
                })

        sids = evo.select_survivors([e.metrics for e in ed])
        sdna = [e.dna for e in ed if e.metrics.agent_id in sids and e.dna]
        for e in ed:
            if e.dna and e.metrics.agent_id in sids:
                e.dna.fitness = e.metrics.fitness
        pool = evo.create_next_generation(sdna, cfg.pool_size)
        tc = max(pc, 1.0)

    return {"seed": seed, "capital": tc, "ret": (tc/100-1)*100, "agents": final_data}


def gstats(agents):
    out = {}
    for g in ALL_GENES:
        v = [a["genes"].get(g, 0.0) for a in agents if a["genes"]]
        out[g] = (statistics.mean(v) if v else 0, statistics.stdev(v) if len(v) >= 2 else 0)
    return out


async def main():
    print("=" * 92)
    print("  DARWIN v4 — FINAL POPULATION DNA ANALYSIS")
    print("  B3 × 400 gens × floor=0.45 × 16 genes")
    print("=" * 92)

    data = []
    for s in SEEDS:
        d = await run_seed(s)
        data.append(d)
        tag = "WIN" if d["ret"] > 0 else "LOSS"
        print(f"  Seed {s}: ${d['capital']:.2f} ({d['ret']:+.1f}%) [{tag}]")

    # Per-seed analysis
    for d in data:
        ag = d["agents"]
        tag = "PROFITABLE" if d["ret"] > 0 else "LOSS"
        has = any(a["genes"] for a in ag)

        ranked = sorted(ag, key=lambda a: a["fitness"], reverse=True)
        n = len(ranked)
        top = ranked[:max(1, n // 5)]
        bot = ranked[-max(1, n // 5):]

        print("\n" + "─" * 92)
        print(f"  SEED {d['seed']}  │  $100 → ${d['capital']:.2f}  │  {d['ret']:+.1f}%  │  {tag}")
        if has:
            af = statistics.mean([a["fitness"] for a in ag])
            print(f"  Best fitness: {ranked[0]['fitness']:.4f}  "
                  f"Avg: {af:.4f}  Top20: {len(top)} agents  Bot20: {len(bot)} agents")
        print("─" * 92)

        if not has:
            print("  Capital wiped — no gene data available\n")
            continue

        a_s = gstats(ag)
        t_s = gstats(top)
        b_s = gstats(bot)

        print(f"\n  {'Gene':<25s} {'Range':>11s}  │ {'PopMean':>8s} {'Std':>6s}"
              f"  │ {'Top20%':>8s} {'Bot20%':>8s}  │ {'T-B':>7s}")
        print(f"  {'─'*25} {'─'*11}  │ {'─'*8} {'─'*6}"
              f"  │ {'─'*8} {'─'*8}  │ {'─'*7}")

        for g in FOCUS:
            lo, hi = DEFAULT_GENE_RANGES[g]
            am, astd = a_s[g]
            tm, _ = t_s[g]
            bm, _ = b_s[g]
            delta = tm - bm
            print(f"  {g:<25s} [{lo:>4.1f},{hi:>4.1f}]  │ "
                  f"{am:>8.4f} {astd:>6.3f}  │ "
                  f"{tm:>8.4f} {bm:>8.4f}  │ {delta:>+7.4f}")

    # Cross-seed
    prof = [d for d in data if d["ret"] > 0]
    loss = [d for d in data if d["ret"] <= 0]

    print("\n\n" + "=" * 92)
    print("  CROSS-SEED AGGREGATION")
    print("=" * 92)
    print(f"  Profitable: seeds {[d['seed'] for d in prof]} "
          f"(avg return: {statistics.mean([d['ret'] for d in prof]):+.0f}%)" if prof else "")
    print(f"  Loss:       seeds {[d['seed'] for d in loss]}" if loss else "")

    if prof:
        # Collect all top-20% agent genes from profitable seeds
        top_all = {}
        pop_all = {}
        for g in ALL_GENES:
            tv, pv = [], []
            for d in prof:
                ranked = sorted(d["agents"], key=lambda a: a["fitness"], reverse=True)
                top_n = ranked[:max(1, len(ranked) // 5)]
                for a in top_n:
                    if a["genes"]:
                        tv.append(a["genes"].get(g, 0.0))
                for a in d["agents"]:
                    if a["genes"]:
                        pv.append(a["genes"].get(g, 0.0))
            top_all[g] = (statistics.mean(tv) if tv else 0, statistics.stdev(tv) if len(tv) >= 2 else 0)
            pop_all[g] = (statistics.mean(pv) if pv else 0, statistics.stdev(pv) if len(pv) >= 2 else 0)

        print(f"\n  WINNING DNA PROFILE (top 20% from profitable seeds)")
        print(f"\n  {'Gene':<25s} {'Range':>11s}  │ {'PopMean':>8s} {'PopStd':>7s}"
              f"  │ {'Top20%':>8s} {'TopStd':>7s}  │ {'Zone':<6s}")
        print(f"  {'─'*25} {'─'*11}  │ {'─'*8} {'─'*7}"
              f"  │ {'─'*8} {'─'*7}  │ {'─'*6}")

        for g in ALL_GENES:
            lo, hi = DEFAULT_GENE_RANGES[g]
            pm, ps = pop_all[g]
            tm, ts = top_all[g]
            rng = hi - lo
            pos = (tm - lo) / rng if rng > 0 else 0.5
            zone = "LOW" if pos < 0.33 else ("HIGH" if pos > 0.66 else "MID")
            star = " ◀" if g in FOCUS else ""
            print(f"  {g:<25s} [{lo:>4.1f},{hi:>4.1f}]  │ "
                  f"{pm:>8.4f} {ps:>7.3f}  │ "
                  f"{tm:>8.4f} {ts:>7.3f}  │ {zone:<6s}{star}")

    print("\n" + "=" * 92)
    print("  ANALYSIS COMPLETE")
    print("=" * 92)


asyncio.run(main())
