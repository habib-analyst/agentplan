"""agentplan CLI: inspect plans, run the optimizer, run benchmarks."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentplan import workflows, optimize, execute, plan_cost, canonical
from agentplan.optimize import PhysicalPlan
from agentplan.stats import StatsStore
from agentplan.viz import render_before_after


def cmd_plan(args):
    p = workflows.BUILDERS[args.workflow]()
    print(f"workflow: {p.name}  nodes: {len(p.nodes)}")
    for nid in p.topo():
        n = p.nodes[nid]
        print(f"  [{n.op:7s}] {n.name:18s} <- {n.inputs}")


def cmd_optimize(args):
    naive = workflows.BUILDERS[args.workflow]()
    stats = StatsStore(args.stats) if args.stats else None
    t0 = time.perf_counter()
    opt, phys, log = optimize(naive, stats)
    dt = time.perf_counter() - t0
    c0 = plan_cost(naive, stats).total
    c1 = plan_cost(opt, stats).total
    print(f"rules fired: {len(log)}   optimize time: {dt * 1000:.1f} ms")
    for r in log:
        print(f"  [{r.rule}] {r.detail}")
    print(f"estimated cost: {c0:.0f} -> {c1:.0f}  ({100 * (c0 - c1) / c0:.1f}% reduction)")
    print(f"pruned: {[naive.nodes[i].name for i in phys.pruned if i in naive.nodes]}")
    if args.viz:
        notes = {r.detail.split('`')[0][:40]: r.rule for r in log}
        html = render_before_after(naive, opt, phys,
                                   {r.rule: r.detail for r in log})
        with open(args.viz, "w") as f:
            f.write(html)
        print(f"wrote {args.viz}")


def cmd_run(args):
    naive = workflows.BUILDERS[args.workflow]()
    stats = StatsStore(args.stats) if args.stats else None
    if args.mode == "sequential":
        res = execute(naive, mode="sequential", stats=stats, record=False)
    else:
        opt, phys, log = optimize(naive, stats)
        res = execute(opt, phys, mode="optimized", stats=stats, record=False)
    print(f"mode={args.mode} rows={len(res.rows)} tokens={res.tokens_total:.0f} "
          f"wall={res.wall_total:.3f}s")
    print(json.dumps(res.rows[:3], indent=1, default=str))


def main():
    ap = argparse.ArgumentParser(prog="agentplan")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("plan"); a.add_argument("workflow", choices=workflows.BUILDERS)
    a.set_defaults(f=cmd_plan)
    b = sub.add_parser("optimize"); b.add_argument("workflow", choices=workflows.BUILDERS)
    b.add_argument("--stats", default=None); b.add_argument("--viz", default=None)
    b.set_defaults(f=cmd_optimize)
    c = sub.add_parser("run"); c.add_argument("workflow", choices=workflows.BUILDERS)
    c.add_argument("--mode", choices=["sequential", "optimized"], default="optimized")
    c.add_argument("--stats", default=None)
    c.set_defaults(f=cmd_run)
    args = ap.parse_args()
    args.f(args)


if __name__ == "__main__":
    main()
