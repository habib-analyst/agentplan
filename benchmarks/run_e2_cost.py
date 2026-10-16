"""E2 cost/latency wins: optimized vs sequential vs hand-optimized."""

import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentplan import workflows, optimize, execute
from agentplan.executor import canonical
from agentplan.optimize import PhysicalPlan

N_RUNS = 10


def hand_physical(plan):
    # a careful engineer's execution: parallel stages, but no planner rewrites
    return PhysicalPlan(stages=plan.stages())


def bench(builder, mode, physical=None, n=N_RUNS):
    toks, walls = [], []
    for _ in range(n):
        p = builder()
        if mode == "sequential":
            res = execute(p, mode="sequential", record=False)
        elif mode == "hand":
            res = execute(p, physical or hand_physical(p), mode="optimized", record=False)
        else:
            opt, phys, _ = optimize(p)
            res = execute(opt, phys, mode="optimized", record=False)
        toks.append(res.tokens_total)
        walls.append(res.wall_total)
    return toks, walls


def main():
    results = {"per_workflow": {}}
    for name in workflows.BUILDERS:
        naive_b = workflows.BUILDERS[name]
        hand_b = workflows.HAND_BUILDERS[name]
        # sanity: hand plan must also be output-equivalent
        seq0 = execute(naive_b(), mode="sequential", record=False)
        hand0 = execute(hand_b(), hand_physical(hand_b()), mode="optimized", record=False)
        assert canonical(hand0.rows) == canonical(seq0.rows), f"hand mismatch {name}"

        st, sw = bench(naive_b, "sequential")
        ot, ow = bench(naive_b, "optimized")
        ht, hw = bench(hand_b, "hand")
        row = {
            "seq_tokens_mean": statistics.mean(st), "seq_wall_mean": statistics.mean(sw),
            "opt_tokens_mean": statistics.mean(ot), "opt_wall_mean": statistics.mean(ow),
            "hand_tokens_mean": statistics.mean(ht), "hand_wall_mean": statistics.mean(hw),
        }
        row["tok_red_vs_seq"] = (row["seq_tokens_mean"] - row["opt_tokens_mean"]) / row["seq_tokens_mean"]
        row["wall_red_vs_seq"] = (row["seq_wall_mean"] - row["opt_wall_mean"]) / row["seq_wall_mean"]
        row["tok_red_vs_hand"] = (row["hand_tokens_mean"] - row["opt_tokens_mean"]) / row["hand_tokens_mean"]
        row["wall_red_vs_hand"] = (row["hand_wall_mean"] - row["opt_wall_mean"]) / row["hand_wall_mean"]
        results["per_workflow"][name] = row
        print(f"{name}: tok -{row['tok_red_vs_seq']*100:.1f}% vs seq, "
              f"-{row['tok_red_vs_hand']*100:.1f}% vs hand | "
              f"wall -{row['wall_red_vs_seq']*100:.1f}% vs seq", flush=True)
    out = os.path.join(os.path.dirname(__file__), "..", "results", "e2.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print("E2 done")


if __name__ == "__main__":
    main()
