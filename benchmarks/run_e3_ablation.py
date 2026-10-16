"""E3 ablation: disable each optimizer rule in turn; measure what it contributed."""

import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentplan import workflows, optimize, execute

RULES = ["pushdown", "reorder", "parallel", "prune"]
N_RUNS = 6


def run_cost(builder, rules, parallel_workers):
    toks, walls = [], []
    for _ in range(N_RUNS):
        p = builder()
        opt, phys, _ = optimize(p, rules=rules)
        res = execute(opt, phys, mode="optimized",
                      max_workers=parallel_workers, record=False)
        toks.append(res.tokens_total)
        walls.append(res.wall_total)
    return statistics.mean(toks), statistics.mean(walls)


def main():
    results = {"per_workflow": {}}
    for name, builder in workflows.BUILDERS.items():
        full = {"pushdown", "reorder", "parallel", "prune"}
        base_t, base_w = run_cost(builder, full, 8)
        row = {"full": {"tokens": base_t, "wall": base_w}, "ablations": {}}
        for rule in RULES:
            rules = set(full) - {rule}
            workers = 1 if rule == "parallel" else 8
            t, w = run_cost(builder, rules, workers)
            row["ablations"][f"no_{rule}"] = {
                "tokens": t, "wall": w,
                "tok_delta_vs_full": (t - base_t) / base_t,
                "wall_delta_vs_full": (w - base_w) / base_w,
            }
            print(f"{name} no_{rule}: tokens {t:.0f} ({(t-base_t)/base_t*100:+.1f}%), "
                  f"wall {w:.2f}s ({(w-base_w)/base_w*100:+.1f}%)", flush=True)
        results["per_workflow"][name] = row
    out = os.path.join(os.path.dirname(__file__), "..", "results", "e3.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print("E3 done")


if __name__ == "__main__":
    main()
