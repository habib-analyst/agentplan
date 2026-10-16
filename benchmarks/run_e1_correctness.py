"""E1 correctness: optimized plan outputs match sequential on 200 workflow runs."""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentplan import workflows, optimize, execute
from agentplan.executor import canonical

N_PER_WORKFLOW = 40


def main():
    results = {"n_total": 0, "n_match": 0, "per_workflow": {}, "mismatches": []}
    t0 = time.perf_counter()
    for name, builder in workflows.BUILDERS.items():
        wf = {"runs": 0, "match": 0, "seq_wall": [], "opt_wall": []}
        for i in range(N_PER_WORKFLOW):
            naive = builder()
            seq = execute(naive, mode="sequential", record=False)
            opt_plan, phys, _ = optimize(naive)
            t1 = time.perf_counter()
            res = execute(opt_plan, phys, mode="optimized", record=False)
            wf["opt_wall"].append(time.perf_counter() - t1)
            wf["seq_wall"].append(seq.wall_total)
            wf["runs"] += 1
            results["n_total"] += 1
            if canonical(res.rows) == canonical(seq.rows):
                wf["match"] += 1
                results["n_match"] += 1
            else:
                results["mismatches"].append({"workflow": name, "run": i})
        wf["seq_wall_mean"] = sum(wf["seq_wall"]) / len(wf["seq_wall"])
        wf["opt_wall_mean"] = sum(wf["opt_wall"]) / len(wf["opt_wall"])
        results["per_workflow"][name] = wf
        print(f"{name}: {wf['match']}/{wf['runs']} match", flush=True)
    results["elapsed_s"] = time.perf_counter() - t0
    out = os.path.join(os.path.dirname(__file__), "..", "results", "e1.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"E1: {results['n_match']}/{results['n_total']} exact matches "
          f"({results['elapsed_s']:.0f}s)")


if __name__ == "__main__":
    main()
