"""E4 robustness: optimizer behavior under wrong cost estimates.

Perturb the declared profiles, optimize on the perturbed values, then score
the chosen plan under TRUE costs (regret vs the oracle plan optimized on
true values). Correctness (output equivalence) must hold regardless.
"""

import copy
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentplan import workflows, optimize, execute, plan_cost
from agentplan.executor import canonical


def perturb(plan, kind, seed=0):
    p = plan.copy()
    rng = random.Random(seed)
    for nid, n in p.nodes.items():
        if kind == "cost_x2" and n.op == "agent":
            n.tok_in *= 2; n.tok_out *= 2; n.latency *= 2
        elif kind == "cheap_xhalf" and n.op == "agent" and (n.tok_in + n.tok_out) < 800:
            n.tok_in *= 0.5; n.tok_out *= 0.5; n.latency *= 0.5
        elif kind == "sel_noise" and n.op == "filter" and n.selectivity is not None:
            n.selectivity = min(0.99, max(0.01, n.selectivity + rng.uniform(-0.25, 0.25)))
        elif kind == "lat_lognormal" and n.op == "agent":
            n.latency *= rng.lognormvariate(0, 0.7)
    return p


def restore_true(opt_pert, true_plan):
    for nid, n in opt_pert.nodes.items():
        t = true_plan.nodes[nid]
        n.tok_in, n.tok_out, n.latency, n.dollar = t.tok_in, t.tok_out, t.latency, t.dollar
        n.selectivity = t.selectivity


PERTURBATIONS = ["cost_x2", "cheap_xhalf", "sel_noise", "lat_lognormal"]


def main():
    results = {"per_workflow": {}}
    for name, builder in workflows.BUILDERS.items():
        naive = builder()
        oracle_plan, _, _ = optimize(naive)
        oracle_cost = plan_cost(oracle_plan).total
        seq_rows = canonical(execute(naive, mode="sequential", record=False).rows)
        row = {"oracle_true_cost": oracle_cost, "perturbations": {}}
        for kind in PERTURBATIONS:
            pert = perturb(naive, kind)
            opt_pert, phys_pert, log = optimize(pert)
            # equivalence must hold even with wrong estimates
            res = execute(opt_pert, phys_pert, mode="optimized", record=False)
            equiv = canonical(res.rows) == seq_rows
            restore_true(opt_pert, naive)
            chosen_cost = plan_cost(opt_pert).total
            regret = (chosen_cost - oracle_cost) / oracle_cost if oracle_cost else 0.0
            row["perturbations"][kind] = {
                "equivalent": equiv,
                "regret": regret,
                "rules_fired": sorted({r.rule for r in log}),
            }
            print(f"{name} {kind}: equiv={equiv} regret={regret*100:.2f}%", flush=True)
            assert equiv, f"equivalence broken under {kind} in {name}"
        results["per_workflow"][name] = row
    out = os.path.join(os.path.dirname(__file__), "..", "results", "e4.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print("E4 done: equivalence held under all perturbations")


if __name__ == "__main__":
    main()
