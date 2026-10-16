"""agentplan quickstart: compile a workflow, optimize it, run it."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentplan import workflows, optimize, execute, plan_cost
from agentplan.executor import canonical

# 1. pick a workflow (naive fixed order, as a straightforward implementation)
naive = workflows.build_data_pipeline()
print(f"naive plan: {len(naive.nodes)} nodes, "
      f"est. cost {plan_cost(naive).total:.0f}")

# 2. optimize: predicate pushdown, join reordering, parallelization, pruning
opt, phys, log = optimize(naive)
print(f"optimized: {len(opt.nodes)} nodes, "
      f"est. cost {plan_cost(opt).total:.0f}")
for r in log:
    print(f"  [{r.rule}] {r.detail}")

# 3. run both; the invariant is OPTIMIZED == SEQUENTIAL on outputs
seq = execute(naive, mode="sequential", record=False)
res = execute(opt, phys, mode="optimized", record=False)
assert canonical(res.rows) == canonical(seq.rows)
print(f"outputs identical: {len(res.rows)} rows")
print(f"tokens: {seq.tokens_total:.0f} -> {res.tokens_total:.0f} "
      f"({100*(seq.tokens_total-res.tokens_total)/seq.tokens_total:.1f}% saved)")
print(f"wall:   {seq.wall_total:.2f}s -> {res.wall_total:.2f}s "
      f"({100*(seq.wall_total-res.wall_total)/seq.wall_total:.1f}% saved)")
print("sink output:", res.rows)
