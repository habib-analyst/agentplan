# agentplan

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)

**Your multi-agent workflow, query-planned.** Agent frameworks run steps in fixed order, like databases in 1975. agentplan compiles a workflow to a *query plan* , each agent call is a table function with estimated cost and selectivity , and a cost-based optimizer applies 50 years of database theory: predicate pushdown, join reordering, parallel scans, dead-agent pruning. Proven output-equivalent to the naive run.

## Quickstart

```bash
pip install agentplan
```

```python
from agentplan import workflows, optimize, execute, plan_cost
from agentplan.executor import canonical

naive = workflows.build_data_pipeline()          # fixed order, as written
opt, phys, log = optimize(naive)                 # cost-based rewrite
for r in log:
    print(f"[{r.rule}] {r.detail}")

seq = execute(naive, mode="sequential", record=False)
res = execute(opt, phys, mode="optimized", record=False)
assert canonical(res.rows) == canonical(seq.rows)  # the invariant
```

Real output (data_pipeline workflow):

```
naive plan: 9 nodes, est. cost 118378
optimized: 9 nodes, est. cost 108778
  [pushdown] filter 'validator' pushed before agent 'enricher'
  [reorder] join 'join_enrich': probe 'enricher' first; semi-join reduces sources ['ref_table']
  [parallel] 9 execution stages; 9 nodes
outputs identical: 1 rows
tokens: 118320 -> 105420 (10.9% saved)
wall:   2.39s -> 0.39s (83.5% saved)
```

See `examples/quickstart.py`. Before/after plan diagrams: `viz_support_triage_before_after.html`, `viz_data_pipeline_before_after.html`.

## Measured results

Five realistic multi-agent workflows, simulated deterministic agents on CPU ($0 spend; token figures are cost-model estimates, wall-clock measured on simulated latencies). Means over 10 runs per cell; full tables in `benchmarks/results/SUMMARY.md`.

| workflow | token reduction vs sequential | wall-clock reduction vs sequential | token reduction vs hand-optimized |
|---|---|---|---|
| research_assistant | 27.9% | 90.6% | 7.9% |
| data_pipeline | 10.9% | 83.7% | 3.0% |
| support_triage | 56.3% | 96.2% | 56.3% |
| code_review | 34.1% | 85.9% | 7.9% |
| medical_intake | 71.9% | 92.5% | 32.3% |

Correctness: **200/200 workflow runs** produce byte-identical outputs between optimized and sequential plans. Optimizer overhead: < 1 ms per workflow (< 0.1% of workflow cost). Sensitivity analysis: output equivalence holds under all estimate perturbations; mean cost regret of wrong estimates under 2% (see PAPER.md).

Why the wins differ: Amdahl's law for agents. The optimizer can only save the fraction of cost sitting in *optimizable* steps , support_triage wastes 2/3 of specialist calls on rows routed elsewhere (route pruning recovers it); data_pipeline's naive plan already validated before aggregating, so only the enricher pushdown and semi-join reduction apply. Honest numbers, all of them.

## How it works

```
workflow  -->  logical plan  -->  optimized physical plan  -->  executor
  |               |                       |                        |
  agent call      table function          predicate pushdown         staged,
  handoff         join                    join reorder + semi-join   parallel,
  guardrail       filter                  route pruning              purity-
  fan-out         parallel scan           dead-agent pruning         checked
```

Cost model: `cost = alpha * expected_tokens + beta * latency + gamma * dollar_cost`, with selectivity from declared contracts or measured run history (auto-learned after 3 runs). The four rewrite rules are provably output-preserving (proof sketches in PAPER.md); the executor *enforces* the purity precondition at runtime, failing loudly instead of miscompiling.

## Citation

```bibtex
@software{rehman2026agentplan,
  author = {Rehman, Habib Ur},
  title = {agentplan: Cost-Based Query Planning for Multi-Agent Orchestration},
  year = {2026},
  url = {https://github.com/habib-analyst/agentplan}
}
```

---
Built by [Habib Ur Rehman](https://github.com/habib-analyst) · ML/AI Engineer & healthcare-AI researcher ·
[LinkedIn](https://www.linkedin.com/in/hur-dev) · [habib.top](https://habib.top)
