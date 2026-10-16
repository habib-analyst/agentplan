# KILL-GATE RECORD , Project #35 `agentplan`

**Date:** 2026-10-16. **Verdict: PASS** (gap holds; build authorized).

**Claim under test:** nobody has ported the database query planner to agent
orchestration , no open tooling compiles multi-agent workflows to a
cost-based plan where each agent call is a table function with estimated
cost/selectivity and a Selinger-style optimizer applies predicate pushdown,
join reordering, parallelization, and dead-agent pruning.

## What was checked (2026-10-16, before any build code)

**Web search: "cost-based query optimizer multi-agent LLM workflow orchestration planner"**
- No direct hits. Results were unrelated (a trip-planner repo, NL-to-BI-dashboard
  LangGraph demos with SQL query optimizers for *database* queries, not agents).

**Web search: "LLMCompiler parallel agent plan compilation tool calls"**
- LLMCompiler (Kim et al., arXiv 2312.04511, ICML 2024): compiles a single
  agent's tool-call plan into a parallel DAG (planner + task-fetching unit +
  executor). Up to 3.7x latency, 6.7x cost vs ReAct. BUT: no cost model, no
  selectivity estimation, no predicate pushdown, no join reordering, no
  pruning , it parallelizes one fixed plan; it does not *choose* between
  plans by cost. Targets single-agent function calling, not multi-agent
  orchestration.

**Web search: "Lotus semantic operators query optimizer LLM declarative cascades"**
- LOTUS / Semantic Operators (Patel et al., arXiv 2407.11418, Stanford/Berkeley,
  open source at github.com/stanford-futuredata/lotus): declarative semantic
  operators (sem_filter, sem_join, ...) over text corpora with model cascades
  and filter/join optimizations (up to 400x on analytics). Closest in spirit.
  BUT: it is a batch *analytics engine over static corpora*, not a planner for
  multi-agent *orchestration*. No agent-call-as-table-function IR, no
  cost-based choice between orchestration plans, no guardrail/filter pushdown
  across agents, no dead-agent elimination, no output-equivalence invariant
  between naive and planned execution.

**Web search: "github predicate pushdown join reordering agent workflow optimizer LLM pipeline"**
- No matching open tooling. Hits were corporate PySpark standards docs and
  unrelated content.

**Known adjacent systems (from training knowledge, verified against the above):**
- **Palimpzest / Abacus**: cost-based optimization over models, prompts, and
  operator variants , but for declarative AI *analytics pipelines* in offline
  batch settings, not multi-agent workflows.
- **DocETL**: LLM-in-the-loop pipeline rewriting for document processing.
- **AFlow**: MCTS/search-based agentic workflow optimization , search, not
  cost-based query planning; no IR, no equivalence invariant.

## Why the gap holds

The specific combination agentplan implements has no open equivalent:
(1) an IR mapping agent calls to table functions, handoffs to joins,
guardrails to filters, fan-out to parallel scans; (2) a cost model
`cost = alpha*expected_tokens + beta*latency + gamma*dollar_cost` with
selectivity from contracts or measured history; (3) a rule-based optimizer
with predicate pushdown, join reordering with semi-join reduction,
route-conditional pruning, and dead-agent elimination; (4) an executor with
the proven invariant OPTIMIZED PLAN == SEQUENTIAL PLAN on outputs.

Closest prior art is cited as such in PAPER.md (Related work). The novelty
claim is scoped honestly: the *combination and the agent-orchestration
target* are new; the individual database techniques are 50 years old, which
is exactly the point.
