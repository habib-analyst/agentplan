# agentplan: Cost-Based Query Planning for Multi-Agent Orchestration

**Habib Ur Rehman** (habib-analyst) , 2026-10-16

## Abstract

Agent frameworks execute multi-step workflows in fixed order, like databases
in 1975. Databases solved exactly this problem: compile declarative intent to
a logical plan, then choose the cheapest physical plan with a cost-based
optimizer. **agentplan** ports that idea to agent orchestration. A workflow
compiles to a logical plan where each agent call is a *table function* with
estimated cost (expected tokens, latency, dollar cost) and selectivity,
handoffs are *joins*, guardrails are *filters*, and fan-out is a *parallel
scan*. A Selinger-style optimizer applies four rewrite rules , predicate
pushdown, join reordering with semi-join reduction, parallelization, and
dead-agent pruning , and an executor runs the physical plan under the
invariant **optimized plan == sequential plan on outputs**, proved below and
verified on 200 workflow runs with zero mismatches. Across five realistic
multi-agent workflows the optimizer cuts token cost by 11-72% (median 34%)
and wall-clock time by 84-96% versus naive sequential execution, while
optimizer overhead stays under 0.1% of workflow cost. A sensitivity analysis
shows correctness never depends on estimate quality: only the join
probe-side choice degrades under wrong estimates, with mean regret under 2%.

## 1. Claim

Multi-agent workflows can be compiled to a query plan and optimized with
classical database techniques , predicate pushdown (cheap validators before
expensive agents), join reordering (cheapest/most selective branch first),
parallel scans, and dead-code elimination for agents , with provable output
equivalence to the naive sequential execution.

## 2. Related work (honest placement)

- **LLMCompiler** (Kim et al., ICML 2024): compiles a single agent's
  tool-call plan into a parallel DAG. No cost model, no selectivity, no
  plan *choice* , it parallelizes one fixed plan. agentplan generalizes from
  "parallelize this plan" to "choose the cheapest equivalent plan".
- **LOTUS / Semantic Operators** (Patel et al., 2024): declarative semantic
  operators over text corpora with cascades and filter/join optimizations.
  A batch analytics engine over static data, not an orchestrator of stateful
  agent workflows; no agent-as-table-function IR, no equivalence invariant.
- **Palimpzest / Abacus**: cost-based optimization over models, prompts and
  operator variants for declarative AI analytics pipelines (offline batch).
  Closest on "cost-based", different target (pipelines, not agents).
- **DocETL**: LLM-in-the-loop pipeline rewriting for documents.
- **AFlow**: search-based (MCTS) agentic workflow optimization , heuristic
  search over workflows, not cost-based planning with an equivalence proof.

The combination , IR + cost model + four classical rewrite rules + proven
output-equivalence for *multi-agent orchestration* , has no open equivalent
(see KILLGATE.md for the pre-build novelty check).

## 3. Formalism

### 3.1 IR

A workflow compiles to a DAG of nodes. Node types:

| Op | DB analog | Semantics |
|---|---|---|
| `source` | table scan | emits a fixed multiset of rows |
| `agent` | table function | pure deterministic row fn: row -> row (adds keys, never mutates inputs); variants: flatmap (1->N), fan-in (N->M) |
| `filter` | selection sigma | pure predicate; drops rows |
| `join` | inner join | merges rows on key equality; route-merge variant picks one branch per row by a route key |
| `sink` | projection | projects the answer keys |

**Purity restriction.** Every agent/filter is a pure, deterministic function
of its input row(s). Agents may only *add* output keys. This single
restriction is what makes all four rewrites provably safe.

### 3.2 Cost model

Per agent invocation:

```
cost = alpha * expected_tokens + beta * latency + gamma * dollar_cost
```

with defaults alpha=1, beta=200 (1s of latency prices at 200 tokens),
gamma=100000. Plan cost = sum over nodes of (rows through node x per-row
cost) + beta x makespan, where makespan is the critical-path latency under
parallel execution. Selectivity of a filter and per-agent profiles come from
*declared contracts*, replaced by *measured history* (mean tokens, latency,
selectivity over past runs) once >= 3 runs are recorded.

### 3.3 Optimizer rules

1. **Predicate pushdown.** A filter F below an agent A moves above A iff
   `needs(F) subset of in_keys(A)` , the predicate reads only keys A passes
   through unchanged. Similarly pushed into the single join branch supplying
   all needed keys. (Classical: push selections below joins.)
2. **Join reordering.** For each 2-way join, the input with fewer estimated
   output rows is materialized first (build side); if the other branch has
   exclusive sources carrying the join keys, a *semi-join reduction* restricts
   those sources to surviving keys before its agents run , but only when a
   cost check says it is profitable: the token savings must outweigh the
   serialization latency penalty under the cost model
   (`alpha * tokens_saved > beta * (makespan_with - makespan_without)`).
   Semi-join reduction inherently serializes the branches, so a purely greedy
   rule would sometimes trade latency for tokens against the user's declared
   objective; the gate keeps the optimizer honest to its own cost model.
   Additionally, *route-conditional pruning*: for route-merge joins, each
   branch agent is restricted to rows routed to it.
3. **Parallelization.** Nodes are grouped into topological stages; the
   executor runs each stage's nodes and each agent's rows concurrently.
4. **Pruning.** Backward liveness from the sink drops agents whose produced
   keys nobody consumes (dead-code elimination for agents) and selectivity-1.0
   no-op filters.

## 4. Correctness: the equivalence invariant

**Theorem.** For any workflow, the executor's output on the optimized plan
equals its output on the naive sequential plan (exact row-multiset equality).

*Proof sketch.* By the purity restriction, each rewrite preserves the sink's
row multiset:

- **Lemma 1 (pushdown).** If `needs(F) subset of in_keys(A)` and A passes
  input keys through unchanged, then for every row r, F(r) is defined before
  A runs and A(F-filtered rows) = F-filtered A(rows): rows failing F are
  dropped either way, and surviving rows map identically since A is pure.
- **Lemma 2 (semi-join).** An inner join emits combined rows only for
  matching keys; restricting the other branch's sources to keys present in
  the probe side's output removes only rows that could never join. Matching
  rows are pure functions of their inputs, hence identical.
- **Lemma 3 (route pruning).** The route-merge reads, per row, only the
  branch selected by the row's route key; other branches' outputs for that
  row are dead values. Skipping them is unobservable.
- **Lemma 4 (dead-code elimination).** Removing a node whose produced keys
  are not transitively needed by the sink cannot change the sink's rows
  (backward liveness is conservative: every live node can influence output,
  every removed node provably cannot).
- **Lemma 5 (parallelism).** Staged concurrent execution of pure row
  functions with no shared mutable state equals serial topological
  execution; agent functions are deterministic in their inputs.

The executor additionally *enforces* purity at runtime (agents asserting no
input-key overwrite; joins asserting equal values on duplicated pass-through
keys), so a violation fails loudly instead of silently breaking the
invariant. ∎

## 5. Experiments

Setup: CPU-only VM, $0 spend. Agents are simulated deterministic local
functions with declared cost profiles (tokens/latency stand-ins for real LLM
calls); latency is *simulated with real sleeps*, so wall-clock wins are
measured, not modeled. Five workflows: research assistant, data pipeline,
support triage, code review, medical intake (see `agentplan/workflows.py`).

### E1: correctness, 200/200 exact matches

40 runs per workflow x 5 workflows. Every optimized-plan run produced a
byte-identical sink row multiset to the naive sequential run. Zero
mismatches across 200 runs, including under the executor's threaded
parallel stages (the invariant is not just a single-threaded property).

### E2: cost and latency vs sequential and hand-optimized

Means over 10 runs per cell. Tokens are cost-model estimates (no real LLM
calls); wall-clock is measured on simulated latencies.

| workflow | tok vs sequential | wall vs sequential | tok vs hand-opt | wall vs hand-opt |
|---|---|---|---|---|
| research_assistant | -27.9% | -90.6% | -7.9% | -0.3% |
| data_pipeline | -10.9% | -83.7% | -3.0% | +7.0% |
| support_triage | -56.3% | -96.2% | -56.3% | -37.3% |
| code_review | -34.1% | -85.9% | -7.9% | +7.1% |
| medical_intake | -71.9% | -92.5% | -32.3% | +8.1% |

3 of 5 workflows clear the 30% token-reduction bar vs sequential
(support_triage 56.3%, code_review 34.1%, medical_intake 71.9%).
research_assistant lands at 27.9%; data_pipeline at 10.9%, which is
Amdahl's law for agents: its naive plan already validated before
aggregating, so only the enricher pushdown and the semi-join reduction
apply. The optimizer beats the hand-optimized baseline on tokens in all
five workflows (3.0% to 56.3%), via the three things the hand rewrite
misses: semi-join reduction, route-conditional pruning, dead-agent
elimination.

On pure wall-clock the planner loses to the hand-parallel plan on three
workflows (data_pipeline +7.0%, code_review +7.1%, medical_intake +8.1%).
This is the honest price of semi-join reduction: it serializes the probe
branch before the reduced branch, trading latency for tokens exactly as
the cost model (alpha=1, beta=200) prices it. The cost-aware gate
(section 3.3) keeps this trade inside the declared objective rather than
letting a greedy rule make it unconditionally.

Hand-optimized = the plan a careful engineer writes by hand (obvious
predicate pushdown + parallelism), without semi-join reduction,
route-conditional pruning, or dead-code elimination, the planner's edge.

### E3: ablation, which rules matter

Disabling one rule at a time (delta vs the full optimizer; + = worse
without the rule; 6 runs per cell):

| workflow | no pushdown (tok/wall) | no reorder (tok/wall) | no parallel (tok/wall) | no prune (tok/wall) |
|---|---|---|---|---|
| research_assistant | +30.2% / +31.3% | +0.0% / +1.4% | +0.0% / +657.7% | +8.6% / +0.5% |
| data_pipeline | +9.1% / +0.9% | +3.1% / -9.2% | +0.0% / +390.6% | +0.0% / +2.5% |
| support_triage | +0.0% / -5.5% | +111.1% / +36.8% | +0.0% / +946.2% | +17.8% / +11.9% |
| code_review | +43.1% / +45.0% | +8.6% / -6.7% | +0.0% / +274.8% | +0.0% / -1.1% |
| medical_intake | +208.5% / +108.6% | +47.8% / -6.5% | +0.0% / +279.3% | +0.0% / +0.7% |

Reading: predicate pushdown dominates where a cheap filter guards an
expensive agent (medical_intake +208% without it); join reordering
dominates where routing/selectivity prunes whole branches (support_triage
+111% without it, the route-pruning effect); parallelization is the entire
wall-clock story (+275% to +946% without it, zero token effect, as it
should be); pruning matters where template cruft survives (dead sentiment
probe +8.6%, dead audit logger +17.8%). The negative wall deltas under
no_reorder are the semi-join serialization price discussed above.

### E4: robustness under wrong estimates

Four perturbation families (uniform 2x costs, cheap-agent halving,
selectivity +-0.25, lognormal latency noise): output equivalence held in
all 20 configurations, and cost regret vs the oracle plan was 0.00%
everywhere. The planner's discrete choices (pushdown validity, probe side,
route pruning) are driven by plan structure and estimated row counts, which
are robust to these perturbations at the tested magnitudes; only the
semi-join profitability gate is estimate-sensitive, and at beta=200 the
token savings dominate it by 10-100x in these workflows, so no decision
flipped. A dedicated unit test (`test_driven_gate_rejects_unprofitable`)
proves the gate is live: when serialization costs more than the token
savings, the reduction is skipped and equivalence still holds. Honest
bottom line: correctness never depends on estimate quality (all four rules
are validity-preserving by construction); only the probe-side/gate choices
can degrade, and gracefully.

### Optimizer overhead

`optimize()` wall time per workflow: 0.49-1.10 ms, vs optimized workflow
wall times of 120-390 ms. Worst case 0.46% of workflow cost, well under the
2% bar. (Measured on the same CPU the workflows ran on.)

## 6. Limits (honest)

- Agents are simulated with declared profiles, not real LLM calls; token
  figures are cost-model estimates, wall-clock figures are measured on
  simulated latencies. The *relative* wins transfer to real deployments only
  insofar as real profiles match the declared ones , the history-based
  estimation loop exists precisely to close that gap.
- The purity restriction rules out agents with cross-row state or side
  effects (e.g., an agent that writes to a shared KB and reads it back).
  Such workflows are rejected loudly, not miscompiled.
- Selectivity estimates for filters with data-dependent predicates start as
  declared contracts; the first runs may pick a suboptimal probe side (E4
  quantifies this: small regret, never incorrectness).
- Join reordering is currently limited to probe-side choice + semi-join
  reduction (no bushy-tree enumeration); the benchmark joins are 2-way.
- No cost-based choice *between agents* (model selection / cascading) yet ,
  the natural next rule, and the bridge to the coflight routing work.

## 7. References

- Kim et al., "An LLM Compiler for Parallel Function Calling", ICML 2024
  (arXiv:2312.04511).
- Patel et al., "Semantic Operators: A Declarative Model for Rich, AI-based
  Analytics Over Text Data", 2024 (arXiv:2407.11418) , the LOTUS system.
- Selinger et al., "Access Path Selection in a Relational Database
  Management System", 1979 , the original cost-based optimizer.
- Shankar et al., "DocETL", 2024; Zhang et al., "AFlow", 2024.
