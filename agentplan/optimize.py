"""Optimizer rules. Each rule is a provably output-preserving plan rewrite
(see PAPER.md for the proof sketches):

  pushdown  - move a Filter before an Agent/Join when the predicate only
              reads keys the agent passes through unchanged (needs(F) subset
              of in_keys(A)). Lemma 1: filter-then-map == map-then-filter.
  reorder   - choose join probe order (cheapest / most selective side first)
              and enable semi-join reduction on the other side's sources.
              Lemma 2: inner join is unaffected by upstream key restriction.
              Also: route-conditional pruning (run a specialist only on rows
              routed to it). Lemma 3: unread branch outputs are dead values.
  parallel  - group nodes into stages; the executor runs stages concurrently.
              Lemma 5: pure row functions commute across threads.
  prune     - dead-agent elimination via backward liveness analysis, plus
              dropping selectivity-1.0 no-op filters. Lemma 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .ir import Plan, PlanNode
from .cost import plan_cost, estimate_rows, resolve


@dataclass
class RuleApplication:
    rule: str
    detail: str
    cost_before: float = 0.0
    cost_after: float = 0.0


@dataclass
class PhysicalPlan:
    stages: List[List[str]]
    probe_first: Dict[str, str] = field(default_factory=dict)   # join_id -> input id
    driven: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # join_id -> {probe, other, sources}
    pruned: List[str] = field(default_factory=list)
    rules_applied: List[RuleApplication] = field(default_factory=list)


def _needs(filter_node: PlanNode) -> Set[str]:
    return set(filter_node.tags.get("needs", []))


def pushdown_filters(plan: Plan, stats=None) -> List[RuleApplication]:
    """Move filters below agents/joins when valid. Fixpoint iteration."""
    applied: List[RuleApplication] = []
    changed = True
    while changed:
        changed = False
        for nid in plan.topo():
            n = plan.nodes[nid]
            if n.op != "filter" or len(n.inputs) != 1:
                continue
            parent = plan.nodes[n.inputs[0]]
            needs = _needs(n)
            if parent.op == "agent" and not parent.fan_in and needs <= set(parent.in_keys):
                # F(A(x)) == A(F(x)) restricted: predicate reads keys A passes through.
                _swap_filter_before_agent(plan, n, parent)
                applied.append(RuleApplication("pushdown",
                    f"filter '{n.name}' pushed before agent '{parent.name}'"))
                changed = True
                break
            if parent.op == "join":
                # push into the single branch that supplies all needed keys
                for inp_id in parent.inputs:
                    if needs <= plan.subgraph_keys(inp_id):
                        _insert_filter_into_branch(plan, n, parent, inp_id)
                        applied.append(RuleApplication("pushdown",
                            f"filter '{n.name}' pushed into join branch '{plan.nodes[inp_id].name}'"))
                        changed = True
                        break
                if changed:
                    break
    return applied


def _swap_filter_before_agent(plan: Plan, f: PlanNode, a: PlanNode) -> None:
    grand = list(a.inputs)
    f.inputs = grand
    f.in_keys = list(a.in_keys)
    f.out_keys = list(a.in_keys)
    a.inputs = [f.id]
    # rewire a's children that pointed at a's old position: children of f-old
    # position now must read from a. Children of f (except a) -> repoint to a.
    for cid in plan.children_of(f.id):
        if cid != a.id:
            c = plan.nodes[cid]
            c.inputs = [a.id if i == f.id else i for i in c.inputs]
    # a keeps its in_keys/out_keys (it still reads the same logical keys)


def _insert_filter_into_branch(plan: Plan, f: PlanNode, j: PlanNode, branch_id: str) -> None:
    f.inputs = [branch_id]
    avail = plan.subgraph_keys(branch_id)
    f.in_keys = sorted(avail)
    f.out_keys = sorted(avail)
    j.inputs = [f.id if i == branch_id else i for i in j.inputs]
    # f's downstream (other than the join) must now read from the join output
    for cid in plan.children_of(f.id):
        if cid != j.id:
            c = plan.nodes[cid]
            c.inputs = [j.id if i == f.id else i for i in c.inputs]


def _lineage_sources(plan: Plan, nid: str, _memo=None) -> Set[str]:
    _memo = _memo if _memo is not None else {}
    if nid in _memo:
        return _memo[nid]
    n = plan.nodes[nid]
    if n.op == "source":
        _memo[nid] = {nid}
    else:
        s: Set[str] = set()
        for i in n.inputs:
            s |= _lineage_sources(plan, i, _memo)
        _memo[nid] = s
    return _memo[nid]


def _branch_cost(plan: Plan, nid: str, stats, row_est) -> float:
    """Estimated cost of the subgraph feeding nid (exclusive of shared upstream)."""
    # approximate: sum over exclusive nodes of rows*per-row-cost
    from .cost import ALPHA, BETA, GAMMA
    total = 0.0
    seen: Set[str] = set()
    stack = [nid]
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        node = plan.nodes[x]
        tok_in, tok_out, lat, dol, _ = resolve(x, plan, stats)
        r = row_est.get(x, 1.0)
        if node.op in ("agent", "filter"):
            r = row_est.get(node.inputs[0], 1.0) if node.inputs else 1.0
        total += r * (ALPHA * (tok_in + tok_out) + GAMMA * dol) + lat
        stack.extend(node.inputs)
    return total


def _driven_profitable(plan: Plan, stats, jid: str, probe: str, other: str,
                     sources: List[str]) -> bool:
    """Cost-aware gate: enable semi-join reduction only if the token savings
    outweigh the serialization latency penalty under the cost model."""
    from .cost import ALPHA, BETA, estimate_rows, resolve
    row_est = estimate_rows(plan, stats)
    up_probe = _upstream(plan, probe)
    excl = _upstream(plan, other) - up_probe
    if not excl:
        return False
    scale = min(1.0, row_est.get(probe, 0.0) / max(row_est.get(other, 1e-9), 1e-9))
    tok_save = 0.0
    for nid in excl:
        n = plan.nodes[nid]
        if n.op != "agent":
            continue
        tok_in, tok_out, _, _, _ = resolve(nid, plan, stats)
        r = row_est.get(n.inputs[0], 1.0) if n.inputs else 1.0
        tok_save += r * (1.0 - scale) * (tok_in + tok_out)

    def stage_latency(stages: List[List[str]]) -> float:
        tot = 0.0
        for st in stages:
            mx = 0.0
            for nid in st:
                _, _, lat, _, _ = resolve(nid, plan, stats)
                mx = max(mx, lat)
            tot += mx
        return tot

    lat_without = stage_latency(plan.stages())
    stages_with = _force_probe_stages(
        plan, {jid: {"probe": probe, "other": other, "sources": sources}})
    lat_with = stage_latency(stages_with)
    return ALPHA * tok_save > BETA * (lat_with - lat_without)


def reorder_joins(plan: Plan, stats=None) -> Tuple[List[RuleApplication], Dict[str, str], Dict[str, Dict]]:
    """Pick probe order per join; enable semi-join reduction where valid AND profitable."""
    applied: List[RuleApplication] = []
    probe_first: Dict[str, str] = {}
    driven: Dict[str, Dict[str, Any]] = {}
    row_est = estimate_rows(plan, stats)
    for nid in plan.topo():
        n = plan.nodes[nid]
        if n.op != "join":
            continue
        # route-conditional pruning (Lemma 3)
        route_key = n.tags.get("route_key")
        route_map: Dict[str, str] = n.tags.get("route_map", {})
        if route_key and route_map:
            inv: Dict[str, List[str]] = {}
            for route_val, branch_name in route_map.items():
                inv.setdefault(branch_name, []).append(route_val)
            for inp_id in n.inputs:
                b = plan.nodes[inp_id]
                vals = inv.get(b.name)
                if vals and b.op == "agent":
                    b.route_key = route_key
                    b.route_values = set(vals)
                    applied.append(RuleApplication("reorder",
                        f"route pruning: agent '{b.name}' restricted to {route_key} in {sorted(vals)}"))
        if len(n.inputs) == 2:
            # probe = the side with fewer estimated output rows (classic
            # build-side choice); ties broken by branch cost.
            costs = {i: _branch_cost(plan, i, stats, row_est) for i in n.inputs}
            first = min(n.inputs, key=lambda i: (row_est.get(i, 0.0), costs[i]))
            probe_first[nid] = first
            other = n.inputs[1] if n.inputs[0] == first else n.inputs[0]
            # semi-join reduction valid iff the other branch has exclusive
            # sources carrying the join keys (Lemma 2).
            excl = _lineage_sources(plan, other) - _lineage_sources(plan, first)
            keyed = [s for s in excl
                     if set(n.join_keys) <= set(plan.nodes[s].out_keys)]
            if keyed and _driven_profitable(plan, stats, nid, first, other, sorted(keyed)):
                driven[nid] = {"probe": first, "other": other, "sources": sorted(keyed)}
                applied.append(RuleApplication(
                    "reorder",
                    f"join '{n.name}': probe '{plan.nodes[first].name}' first; "
                    f"semi-join reduces sources {[plan.nodes[s].name for s in keyed]}"))
            elif keyed:
                applied.append(RuleApplication(
                    "reorder",
                    f"join '{n.name}': probe '{plan.nodes[first].name}' first; "
                    f"semi-join valid but not cost-profitable, skipped"))
            else:
                applied.append(RuleApplication(
                    "reorder",
                    f"join '{n.name}': probe '{plan.nodes[first].name}' first (no exclusive keyed sources)"))
    return applied, probe_first, driven


def _drop_noop_filters(plan: Plan) -> List[RuleApplication]:
    """Forward pass: remove selectivity-1.0 filters, rewiring children to the input."""
    applied: List[RuleApplication] = []
    for nid in plan.topo():
        n = plan.nodes.get(nid)
        if n is None or n.op != "filter" or n.selectivity != 1.0 or len(n.inputs) != 1:
            continue
        parent = n.inputs[0]
        for cid in plan.children_of(nid):
            c = plan.nodes[cid]
            c.inputs = [parent if i == nid else i for i in c.inputs]
        applied.append(RuleApplication("prune", f"no-op filter '{n.name}' (selectivity 1.0) dropped"))
        del plan.nodes[nid]
    return applied


def prune_dead(plan: Plan) -> Tuple[List[RuleApplication], List[str]]:
    """Backward liveness: drop agents/filters whose outputs nobody needs."""
    applied: List[RuleApplication] = []
    applied.extend(_drop_noop_filters(plan))
    assert plan.sink_id
    needed: Set[str] = set(plan.nodes[plan.sink_id].keep)
    live: Set[str] = {plan.sink_id}
    for nid in reversed(plan.topo()):
        n = plan.nodes[nid]
        if nid not in live:
            continue
        if n.op == "sink":
            needed |= set(n.keep)
        elif n.op == "filter":
            # live only if some needed key flows through it
            if needed & (set(n.in_keys) | set(n.out_keys)):
                needed |= set(n.in_keys) | _needs(n)
            else:
                live.discard(nid)
                applied.append(RuleApplication("prune", f"dead filter '{n.name}' eliminated"))
                continue
        elif n.op == "agent":
            new_keys = set(n.out_keys) - set(n.in_keys)
            if needed & new_keys:
                needed |= set(n.in_keys)
            else:
                live.discard(nid)
                applied.append(RuleApplication("prune", f"dead agent '{n.name}' eliminated"))
                continue
        elif n.op == "join":
            if needed & (set(n.in_keys) | set(n.out_keys) | set(n.join_keys)):
                needed |= set(n.join_keys)
                for i in n.inputs:
                    needed |= set(plan.nodes[i].out_keys) | set(plan.nodes[i].in_keys)
            else:
                live.discard(nid)
                continue
        elif n.op == "source":
            pass
        for i in n.inputs:
            if nid in live:
                live.add(i)
    pruned = [nid for nid in plan.nodes if nid not in live]
    logged = set()
    for r in applied:
        # extract node names already logged (best-effort)
        logged.add(r.detail)
    for nid in pruned:
        n = plan.nodes[nid]
        detail = f"dead {n.op} '{n.name}' eliminated"
        if detail not in logged:
            applied.append(RuleApplication("prune", detail))
    for nid in pruned:
        del plan.nodes[nid]
    plan.validate()
    return applied, pruned


def _upstream(plan: Plan, nid: str, _memo=None) -> Set[str]:
    _memo = _memo if _memo is not None else {}
    if nid in _memo:
        return _memo[nid]
    n = plan.nodes[nid]
    s = {nid}
    for i in n.inputs:
        s |= _upstream(plan, i, _memo)
    _memo[nid] = s
    return s


def _force_probe_stages(plan: Plan, driven: Dict[str, Dict]) -> List[List[str]]:
    """Stage assignment guaranteeing: for each driven join, the probe branch
    fully materializes before any node exclusive to the other branch runs."""
    import collections
    base: Dict[str, int] = {}
    for nid in plan.topo():
        n = plan.nodes[nid]
        base[nid] = 0 if not n.inputs else 1 + max(base[i] for i in n.inputs)
    # propagate constraints
    changed = True
    while changed:
        changed = False
        for jid, d in driven.items():
            probe, other = d["probe"], d["other"]
            up_probe = _upstream(plan, probe)
            excl_other = _upstream(plan, other) - up_probe
            if not excl_other:
                continue
            lo = max(base[x] for x in up_probe)
            for y in excl_other:
                # y must come after the whole probe branch
                need = lo + 1 + _depth_within(plan, y, excl_other)
                if base[y] < need:
                    base[y] = need
                    changed = True
        # re-propagate plain topo after bumps
        for nid in plan.topo():
            n = plan.nodes[nid]
            need = 0 if not n.inputs else 1 + max(base[i] for i in n.inputs)
            if base[nid] < need:
                base[nid] = need
                changed = True
    by: Dict[int, List[str]] = collections.defaultdict(list)
    for nid, lv in base.items():
        by[lv].append(nid)
    return [sorted(by[lv]) for lv in sorted(by)]


def _depth_within(plan: Plan, nid: str, allowed: Set[str]) -> int:
    n = plan.nodes[nid]
    ins = [i for i in n.inputs if i in allowed]
    return 0 if not ins else 1 + max(_depth_within(plan, i, allowed) for i in ins)


def optimize(plan: Plan, stats=None,
             rules: Optional[Set[str]] = None) -> Tuple[Plan, PhysicalPlan, List[RuleApplication]]:
    """Run the rewrite pipeline. Returns (optimized plan, physical plan, log)."""
    rules = rules if rules is not None else {"pushdown", "reorder", "parallel", "prune"}
    plan = plan.copy()
    plan.validate()
    log: List[RuleApplication] = []
    pruned: List[str] = []
    probe_first: Dict[str, str] = {}
    driven: Dict[str, Dict] = {}
    if "prune" in rules:
        a, pruned = prune_dead(plan)
        log.extend(a)
    if "pushdown" in rules:
        log.extend(pushdown_filters(plan, stats))
    if "reorder" in rules:
        a, probe_first, driven = reorder_joins(plan, stats)
        log.extend(a)
    if "parallel" in rules:
        stages = _force_probe_stages(plan, driven) if driven else plan.stages()
        log.append(RuleApplication("parallel",
            f"{len(stages)} execution stages; {sum(len(s) for s in stages)} nodes"))
    else:
        # serial execution, but keep probe-first ordering so driven semi-joins
        # still fire (isolates parallelism's effect in ablations)
        if driven and "reorder" in rules:
            forced = _force_probe_stages(plan, driven)
            stages = [[nid] for stage in forced for nid in stage]
        else:
            stages = [[nid] for nid in plan.topo()]
    plan.validate()
    phys = PhysicalPlan(stages=stages, probe_first=probe_first, driven=driven,
                        pruned=pruned,
                        rules_applied=[r for r in log if r.rule in rules])
    return plan, phys, log
