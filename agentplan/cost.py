"""Cost model: per-agent cost = alpha*expected_tokens + beta*latency + gamma*dollar.

Selectivity and cost estimates come from declared contracts first, then get
replaced by measured history (stats.StatsStore) once >= MIN_SAMPLES runs
have been observed. Row counts propagate through the plan so the optimizer
can price filters (fewer rows downstream) and joins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .ir import Plan

ALPHA = 1.0        # cost per token
BETA = 200.0       # cost per second of latency (1s == 200 tokens)
GAMMA = 100000.0   # cost per dollar (1$ == 100k tokens)
MIN_SAMPLES = 3


@dataclass
class PlanCost:
    tokens: float
    latency: float      # makespan in seconds under parallel execution
    dollars: float
    total: float        # alpha*tokens + beta*latency + gamma*dollars
    rows_out: float = 0.0


def resolve(node_id: str, plan, stats) -> Tuple[float, float, float, float, Optional[float]]:
    """Return (tok_in, tok_out, latency, dollar, selectivity) for a node.

    Measured stats override declared contracts once enough samples exist.
    """
    n = plan.nodes[node_id]
    tok_in, tok_out, lat, dol = n.tok_in, n.tok_out, n.latency, n.dollar
    sel = n.selectivity
    if stats is not None:
        m = stats.get(plan.name, node_id)
        if m and m.get("n", 0) >= MIN_SAMPLES:
            if m.get("mean_tokens") is not None:
                # measured total tokens per row -> split proportionally
                tot = m["mean_tokens"]
                denom = (tok_in + tok_out) or 1.0
                tok_in, tok_out = tot * tok_in / denom, tot * tok_out / denom
            if m.get("mean_latency") is not None:
                lat = m["mean_latency"]
            if n.op == "filter" and m.get("selectivity") is not None:
                sel = m["selectivity"]
    return tok_in, tok_out, lat, dol, sel


def estimate_rows(plan: Plan, stats=None) -> Dict[str, float]:
    """Estimated output row count per node (topological propagation)."""
    rows: Dict[str, float] = {}
    for nid in plan.topo():
        n = plan.nodes[nid]
        if n.op == "source":
            rows[nid] = float(n.rows or 1.0)
        elif n.op == "agent":
            in_rows = rows[n.inputs[0]] if n.inputs else 1.0
            if n.fan_in:
                rows[nid] = 1.0
            elif n.route_values is not None:
                # routed agent: fraction of rows routed to it
                frac = 1.0 / max(1, len(n.route_values))
                rows[nid] = in_rows * frac * n.expand_factor
            else:
                rows[nid] = in_rows * n.expand_factor
        elif n.op == "filter":
            in_rows = rows[n.inputs[0]] if n.inputs else 1.0
            _, _, _, _, sel = resolve(nid, plan, stats)
            rows[nid] = in_rows * (sel if sel is not None else 1.0)
        elif n.op == "join":
            vals = [rows[i] for i in n.inputs]
            rows[nid] = min(vals)  # 1:1 key merge estimate
        elif n.op == "sink":
            rows[nid] = rows[n.inputs[0]] if n.inputs else 1.0
        else:
            rows[nid] = 1.0
    return rows


def critical_path_latency(plan: Plan, stats=None, row_est: Optional[Dict[str, float]] = None) -> float:
    """Makespan: longest latency path assuming independent nodes run in parallel."""
    row_est = row_est or estimate_rows(plan, stats)
    depth_lat: Dict[str, float] = {}
    for nid in plan.topo():
        n = plan.nodes[nid]
        _, _, lat, _, _ = resolve(nid, plan, stats)
        if n.op == "agent" and not n.fan_in:
            node_lat = lat          # parallel scan: one row-time
        elif n.op == "agent" and n.fan_in:
            node_lat = lat * max(1.0, row_est.get(n.inputs[0], 1.0)) ** 0.5
        else:
            node_lat = lat
        depth_lat[nid] = node_lat + (max([depth_lat[i] for i in n.inputs]) if n.inputs else 0.0)
    return depth_lat.get(plan.sink_id or "", 0.0)


def plan_cost(plan: Plan, stats=None,
              alpha: float = ALPHA, beta: float = BETA, gamma: float = GAMMA) -> PlanCost:
    row_est = estimate_rows(plan, stats)
    tokens = dollars = 0.0
    for nid in plan.topo():
        n = plan.nodes[nid]
        tok_in, tok_out, _, dol, _ = resolve(nid, plan, stats)
        r = row_est[nid] if n.op in ("agent", "filter") else 1.0
        # agent/filter cost scales with rows THROUGH the node (input rows)
        if n.op in ("agent", "filter"):
            r = row_est[n.inputs[0]] if n.inputs else 1.0
            if n.op == "agent" and n.route_values is not None:
                r = r / max(1, len(n.route_values))
        tokens += r * (tok_in + tok_out)
        dollars += r * dol
    lat = critical_path_latency(plan, stats, row_est)
    total = alpha * tokens + beta * lat + gamma * dollars
    return PlanCost(tokens=tokens, latency=lat, dollars=dollars, total=total,
                    rows_out=row_est.get(plan.sink_id or "", 0.0))
