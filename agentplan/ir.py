"""agentplan IR: workflows compile to a logical query plan.

Mapping (the core idea):
  agent call   -> table function  (row in -> row out, with cost + selectivity)
  handoff      -> join            (merge agent outputs on keys)
  guardrail    -> filter          (cheap predicate, drops rows)
  fan-out      -> parallel scan   (one agent applied per row, concurrently)
  final answer -> sink            (projection onto answer keys)

Every agent is a PURE, DETERMINISTIC row function: it may only ADD output
keys and must never mutate its input keys. This single restriction is what
makes predicate pushdown, semi-join reduction and route pruning provably
output-equivalent (see optimize.py lemmas, PAPER.md proofs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

Row = Dict[str, Any]
RowFn = Callable[[Row], Dict[str, Any]]       # agent: row -> {new keys}
PredFn = Callable[[Row], bool]                # filter predicate
ReduceFn = Callable[[List[Row]], List[Row]]   # fan-in agent: rows -> rows


@dataclass
class PlanNode:
    id: str
    op: str                      # 'source' | 'agent' | 'filter' | 'join' | 'sink'
    name: str
    inputs: List[str] = field(default_factory=list)
    in_keys: List[str] = field(default_factory=list)
    out_keys: List[str] = field(default_factory=list)
    # cost profile (per ROW for agent/filter, per NODE for source/join/sink)
    tok_in: float = 0.0
    tok_out: float = 0.0
    latency: float = 0.0         # seconds (simulated); per row for agents
    dollar: float = 0.0
    selectivity: Optional[float] = None   # filters: declared fraction passing
    rows: Optional[float] = None          # sources: declared row count
    fn: Optional[Callable] = None
    join_keys: List[str] = field(default_factory=list)
    keep: List[str] = field(default_factory=list)   # sink: projected keys
    fan_in: bool = False                   # agent consumes ALL rows at once
    expand_factor: float = 1.0             # flatmap agents: output rows per input row
    route_key: Optional[str] = None        # agent only runs rows where
    route_values: Optional[Set[str]] = None  # row[route_key] in route_values
    tags: Dict[str, Any] = field(default_factory=dict)

    def node_cost_per_row(self, alpha: float, beta: float, gamma: float) -> float:
        return alpha * (self.tok_in + self.tok_out) + beta * self.latency + gamma * self.dollar


class Plan:
    """A DAG of PlanNodes with one sink."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.nodes: Dict[str, PlanNode] = {}
        self.sink_id: Optional[str] = None
        self.meta: Dict[str, Any] = {}

    def add(self, node: PlanNode) -> PlanNode:
        if node.id in self.nodes:
            raise ValueError(f"duplicate node id {node.id}")
        self.nodes[node.id] = node
        if node.op == "sink":
            self.sink_id = node.id
        return node

    def children_of(self, nid: str) -> List[str]:
        return [n.id for n in self.nodes.values() if nid in n.inputs]

    def topo(self) -> List[str]:
        """Topological order (Kahn). Raises on cycles."""
        indeg = {nid: 0 for nid in self.nodes}
        for n in self.nodes.values():
            for i in n.inputs:
                indeg[n.id] += 1
        ready = sorted([nid for nid, d in indeg.items() if d == 0])
        order: List[str] = []
        while ready:
            nid = ready.pop(0)
            order.append(nid)
            for c in self.children_of(nid):
                indeg[c] -= 1
                if indeg[c] == 0:
                    ready.append(c)
            ready.sort()
        if len(order) != len(self.nodes):
            raise ValueError("plan has a cycle")
        return order

    def stages(self) -> List[List[str]]:
        """Topological levels; nodes in one stage are mutually independent."""
        depth: Dict[str, int] = {}
        for nid in self.topo():
            n = self.nodes[nid]
            depth[nid] = 0 if not n.inputs else 1 + max(depth[i] for i in n.inputs)
        by_depth: Dict[int, List[str]] = {}
        for nid, d in depth.items():
            by_depth.setdefault(d, []).append(nid)
        return [sorted(by_depth[d]) for d in sorted(by_depth)]

    def sources(self) -> List[str]:
        return [nid for nid, n in self.nodes.items() if n.op == "source"]

    def copy(self) -> "Plan":
        import copy as _copy
        p = Plan(self.name)
        p.nodes = {nid: _copy.deepcopy(n) for nid, n in self.nodes.items()}
        # fn objects are not deepcopy-safe in general; restore by identity
        for nid, n in self.nodes.items():
            p.nodes[nid].fn = n.fn
        p.sink_id = self.sink_id
        p.meta = dict(self.meta)
        return p

    def live_keys(self, nid: str) -> Set[str]:
        """Keys available at the output of node nid."""
        keys: Set[str] = set()
        for up in self.topo():
            n = self.nodes[up]
            keys |= set(n.in_keys) | set(n.out_keys)
            if up == nid:
                break
        return keys

    def subgraph_keys(self, nid: str) -> Set[str]:
        """Keys produced by the subgraph feeding nid (nid + all ancestors)."""
        keys: Set[str] = set()
        seen: Set[str] = set()
        stack = [nid]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            n = self.nodes[x]
            keys |= set(n.out_keys)
            stack.extend(n.inputs)
        return keys

    def validate(self) -> None:
        assert self.sink_id, "plan needs a sink"
        self.topo()  # raises on cycle
        for nid, n in self.nodes.items():
            for i in n.inputs:
                assert i in self.nodes, f"{nid} references missing input {i}"
            if n.op == "filter":
                assert n.fn is not None, f"filter {nid} needs predicate"
            if n.op in ("agent", "source"):
                assert n.fn is not None, f"{n.op} {nid} needs fn"
            if n.op == "join":
                assert len(n.inputs) >= 2 and n.join_keys, f"join {nid} malformed"
                if len(n.inputs) > 2:
                    assert n.tags.get("route_map"), f"join {nid}: >2 inputs need route_map"
