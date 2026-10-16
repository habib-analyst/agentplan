"""Executors: a naive sequential runner and the optimized staged runner.

Both execute the same logical plan; the optimizer's rewrites (pushdown,
reorder + semi-join, route pruning, dead-code elimination) are provably
output-preserving, so the optimized runner must return EXACTLY the same
sink rows as the sequential runner (the E1 correctness invariant).
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .ir import Plan, PlanNode, Row
from .optimize import PhysicalPlan


@dataclass
class NodeMetrics:
    rows_in: int = 0
    rows_out: int = 0
    tokens: float = 0.0
    wall: float = 0.0


@dataclass
class RunResult:
    rows: List[Row]
    metrics: Dict[str, NodeMetrics]
    wall_total: float
    tokens_total: float


def _keytuple(row: Row, keys: List[str]) -> Tuple:
    return tuple(row[k] for k in keys)


def canonical(rows: List[Row]) -> Tuple:
    """Sortable canonical form for exact output comparison."""
    return tuple(sorted(
        tuple(sorted((k, repr(v)) for k, v in r.items()))
        for r in rows
    ))


class Runner:
    def __init__(self, plan: Plan, physical: Optional[PhysicalPlan] = None,
                 mode: str = "sequential", max_workers: int = 8,
                 stats=None, record: bool = True) -> None:
        assert mode in ("sequential", "optimized")
        self.plan = plan
        self.mode = mode
        self.max_workers = 1 if mode == "sequential" else max_workers
        self.stats = stats
        self.record = record
        self.physical = physical
        self.outputs: Dict[str, List[Row]] = {}
        self.overrides: Dict[str, List[Row]] = {}
        self.pending_filters: Dict[str, Tuple[List[str], set]] = {}
        self.metrics: Dict[str, NodeMetrics] = {}
        self._driven_installed: set = set()

    # ---- node kernels -------------------------------------------------
    def _run_source(self, n: PlanNode) -> List[Row]:
        t0 = time.perf_counter()
        rows = n.fn()
        assert isinstance(rows, list)
        m = self.metrics[n.id]
        m.rows_in = len(rows)
        m.rows_out = len(rows)
        m.wall = time.perf_counter() - t0
        return rows

    def _apply_agent_row(self, n: PlanNode, row: Row) -> List[Row]:
        if n.latency:
            time.sleep(n.latency)
        if n.fan_in:
            raise AssertionError("fan_in handled separately")
        out = n.fn(row)
        if isinstance(out, list):  # flatmap expand
            new_rows = []
            for o in out:
                assert not (set(o) & set(row)), f"agent {n.name} overwrote input keys"
                new_rows.append({**row, **o})
            return new_rows
        assert isinstance(out, dict), f"agent {n.name} must return dict"
        assert not (set(out) & set(row)), f"agent {n.name} overwrote input keys: {set(out) & set(row)}"
        return [{**row, **out}]

    def _run_agent(self, n: PlanNode, rows: List[Row]) -> List[Row]:
        t0 = time.perf_counter()
        if n.route_key and n.route_values is not None:
            rows = [r for r in rows if r.get(n.route_key) in n.route_values]
        m = self.metrics[n.id]
        m.rows_in = len(rows)
        if n.fan_in:
            if n.latency:
                time.sleep(n.latency)
            out_rows = n.fn(rows)
            assert isinstance(out_rows, list)
            m.rows_out = len(out_rows)
            m.tokens = (n.tok_in + n.tok_out) * max(1, len(rows))
            m.wall = time.perf_counter() - t0
            return out_rows
        out: List[Row] = []
        if self.max_workers > 1 and rows:
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                for res in ex.map(lambda r: self._apply_agent_row(n, r), rows):
                    out.extend(res)
        else:
            for r in rows:
                out.extend(self._apply_agent_row(n, r))
        m.rows_out = len(out)
        m.tokens = (n.tok_in + n.tok_out) * len(rows)
        m.wall = time.perf_counter() - t0
        return out

    def _run_filter(self, n: PlanNode, rows: List[Row]) -> List[Row]:
        t0 = time.perf_counter()
        keep: List[Row] = []
        for r in rows:
            if n.latency:
                time.sleep(n.latency)
            if n.fn(r):
                keep.append(r)
        m = self.metrics[n.id]
        m.rows_in = len(rows)
        m.rows_out = len(keep)
        m.tokens = (n.tok_in + n.tok_out) * len(rows)
        m.wall = time.perf_counter() - t0
        return keep

    def _run_join(self, n: PlanNode) -> List[Row]:
        t0 = time.perf_counter()
        route_map = n.tags.get("route_map")
        if route_map:
            out = self._run_route_merge(n, route_map)
        else:
            assert len(n.inputs) == 2
            left = self.outputs[n.inputs[0]]
            right = self.outputs[n.inputs[1]]
            table: Dict[Tuple, List[Row]] = {}
            for r in right:
                table.setdefault(_keytuple(r, n.join_keys), []).append(r)
            out = []
            for l in left:
                for r in table.get(_keytuple(l, n.join_keys), []):
                    for k in (set(l) & set(r)) - set(n.join_keys):
                        assert l[k] == r[k], f"join {n.name}: conflicting values for {k}"
                    extra = {k: v for k, v in r.items() if k not in l}
                    out.append({**l, **extra})
        m = self.metrics[n.id]
        m.rows_in = sum(self.metrics[i].rows_out for i in n.inputs)
        m.rows_out = len(out)
        m.wall = time.perf_counter() - t0
        return out

    def _run_route_merge(self, n: PlanNode, route_map: Dict[str, str]) -> List[Row]:
        route_key = n.tags["route_key"]
        name_to_id = {self.plan.nodes[i].name: i for i in n.inputs}
        by_branch: Dict[str, Dict[Tuple, Row]] = {}
        for route_val, branch_name in route_map.items():
            bid = name_to_id[branch_name]
            by_branch[branch_name] = {_keytuple(r, n.join_keys): r for r in self.outputs[bid]}
        # the classifier/router is the input whose name is not a branch target
        router_id = next(i for i in n.inputs if self.plan.nodes[i].name not in route_map.values())
        out = []
        for r in self.outputs[router_id]:
            branch = by_branch[route_map[r[route_key]]]
            b = branch.get(_keytuple(r, n.join_keys))
            assert b is not None, f"route merge: no {route_map[r[route_key]]} row for {r}"
            for k in (set(r) & set(b)) - set(n.join_keys):
                assert r[k] == b[k], f"route merge: conflicting values for {k}"
            extra = {k: v for k, v in b.items() if k not in r}
            out.append({**r, **extra})
        return out

    def _run_sink(self, n: PlanNode, rows: List[Row]) -> List[Row]:
        return [{k: r[k] for k in n.keep} for r in rows]

    # ---- driven semi-join ----------------------------------------------
    def _maybe_install_driven(self) -> None:
        if not self.physical:
            return
        for jid, d in self.physical.driven.items():
            if jid in self._driven_installed or d["probe"] not in self.outputs:
                continue
            probe_rows = self.outputs[d["probe"]]
            keys = self.plan.nodes[jid].join_keys
            kset = {_keytuple(r, keys) for r in probe_rows}
            for s in d["sources"]:
                # source may not have executed yet: stash a pending key filter
                self.pending_filters[s] = (keys, kset)
            self._driven_installed.add(jid)

    def _exec_node(self, nid: str) -> None:
        n = self.plan.nodes[nid]
        self.metrics[nid] = NodeMetrics()
        if nid in self.overrides:
            assert n.op == "source"
            self.outputs[nid] = self.overrides[nid]
            m = self.metrics[nid]
            m.rows_in = m.rows_out = len(self.outputs[nid])
            return
        if n.op == "source":
            rows = self._run_source(n)
            if nid in self.pending_filters:
                keys, kset = self.pending_filters[nid]
                rows = [r for r in rows if _keytuple(r, keys) in kset]
                m = self.metrics[nid]
                m.rows_in = m.rows_out = len(rows)
            self.outputs[nid] = rows
        elif n.op == "agent":
            rows = self.outputs[n.inputs[0]] if n.inputs else []
            self.outputs[nid] = self._run_agent(n, rows)
        elif n.op == "filter":
            rows = self.outputs[n.inputs[0]]
            self.outputs[nid] = self._run_filter(n, rows)
        elif n.op == "join":
            self.outputs[nid] = self._run_join(n)
        elif n.op == "sink":
            rows = self.outputs[n.inputs[0]]
            self.outputs[nid] = self._run_sink(n, rows)
        else:
            raise ValueError(f"unknown op {n.op}")

    def run(self) -> RunResult:
        t0 = time.perf_counter()
        if self.mode == "sequential" or not self.physical:
            order = self.plan.topo()
            for nid in order:
                self._maybe_install_driven()
                self._exec_node(nid)
        else:
            for stage in self.physical.stages:
                self._maybe_install_driven()
                if self.max_workers > 1 and len(stage) > 1:
                    with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                        list(ex.map(self._exec_node, stage))
                else:
                    for nid in stage:
                        self._exec_node(nid)
        wall = time.perf_counter() - t0
        tokens = sum(m.tokens for m in self.metrics.values())
        if self.record and self.stats is not None:
            self.stats.record_run(self.plan.name, self.plan, self.metrics)
        return RunResult(rows=self.outputs[self.plan.sink_id or ""],
                         metrics=self.metrics, wall_total=wall, tokens_total=tokens)


def execute(plan: Plan, physical: Optional[PhysicalPlan] = None,
            mode: str = "sequential", max_workers: int = 8,
            stats=None, record: bool = True) -> RunResult:
    plan.validate()
    return Runner(plan, physical, mode, max_workers, stats, record).run()
