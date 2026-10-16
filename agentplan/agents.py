"""Simulated local agents with deterministic, measured cost profiles.

No LLM APIs, no network: each agent is a pure deterministic function of its
input row (seeded by content hash), with a declared cost profile
(input/output tokens, simulated latency seconds, dollar cost). The profiles
stand in for real LLM calls; the optimizer prices them, the executor
simulates their latency with real sleeps so wall-clock wins are measured,
not modeled.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List

from .ir import PlanNode, Row


def det(*parts: Any) -> random.Random:
    h = hashlib.sha256("|".join(repr(p) for p in parts).encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def agent(node_id: str, name: str, in_keys: List[str], new_keys: List[str],
          fn, tok_in: float, tok_out: float, latency: float,
          fan_in: bool = False, expand_factor: float = 1.0,
          inputs: List[str] = None, dollar: float = 0.0) -> PlanNode:
    return PlanNode(id=node_id, op="agent", name=name, inputs=inputs or [],
                    in_keys=list(in_keys), out_keys=list(in_keys) + list(new_keys),
                    tok_in=tok_in, tok_out=tok_out, latency=latency, dollar=dollar,
                    fn=fn, fan_in=fan_in, expand_factor=expand_factor)


def filt(node_id: str, name: str, in_keys: List[str], needs: List[str],
         pred, selectivity: float, inputs: List[str],
         tok: float = 10.0, latency: float = 0.001) -> PlanNode:
    return PlanNode(id=node_id, op="filter", name=name, inputs=inputs,
                    in_keys=list(in_keys), out_keys=list(in_keys),
                    tok_in=tok, tok_out=0.0, latency=latency,
                    selectivity=selectivity, fn=pred,
                    tags={"needs": list(needs)})


def join(node_id: str, name: str, inputs: List[str], join_keys: List[str],
         out_keys: List[str], **tags) -> PlanNode:
    in_keys = sorted({k for i in inputs for k in []} | set(join_keys))
    return PlanNode(id=node_id, op="join", name=name, inputs=inputs,
                    in_keys=in_keys, out_keys=list(out_keys),
                    join_keys=list(join_keys), tags=dict(tags))


def source(node_id: str, name: str, rows: List[Row]) -> PlanNode:
    keys = sorted({k for r in rows for k in r})
    return PlanNode(id=node_id, op="source", name=name, inputs=[],
                    in_keys=[], out_keys=keys, rows=float(len(rows)),
                    fn=lambda: [dict(r) for r in rows])


def sink(node_id: str, inputs: List[str], keep: List[str]) -> PlanNode:
    return PlanNode(id=node_id, op="sink", name="sink", inputs=inputs,
                    in_keys=list(keep), out_keys=list(keep), keep=list(keep))


# ---------------------------------------------------------------- fixtures

DOMAINS = ["ml", "health", "law", "sports"]  # sports is out of scope


def research_queries() -> List[Row]:
    topics = ["transformer interpretability", "vaccine efficacy",
              "contract clause extraction", "football transfers"]
    return [{"qid": f"q{i}", "query": t, "domain": DOMAINS[i]}
            for i, t in enumerate(topics)]


def ticket_rows(n: int = 16) -> List[Row]:
    kinds = ["billing", "outage", "howto"]
    return [{"tid": f"t{i:02d}", "text": f"{kinds[i % 3]} issue #{i} " * 4}
            for i in range(n)]


def file_rows(n: int = 12) -> List[Row]:
    return [{"fid": f"f{i:02d}", "lang": "python",
             "code": f"def fn_{i}():\n    return {i}  # {'x' * (i * 7 % 40)}"}
            for i in range(n)]


def patient_rows(n: int = 20) -> List[Row]:
    syms = ["headache", "chest pain", "fever", "back pain", "cough"]
    rows = []
    for i in range(n):
        age = 24 + (i * 13) % 60
        s = syms[i % len(syms)]
        rows.append({"pid": f"p{i:02d}", "symptoms": s, "age": age})
    return rows


def data_rows(n: int = 24) -> List[Row]:
    return [{"rid": f"r{i:02d}", "lang": "en",
             "text": f"record {i} about topic {i % 6} " * 6} for i in range(n)]
