"""History-based estimation: measured selectivities and costs from past runs.

The optimizer uses declared contracts until >= MIN_SAMPLES runs have been
recorded for a (workflow, node), then switches to measured means. Stored as
plain JSON; atomic replace on write.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from .ir import Plan


class StatsStore:
    def __init__(self, path: str) -> None:
        self.path = path
        self.data: Dict[str, Dict[str, Dict[str, Any]]] = {}
        if os.path.exists(path):
            with open(path) as f:
                self.data = json.load(f)

    def get(self, workflow: str, node_id: str) -> Optional[Dict[str, Any]]:
        return self.data.get(workflow, {}).get(node_id)

    def record_run(self, workflow: str, plan: Plan, metrics: Dict) -> None:
        wf = self.data.setdefault(workflow, {})
        for nid, m in metrics.items():
            n = plan.nodes[nid]
            e = wf.setdefault(nid, {"n": 0, "rows_in": 0, "rows_out": 0,
                                   "tokens": 0.0, "wall": 0.0})
            e["n"] += 1
            e["rows_in"] += m.rows_in
            e["rows_out"] += m.rows_out
            e["tokens"] += m.tokens
            e["wall"] += m.wall
            if e["rows_in"] > 0:
                if n.op == "filter":
                    e["selectivity"] = e["rows_out"] / e["rows_in"]
                e["mean_tokens"] = e["tokens"] / e["rows_in"]
                e["mean_latency"] = e["wall"] / max(1, e["rows_in"])
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=1)
        os.replace(tmp, self.path)

    def summary(self, workflow: str) -> Dict[str, Dict[str, Any]]:
        return self.data.get(workflow, {})
