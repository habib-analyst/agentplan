"""Before/after plan visualization as standalone SVG (stdlib only)."""

from __future__ import annotations

from typing import Dict, List, Optional

from .ir import Plan
from .optimize import PhysicalPlan

_COLORS = {
    "source": "#dbeafe", "agent": "#fef3c7", "filter": "#dcfce7",
    "join": "#f3e8ff", "sink": "#fee2e2",
}
_LABEL = {"source": "SCAN", "agent": "AGENT", "filter": "FILTER", "join": "JOIN", "sink": "SINK"}


def _layout(plan: Plan, stages: List[List[str]]) -> Dict[str, tuple]:
    pos: Dict[str, tuple] = {}
    W, H, DX, DY = 210, 64, 250, 96
    maxw = max(len(s) for s in stages)
    for si, stage in enumerate(stages):
        for j, nid in enumerate(stage):
            x = 20 + si * DX
            y = 20 + j * DY + (maxw - len(stage)) * DY / 2
            pos[nid] = (x, y)
    return pos


def render(plan: Plan, physical: Optional[PhysicalPlan], title: str,
           highlight: Optional[Dict[str, str]] = None) -> str:
    stages = physical.stages if physical else plan.stages()
    pos = _layout(plan, stages)
    W = 20 + len(stages) * 250
    H = int(max(y for _, y in pos.values()) + 110) if pos else 120
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
    parts.append(f'<text x="20" y="18" font-size="15" font-weight="bold" fill="#111">{title}</text>')
    for nid, n in plan.nodes.items():
        for i in n.inputs:
            if i not in pos:
                continue
            x1, y1 = pos[i]; x2, y2 = pos[nid]
            parts.append(f'<line x1="{x1 + 210}" y1="{y1 + 30}" x2="{x2}" y2="{y2 + 30}" '
                         f'stroke="#9ca3af" stroke-width="1.5" marker-end="url(#ar)"/>')
    parts.append('<defs><marker id="ar" markerWidth="8" markerHeight="8" refX="7" refY="4" '
                 'orient="auto"><path d="M0,0 L8,4 L0,8" fill="none" stroke="#9ca3af"/></marker></defs>')
    for nid, n in plan.nodes.items():
        x, y = pos[nid]
        c = _COLORS.get(n.op, "#eee")
        extra = ""
        if highlight and nid in highlight:
            extra = f'<text x="{x}" y="{y - 6}" font-size="10" fill="#b91c1c" font-weight="bold">[{highlight[nid]}]</text>'
        sel = f" sel={n.selectivity:.2f}" if n.selectivity is not None else ""
        cost = f"{n.tok_in + n.tok_out:.0f}tok/{n.latency * 1000:.0f}ms" if n.op in ("agent", "filter") else ""
        parts.append(extra)
        parts.append(f'<rect x="{x}" y="{y}" width="210" height="60" rx="8" fill="{c}" '
                     f'stroke="#374151"/>')
        parts.append(f'<text x="{x + 10}" y="{y + 20}" font-size="11" font-weight="bold" fill="#111">'
                     f'{_LABEL.get(n.op, n.op)} {n.name}</text>')
        parts.append(f'<text x="{x + 10}" y="{y + 38}" font-size="10" fill="#4b5563">{cost}{sel}</text>')
        if n.op == "join" and physical and nid in (physical.driven or {}):
            parts.append(f'<text x="{x + 10}" y="{y + 52}" font-size="10" fill="#7c3aed">semi-join driven</text>')
        if getattr(n, "route_values", None):
            parts.append(f'<text x="{x + 10}" y="{y + 52}" font-size="10" fill="#7c3aed">'
                         f'routed: {n.route_key} in {sorted(n.route_values)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def render_before_after(naive: Plan, opt: Plan, physical: PhysicalPlan,
                        rule_notes: Dict[str, str]) -> str:
    """Side-by-side HTML page with the two SVGs and rule annotations."""
    before = render(naive, None, "BEFORE: naive fixed-order plan")
    after = render(opt, physical, "AFTER: cost-optimized physical plan", rule_notes)
    notes = "".join(f"<li><b>{k}</b>: {v}</li>" for k, v in rule_notes.items())
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>agentplan: before / after</title></head>
<body style="font-family:monospace;max-width:1400px;margin:auto;padding:16px">
<h2>agentplan: your agent workflow, query-planned</h2>
<h3>Optimizer rules fired</h3><ul>{notes}</ul>
<h3>Before</h3>{before}
<h3>After</h3>{after}
</body></html>"""
