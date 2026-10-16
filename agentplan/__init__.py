"""agentplan: cost-based query planning for multi-agent orchestration.

Each agent call is a table function with estimated cost (tokens, latency,
dollars) and selectivity; handoffs are joins, guardrails are filters,
fan-out is a parallel scan. A Selinger-style optimizer applies predicate
pushdown, join reordering (+ semi-join reduction), parallelization, and
dead-agent pruning. The executor guarantees: optimized plan == sequential
plan on outputs.
"""

from .ir import Plan, PlanNode
from .cost import plan_cost, ALPHA, BETA, GAMMA
from .optimize import optimize, PhysicalPlan
from .executor import execute, canonical
from .stats import StatsStore
from . import workflows

__version__ = "0.1.0"
__all__ = ["Plan", "PlanNode", "plan_cost", "optimize", "PhysicalPlan",
           "execute", "canonical", "StatsStore", "workflows",
           "ALPHA", "BETA", "GAMMA"]
