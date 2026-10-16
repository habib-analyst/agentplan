import os

from agentplan import optimize, execute, plan_cost
from agentplan.stats import StatsStore
from agentplan import workflows
from agentplan.cost import resolve


def test_stats_learn_selectivity(tmp_path):
    sp = str(tmp_path / "stats.json")
    stats = StatsStore(sp)
    naive = workflows.build_data_pipeline()
    # declared contract first
    _, _, _, _, sel0 = resolve("val", naive, stats)
    assert sel0 == 0.75
    # record 3 runs -> measured takes over
    for _ in range(3):
        execute(naive, mode="sequential", stats=stats, record=True)
    s = stats.get("data_pipeline", "val")
    assert s["n"] == 3
    assert abs(s["selectivity"] - 0.75) < 1e-9  # fixture is exact by construction
    _, _, _, _, sel1 = resolve("val", naive, stats)
    assert sel1 == s["selectivity"]


def test_optimizer_uses_measured_stats(tmp_path):
    sp = str(tmp_path / "stats.json")
    stats = StatsStore(sp)
    naive = workflows.build_support_triage()
    for _ in range(4):
        opt, phys, _ = optimize(naive, stats)
        execute(opt, phys, mode="optimized", stats=stats, record=True)
    m = stats.get("support_triage", "spec_a")
    assert m["n"] == 4
    # 16 tickets routed A/B/C by tid%3 -> spec_a sees 6 of 16 per run
    assert m["rows_in"] == 6 * 4


def test_stats_persist(tmp_path):
    sp = str(tmp_path / "s.json")
    s1 = StatsStore(sp)
    naive = workflows.build_code_review()
    execute(naive, mode="sequential", stats=s1, record=True)
    s2 = StatsStore(sp)
    assert s2.get("code_review", "lint")["n"] == 1
