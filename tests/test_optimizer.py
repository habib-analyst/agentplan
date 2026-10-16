import pytest

from agentplan import Plan, PlanNode, optimize, execute, canonical, plan_cost
from agentplan.agents import agent, filt, join, source, sink
from agentplan import workflows


def toy_workflow():
    """src -> expensive_agent -> cheap_filter -> sink, filter written late."""
    p = Plan("toy")
    p.add(source("src", "src", [{"id": i, "x": i * 2} for i in range(6)]))
    p.add(agent("a", "expensive", ["id", "x"], ["y"],
                lambda r: {"y": r["x"] + 1}, 1000, 500, 0.02, inputs=["src"]))
    p.add(filt("f", "cheap", ["id", "x", "y"], ["x"],
                lambda r: r["x"] % 4 == 0, 0.5, ["a"], tok=5, latency=0.001))
    p.add(sink("sink", ["f"], ["id", "y"]))
    p.validate()
    return p


def test_topo_and_stages():
    p = toy_workflow()
    assert p.topo() == ["src", "a", "f", "sink"]
    stages = p.stages()
    assert stages == [["src"], ["a"], ["f"], ["sink"]]


def test_pushdown_moves_filter_and_preserves_output():
    naive = toy_workflow()
    seq = execute(naive, mode="sequential", record=False)
    opt, phys, log = optimize(naive)
    rules = [r.rule for r in log]
    assert "pushdown" in rules
    # filter now sits before the agent
    assert opt.nodes["f"].inputs == ["src"]
    assert opt.nodes["a"].inputs == ["f"]
    res = execute(opt, phys, mode="optimized", record=False)
    assert canonical(res.rows) == canonical(seq.rows)
    # and it actually saved tokens (filter dropped half before the agent)
    assert res.tokens_total < seq.tokens_total


def test_prune_dead_agent():
    p = Plan("dead")
    p.add(source("src", "src", [{"id": 1}]))
    p.add(agent("a", "used", ["id"], ["y"], lambda r: {"y": 1},
                100, 50, 0.001, inputs=["src"]))
    p.add(agent("d", "dead", ["id"], ["z"], lambda r: {"z": 2},
                100, 50, 0.001, inputs=["src"]))
    p.add(sink("sink", ["a"], ["id", "y"]))
    p.validate()
    seq = execute(p, mode="sequential", record=False)
    opt, phys, log = optimize(p)
    assert "d" not in opt.nodes
    assert any(r.rule == "prune" and "dead" in r.detail for r in log)
    res = execute(opt, phys, mode="optimized", record=False)
    assert canonical(res.rows) == canonical(seq.rows)


def test_route_pruning_equivalence():
    naive = workflows.build_support_triage()
    seq = execute(naive, mode="sequential", record=False)
    opt, phys, log = optimize(naive)
    assert any("route pruning" in r.detail for r in log)
    res = execute(opt, phys, mode="optimized", record=False)
    assert canonical(res.rows) == canonical(seq.rows)
    routed = sum(1 for r in log if "route pruning" in r.detail)
    assert routed == 3
    # routed specialists saw fewer rows
    assert res.metrics["spec_a"].rows_in < seq.metrics["spec_a"].rows_in


def test_semi_join_driven_reduction():
    naive = workflows.build_data_pipeline()
    seq = execute(naive, mode="sequential", record=False)
    opt, phys, log = optimize(naive)
    assert opt.nodes["val"].inputs == ["clean"], "validator pushed before enricher"
    assert "j" in phys.driven, "driven semi-join expected on join_enrich"
    res = execute(opt, phys, mode="optimized", record=False)
    assert canonical(res.rows) == canonical(seq.rows)
    assert res.metrics["ref"].rows_in == 18  # only surviving rids looked up
    assert seq.metrics["ref"].rows_in == 24


def test_purity_enforced():
    p = Plan("impure")
    p.add(source("src", "src", [{"id": 1, "x": 1}]))
    p.add(agent("a", "bad", ["id", "x"], ["x"],
                lambda r: {"x": 999}, 10, 10, 0.0, inputs=["src"]))
    p.add(sink("sink", ["a"], ["id", "x"]))
    with pytest.raises(AssertionError):
        execute(p, mode="sequential", record=False)


def test_all_workflows_equivalent():
    for name, builder in workflows.BUILDERS.items():
        naive = builder()
        seq = execute(naive, mode="sequential", record=False)
        opt, phys, _ = optimize(naive)
        res = execute(opt, phys, mode="optimized", record=False)
        assert canonical(res.rows) == canonical(seq.rows), f"mismatch in {name}"
        assert len(res.rows) > 0, f"empty output in {name}"


def test_driven_gate_rejects_unprofitable():
    """Semi-join reduction serializes branches; the cost gate must say no
    when the latency penalty outweighs the token savings."""
    p = Plan("gate")
    p.add(source("sa", "a", [{"k": i} for i in range(4)]))
    p.add(agent("ea", "slow_a", ["k"], ["x"], lambda r: {"x": 1},
                10, 10, 0.5, inputs=["sa"]))
    p.add(filt("f", "sel_filter", ["k", "x"], ["x"],
                lambda r: int(r["k"]) % 2 == 0, 0.5, ["ea"], tok=5, latency=0.001))
    p.add(source("sb", "b", [{"k": i} for i in range(4)]))
    p.add(agent("eb", "slow_b", ["k"], ["y"], lambda r: {"y": 2},
                10, 10, 0.5, inputs=["sb"]))
    p.add(join("j", "j", ["f", "eb"], ["k"], ["k", "x", "y"]))
    p.add(sink("sink", ["j"], ["k", "x", "y"]))
    p.validate()
    seq = execute(p, mode="sequential", record=False)
    opt, phys, log = optimize(p)
    assert phys.driven == {}, "unprofitable semi-join must be skipped"
    assert any("not cost-profitable" in r.detail for r in log)
    res = execute(opt, phys, mode="optimized", record=False)
    assert canonical(res.rows) == canonical(seq.rows)


def test_optimized_cost_estimate_no_worse():
    for name, builder in workflows.BUILDERS.items():
        naive = builder()
        opt, _, _ = optimize(naive)
        assert plan_cost(opt).total <= plan_cost(naive).total + 1e-9, name
