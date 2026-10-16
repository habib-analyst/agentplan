"""Five realistic multi-agent workflows, each built in NAIVE form (the fixed
order a straightforward implementation would use). The optimizer rewrites
them; the executor's sequential mode runs the naive form.

Each workflow returns a Plan. `hand_plan(name)` returns the version a
careful engineer would hand-write: obvious predicate pushdown + parallelism,
but no semi-join reduction, no route-conditional pruning, no dead-code
elimination (the planner's edge).
"""

from __future__ import annotations

from typing import Dict, List

from .ir import Plan, Row
from .agents import (agent, filt, join, source, sink, det,
                     research_queries, ticket_rows, file_rows, patient_rows, data_rows)


def _union_keys(*nodes) -> List[str]:
    seen: List[str] = []
    for n in nodes:
        for k in n.out_keys:
            if k not in seen:
                seen.append(k)
    return seen

# ---------------------------------------------------------------- W1: research assistant


def _searcher(tag: str):
    def fn(row: Row):
        rng = det("search", tag, row["qid"], row["subq"])
        hits = rng.randint(3, 9)
        return {f"{tag}_hits": hits,
                f"{tag}_snippet": f"{tag}:{row['qid']}:{row['subq'][:14]}:{hits}"}
    return fn


def build_research_assistant() -> Plan:
    p = Plan("research_assistant")
    src = p.add(source("src", "queries", research_queries()))
    exp = p.add(agent("expand", "expand_subqueries",
                      ["qid", "query", "domain"], ["subq", "sq_idx"],
                      lambda r: [{"subq": f"{r['query']} aspect {k}", "sq_idx": k}
                                 for k in range(3)],
                      200, 150, 0.010, expand_factor=3.0, inputs=["src"]))
    sa = p.add(agent("sa", "search_a", ["qid", "subq", "sq_idx", "domain"],
                     ["a_hits", "a_snippet"], _searcher("a"), 900, 400, 0.050, inputs=["expand"]))
    sb = p.add(agent("sb", "search_b", ["qid", "subq", "sq_idx", "domain"],
                     ["b_hits", "b_snippet"], _searcher("b"), 900, 400, 0.050, inputs=["expand"]))
    sc = p.add(agent("sc", "search_c", ["qid", "subq", "sq_idx", "domain"],
                     ["c_hits", "c_snippet"], _searcher("c"), 900, 400, 0.050, inputs=["expand"]))
    jab = p.add(join("jab", "join_ab", ["sa", "sb"], ["qid", "subq"],
                     _union_keys(sa, sb)))
    jabc = p.add(join("jabc", "join_abc", ["jab", "sc"], ["qid", "subq"],
                      _union_keys(jab, sc)))
    # naive: guardrail placed LATE (after the expensive fan-out)
    guard = p.add(filt("guard", "scope_guard", _union_keys(jabc), ["domain"],
                       lambda r: r["domain"] in ("ml", "health", "law"),
                       0.75, ["jabc"], tok=5, latency=0.001))
    synth = p.add(agent("synth", "synthesize",
                        ["qid", "subq", "sq_idx", "domain",
                         "a_snippet", "b_snippet", "c_snippet"],
                        ["draft"],
                        lambda r: {"draft": f"draft({r['qid']},{r['sq_idx']}):"
                                            f"{r['a_snippet']}|{r['b_snippet']}|{r['c_snippet']}"},
                        1500, 800, 0.080, inputs=["guard"]))
    # naive: cheap evidence triage AFTER the expensive synthesizer
    fact = p.add(filt("fact", "evidence_triage",
                      ["qid", "subq", "sq_idx", "domain", "a_snippet", "b_snippet",
                       "c_snippet", "draft"],
                      ["a_snippet", "b_snippet", "c_snippet", "sq_idx"],
                      lambda r: all(r[k] for k in ("a_snippet", "b_snippet", "c_snippet"))
                      and r["sq_idx"] % 3 != 0,
                      2 / 3, ["synth"], tok=40, latency=0.004))
    # dead agent: sentiment never read by the sink
    sent = p.add(agent("sent", "sentiment_probe",
                       ["qid", "draft"], ["sentiment"],
                       lambda r: {"sentiment": "pos" if det("s", r["qid"]).random() > 0.5 else "neg"},
                       800, 100, 0.040, inputs=["fact"]))
    writer = p.add(agent("writer", "writer",
                         ["qid", "subq", "sq_idx", "draft"], ["article"],
                         lambda r: {"article": f"article({r['qid']}): {r['draft'][:64]}"},
                         1200, 900, 0.070, inputs=["fact"]))
    p.add(sink("sink", ["writer"], ["qid", "article"]))
    p.meta["dead"] = ["sent"]
    p.validate()
    return p


def hand_research_assistant() -> Plan:
    """Engineer's version: guard first, triage before synthesis, parallel.
    Keeps the dead sentiment probe (template cruft nobody removed)."""
    p = Plan("research_assistant")
    src = p.add(source("src", "queries", research_queries()))
    guard = p.add(filt("guard", "scope_guard", ["qid", "query", "domain"], ["domain"],
                       lambda r: r["domain"] in ("ml", "health", "law"),
                       0.75, ["src"], tok=5, latency=0.001))
    exp = p.add(agent("expand", "expand_subqueries",
                      ["qid", "query", "domain"], ["subq", "sq_idx"],
                      lambda r: [{"subq": f"{r['query']} aspect {k}", "sq_idx": k}
                                 for k in range(3)],
                      200, 150, 0.010, expand_factor=3.0, inputs=["guard"]))
    sa = p.add(agent("sa", "search_a", ["qid", "subq", "sq_idx", "domain"],
                     ["a_hits", "a_snippet"], _searcher("a"), 900, 400, 0.050, inputs=["expand"]))
    sb = p.add(agent("sb", "search_b", ["qid", "subq", "sq_idx", "domain"],
                     ["b_hits", "b_snippet"], _searcher("b"), 900, 400, 0.050, inputs=["expand"]))
    sc = p.add(agent("sc", "search_c", ["qid", "subq", "sq_idx", "domain"],
                     ["c_hits", "c_snippet"], _searcher("c"), 900, 400, 0.050, inputs=["expand"]))
    jab = p.add(join("jab", "join_ab", ["sa", "sb"], ["qid", "subq"], _union_keys(sa, sb)))
    jabc = p.add(join("jabc", "join_abc", ["jab", "sc"], ["qid", "subq"], _union_keys(jab, sc)))
    fact = p.add(filt("fact", "evidence_triage", _union_keys(jabc),
                      ["a_snippet", "b_snippet", "c_snippet", "sq_idx"],
                      lambda r: all(r[k] for k in ("a_snippet", "b_snippet", "c_snippet"))
                      and r["sq_idx"] % 3 != 0,
                      2 / 3, ["jabc"], tok=40, latency=0.004))
    synth = p.add(agent("synth", "synthesize",
                        ["qid", "subq", "sq_idx", "domain", "a_snippet", "b_snippet", "c_snippet"],
                        ["draft"],
                        lambda r: {"draft": f"draft({r['qid']},{r['sq_idx']}):"
                                            f"{r['a_snippet']}|{r['b_snippet']}|{r['c_snippet']}"},
                        1500, 800, 0.080, inputs=["fact"]))
    sent = p.add(agent("sent", "sentiment_probe", ["qid", "draft"], ["sentiment"],
                       lambda r: {"sentiment": "pos" if det("s", r["qid"]).random() > 0.5 else "neg"},
                       800, 100, 0.040, inputs=["synth"]))
    writer = p.add(agent("writer", "writer", ["qid", "subq", "sq_idx", "draft"], ["article"],
                         lambda r: {"article": f"article({r['qid']}): {r['draft'][:64]}"},
                         1200, 900, 0.070, inputs=["synth"]))
    p.add(sink("sink", ["writer"], ["qid", "article"]))
    p.validate()
    return p

# ---------------------------------------------------------------- W2: data pipeline


def build_data_pipeline() -> Plan:
    p = Plan("data_pipeline")
    src = p.add(source("src", "records", data_rows()))
    clean = p.add(agent("clean", "cleaner", ["rid", "text", "lang"], ["clean_text"],
                        lambda r: {"clean_text": r["text"].strip().lower()[:60]},
                        300, 200, 0.015, inputs=["src"]))
    # naive: expensive enricher BEFORE the cheap validator
    enr = p.add(agent("enr", "enricher", ["rid", "clean_text", "lang"], ["entities"],
                      lambda r: {"entities": f"ent({r['rid']}):" + ",".join(
                          sorted(set(r["clean_text"].split()))[:5])},
                      1100, 500, 0.060, inputs=["clean"]))
    val = p.add(filt("val", "validator", ["rid", "clean_text", "lang", "entities"],
                     ["clean_text", "rid"],
                     lambda r: len(r["clean_text"]) > 10 and int(r["rid"][1:]) % 10 < 7,
                     0.75, ["enr"], tok=30, latency=0.002))
    sref = p.add(source("sref", "ref_table",
                        [{"rid": f"r{i:02d}", "ref": f"ref-r{i:02d}"} for i in range(24)]))
    ref = p.add(agent("ref", "ref_lookup", ["rid", "ref"], ["ref_meta"],
                      lambda r: {"ref_meta": f"meta({r['ref']})"},
                      400, 150, 0.020, inputs=["sref"]))
    j = p.add(join("j", "join_enrich", ["val", "ref"], ["rid"], _union_keys(val, ref)))
    agg = p.add(agent("agg", "aggregator",
                      ["rid", "clean_text", "entities", "ref_meta"], ["report"],
                      lambda rows: [{"report": f"report: {len(rows)} rows enriched"}],
                      2000, 1000, 0.050, fan_in=True, inputs=["j"]))
    p.add(sink("sink", ["agg"], ["report"]))
    p.validate()
    return p


def hand_data_pipeline() -> Plan:
    p = Plan("data_pipeline")
    src = p.add(source("src", "records", data_rows()))
    clean = p.add(agent("clean", "cleaner", ["rid", "text", "lang"], ["clean_text"],
                        lambda r: {"clean_text": r["text"].strip().lower()[:60]},
                        300, 200, 0.015, inputs=["src"]))
    val = p.add(filt("val", "validator", ["rid", "clean_text", "lang"],
                     ["clean_text", "rid"],
                     lambda r: len(r["clean_text"]) > 10 and int(r["rid"][1:]) % 10 < 7,
                     0.75, ["clean"], tok=30, latency=0.002))
    enr = p.add(agent("enr", "enricher", ["rid", "clean_text", "lang"], ["entities"],
                      lambda r: {"entities": f"ent({r['rid']}):" + ",".join(
                          sorted(set(r["clean_text"].split()))[:5])},
                      1100, 500, 0.060, inputs=["val"]))
    sref = p.add(source("sref", "ref_table",
                        [{"rid": f"r{i:02d}", "ref": f"ref-r{i:02d}"} for i in range(24)]))
    ref = p.add(agent("ref", "ref_lookup", ["rid", "ref"], ["ref_meta"],
                      lambda r: {"ref_meta": f"meta({r['ref']})"},
                      400, 150, 0.020, inputs=["sref"]))
    j = p.add(join("j", "join_enrich", ["enr", "ref"], ["rid"], _union_keys(enr, ref)))
    agg = p.add(agent("agg", "aggregator",
                      ["rid", "clean_text", "entities", "ref_meta"], ["report"],
                      lambda rows: [{"report": f"report: {len(rows)} rows enriched"}],
                      2000, 1000, 0.050, fan_in=True, inputs=["j"]))
    p.add(sink("sink", ["agg"], ["report"]))
    p.validate()
    return p

# ---------------------------------------------------------------- W3: support triage


def _specialist(name: str):
    def fn(row: Row):
        rng = det("spec", name, row["tid"])
        return {"spec_text": f"{name} fix for {row['tid']}: step{rng.randint(1,5)}"}
    return fn


def build_support_triage() -> Plan:
    p = Plan("support_triage")
    src = p.add(source("src", "tickets", ticket_rows()))
    clf = p.add(agent("clf", "classifier", ["tid", "text"], ["route", "conf"],
                      lambda r: {"route": "ABC"[int(r["tid"][1:]) % 3], "conf": 0.9},
                      250, 50, 0.008, inputs=["src"]))
    specs = []
    for tag, nm in (("a", "spec_a"), ("b", "spec_b"), ("c", "spec_c")):
        specs.append(p.add(agent(nm, nm, ["tid", "text", "route", "conf"], ["spec_text"],
                                 _specialist(nm), 900, 350, 0.050, inputs=["clf"])))
    # naive: run ALL specialists on every ticket, merge picks the routed one
    merge = p.add(join("merge", "merge", ["clf"] + [s.id for s in specs], ["tid"],
                       ["tid", "text", "route", "conf", "spec_text"],
                       route_key="route",
                       route_map={"A": "spec_a", "B": "spec_b", "C": "spec_c"}))
    res = p.add(agent("res", "resolver", ["tid", "route", "spec_text"], ["resolution"],
                      lambda r: {"resolution": f"resolved({r['tid']},{r['route']}): "
                                               f"{r['spec_text'][:40]}"},
                      500, 200, 0.020, inputs=["merge"]))
    audit = p.add(agent("audit", "audit_logger", ["tid", "resolution"], ["audit_note"],
                        lambda r: {"audit_note": f"audit {r['tid']}"},
                        300, 100, 0.010, inputs=["res"]))
    p.add(sink("sink", ["res"], ["tid", "route", "resolution"]))
    p.meta["dead"] = ["audit"]
    p.validate()
    return p


def hand_support_triage() -> Plan:
    # A careful engineer still writes run-all-then-pick in a fixed-order DSL;
    # the obvious wins here are parallelism only.
    return build_support_triage()

# ---------------------------------------------------------------- W4: code review


def build_code_review() -> Plan:
    p = Plan("code_review")
    src = p.add(source("src", "files", file_rows()))
    # naive: expensive reviewer BEFORE the cheap linter
    rev = p.add(agent("rev", "reviewer", ["fid", "code", "lang"], ["review"],
                      lambda r: {"review": f"review({r['fid']}): "
                                          f"issues={det('rev', r['fid']).randint(0, 5)}"},
                      1400, 600, 0.070, inputs=["src"]))
    lint = p.add(filt("lint", "linter", ["fid", "code", "lang", "review"],
                      ["code", "fid"],
                      lambda r: int(r["fid"][1:]) % 2 == 0,
                      0.5, ["rev"], tok=20, latency=0.002))
    smeta = p.add(source("smeta", "file_meta",
                         [{"fid": f"f{i:02d}", "author": f"dev{i % 4}"} for i in range(12)]))
    menr = p.add(agent("menr", "meta_enrich", ["fid", "author"], ["author_meta"],
                       lambda r: {"author_meta": f"meta({r['author']})"},
                       300, 100, 0.015, inputs=["smeta"]))
    j = p.add(join("j", "join_meta", ["lint", "menr"], ["fid"], _union_keys(lint, menr)))
    summ = p.add(agent("summ", "summarizer",
                       ["fid", "review", "author_meta"], ["summary"],
                       lambda rows: [{"summary": f"summary of {len(rows)} reviews"}],
                       1500, 700, 0.040, fan_in=True, inputs=["j"]))
    p.add(sink("sink", ["summ"], ["summary"]))
    p.validate()
    return p


def hand_code_review() -> Plan:
    p = Plan("code_review")
    src = p.add(source("src", "files", file_rows()))
    lint = p.add(filt("lint", "linter", ["fid", "code", "lang"], ["code", "fid"],
                      lambda r: int(r["fid"][1:]) % 2 == 0,
                      0.5, ["src"], tok=20, latency=0.002))
    rev = p.add(agent("rev", "reviewer", ["fid", "code", "lang"], ["review"],
                      lambda r: {"review": f"review({r['fid']}): "
                                          f"issues={det('rev', r['fid']).randint(0, 5)}"},
                      1400, 600, 0.070, inputs=["lint"]))
    smeta = p.add(source("smeta", "file_meta",
                         [{"fid": f"f{i:02d}", "author": f"dev{i % 4}"} for i in range(12)]))
    menr = p.add(agent("menr", "meta_enrich", ["fid", "author"], ["author_meta"],
                       lambda r: {"author_meta": f"meta({r['author']})"},
                       300, 100, 0.015, inputs=["smeta"]))
    j = p.add(join("j", "join_meta", ["rev", "menr"], ["fid"], _union_keys(rev, menr)))
    summ = p.add(agent("summ", "summarizer",
                       ["fid", "review", "author_meta"], ["summary"],
                       lambda rows: [{"summary": f"summary of {len(rows)} reviews"}],
                       1500, 700, 0.040, fan_in=True, inputs=["j"]))
    p.add(sink("sink", ["summ"], ["summary"]))
    p.validate()
    return p

# ---------------------------------------------------------------- W5: medical intake


_SPEC = {"headache": "neurology", "chest pain": "cardiology", "fever": "internal",
         "back pain": "orthopedics", "cough": "pulmonology"}


def build_medical_intake() -> Plan:
    p = Plan("medical_intake")
    src = p.add(source("src", "patients", patient_rows()))
    ext = p.add(agent("ext", "symptom_extractor", ["pid", "symptoms", "age"],
                      ["symptom_codes"],
                      lambda r: {"symptom_codes": r["symptoms"][:3].upper()},
                      500, 150, 0.020, inputs=["src"]))
    # naive: expensive recommender BEFORE the cheap red-flag filter
    rec = p.add(agent("rec", "specialist_recommender",
                      ["pid", "symptom_codes", "age", "symptoms"], ["specialty"],
                      lambda r: {"specialty": _SPEC[r["symptoms"]]},
                      1100, 300, 0.060, inputs=["ext"]))
    flag = p.add(filt("flag", "redflag_filter",
                      ["pid", "symptom_codes", "age", "symptoms", "specialty"],
                      ["symptoms", "age"],
                      lambda r: r["symptoms"] == "chest pain" or r["age"] >= 80,
                      0.3, ["rec"], tok=25, latency=0.002))
    sins = p.add(source("sins", "insurance_feed",
                        [{"pid": f"p{i:02d}", "plan": f"plan-{i % 3}"} for i in range(20)]))
    ins = p.add(agent("ins", "insurance_lookup", ["pid", "plan"], ["coverage"],
                      lambda r: {"coverage": f"cov({r['plan']})"},
                      350, 120, 0.015, inputs=["sins"]))
    j = p.add(join("j", "join_insurance", ["flag", "ins"], ["pid"], _union_keys(flag, ins)))
    sched = p.add(agent("sched", "scheduler",
                        ["pid", "specialty", "coverage"], ["slot"],
                        lambda r: {"slot": f"slot({r['pid']},{r['specialty']})"},
                        250, 80, 0.010, inputs=["j"]))
    p.add(sink("sink", ["sched"], ["pid", "specialty", "slot"]))
    p.validate()
    return p


def hand_medical_intake() -> Plan:
    p = Plan("medical_intake")
    src = p.add(source("src", "patients", patient_rows()))
    flag = p.add(filt("flag", "redflag_filter", ["pid", "symptoms", "age"],
                      ["symptoms", "age"],
                      lambda r: r["symptoms"] == "chest pain" or r["age"] >= 80,
                      0.3, ["src"], tok=25, latency=0.002))
    ext = p.add(agent("ext", "symptom_extractor", ["pid", "symptoms", "age"],
                      ["symptom_codes"],
                      lambda r: {"symptom_codes": r["symptoms"][:3].upper()},
                      500, 150, 0.020, inputs=["flag"]))
    rec = p.add(agent("rec", "specialist_recommender",
                      ["pid", "symptom_codes", "age", "symptoms"], ["specialty"],
                      lambda r: {"specialty": _SPEC[r["symptoms"]]},
                      1100, 300, 0.060, inputs=["ext"]))
    sins = p.add(source("sins", "insurance_feed",
                        [{"pid": f"p{i:02d}", "plan": f"plan-{i % 3}"} for i in range(20)]))
    ins = p.add(agent("ins", "insurance_lookup", ["pid", "plan"], ["coverage"],
                      lambda r: {"coverage": f"cov({r['plan']})"},
                      350, 120, 0.015, inputs=["sins"]))
    j = p.add(join("j", "join_insurance", ["rec", "ins"], ["pid"], _union_keys(rec, ins)))
    sched = p.add(agent("sched", "scheduler", ["pid", "specialty", "coverage"], ["slot"],
                        lambda r: {"slot": f"slot({r['pid']},{r['specialty']})"},
                        250, 80, 0.010, inputs=["j"]))
    p.add(sink("sink", ["sched"], ["pid", "specialty", "slot"]))
    p.validate()
    return p


BUILDERS: Dict[str, callable] = {
    "research_assistant": build_research_assistant,
    "data_pipeline": build_data_pipeline,
    "support_triage": build_support_triage,
    "code_review": build_code_review,
    "medical_intake": build_medical_intake,
}

HAND_BUILDERS: Dict[str, callable] = {
    "research_assistant": hand_research_assistant,
    "data_pipeline": hand_data_pipeline,
    "support_triage": hand_support_triage,
    "code_review": hand_code_review,
    "medical_intake": hand_medical_intake,
}
