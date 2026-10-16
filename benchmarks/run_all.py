"""Run the full benchmark suite E1-E4 + optimizer overhead; write SUMMARY.md."""

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
VENV_PY = os.path.join(HERE, "..", ".venv", "bin", "python")


def main():
    # E2, E3, E4 run as subprocesses (E1 runs separately; it is the long pole)
    for script in ["run_e2_cost.py", "run_e3_ablation.py", "run_e4_sensitivity.py"]:
        print(f"=== {script} ===", flush=True)
        r = subprocess.run([VENV_PY, os.path.join(HERE, script)], capture_output=True, text=True)
        print(r.stdout[-1500:])
        if r.returncode != 0:
            print(r.stderr[-2000:])
            raise SystemExit(f"{script} failed")

    # optimizer overhead: optimize() time vs workflow execution cost
    from agentplan import workflows, optimize
    overhead = {}
    for name, builder in workflows.BUILDERS.items():
        naive = builder()
        t0 = time.perf_counter()
        optimize(naive)
        dt = time.perf_counter() - t0
        overhead[name] = {"optimize_ms": dt * 1000}
        print(f"{name}: optimize() {dt*1000:.2f} ms", flush=True)
    res_dir = os.path.join(HERE, "..", "results")
    with open(os.path.join(res_dir, "overhead.json"), "w") as f:
        json.dump(overhead, f, indent=1)

    # summary
    e1 = json.load(open(os.path.join(res_dir, "e1.json")))
    e2 = json.load(open(os.path.join(res_dir, "e2.json")))
    e3 = json.load(open(os.path.join(res_dir, "e3.json")))
    e4 = json.load(open(os.path.join(res_dir, "e4.json")))
    lines = ["# agentplan benchmark summary", ""]
    lines.append(f"E1 correctness: {e1['n_match']}/{e1['n_total']} exact matches")
    lines.append("")
    lines.append("| workflow | tok vs seq | wall vs seq | tok vs hand | wall vs hand |")
    lines.append("|---|---|---|---|---|")
    n30 = 0
    for name, r in e2["per_workflow"].items():
        lines.append(f"| {name} | {r['tok_red_vs_seq']*100:.1f}% | {r['wall_red_vs_seq']*100:.1f}% | "
                     f"{r['tok_red_vs_hand']*100:.1f}% | {r['wall_red_vs_hand']*100:.1f}% |")
        if r["tok_red_vs_seq"] >= 0.30:
            n30 += 1
    lines.append(f"\nWorkflows with >=30% token reduction vs sequential: {n30}/5")
    lines.append("\n## E3 ablation (delta vs full optimizer; + means worse without the rule)")
    for name, r in e3["per_workflow"].items():
        abl = "; ".join(f"{k}: tok {v['tok_delta_vs_full']*100:+.1f}%, wall {v['wall_delta_vs_full']*100:+.1f}%"
                        for k, v in r["ablations"].items())
        lines.append(f"- {name}: {abl}")
    lines.append("\n## E4 sensitivity (regret vs oracle under perturbed estimates)")
    for name, r in e4["per_workflow"].items():
        bits = "; ".join(f"{k}: equiv={v['equivalent']}, regret={v['regret']*100:.2f}%"
                         for k, v in r["perturbations"].items())
        lines.append(f"- {name}: {bits}")
    lines.append("\n## Optimizer overhead")
    for name, o in overhead.items():
        lines.append(f"- {name}: {o['optimize_ms']:.2f} ms")
    with open(os.path.join(res_dir, "SUMMARY.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
