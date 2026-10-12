"""Check whether a frozen WHALE result improves both conditional axes."""
import argparse
import json
from pathlib import Path

import numpy as np

from pipeline import sha, write_json


def paired_change(before, after):
    delta = np.asarray(after, dtype=float) - np.asarray(before, dtype=float)
    rng = np.random.default_rng(20260914)
    bootstrap = rng.choice(delta, size=(20000, len(delta)), replace=True).mean(axis=1)
    return {"difference_pp": float(100 * delta.mean()),
            "new_successes": int((delta > 0).sum()),
            "regressions": int((delta < 0).sum()),
            "paired_bootstrap_95_percent_pp": (100 * np.quantile(bootstrap, [0.025, 0.975])).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--minimum-gain-pp", type=float, default=5.0)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    config = json.loads((run / "config.json").read_text())
    assert not config.get("gate"), "Qualify native WHALE before adding a gate"
    assert json.loads((run / "verification.json").read_text())["passed"]
    names = {"W0_H0": "test_initial", "Wf_H0": "test_weights_only",
             "W0_Hf": "test_harness_only", "Wf_Hf": "test_final"}
    scores, outcomes, identities = {}, {}, {}
    for cell, name in names.items():
        report = json.loads((run / name / "val.json").read_text())
        assert sha(report["metadata"]["dataset_path"]) == config["data"]["splits"]["test"]["sha256"]
        assert report["metadata"]["seed"] == 42
        rows = [json.loads(line) for line in (run / name / "trajectories.jsonl").read_text().splitlines()]
        outcomes[cell] = [int(r["reward"] > 0.5) for r in rows]
        assert len(rows) == config["data"]["splits"]["test"]["count"]
        assert sum(outcomes[cell]) == report["solved_examples"]
        scores[cell] = {"solved": sum(outcomes[cell]), "total": len(rows)}
        identities[cell] = json.loads((run / name / "identity.json").read_text())
    assert identities["W0_H0"]["weight_digest"] == identities["W0_Hf"]["weight_digest"]
    assert identities["Wf_H0"]["weight_digest"] == identities["Wf_Hf"]["weight_digest"]
    assert identities["W0_H0"]["harness_sha256"] == identities["Wf_H0"]["harness_sha256"]
    assert identities["W0_Hf"]["harness_sha256"] == identities["Wf_Hf"]["harness_sha256"]
    contrasts = {"harness_at_final_weights": paired_change(outcomes["Wf_H0"], outcomes["Wf_Hf"]),
                 "weights_at_final_harness": paired_change(outcomes["W0_Hf"], outcomes["Wf_Hf"]),
                 "weights_at_base_harness": paired_change(outcomes["W0_H0"], outcomes["Wf_H0"]),
                 "harness_at_base_weights": paired_change(outcomes["W0_H0"], outcomes["W0_Hf"]),
                 "joint_change": paired_change(outcomes["W0_H0"], outcomes["Wf_Hf"])}
    eligible = all(contrasts[key]["difference_pp"] >= args.minimum_gain_pp
                   for key in ("harness_at_final_weights", "weights_at_final_harness"))
    result = {"qualified_for_gate": eligible, "minimum_conditional_gain_pp": args.minimum_gain_pp,
              "scores": scores, "contrasts": contrasts,
              "interpretation": "Small-scale setting qualification, not a gate improvement or full paper replication",
              "source_run": str(run), "test_sha256": config["data"]["splits"]["test"]["sha256"]}
    write_json(run / "viability.json", result)
    lines = ["# WHALE 双轴确认", "", f"是否达到预设 gate 场景门槛：{'是' if eligible else '否'}。", "",
             "| 组合 | 解题数 |", "|---|---:|"]
    lines += [f"| {cell} | {s['solved']}/{s['total']} |" for cell, s in scores.items()]
    lines += ["", "| 成对比较 | 净变化（百分点） | 95% bootstrap 区间 | 新成功 / 退步 |", "|---|---:|---|---:|"]
    for key, value in contrasts.items():
        low, high = value["paired_bootstrap_95_percent_pp"]
        lines.append(f"| {key} | {value['difference_pp']:+.2f} | [{low:+.2f}, {high:+.2f}] | {value['new_successes']} / {value['regressions']} |")
    lines += ["", "判定使用固定最终模型/最终 harness 的两个条件收益，均须达到预设 5 个百分点。",
              "此结果仅确认场景可行性。后续 gate 对比应保持训练数据和预算一致，并使用另外预留的未查看测试集。", ""]
    (run / "VIABILITY.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
