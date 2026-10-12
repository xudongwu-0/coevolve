"""Compare an audited teacher-gate pilot with the two preserved reference runs."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from compare_runs import cost, heldout, lines, paired, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--previous-gate", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-training-rollouts", type=int, default=512)
    args = parser.parse_args()
    runs = {"whale": args.baseline.resolve(), "gate_v1": args.previous_gate.resolve(),
            "teacher_gate": args.teacher.resolve()}
    configs = {k: read(p / "config.json") for k, p in runs.items()}
    identities = {k: read(p / "initial_identity.json") for k, p in runs.items()}
    assert not configs["whale"].get("gate")
    for key, run in runs.items():
        assert read(run / "summary.json")["status"] == "complete"
        assert read(run / "verification.json")["passed"]
        assert identities[key]["weight_digest"] == identities["whale"]["weight_digest"]
        for field in ("rounds", "rollouts_per_puzzle", "learning_rate", "sft_epochs", "seed",
                      "assistant_token_budget", "policy_max_tokens", "upstream_commit"):
            assert configs[key][field] == configs["whale"][field], field
        for split in ("train", "mh_val", "test"):
            entry = configs[key]["data"]["splits"][split]
            assert entry["sha256"] == configs["whale"]["data"]["splits"][split]["sha256"]
            assert hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
    data = configs["whale"]["data"]["splits"]["test"]
    expected = {" ".join(json.loads(x)["start_fen"].split()[:4])
                for x in pd.read_parquet(data["path"])["extra_info"]}
    assert len(expected) == data["count"]
    names = ("test_initial", "test_weights_only", "test_harness_only", "test_final")
    cells = {k: {n: heldout(p, n, expected) for n in names} for k, p in runs.items()}
    assert all(np.array_equal(cells[k]["test_initial"], cells["whale"]["test_initial"]) for k in runs)

    # Only the common first wave must be identical; subsequent online models
    # and adaptive sampling choices are allowed to diverge.
    initial = {k: lines(p / "round_01/train_rollouts.jsonl") for k, p in runs.items()}
    n = configs["whale"]["data"]["splits"]["train"]["count"]
    fields = ("task_key", "rollout_seed", "harness_sha256", "reward", "stop_condition", "policy_calls")
    mismatch = [i for i, (left, right) in enumerate(zip(initial["whale"][:n], initial["teacher_gate"][:n]))
                if any(left.get(f) != right.get(f) for f in fields)]
    assert len(initial["teacher_gate"]) >= n and not mismatch, "First wave did not reproduce"
    budget = []
    for round_n in range(1, configs["whale"]["rounds"] + 1):
        per_round = {}
        for key in ("whale", "teacher_gate"):
            state = read(runs[key] / f"round_{round_n:02d}/checkpoint/training_state.json")
            per_round[key] = state["rollouts"] + (state.get("gate") or {}).get("extra_rollouts", 0)
        assert per_round["whale"] == per_round["teacher_gate"] == args.expected_training_rollouts
        budget.append({"round": round_n, **per_round})
    contrasts = {
        "teacher_vs_initial": paired(cells["teacher_gate"]["test_initial"], cells["teacher_gate"]["test_final"]),
        "teacher_vs_whale": paired(cells["whale"]["test_final"], cells["teacher_gate"]["test_final"]),
        "teacher_vs_gate_v1": paired(cells["gate_v1"]["test_final"], cells["teacher_gate"]["test_final"]),
        "teacher_weights_vs_whale_weights": paired(cells["whale"]["test_weights_only"], cells["teacher_gate"]["test_weights_only"]),
    }
    costs = {k: cost(p) for k, p in runs.items()}
    teacher_usage = [read(p) for p in sorted(runs["teacher_gate"].glob("round_*/teacher_usage.json"))]
    result = {"runs": {k: str(p) for k, p in runs.items()},
              "scores": {k: {n: {"solved": int(a.sum()), "n": len(a)} for n, a in group.items()} for k, group in cells.items()},
              "contrasts": contrasts, "costs": costs, "equal_training_rollout_budget": budget,
              "first_wave_exact_token_match": {"n": n, "passed": not mismatch},
              "teacher_usage": teacher_usage, "external_teacher_tokens_known": False,
              "limitations": ["Previously used 128-position test; exploratory follow-up, not a new blind evaluation",
                              "Equal small-model trajectory count does not equal total compute, tokens or teacher cost",
                              "Single seed; position bootstrap does not cover seed variation",
                              "gate-v1 is a historical reference with extra rollout budget",
                              "No random-gate control; label value is not isolated from other selection rules"]}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    labels = {"whale": "WHALE", "gate_v1": "WHALE + gate-v1（历史）", "teacher_gate": "WHALE + 外部教师 gate"}
    report = ["# 外部教师 gate：同采样预算两轮比较\n",
              "相同 Qwen3.5-2B 起点、128 train / 64 MH / 128 test、两轮、LR1e-6。",
              f"WHALE 与新版本每轮均使用 {args.expected_training_rollouts} 条训练/诊断轨迹；首轮前 {n} 条轨迹逐 token 相同。",
              "沿用已有测试集，结果属于探索性后续比较；未按本次测试分数挑 checkpoint、参数或种子。\n",
              "| 方法 | 最终解题数 | 准确率 |\n|---|---:|---:|",
              f"| 初始模型 | {int(cells['whale']['test_initial'].sum())}/{len(expected)} | {cells['whale']['test_initial'].mean()*100:.2f}% |"]
    for k in runs:
        a = cells[k]["test_final"]
        report.append(f"| {labels[k]} | {int(a.sum())}/{len(a)} | {a.mean()*100:.2f}% |")
    report += ["\n| 新版相对参照 | 净变化（百分点） | 新增成功 / 退步 | 配对位置 bootstrap 95% 区间 |",
               "|---|---:|---:|---:|"]
    for key, label in [("teacher_vs_initial", "初始模型"), ("teacher_vs_whale", "WHALE"), ("teacher_vs_gate_v1", "旧 gate")]:
        d = contrasts[key]; lo, hi = d["paired_position_bootstrap_95_percent_interval_pp"]
        report.append(f"| {label} | {d['delta_percentage_points']:+.2f} | {d['gain_positions']} / {d['regression_positions']} | [{lo:+.2f}, {hi:+.2f}] |")
    report += ["\n| 权重 / harness | WHALE | 旧 gate | 外部教师 gate |", "|---|---:|---:|---:|"]
    for name, label in zip(names, ["初始 / 初始", "最终 / 初始", "初始 / 最终", "最终 / 最终"]):
        report.append("| " + label + " | " + " | ".join(f"{int(cells[k][name].sum())}/{len(expected)}" for k in runs) + " |")
    report += ["\n| 实际工作量 | WHALE | 旧 gate | 外部教师 gate |", "|---|---:|---:|---:|"]
    for key, label, divisor in [("accepted_training_trajectories", "进入 SFT 的轨迹", 1),
            ("sft_response_tokens", "SFT 响应 token", 1), ("optimizer_updates", "优化步数", 1),
            ("total_rollouts", "全部小模型轨迹（含 MH/test）", 1),
            ("total_generated_tokens", "全部小模型生成 token", 1),
            ("wall_seconds", "完整墙钟分钟", 60),
            ("active_agent_wait_seconds", "外部标注/提案队列等待分钟", 60),
            ("wall_minus_proposal_queue_wait_seconds", "扣除外部队列等待后的分钟", 60),
            ("sft_and_hash_seconds", "SFT + 参数 hash 分钟", 60)]:
        report.append("| " + label + " | " + " | ".join(f"{costs[k][key]/divisor:.2f}" for k in runs) + " |")
    report += ["\n大模型标注与提案 token、费用不可由当前活跃会话队列计量，不记为零。相同轨迹数不等于相同 token 或总算力。",
               "旧 gate 有额外诊断轨迹，作为历史成绩参照；本次公平预算断言针对 WHALE 与新版。",
               "单种子与已用测试集不能证明普遍优越性或‘只升不降’。需要随机 gate 消融才能拆分教师标注与样本筛选等规则的贡献。",
               "\n数据与配对统计见 comparison.json；每条运行的 verification.json 保存训练、checkpoint 和测试身份审计。"]
    (args.output / "REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"scores": result["scores"], "contrasts": contrasts, "budgets": budget}, indent=2))


if __name__ == "__main__":
    main()
