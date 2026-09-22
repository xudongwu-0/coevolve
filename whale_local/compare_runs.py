"""Compare a completed WHALE run and its gate variant on the same held-out positions."""
from pathlib import Path
import argparse
import hashlib
import json
import re

import numpy as np
import pandas as pd


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]


def heldout(run, name, expected):
    scores = {}
    for row in lines(run / name / "trajectories.jsonl"):
        match = re.search(r"(?m)^FEN: (.+)$", row["question"])
        assert match, "Cannot pair redacted example IDs without a visible initial FEN"
        key = " ".join(match.group(1).split()[:4])
        assert key in expected and key not in scores
        assert row["answer_redacted"] and not row["answer"]
        scores[key] = int(row["reward"] > 0.5)
    assert set(scores) == expected
    return np.array([scores[k] for k in sorted(expected)], dtype=np.int8)


def paired(old, new):
    delta = new.astype(float) - old
    rng = np.random.default_rng(20260914)
    boot = delta[rng.integers(0, len(delta), size=(20000, len(delta)))].mean(axis=1)
    return {"before_solved": int(old.sum()), "after_solved": int(new.sum()), "n": len(delta),
            "gain_positions": int((delta > 0).sum()), "regression_positions": int((delta < 0).sum()),
            "delta_percentage_points": float(delta.mean() * 100),
            "paired_position_bootstrap_95_percent_interval_pp": (np.quantile(boot, [0.025, 0.975]) * 100).tolist(),
            "bootstrap_samples": 20000, "bootstrap_seed": 20260914}


def cost(run):
    summary = read(run / "summary.json")
    stages = [read(p) for p in sorted(run.glob("round_*/checkpoint/training_state.json"))]
    waiting = []
    for p in sorted((run / "proposer_jobs").glob("*/request.json")):
        response = p.with_name("response.json")
        assert response.exists()
        waiting.append({"job": p.parent.name, "seconds": max(0, response.stat().st_mtime - p.stat().st_mtime)})
    mh_paths = {str(Path(p).resolve()) for stage in run.glob("round_*/mh_result.json")
                for p in read(stage)["evaluation_files"]}
    eval_paths = [run / name / "val.json" for name in
                  ("test_initial", "test_final", "test_weights_only", "test_harness_only")]
    n_mh = sum(read(p)["num_examples"] for p in mh_paths)
    n_test = sum(read(p)["num_examples"] for p in eval_paths)
    train_tokens = 0
    for stage in run.glob("round_*"):
        if not (stage / "train_rollouts.jsonl").exists():
            continue
        train_tokens += sum(len(c["response_ids"]) for r in lines(stage / "train_rollouts.jsonl") for c in r["policy_calls"])
    extra_tokens = sum((s.get("gate") or {}).get("extra_generated_tokens", 0) for s in stages)
    eval_tokens = sum(read(p)["avg_metrics"]["assistant_tokens_used"] * read(p)["num_examples"] for p in [*mh_paths, *eval_paths])
    wait_seconds = sum(p["seconds"] for p in waiting)
    elapsed = summary["elapsed_seconds_this_invocation"]
    base_rollouts = sum(s["rollouts"] for s in stages)
    extra_rollouts = sum((s.get("gate") or {}).get("extra_rollouts", 0) for s in stages)
    return {"base_training_rollouts": base_rollouts, "gate_extra_rollouts": extra_rollouts,
            "mh_eval_rollouts": n_mh, "four_cell_test_rollouts": n_test,
            "total_rollouts": base_rollouts + extra_rollouts + n_mh + n_test,
            "base_generated_tokens": train_tokens, "gate_generated_tokens": extra_tokens,
            "eval_generated_tokens": round(eval_tokens), "total_generated_tokens": round(train_tokens + extra_tokens + eval_tokens),
            "accepted_training_trajectories": sum(s["accepted_trajectories"] for s in stages),
            "sft_response_tokens": sum(s["response_tokens"] for s in stages),
            "optimizer_updates": sum(int(s["metrics"]["actor/sft_updates"]) for s in stages),
            "sampling_seconds": sum(s["sampling_seconds"] for s in stages),
            "gate_seconds": sum((s.get("gate") or {}).get("elapsed_seconds", 0) for s in stages),
            "sft_and_hash_seconds": sum(s["sft_and_hash_seconds"] for s in stages),
            "training_phase_seconds_including_checkpoints": sum(s["elapsed_seconds"] for s in stages),
            "peak_gpu_allocated_gib": max(s["peak_gpu_allocated_bytes"] for s in stages) / 2**30,
            "peak_gpu_reserved_gib": max(s["peak_gpu_reserved_bytes"] for s in stages) / 2**30,
            "wall_seconds": elapsed, "active_agent_wait_seconds": wait_seconds,
            "wall_minus_proposal_queue_wait_seconds": max(0, elapsed - wait_seconds), "proposal_waits": waiting}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gate-diagnostics", type=Path)
    args = parser.parse_args()
    runs = {"whale": args.baseline.resolve(), "whale_gate": args.gate.resolve()}
    configs = {k: read(p / "config.json") for k, p in runs.items()}
    for run in runs.values():
        assert read(run / "summary.json")["status"] == "complete"
        assert read(run / "verification.json")["passed"]
    assert not configs["whale"].get("gate") and configs["whale_gate"]["gate"]
    for key in ("rounds", "rollouts_per_puzzle", "learning_rate", "sft_epochs", "seed", "assistant_token_budget", "policy_max_tokens", "upstream_commit"):
        assert configs["whale"][key] == configs["whale_gate"][key], key
    for split in ("train", "mh_val", "test"):
        assert configs["whale"]["data"]["splits"][split]["sha256"] == configs["whale_gate"]["data"]["splits"][split]["sha256"]
    assert read(runs["whale"] / "initial_identity.json")["weight_digest"] == read(runs["whale_gate"] / "initial_identity.json")["weight_digest"]
    data = configs["whale"]["data"]["splits"]["test"]
    expected = {" ".join(json.loads(x)["start_fen"].split()[:4]) for x in pd.read_parquet(data["path"])["extra_info"]}
    assert len(expected) == data["count"]
    cells = {k: {name: heldout(p, name, expected) for name in
                  ("test_initial", "test_weights_only", "test_harness_only", "test_final")} for k, p in runs.items()}
    assert np.array_equal(cells["whale"]["test_initial"], cells["whale_gate"]["test_initial"]), "Base evaluation did not reproduce"
    a, b = [lines(p / "round_01/train_rollouts.jsonl") for p in runs.values()]
    assert len(a) == len(b)
    mismatches = []
    for i, (left, right) in enumerate(zip(a, b)):
        if any(left.get(k) != right.get(k) for k in ("task_key", "rollout_seed", "harness_sha256", "reward", "stop_condition", "policy_calls")):
            mismatches.append(i)
    score_table = {k: {name: {"solved": int(v.sum()), "n": len(v), "success_rate": float(v.mean())}
                      for name, v in group.items()} for k, group in cells.items()}
    contrasts = {"whale_vs_base": paired(cells["whale"]["test_initial"], cells["whale"]["test_final"]),
                 "gate_vs_base": paired(cells["whale_gate"]["test_initial"], cells["whale_gate"]["test_final"]),
                 "gate_vs_whale": paired(cells["whale"]["test_final"], cells["whale_gate"]["test_final"])}
    costs = {k: cost(p) for k, p in runs.items()}
    result = {"scope": "single-seed local WHALE online RSFT+MH versus the same pipeline with failure gate",
              "position_pairing": "visible initial FEN first four fields; redacted IDs are not used",
              "n_unique_test_positions": len(expected), "seed": configs["whale"]["seed"],
              "first_round_sampling": {"n": len(a), "exact_match": not mismatches, "mismatched_indices": mismatches},
              "scores": score_table, "contrasts": contrasts, "costs": costs,
              "limitations": ["single seed, easy filtered 128-position pilot, not paper configuration",
                              "same base rollout budget; gate uses additional diagnostic compute",
                              "bootstrap intervals describe position sampling, not seed-to-seed variation",
                              "local adapter backend; original distributed Ray/vLLM launcher not validated"],
              "runs": {k: str(v) for k, v in runs.items()}}
    if args.gate_diagnostics:
        diagnostic = read(args.gate_diagnostics)
        assert all(c["post_move_checkmate_verified"] for r in diagnostic["rounds"] for c in r["cases"])
        result["posthoc_gate_diagnostics"] = diagnostic
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    header = "# WHALE 与 WHALE + gate：本机小规模结果\n\n"
    report = [header, "同一 2B 初始模型、128 train / 64 MH validation / 128 held-out test、两轮、G=4、LR=1e-6。",
              "第一轮原始采样逐 token 配对：" + (f"{len(a)}/{len(a)} 相同。" if not mismatches else f"{len(mismatches)} 条不一致，不能宣称完全配对。"),
              "\n| 组合 | WHALE | WHALE + gate |\n|---|---:|---:|"]
    for name, label in [("test_initial", "初始权重 + 初始 harness"), ("test_weights_only", "最终权重 + 初始 harness"),
                        ("test_harness_only", "初始权重 + 最终 harness"), ("test_final", "最终权重 + 最终 harness")]:
        report.append(f"| {label} | {score_table['whale'][name]['solved']}/{len(expected)} | {score_table['whale_gate'][name]['solved']}/{len(expected)} |")
    report.append("\n| 对照 | 差值（百分点） | 新增成功 / 退步题数 | 配对位置 bootstrap 95% 区间 |\n|---|---:|---:|---:|")
    for key, label in [("whale_vs_base", "WHALE − 初始模型"), ("gate_vs_base", "WHALE+gate − 初始模型"), ("gate_vs_whale", "WHALE+gate − WHALE")]:
        d = contrasts[key]; lo, hi = d["paired_position_bootstrap_95_percent_interval_pp"]
        report.append(f"| {label} | {d['delta_percentage_points']:+.2f} | {d['gain_positions']} / {d['regression_positions']} | [{lo:+.2f}, {hi:+.2f}] |")
    report.append("\n| 实际工作量/耗时 | WHALE | WHALE + gate |\n|---|---:|---:|")
    for key, label, divisor in [("optimizer_updates", "optimizer 更新次数", 1), ("sft_response_tokens", "训练响应 token", 1),
            ("total_rollouts", "总轨迹（含评测）", 1), ("gate_extra_rollouts", "gate 额外轨迹", 1),
            ("training_phase_seconds_including_checkpoints", "训练阶段分钟（含采样/诊断/存盘）", 60),
            ("sft_and_hash_seconds", "SFT + 参数 hash 分钟", 60),
            ("wall_minus_proposal_queue_wait_seconds", "完整流程分钟（扣除提案队列等待）", 60),
            ("active_agent_wait_seconds", "提案队列等待分钟", 60), ("peak_gpu_allocated_gib", "峰值 allocated 显存 GiB", 1),
            ("peak_gpu_reserved_gib", "峰值 reserved 显存 GiB", 1)]:
        report.append(f"| {label} | {costs['whale'][key]/divisor:.2f} | {costs['whale_gate'][key]/divisor:.2f} |")
    report += ["\n该比较使用相同基础采样预算；gate 额外诊断和补采样不是免费的。两组均直接调用未修改的 WHALE RSFT actor、Chess verifier 和 MH 选择器。",
               "单个种子、128 个简单位置只支持探索性结论；区间不包括训练随机种子变化。没有验证原版分布式 launcher，也不是论文成绩复现。",
               "\n结构化证据见 comparison.json；每条运行的 verification.json 验证 checkpoint、optimizer 连续性和 SFT token 对齐。"]
    if args.gate_diagnostics:
        counts = [r["legal_wrong_with_visible_mate_SAN"] for r in diagnostic["rounds"]]
        routed = sum(r["labels"].get("weight_candidate", 0) for r in diagnostic["rounds"])
        report.append(f"\n事后训练轨迹审计：两轮分别有 {counts} 条轨迹选择合法将死着法，但因不匹配数据指定续着被原生 verifier 判错，其中 {routed} 条被 gate 标为 weight_candidate。已在可见 FEN 上单独验证这些已选着法确实将死，无搜索、无读取隐藏答案。原评测规则保持不变；这说明 weight_candidate 是有限干预下的训练候选标签，不能当作确定的能力缺陷，后续需要显式处理评分歧义。该检查未用于本轮训练、harness 选择或测试筛选。")
    (args.output / "REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"scores": score_table, "contrasts": contrasts, "first_round_sampling": result["first_round_sampling"]}, indent=2))


if __name__ == "__main__":
    main()
