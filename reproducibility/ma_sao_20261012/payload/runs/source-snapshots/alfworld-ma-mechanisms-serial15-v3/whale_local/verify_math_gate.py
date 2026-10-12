"""Audit a qualified-reference, matched-proposal ReTool Math sampling-gate trial."""
import argparse
from collections import Counter
import json
from pathlib import Path

import pandas as pd
import numpy as np
from safetensors import safe_open
import torch

from assess_viability import paired_change
from pipeline import sha, write_json
from math_gate import make_request, plan_routes
from verify_math_run import jsonl, native_em_scope, read, recheck_rewards


def check_calls(rows, run, digest, harness_sha):
    for row in rows:
        assert row["answer_redacted"] and row["sampling_identity"]["weight_digest"] == digest
        assert row["harness_sha256"] == harness_sha
        assert row["assistant_tokens"] == sum(len(c["response_ids"]) for c in row["policy_calls"]) <= 2048
        for call in row["policy_calls"]:
            assert call == read(run / "calls" / (call["call_id"] + ".json"))
            assert call["sampling_identity"]["weight_digest"] == digest
            assert call["stop_at_chat_end"] and len(call["response_ids"]) <= call["requested_max_tokens"] <= 1024
            assert len(call["response_logprobs"]) == len(call["response_ids"])
            assert not any(t in (248044, 248046) for t in call["response_ids"][:-1])


def cost(rows):
    calls = [c for row in rows for c in row["policy_calls"]]
    return {"trajectories": len(rows), "policy_calls": len(calls),
            "generated_tokens": sum(len(c["response_ids"]) for c in calls),
            "prompt_tokens": sum(len(c["prompt_ids"]) for c in calls)}


def repeated_outcomes(rows, task_ids, draws):
    """Keep every draw and return one mean per question, never best/last draw."""
    assert draws >= 1 and task_ids and len(set(task_ids)) == len(task_ids)
    assert len(rows) == draws * len(task_ids)
    assert [[row["draw"], row["example_index"]] for row in rows] == [
        [draw, index] for draw in range(draws) for index in range(len(task_ids))]
    assert all(row["task_id"] == task_ids[row["example_index"]] for row in rows)
    rewards = np.asarray([row["reward"] for row in rows], dtype=float).reshape(draws, len(task_ids))
    assert set(np.unique(rewards)) <= {0, 1}
    score = {"successful_draws": int(rewards.sum()), "total_draws": len(rows),
        "questions": len(task_ids), "draws_per_question": draws,
        "mean_accuracy_percent": float(100 * rewards.mean())}
    return score, dict(zip(task_ids, rewards.mean(axis=0).tolist()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    config, summary = read(run / "config.json"), read(run / "summary.json")
    assert config["gate_enabled"] and summary["gate_enabled"] and summary["status"] == "complete"
    assert config["gate_proposal_mode"] == "reference_pool" and config["parameter_dtype"] == "float32"
    reference = Path(config["gate_reference"])
    ref_config, ref_audit = read(reference / "config.json"), read(reference / "verification.json")
    assert sha(reference / "verification.json") == config["qualification_sha256"]
    assert ref_audit["passed_audit"] and ref_audit["qualified_for_gate"] and ref_audit["fp32_master_and_moments"]
    for key in ["data", "rounds", "rollouts_per_task", "learning_rate", "parameter_dtype", "mh_iterations",
                "mh_proposals", "assistant_token_budget", "policy_max_tokens", "stop_at_chat_end", "seed", "base_model"]:
        assert config[key] == ref_config[key], f"Comparison differs in {key}"
    draws = config.get("evaluation_rollouts", 1)
    temperature = config.get("evaluation_temperature", 0.0)
    assert 1 <= draws <= 8 and 0 <= temperature <= 2
    assert draws == ref_config.get("evaluation_rollouts", 1)
    assert temperature == ref_config.get("evaluation_temperature", 0.0)
    keys = {}
    for name, entry in config["data"]["splits"].items():
        assert sha(entry["path"]) == entry["sha256"]
        keys[name] = {r["task_id"] for r in pd.read_parquet(entry["path"])["extra_info"]}
    assert len(set.union(*keys.values())) == sum(map(len, keys.values()))
    scope = native_em_scope()
    previous = read(run / "initial_identity.json")
    assert previous["weight_digest"] == read(reference / "initial_identity.json")["weight_digest"]
    previous_harness, rounds = None, []
    stage_rows = {"training": [], "harness_search": [], "test": []}
    annotation_routes = Counter()
    shared_probe = None
    for number in range(1, config["rounds"] + 1):
        phase = run / f"round_{number:02d}"
        gate = phase / "gate"
        training, reload, mh = (read(phase / path) for path in
            ["checkpoint/training_state.json", "reload_verification.json", "mh_result.json"])
        first = jsonl(gate / "initial/trajectories.jsonl")
        request = make_request(first, Path(training["harness"]))
        assert request == read(gate / "teacher_request.json")
        plan = plan_routes(first, request, read(gate / "teacher_response.json"),
            seed=config["seed"] + number, total_rollouts=len(keys["train"]) * config["rollouts_per_task"])
        assert plan == read(gate / "sampling_plan.json")
        followup = jsonl(gate / "followup/trajectories.jsonl") if plan["followup_schedule"] else []
        assert [[r["draw"], r["example_index"]] for r in followup] == plan["followup_schedule"]
        rows = jsonl(phase / "rollouts/trajectories.jsonl")
        assert rows == first + followup and len(rows) == plan["total_rollouts"] == training["rollouts"]
        assert {r["task_id"] for r in first} == keys["train"]
        for row in rows:
            assert row["rollout_seed"] == (config["seed"] + number) * 100003 + row["draw"] * 1009 + row["example_index"]
        if number == 1:
            control = jsonl(reference / "round_01/rollouts/trajectories.jsonl")[:len(first)]
            for baseline, treatment in zip(control, first):
                assert baseline["task_id"] == treatment["task_id"] and baseline["reward"] == treatment["reward"]
                assert len(baseline["policy_calls"]) == len(treatment["policy_calls"])
                for a, b in zip(baseline["policy_calls"], treatment["policy_calls"]):
                    assert a["prompt_ids"] == b["prompt_ids"] and a["response_ids"] == b["response_ids"]
            if config.get("gate_first_probe_mode"):
                from math_common_probe import VERSION, first_wave
                assert config["gate_first_probe_mode"] == VERSION
                expected_rows, expected_text = first_wave(reference, identity=previous,
                    harness=Path(training["harness"]), data_entry=config["data"]["splits"]["train"],
                    seed=config["seed"] + 1)
                assert first == expected_rows
                assert (gate / "initial/trajectories.jsonl").read_text() == expected_text
                shared_probe = read(gate / "initial/SHARED_PROBE.json")
                assert shared_probe["version"] == VERSION
                assert shared_probe["reference"] == str(reference)
                assert shared_probe["reference_audit_sha256"] == sha(reference / "verification.json")
                assert shared_probe["source_trajectories_sha256"] == sha(reference / "round_01/rollouts/trajectories.jsonl")
                assert shared_probe["shared_trajectories_sha256"] == sha(gate / "initial/trajectories.jsonl")
                assert shared_probe["shared_trajectories"] == len(first)
                assert shared_probe["shared_policy_calls"] == sum(len(r["policy_calls"]) for r in first)
                assert shared_probe["shared_generated_tokens"] == sum(r["assistant_tokens"] for r in first)
                assert shared_probe["new_generations_during_import"] == 0
                assert shared_probe["counts_in_each_arm_sample_budget"] and shared_probe["not_an_algorithmic_compute_saving"]
                expected_receipts = {"calls/" + c["call_id"] + ".json" for r in first for c in r["policy_calls"]}
                expected_receipts.update("context_rejections/" + r["context_rejection_id"] + ".json"
                                         for r in first if r.get("context_rejection_id"))
                assert set(shared_probe["receipt_sha256"]) == expected_receipts
                for relative, digest in shared_probe["receipt_sha256"].items():
                    assert sha(reference / relative) == sha(run / relative) == digest
        recheck_rewards(rows, config["data"]["splits"]["train"], scope)
        check_calls(rows, run, previous["weight_digest"], training["harness_sha256"])
        assert training["sampling_identity"]["weight_digest"] == previous["weight_digest"]
        assert training["optimizer_steps_before"] == previous["optimizer_steps"]
        assert training["weight_digest"] == reload["weight_digest"] == mh["model_identity"]["weight_digest"]
        assert reload["all_parameter_hashes_equal"] and training["changed_parameter_tensors"]
        assert sha(training["harness"]) == training["harness_sha256"]
        if previous_harness:
            assert previous_harness == training["harness_sha256"]
        calls = [c for row in rows if row["reward"] for c in row["policy_calls"]]
        assert sum(r["reward"] for r in rows) == training["accepted_trajectories"]
        assert len(calls) == training["accepted_policy_calls"]
        batch = torch.load(phase / "accepted_batch.pt", map_location="cpu", weights_only=True)
        assert len(calls) == len(batch["responses"])
        pmax = batch["input_ids"].shape[1] - batch["responses"].shape[1]
        for index, call in enumerate(calls):
            assert batch["input_ids"][index, :pmax][batch["attention_mask"][index, :pmax].bool()].tolist() == call["prompt_ids"]
            assert batch["responses"][index][batch["response_mask"][index].bool()].tolist() == call["response_ids"]
        assert int(batch["response_mask"].sum()) == training["response_tokens"]
        assert training["metrics"]["actor/sft_updates"] == (len(calls) + 3) // 4
        assert max(training["optimizer_steps_after"]) == max(previous["optimizer_steps"] or [0]) + int(training["metrics"]["actor/sft_updates"])
        optimizer = torch.load(phase / "checkpoint/optimizer.pt", map_location="cpu", weights_only=True)
        assert {value.dtype for state in optimizer["state"].values() for key, value in state.items()
                if key != "step" and torch.is_tensor(value)} == {torch.float32}
        del optimizer
        for path in (phase / "checkpoint").glob("*.safetensors"):
            with safe_open(path, framework="pt", device="cpu") as saved:
                assert all(saved.get_slice(name).get_dtype() == "F32" for name in saved.keys() if "visual" not in name)
        assert training["parameter_dtype"] == "float32"
        assert training["qwen35_padding_compatibility_fix"]["name"] == "qwen35-singleton-padding-mask-v1"
        assert len(mh["evaluation_files"]) == 1 + config["mh_iterations"] * config["mh_proposals"]
        mh_run = run / "mh" / f"round_{number:02d}"
        for index in range(1, config["mh_iterations"] * config["mh_proposals"] + 1):
            relative = Path("mh") / f"round_{number:02d}" / "harnesses" / f"h{index}" / "harness.py"
            assert sha(run / relative) == sha(reference / relative)
        for path in mh["evaluation_files"]:
            value = read(path)
            assert value["metadata"]["weight_digest"] == training["weight_digest"]
            assert sha(value["metadata"]["dataset_path"]) == config["data"]["splits"]["mh_val"]["sha256"]
            mh_rows = jsonl(Path(path).parent / "trajectories.jsonl")
            assert len(mh_rows) == len(keys["mh_val"]) and {r["task_id"] for r in mh_rows} == keys["mh_val"]
            recheck_rewards(mh_rows, config["data"]["splits"]["mh_val"], scope)
            name = Path(path).parents[1].name
            receipts = run  # Math MH receipts from all replicas are preserved at the root.
            check_calls(mh_rows, receipts, training["weight_digest"], value["metadata"]["harness_sha256"])
            stage_rows["harness_search"].extend(mh_rows)
        annotation_routes.update(label["route"] for label in plan["labels"])
        rounds.append({"round": number, "rollouts": len(rows), "accepted_trajectories": training["accepted_trajectories"],
            "response_tokens": training["response_tokens"], "updates": training["metrics"]["actor/sft_updates"],
            "teacher_cases": len(request["payload"]["cases"]), "routes": dict(Counter(label["route"] for label in plan["labels"]))})
        stage_rows["training"].extend(rows)
        previous, previous_harness = reload, mh["harness_sha256"]
    frozen = read(run / "frozen_comparison.json")
    assert frozen["gate"]["model_identity"]["weight_digest"] == previous["weight_digest"]
    assert frozen["gate"]["harness_sha256"] == previous_harness
    ref_summary = read(reference / "summary.json")
    assert frozen["whale"]["model_identity"]["weight_digest"] == ref_summary["final_identity"]["weight_digest"]
    assert frozen["whale"]["harness_sha256"] == sha(ref_summary["final_harness"])
    scores, cells = {}, {}
    test_ids = [info["task_id"] for info in pd.read_parquet(config["data"]["splits"]["test"]["path"])["extra_info"]]
    for name in ["initial", "whale", "gate"]:
        value = read(run / "test" / name / "val.json")
        rows = jsonl(run / "test" / name / "trajectories.jsonl")
        assert len(rows) == len(keys["test"]) * draws and {r["task_id"] for r in rows} == keys["test"]
        expected_schedule = [[draw, index] for draw in range(draws) for index in range(len(test_ids))]
        assert [[r["draw"], r["example_index"]] for r in rows] == expected_schedule
        assert value["metadata"]["sampling_schedule"] == expected_schedule
        assert value["metadata"]["temperature"] == temperature and value["metadata"]["seed"] == config["seed"]
        for row in rows:
            assert row["task_id"] == test_ids[row["example_index"]]
            assert row["rollout_seed"] == config["seed"] * 100003 + row["draw"] * 1009 + row["example_index"]
        assert value["metadata"]["weight_digest"] == frozen[name]["model_identity"]["weight_digest"]
        assert value["metadata"]["harness_sha256"] == frozen[name]["harness_sha256"]
        receipts = run if name == "gate" else run / "frozen_eval_workers/test" / name
        check_calls(rows, receipts, value["metadata"]["weight_digest"], value["metadata"]["harness_sha256"])
        recheck_rewards(rows, config["data"]["splits"]["test"], scope)
        scores[name], cells[name] = repeated_outcomes(rows, test_ids, draws)
        assert scores[name]["successful_draws"] == value["solved_examples"] == int(sum(r["reward"] for r in rows))
        assert value["num_examples"] == len(rows)
        stage_rows["test"].extend(rows)
    ordered = sorted(keys["test"])
    contrasts = {"gate_minus_whale": paired_change([cells["whale"][k] for k in ordered], [cells["gate"][k] for k in ordered]),
        "whale_minus_initial": paired_change([cells["initial"][k] for k in ordered], [cells["whale"][k] for k in ordered])}
    for contrast in contrasts.values():
        contrast["question_means_increased"] = contrast.pop("new_successes")
        contrast["question_means_decreased"] = contrast.pop("regressions")
        contrast["bootstrap_unit"] = "question"
    costs = {name: cost(rows) for name, rows in stage_rows.items()}
    for stage in ["training", "harness_search"]:
        assert costs[stage]["trajectories"] == ref_audit["costs"][stage]["trajectories"]
    canonical = {c["call_id"] for rows in stage_rows.values() for row in rows for c in row["policy_calls"]}
    assert canonical == {p.stem for p in run.rglob("calls/*.json")}, "Unaccounted policy calls"
    result = {"passed_audit": True, "scores": scores, "contrasts": contrasts, "training": rounds,
        "costs": costs, "reference_training_costs": {k: ref_audit["costs"][k] for k in ["training", "harness_search"]},
        "same_candidate_sources_verified": True, "first_wave_prompt_and_response_tokens_identical": True,
        "all_verified_successes_retained": True, "fp32_master_and_moments": True,
        "scope": "controlled sampling-allocation gate, shared externally generated harness proposal pool",
        "external_teacher_token_usage_known": False, "routes": dict(annotation_routes)}
    result.update(evaluation_temperature=temperature, evaluation_rollouts=draws,
                  evaluation_metric=f"mean_at_{draws}" if draws > 1 else "accuracy")
    if shared_probe is not None:
        result["shared_initial_probe"] = shared_probe
        first = jsonl(run / "round_01/gate/initial/trajectories.jsonl")
        shared_cost = cost(first)
        result["physically_new_gate_training_costs"] = {
            key: value - shared_cost[key] for key, value in costs["training"].items()}
        assert all(value >= 0 for value in result["physically_new_gate_training_costs"].values())
        result["shared_probe_is_not_algorithmic_cost_saving"] = True
    write_json(run / "verification.json", result)
    lines = ["# ReTool Math 同采样预算 gate 对照", "", "审计通过。两组每轮 512 条训练轨迹，同一批 harness 候选、原生选择规则和 FP32 训练配置。", "",
             f"每道独立测试题固定采样 {draws} 次，温度 {temperature}；按题计算平均成功率和配对区间。", "",
             "| 方法 | 平均成功率 | 成功采样 / 总采样 |", "|---|---:|---:|"]
    lines += [f"| {name} | {s['mean_accuracy_percent']:.2f}% | {s['successful_draws']}/{s['total_draws']} |" for name, s in scores.items()]
    delta = contrasts["gate_minus_whale"]
    lines += ["", f"Gate 相对 WHALE：{delta['difference_pp']:+.2f} 个百分点；平均成功率提高 {delta['question_means_increased']} 题，降低 {delta['question_means_decreased']} 题。",
        f"95% 成对 bootstrap 区间：{delta['paired_bootstrap_95_percent_pp']}。", "",
        "| 方法 | 训练采样 | MH 采样 | SFT 成功轨迹 | SFT 响应 token | 更新次数 |", "|---|---:|---:|---:|---:|---:|"]
    for name, audit, training in [("WHALE", ref_audit["costs"], ref_audit["training"]), ("gate", costs, rounds)]:
        lines.append(f"| {name} | {audit['training']['trajectories']} | {audit['harness_search']['trajectories']} | {sum(r['accepted_trajectories'] for r in training)} | {sum(r['response_tokens'] for r in training)} | {sum(r['updates'] for r in training):g} |")
    if shared_probe is not None:
        lines += ["", f"首批 {shared_probe['shared_trajectories']} 条 TRAIN 是显式共享输入，计入两臂的逻辑样本预算；物理生成一次。缓存复用不算 gate 的算法收益。"]
    lines += ["", "本次仅检验采样分配 gate；未同时替换 harness 候选生成器。外部标签是路由预测，未证明是因果归因。",
        "单种子小规模实验；同轨迹数不等于同 token、计算或费用。完整输入/生成 token 计数见 verification.json，教师费用未计量。",
        "资格筛选和工程诊断成本独立保留，没有计入每臂的正式训练预算。测试结果不用于继续改候选或调 gate 权重。", ""]
    (run / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
