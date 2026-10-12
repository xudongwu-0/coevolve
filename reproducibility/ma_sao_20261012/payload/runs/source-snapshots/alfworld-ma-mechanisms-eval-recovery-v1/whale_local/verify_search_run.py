"""Audit actual SearchQA RSFT targets and independently assess both axes."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

import pandas as pd
import torch
from safetensors import safe_open

from assess_viability import paired_change
from pipeline import sha, write_json


def native_em_scope():
    source = Path(__file__).resolve().parents[1] / "WHALE/domains/search_qa/environments/search_r1/env.py"
    tree = ast.parse(source.read_text())
    names = {"_verl_normalize", "_verl_em_check", "_inner_most_extract"}
    scope = {"re": re, "_string": __import__("string")}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
        and n.name in names], type_ignores=[]), str(source), "exec"), scope)
    return scope


def recheck_rewards(rows, entry, scope):
    metadata = pd.read_parquet(entry["path"]).to_dict("records")
    targets = {r["extra_info"]["task_id"]: r["reward_model"]["ground_truth"]["target"] for r in metadata}
    for row in rows:
        extracted = scope["_inner_most_extract"](row["last_assistant"])
        expected = float(extracted is not None and scope["_verl_em_check"](extracted, targets[row["task_id"]]))
        assert row["reward"] == expected, "Independent native EM mismatch"
        assert row["last_assistant"] == row["policy_calls"][-1]["parsed_message"]["content"]


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    config, summary = read(run / "config.json"), read(run / "summary.json")
    assert summary["status"] == "complete" and not config["gate_enabled"]
    prefix_path = run / "prefix_reuse.json"
    prefix = read(prefix_path) if prefix_path.exists() else None
    if prefix:
        parent = Path(prefix["parent_run"])
        assert sha(parent / "verification.json") == prefix["parent_audit_sha256"]
        parent_audit = read(parent / "verification.json")
        assert parent_audit["passed_audit"] and parent_audit["fp32_master_and_moments"]
        assert 0 < prefix["completed_rounds"] <= read(parent / "config.json")["rounds"]
        assert prefix["new_rollouts_in_prefix"] == 0
        if prefix.get("reuse_mode") == "completed_stopping_point":
            assert prefix["completed_rounds"] == config["rounds"]
            selection = prefix["development_selection"]
            assert sha(selection["path"]) == selection["sha256"]
            selected = read(selection["path"])
            assert selected["passed_audit"] and selected["selection_split"] == "mh_val"
            assert selected["selected_round"] == config["rounds"]
        else:
            assert prefix.get("reuse_mode") in {None, "continuation"}
            assert prefix["completed_rounds"] < config["rounds"]
        for relative, expected in prefix["original_relative_file_sha256"].items():
            assert sha(run / relative) == sha(parent / relative) == expected
        assert prefix["confirmation_from_parent_not_reused"]
        old_data = read(parent / "config.json")["data"]
        old_keys = {r["task_id"] for entry in old_data["splits"].values()
                    for r in pd.read_parquet(entry["path"])["extra_info"]}
        new_confirmation = {r["task_id"] for r in pd.read_parquet(config["data"]["splits"]["confirmation"]["path"])["extra_info"]}
        assert not old_keys.intersection(new_confirmation)
    reward_scope = native_em_scope()
    keys = {}
    for name, entry in config["data"]["splits"].items():
        assert sha(entry["path"]) == entry["sha256"]
        keys[name] = {r["task_id"] for r in pd.read_parquet(entry["path"])["extra_info"]}
        assert len(keys[name]) == entry["count"]
    assert len(set.union(*keys.values())) == sum(map(len, keys.values()))
    assert sha(config["data"]["corpora"]["path"]) == config["data"]["corpora"]["sha256"]
    training_tasks = [r["task_id"] for r in pd.read_parquet(config["data"]["splits"]["train"]["path"])["extra_info"]]
    previous = read(run / "initial_identity.json")
    previous_harness = None
    rounds = []
    precision_ok = True
    stage_rows = {"training": [], "harness_search": [], "confirmation": []}
    for n in range(1, config["rounds"] + 1):
        phase = run / f"round_{n:02d}"
        training = read(phase / "checkpoint/training_state.json")
        optimizer = torch.load(phase / "checkpoint/optimizer.pt", map_location="cpu", weights_only=True)
        moment_dtypes = sorted({str(value.dtype) for state in optimizer["state"].values()
                               for key, value in state.items() if key != "step" and torch.is_tensor(value)})
        parameter_dtypes = set()
        for path in (phase / "checkpoint").glob("*.safetensors"):
            with safe_open(path, framework="pt", device="cpu") as saved:
                parameter_dtypes.update(saved.get_slice(name).get_dtype() for name in saved.keys() if "visual" not in name)
        round_precision_ok = (training.get("parameter_dtype", "bfloat16") == "float32"
                              and moment_dtypes == ["torch.float32"] and parameter_dtypes == {"F32"})
        precision_ok = precision_ok and round_precision_ok
        del optimizer
        reload = read(phase / "reload_verification.json")
        mh = read(phase / "mh_result.json")
        assert training["sampling_identity"]["weight_digest"] == previous["weight_digest"]
        assert training["optimizer_steps_before"] == previous["optimizer_steps"]
        assert training["weight_digest"] == reload["weight_digest"] == mh["model_identity"]["weight_digest"]
        assert reload["all_parameter_hashes_equal"]
        assert training["changed_parameter_tensors"]
        assert max(training["optimizer_steps_after"]) == max(previous["optimizer_steps"] or [0]) + int(training["metrics"]["actor/sft_updates"])
        assert sha(training["harness"]) == training["harness_sha256"]
        if previous_harness:
            assert previous_harness == training["harness_sha256"]
        rows = jsonl(phase / "rollouts/trajectories.jsonl")
        stage_rows["training"].extend(rows)
        recheck_rewards(rows, config["data"]["splits"]["train"], reward_scope)
        assert len(rows) == len(keys["train"]) * config["rollouts_per_task"] == training["rollouts"]
        calls = []
        for position, row in enumerate(rows):
            draw, index = divmod(position, len(training_tasks))
            assert row["draw"] == draw and row["example_index"] == index
            assert row["task_id"] == training_tasks[index]
            assert row["rollout_seed"] == (config["seed"] + n) * 100003 + draw * 1009 + index
            assert row["task_id"] in keys["train"] and row["answer_redacted"]
            assert row["sampling_identity"]["weight_digest"] == previous["weight_digest"]
            assert row["harness_sha256"] == training["harness_sha256"]
            assert row["assistant_tokens"] == sum(len(c["response_ids"]) for c in row["policy_calls"]) <= config["assistant_token_budget"]
            for call in row["policy_calls"]:
                assert call == read(run / "calls" / (call["call_id"] + ".json"))
                assert call["stop_at_chat_end"] and len(call["response_ids"]) <= call["requested_max_tokens"] <= 256
                assert len(call["response_logprobs"]) == len(call["response_ids"])
                assert not any(t in (248044, 248046) for t in call["response_ids"][:-1]), "Generated past message end"
            if row["reward"]:
                calls.extend(row["policy_calls"])
        assert sum(r["reward"] for r in rows) == training["accepted_trajectories"]
        assert len(calls) == training["accepted_policy_calls"]
        batch = torch.load(phase / "accepted_batch.pt", map_location="cpu", weights_only=True)
        assert len(calls) == len(batch["responses"])
        rmax = batch["responses"].shape[1]
        pmax = batch["input_ids"].shape[1] - rmax
        for i, call in enumerate(calls):
            prompt = batch["input_ids"][i, :pmax][batch["attention_mask"][i, :pmax].bool()].tolist()
            response = batch["responses"][i][batch["response_mask"][i].bool()].tolist()
            assert prompt == call["prompt_ids"] and response == call["response_ids"]
        assert int(batch["response_mask"].sum()) == training["response_tokens"]
        assert len(mh["evaluation_files"]) == 1 + config["mh_iterations"] * config["mh_proposals"]
        for path in mh["evaluation_files"]:
            value = read(path)
            assert value["num_examples"] == len(keys["mh_val"])
            assert value["metadata"]["weight_digest"] == training["weight_digest"]
            assert sha(value["metadata"]["dataset_path"]) == config["data"]["splits"]["mh_val"]["sha256"]
            mh_rows = jsonl(Path(path).parent / "trajectories.jsonl")
            stage_rows["harness_search"].extend(mh_rows)
            recheck_rewards(mh_rows, config["data"]["splits"]["mh_val"], reward_scope)
            name = Path(path).parents[1].name
            receipt_root = run if name == "h0" else run / "eval_workers" / f"round_{n:02d}" / name
            for row in mh_rows:
                assert row["sampling_identity"]["weight_digest"] == training["weight_digest"]
                assert row["harness_sha256"] == value["metadata"]["harness_sha256"]
                for call in row["policy_calls"]:
                    assert call == read(receipt_root / "calls" / (call["call_id"] + ".json"))
        rounds.append({"round": n, "rollouts": len(rows), "accepted_trajectories": training["accepted_trajectories"],
                       "exact_sft_calls_verified": len(calls), "response_tokens": training["response_tokens"],
                       "updates": training["metrics"]["actor/sft_updates"], "fp32_master_and_moments": round_precision_ok,
                       "optimizer_moment_dtypes": moment_dtypes, "saved_language_parameter_dtypes": sorted(parameter_dtypes)})
        previous, previous_harness = reload, mh["harness_sha256"]
    cells, scores, identities = {}, {}, {}
    for name in ["W0_H0", "Wf_H0", "W0_Hf", "Wf_Hf"]:
        output = run / "confirmation" / name
        value, rows = read(output / "val.json"), jsonl(output / "trajectories.jsonl")
        stage_rows["confirmation"].extend(rows)
        recheck_rewards(rows, config["data"]["splits"]["confirmation"], reward_scope)
        assert {r["task_id"] for r in rows} == keys["confirmation"]
        assert len(rows) == len(keys["confirmation"])
        assert sum(r["reward"] for r in rows) == value["solved_examples"]
        cells[name] = {r["task_id"]: int(r["reward"]) for r in rows}
        scores[name] = {"solved": value["solved_examples"], "total": len(rows)}
        identities[name] = value["metadata"]
        worker = run if name == "Wf_Hf" else run / "frozen_eval_workers/confirmation" / name
        for row in rows:
            assert row["sampling_identity"]["weight_digest"] == value["metadata"]["weight_digest"]
            assert row["harness_sha256"] == value["metadata"]["harness_sha256"]
            assert row["assistant_tokens"] == sum(len(c["response_ids"]) for c in row["policy_calls"]) <= 512
            for call in row["policy_calls"]:
                assert call == read(worker / "calls" / (call["call_id"] + ".json"))
                assert len(call["response_ids"]) <= call["requested_max_tokens"] <= 256
                assert call["stop_at_chat_end"]
                assert not any(t in (248044, 248046) for t in call["response_ids"][:-1])
    initial = read(run / "initial_identity.json")
    first_harness = read(run / "round_01/checkpoint/training_state.json")["harness_sha256"]
    for name in cells:
        expected_weight = initial["weight_digest"] if name.startswith("W0") else previous["weight_digest"]
        expected_harness = first_harness if name.endswith("H0") else previous_harness
        assert identities[name]["weight_digest"] == expected_weight
        assert identities[name]["harness_sha256"] == expected_harness
    ordered = sorted(keys["confirmation"])
    values = {name: [cell[key] for key in ordered] for name, cell in cells.items()}
    contrasts = {"harness_at_final_weights": paired_change(values["Wf_H0"], values["Wf_Hf"]),
                 "weights_at_final_harness": paired_change(values["W0_Hf"], values["Wf_Hf"]),
                 "joint": paired_change(values["W0_H0"], values["Wf_Hf"])}
    score_threshold_passed = all(contrasts[k]["difference_pp"] >= config["minimum_conditional_gain_pp"]
                    for k in ["harness_at_final_weights", "weights_at_final_harness"])
    qualified = score_threshold_passed and precision_ok
    resume_path = run / "resume_amendment_20260915.json"
    recovery = read(resume_path) if resume_path.exists() else {}
    if recovery:
        rows = (run / "round_02/rollouts/trajectories.jsonl").read_bytes().splitlines(keepends=True)
        assert hashlib.sha256(b"".join(rows[:recovery["retained_rows"]])).hexdigest() == recovery["retained_prefix_sha256"]
        assert all(r["prompt_response_tokens_identical"] for r in read(run / "resume_replay_verification.json")["matched_calls"])
    result = {"passed_audit": True, "qualified_for_gate": qualified, "scores": scores,
              "contrasts": contrasts, "training": rounds, "independent_gate_test_not_evaluated": True}
    result["reused_completed_prefix"] = prefix
    result["native_em_independently_recomputed"] = True
    result["score_threshold_passed"] = score_threshold_passed
    result["fp32_master_and_moments"] = precision_ok
    result["incomplete_retry_generated_tokens"] = recovery.get("incomplete_generated_tokens", 0)
    costs = {}
    canonical_calls = set()
    for stage, rows in stage_rows.items():
        calls = [c for r in rows for c in r["policy_calls"]]
        assert not canonical_calls.intersection(c["call_id"] for c in calls)
        canonical_calls.update(c["call_id"] for c in calls)
        costs[stage] = {"trajectories": len(rows), "policy_calls": len(calls),
            "generated_tokens": sum(len(c["response_ids"]) for c in calls),
            "prompt_tokens": sum(len(c["prompt_ids"]) for c in calls)}
    physical_calls = {p.stem for p in run.rglob("calls/*.json")}
    assert physical_calls - canonical_calls == set(recovery.get("incomplete_call_ids", []))
    assert canonical_calls <= physical_calls
    result["costs"] = costs
    result["external_proposer_token_usage_known"] = False
    write_json(run / "verification.json", result)
    lines = ["# WHALE 小规模双轴确认", "", f"审计通过；双轴门槛：{'通过' if qualified else '未通过'}。", "",
             "| 组合 | 解题数 |", "|---|---:|"]
    lines += [f"| {name} | {s['solved']}/{s['total']} |" for name, s in scores.items()]
    lines += ["", "| 比较 | 净变化（百分点） | 95% 成对 bootstrap 区间 | 新成功 / 退步 |", "|---|---:|---|---:|"]
    for name, value in contrasts.items():
        low, high = value["paired_bootstrap_95_percent_pp"]
        lines.append(f"| {name} | {value['difference_pp']:+.2f} | [{low:+.2f}, {high:+.2f}] | {value['new_successes']} / {value['regressions']} |")
    lines += ["", "场景：HotpotQA 官方 distractor 上下文上的 bridge 子集，固定 BM25 检索与 WHALE 默认 EM 验证。",
              "Qwen3.5-2B，64 train / 32 MH / 128 confirmation；另留 128 test，尚未评测。",
              f"{config['rounds']} 轮在线 RSFT，每轮 {len(keys['train']) * config['rollouts_per_task']} 条采样、全部成功轨迹入训、LR={config['learning_rate']:g}；每轮原生 MH 搜索 {config['mh_iterations']}×{config['mh_proposals']} 个候选。",
              "", "| 阶段 | 轨迹数 | 模型调用数 | 生成 token | 输入 token |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {stage} | {c['trajectories']} | {c['policy_calls']} | {c['generated_tokens']} | {c['prompt_tokens']} |"
              for stage, c in costs.items()]
    lines += ["", f"中断恢复另产生 {result['incomplete_retry_generated_tokens']} 个未入训的生成 token，已单列核对。",
              "场景筛选与工程调试在独立目录保存；不属于上述正式训练/搜索预算。外部 proposer token 费用未计量。",
              "这是单种子、小规模 WHALE 实现与场景可行性确认，使用本地 Transformers/BM25 适配，",
              "不是论文全量 Wikipedia/分布式训练复现，尚不是 gate 改进证据。", ""]
    if not precision_ok:
        lines += ["数值实现诊断：本次使用 BF16 参数和 AdamW moments，未匹配上游 actor 默认的 FP32 主参数。",
                  "该问题在读取任何确认集分数前发现；此运行保留作诊断，禁止据此启动 gate。",
                  "小学习率更新可能被 BF16 舍入；见 PRECISION_AUDIT.json。需要精度修正后的独立 WHALE 验证。", ""]
    if prefix:
        lines += [f"前 {prefix['completed_rounds']} 轮完整复用已审计的训练/搜索前缀，逐文件校验一致；其轨迹计入本表总预算，并未重新生成。",
                  "父运行的确认结果仅作失败筛选记录；本次使用新的、不重叠的确认题，未复用父运行确认轨迹。", ""]
    (run / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
