"""Independently audit the completed alternating run and its actual SFT labels."""
from pathlib import Path
import argparse
import hashlib
import json
import pandas as pd
import torch


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    run = parser.parse_args().run_dir.resolve()
    config, summary = read(run / "config.json"), read(run / "summary.json")
    assert summary["status"] == "complete"
    positions = {}
    for name, entry in config["data"]["splits"].items():
        path = Path(entry["path"])
        assert sha(path) == entry["sha256"]
        positions[name] = {json.loads(r)["position_key"] for r in pd.read_parquet(path)["extra_info"]}
    assert not any(positions[a] & positions[b] for a, b in [("train", "mh_val"), ("train", "test"), ("mh_val", "test")])
    previous = read(run / "initial_identity.json")
    prev_harness = None
    checks = []
    for n in range(1, config["rounds"] + 1):
        phase = run / f"round_{n:02d}"
        training = read(phase / "checkpoint/training_state.json")
        reload = read(phase / "reload_verification.json")
        mh = read(phase / "mh_result.json")
        assert training["sampling_identity"]["weight_digest"] == previous["weight_digest"]
        assert training["optimizer_steps_before"] == previous["optimizer_steps"]
        updates = int(training["metrics"]["actor/sft_updates"])
        assert (training["weight_digest"] != previous["weight_digest"]) == bool(updates)
        assert reload["all_parameter_hashes_equal"]
        assert training["weight_digest"] == reload["weight_digest"] == mh["model_identity"]["weight_digest"]
        assert max(training["optimizer_steps_after"] or [0]) == max(previous["optimizer_steps"] or [0]) + updates
        if prev_harness:
            assert training["harness_sha256"] == prev_harness
        assert training["harness_sha256"] == sha(Path(training["harness"]))
        rollouts = [json.loads(s) for s in (phase / "train_rollouts.jsonl").read_text().splitlines()]
        assert len(rollouts) == training["rollouts"]
        sources = rollouts
        gate_checks = None
        if config.get("gate", {}).get("version", "").startswith("teacher-v2"):
            if config["gate"]["version"] == "teacher-v2-equal-budget":
                from teacher_budget_gate import verify_artifacts
            else:
                from teacher_gate import verify_artifacts
            sources, gate_checks = verify_artifacts(phase, rollouts)
            assert len(sources) == training["accepted_trajectories"]
            assert training["gate"]["harness_search_enabled"] != mh.get("skipped_by_gate", False)
        elif config.get("gate"):
            from failure_gate import decide
            labels = [json.loads(s) for s in (phase / "gate_labels.jsonl").read_text().splitlines()]
            probes = [json.loads(s) for s in (phase / "gate_probes.jsonl").read_text().splitlines()]
            assert len(labels) == len(rollouts)
            by_index = {r["sample_index"]: r for r in rollouts}
            for probe in probes:
                decision = decide(by_index[probe["sample_index"]], probe["control"], probe["treatment"])
                assert decision == probe["decision"]
                assert all(labels[probe["sample_index"]][k] == v for k, v in decision.items())
            g = read(phase / "gate_summary.json")
            assert len(probes) <= g["probe_budget"]
            sources = [json.loads(s) for s in (phase / "sft_sources.jsonl").read_text().splitlines()]
            supplemental = [json.loads(s) for s in (phase / "gate_resamples.jsonl").read_text().splitlines()]
            for source in sources:
                assert source["reward"] > 0.5 and source["task_key"] in g["weight_task_keys"]
                assert source["task_key"] not in g["mixed_task_keys_excluded"]
                if source["sft_source"] == "original_success":
                    assert source["policy_calls"] == by_index[source["sample_index"]]["policy_calls"]
                else:
                    assert source in supplemental
            assert len(sources) == training["accepted_trajectories"]
            assert g["harness_search_enabled"] != mh.get("skipped_by_gate", False)
            gate_checks = {"paired_interventions_recomputed": len(probes),
                           "only_verified_weight_routed_targets": True,
                           "unknown_failures_excluded": True,
                           "harness_search_follows_gate": True,
                           "labels": g["labels"], "extra_rollouts": g["extra_rollouts"]}
        calls = []
        for row in rollouts:
            assert row["sampling_identity"]["weight_digest"] == previous["weight_digest"]
            assert row["harness_sha256"] == training["harness_sha256"]
            assert row["answer_redacted"] and row["answer"] == ""
        for row in sources:
            assert row["sampling_identity"]["weight_digest"] == previous["weight_digest"]
            assert row["harness_sha256"] == training["harness_sha256"]
            if row["reward"] > 0.5:
                calls.extend(row["policy_calls"])
        if calls:
            batch = torch.load(phase / "accepted_batch.pt", map_location="cpu", weights_only=True)
        else:
            assert config.get("gate") and not updates and not (phase / "accepted_batch.pt").exists()
            batch = {"input_ids": torch.empty((0, 0)), "responses": torch.empty((0, 0))}
        assert len(calls) == batch["input_ids"].shape[0] == training["accepted_policy_calls"]
        rmax = batch["responses"].shape[1]
        pmax = batch["input_ids"].shape[1] - rmax
        for i, call in enumerate(calls):
            prompt = batch["input_ids"][i, :pmax][batch["attention_mask"][i, :pmax].bool()].tolist()
            response = batch["responses"][i][batch["response_mask"][i].bool()].tolist()
            assert prompt == call["prompt_ids"] and response == call["response_ids"]
            assert batch["input_ids"][i, pmax:pmax+len(response)].tolist() == response
            assert batch["response_mask"][i].sum().item() == len(call["response_ids"])
        for p in mh["evaluation_files"]:
            assert mh["model_identity"]["model"] in Path(p).parts
            evaluation = read(Path(p))
            assert evaluation["num_examples"] == evaluation["metadata"]["limit"] == config["data"]["splits"]["mh_val"]["count"]
        if mh.get("skipped_by_gate"):
            assert not mh["evaluation_files"] and mh["harness_sha256"] == training["harness_sha256"]
        else:
            assert len(mh["evaluation_files"]) >= 2
        checks.append({"round": n, "sampled_trajectories": len(rollouts),
            "accepted_calls_exact_tokens_verified": len(calls), "checkpoint_handoff": True,
            "optimizer_continuity": True, "input_harness_verified": True,
            "previous_round_harness_verified": True if prev_harness is not None else None,
            "mh_same_updated_model": True, "gate": gate_checks})
        previous = reload
        prev_harness = mh["harness_sha256"]
    final_id = read(run / "test_final/identity.json")
    assert final_id["weight_digest"] == previous["weight_digest"]
    assert final_id["harness_sha256"] == prev_harness
    initial_id = read(run / "initial_identity.json")
    baseline_harness = read(run / "round_01/checkpoint/training_state.json")["harness_sha256"]
    expected_tests = {
        "test_initial": (initial_id["weight_digest"], baseline_harness),
        "test_final": (previous["weight_digest"], prev_harness),
    }
    if config.get("heldout_ablation"):
        expected_tests.update(test_weights_only=(previous["weight_digest"], baseline_harness),
                              test_harness_only=(initial_id["weight_digest"], prev_harness))
    for name, (weight_digest, harness_digest) in expected_tests.items():
        model_id, evaluation = read(run / name / "identity.json"), read(run / name / "val.json")
        assert model_id["weight_digest"] == weight_digest and model_id["harness_sha256"] == harness_digest
        assert evaluation["num_examples"] == evaluation["metadata"]["limit"] == config["data"]["splits"]["test"]["count"]
        assert sha(Path(evaluation["metadata"]["dataset_path"])) == config["data"]["splits"]["test"]["sha256"]
        assert sha(Path(evaluation["metadata"]["harness_path"])) == harness_digest
    result = {"passed": True, "position_disjoint": True, "rounds": checks,
              "final_test_uses_final_pair": True, "heldout_pairs_verified": list(expected_tests),
              "stale_identity_rejected": read(run / "identity_guard.json")["stale_checkpoint_rejected"]}
    (run / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
