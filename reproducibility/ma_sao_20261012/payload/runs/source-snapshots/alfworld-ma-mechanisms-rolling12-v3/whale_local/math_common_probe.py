"""Reuse an audited first-wave TRAIN probe as an explicitly shared input.

Equal logical sample budgets include these rows in both arms. They are generated
only once physically; reuse must not be presented as an algorithmic cost saving.
This module cannot start a gate or bypass baseline qualification.
"""
import json
from pathlib import Path
import shutil

from pipeline import sha, write_json


VERSION = "math-shared-first-probe-v1"


def read(path):
    return json.loads(Path(path).read_text())


def first_wave(reference, *, identity, harness, data_entry, seed):
    reference = Path(reference)
    config = read(reference / "config.json")
    assert config["data"]["splits"]["train"] == data_entry
    assert seed == config["seed"] + 1
    source = reference / "round_01/rollouts/trajectories.jsonl"
    lines = source.read_text().splitlines(keepends=True)
    rows = [json.loads(line) for line in lines[:data_entry["count"]]]
    assert len(rows) == data_entry["count"]
    assert [(r["draw"], r["example_index"]) for r in rows] == [
        (0, i) for i in range(data_entry["count"])]
    assert identity["weight_digest"] == read(reference / "initial_identity.json")["weight_digest"]
    assert identity["optimizer_steps"] == []
    for row in rows:
        assert row["answer_redacted"] and row["harness_sha256"] == sha(harness)
        assert row["sampling_identity"]["weight_digest"] == identity["weight_digest"]
        assert row["rollout_seed"] == seed * 100003 + row["example_index"]
        for call in row["policy_calls"]:
            assert call == read(reference / "calls" / (call["call_id"] + ".json"))
    return rows, "".join(lines[:data_entry["count"]])


def install(reference, *, run, output, identity, harness, data_entry, seed):
    reference, run, output = map(Path, (reference, run, output))
    qualification = read(reference / "verification.json")
    assert qualification["passed_audit"] and qualification["qualified_for_gate"]
    config = read(run / "config.json")
    assert config["qualification_sha256"] == sha(reference / "verification.json")
    assert config["gate_reference"] == str(reference.resolve())
    assert config["gate_first_probe_mode"] == VERSION
    rows, text = first_wave(reference, identity=identity, harness=harness,
                            data_entry=data_entry, seed=seed)
    copied = {}
    for row in rows:
        paths = [Path("calls") / (call["call_id"] + ".json") for call in row["policy_calls"]]
        if row.get("context_rejection_id"):
            paths.append(Path("context_rejections") / (row["context_rejection_id"] + ".json"))
        for relative in paths:
            source, target = reference / relative, run / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                assert target.read_bytes() == source.read_bytes()
            else:
                shutil.copy2(source, target)
            copied[str(relative)] = sha(source)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "trajectories.jsonl"
    if path.exists():
        assert path.read_text() == text
    else:
        path.write_text(text)
    ledger = {"version": VERSION, "reference": str(reference.resolve()),
        "reference_audit_sha256": sha(reference / "verification.json"),
        "source_trajectories_sha256": sha(reference / "round_01/rollouts/trajectories.jsonl"),
        "shared_trajectories_sha256": sha(path), "shared_trajectories": len(rows),
        "shared_policy_calls": sum(len(r["policy_calls"]) for r in rows),
        "shared_generated_tokens": sum(r["assistant_tokens"] for r in rows),
        "new_generations_during_import": 0, "counts_in_each_arm_sample_budget": True,
        "not_an_algorithmic_compute_saving": True, "receipt_sha256": copied}
    write_json(output / "SHARED_PROBE.json", ledger)
    write_json(output / "val.json", {"num_examples": len(rows),
        "solved_examples": int(sum(r["reward"] for r in rows)),
        "success_rate": sum(r["reward"] for r in rows) / len(rows),
        "shared_probe": ledger})
    return ledger
