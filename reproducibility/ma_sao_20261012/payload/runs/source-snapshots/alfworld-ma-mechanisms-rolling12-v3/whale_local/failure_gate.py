"""Budgeted failure routing using frozen-weight, common-seed interventions.

Labels describe evidence for the NEXT intervention, not identified root causes.
Neither hidden solutions nor failed completions are converted into SFT targets.
"""
from collections import Counter
from pathlib import Path
import hashlib
import json
import time

VERSION = "counterfactual-v1"


def solved(row):
    return row["reward"] > 0.5


def legal_wrong(row):
    events = [x for x in row.get("harness_trace", []) if x.get("actor") == "verifier"]
    return (row.get("stop_condition") == "wrong_move" and bool(events)
            and events[-1].get("info", {}).get("legal_by_board") is True
            and events[-1].get("info", {}).get("legal_by_harness") is True
            and not any(x.get("source", "").endswith("error")
                        for x in row.get("harness_trace", [])))


def response_signature(row):
    return [(c["prompt_ids"], c["response_ids"]) for c in row["policy_calls"]]


def decide(original, control=None, treatment=None):
    if solved(original):
        return {"label": "success", "route": "none", "evidence": "original_verifier_success"}
    if control is None or treatment is None:
        return {"label": "uncertain", "route": "uncertain", "evidence": "probe_budget_exhausted"}
    expected = original["sampling_identity"]["weight_digest"]
    assert original["task_key"] == control["task_key"] == treatment["task_key"]
    assert control["sampling_identity"]["weight_digest"] == expected
    assert treatment["sampling_identity"]["weight_digest"] == expected
    assert original["rollout_seed"] == control["rollout_seed"] == treatment["rollout_seed"]
    assert original["harness_sha256"] == control["harness_sha256"]
    if treatment["harness_sha256"] == control["harness_sha256"]:
        return {"label": "uncertain", "route": "uncertain", "evidence": "identical_harness_intervention"}
    if (original["reward"] != control["reward"]
            or original["stop_condition"] != control["stop_condition"]
            or response_signature(original) != response_signature(control)):
        return {"label": "uncertain", "route": "uncertain", "evidence": "control_replay_not_reproducible"}
    if original.get("stop_condition") in {"invalid_solution_line", "error"}:
        return {"label": "uncertain", "route": "uncertain", "evidence": "environment_or_data_error"}
    if solved(treatment):
        return {"label": "harness_recoverable", "route": "harness",
                "evidence": "same_weights_seed_harness_change_succeeds",
                "confidence_scope": "one_seed_one_candidate_not_global_causality"}
    if legal_wrong(control) and legal_wrong(treatment):
        return {"label": "weight_candidate", "route": "weight",
                "evidence": "legal_wrong_under_both_tested_harnesses",
                "confidence_scope": "one_candidate_failed_to_recover_not_proven_weight_cause"}
    return {"label": "uncertain", "route": "uncertain",
            "evidence": "format_legality_budget_or_mixed_failure_needs_more_diagnosis"}


def probe_indices(trajectories, budget):
    """Cover different training positions before a second failure on one position."""
    first, rest, seen = [], [], set()
    for i, row in enumerate(trajectories):
        if solved(row):
            continue
        if row["task_key"] in seen:
            rest.append(i)
        else:
            seen.add(row["task_key"])
            first.append(i)
    return (first + rest)[:budget]


def dump_lines(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def route_and_select(trajectories, *, replay, original_harness, probe_harness,
                     out, probe_budget=16, resamples=2, seed=42):
    """replay(task_key, harness_path, seed) invokes the REAL frozen policy/verifier."""
    started = time.monotonic()
    out = Path(out)
    labels = [dict(sample_index=r["sample_index"], task_key=r["task_key"], **decide(r))
              for r in trajectories]
    probes, supplemental = [], []
    with (out / "gate_probes.jsonl").open("w") as stream:
        for i in probe_indices(trajectories, probe_budget):
            original = trajectories[i]
            control = replay(original["task_key"], original_harness, original["rollout_seed"])
            treatment = replay(original["task_key"], probe_harness, original["rollout_seed"])
            decision = decide(original, control, treatment)
            labels[i].update(decision)
            probe = {"sample_index": original["sample_index"], "task_key": original["task_key"],
                     "control": control, "treatment": treatment, "decision": decision}
            probes.append(probe)
            stream.write(json.dumps(probe, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"gate sample={i} label={decision['label']}", flush=True)
    weight_keys = {x["task_key"] for x in labels if x["route"] == "weight"}
    harness_keys = {x["task_key"] for x in labels if x["route"] == "harness"}
    # If different sampled failures on one task disagree, abstain from its SFT.
    mixed_keys = weight_keys & harness_keys
    weight_keys -= mixed_keys
    selected = [r | {"sft_source": "original_success"} for r in trajectories
                if solved(r) and r["task_key"] in weight_keys]
    for offset, key in enumerate(sorted(weight_keys)):
        if any(r["task_key"] == key for r in selected):
            continue
        for attempt in range(resamples):
            sample = replay(key, original_harness, seed + 100000 + offset * resamples + attempt)
            sample.update(sft_source="targeted_resample", resample_index=len(supplemental))
            supplemental.append(sample)
            if solved(sample):
                selected.append(sample)
                break
    dump_lines(out / "gate_labels.jsonl", labels)
    dump_lines(out / "gate_resamples.jsonl", supplemental)
    dump_lines(out / "sft_sources.jsonl", selected)
    routed = []
    for label, original in zip(labels, trajectories):
        if label["route"] == "harness":
            routed.append({"label": label, "prompt": original["prompt"],
                           "completion": original["completion"],
                           "harness_trace": original["harness_trace"]})
    context = {"gate_version": VERSION, "scope": "training failures only; no hidden solutions",
               "probe_harness": str(probe_harness),
               "probe_harness_sha256": hashlib.sha256(Path(probe_harness).read_bytes()).hexdigest(),
               "failures": routed}
    (out / "gate_harness_context.json").write_text(json.dumps(context, indent=2, ensure_ascii=False) + "\n")
    extra = [r for p in probes for r in (p["control"], p["treatment"])] + supplemental
    report = {"version": VERSION, "labels": dict(Counter(x["label"] for x in labels)),
              "routes": dict(Counter(x["route"] for x in labels)),
              "probe_budget": probe_budget, "paired_probes": len(probes),
              "extra_rollouts": len(extra),
              "extra_generated_tokens": sum(len(c["response_ids"]) for r in extra for c in r["policy_calls"]),
              "weight_task_keys": sorted(weight_keys), "mixed_task_keys_excluded": sorted(mixed_keys),
              "selected_successful_trajectories": len(selected),
              "selected_policy_calls": sum(len(r["policy_calls"]) for r in selected),
              "harness_search_enabled": bool(routed), "elapsed_seconds": time.monotonic() - started,
              "all_diagnosis_weights_frozen": True, "causal_root_cause_claimed": False}
    (out / "gate_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return selected, report
