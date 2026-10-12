"""Teacher routing with an exact small-model rollout budget, including audits.

Labels allocate the remaining samples. A seeded round-robin schedule spends 90%
on weight/both candidates and 10% on uncertain exploration when both exist.
Every planned sample runs, including after an earlier success. This deliberately
differs from teacher-v2's stop-after-success, lower-budget mode.
"""
from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import random
import time

from failure_gate import decide, dump_lines, solved
from teacher_gate import make_request, request_annotations, validate_response

VERSION = "teacher-v2-equal-budget"


def plan_routes(rows, request, response, *, seed, rollouts_per_puzzle, audit_pairs):
    assert make_request(rows) == request
    assert rollouts_per_puzzle >= 1 and 0 <= audit_pairs <= 8
    annotations = validate_response(request, response)
    labels = [{"sample_index": r["sample_index"], "task_key": r["task_key"],
               **annotations.get(r["sample_index"], {"label": "success", "route": "none", "label_source": "native_verifier"})}
              for r in rows]
    rng = random.Random(seed)
    success = [r["sample_index"] for r in rows if solved(r)]; rng.shuffle(success)
    replay = success[:math.ceil(len(success) * 0.25)]
    failures = sorted(annotations); rng.shuffle(failures)
    total = len(rows) * rollouts_per_puzzle
    audits = failures[:min(audit_pairs, (total - len(rows)) // 2)]
    remaining = total - len(rows) - 2 * len(audits)
    primary = [x["sample_index"] for x in labels if x["route"] in {"weight", "both"}]
    uncertain = [x["sample_index"] for x in labels if x["route"] == "uncertain"]
    fallback = [x["sample_index"] for x in labels if x["route"] not in {"harness", "scoring_ambiguity", "environment"}]
    for pool in (primary, uncertain, fallback):
        rng.shuffle(pool)
    # In a degenerate all-excluded batch, diagnostic-only calls still account
    # for the fixed budget. They never enter SFT or change labels.
    diagnostic = [r["sample_index"] for r in rows]; rng.shuffle(diagnostic)
    counters = Counter()
    schedule = []
    for j in range(remaining):
        if uncertain and (not primary or j % 10 == 9):
            pool, reason = uncertain, "uncertainty_exploration"
        elif primary:
            pool, reason = primary, "teacher_weight_candidate"
        elif fallback:
            pool, reason = fallback, "budget_replay"
        else:
            pool, reason = diagnostic, "diagnostic_only"
        i = pool[counters[reason] % len(pool)]; counters[reason] += 1
        schedule.append({"sample_index": i, "reason": reason,
                         "rollout_seed": seed * 1000000 + 100000 + j})
    assert len(rows) + 2 * len(audits) + len(schedule) == total
    return {"version": VERSION, "seed": seed, "rollouts_per_puzzle": rollouts_per_puzzle,
            "audit_pairs": audit_pairs, "total_small_model_rollouts": total,
            "labels": labels, "audit_indices": audits, "initial_success_replay_indices": replay,
            "followup_schedule": schedule, "success_replay_fraction": 0.25,
            "uncertain_exploration_fraction_when_primary_exists": 0.1,
            "stop_after_success": False}


def route_and_select(rows, *, replay, original_harness, probe_harness, out,
                     rollouts_per_puzzle, audit_pairs=2, seed=42):
    started = time.monotonic()
    request, response, usage = request_annotations(rows, out)
    plan = plan_routes(rows, request, response, seed=seed,
                       rollouts_per_puzzle=rollouts_per_puzzle, audit_pairs=audit_pairs)
    (out / "teacher_sampling_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    by_index = {r["sample_index"]: r for r in rows}
    probes, supplemental = [], []
    selected = [r | {"sft_source": "initial_success_replay"} for r in rows
                if r["sample_index"] in plan["initial_success_replay_indices"]]
    for i in plan["audit_indices"]:
        row = by_index[i]
        control = replay(row["task_key"], original_harness, row["rollout_seed"])
        treatment = replay(row["task_key"], probe_harness, row["rollout_seed"])
        probes.append({"sample_index": i, "task_key": row["task_key"], "control": control,
                       "treatment": treatment, "decision": decide(row, control, treatment)})
    for j, item in enumerate(plan["followup_schedule"]):
        row = by_index[item["sample_index"]]
        sample = replay(row["task_key"], original_harness, item["rollout_seed"])
        sample.update(sft_source=item["reason"], resample_index=j, initial_sample_index=row["sample_index"])
        supplemental.append(sample)
        if solved(sample) and item["reason"] != "diagnostic_only":
            selected.append(sample)
    for name, data in [("gate_labels.jsonl", plan["labels"]), ("gate_probes.jsonl", probes),
                       ("gate_resamples.jsonl", supplemental), ("sft_sources.jsonl", selected)]:
        dump_lines(out / name, data)
    routed = [{"label": label, "prompt": row["prompt"], "completion": row["completion"],
               "harness_trace": row["harness_trace"]} for label, row in zip(plan["labels"], rows)
              if label["route"] in {"harness", "both"}]
    context = {"gate_version": VERSION, "scope": "external TRAIN predictions; no hidden solutions",
               "probe_harness": str(probe_harness),
               "probe_harness_sha256": hashlib.sha256(Path(probe_harness).read_bytes()).hexdigest(), "failures": routed}
    (out / "gate_harness_context.json").write_text(json.dumps(context, indent=2, ensure_ascii=False) + "\n")
    extra = [r for p in probes for r in (p["control"], p["treatment"])] + supplemental
    report = {"version": VERSION, "labels": dict(Counter(x["label"] for x in plan["labels"])),
              "routes": dict(Counter(x["route"] for x in plan["labels"])), "probe_budget": audit_pairs,
              "paired_probes": len(probes), "extra_rollouts": len(extra),
              "extra_generated_tokens": sum(len(c["response_ids"]) for r in extra for c in r["policy_calls"]),
              "initial_rollouts": len(rows), "nominal_ungated_rollouts": plan["total_small_model_rollouts"],
              "actual_small_model_rollouts": len(rows) + len(extra),
              "selected_successful_trajectories": len(selected), "selected_policy_calls": sum(len(r["policy_calls"]) for r in selected),
              "harness_search_enabled": bool(routed), "all_diagnosis_weights_frozen": True,
              "causal_root_cause_claimed": False, "teacher": usage,
              "elapsed_seconds": time.monotonic() - started,
              "note": "Exact small-model rollout budget; generated tokens, SFT volume and teacher cost can differ"}
    assert report["actual_small_model_rollouts"] == plan["total_small_model_rollouts"]
    (out / "gate_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return selected, report


def verify_artifacts(out, rows):
    read = lambda n: json.loads((out / n).read_text())
    lines = lambda n: [json.loads(x) for x in (out / n).read_text().splitlines()]
    stored = read("teacher_sampling_plan.json")
    rebuilt = plan_routes(rows, read("teacher_request.json"), read("teacher_response.json"),
                          **{k: stored[k] for k in ("seed", "rollouts_per_puzzle", "audit_pairs")})
    assert rebuilt == stored and lines("gate_labels.jsonl") == rebuilt["labels"]
    by_index = {r["sample_index"]: r for r in rows}
    probes = lines("gate_probes.jsonl")
    assert [p["sample_index"] for p in probes] == rebuilt["audit_indices"]
    for p in probes:
        assert decide(by_index[p["sample_index"]], p["control"], p["treatment"]) == p["decision"]
    expected = [r | {"sft_source": "initial_success_replay"} for r in rows
                if r["sample_index"] in rebuilt["initial_success_replay_indices"]]
    supplemental = lines("gate_resamples.jsonl")
    assert len(supplemental) == len(rebuilt["followup_schedule"])
    for j, (r, item) in enumerate(zip(supplemental, rebuilt["followup_schedule"])):
        row = by_index[item["sample_index"]]
        assert r["initial_sample_index"] == row["sample_index"] and r["resample_index"] == j
        assert r["task_key"] == row["task_key"] and r["harness_sha256"] == row["harness_sha256"]
        assert r["sampling_identity"]["weight_digest"] == row["sampling_identity"]["weight_digest"]
        assert r["rollout_seed"] == item["rollout_seed"] and r["sft_source"] == item["reason"]
        assert r["answer_redacted"] and not r["answer"]
        if solved(r) and item["reason"] != "diagnostic_only":
            expected.append(r)
    assert expected == lines("sft_sources.jsonl")
    report = read("gate_summary.json")
    assert report["actual_small_model_rollouts"] == len(rows) + 2 * len(probes) + len(supplemental) == rebuilt["total_small_model_rollouts"]
    assert report["harness_search_enabled"] == any(x["route"] in {"harness", "both"} for x in rebuilt["labels"])
    return expected, {"teacher_request_bound_to_current_evidence": True, "sampling_plan_reconstructed": True,
                      "annotation_schema_and_quotes_verified": True, "only_native_successful_small_model_responses_used": True,
                      "paired_audit_interventions_recomputed": len(probes), "exact_total_rollout_budget_verified": True,
                      "actual_small_model_rollouts": report["actual_small_model_rollouts"], "teacher_usage_metered": False}
