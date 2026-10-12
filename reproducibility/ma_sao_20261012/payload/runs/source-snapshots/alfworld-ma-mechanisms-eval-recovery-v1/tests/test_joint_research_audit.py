from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.joint_research_audit import research_gate


def gate(scores, interval=(0.005, 0.06)):
    points = [{"rounds": 48 * (i + 1), "accuracy": score} for i, score in enumerate(scores)]
    comparison = {
        "control_accuracy": 0.92, "candidate_accuracy": scores[-1],
        "paired_question_bootstrap_95pct": interval,
    }
    return research_gate(points, comparison)


def test_plateau_advantage_unlocks_only_exploratory_efficiency():
    result = gate([0.935, 0.95, 0.95, 0.945, 0.95])
    assert result["working_advantage_gate_passed"]
    assert result["next_phase"] == "freeze_candidate_and_design_matched_efficiency"
    assert "not confirmatory" in result["interpretation"]


@pytest.mark.parametrize("scores,interval", [
    ([0.935, 0.95], (0.005, 0.06)),
    ([0.935, 0.94, 0.945, 0.95, 0.955], (0.005, 0.06)),
    ([0.935, 0.94, 0.94, 0.92, 0.94], (0.005, 0.06)),
    ([0.935, 0.935, 0.935, 0.935, 0.935], (0.0, 0.06)),
    ([0.935, 0.935, 0.935, 0.935, 0.935], (-0.015, 0.05)),
])
def test_short_rising_unstable_or_uncertain_curve_stays_in_tuning(scores, interval):
    result = gate(scores, interval)
    assert not result["working_advantage_gate_passed"]
    assert result["next_phase"] == "diagnose_then_one_frozen_candidate"


def test_wrong_endpoint_comparison_is_an_error():
    with pytest.raises(ValueError, match="final endpoint"):
        research_gate([{"rounds": 48, "accuracy": 0.935}], {
            "control_accuracy": 0.92, "candidate_accuracy": 0.95,
            "paired_question_bootstrap_95pct": [0.005, 0.06],
        })


def test_invalid_interval_is_an_error():
    with pytest.raises(ValueError, match="confidence interval"):
        gate([0.935], (float("nan"), 0.06))


def test_matrix_startup_recovery_preserves_every_scientific_input(tmp_path):
    from sao_bridge.common import file_sha, write_json
    from sao_bridge.joint_research_audit import endpoint_matrix_config

    stage = tmp_path / "evaluation200"
    failed = stage / "attempt-01"
    original = {"run": str(failed), "solver_model": "fixed-checkpoint",
                "seed": 42, "task_ids": [1, 2], "harnesses": {"joint_harness": "fixed-harness"}}
    write_json(stage / "config-01.json", original)
    assert endpoint_matrix_config(stage) == original
    write_json(failed / "FAILED.json", {"error": "Resource/budget watchdog stopped experiment"})
    evidence = failed / "supervision/foreign_workload.json"
    write_json(evidence, {"processes": [{"pid": 123, "cmd": ["python", "-m", "pytest"]}]})
    current = {**original, "run": str(stage / "attempt-02")}
    write_json(stage / "config-02.json", current)
    receipt = {
        "reason": "cpu_test_gpu_context_before_rollout", "foreign_pid": 123,
        "original_config_sha256": file_sha(stage / "config-01.json"),
        "failure_sha256": file_sha(failed / "FAILED.json"),
        "foreign_workload_sha256": file_sha(evidence),
        "new_config_sha256": file_sha(stage / "config-02.json"),
    }
    write_json(stage / "INFRASTRUCTURE_RECOVERY.json", receipt)
    assert endpoint_matrix_config(stage) == current
    write_json(stage / "config-02.json", {**current, "seed": 43})
    receipt["new_config_sha256"] = file_sha(stage / "config-02.json")
    write_json(stage / "INFRASTRUCTURE_RECOVERY.json", receipt)
    with pytest.raises(ValueError, match="changed an input"):
        endpoint_matrix_config(stage)
    write_json(stage / "config-02.json", current)
    receipt["new_config_sha256"] = file_sha(stage / "config-02.json")
    write_json(stage / "INFRASTRUCTURE_RECOVERY.json", receipt)
    write_json(failed / "reports/partial.json", {"reward": 0})
    with pytest.raises(ValueError, match="zero-rollout"):
        endpoint_matrix_config(stage)


@pytest.fixture
def completed_curve(tmp_path, monkeypatch):
    from sao_bridge import joint_research_audit as module

    root, parent = tmp_path / "run", tmp_path / "parent"
    documents, matrices, training = {}, {}, {}
    comparison = {"control_accuracy": 0.92, "candidate_accuracy": 0.95,
                  "paired_question_bootstrap_95pct": [0.005, 0.06]}
    points = [{"rounds": 48, "accuracy": 0.935}]
    documents[parent / "matched_joint_harness_report.json"] = {"accuracy": 0.935}
    documents[parent / "matched_COMPLETE.json"] = {"selection": {"harness": "incumbent"}}
    documents[Path(module.EVAL_PROTOCOL)] = {"all_task_ids": [0]}
    documents[module.ROOT / "runs/alfworld_core_200_oneshot_v2/merged/w1_h2.json"] = {}
    documents[root / "DESIGN.json"] = {
        "parent": str(parent), "transport_recovery": {"failed_cost": str(tmp_path / "failed.json")},
    }
    documents[tmp_path / "failed.json"] = {"consumed_total": 400}
    for block_index in (1, 2):
        block = root / f"block-{block_index:02d}"
        documents[block / "protocol.json"] = {"incumbent_min_gain": 2 / 48}
        training[block] = {}
        for role in ("editor", "solver"):
            documents[block / f"{role}_spec.json"] = {
                "role": role, "schedule": "joint_joint_anchor", "steps": 96,
                "critic_warmup_steps": 0, "critic_updates_per_step": 2,
            }
            training[block][role] = {
                "consumed_policy_trajectories": 512 if role == "editor" else 768,
                "generated_policy_trajectories": 768,
            }
            optimizer = block / role / "optimizer"
            documents[optimizer / "OPTIMIZER_LOAD.json"] = {
                "mode": "initial_fresh_adam" if block_index == 1 else "native_dcp_with_adam",
            }
            documents[optimizer / "OPTIMIZER_SAVED.json"] = {"paths": {
                component: str(optimizer / component) for component in ("actor", "critic")
            }}
            if block_index > 1:
                inputs = documents[root / f"block-{block_index - 1:02d}" / role
                                   / "optimizer/OPTIMIZER_SAVED.json"]["paths"]
                documents[block / f"{role}_spec.json"]["optimizer_state_input"] = inputs
                documents[optimizer / "OPTIMIZER_LOAD.json"]["paths"] = inputs
            for component in ("actor", "critic"):
                (optimizer / component).mkdir(parents=True)
                steps = block_index * (64 if role == "editor" else 96) * (2 if component == "critic" else 1)
                for rank in range(4):
                    path = optimizer / component / f"adam-rank{rank}.json"
                    path.touch()
                    documents[path] = {"min_adam_step": steps, "max_adam_step": steps}
        documents[block / "TRAINING_AUDIT.json"] = {"roles": training[block]}
        for rounds in (48, 96):
            label = f"round{rounds:04d}"
            selected = {"rounds": rounds, "global_step": rounds - 1, "patch_label": "incumbent_harness",
                        "harness": "incumbent", "pair": {"solver": {"actor": f"solver-{block_index}-{rounds}"}}}
            endpoint = {"selection": selected, "reports": {}, "scores": {}, "versus_w1_h2": comparison}
            endpoint_path = block / f"{label}_COMPLETE.json"
            documents[endpoint_path] = endpoint
            points.append({
                "rounds": 48 + (block_index - 1) * 96 + rounds, "accuracy": 0.95,
                "no_patch_accuracy": 0.9, "result": str(endpoint_path), "harness": "incumbent",
                "versus_historical_sao": comparison, "versus_parent": comparison,
            })
            for stage in (f"selection-{rounds - 1}", "evaluation200"):
                output = block / label / stage / "attempt-01"
                documents[output.parent / "config-01.json"] = {
                    "run": str(output), "solver_model": selected["pair"]["solver"]["actor"],
                    "seed": 6000045, "temperature": 1, "solver_tokens": 4096, "samples": 1,
                    "task_ids": [0], "harnesses": {
                        "joint_harness" if stage == "evaluation200" else "incumbent_harness": "incumbent",
                    },
                }
                matrices[output] = {"unique_harnesses": 2, "question_count": 200 if stage == "evaluation200" else 48}
                if stage == "evaluation200":
                    matrices[output]["variants"] = {}
                    for variant, accuracy in (("joint_harness", 0.95), ("no_patch", 0.9)):
                        compact = block / f"{label}_{variant}_report.json"
                        documents[compact] = {"accuracy": accuracy, "results": [{"reward": 1}]}
                        matrices[output]["variants"][variant] = {"report": str(compact)}
                        endpoint["reports"][variant] = {"sha256": "digest"}
                        endpoint["scores"][variant] = {"all200": {"accuracy": accuracy}}
    documents[root / "RESULT.json"] = {
        "status": "completed", "stop_reason": "operational_development_plateau", "points": points,
        "convergence": module.convergence_status(points), "new_consumed_role_trajectories": 2560,
        "new_generated_role_trajectories": 3072, "new_training_environment_episodes": 1536,
        "release": {"node3": {"foreign_gpu_pids": [], "filler_unit_active": True}},
    }
    documents[root / "CURVE.json"] = {"points": points}
    monkeypatch.setattr(module, "read_json", lambda path: documents[Path(path)])
    monkeypatch.setattr(module, "file_sha", lambda path: "digest")
    monkeypatch.setattr(module, "audit_training", lambda path: training[path])
    monkeypatch.setattr(module, "checked_matrix", lambda path, config: matrices[path])
    monkeypatch.setattr(module, "matrix_reports", lambda matrix, protocol: {})
    monkeypatch.setattr(module, "select_pair", lambda reports, **kwargs: (next(iter(reports)), "incumbent_harness"))
    monkeypatch.setattr(module, "checked_native_report", lambda report, tasks, config: report)
    monkeypatch.setattr(module, "paired_difference", lambda *args, **kwargs: comparison)
    return module, root, documents


def test_completed_audit_accounts_failure_and_all_fixed_endpoints(completed_curve):
    module, root, _ = completed_curve
    result = module.audit(root)
    assert result["research_gate"]["working_advantage_gate_passed"]
    assert len(result["endpoints"]) == 4
    assert result["successful_continuation_cost"]["evaluation_environment_episodes"] == 1984
    assert result["continuation_attempts_consumed_including_failure"] == 2960


@pytest.mark.parametrize("fault", ["reset", "missing_critic", "changed_incumbent", "cost"])
def test_completed_audit_rejects_broken_provenance(completed_curve, fault):
    module, root, documents = completed_curve
    optimizer = root / "block-02/editor/optimizer"
    if fault == "reset":
        documents[optimizer / "OPTIMIZER_LOAD.json"]["mode"] = "initial_fresh_adam"
    elif fault == "missing_critic":
        del documents[optimizer / "OPTIMIZER_SAVED.json"]["paths"]["critic"]
    elif fault == "changed_incumbent":
        documents[root / "block-02/round0048/selection-47/config-01.json"]["harnesses"]["incumbent_harness"] = "test-chosen"
    else:
        documents[root / "RESULT.json"]["new_consumed_role_trajectories"] += 1
    with pytest.raises(ValueError):
        module.audit(root)


def test_explicit_user_boundary_does_not_claim_convergence(completed_curve):
    module, root, documents = completed_curve
    result = documents[root / "RESULT.json"]
    result["points"][-1]["accuracy"] = 0.99
    result["convergence"] = module.convergence_status(result["points"])
    result["stop_reason"] = "user_requested_schedule_change_not_converged"
    documents[root / "REQUESTED_BOUNDARY.json"] = {"last_block": 3}
    result["user_requested_boundary"] = {"last_block": 3}
    with pytest.raises(ValueError, match="boundary"):
        module.audit(root)


def test_valid_user_boundary_is_reported_without_plateau(completed_curve, monkeypatch):
    module, root, documents = completed_curve
    result = documents[root / "RESULT.json"]
    state = {**result["convergence"], "operational_plateau": False}
    monkeypatch.setattr(module, "convergence_status", lambda points: state)
    result["convergence"] = state
    result["stop_reason"] = "user_requested_schedule_change_not_converged"
    request = {"last_block": 2}
    documents[root / "REQUESTED_BOUNDARY.json"] = request
    result["user_requested_boundary"] = request
    audited = module.audit(root)
    assert not audited["research_gate"]["working_advantage_gate_passed"]


@pytest.fixture
def peak_resume_curve(completed_curve):
    module, root, documents = completed_curve
    design = documents[root / "DESIGN.json"]
    parent = Path(design["parent"])
    inherited = {"editor": {"actor": 64, "critic": 128}, "solver": {"actor": 96, "critic": 192}}
    design.update(initial_rounds=144, parent_endpoint="round0096", inherited_optimizer_steps=inherited)
    for suffix in ("joint_harness_report.json", "COMPLETE.json"):
        documents[parent / f"round0096_{suffix}"] = documents.pop(parent / f"matched_{suffix}")
    points = documents[root / "RESULT.json"]["points"]
    for point in points:
        point["rounds"] += 96
    points[0]["accuracy"] = 0.97
    documents[parent / "round0096_joint_harness_report.json"]["accuracy"] = 0.97
    documents[root / "RESULT.json"]["convergence"] = module.convergence_status(points)
    documents[root / "RESULT.json"]["new_consumed_role_trajectories"] = 3072
    for role in ("editor", "solver"):
        native = parent / role / "optimizer"
        inputs = {component: str(native / component) for component in ("actor", "critic")}
        documents[native / "OPTIMIZER_SAVED.json"] = {"paths": inputs}
        first = root / "block-01"
        documents[first / f"{role}_spec.json"]["optimizer_state_input"] = inputs
        documents[first / role / "optimizer/OPTIMIZER_LOAD.json"] = {
            "mode": "native_dcp_with_adam", "paths": inputs,
        }
        for component, path in inputs.items():
            Path(path).mkdir(parents=True)
            for rank in range(4):
                fingerprint = Path(path) / f"adam-rank{rank}.json"
                fingerprint.touch()
                steps = inherited[role][component]
                documents[fingerprint] = {"min_adam_step": steps, "max_adam_step": steps}
        for block_index in (1, 2):
            block = root / f"block-{block_index:02d}"
            spec = documents[block / f"{role}_spec.json"]
            spec.update(schedule="simultaneous", inherited_optimizer_steps=inherited[role])
            documents[block / "TRAINING_AUDIT.json"]["roles"][role]["consumed_policy_trajectories"] = 768
            for component in ("actor", "critic"):
                steps = inherited[role][component] + block_index * 96 * (2 if component == "critic" else 1)
                for rank in range(4):
                    path = block / role / "optimizer" / component / f"adam-rank{rank}.json"
                    documents[path] = {"min_adam_step": steps, "max_adam_step": steps}
    return module, root, documents


def test_native_peak_resume_keeps_actor_and_critic_offsets(peak_resume_curve):
    module, root, documents = peak_resume_curve
    inherited = documents[root / "DESIGN.json"]["inherited_optimizer_steps"]
    first = module.audit_optimizer(root / "block-01", 1, inherited_steps=inherited)
    assert first["editor"]["cumulative_actor_updates"] == 160
    assert first["solver"]["cumulative_actor_updates"] == 192
    assert module.audit(root)["status"] == "passed"


@pytest.mark.parametrize("fault", ["reset", "offset", "loaded_source", "rank", "input_step", "parent_source"])
def test_native_peak_resume_rejects_wrong_lineage(peak_resume_curve, fault):
    module, root, documents = peak_resume_curve
    spec = documents[root / "block-01/editor_spec.json"]
    loaded = documents[root / "block-01/editor/optimizer/OPTIMIZER_LOAD.json"]
    if fault == "reset":
        loaded["mode"] = "initial_fresh_adam"
    elif fault == "offset":
        spec["inherited_optimizer_steps"] = {"actor": 0, "critic": 0}
    elif fault == "loaded_source":
        loaded["paths"] = {"actor": "/wrong", "critic": "/wrong"}
    elif fault == "rank":
        (root / "block-01/editor/optimizer/actor/adam-rank3.json").unlink()
    elif fault == "input_step":
        documents[Path(spec["optimizer_state_input"]["actor"]) / "adam-rank0.json"]["max_adam_step"] = 63
    else:
        parent = Path(documents[root / "DESIGN.json"]["parent"])
        documents[parent / "editor/optimizer/OPTIMIZER_SAVED.json"] = {
            "paths": {"actor": "/unselected", "critic": "/unselected"},
        }
    with pytest.raises(ValueError):
        module.audit(root)
