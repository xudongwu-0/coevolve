import copy
import importlib.util
import json
from pathlib import Path
import random
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
loader = importlib.util.spec_from_file_location("reproduce_ma", ROOT / "scripts/reproduce_ma_progress.py")
assert loader is not None and loader.loader is not None
reproduce = importlib.util.module_from_spec(loader)
loader.loader.exec_module(reproduce)
from export_ma_handoff import export_git_payload, sha256


@pytest.fixture
def bundle(tmp_path):
    project = tmp_path / "project"
    output = tmp_path / "bundle"
    study = Path("runs/study")
    attempt = study / "without_peer/attempts/attempt-0003"

    def save(name, value):
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    paths = [
        f"json_2.1.1/valid_seen/goal{index if index not in (240, 241, 242) else index - 240}/trial/game.tw-pddl"
        for index in range(500)
    ]
    for index in (240, 241, 242):
        paths[index] = paths[index].replace("/trial/", "/different_trial/")
    save("assets/catalogue.json", {"tasks": paths})
    order = list(range(240))
    random.Random(1000042).shuffle(order)
    source = [
        {"id": f"h0w0-{index:05d}-task{order[index % 240]}",
         "task_id": order[index % 240], "task_context": paths[order[index % 240]]}
        for index in range(768)
    ]
    save(study / "schedule_source_data.json", source)
    save(study / "joint_data.json", reproduce.build_rows(source, cycles=48, batch_size=8))
    main, chosen = list(range(240, 440)), list(range(240, 288))
    core = {
        "all_task_ids": main, "strict_goal_disjoint_task_ids": main[3:],
        "training_task_ids_excluded": list(range(240)),
        "untrained_instance_but_train_goal_related_task_ids": main[:3],
        "seed": 6000045, "temperature": 1.0, "solver_tokens": 4096,
    }
    save("runs/core.json", core)
    save("runs/selection.json", {"selection_task_ids": chosen, "report_task_ids": main[48:]})
    save(study / "PROTOCOL.json", {
        "joint_rounds": 144, "batch_size": 8, "core200_protocol": str(project / "runs/core.json"),
        "evaluation": {"task_manifest": str(project / "runs/selection.json")},
    })
    pause = study / "serial/CANCELLED.json"
    save(pause, {"status": "cancelled", "h200_training_allowed": False})
    endpoints = {}
    for rounds, selected_count, no_patch_count in ((96, 180, 178), (144, 186, 179)):
        endpoint = study / "full" / f"round{rounds}"
        harness = save(endpoint / "harness.json", {"code": "example"})
        matrix = endpoint / "matrix"
        solver = str(project / f"models/solver{rounds}")
        save(matrix / "config.json", {
            "task_ids": main, "samples": 1, "solver_model": solver,
            "harnesses": {"selected": str(harness)},
            **{key: core[key] for key in ("seed", "temperature", "solver_tokens")},
        })
        scores = {}
        for name, count in (("selected", selected_count), ("no_patch", no_patch_count)):
            path = save(matrix / f"{name}.json", {
                "question_count": 200, "samples_per_question": 1, "metric": "mean@1",
                "accuracy": count / 200, "results": [
                    {"task_id": task, "sample_index": 0, "reward": int(index < count)}
                    for index, task in enumerate(main)
                ],
            })
            scores[name] = {
                "report": str(path), "sha256": sha256(path), "successes": count,
                "questions": 200, "accuracy": count / 200,
            }
        endpoints[str(rounds)] = {
            "rounds": rounds, "arm": "full", "main_matrix": str(project / matrix),
            "selection": {
                "harness": str(harness), "pair": {"solver": {"actor": solver}},
                "main200_used_to_select": False, "at_unix": 1,
            },
            "main_started_at_unix": 2, "scores": scores,
        }
    save(study / "full/RESULT.json", {
        "status": "verified", "primary_endpoint": 144, "endpoints": endpoints,
    })
    for arm, rounds in (("full", 144), ("without_peer", 72)):
        recovery = study / arm / "recovery"
        generation = f"round-{rounds:04d}-attempt-0001"
        roles = {}
        for role in ("editor", "solver"):
            path = save(recovery / generation / f"{role}.json", {
                "role": role, "completed_rounds": rounds,
                "committed_sample_cursor": rounds * 8, "batch_size": 8,
            })
            roles[role] = sha256(path)
        path = save(recovery / generation / "COMMITTED.json", {
            "status": "committed", "completed_rounds": rounds, "roles": roles,
        })
        save(recovery / "LATEST.json", {
            "generation": generation, "completed_rounds": rounds, "sha256": sha256(path),
        })
    for role in ("editor", "solver"):
        path = project / attempt / "without_peer" / role / "optimizer/metrics.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("".join(json.dumps({"global_step": step, "metrics": {"loss": 0.2}}) + "\n"
                                for step in range(76)))
    best97 = reproduce.BEST97_RUN
    save(best97 / "round0096/patches/95-proposal07.json", {
        "schema_version": "harness-r1-patch-v1", "benchmark": "alfworld", "actions": [],
    })
    for suffix, successes in (("joint_harness", 194), ("no_patch", 185)):
        report = {
            "domain": "alfworld", "question_count": 200, "samples_per_question": 1,
            "metric": "mean@1", "accuracy": successes / 200,
            "results": [
                {"task_id": task, "sample_index": 0, "reward": int(index < successes)}
                for index, task in enumerate(main)
            ],
        }
        save(best97 / f"round0096_{suffix}_report.json", report)
        if suffix == "joint_harness":
            save(
                best97 / "round0096/evaluation200/attempt-01/reports"
                / "af8f607201d8a33fc24700422ee1b1169e830df854ff3022a63e0842d2dcdb76.json",
                report,
            )
    export_git_payload(output, [path for path in project.rglob("*") if path.is_file()], project=project)
    (output / "STATE.json").write_text(json.dumps({
        "schema": "ma-sao-progress-handoff-v1", "study": study.as_posix(),
        "attempt": attempt.as_posix(), "pause": pause.as_posix(), "catalogue": "assets/catalogue.json",
        "source_snapshots": [],
        "expected": {
            "without_peer_metric_rounds": {"editor": 76, "solver": 76}, "without_peer_native_rounds": 72,
            "full144_selected_successes": 186, "full144_weight_only_successes": 179,
        },
    }))
    return output


def test_offline_audit_reproduces_exact_scores_splits_and_recovery_boundary(bundle):
    result = reproduce.audit(bundle)
    assert result["full"]["96"]["selected"]["successes"] == 180
    assert result["full"]["144"]["selected"]["accuracy"] == 0.93
    assert result["full"]["144"]["no_patch"]["accuracy"] == 0.895
    assert result["data_split"]["selection_main_overlap"] == 48
    assert result["data_split"]["train_main_task_overlap"] == 0
    assert result["without_peer_metric_rounds"] == {"editor": 76, "solver": 76}
    assert result["native_metadata"]["without_peer"]["completed_rounds"] == 72
    assert result["training_allowed"] is False and result["gpu_or_api_used"] is False


def test_best97_recomputes_scores_and_binds_the_harness(bundle):
    evidence = reproduce.Evidence(bundle)
    result = reproduce.best97_audit(evidence)
    assert result["selected"]["successes"] == 194
    assert result["selected"]["questions"] == 200
    assert result["selected"]["accuracy"] == 0.97
    assert result["no_patch"]["successes"] == 185
    assert result["no_patch"]["accuracy"] == 0.925
    assert result["harness_sha256"] == sha256(evidence.path(result["harness"]))
    for name in ("selected", "no_patch"):
        assert result[name]["sha256"] == sha256(evidence.path(result[name]["report"]))


@pytest.mark.parametrize("change", ["domain", "outcomes", "coverage", "score"])
def test_best97_rejects_mismatched_reports(bundle, change):
    evidence = reproduce.Evidence(bundle)
    suffix = "no_patch" if change == "score" else "joint_harness"
    path = evidence.path(reproduce.BEST97_RUN / f"round0096_{suffix}_report.json")
    report = json.loads(path.read_text())
    if change == "domain":
        report["domain"] = "webshop"
    elif change == "outcomes":
        report["results"][0]["reward"] = 0
        report["results"][-1]["reward"] = 1
    elif change == "coverage":
        report["results"][-1]["task_id"] = report["results"][0]["task_id"]
    else:
        report["results"][0]["reward"] = 0
        report["accuracy"] = 184 / 200
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        reproduce.best97_audit(evidence)


def test_best97_requires_the_matching_alfworld_harness(bundle):
    evidence = reproduce.Evidence(bundle)
    path = evidence.path(reproduce.BEST97_RUN / "round0096/patches/95-proposal07.json")
    path.write_text(json.dumps({"schema_version": "harness-r1-patch-v1", "benchmark": "dbbench"}))
    with pytest.raises(ValueError, match="ALFWorld patch schema"):
        reproduce.best97_audit(evidence)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        reproduce.best97_audit(evidence)


def test_best97_cli_preserves_existing_audits(bundle, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["reproduce_ma_progress.py", "--bundle", str(bundle), "--best97"])
    reproduce.main()
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "passed"
    assert result["best97"]["selected"]["accuracy"] == 0.97
    assert result["full"]["144"]["selected"]["accuracy"] == 0.93
    assert result["training_allowed"] is False and result["gpu_or_api_used"] is False


def test_materialization_is_no_clobber_and_does_not_rewrite_original_paths(bundle, tmp_path):
    destination = tmp_path / "materialized"
    result = reproduce.materialize(bundle, destination)
    assert result["training_started"] is False
    assert result["absolute_paths_rewritten"] is False
    with pytest.raises(FileExistsError, match="new directory"):
        reproduce.materialize(bundle, destination)
    assert (destination / "runs/study/PROTOCOL.json").read_bytes() == (
        bundle / "payload/runs/study/PROTOCOL.json"
    ).read_bytes()
    with pytest.raises(ValueError, match="must not modify"):
        reproduce.materialize(bundle, bundle / "payload/new-output")


@pytest.mark.parametrize("path", ["/not-the-source/private.json", "../private.json", "absent.json"])
def test_evidence_never_reads_outside_the_declared_bundle(bundle, path):
    with pytest.raises((ValueError, FileNotFoundError)):
        reproduce.Evidence(bundle).read(path)


@pytest.mark.parametrize("change", ["duplicate", "nan", "bool_reward", "accuracy"])
def test_raw_score_recomputation_rejects_invalid_reports(bundle, change):
    evidence = reproduce.Evidence(bundle)
    endpoint = evidence.read("runs/study/full/RESULT.json")["endpoints"]["144"]
    report = copy.deepcopy(evidence.read(endpoint["scores"]["selected"]["report"]))
    if change == "duplicate":
        report["results"][-1] = report["results"][0]
    elif change == "nan":
        report["results"][0]["reward"] = float("nan")
    elif change == "bool_reward":
        report["results"][0]["reward"] = True
    else:
        report["accuracy"] = 1.0
    with pytest.raises(ValueError):
        reproduce.report_scores(report, list(range(240, 440)), list(range(243, 440)), list(range(288, 440)))


def test_wrong_declared_progress_fails_instead_of_reporting_success(bundle):
    state = json.loads((bundle / "STATE.json").read_text())
    state["expected"]["without_peer_native_rounds"] = 76
    (bundle / "STATE.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="declared progress"):
        reproduce.audit(bundle)
