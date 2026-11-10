#!/usr/bin/env python3
"""Verify archived JoMoHa/MA-SAO results without GPUs, model weights, or APIs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import shutil
import sys

from export_ma_handoff import payload_path, read_json, sha256, verify_git_payload

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "whale_local"))

from sao_bridge.joint_anchor import build_rows

DEFAULT_BUNDLE = ROOT / "reproducibility/ma_sao_20261012"
BEST97_RUN = Path("runs/alfworld_ma_convergence_v1r4/block-01")


class Evidence:
    def __init__(self, bundle: Path):
        self.bundle = bundle
        self.manifest = read_json(bundle / "MANIFEST.json")
        self.source = Path(self.manifest["source_project"])
        self.files = {entry["path"] for entry in self.manifest["files"]}
        self.state = read_json(bundle / "STATE.json")
        if not self.source.is_absolute() or self.state["schema"] != "ma-sao-progress-handoff-v1":
            raise ValueError("Invalid handoff source or state schema")

    def path(self, name: str | Path) -> Path:
        relative = Path(name)
        if relative.is_absolute():
            relative = relative.relative_to(self.source)
        if relative.as_posix() not in self.files:
            raise FileNotFoundError(f"Required evidence is not included: {relative}")
        return payload_path(self.bundle / "payload", relative.as_posix())

    def read(self, name: str | Path):
        return read_json(self.path(name))


def unique_ids(values: list[int], *, count: int, name: str) -> set[int]:
    if (
        len(values) != count or len(set(values)) != count
        or any(type(value) is not int or not 0 <= value < 500 for value in values)
    ):
        raise ValueError(f"Wrong or duplicated task IDs in {name}")
    return set(values)


def split_audit(evidence: Evidence, protocol: dict) -> dict:
    study = Path(evidence.state["study"])
    data = evidence.read(study / "joint_data.json")
    source = evidence.read(study / "schedule_source_data.json")
    catalogue = evidence.read(evidence.state["catalogue"])
    paths = [path for group in catalogue.values() for path in group]
    if len(paths) != 500 or len(set(paths)) != 500:
        raise ValueError("The official task catalogue must have500 distinct ordered paths")
    core = evidence.read(protocol["core200_protocol"])
    selection = evidence.read(protocol["evaluation"]["task_manifest"])
    train = set(range(240))
    main = unique_ids(core["all_task_ids"], count=200, name="main200")
    strict = unique_ids(core["strict_goal_disjoint_task_ids"], count=197, name="strict197")
    chosen = unique_ids(selection["selection_task_ids"], count=48, name="selection48")
    remainder = unique_ids(selection["report_task_ids"], count=152, name="remaining152")
    if (
        train & (main | chosen) or chosen & remainder or chosen | remainder != main
        or not strict < main or set(core["training_task_ids_excluded"]) != train
    ):
        raise ValueError("Frozen training/selection/main partition relationships changed")
    related = set(core["untrained_instance_but_train_goal_related_task_ids"])
    if main - strict != related:
        raise ValueError("The three goal-related task exclusions changed")
    goals = [Path(path).parts[2] for path in paths]
    train_goals = {goals[index] for index in train}
    actual_related = {index for index in main if goals[index] in train_goals}
    if actual_related != related:
        raise ValueError("The catalogue does not reproduce the recorded goal overlap")
    if protocol["joint_rounds"] != 144 or protocol["batch_size"] != 8:
        raise ValueError("This bundle describes the fixed144-round, batch8 experiment")
    if build_rows(source, cycles=48, batch_size=8) != data:
        raise ValueError("The joint/joint/anchor input ledger cannot be reconstructed exactly")
    order = list(range(240))
    random.Random(1000042).shuffle(order)
    for index, row in enumerate(source):
        task = order[index % len(order)]
        if (
            row["task_id"] != task or row["id"] != f"h0w0-{index:05d}-task{task}"
            or row["task_context"] != paths[task]
        ):
            raise ValueError("The fixed source ordering or task-to-game mapping changed")
    return {
        "catalogue_tasks": 500, "training_tasks": 240, "training_source_rows": len(source),
        "shared_training_episodes": len(data), "selection_tasks": 48, "main_tasks": 200,
        "strict_goal_disjoint_tasks": 197,
        "train_main_task_overlap": 0, "train_selection_task_overlap": 0,
        "selection_main_overlap": len(chosen & main),
        "main_goal_related_to_training": sorted(related),
        "remaining_nonselection_tasks": 152,
        "selection48_is_part_of_main200": True,
        "interpretation": "Reused development evaluation, not an untouched held-out test.",
    }


def report_outcomes(report: dict, tasks: list[int]) -> dict[int, float]:
    rows = report["results"]
    actual = [(row["task_id"], row["sample_index"]) for row in rows]
    if (
        len(actual) != len(tasks) or set(actual) != {(task, 0) for task in tasks}
        or report["question_count"] != len(tasks) or report["samples_per_question"] != 1
        or report["metric"] != "mean@1"
        or any(type(row["reward"]) not in (int, float) or row["reward"] not in (0, 1) for row in rows)
    ):
        raise ValueError("Raw report lost its exact one-sample question/reward coverage")
    successes = int(sum(row["reward"] for row in rows))
    if not math.isclose(report["accuracy"], successes / len(tasks), rel_tol=0, abs_tol=1e-12):
        raise ValueError("Raw report mean differs from its recorded outcomes")
    return {row["task_id"]: row["reward"] for row in rows}


def report_scores(report: dict, tasks: list[int], strict: list[int], remaining: list[int]) -> dict:
    by_task = report_outcomes(report, tasks)
    successes = int(sum(by_task.values()))
    return {
        "successes": successes, "questions": len(tasks), "accuracy": successes / len(tasks),
        "strict197_successes": int(sum(by_task[task] for task in strict)),
        "nonselection152_successes": int(sum(by_task[task] for task in remaining)),
        "nonselection152_scope": "Post-hoc descriptive subset, not a new primary or untouched test.",
    }


def endpoint_audit(evidence: Evidence, protocol: dict) -> dict:
    result = evidence.read(Path(evidence.state["study"]) / "full/RESULT.json")
    core = evidence.read(protocol["core200_protocol"])
    selection_manifest = evidence.read(protocol["evaluation"]["task_manifest"])
    if result["status"] != "verified" or result["primary_endpoint"] != 144:
        raise ValueError("The fixed Full144 result is not a verified primary endpoint")
    endpoints = {}
    for rounds in ("96", "144"):
        endpoint = result["endpoints"][rounds]
        selection = endpoint["selection"]
        matrix = Path(endpoint["main_matrix"])
        config = evidence.read(matrix / "config.json")
        if (
            endpoint["rounds"] != int(rounds) or endpoint["arm"] != "full"
            or selection["main200_used_to_select"] is not False
            or selection["at_unix"] >= endpoint["main_started_at_unix"]
            or config["task_ids"] != core["all_task_ids"] or config["samples"] != 1
            or config["solver_model"] != selection["pair"]["solver"]["actor"]
            or config["harnesses"] != {"selected": selection["harness"]}
            or any(config[key] != core[key] for key in ("seed", "temperature", "solver_tokens"))
        ):
            raise ValueError("The selected Harness, Solver, endpoint timing, or matrix inputs changed")
        evidence.path(selection["harness"])
        scores = {}
        if set(endpoint["scores"]) != {"selected", "no_patch"}:
            raise ValueError("Both selected-Harness and weight-only endpoints are required")
        for name, recorded in endpoint["scores"].items():
            path = evidence.path(recorded["report"])
            if sha256(path) != recorded["sha256"]:
                raise ValueError("Endpoint report no longer matches the original result binding")
            score = report_scores(
                read_json(path), core["all_task_ids"], core["strict_goal_disjoint_task_ids"],
                selection_manifest["report_task_ids"],
            )
            if any(score[key] != recorded[key] for key in ("successes", "questions", "accuracy")):
                raise ValueError("Recomputed endpoint differs from its original result")
            scores[name] = score
        endpoints[rounds] = scores
    return endpoints


def best97_audit(evidence: Evidence) -> dict:
    protocol = evidence.read(Path(evidence.state["study"]) / "PROTOCOL.json")
    tasks = evidence.read(protocol["core200_protocol"])["all_task_ids"]
    unique_ids(tasks, count=200, name="best97 evaluation")
    harness = BEST97_RUN / "round0096/patches/95-proposal07.json"
    patch = evidence.read(harness)
    if patch["schema_version"] != "harness-r1-patch-v1" or patch["benchmark"] != "alfworld":
        raise ValueError("The best97 harness must use the recorded ALFWorld patch schema")
    original = evidence.read(
        BEST97_RUN / "round0096/evaluation200/attempt-01/reports"
        / "af8f607201d8a33fc24700422ee1b1169e830df854ff3022a63e0842d2dcdb76.json"
    )
    if original["domain"] != "alfworld":
        raise ValueError("The best97 original report must be an ALFWorld evaluation")
    original_outcomes = report_outcomes(original, tasks)
    scores = {}
    for name, suffix, expected in (("selected", "joint_harness", 194), ("no_patch", "no_patch", 185)):
        report_path = BEST97_RUN / f"round0096_{suffix}_report.json"
        report = evidence.read(report_path)
        if report["domain"] != "alfworld":
            raise ValueError("The best97 reports must be ALFWorld evaluations")
        outcomes = report_outcomes(report, tasks)
        if name == "selected" and outcomes != original_outcomes:
            raise ValueError("The best97 summary differs from the original per-task report")
        successes = int(sum(outcomes.values()))
        if successes != expected:
            raise ValueError(f"The best97 {name} report must contain {expected}/200 successes")
        scores[name] = {
            "successes": successes, "questions": len(tasks), "accuracy": successes / len(tasks),
            "report": report_path.as_posix(), "sha256": sha256(evidence.path(report_path)),
        }
    return {
        **scores, "harness": harness.as_posix(), "harness_sha256": sha256(evidence.path(harness)),
    }


def finite_metrics(value) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_metrics(child) for child in value.values())
    if isinstance(value, list):
        return all(finite_metrics(child) for child in value)
    return True


def native_metadata_audit(evidence: Evidence, arm: str) -> dict:
    root = Path(evidence.state["study"]) / arm / "recovery"
    pointer = evidence.read(root / "LATEST.json")
    generation = pointer["generation"]
    if Path(generation).name != generation:
        raise ValueError("Native generation must remain inside its recovery directory")
    committed_path = evidence.path(root / generation / "COMMITTED.json")
    committed = read_json(committed_path)
    if (
        sha256(committed_path) != pointer["sha256"]
        or committed["status"] != "committed"
        or committed["completed_rounds"] != pointer["completed_rounds"]
        or set(committed["roles"]) != {"editor", "solver"}
    ):
        raise ValueError("Paired native metadata no longer matches LATEST")
    for role in ("editor", "solver"):
        path = evidence.path(root / generation / f"{role}.json")
        record = read_json(path)
        if (
            sha256(path) != committed["roles"][role] or record["role"] != role
            or record["completed_rounds"] != committed["completed_rounds"]
            or record["committed_sample_cursor"] != committed["completed_rounds"] * record["batch_size"]
        ):
            raise ValueError("The paired checkpoint omitted a role or consistent data cursor")
    return {
        "completed_rounds": committed["completed_rounds"], "sha256": pointer["sha256"],
        "metadata_verified": True, "model_tensors_present_in_git": False,
        "scope": "Git verifies saved receipts only; native tensors require separate artifact download.",
    }


def audit(bundle: Path) -> dict:
    verified = verify_git_payload(bundle)
    evidence = Evidence(bundle)
    study = Path(evidence.state["study"])
    protocol = evidence.read(study / "PROTOCOL.json")
    paused = evidence.read(evidence.state["pause"])
    if paused["status"] != "cancelled" or paused["h200_training_allowed"] is not False:
        raise ValueError("This handoff must retain the explicit experiment pause")
    progress = {}
    for role in ("editor", "solver"):
        path = evidence.path(Path(evidence.state["attempt"]) / "without_peer" / role / "optimizer/metrics.jsonl")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if [row["global_step"] for row in rows] != list(range(len(rows))) or not finite_metrics(rows):
            raise ValueError("Interrupted training metrics have missing steps or nonfinite values")
        progress[role] = len(rows)
    snapshots = {}
    for name in evidence.state["source_snapshots"]:
        manifest = evidence.read(Path(name) / "SOURCE_MANIFEST.json")
        if "runtime_files" in manifest:
            for path, digest in manifest["runtime_files"].items():
                if sha256(evidence.path(Path(name) / path)) != digest:
                    raise ValueError("A frozen runtime file changed in the handoff")
            snapshots[name] = len(manifest["runtime_files"])
    result = {
        "status": "passed", "payload": verified,
        "data_split": split_audit(evidence, protocol),
        "full": endpoint_audit(evidence, protocol),
        "without_peer_metric_rounds": progress,
        "native_metadata": {arm: native_metadata_audit(evidence, arm) for arm in ("full", "without_peer")},
        "verified_snapshot_runtime_files": snapshots,
        "training_allowed": False, "gpu_or_api_used": False,
        "interpretation": "Recomputed stored evidence; not a new training or model-evaluation run.",
    }
    expected = evidence.state["expected"]
    if (
        progress != expected["without_peer_metric_rounds"]
        or result["native_metadata"]["without_peer"]["completed_rounds"] != expected["without_peer_native_rounds"]
        or result["native_metadata"]["full"]["completed_rounds"] != 144
        or result["full"]["144"]["selected"]["successes"] != expected["full144_selected_successes"]
        or result["full"]["144"]["no_patch"]["successes"] != expected["full144_weight_only_successes"]
    ):
        raise ValueError("The handoff no longer describes its declared progress")
    return result


def materialize(bundle: Path, destination: Path) -> dict:
    if destination.resolve().is_relative_to(bundle.resolve()):
        raise ValueError("Materialization must not modify the reproducibility bundle")
    if destination.exists():
        raise FileExistsError("Materialize only into a new directory, never over a live experiment")
    verify_git_payload(bundle)
    manifest = read_json(bundle / "MANIFEST.json")
    destination.mkdir(parents=True)
    for entry in manifest["files"]:
        source = payload_path(bundle / "payload", entry["path"])
        target = payload_path(destination, entry["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as incoming, target.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=2**20)
        if sha256(target) != entry["sha256"]:
            raise ValueError(f"Materialized evidence differs: {entry['path']}")
    return {
        "status": "materialized", "destination": str(destination), "files": len(manifest["files"]),
        "absolute_paths_rewritten": False, "model_tensors_included": False,
        "training_started": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--materialize", type=Path)
    parser.add_argument(
        "--best97", action="store_true",
        help="Also verify the best97 harness and its 194/200 ALFWorld result.",
    )
    args = parser.parse_args()
    result = audit(args.bundle)
    if args.best97:
        result["best97"] = best97_audit(Evidence(args.bundle))
    if args.materialize is not None:
        result["materialized"] = materialize(args.bundle, args.materialize)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
