"""Compare the target-specific released Engineer with frozen, selected SAO proposals."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path
import sys
import threading
import time

from . import alfworld_engineer_eval as evaluation
from .alfworld_strong_arm import wait_stopped_unit
from .alfworld_train_data import _immutable_json
from .analysis import paired_difference
from .budget import contract_digest
from .common import ROOT, file_sha, read_json, write_json
from .gpu_telemetry import monitor, summarize
from .launch import Launcher
from .resources import gpu_release_audit, require_research_node


OFFICIAL_LABEL = "official_target_specific"


def validate_inputs(path: Path) -> tuple:
    request = read_json(path)
    if request["protocol"] != "alfworld-matched-official-engineer-v1":
        raise ValueError("Unknown matched official Engineer protocol")
    reference_spec = Path(request["reference_spec"])
    spec, manifest, prompts = evaluation.validate_spec(reference_spec)
    reference = Path(spec["run"])
    root = Path(request["run"]).resolve()
    if root in (ROOT, ROOT / "runs", reference.parent.resolve()) or root.is_relative_to(reference.resolve()):
        raise ValueError("Matched baseline must not overwrite its reference experiment")
    completion = read_json(reference / "COMPLETE.json")
    selected = completion.get("selected")
    if completion["status"] != "complete" or not completion["report_opened"] or selected is None:
        raise ValueError("Matched baseline needs an already selected and reported Engineer")
    label = selected["label"]
    if label != f"step{selected['global_step']}" or selected["global_step"] not in spec["checkpoint_steps"]:
        raise ValueError("Reference Engineer was not a predeclared checkpoint")
    files = {
        "protocol": reference_spec,
        "completion": reference / "COMPLETE.json",
        "identity": reference / "identity.json",
        "models": reference / "models.json",
        "proposals": reference / "proposals" / f"{label}.json",
    }
    reference_sha = {name: file_sha(item) for name, item in files.items()}
    if reference_sha != request["reference_sha256"]:
        raise ValueError("Frozen reference evidence changed")
    proposals = read_json(files["proposals"])
    if len(proposals) != 8 or len(prompts) != 8:
        raise ValueError("Matched comparison must retain all eight proposals")
    for index, (proposal, prompt) in enumerate(zip(proposals, prompts)):
        if (
            proposal["sample_index"] != index or proposal["prompt_id"] != prompt["id"]
            or proposal["decode_seed"] != spec["seed"] + 5_000_003 + index
            or proposal["model_label"] != label or type(proposal["valid"]) is not bool
            or proposal["patch_sha256"] != (
                contract_digest(proposal["patch"]) if proposal["valid"] else None
            )
            or (not proposal["valid"] and proposal["patch"] is not None)
        ):
            raise ValueError("Reference proposal slots, patches or decode seeds changed")
    model = Path(request["official_model"])
    provenance_path = model / "provenance.json"
    provenance = read_json(provenance_path)
    initial = read_json(Path(spec["initial_engineer"]) / "provenance.json")
    solver = read_json(Path(spec["solver_model"]) / "provenance.json")
    status_path = model.parent / "agent-sft-harness-r1_DOWNLOAD_STATUS.json"
    status = read_json(status_path)
    if (
        provenance["subfolder"] != "agent-sft-harness-r1"
        or solver["subfolder"] != "target-agent-sft"
        or provenance["model"] != initial["model"] or provenance["model"] != solver["model"]
        or provenance["revision"] != initial["revision"] or provenance["revision"] != solver["revision"]
        or status["status"] != "verified_complete"
        or status["receipt_sha256"] != file_sha(provenance_path)
        or Path(status["model_path"]).resolve() != model.resolve()
        or not provenance["files"]
        or not any(item["path"] == "model.safetensors" for item in provenance["files"])
        or any(
            not item["verified"] or (model / item["path"]).stat().st_size != item["size"]
            for item in provenance["files"]
        )
    ):
        raise ValueError("Official baseline is not the verified target-specific checkpoint")
    sources = evaluation.evaluation_sources()
    sources[str(Path(__file__).relative_to(ROOT))] = file_sha(Path(__file__))
    frozen = {
        "request": request, "reference_sha256": reference_sha,
        "official_provenance_sha256": file_sha(provenance_path),
        "download_status_sha256": file_sha(status_path),
        "sources": sources, "selected_reference": label,
        "reference_engineer": read_json(files["models"])[label],
        "report_task_ids": sorted(manifest["report_task_ids"]),
        "evaluation_prompts_sha256": file_sha(Path(spec["evaluation_prompts"])),
    }
    return request, spec, manifest, prompts, label, proposals, frozen


def require_report_release(root: Path) -> None:
    complete = read_json(root / "COMPLETE.json")
    release = read_json(root / "RELEASED.json")
    if complete["status"] != "completed" or release["foreign_gpu_pids"]:
        raise ValueError("Prior strong report did not finish and release its GPUs")


def run(path: Path) -> dict:
    request, spec, manifest, prompts, label, reference_proposals, frozen = validate_inputs(path)
    root = Path(request["run"])
    _immutable_json(root / "identity.json", frozen)
    if (root / "COMPLETE.json").exists():
        return read_json(root / "COMPLETE.json")
    evaluation.check_attempt_budget(root)
    for name in ("FAILED.json", "RELEASED.json"):
        previous = root / name
        if previous.exists():
            archive = root / "controller_history" / f"{time.time_ns()}-{name}"
            archive.parent.mkdir(exist_ok=True)
            previous.rename(archive)
    require_research_node()
    release = gpu_release_audit()
    if release["foreign_gpu_pids"] or release["filler_kind"] != "adaptive" or not release["filler_unit_active"]:
        raise RuntimeError("Matched baseline cannot claim GPUs from another workload")
    started = time.time()
    launch = Launcher(root / "supervision")
    stop = threading.Event()
    telemetry = threading.Thread(
        target=monitor, args=(root / "gpu_process_samples.jsonl", stop, 5.0), daemon=True,
    )
    telemetry.start()
    try:
        task_ids = sorted(manifest["report_task_ids"])
        proposals = {label: reference_proposals}
        evaluation._status(root, "baseline_and_official_generation")
        with launch.vllm(**evaluation._model_options(
            spec["solver_model"], port=19701, gpus="1,2,3,4,5,6,7",
            name="fixed_solver_7", replicas=7, context=32768,
        )) as solver_urls:
            urls = solver_urls.split(",")
            jobs = evaluation.Jobs(root / "jobs", launch, 15)
            baseline_names = []
            for index, url in enumerate(urls):
                chunk = task_ids[index::len(urls)]
                if not chunk:
                    continue
                name = f"baseline-{index}"
                baseline_names.append(name)
                jobs.add({
                    "name": name, "kind": "baseline", "task_ids": chunk,
                    "url": url, "concurrency": spec["target_rollout_concurrency"],
                    "port_index": index,
                })
            jobs.tick()
            names = [f"generate-{OFFICIAL_LABEL}-{index}" for index in range(8)]
            with ExitStack() as engineer:
                url = "http://127.0.0.1:19601/v1"
                if not all((root / "jobs" / name / "result.json").is_file() for name in names):
                    url = engineer.enter_context(launch.vllm(**evaluation._model_options(
                        request["official_model"], port=19601, gpus="0",
                        name=OFFICIAL_LABEL, replicas=1, context=16384,
                    )))
                for index, prompt in enumerate(prompts):
                    jobs.add({
                        "name": names[index], "kind": "generate", "model_label": OFFICIAL_LABEL,
                        "model": request["official_model"], "url": url,
                        "messages": prompt["messages"], "prompt_id": prompt["id"],
                        "sample_index": index, "decode_seed": spec["seed"] + 5_000_003 + index,
                        "temperature": spec["engineer_temperature"], "tokens": spec["engineer_tokens"],
                    })
                results = jobs.wait()
            proposals[OFFICIAL_LABEL] = [results[name] for name in names]
            for model_label, rows in proposals.items():
                _immutable_json(root / "proposals" / f"{model_label}.json", rows)
            baseline = {}
            for name in baseline_names:
                scores = {int(key): value for key, value in results[name]["scores"].items()}
                if baseline.keys() & scores.keys():
                    raise ValueError("Matched no-patch baseline cohorts overlap")
                baseline.update(scores)
            baseline = evaluation.normalize_rewards(baseline, task_ids)
            write_json(root / "baseline.json", baseline)
            with launch.vllm(**evaluation._model_options(
                spec["solver_model"], port=19801, gpus="0",
                name="fixed_solver_extra_1", replicas=1, context=32768,
            )) as extra_url:
                evaluation._status(root, "matched_report")
                reports = evaluation._score_models(
                    root, launch, spec, proposals, [OFFICIAL_LABEL, label],
                    task_ids, baseline, urls + extra_url.split(","), "report",
                )
        if validate_inputs(path)[-1] != frozen:
            raise ValueError("Matched baseline inputs or source changed before publication")
        difference = paired_difference(
            reports[OFFICIAL_LABEL], reports[label],
            seed=spec["seed"] + 9_000_003, draws=spec["bootstrap_draws"],
        )
        complete = {
            "status": "complete", "no_patch_accuracy": sum(baseline.values()) / len(task_ids),
            "system_accuracies": {key: value["accuracy"] for key, value in reports.items()},
            "mean_gains_vs_no_patch": {key: value["mean_gain_vs_no_patch"] for key, value in reports.items()},
            "sao_minus_official": difference,
            "valid_proposals": {key: value["valid_proposals"] for key, value in reports.items()},
            "reference_selected": label, "question_count": len(task_ids),
            "samples_per_question": 8, "fresh_engineer_proposals": 8,
            "reused_sao_proposals": 8, "sealed_test_opened": False,
            "compute_started_at_unix": started, "elapsed_compute_seconds": time.time() - started,
            "interpretation": (
                "Same Agent-SFT Solver and reused development report tasks; all eight proposal slots "
                "count. Frozen SAO proposals are re-evaluated alongside the target-specific official "
                "Engineer, with no new SAO selection. This is not the official 500-task score, "
                "a from-SFT SAO-versus-GRPO experiment, or a multi-seed result."
            ),
        }
        write_json(root / "COMPLETE.json", complete)
        evaluation._status(root, "complete")
        return complete
    except BaseException as exc:
        write_json(root / "FAILED.json", {"error": repr(exc), "failed_at_unix": time.time()})
        evaluation._status(root, "failed", error=repr(exc))
        raise
    finally:
        launch.close()
        stop.set()
        telemetry.join(timeout=10)
        samples = root / "gpu_process_samples.jsonl"
        if samples.exists():
            try:
                utilization = summarize(samples, start_time=started)
            except ValueError as exc:
                utilization = {"status": "invalid", "error": repr(exc)}
                print(f"GPU telemetry audit failed: {exc}", file=sys.stderr, flush=True)
            write_json(root / "gpu_utilization_summary.json", utilization)
        write_json(root / "RELEASED.json", gpu_release_audit())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--after-report-unit")
    parser.add_argument("--require-report-release", type=Path)
    args = parser.parse_args()
    if args.after_report_unit:
        wait_stopped_unit(args.after_report_unit)
    if args.require_report_release:
        require_report_release(args.require_report_release)
    try:
        print(run(args.spec), flush=True)
    except evaluation.EvaluationAttemptsExhausted as exc:
        print(str(exc), file=sys.stderr, flush=True)
        raise SystemExit(78) from exc


if __name__ == "__main__":
    main()
