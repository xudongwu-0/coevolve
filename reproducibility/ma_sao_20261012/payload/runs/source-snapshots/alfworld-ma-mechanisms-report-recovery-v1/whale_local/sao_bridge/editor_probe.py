"""Frozen-editor capability comparison with a fixed solver and no training."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from .analysis import paired_difference, paired_lower_bound
from .budget import initial_model_digest
from .common import file_sha, read_json, write_json
from .evaluation import Sampler, evaluate
from .experiment import propose
from .launch import Launcher
from .resources import require_research_node, stop_known
from .retool import harness_spec


def settings(source: str) -> dict:
    return {key: value for key, value in harness_spec(source).items() if key != "source"}


def configuration_key(source: str) -> str:
    return hashlib.sha256(json.dumps(settings(source), sort_keys=True).encode()).hexdigest()


def validate_origin(spec: dict) -> None:
    previous = Path(spec["frozen_run"])
    if spec.get("first_h1"):
        initial = read_json(previous / "spec.json")
        if (spec["solver"] != initial["model"]
                or Path(spec["incumbent"]).resolve() != (previous / "h0.py").resolve()):
            raise ValueError("First H1 must use the original SFT solver and initial harness")
        if len(spec["arms"]) != 1 or spec["proposals"] != 1:
            raise ValueError("First H1 requires exactly one proposal, not best-of-N")
        arm = spec["arms"][0]
        config = read_json(Path(arm["model"]) / "config.json")
        native_context = config.get("text_config", config)["max_position_embeddings"]
        if (arm["tokens"] is not None or not arm["thinking"]
                or arm["max_model_len"] != native_context):
            raise ValueError("Full H1 generation must use thinking, no short token cap, and native context")
    else:
        frozen = read_json(previous / "FROZEN.json")["stages"][-1]
        if spec["solver"] != frozen["solver"] or spec["incumbent"] != frozen["harness"]:
            raise ValueError("Probe must retain the completed MA solver and incumbent")


def require_complete_output(proposals: list) -> None:
    if len(proposals) != 1 or proposals[0]["finish_reason"] != "stop":
        raise ValueError("The first rewrite did not finish naturally; a truncated output cannot be evaluated")


def proposal_summary(proposals: list, incumbent: str) -> dict:
    initial = settings(incumbent)
    valid = [row for row in proposals if row["valid"]]
    keys = {configuration_key(row["source"]) for row in valid}
    changed = [row for row in valid if settings(row["source"]) != initial]
    return {
        "slots": len(proposals), "valid": len(valid),
        "distinct_valid_configurations": len(keys),
        "modified_slots": len(changed),
        "distinct_modified_configurations": len({configuration_key(row["source"]) for row in changed}),
        "length_limited_slots": sum(row.get("finish_reason") == "length" for row in proposals),
    }


def choose_candidate(proposals: list, outcomes: list, baseline: dict,
                     incumbent: str, *, seed: int, total_slots: int) -> dict:
    if len(proposals) != len(outcomes) or not proposals:
        raise ValueError("Every declared proposal slot needs a selection evaluation")
    seen = {configuration_key(incumbent)}
    candidates = []
    for slot, (proposal, outcome) in enumerate(zip(proposals, outcomes)):
        if not proposal["valid"]:
            continue
        key = configuration_key(proposal["source"])
        if key in seen:
            continue
        seen.add(key)
        candidates.append({
            "slot": slot, "source": proposal["source"], "configuration_sha256": key,
            "selection_accuracy": outcome["accuracy"],
            "original_rule_lcb": paired_lower_bound(
                baseline, outcome, alpha=0.05 / len(proposals), seed=seed + 31_337 + slot),
            "study_rule_lcb": paired_lower_bound(
                baseline, outcome, alpha=0.05 / total_slots, seed=seed + 31_337 + slot),
        })
    best = max(candidates, key=lambda row: row["selection_accuracy"]) if candidates else None
    return {
        "candidates": candidates, "best": best,
        "any_passes_original_gate": any(row["original_rule_lcb"] > 0 for row in candidates),
        "best_passes_study_gate": best is not None and best["study_rule_lcb"] > 0,
        "rule": "First slot per semantic configuration; highest selection accuracy among changed "
                "valid proposals. Report this frozen candidate even if the conservative gate fails. "
                "No automatic deployment or report-driven reselection.",
    }


def load_rows(spec: dict) -> tuple[list, list]:
    data = Path(spec["data"])
    manifest = read_json(data / "manifest.json")
    for split in ("train", "dev"):
        if file_sha(data / f"{split}.json") != manifest["splits"][split]["sha256"]:
            raise ValueError(f"Input data changed: {split}")
    dev, train = read_json(data / "dev.json"), read_json(data / "train.json")
    select = dev[:spec["select_questions"]]
    report = dev[spec["select_questions"]:spec["select_questions"] + spec["report_questions"]]
    report += train[-spec["report_train_tail"]:]
    select_ids, report_ids = ({row["id"] for row in rows} for rows in (select, report))
    if (len(select_ids) != spec["select_questions"]
            or len(report_ids) != spec["report_questions"] + spec["report_train_tail"]
            or select_ids & report_ids):
        raise ValueError("Selection/report questions must be complete, unique and disjoint")
    previous = Path(spec["frozen_run"])
    reference_ids = {row["id"] for row in read_json(previous / "h2_selected_report.json")["results"]}
    trained_ids = {row["id"] for row in read_json(previous / "w1_solver_data.json")
                   if not row.get("warmup")}
    recycled_ids = {row["question_id"] for row in read_json(previous / "w1_recycle.json")}
    if report_ids != reference_ids or report_ids & (trained_ids | recycled_ids):
        raise ValueError("Frozen-solver reporting split changed or overlaps its training data")
    return select, report


def reuse_completed_proposals(spec: dict, model_hashes: dict, root: Path) -> dict:
    if not spec.get("resume_from"):
        return {}
    previous = Path(spec["resume_from"])
    old_spec = read_json(previous / "spec.json")
    if (read_json(previous / "STATUS.json")["phase"] != "failed"
            or list(previous.glob("*_select.json")) or list(previous.glob("*_report.json"))):
        raise ValueError("Only a failed pre-selection probe may reuse completed proposal slots")
    def common(value):
        return {k: v for k, v in value.items() if k not in {"run", "resume_from", "arms"}}
    if common(old_spec) != common(spec) or read_json(previous / "initialization.json")["models"] != model_hashes:
        raise ValueError("Recovery must preserve prompts, budgets, data and all model digests")
    old_arms = {arm["name"]: arm for arm in old_spec["arms"]}
    if set(old_arms) != {arm["name"] for arm in spec["arms"]}:
        raise ValueError("Recovery cannot add or remove editor conditions")
    reused, receipts = {}, {}
    for arm in spec["arms"]:
        old = old_arms[arm["name"]]
        if {k: v for k, v in arm.items() if k != "server_args"} != {
                k: v for k, v in old.items() if k != "server_args"}:
            raise ValueError("Recovery may only change an unfinished editor's serving backend")
        path = previous / f"{arm['name']}_proposals.json"
        if not path.exists():
            continue
        if old != arm:
            raise ValueError("A completed editor condition cannot change on recovery")
        proposals = read_json(path)
        if (len(proposals) != spec["proposals"]
                or [row["id"] for row in proposals] != list(range(spec["proposals"]))):
            raise ValueError("Only a complete deterministic proposal slot set may be reused")
        for row in proposals:
            if row["valid"] and harness_spec(row["text"])["source"] != row["source"]:
                raise ValueError("Saved proposal source differs from its raw generated text")
        reused[arm["name"]] = proposals
        receipts[arm["name"]] = {"source": str(path), "sha256": file_sha(path), "slots": len(proposals)}
        write_json(root / path.name, proposals)
    write_json(root / "REUSED_PROPOSALS.json", {
        "previous_failed_run": str(previous), "proposals": receipts,
        "note": "Earlier startup failure and partial inference cost remain in the preserved failed run.",
    })
    return reused


async def evaluate_jobs(url: str, jobs: list, *, tokens: int, samples: int, seed: int) -> list:
    sampler = Sampler(url, concurrency=256)
    try:
        async def one(source, rows, output):
            result = await evaluate(sampler, rows, source, tokens=tokens, samples=samples, seed=seed)
            write_json(output, result)
            return result
        return await asyncio.gather(*(one(*job) for job in jobs))
    finally:
        await sampler.close()


def run_probe(path: Path):
    require_research_node()
    spec = read_json(path)
    root = Path(spec["run"])
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "spec.json", spec)
    start = time.monotonic()
    launcher = None
    allocated = False

    def status(phase, **extra):
        value = {"phase": phase, "updated_at": time.time(), "elapsed_seconds": time.monotonic() - start,
                 "optimizer_updates": 0, **extra}
        write_json(root / "STATUS.json", value)
        print(json.dumps(value), flush=True)

    try:
        status("validating")
        validate_origin(spec)
        arms = spec["arms"]
        if (not arms or len({arm["name"] for arm in arms}) != len(arms)
                or any(not arm["name"].replace("_", "").isalnum() for arm in arms)):
            raise ValueError("Probe arms require unique simple names")
        incumbent = harness_spec(Path(spec["incumbent"]).read_text())["source"]
        select_rows, report_rows = load_rows(spec)
        models = sorted({spec["solver"], *(arm["model"] for arm in arms)})
        model_hashes = {model: initial_model_digest(Path(model)) for model in models}
        proposals_by_arm = reuse_completed_proposals(spec, model_hashes, root)
        reused_slots = sum(len(rows) for rows in proposals_by_arm.values())
        for name, proposals in proposals_by_arm.items():
            write_json(root / f"{name}_diversity.json", proposal_summary(proposals, incumbent))
        write_json(root / "initialization.json", {
            "models": model_hashes, "protocol_sha256": file_sha(path),
            "incumbent_configuration": settings(incumbent),
            "model_provenance": read_json(Path(spec["new_model_provenance"])),
            "training": False, "report_split_previously_observed": True,
        })
        write_json(root / "source_manifest.json", {
            p.name: file_sha(p) for p in Path(__file__).parent.glob("*.py")})
        stop_known(require_exclusive=True)
        allocated = True
        launcher = Launcher(root / "services")
        pending = [arm for arm in arms if arm["name"] not in proposals_by_arm]
        for model in dict.fromkeys(arm["model"] for arm in pending):
            group = [arm for arm in pending if arm["model"] == model]
            if any(arm.get("server_args", []) != group[0].get("server_args", []) for arm in group):
                raise ValueError("Arms sharing a model must use the same server configuration")
            if any(arm.get("max_model_len", 8192) != group[0].get("max_model_len", 8192) for arm in group):
                raise ValueError("Arms sharing a model must use the same context limit")
            status("generating", arms=[arm["name"] for arm in group])
            with launcher.vllm(model, port=19803, gpus="0", name=group[0]["name"],
                               server_args=group[0].get("server_args"),
                               max_model_len=group[0].get("max_model_len", 8192)) as url:
                for arm in group:
                    proposals = asyncio.run(propose(
                        url, incumbent, spec["proposals"], arm["tokens"], spec["proposal_seed"],
                        temperature=spec["editor_temperature"],
                        chat_template_kwargs={"enable_thinking": arm["thinking"]},
                        request_timeout=arm.get("request_timeout", 900),
                    ))
                    proposals_by_arm[arm["name"]] = proposals
                    write_json(root / f"{arm['name']}_proposals.json", proposals)
                    write_json(root / f"{arm['name']}_diversity.json",
                               proposal_summary(proposals, incumbent))
        if spec.get("first_h1"):
            require_complete_output(proposals_by_arm[arms[0]["name"]])
        status("selecting")
        selections = {}
        with launcher.vllm(spec["solver"], port=19821, gpus="0,1,2,3,4,5,6,7",
                           data_parallel=8, name="frozen_solver") as url:
            jobs = [(incumbent, select_rows, root / "incumbent_select.json")]
            for arm in arms:
                jobs.extend((proposal["source"] if proposal["valid"] else incumbent,
                             select_rows, root / f"{arm['name']}_{index}_select.json")
                            for index, proposal in enumerate(proposals_by_arm[arm["name"]]))
            results = asyncio.run(evaluate_jobs(
                url, jobs, tokens=spec["solver_tokens"], samples=spec["select_samples"], seed=spec["seed"]))
            cursor = 1
            reports = {configuration_key(incumbent): (incumbent, root / "incumbent_report.json")}
            for arm in arms:
                name = arm["name"]
                selected = choose_candidate(
                    proposals_by_arm[name], results[cursor:cursor + spec["proposals"]], results[0],
                    incumbent, seed=spec["seed"], total_slots=len(arms) * spec["proposals"])
                cursor += spec["proposals"]
                selections[name] = selected
                write_json(root / f"{name}_selection.json", selected)
                if selected["best"]:
                    best = selected["best"]
                    reports.setdefault(best["configuration_sha256"],
                                       (best["source"], root / f"{name}_best_report.json"))
            status("reporting", frozen_candidates={name: row["best"] for name, row in selections.items()})
            jobs = [(source, report_rows, output) for source, output in reports.values()]
            outcomes = asyncio.run(evaluate_jobs(
                url, jobs, tokens=spec["solver_tokens"], samples=spec["report_samples"],
                seed=spec["seed"] + 5_000_003))
            by_configuration = dict(zip(reports, outcomes))
            baseline = by_configuration[configuration_key(incumbent)]
            summaries = {}
            for arm in arms:
                name = arm["name"]
                best = selections[name]["best"]
                key = best["configuration_sha256"] if best else configuration_key(incumbent)
                report = by_configuration[key]
                summaries[name] = {
                    "model": arm["model"], "thinking": arm["thinking"], "editor_token_cap": arm["tokens"],
                    "editor_context_limit": arm.get("max_model_len", 8192),
                    "generation_finish_reasons": [p["finish_reason"] for p in proposals_by_arm[name]],
                    "diversity": proposal_summary(proposals_by_arm[name], incumbent),
                    "selection": selections[name], "report_file": str(reports[key][1]),
                    "report_comparison": paired_difference(baseline, report),
                    "report_family_adjusted_lower_bound": paired_lower_bound(
                        baseline, report, alpha=0.05 / len(arms), seed=spec["seed"]),
                    "changed_fields": {field: {"before": value, "after": settings(best["source"])[field]}
                                       for field, value in settings(incumbent).items()
                                       if best and settings(best["source"])[field] != value},
                }
        launcher.close()
        launcher = None
        status("verifying_immutability")
        final_hashes = {model: initial_model_digest(Path(model)) for model in models}
        if final_hashes != model_hashes:
            raise RuntimeError("An editor or the frozen solver changed during the no-training probe")
        write_json(root / "COMPLETE.json", {
            "optimizer_updates": 0, "models_unchanged": True, "test_evaluated": False,
            "first_h1": bool(spec.get("first_h1")),
            "solver": spec["solver"], "baseline_accuracy": baseline["accuracy"],
            "question_count": len(report_rows), "report_samples": spec["report_samples"],
            "arms": summaries, "elapsed_seconds": time.monotonic() - start,
            "reused_proposal_slots": reused_slots,
            "physical_completed_samples": {
                "editor_proposals": len(arms) * spec["proposals"],
                "solver_selection": (1 + len(arms) * spec["proposals"]) * len(select_rows) * spec["select_samples"],
                "solver_reporting": len(reports) * len(report_rows) * spec["report_samples"],
            },
            "interpretation": (
                "Single first EOS-complete zero-shot H1 proposal; no best-of-N. SFT1152 solver and "
                "2048-token solver budget fixed. No artificial editor output-token cap; native context "
                "and runtime safety limits still apply. Reporting data was previously observed; "
                "this is exploratory, not an untouched test. No automatic harness deployment."
                if spec.get("first_h1") else
                "Exploratory editor/model-and-generation comparison on previously observed "
                "development reporting data. Thinking has a separate larger token budget. "
                "Valid changes do not imply task gains; no automatic harness deployment."),
        })
        status("completed", result=str(root / "COMPLETE.json"))
    except BaseException as exc:
        status("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if launcher is not None:
            launcher.close()
        if allocated:
            remaining = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                text=True).strip()
            if not remaining:
                subprocess.run(["sudo", "-n", "systemctl", "start", "bruli-h200-load.service"],
                               check=True, timeout=45)
            else:
                print(f"GPU processes remain; filler not restored: {remaining}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()

    def interrupted(signum, frame):
        raise InterruptedError(f"Editor probe received signal {signum}")

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    run_probe(args.spec)


if __name__ == "__main__":
    main()
