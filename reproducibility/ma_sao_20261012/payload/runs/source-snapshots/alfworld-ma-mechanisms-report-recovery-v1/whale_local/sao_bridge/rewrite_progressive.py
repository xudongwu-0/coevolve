"""Timeboxed, once-per-record SFT with immutable development gates."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import signal
import time

from .common import ROOT, file_sha, read_json, write_json
from .launch import Launcher
from .resources import stop_known
from .rewrite_audit import audit_rows
from .rewrite_compare import compare
from .rewrite_data import TeacherResponse, digest, validate_example


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def accepted_rows(queue: Path, previous: dict) -> tuple[list[dict], dict, list[str]]:
    rows, hashes, pending = [], {}, []
    for path in sorted(queue.glob("*/response.json")):
        # The queue predates streaming collection and writes responses in place.
        if time.time() - path.stat().st_mtime < 5:
            pending.append(path.parent.name)
            continue
        response = TeacherResponse.model_validate(read_json(path))
        request = read_json(path.parent / "request.json")
        if response.request_sha256 != digest(request):
            raise ValueError(f"Teacher request digest mismatch: {path.parent.name}")
        counts = {s: sum(e.split == s for e in response.examples)
                  for s in ("train", "dev", "test")}
        if counts != request["counts"]:
            raise ValueError(f"Incomplete teacher response: {path.parent.name}")
        for example in response.examples:
            validate_example(example)
            rows.append(example.model_dump())
        hashes[path.parent.name] = file_sha(path)
    if any(hashes.get(name) != sha for name, sha in previous.items()):
        raise ValueError("A previously accepted teacher response changed or disappeared")
    return rows, hashes, pending


def unseen_train(rows: list[dict], consumed: set[str], world_size=8) -> list[dict]:
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate teacher IDs")
    if not consumed.issubset(ids):
        raise ValueError("Previously trained examples disappeared")
    fresh = [row for row in rows if row["split"] == "train" and row["id"] not in consumed]
    if len(fresh) % world_size:
        raise ValueError("New train records must partition without dropping any examples")
    return fresh


def selection_gate(control_comparison: dict, best_comparison: dict, *,
                   min_gain=0.005, math_margin=0.03, legality_margin=0.01) -> dict:
    edit = control_comparison["edits"]["dev"]
    math_result = control_comparison["math_dev"]
    harness = control_comparison["harness_generation"]["candidate"]
    values = [edit["legal"]["candidate_minus_control"],
              edit["instruction_following"]["candidate_minus_control"],
              math_result["candidate_minus_control"],
              *math_result["paired_question_bootstrap_95pct"],
              best_comparison["edits"]["dev"]["instruction_following"]["candidate_minus_control"],
              best_comparison["math_dev"]["candidate_minus_control"],
              harness["legal_fraction"]]
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Nonfinite development score")
    reasons = []
    if edit["legal"]["candidate_minus_control"] < -legality_margin:
        reasons.append("edit_legality_regression")
    if edit["instruction_following"]["candidate_minus_control"] < -legality_margin:
        reasons.append("edit_instruction_regression")
    if math_result["paired_question_bootstrap_95pct"][0] < -math_margin:
        reasons.append("math_noninferiority_not_established")
    if harness["legal_fraction"] < 0.95:
        reasons.append("actual_harness_legality_below_95pct")
    if harness["distinct_valid_configurations"] < 2:
        reasons.append("actual_harness_proposal_collapse")
    edit_gain = best_comparison["edits"]["dev"]["instruction_following"]["candidate_minus_control"]
    math_gain = best_comparison["math_dev"]["candidate_minus_control"]
    gain = (edit_gain + math_gain) / 2
    regression = (math_result["candidate_minus_control"] < -math_margin
                  or edit["legal"]["candidate_minus_control"] < -legality_margin
                  or edit["instruction_following"]["candidate_minus_control"] < -legality_margin)
    return {
        "eligible": not reasons, "improved": not reasons and gain >= min_gain,
        "stop_for_regression": regression, "reasons": reasons,
        "instruction_gain_over_best": edit_gain, "math_gain_over_best": math_gain,
        "combined_gain_over_best": gain, "minimum_gain": min_gain,
        "selection_score": "0.5 * edit instruction fidelity + 0.5 * math accuracy",
        "math_noninferiority_margin": math_margin,
        "scope": "Exploratory fixed-development selection, not broad rewriting capability.",
    }


def validate_control(control: Path, args):
    if not (control / "COMPLETE.json").is_file() or (control / "FAILED.json").exists():
        raise ValueError("A completed original-model control is required")
    spec = read_json(control / "spec.json")
    expected = {
        "initial_model": str(args.model), "input_model": str(args.model),
        "data_sha256": file_sha(args.fixed_data), "math_data_sha256": file_sha(args.math_data),
        "math_questions": args.math_questions, "samples": args.samples,
        "math_samples": args.math_samples, "edit_tokens": 1024,
        "math_tokens": 2048, "server_context": 8192, "evaluation_splits": ["dev"],
    }
    for key, value in expected.items():
        if spec.get(key) != value:
            raise ValueError(f"Original control protocol mismatch: {key}")


def evaluation_args(args, model: Path, output: Path, *, heldout=False):
    values = [
        "--mode", "evaluate", "--model", model, "--origin-model", args.model,
        "--data", args.fixed_data, "--output", output, "--math-data", args.math_data,
        "--math-questions", args.math_questions, "--math-samples", args.math_samples,
        "--samples", args.samples, "--edit-tokens", 1024, "--math-tokens", 2048,
        "--server-context", 8192, "--evaluation-gpus", "0,1,2,3,4,5,6,7",
    ]
    if not heldout:
        values.append("--dev-only")
    return values


def execute(args):
    args.run.mkdir(parents=True, exist_ok=False)
    soft_end = datetime.fromisoformat(args.train_until).timestamp()
    hard_end = datetime.fromisoformat(args.finish_by).timestamp()
    if not time.time() < soft_end < hard_end:
        raise ValueError("Training and final deadlines must be ordered and in the future")
    spec = {key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()}
    spec.update(created_at=utc_now(), fixed_data_sha256=file_sha(args.fixed_data),
                repeated_training_records=False, optimizer_continuity=True,
                test_used_for_selection=False, fixed_target_count_required=False,
                selection_score="0.5 * edit instruction fidelity + 0.5 * math accuracy",
                edit_ceiling_note="Original model is already 100% on this exact-edit dev set.")
    write_json(args.run / "protocol.json", spec)
    state = {
        "phase": "waiting_for_predecessor", "created_at": utc_now(),
        "training_deadline": args.train_until, "completion_deadline": args.finish_by,
        "rounds": [], "consumed_train_ids": [], "teacher_response_hashes": {},
        "best_model": str(args.model), "best_evaluation": str(args.control),
        "sao_ready": False, "AIME_opened": False,
    }

    def status(phase, **values):
        state.update(phase=phase, updated_at=utc_now(), **values)
        write_json(args.run / "STATUS.json", state)
        print(utc_now(), phase, flush=True)

    launcher = None
    try:
        status("waiting_for_predecessor")
        while not args.predecessor_exit.exists():
            if time.time() >= soft_end - 900:
                raise TimeoutError("Predecessor did not release GPUs within the training budget")
            time.sleep(10)
        if args.predecessor_exit.read_text().strip() != "0":
            raise RuntimeError("Predecessor evaluation failed; inspect its logs before allocation")
        validate_control(args.control, args)

        def wait(process, name, deadline):
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(f"No time remains for {name}")
            launcher.wait(process, name, timeout=remaining)

        targets = read_json(args.campaign)["targets"]
        current, resume_state = args.model, None
        best_eval, best_model = args.control, args.model
        consumed, stale, round_number = set(), 0, 0
        hashes = {}
        stop_reason = "training_deadline"
        if args.initial_training:
            initial = args.initial_training
            result = read_json(initial / "result.json")
            config = read_json(initial / "config.json")
            if (result["status"] != "completed" or not result["hf_cpu_reload_verified"]
                    or config["lr"] != args.lr or config["epochs"] != 1
                    or config["world_size"] != 8 or config["batch_size"] != 128
                    or config["seed"] != 42):
                raise ValueError("Initial stage is incompatible with incremental continuation")
            trained_rows = read_json(initial / "train_records.json")
            consumed = {row["id"] for row in trained_rows}
            if (len(consumed) != result["examples_seen"]
                    or any(row["split"] != "train" for row in trained_rows)):
                raise ValueError("Initial stage training identities are inconsistent")
            current = initial / "checkpoint"
            resume_state = initial / "training_state.pt"
            if not resume_state.is_file():
                raise FileNotFoundError("Initial stage has no optimizer state")
            if read_json(args.initial_evaluation / "spec.json")["input_model"] != str(current):
                raise ValueError("Initial evaluation is not of the imported checkpoint")
            baseline = compare(args.control, args.initial_evaluation)
            gate = selection_gate(baseline, baseline, min_gain=args.min_gain,
                                  math_margin=args.math_margin)
            write_json(args.run / "initial_selection.json", gate)
            if gate["improved"]:
                best_model, best_eval = current, args.initial_evaluation
            else:
                stale = 1
            state["rounds"].append({
                "round": 0, "imported_training": str(initial), "new_train": len(consumed),
                "cumulative_train": len(consumed), "model": str(current), "selection": gate,
            })
            status("imported_verified_stage", consumed_train_ids=sorted(consumed),
                   best_model=str(best_model), best_evaluation=str(best_eval),
                   plateau_rounds=stale)
            if gate["stop_for_regression"]:
                raise ValueError("Initial candidate regressed; do not continue its optimizer")
        while time.time() < soft_end - 900:
            rows, hashes, pending = accepted_rows(args.queue, hashes)
            if not rows:
                status("waiting_for_teacher_data", pending_writes=pending)
                time.sleep(30)
                continue
            audit = audit_rows(rows, targets, full_budget=False)
            if audit["errors"]:
                raise ValueError(f"Teacher structural audit failed: {audit['errors']}")
            fresh = unseen_train(rows, consumed)
            status("collecting", accepted_train=audit["counts"]["train"],
                   new_train=len(fresh), teacher_response_hashes=hashes,
                   pending_writes=pending)
            if len(fresh) < args.minimum_new:
                status("waiting_for_teacher_data", new_train=len(fresh))
                if launcher:
                    launcher.check()
                time.sleep(30)
                continue
            if launcher is None:
                stop_known(require_exclusive=True)
                launcher = Launcher(args.run)
            round_number += 1
            root = args.run / f"round_{round_number:02d}"
            root.mkdir()
            write_json(root / "audit.json", audit)
            write_json(root / "new_train.json", fresh)
            write_json(root / "lineage.json", {
                "parent_model": str(current),
                "resume_state": str(resume_state) if resume_state else None,
                "teacher_response_hashes": hashes,
                "new_train_ids": [r["id"] for r in fresh],
                "previously_consumed_train_ids": sorted(consumed),
                "epochs": 1,
            })
            training = root / "training"
            train_args = [
                "--standalone", "--local-addr=127.0.0.1", "--nnodes=1",
                "--nproc-per-node=8", "--max-restarts=0",
                "-m", "sao_bridge.rewrite_sft_train", "--model", current,
                "--data", root / "new_train.json", "--output", training,
                "--batch-size", 128, "--epochs", 1, "--lr", args.lr, "--seed", 42,
                "--max-length", 4096, "--distributed-timeout", 900, "--save-training-state",
            ]
            if resume_state:
                train_args.extend(["--resume-state", resume_state])
            status("training", active_round=round_number, new_train=len(fresh))
            process = launcher.module("torch.distributed.run", train_args,
                                      gpus="0,1,2,3,4,5,6,7", name=f"train_{round_number:02d}")
            wait(process, "incremental SFT", soft_end)
            result = read_json(training / "result.json")
            if (result["status"] != "completed" or not result["hf_cpu_reload_verified"]
                    or result["examples_seen"] != len(fresh)):
                raise RuntimeError("Training completion or once-per-record receipt failed")
            current = training / "checkpoint"
            resume_state = training / "training_state.pt"
            if not resume_state.is_file():
                raise FileNotFoundError("Incremental training did not persist optimizer state")
            consumed.update(r["id"] for r in fresh)
            status("development_evaluation", consumed_train_ids=sorted(consumed))
            evaluation = root / "dev"
            process = launcher.module("sao_bridge.rewrite_trial",
                                      evaluation_args(args, current, evaluation),
                                      gpus="0,1,2,3,4,5,6,7", name=f"dev_{round_number:02d}")
            wait(process, "development gate", soft_end)
            baseline = compare(args.control, evaluation)
            incumbent = compare(best_eval, evaluation)
            gate = selection_gate(baseline, incumbent, min_gain=args.min_gain,
                                  math_margin=args.math_margin)
            write_json(root / "comparison.json", baseline)
            write_json(root / "selection.json", gate)
            if gate["improved"]:
                best_model, best_eval, stale = current, evaluation, 0
            else:
                stale += 1
            state["rounds"].append({
                "round": round_number, "new_train": len(fresh),
                "cumulative_train": len(consumed), "model": str(current),
                "training_seconds": result["training_seconds"], "selection": gate,
            })
            status("development_checked", best_model=str(best_model),
                   best_evaluation=str(best_eval), plateau_rounds=stale)
            if gate["stop_for_regression"]:
                stop_reason = "development_regression"
                break
            if stale >= args.patience:
                stop_reason = "development_plateau"
                break
            if round_number >= args.max_rounds:
                stop_reason = "maximum_rounds"
                break

        selected = best_model != args.model
        frozen = {
            "model": str(best_model), "dev_evaluation": str(best_eval),
            "selected_at": utc_now(), "stop_reason": stop_reason,
            "selected_new_sft_model": selected, "test_used_for_selection": False,
        }
        write_json(args.run / "FROZEN.json", frozen)
        status("frozen", **frozen)
        final_passed = False
        if selected:
            if launcher is None:
                stop_known(require_exclusive=True)
                launcher = Launcher(args.run)
            for label, model in (("control", args.model), ("selected", best_model)):
                status("heldout_evaluation", final_arm=label)
                output = args.run / f"final_{label}"
                process = launcher.module("sao_bridge.rewrite_trial",
                                          evaluation_args(args, model, output, heldout=True),
                                          gpus="0,1,2,3,4,5,6,7", name=f"final_{label}")
                wait(process, "final heldout verification", hard_end)
            final = compare(args.run / "final_control", args.run / "final_selected")
            write_json(args.run / "final_comparison.json", final)
            test = final["edits"]["test"]
            final_passed = (
                test["legal"]["candidate_minus_control"] >= -0.01
                and test["instruction_following"]["candidate_minus_control"] >= -0.01
                and (test["instruction_following"]["candidate_minus_control"]
                     + final["math_dev"]["candidate_minus_control"]) / 2 >= args.min_gain
                and final["math_dev"]["paired_question_bootstrap_95pct"][0] >= -args.math_margin
            )
        completion = {
            **frozen, "finished_at": utc_now(), "trained_records": len(consumed),
            "sao_ready": final_passed,
            "reason": ("selected_checkpoint_passed_gates" if final_passed
                       else "no_validated_improvement_do_not_replace_baseline"),
            "limitations": [
                "Small fixed synthetic heldout set; not broad teacher capability transfer.",
                "Repeated development selection is exploratory; AIME remains sealed.",
                "No SAO run is automatically launched by this SFT controller.",
            ],
        }
        status("completed", **completion)
        write_json(args.run / "COMPLETE.json", completion)
    except BaseException as exc:
        status("failed", error=f"{type(exc).__name__}: {exc}")
        write_json(args.run / "FAILED.json", {"error": repr(exc), "at": utc_now()})
        raise
    finally:
        if launcher:
            launcher.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--fixed-data", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--predecessor-exit", type=Path, required=True)
    parser.add_argument("--initial-training", type=Path)
    parser.add_argument("--initial-evaluation", type=Path)
    parser.add_argument("--model", type=Path, default=ROOT / "assets/Qwen3-4B-Instruct-2507")
    parser.add_argument("--campaign", type=Path, default=ROOT / "assets/rewrite_round1_campaign.json")
    parser.add_argument("--math-data", type=Path, default=ROOT / "assets/dapo_aime_protocol/dev.json")
    parser.add_argument("--train-until", required=True)
    parser.add_argument("--finish-by", required=True)
    parser.add_argument("--math-questions", type=int, default=256)
    parser.add_argument("--math-samples", type=int, default=8)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--minimum-new", type=int, default=128)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--min-gain", type=float, default=0.005)
    parser.add_argument("--math-margin", type=float, default=0.03)
    args = parser.parse_args()
    if bool(args.initial_training) != bool(args.initial_evaluation):
        parser.error("--initial-training and --initial-evaluation must be supplied together")
    if (args.minimum_new < 8 or args.minimum_new % 8 or args.patience < 1
            or args.max_rounds < 1 or min(args.samples, args.math_samples, args.math_questions) < 1
            or not 0 < args.lr < 1 or not 0 < args.min_gain < 1
            or not 0 < args.math_margin < 1):
        parser.error("Invalid incremental training or selection budget")
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())

    def interrupted(signum, frame):
        raise InterruptedError(f"Controller received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    execute(args)


if __name__ == "__main__":
    main()
