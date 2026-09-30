"""Finite H-W-H experiment with frozen stage states and dev-only selection."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import hashlib
from pathlib import Path
import subprocess
import time

import httpx

from .analysis import paired_lower_bound, verify_sample_ledger
from .budget import ComputeBudget, claim_formal_baseline, initial_role_models, validate_trial
from .common import ROOT, file_sha, read_json, write_json
from .domain_api import get_domain
from .evaluation import Sampler, evaluate
from .launch import Launcher
from .phase_data import editor_dataset, solver_dataset
from .phase_recovery import cached_evaluation, editor_pair_audit, validate_resume
from .replay import recycle_pool
from .resources import stop_known
from .retool import EDITOR_SYSTEM, INITIAL_SOURCE, harness_spec


async def propose(url, harness, count, tokens, seed, *, temperature=0.5,
                  chat_template_kwargs=None, request_timeout=900, domain="math", feedback=""):
    task = get_domain(domain)
    instruction = (
        "Improve only the six allowed literal values; keep exact class and import.\n"
        if domain == "math" else
        "Improve the permitted settings and executable hooks in this reusable harness.\n"
    )
    prompt = instruction + harness
    if feedback:
        prompt += "\nTraining-only failure feedback:\n" + feedback
    sampler = Sampler(url, concurrency=16, temperature=temperature,
                      chat_template_kwargs=chat_template_kwargs, request_timeout=request_timeout)
    try:
        async def one(index):
            reply = await sampler([
                {"role": "system", "content": task.EDITOR_SYSTEM},
                {"role": "user", "content": prompt},
            ], None, tokens, seed + index)
            choice = reply["choices"][0]
            message = choice["message"]
            text = message.get("content") or ""
            record = {"id": index, "text": text, "finish_reason": choice.get("finish_reason"),
                      "usage": reply.get("usage"),
                      "reasoning_characters": len(message.get("reasoning")
                                                 or message.get("reasoning_content") or "")}
            try:
                record.update(valid=True, source=task.harness_spec(text)["source"])
            except (ValueError, SyntaxError, TypeError) as exc:
                record.update(valid=False, error=str(exc))
            return record
        return await asyncio.gather(*(one(i) for i in range(count)))
    finally:
        await sampler.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    spec = read_json(args.spec)
    root = Path(spec["run"]).resolve()
    elapsed_before = validate_resume(root, spec) if args.resume else 0.0
    if root.exists() and not args.resume:
        raise FileExistsError("Use a fresh experiment directory")
    initial_models = initial_role_models(spec)
    if spec.get("formal_baseline") and not spec.get("contract"):
        raise ValueError("Formal baseline requires a frozen compute/protocol contract")
    if spec.get("contract") and not spec.get("formal_baseline") and not spec.get("reference_run"):
        raise ValueError("Formal alternatives require the completed frozen baseline reference")
    if spec.get("contract") and not spec.get("formal_baseline") and spec["method"] == "independent":
        raise ValueError("The formal independent baseline must be claimed exactly once")
    if spec.get("contract"):
        validate_trial(Path(spec["contract"]), spec)
        names = subprocess.check_output([
            "nvidia-smi", "--query-gpu=name", "--format=csv,noheader",
        ], text=True).splitlines()
        if len(names) != 8 or set(names) != {"NVIDIA H200"}:
            raise ValueError("Formal protocol requires eight actual NVIDIA H200 GPUs")
    stop_known(require_exclusive=True)
    root.mkdir(parents=True, mode=0o700, exist_ok=args.resume)
    if spec.get("formal_baseline"):
        claim_formal_baseline(Path(spec["contract"]), root, spec["seed"])
    budget = ComputeBudget(
        Path(spec["contract"]), root / "compute.json",
        hardware="NVIDIA H200", gpu_count=8,
        reference_run=Path(spec["reference_run"]) if spec.get("reference_run") else None,
        trial_seed=spec["seed"],
    ) if spec.get("contract") else None
    launch = Launcher(root, budget)
    start = time.monotonic() - elapsed_before
    data = Path(spec["data"]).resolve()
    editor, solver = initial_models["editor"], initial_models["solver"]
    editor_critic, solver_critic, shared_critic = editor, solver, editor
    harness = root / "h0.py"
    harness.write_text(INITIAL_SOURCE)
    stages = [{"stage": "h0w0", "solver": solver, "editor": editor, "harness": str(harness)}]
    h_receipts = None
    previous_h_steps = 0
    method = spec["method"]
    if method not in {"independent", "central", "ma_sao"}:
        raise ValueError("Unknown experimental method")
    central = method == "central" or bool(spec.get("central_critic"))
    write_json(root / "spec.json", spec)
    manifest_name = f"source_manifest_resume_{time.time_ns()}.json" if args.resume else "source_manifest.json"
    write_json(root / manifest_name, {
        p.name: file_sha(p) for p in Path(__file__).parent.glob("*.py")
    })

    def persist(status, **extra):
        write_json(root / "STATUS.json", {
            "status": status, "elapsed_seconds": time.monotonic() - start,
            "stages": stages, **extra,
        })
        if budget:
            budget.record(status)
            if status == "training_frozen":
                budget.check()

    @contextmanager
    def solver_server(name):
        with launch.vllm(solver, port=18801, gpus="4,5,6,7", name=name,
                         data_parallel=4) as url:
            yield url

    dev_rows = read_json(data / "dev.json")
    n_select = spec.get("select_questions", 128)
    n_report = spec.get("report_questions", 128)
    select_rows = dev_rows[:n_select]
    report_rows = dev_rows[n_select:n_select + n_report]
    if len(report_rows) != n_report:
        raise ValueError("Development split too small for disjoint selection/report questions")
    tail = spec.get("report_train_tail", 0)
    if tail:
        # Never-trained tail of the training split widens the report set; solver phases
        # read training questions from the head, so assert they cannot reach the tail.
        train_rows = read_json(data / "train.json")
        used = spec.get("cycles", 1) * spec["solver_steps"] * spec.get(
            "solver_batch_size", spec["batch_size"])
        if used > len(train_rows) - tail:
            raise ValueError("Solver training would reach the report tail")
        report_rows = report_rows + train_rows[-tail:]
    report_samples = spec.get("report_samples", 4)
    select_samples = spec.get("select_samples", 1)

    def evaluate_many(url, jobs):
        """jobs: (source, rows, samples, output). Distinct sources run concurrently."""
        async def run():
            sampler = Sampler(url, concurrency=spec.get("eval_concurrency", 256))
            try:
                tasks = []
                for index, (source, rows, samples, output) in enumerate(jobs):
                    # Report uses common random numbers across every stage/method;
                    # selection slots use disjoint seed blocks so duplicate proposals
                    # still consume distinct, fixed samples.
                    seed = spec["seed"] + (5_000_003 if rows is report_rows else 0)
                    async def evaluate_or_reuse(source=source, rows=rows, samples=samples,
                                                output=output, seed=seed):
                        if args.resume and output.is_file():
                            return cached_evaluation(output, rows, samples)
                        return await evaluate(sampler, rows, source, tokens=spec["solver_tokens"],
                                              samples=samples, seed=seed)
                    tasks.append(evaluate_or_reuse())
                return await asyncio.gather(*tasks)
            finally:
                await sampler.close()
        results = asyncio.run(run())
        for (_, _, _, output), result in zip(jobs, results):
            write_json(output, result)
        launch.check()
        return results

    def train(role, phase, model, critic, source_data, steps, feedback, evaluator=None,
              recycle=None, warmup=0):
        nonlocal shared_critic
        out = root / phase
        context = None
        if central:
            context = root / (phase + "_joint_context.json")
            write_json(context, {
                "phase": "w" if role == "solver" else ("h1" if phase == "h1" else "h2"),
                "actor_role": role, "partner_checkpoint": solver if role == "editor" else editor,
                "harness_source": harness.read_text(), "calibration_accuracy": feedback,
                "calibration_split": "dev", "partner_version": len(stages) - 1,
            })
        actor_gpus, rollout_gpus = spec.get(f"{role}_layout", [2, 2] if role == "editor" else [4, 4])
        phase_spec = {
            "run": str(out), "role": role, "model": model,
            "critic_model": shared_critic if context else critic,
            "phase": phase, "experiment": root.name,
            "steps": steps, "batch_size": spec.get(f"{role}_batch_size", spec["batch_size"]),
            "seed": spec["seed"],
            "lr": spec[f"{role}_lr"], "tokens": spec[f"{role}_tokens"],
            "temperature": spec.get("editor_temperature", 0.7) if role == "editor" else 1.0,
            "staleness": spec.get("staleness", 1),
            "harness": str(harness), "data": str(source_data),
            "evaluator_url": evaluator, "central_context": str(context) if context else None,
            "interaction_credit": spec.get("interaction_credit", 0.0) if method == "ma_sao" else 0.0,
            "weight_update_mode": spec.get("weight_update_mode", "xccl"),
            "actor_gpus": actor_gpus, "rollout_gpus": rollout_gpus,
            "recycle_pool": str(recycle) if recycle else None,
            "critic_warmup_steps": warmup,
        }
        phase_spec_path = root / (phase + "_spec.json")
        if args.resume and (out / "trained.json").is_file():
            if read_json(phase_spec_path) != phase_spec:
                raise ValueError(f"Cannot change a completed phase specification: {phase}")
            receipt = read_json(out / "trained.json")
            if context:
                shared_critic = receipt["checkpoints"]["critic"]
            return receipt["checkpoints"]
        if out.exists():
            raise ValueError(f"Refusing to repeat a partial optimizer phase: {phase}")
        write_json(phase_spec_path, phase_spec)
        gpus = ",".join(str(i) for i in range(actor_gpus + rollout_gpus))
        process = launch.module("sao_bridge.large_train", ["--spec", phase_spec_path],
                                gpus=gpus, name=phase + "_train")
        launch.wait(process, phase + "_train")
        receipt = read_json(out / "trained.json")
        if context:
            shared_critic = receipt["checkpoints"]["critic"]
        return receipt["checkpoints"]

    reward_n = spec["reward_questions"]
    try:
        for cycle in range(spec.get("cycles", 1) + 1):
            phase = f"h{cycle + 1}"
            persist(phase + "_start")
            with solver_server(phase + "_frozen_solver") as url:
                incumbent = harness.read_text()
                select, report = evaluate_many(url, [
                    (incumbent, select_rows, select_samples, root / (phase + "_before_select.json")),
                    (incumbent, report_rows, report_samples, root / (phase + "_before_report.json")),
                ])
                stages[-1].update(select_accuracy=select["accuracy"],
                                  report_accuracy=report["accuracy"],
                                  report_metric=report["metric"])
                persist(phase + "_incumbent_reported")
                # Disjoint reward questions per H phase: later solver phases may train on
                # recycled episodes from earlier reward questions without contaminating them.
                completed_h = args.resume and (root / phase / "trained.json").is_file()
                evaluator = None if completed_h else launch.module("sao_bridge.evaluation", [
                    "--url", url, "--data", data / "reward.json", "--harness", harness,
                    "--output", root / (phase + "_reward_evals"),
                    "--tokens", spec["solver_tokens"], "--limit", reward_n,
                    "--offset", cycle * reward_n, "--port", 18802, "--record-tokens",
                    "--concurrency", spec.get("eval_concurrency", 256),
                ], gpus="", name=phase + "_reward_service")
                try:
                    if evaluator is not None:
                        launch.ready(evaluator, "http://127.0.0.1:18802/health")
                    h_data = root / (phase + "_editor_data.json")
                    editor_batch = spec.get("editor_batch_size", spec["batch_size"])
                    count = spec["editor_steps"] * editor_batch
                    # Extra fresh input rows cover asynchronous prefetch; optimizer budget stays fixed.
                    count += editor_batch  # exactly one asynchronous prefetch batch
                    write_json(h_data, editor_dataset(
                        incumbent, select["accuracy"], count, cycle,
                        replay_dir=h_receipts,
                        replay_fraction=spec.get("replay_fraction", 0.25)
                        if method == "ma_sao" and h_receipts else 0,
                        editor_step_offset=previous_h_steps,
                    ))
                    checkpoints = train(
                        "editor", phase, editor, editor_critic, h_data,
                        spec["editor_steps"], select["accuracy"], "http://127.0.0.1:18802",
                    )
                    editor, editor_critic = checkpoints["actor"], checkpoints["critic"]
                    h_receipts = root / phase / "receipts"
                    previous_h_steps = spec["editor_steps"]
                finally:
                    if evaluator is not None:
                        launch.stop(evaluator)
                audit = editor_pair_audit(root, phase, spec)
                write_json(root / f"{phase}_pair_audit.json", audit)
                if audit["unused_request_indices"]:
                    print(f"{phase}: unused async reward requests retained as overhead: "
                          f"{audit['unused_request_indices']}; not strict compute-matched", flush=True)
                # Fixed proposal budget: always generate exactly N post-training slots.
                # Invalid slots are evaluated as the incumbent (but excluded from selection),
                # preserving physical solver samples. Duplicate valid sources are aggregated.
                proposal_path = root / (phase + "_proposals.json")
                if args.resume and proposal_path.is_file():
                    candidates = read_json(proposal_path)
                    if len(candidates) != spec["proposals"]:
                        raise ValueError("Incomplete cached proposal slots")
                else:
                    with launch.vllm(editor, port=18803, gpus="0", name=phase + "_proposal") as editor_url:
                        candidates = asyncio.run(propose(
                            editor_url, incumbent, spec["proposals"],
                            spec["editor_tokens"], spec["seed"] + 1000 + 100 * cycle,
                        ))
                write_json(root / (phase + "_proposals.json"), candidates)
                slot_sources = [candidate["source"] if candidate["valid"] else incumbent
                                for candidate in candidates]
                outcomes = evaluate_many(url, [
                    (source, select_rows, select_samples, root / f"{phase}_candidate_{i}_select.json")
                    for i, source in enumerate(slot_sources)
                ])
                grouped = {}
                for slot, (candidate, outcome) in enumerate(zip(candidates, outcomes)):
                    if candidate["valid"] and candidate["source"] != incumbent:
                        grouped.setdefault(candidate["source"], (outcome, slot))
                scored = [{"source": incumbent, "accuracy": select["accuracy"],
                           "selection_lcb": 0.0, "incumbent": True}]
                for source, (outcome, slot) in grouped.items():
                    scored.append({
                        "source": source, "accuracy": outcome["accuracy"], "first_slot": slot,
                        "selection_lcb": paired_lower_bound(
                            select, outcome, alpha=0.05 / spec["proposals"],
                            seed=spec["seed"] + 31_337 + slot),
                        "incumbent": False,
                    })
                qualified = [row for row in scored if not row["incumbent"]
                             and row["selection_lcb"] > 0]
                chosen = max(qualified, key=lambda row: row["accuracy"]) if qualified else scored[0]
                harness = root / (phase + ".py")
                harness.write_text(chosen["source"])
                chosen_report, = evaluate_many(url, [
                    (chosen["source"], report_rows, report_samples,
                     root / (phase + "_selected_report.json"))])
                write_json(root / (phase + "_selection.json"), {
                    "candidates": scored, "selected": chosen, "split": "dev_select",
                })
                stages.append({"stage": phase + f"w{cycle}", "solver": solver,
                               "editor": editor, "harness": str(harness),
                               "select_accuracy": chosen["accuracy"],
                               "report_accuracy": chosen_report["accuracy"],
                               "report_metric": chosen_report["metric"]})
                persist(phase + "_selected")
            if cycle == spec.get("cycles", 1):
                break
            phase = f"w{cycle + 1}"
            persist(phase + "_start")
            solver_batch = spec.get("solver_batch_size", spec["batch_size"])
            count = spec["solver_steps"] * solver_batch
            recycle = None
            recycle_fraction = spec.get("solver_recycle_fraction", 0.0) if method == "ma_sao" else 0.0
            if not 0 <= recycle_fraction <= 1:
                raise ValueError("solver_recycle_fraction must be in [0, 1]")
            warm_steps = spec.get("critic_warmup_steps", 0) if method == "ma_sao" else 0
            warm_ids, mix_ids = [], []
            if recycle_fraction or warm_steps:
                audit = editor_pair_audit(root, f"h{cycle + 1}", spec)
                pool = recycle_pool(root / f"h{cycle + 1}_reward_evals",
                                    request_indices=set(audit["committed_request_indices"]))
                ids = [e["id"] for e in pool]
                warm_n = warm_steps * solver_batch * spec.get("warmup_recycled_per_row", 4)
                # Warm-up and actor mixing take disjoint episodes in content-hash order.
                mix_n = int(count * recycle_fraction)
                if warm_n + mix_n > len(ids):
                    raise ValueError(
                        f"Partner pool pseudoreplication forbidden: need {warm_n + mix_n}, "
                        f"have {len(ids)}")
                warm_ids = ids[:warm_n]
                mix_ids = ids[warm_n:warm_n + mix_n]
                recycle = root / (phase + "_recycle.json")
                used = set(warm_ids) | set(mix_ids)
                write_json(recycle, [e for e in pool if e["id"] in used])
            w_data = root / (phase + "_solver_data.json")
            solver_dataset(data / "train.json", w_data, count, cycle * count,
                           recycled_ids=mix_ids or None, warmup_ids=warm_ids or None,
                           warmup_rows=warm_steps * solver_batch)
            checkpoints = train(
                "solver", phase, solver, solver_critic, w_data, spec["solver_steps"] + warm_steps,
                stages[-1]["select_accuracy"], recycle=recycle, warmup=warm_steps,
            )
            solver, solver_critic = checkpoints["actor"], checkpoints["critic"]
            stages.append({"stage": f"h{cycle + 1}" + phase, "solver": solver,
                           "editor": editor, "harness": str(harness)})
        ledger = verify_sample_ledger(root)
        persist("training_frozen", physical_sample_ledger=ledger)
        write_json(root / "FROZEN.json", {"stages": stages, "spec": spec,
                                        "physical_sample_ledger": ledger,
                                        "training_seconds": time.monotonic() - start})
        # Test is opt-in after predeclared final configuration, never used by tuning.
        if spec.get("evaluate_test"):
            launch.budget = None
            with solver_server("final_test_solver") as url:
                evaluate_many(url, [(harness.read_text(), read_json(data / "test.json"), 8,
                                     root / "test.json")])
        write_json(root / "COMPLETE.json", {
            "method": method, "stages": stages,
            "final_report_accuracy": stages[-1]["report_accuracy"],
            "report_metric": stages[-1]["report_metric"],
            "test_evaluated": bool(spec.get("evaluate_test")),
            "elapsed_seconds": time.monotonic() - start,
            "claim": "single-seed controlled run; no automatic superiority claim",
        })
        print("Completed", root, flush=True)
    except BaseException as exc:
        persist("failed", error=repr(exc))
        raise
    finally:
        launch.close()


if __name__ == "__main__":
    main()
