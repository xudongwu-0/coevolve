"""Small WHALE ReTool Math qualification: native RSFT, native dynamic MH, four cells."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pipeline
from math_runtime import BASE_HARNESS, DOMAIN, ROOT, evaluate, service


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gpu", default="8")
    parser.add_argument("--policy-port", type=int, default=18841)
    parser.add_argument("--proposer-port", type=int, default=18842)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--mh-iterations", type=int, default=2)
    parser.add_argument("--mh-proposals", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--parameter-dtype", choices=["bfloat16", "float32"], default="float32")
    parser.add_argument("--gate-reference", type=Path,
                        help="Enable soft routing only after this baseline passes its two-axis audit")
    parser.add_argument("--evaluation-temperature", type=float, default=None)
    parser.add_argument("--evaluation-rollouts", type=int, default=None)
    parser.add_argument("--share-first-probe", action="store_true",
                        help="Reuse the audited initial TRAIN probe in both logical sample budgets")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    assert 1 <= args.rounds <= 4 and 1 <= args.rollouts <= 8
    assert 1 <= args.mh_iterations <= 6 and 1 <= args.mh_proposals <= 3
    assert args.evaluation_temperature is None or 0 <= args.evaluation_temperature <= 2
    assert args.evaluation_rollouts is None or 1 <= args.evaluation_rollouts <= 8
    assert not args.share_first_probe or args.gate_reference
    run = args.run_dir.resolve()
    if run.exists() and any(run.iterdir()) and not args.resume:
        raise FileExistsError("Use a fresh directory or --resume")
    run.mkdir(parents=True, exist_ok=True)
    data = read(args.manifest)
    for entry in data["splits"].values():
        assert pipeline.sha(entry["path"]) == entry["sha256"]
    base = Path((ROOT / "runs/whale_setup/base_model_path.txt").read_text().strip())
    config = {"scope": "native WHALE ReToolEnv, strict boxed binary reward, native math RSFT actor and MH selection; local Transformers and isolated SandboxFusion-compatible Python transport",
        "data": data, "base_model": str(base), "rounds": args.rounds, "rollouts_per_task": args.rollouts,
        "learning_rate": args.learning_rate, "mh_iterations": args.mh_iterations, "mh_proposals": args.mh_proposals,
        "gpu": args.gpu, "policy_port": args.policy_port, "proposer_port": args.proposer_port,
        "assistant_token_budget": 2048, "policy_max_tokens": 1024, "stop_at_chat_end": True,
        "minimum_conditional_gain_pp": 5.0, "confirmation_split": "confirmation",
        "gate_test_split_unseen": "test", "gate_enabled": bool(args.gate_reference),
        "proposer": "active Codex via localhost FastAPI", "seed": 42}
    if args.parameter_dtype != "bfloat16":
        config["parameter_dtype"] = args.parameter_dtype
    if args.evaluation_temperature is not None or args.evaluation_rollouts is not None:
        config["evaluation_temperature"] = args.evaluation_temperature if args.evaluation_temperature is not None else 0.0
        config["evaluation_rollouts"] = args.evaluation_rollouts if args.evaluation_rollouts is not None else 1
        config["evaluation_metric"] = (f"mean_at_{config['evaluation_rollouts']}"
                                       if config["evaluation_rollouts"] > 1 else "accuracy")
    if args.gate_reference:
        reference = args.gate_reference.resolve()
        qualification, reference_config = read(reference / "verification.json"), read(reference / "config.json")
        assert qualification["passed_audit"] and qualification["qualified_for_gate"]
        evaluation_temperature = reference_config.get("evaluation_temperature", 0.0)
        evaluation_rollouts = reference_config.get("evaluation_rollouts", 1)
        if args.evaluation_temperature is not None:
            assert args.evaluation_temperature == evaluation_temperature
        if args.evaluation_rollouts is not None:
            assert args.evaluation_rollouts == evaluation_rollouts
        if "evaluation_rollouts" in reference_config or "evaluation_rollouts" in config:
            config.update(evaluation_temperature=evaluation_temperature,
                evaluation_rollouts=evaluation_rollouts,
                evaluation_metric=reference_config.get("evaluation_metric",
                    f"mean_at_{evaluation_rollouts}" if evaluation_rollouts > 1 else "accuracy"))
        assert config["data"] == reference_config["data"]
        assert config.get("parameter_dtype", "bfloat16") == reference_config.get("parameter_dtype", "bfloat16")
        for key in ["rounds", "rollouts_per_task", "learning_rate", "mh_iterations", "mh_proposals",
                    "assistant_token_budget", "policy_max_tokens", "stop_at_chat_end", "seed", "base_model"]:
            assert config[key] == reference_config[key], f"Unmatched comparison setting: {key}"
        config.update(gate_reference=str(reference), gate_version="math-soft-routing-v1",
            gate_proposal_mode="reference_pool",
            qualification_sha256=pipeline.sha(reference / "verification.json"),
            confirmation_split=None, final_comparison_split="test", gate_test_split_unseen=None,
            routing_mass={"weight_or_both": 2.0, "success_or_uncertain": 1.0,
                          "harness_scoring_or_environment": 0.25}, retain_all_verified_successes=True)
        if args.share_first_probe:
            from math_common_probe import VERSION
            config["gate_first_probe_mode"] = VERSION
    if (run / "config.json").exists():
        assert read(run / "config.json") == config
    else:
        pipeline.write_json(run / "config.json", config)
    started = time.monotonic()
    proposer_base = f"http://127.0.0.1:{args.proposer_port}"
    child_env = dict(os.environ, PYTHONPATH=str(ROOT / "whale_local") + ":" + str(DOMAIN),
        WHALE_PROPOSER_JOBS=str(run / "proposer_jobs"), WHALE_MH_RUNS=str(run / "mh"),
        WHALE_PROPOSER_PORT=str(args.proposer_port), NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost")
    with (run / "proposer.log").open("a") as log:
        proposer = subprocess.Popen([str(ROOT / ".venv-whale-portable/bin/python"), "-u",
            str(ROOT / "whale_local/proposer_api.py")], env=child_env, stdout=log, stderr=subprocess.STDOUT)
        pipeline.write_json(run / "proposer_service.json", {"pid": proposer.pid})
        try:
            pipeline.wait_ready(proposer, proposer_base)
            with service(run, base, args.gpu, args.policy_port, data, args.learning_rate,
                         args.parameter_dtype) as (base_url, current):
                initial = current
                pipeline.write_json(run / "initial_identity.json", initial)
                harness = BASE_HARNESS
                rounds = []
                for number in range(1, args.rounds + 1):
                    phase = run / f"round_{number:02d}"
                    phase.mkdir(exist_ok=True)
                    checkpoint = phase / "checkpoint"
                    if not (checkpoint / "training_state.json").exists():
                        if not (phase / "rollouts/val.json").exists():
                            print(f"ROUND {number}: sample native ReTool Math rollouts", flush=True)
                            if args.gate_reference:
                                from math_gate import collect
                                collect(reference=args.gate_reference, phase=phase, base_url=base_url,
                                    identity=current, harness=harness, data_entry=data["splits"]["train"],
                                    seed=42 + number, total_rollouts=data["splits"]["train"]["count"] * args.rollouts,
                                    proposer_base=proposer_base, resume=args.resume)
                            else:
                                hardware = read(run / "eval_hardware.json") if (run / "eval_hardware.json").exists() else {}
                                if hardware.get("sampler_gpus"):
                                    from math_sampling import sample_parallel
                                    sample_parallel(run=run, phase=phase, output=phase / "rollouts",
                                        base_url=base_url, identity=current, harness=harness,
                                        data_entry=data["splits"]["train"], rollouts=args.rollouts,
                                        seed=42 + number, resume=args.resume)
                                else:
                                    asyncio.run(evaluate(base_url=base_url, identity=current, harness=harness,
                                        data_entry=data["splits"]["train"], output=phase / "rollouts",
                                        rollouts=args.rollouts, temperature=0.8, seed=42 + number,
                                        resume=args.resume))
                        pipeline.api(base_url, "/rsft/train", {"round": number,
                            "rollouts": str(phase / "rollouts/trajectories.jsonl"),
                            "train_data": data["splits"]["train"]["path"], "harness": str(harness),
                            "expected_weight_digest": current["weight_digest"]}, timeout=7200)
                    training = read(checkpoint / "training_state.json")
                    assert training["sampling_identity"]["weight_digest"] == current["weight_digest"]
                    assert training["harness_sha256"] == pipeline.sha(harness)
                    current = pipeline.api(base_url, "/checkpoint/load", {"checkpoint": str(checkpoint)}, timeout=240)
                    assert current["weight_digest"] == training["weight_digest"]
                    mh_dir = run / "mh" / f"round_{number:02d}"
                    mh_config = {"meta": {"proposer_model": "active-codex", "proposer_effort": "high",
                                          "proposals_per_iter": args.mh_proposals, "propose_timeout_s": 3600},
                        "task_profiles": {"mh_val": {}}, "models": [{"model": current["model"], "api_base": base_url + "/v1"}],
                        "seeds": [42], "eval": {"enable_thinking": False},
                        "local": {"policy_base": base_url, "weight_digest": current["weight_digest"], "data_entry": data["splits"]["mh_val"]}}
                    pipeline.write_json(phase / "mh_config.json", mh_config)
                    if not (phase / "mh_result.json").exists():
                        completed = len(list((mh_dir / "logs").glob("iteration_*/comparison.json")))
                        remaining = max(0, args.mh_iterations - completed)
                        if remaining:
                            with (phase / "mh.log").open("a") as mh_log:
                                process = subprocess.Popen([sys.executable, "-u", str(ROOT / "whale_local/run_math_mh.py"),
                                    "--run-name", f"round_{number:02d}", "--config", str(phase / "mh_config.json"),
                                    "--iterations", str(args.mh_iterations), "--proposals-per-iter", str(args.mh_proposals)],
                                    env=child_env | {"BASELINE_HARNESS_OVERRIDE": str(harness)}, stdout=mh_log, stderr=subprocess.STDOUT)
                                try:
                                    if process.wait():
                                        raise RuntimeError(f"Native MH round {number} failed")
                                finally:
                                    if process.poll() is None:
                                        process.terminate()
                                        process.wait(timeout=30)
                        selected = (mh_dir / "logs/accepted_harness.txt").read_text().strip()
                        chosen = mh_dir / "harnesses" / selected / "harness.py"
                        evaluations = list((mh_dir / "logs/mh_val").rglob("val.json"))
                        assert len(evaluations) >= 1 + args.mh_iterations * args.mh_proposals
                        pipeline.write_json(phase / "mh_result.json", {"model_identity": current,
                            "harness_path": str(chosen), "harness_sha256": pipeline.sha(chosen),
                            "selected": selected, "evaluation_files": [str(p) for p in evaluations],
                            "frontier": read(mh_dir / "logs/frontier_val.json")})
                    mh_result = read(phase / "mh_result.json")
                    assert mh_result["model_identity"]["weight_digest"] == current["weight_digest"]
                    harness = Path(mh_result["harness_path"])
                    assert pipeline.sha(harness) == mh_result["harness_sha256"]
                    rounds.append({"round": number, "model_identity": current, "harness": str(harness),
                                   "accepted_trajectories": training["accepted_trajectories"]})
                    pipeline.write_json(run / "progress.json", {"status": "running", "rounds": rounds})
                frozen_final = current
                pipeline.write_json(run / "frozen_final.json", {"model_identity": current,
                    "harness_path": str(harness), "harness_sha256": pipeline.sha(harness)})
                # First and only confirmation-set use, after both search rounds.
                from math_evaluation import evaluate_four_cells
                if args.gate_reference:
                    from math_evaluation import evaluate_gate_comparison
                    evaluate_gate_comparison(run=run, config=config, base_url=base_url,
                        initial=initial, final=current, final_harness=harness, resume=args.resume)
                else:
                    evaluate_four_cells(run=run, config=config, split="confirmation", base_url=base_url,
                        initial=initial, final=current, final_harness=harness, resume=args.resume)
                assert pipeline.api(base_url, "/health")["weight_digest"] == frozen_final["weight_digest"]
                pipeline.write_json(run / "summary.json", {"status": "complete", "rounds": rounds,
                    "initial_identity": initial, "final_identity": frozen_final, "final_harness": str(harness),
                    "gate_enabled": bool(args.gate_reference), "elapsed_seconds": time.monotonic() - started})
        finally:
            if proposer.poll() is None:
                proposer.send_signal(signal.SIGTERM)
                try:
                    proposer.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proposer.kill()
                    proposer.wait()
            pipeline.write_json(run / "proposer_stopped.json", {"pid": proposer.pid, "returncode": proposer.returncode})


if __name__ == "__main__":
    main()
