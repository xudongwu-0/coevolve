"""Run two or more real WHALE weight/harness alternations on a local GPU.

Resume skips atomically committed checkpoints and completed MH stages. During
each MH stage the active Codex conversation must answer the FastAPI job queue.
"""
from pathlib import Path
from urllib.request import Request, build_opener, ProxyHandler
import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DOMAIN = ROOT / "WHALE/domains/chess_puzzles"
POLICY = "http://127.0.0.1:18541"
PROPOSER = "http://127.0.0.1:18542"
opener = build_opener(ProxyHandler({}))


def api(base, endpoint, payload=None, timeout=30):
    request = Request(base + endpoint,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wait_ready(process, base, timeout=240):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Service {base} exited with {process.returncode}; inspect worker/proposer.log")
        try:
            result = api(base, "/health", timeout=2)
            if result["ready"]:
                return result
        except (OSError, ValueError):
            pass
        time.sleep(1)
    raise TimeoutError(f"Service startup timeout: {base}")


def llm_config(identity):
    return dict(provider="openai-compatible", model=identity["model"], api_base=POLICY + "/v1",
                temperature=0, max_tokens=128, batch_concurrency=1,
                chat_template_kwargs={"enable_thinking": False})


def main():
    global POLICY, PROPOSER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--mh-iterations", type=int, default=1,
                        help="Native dynamic harness search iterations after each weight update")
    parser.add_argument("--mh-proposals", type=int, default=1,
                        help="Independent candidate mechanisms in each native MH iteration")
    parser.add_argument("--gpu", default="6")
    parser.add_argument("--rollouts-per-puzzle", type=int, default=4)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--data-manifest", type=Path, default=ROOT / "runs/whale_setup/data/manifest.json")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--gate-mode", choices=["counterfactual", "teacher"], default="counterfactual")
    parser.add_argument("--teacher-audit-pairs", type=int, default=2)
    parser.add_argument("--teacher-equal-budget", action="store_true",
                        help="Allocate exactly N*G small-model rollouts per round, including audits")
    parser.add_argument("--gate-probe-budget", type=int, default=16)
    parser.add_argument("--gate-resamples", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--matched-sampling", action="store_true",
                        help="Use the gate's per-trajectory seeds for the ungated control too")
    parser.add_argument("--policy-port", type=int, default=18541)
    parser.add_argument("--proposer-port", type=int, default=18542)
    parser.add_argument("--ablation", action="store_true",
                        help="Evaluate final weights with the base harness and vice versa on the same held-out set")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.mh_iterations <= 20 or not 1 <= args.mh_proposals <= 8:
        parser.error("mh-iterations must be in [1, 20]; mh-proposals in [1, 8]")
    if args.gate_mode == "teacher" and not args.gate:
        parser.error("--gate-mode teacher requires --gate")
    if args.teacher_equal_budget and not (args.gate and args.gate_mode == "teacher"):
        parser.error("--teacher-equal-budget requires --gate --gate-mode teacher")
    if not 0 <= args.teacher_audit_pairs <= 8:
        parser.error("teacher-audit-pairs must be in [0, 8]")
    if args.policy_port == args.proposer_port or not all(1024 <= p <= 65535 for p in (args.policy_port, args.proposer_port)):
        parser.error("policy/proposer ports must be different unprivileged ports")
    POLICY = f"http://127.0.0.1:{args.policy_port}"
    PROPOSER = f"http://127.0.0.1:{args.proposer_port}"
    if not 0 < args.learning_rate <= 1e-3:
        parser.error("learning-rate must be positive and no greater than 1e-3")
    run = args.run_dir.resolve()
    if run.exists() and any(run.iterdir()) and not args.resume:
        raise FileExistsError("Use a new run directory or --resume; existing evidence is never overwritten")
    run.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    base = args.base_model or Path((ROOT / "runs/whale_setup/base_model_path.txt").read_text().strip())
    data = json.loads(args.data_manifest.read_text())
    for split in data["splits"].values():
        assert sha(split["path"]) == split["sha256"]
    config = {"backend": "local Transformers + unmodified WHALE runner, RSFT actor and MH selection",
        "base_model": str(base), "base_revision": base.name,
        "upstream_commit": subprocess.check_output(["git", "-C", str(ROOT / "WHALE"), "rev-parse", "HEAD"], text=True).strip(),
        "rounds": args.rounds, "gpu": args.gpu, "rollouts_per_puzzle": args.rollouts_per_puzzle,
        "data": data, "learning_rate": args.learning_rate, "sft_epochs": 1,
        "proposer": "active Codex agent via localhost FastAPI queue",
        "assistant_token_budget": 256, "policy_max_tokens": 128, "seed": args.seed}
    if args.seed != 42:
        config["evaluation_seed"] = 42
    if args.matched_sampling:
        config["sampling_rng"] = "per_trajectory_common_seed"
    if (args.policy_port, args.proposer_port) != (18541, 18542):
        config["ports"] = {"policy": args.policy_port, "proposer": args.proposer_port}
    if args.ablation:
        config["heldout_ablation"] = True
    if (args.mh_iterations, args.mh_proposals) != (1, 1):
        config["harness_search"] = {"iterations_per_round": args.mh_iterations,
                                    "proposals_per_iteration": args.mh_proposals}
    if args.gate:
        probes = sorted((ROOT / "whale_local/gate_harnesses").glob("*.py"), reverse=True)
        config["gate"] = {"version": "counterfactual-v1", "probe_budget": args.gate_probe_budget,
                          "resamples": args.gate_resamples,
                          "probes": [{"path": str(p), "sha256": sha(p)} for p in probes]}
        if args.gate_mode == "teacher":
            config["gate"] = {"version": "teacher-v2", "audit_pairs": args.teacher_audit_pairs,
                              "allocation": "one initial rollout per task; teacher selects followups",
                              "probes": [{"path": str(p), "sha256": sha(p)} for p in probes]}
            if args.teacher_equal_budget:
                config["gate"].update(version="teacher-v2-equal-budget",
                                      allocation="exact N*G small-model rollouts including audits; teacher-directed followups")
    config_path = run / "config.json"
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, "Resume configuration changed"
    else:
        write_json(config_path, config)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "whale_local") + ":" + str(DOMAIN),
        CUDA_VISIBLE_DEVICES=args.gpu, WHALE_RUN_ROOT=str(run), WHALE_MODEL_ROOT=str(base),
        WHALE_PROPOSER_JOBS=str(run / "proposer_jobs"), WHALE_MH_RUNS=str(run / "mh"),
        CHESS_PUZZLE_PROPOSER_RETRIES="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        WHALE_LEARNING_RATE=str(args.learning_rate),
        WHALE_POLICY_PORT=str(args.policy_port), WHALE_PROPOSER_PORT=str(args.proposer_port),
        TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    # Do not silently attach to any pre-existing service on these ports.
    for port in (args.policy_port, args.proposer_port):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
    processes, handles = [], []
    try:
        for script, log in (("worker.py", "worker.log"), ("proposer_api.py", "proposer.log")):
            handle = (run / log).open("a")
            handles.append(handle)
            processes.append(subprocess.Popen([sys.executable, "-u", str(ROOT / "whale_local" / script)],
                env=env, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT))
        write_json(run / "services.json", {"worker_pid": processes[0].pid, "proposer_pid": processes[1].pid})
        current = wait_ready(processes[0], POLICY)
        wait_ready(processes[1], PROPOSER)
        write_json(run / "initial_identity.json", current)
        # A stale model name must be rejected, not silently served using old weights.
        from urllib.error import HTTPError
        try:
            api(POLICY, "/v1/chat/completions", {"model": "stale-checkpoint", "messages": [{"role": "user", "content": "test"}]})
            raise AssertionError("Worker accepted a stale model identity")
        except HTTPError as exc:
            assert exc.code == 409
        write_json(run / "identity_guard.json", {"stale_checkpoint_rejected": True, "status": 409})
        from autoharness_chess_puzzle.runner import evaluate_harness
        baseline = DOMAIN / "environments/chess_puzzle/base_harness.py"
        # Fixed held-out baseline; never used by the proposer or for model selection.
        if not args.ablation and not (run / "test_initial/val.json").exists():
            evaluate_harness(harness_path=baseline, dataset_path=data["splits"]["test"]["path"],
                llm_config=llm_config(current), output_dir=run / "test_initial", limit=data["splits"]["test"]["count"],
                seed=42, assistant_token_budget=256, policy_max_tokens=128)
            write_json(run / "test_initial/identity.json", current | {"harness_sha256": sha(baseline)})
        harness = baseline
        rounds = []
        for number in range(1, args.rounds + 1):
            phase = run / f"round_{number:02d}"
            phase.mkdir(exist_ok=True)
            checkpoint = phase / "checkpoint"
            if not (checkpoint / "training_state.json").exists():
                print(f"ROUND {number}: sample and update {current['model']} with {harness}", flush=True)
                gate_request = {}
                if args.gate:
                    probe = next((p for p in probes if sha(p) != sha(harness)), None)
                    assert probe is not None, "Gate needs a different harness intervention"
                    gate_request = {"gate_probe_harness": str(probe),
                                    "gate_probe_budget": args.gate_probe_budget,
                                    "gate_resamples": args.gate_resamples,
                                    "gate_mode": args.gate_mode, "teacher_audit_pairs": args.teacher_audit_pairs,
                                    "teacher_equal_budget": args.teacher_equal_budget}
                api(POLICY, "/rsft/train", {"round": number, "harness": str(harness),
                    "train_data": data["splits"]["train"]["path"], "expected_weight_digest": current["weight_digest"],
                    "rollouts_per_puzzle": args.rollouts_per_puzzle, "seed": args.seed + number,
                    "matched_sampling": args.matched_sampling,
                    **gate_request}, timeout=7200)
            training = json.loads((checkpoint / "training_state.json").read_text())
            assert training["sampling_identity"]["weight_digest"] == current["weight_digest"]
            assert training["harness_sha256"] == sha(harness)
            current = api(POLICY, "/checkpoint/load", {"checkpoint": str(checkpoint)}, timeout=240)
            assert current["weight_digest"] == training["weight_digest"]
            assert current["optimizer_steps"] == training["optimizer_steps_after"]
            mh_dir = run / "mh" / f"round_{number:02d}"
            mh_cfg = dict(meta={"proposer_model": "active-codex-agent", "propose_timeout_s": 3600,
                                "proposals_per_iter": args.mh_proposals},
                task_profiles={"mh_val": {"dataset_path": data["splits"]["mh_val"]["path"], "limit": data["splits"]["mh_val"]["count"]}},
                models=[llm_config(current)], seeds=[42],
                eval={"assistant_token_budget": 256, "policy_max_tokens": 128})
            write_json(phase / "mh_config.json", mh_cfg)
            done = phase / "mh_result.json"
            if args.gate and not training["gate"]["harness_search_enabled"] and not done.exists():
                write_json(done, {"model_identity": current, "input_harness_sha256": sha(harness),
                    "selected": "unchanged", "harness_path": str(harness), "harness_sha256": sha(harness),
                    "frontier": {}, "evaluation_files": [], "skipped_by_gate": True,
                    "reason": "no harness-recoverable training failure within probe budget"})
            if not done.exists():
                mh_env = env | {"BASELINE_HARNESS_OVERRIDE": str(harness)}
                if args.gate:
                    mh_env["WHALE_GATE_CONTEXT"] = str(phase / "gate_harness_context.json")
                # On an interrupted MH stage, reuse existing evaluated candidates;
                # choose another proposal slot rather than overwriting artifacts.
                with (phase / "mh.log").open("a") as handle:
                    previous_iterations = [int(p.name.split("_")[-1]) for p in (mh_dir / "logs").glob("iteration_*")]
                    start_iteration = max(previous_iterations, default=0) + 1
                    completed = len(list((mh_dir / "logs").glob("iteration_*/comparison.json")))
                    remaining = max(0, args.mh_iterations - completed)
                    if remaining:
                        child = subprocess.Popen([sys.executable, "-u", str(ROOT / "whale_local/run_chess_mh.py"),
                            "--run-name", f"round_{number:02d}", "--config", str(phase / "mh_config.json"),
                            "--iterations", str(remaining), "--start-iteration", str(start_iteration),
                            "--proposals-per-iter", str(args.mh_proposals), "--propose-timeout", "3600"],
                            env=mh_env, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
                        processes.append(child)
                        code = child.wait()
                        if code:
                            raise RuntimeError(f"MH round {number} failed: exit {code}")
                errors = list((mh_dir / "logs").rglob("error.txt"))
                if errors:
                    raise RuntimeError(f"MH evaluation errors: {errors}")
                selected = (mh_dir / "logs/accepted_harness.txt").read_text().strip()
                chosen = mh_dir / "harnesses" / selected / "harness.py"
                evaluations = list((mh_dir / "logs/mh_val").rglob("val.json"))
                assert len(evaluations) >= 2, "Both baseline and candidate must be genuinely evaluated"
                assert api(POLICY, "/health")["weight_digest"] == current["weight_digest"]
                write_json(done, {"model_identity": current, "input_harness_sha256": sha(harness),
                    "selected": selected, "harness_path": str(chosen), "harness_sha256": sha(chosen),
                    "frontier": json.loads((mh_dir / "logs/frontier_val.json").read_text()),
                    "evaluation_files": [str(p) for p in evaluations]})
            result = json.loads(done.read_text())
            assert result["model_identity"]["weight_digest"] == current["weight_digest"]
            harness = Path(result["harness_path"])
            assert sha(harness) == result["harness_sha256"]
            rounds.append({"round": number, "weight_digest": current["weight_digest"],
                "optimizer_steps": current["optimizer_steps"], "accepted_trajectories": training["accepted_trajectories"],
                "sampled_trajectories": training["rollouts"], "sft_updates": training["metrics"]["actor/sft_updates"],
                "changed_parameter_tensors": len(training["changed_parameter_tensors"]),
                "selected_harness": str(harness), "harness_sha256": sha(harness),
                "gate": training.get("gate"), "update_status": training.get("update_status", "updated"),
                "mh_skipped_by_gate": result.get("skipped_by_gate", False)})
            write_json(run / "progress.json", {"completed_rounds": rounds, "status": "running"})
            print(f"ROUND {number} COMPLETE: {current['model']} + {harness}", flush=True)
        if not (run / "test_final/val.json").exists():
            evaluate_harness(harness_path=harness, dataset_path=data["splits"]["test"]["path"],
                llm_config=llm_config(current), output_dir=run / "test_final", limit=data["splits"]["test"]["count"],
                seed=42, assistant_token_budget=256, policy_max_tokens=128)
            write_json(run / "test_final/identity.json", current | {"harness_sha256": sha(harness)})
        if args.ablation:
            # The final pair is frozen before any ablation. These results never
            # enter MH, the gate, checkpoint selection, or another training round.
            ablations = [("test_weights_only", current, baseline)]
            for name, model_identity, test_harness in ablations:
                if not (run / name / "val.json").exists():
                    evaluate_harness(harness_path=test_harness, dataset_path=data["splits"]["test"]["path"],
                        llm_config=llm_config(model_identity), output_dir=run / name,
                        limit=data["splits"]["test"]["count"], seed=42,
                        assistant_token_budget=256, policy_max_tokens=128)
                    write_json(run / name / "identity.json", model_identity | {"harness_sha256": sha(test_harness)})
            if any(not (run / name / "val.json").exists() for name in ("test_initial", "test_harness_only")):
                base_identity = api(POLICY, "/checkpoint/load", {"checkpoint": str(base)}, timeout=240)
                assert base_identity["weight_digest"] == json.loads((run / "initial_identity.json").read_text())["weight_digest"]
                for name, test_harness in (("test_initial", baseline), ("test_harness_only", harness)):
                    if not (run / name / "val.json").exists():
                        evaluate_harness(harness_path=test_harness, dataset_path=data["splits"]["test"]["path"],
                            llm_config=llm_config(base_identity), output_dir=run / name,
                            limit=data["splits"]["test"]["count"], seed=42,
                            assistant_token_budget=256, policy_max_tokens=128)
                        write_json(run / name / "identity.json", base_identity | {"harness_sha256": sha(test_harness)})
        initial = json.loads((run / "test_initial/val.json").read_text())
        final = json.loads((run / "test_final/val.json").read_text())
        write_json(run / "summary.json", {"status": "complete", "scope": config["backend"], "rounds": rounds,
            "test_initial": initial, "test_final": final, "final_checkpoint": current["weights"],
            "final_harness": str(harness), "continuous_optimizer": True,
            "checkpoint_reload_verified_every_round": True, "distributed_launcher_tested": False,
            "elapsed_seconds_this_invocation": time.monotonic() - started,
            "gate_enabled": args.gate, "heldout_ablation": args.ablation})
        write_json(run / "progress.json", {"completed_rounds": rounds, "status": "complete"})
        print(f"COMPLETE: {run / 'summary.json'}", flush=True)
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for handle in handles:
            handle.close()
        write_json(run / "services_stopped.json", {"processes": [{"pid": p.pid, "returncode": p.returncode} for p in processes]})


if __name__ == "__main__":
    main()
