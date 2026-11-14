"""Explicit preflight, paired retesting, and bounded LoRA pilot entry point."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from .core import (
    Protocol, REFERENCE, collect_episode, compare_reports, digest, evaluation_report,
    parse_action, training_rows,
)
from .runtime import (
    HFPolicy, WebShopEnvironment, check_lease, implementation_identity,
    model_identity, preflight, process_identity, write_json,
)


def optimize_group(policy, episodes, protocol, method, optimizer):
    """One accumulation/update over a complete task group; no teacher gradients."""
    import torch
    from .objective import policy_loss
    rows, statistics = training_rows(
        episodes, policy, protocol, use_distillation=method == "success_sd",
    )
    if any(not e.turns for e in episodes):
        raise ValueError("An empty rollout cannot be silently removed from a GRPO group")
    # Freeze all hindsight scores before the first student update.
    teacher_scores = [
        policy.log_probs(row.teacher_prompt_ids, row.response_ids).detach()
        if row.teacher_prompt_ids is not None else None
        for row in rows
    ]
    reference_scores = [
        policy.reference_log_probs(row.student_prompt_ids, row.response_ids).detach()
        if protocol.reference_kl_coefficient else None
        for row in rows
    ]
    counts = [
        sum(len(t.sample.token_ids) for t in episode.turns)
        for episode in episodes
    ]
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    teacher_tokens, teacher_gap_absolute = 0, 0.0
    for row, teacher, reference in zip(rows, teacher_scores, reference_scores):
        student = policy.log_probs(row.student_prompt_ids, row.response_ids, grad=True)
        old = torch.tensor(row.old_log_probs, device=student.device)
        if teacher is not None:
            teacher_tokens += teacher.numel()
            teacher_gap_absolute += float((teacher - old).abs().sum())
        response_mask = torch.ones_like(student, dtype=torch.bool)[None, :]
        loss, _metrics = policy_loss(
            student[None, :], old[None, :], response_mask,
            torch.tensor([row.advantage], device=student.device),
            teacher_log_probs=teacher[None, :] if teacher is not None else None,
            teacher_mask=response_mask if teacher is not None else None,
            coefficient=protocol.distill_coefficient if method == "success_sd" else 0.0,
            gap_clip=protocol.gap_clip, ppo_clip=protocol.ppo_clip,
            reference_log_probs=reference[None, :] if reference is not None else None,
            reference_kl_coefficient=protocol.reference_kl_coefficient,
        )
        weight = len(row.response_ids) / counts[row.episode_index] / len(episodes)
        (loss * weight).backward()
        total_loss += float(loss.detach()) * weight
    parameters = [p for p in policy.model.parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0, error_if_nonfinite=True)
    optimizer.step()
    return {
        **statistics, "loss": total_loss, "gradient_norm": float(norm), "optimizer_steps": 1,
        "teacher_tokens": teacher_tokens,
        "teacher_gap_abs_mean": teacher_gap_absolute / teacher_tokens if teacher_tokens else 0.0,
    }


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    check = commands.add_parser("preflight", help="CPU file/dependency checks only")
    check.add_argument("--env-root", type=Path, required=True)
    check.add_argument("--model", type=Path)
    check.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("compare", help="Offline paired result comparison")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--improved", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    environment = commands.add_parser("environment-check", help="CPU-only real environment reset/action check")
    environment.add_argument("--config", type=Path, required=True)
    environment.add_argument("--env-root", type=Path, required=True)
    environment.add_argument("--task-id", type=int, default=0)
    environment.add_argument("--action", default="search[shoes]")
    environment.add_argument("--output", type=Path, required=True)
    for name in ("train", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--env-root", type=Path, required=True)
        command.add_argument("--model", type=Path, required=True)
        command.add_argument("--lease", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "train":
            command.add_argument("--method", choices=("grpo", "success_sd"), required=True)
        else:
            command.add_argument("--adapter", type=Path)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.output.exists():
        raise ValueError("Output already exists; preserve previous experimental evidence")
    if args.command == "preflight":
        result = preflight(args.env_root.resolve(), args.model.resolve() if args.model else None)
        write_json(args.output, result)
        print(json.dumps(result, indent=2))
        return 0 if result["ready"] else 2
    if args.command == "compare":
        result = compare_reports(
            json.loads(args.baseline.read_text()), json.loads(args.improved.read_text()),
        )
        write_json(args.output, result)
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "environment-check":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("CPU environment check requires explicit CUDA_VISIBLE_DEVICES=''")
        protocol = Protocol.from_dict(json.loads(args.config.read_text()))
        if args.task_id not in (*protocol.train_ids, *protocol.eval_ids):
            raise ValueError("Environment-check task is absent from the fixed protocol")
        if parse_action(args.action) is None:
            raise ValueError("Environment-check action must be a single valid search/click")
        prerequisites = preflight(args.env_root.resolve())
        if not prerequisites["ready"]:
            raise RuntimeError("Preflight failed: " + json.dumps(prerequisites))
        env = WebShopEnvironment(args.env_root, protocol)
        try:
            repeats = []
            for _ in range(2):
                initial, _instruction, manifest = env.reset(args.task_id)
                observation, score, done, info = env.step(args.action)
                repeats.append({
                    "task_manifest": manifest, "initial_observation": initial,
                    "action": args.action, "observation": observation, "task_score": score,
                    "terminated": done, "invalid": bool(info.get("invalid", False)),
                    "available_actions": env.available_actions(),
                })
            if digest(repeats[0]) != digest(repeats[1]):
                raise RuntimeError("Repeated real environment reset/action is not deterministic")
            if repeats[0]["invalid"] or repeats[0]["initial_observation"] == repeats[0]["observation"]:
                raise RuntimeError("Action did not produce a valid environment transition")
            result = {
                "schema": "coevolve_webshop_environment_check_v1",
                "healthy": True, "deterministic_repeats": 2, "assets": env.assets,
                "transition": repeats[0], "goal_seed": protocol.goal_seed,
                "model_used": False, "gpu_used": False, "http_service_started": False,
                "interpretation": "Real in-process WebShop SimServer/search check; not a model benchmark.",
            }
            write_json(args.output, result)
            print(json.dumps({key: value for key, value in result.items()
                              if key not in {"assets", "transition"}}, indent=2))
        finally:
            env.close()
        return 0
    protocol = Protocol.from_dict(json.loads(args.config.read_text()))
    episodes_needed = (
        protocol.updates * protocol.group_size if args.command == "train"
        else len(protocol.eval_ids) * len(protocol.eval_seeds)
    )

    def guard():
        return check_lease(args.lease, mode=args.command, episodes=episodes_needed)

    guard()
    prerequisites = preflight(args.env_root.resolve(), args.model.resolve())
    if not prerequisites["ready"]:
        raise RuntimeError("Preflight failed: " + json.dumps(prerequisites))
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "PROCESS.json", process_identity())
    wall_started = time.monotonic()
    env = WebShopEnvironment(args.env_root, protocol)
    try:
        guard()
        policy = HFPolicy(args.model.resolve(), protocol.seed, train=args.command == "train",
                          adapter=getattr(args, "adapter", None), guard=guard,
                          enable_thinking=protocol.enable_thinking)
        started = time.monotonic()
        if args.command == "train":
            import torch
            optimizer = torch.optim.AdamW(
                (p for p in policy.model.parameters() if p.requires_grad),
                lr=protocol.learning_rate, weight_decay=0.0,
            )
            metrics = []
            for update in range(protocol.updates):
                task = protocol.train_ids[update % len(protocol.train_ids)]
                group = []
                for member in range(protocol.group_size):
                    guard()
                    group.append(collect_episode(
                        env, policy, protocol, task, "train",
                        protocol.seed + update * 100000 + member * protocol.max_steps,
                        policy_version=update, temperature=1.0,
                    ))
                write_json(args.output / f"rollouts-{update:04d}.json",
                           [e.as_dict() for e in group])
                guard()
                statistics = optimize_group(policy, group, protocol, args.method, optimizer)
                metrics.append({"update": update, "task_id": task, **statistics})
                write_json(args.output / "metrics.json", metrics)
            policy.model.save_pretrained(args.output / "adapter")
            policy.tokenizer.save_pretrained(args.output / "adapter")
            write_json(args.output / "TRAINING.json", {
                "schema": "coevolve_webshop_training_v1", "reference": REFERENCE,
                "method": args.method, "protocol": protocol.as_dict(),
                "base_model": str(args.model.resolve()), "assets": env.assets,
                "base_model_identity": policy.model_identity,
                "adapter_identity": model_identity(args.output / "adapter"),
                "implementation_identity": implementation_identity(),
                "metrics": metrics, "elapsed_seconds": time.monotonic() - started,
                "total_wall_seconds": time.monotonic() - wall_started,
                "runtime": policy.runtime_metrics() if hasattr(policy, "runtime_metrics") else {},
                "trainable_parameters": sum(
                    p.numel() for p in policy.model.parameters() if p.requires_grad
                ),
                "note": "Final LoRA inference adapter, not an optimizer/RNG recovery checkpoint.",
            })
        else:
            training = None
            if args.adapter is not None:
                training_path = args.adapter.resolve().parent / "TRAINING.json"
                training = json.loads(training_path.read_text())
                if training["base_model"] != str(args.model.resolve()):
                    raise ValueError("Adapter was trained against a different base model")
                if training["protocol"] != json.loads(json.dumps(protocol.as_dict())):
                    raise ValueError("Adapter training/evaluation protocol mismatch")
                if training["assets"] != env.assets:
                    raise ValueError("Adapter benchmark asset identity mismatch")
                if training["base_model_identity"] != policy.model_identity:
                    raise ValueError("Base model contents changed since training")
                if training["implementation_identity"] != implementation_identity():
                    raise ValueError("Pilot implementation changed since training")
                if training["adapter_identity"] != model_identity(args.adapter):
                    raise ValueError("Adapter contents changed since training")
            episodes = []
            for seed in protocol.eval_seeds:
                for task in protocol.eval_ids:
                    guard()
                    episode = collect_episode(
                        env, policy, protocol, task, "eval", seed,
                        temperature=protocol.eval_temperature,
                    )
                    episodes.append(episode)
                    write_json(args.output / f"task-{task:04d}-seed-{seed}.json", episode.as_dict())
            report = evaluation_report(
                episodes, protocol, str(args.adapter or args.model), env.assets, training,
            )
            report["elapsed_seconds"] = time.monotonic() - started
            report["total_wall_seconds"] = time.monotonic() - wall_started
            report["runtime"] = policy.runtime_metrics() if hasattr(policy, "runtime_metrics") else {}
            report["base_model_identity"] = policy.model_identity
            report["implementation_identity"] = implementation_identity()
            report["adapter_identity"] = model_identity(args.adapter) if args.adapter else None
            write_json(args.output / "REPORT.json", report)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
