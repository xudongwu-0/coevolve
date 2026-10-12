"""Atomic, paired native recovery points without restarting healthy learners."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
from types import SimpleNamespace

from .alfworld_train_data import _immutable_json
from .common import file_sha, read_json, write_json
from .joint_health import metric_rows
from .joint_round import ROLES, check_failures
from .joint_schedule import checkpoint_metadata, update_counts, validate_checkpoint_metadata


SCHEMA = "paired-native-recovery-v1"
GENERATION = re.compile(r"round-[0-9]{4}-attempt-[0-9]{4}")
OPERATIONAL_FIELDS = {
    "run", "joint_root", "pinned_files", "rolling_recovery", "recovery_start_round",
}


def science_digest(spec: dict) -> str:
    value = {key: child for key, child in spec.items() if key not in OPERATIONAL_FIELDS}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_recovery_spec(spec: dict) -> None:
    options = spec.get("rolling_recovery")
    start = spec.get("recovery_start_round", 0)
    if options is None:
        if start:
            raise ValueError("A nonzero collection cursor requires explicit paired recovery")
        return
    finalizing = isinstance(options, dict) and options.get("finalize_only", False)
    if (
        not isinstance(options, dict) or not spec.get("joint_root")
        or spec.get("preserve_optimizer_state", False)
        or options.get("interval_rounds") != 12
        or not re.fullmatch(r"attempt-[0-9]{4}", options.get("attempt", ""))
        or type(finalizing) is not bool
        or type(start) is not int or not 0 <= start <= spec["steps"]
        or (start % 12 and not finalizing)
        or finalizing != (start == spec["steps"])
        or bool(options.get("resume")) != bool(start)
        or not Path(options.get("root", "")).is_absolute()
    ):
        raise ValueError("Invalid twelve-round paired recovery configuration")
    milestones = options.get("hf_rounds")
    if (
        not isinstance(milestones, list) or not milestones
        or any(type(step) is not int or not 0 < step <= spec["steps"] for step in milestones)
        or milestones != sorted(set(milestones)) or milestones[-1] != spec["steps"]
    ):
        raise ValueError("Recovery needs explicit, ordered HF evaluation milestones")


def generation_path(root: Path, name: str) -> Path:
    if GENERATION.fullmatch(name) is None:
        raise ValueError("Recovery generation is not an explicitly owned round/attempt directory")
    path = root / name
    if path.is_symlink() or path.resolve().parent != root.resolve():
        raise ValueError("Recovery generation escaped its owned parent")
    return path


def file_inventory(root: Path, *, allow_empty: bool = False) -> dict:
    entries = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Native recovery state cannot contain symlinks")
        if path.is_file():
            size = path.stat().st_size
            if not size and not allow_empty:
                raise ValueError(f"Empty native recovery file: {path}")
            entries[str(path.relative_to(root))] = {
                "bytes": size,
                **({"sha256": file_sha(path)} if size <= 2**20 else {}),
            }
    return entries


def validate_inventory(root: Path, expected: dict) -> None:
    if not expected or file_inventory(root) != expected:
        raise ValueError(f"Recovery file inventory or metadata changed: {root}")


def validate_role_record(path: Path, *, expected_round: int, role: str) -> dict:
    record = read_json(path)
    if (
        record.get("schema") != SCHEMA or record.get("role") != role
        or record.get("completed_rounds") != expected_round
        or record.get("committed_sample_cursor") != expected_round * record["batch_size"]
    ):
        raise ValueError("Recovery role or committed cursor does not match its paired round")
    payload = path.parent / role
    validate_inventory(payload, record["files"])
    spec = read_json(payload / "spec.json")
    if science_digest(spec) != record["science_sha256"]:
        raise ValueError("Recovery scientific settings changed")
    rows = metric_rows(payload / "metrics.jsonl")
    if [row["global_step"] for row in rows] != list(range(expected_round)):
        raise ValueError("Recovery omitted committed metric rows")
    counts = update_counts(spec, expected_round)
    for engine in ("actor", "critic"):
        state = payload / engine
        validate_checkpoint_metadata(state, spec, expected_round)
        ranks = spec.get("actor_gpus", 4)
        if not (state / ".metadata").is_file() or len(list(state.glob("*.distcp"))) != ranks:
            raise ValueError("Incomplete distributed model shards")
        for rank in range(ranks):
            fingerprint = read_json(state / f"adam-rank{rank}.json")
            if (
                fingerprint["max_adam_step"] != counts[f"{engine}_steps"]
                or fingerprint["min_adam_step"] <= 0
                or not (state / f"optim_world_size_{ranks}_rank_{rank}.pt").is_file()
                or not (state / f"training-rank{rank}.pt").is_file()
            ):
                raise ValueError("Missing native Adam/RNG state or incorrect optimizer step")
        if engine == "critic":
            settings = read_json(state / "joint_critic_settings.json")
            if settings != {
                "cooperative_context": spec.get("critic_cooperative_context", True),
                "value_normalization": spec.get("value_normalization", False),
            }:
                raise ValueError("Recovery changed Critic settings")
            norm = state / "joint_value_norm.json"
            if spec.get("value_normalization", False):
                if read_json(norm)["updates"] != counts["critic_steps"]:
                    raise ValueError("Recovery lost ValueNorm update history")
            elif norm.exists():
                raise ValueError("ValueNorm ablation unexpectedly saved normalization state")
    return record


def validate_commit(path: Path, *, root: Path) -> dict:
    expected_path = generation_path(root, path.parent.name) / "COMMITTED.json"
    if path.resolve() != expected_path.resolve():
        raise ValueError("Recovery commit is outside its explicitly owned generation")
    record = read_json(path)
    if record.get("schema") != SCHEMA or record.get("status") != "committed":
        raise ValueError("Recovery point has no completed paired transaction")
    completed = record["completed_rounds"]
    if type(completed) is not int or completed <= 0 or set(record["roles"]) != set(ROLES):
        raise ValueError("Recovery point omitted a role or valid collection round")
    identities = []
    for role in ROLES:
        role_path = path.parent / f"{role}.json"
        if file_sha(role_path) != record["roles"][role]:
            raise ValueError("Paired recovery role receipt changed")
        current = validate_role_record(role_path, expected_round=completed, role=role)
        identities.append((current["data_sha256"], current["batch_size"]))
    if identities[0] != identities[1]:
        raise ValueError("Paired recovery roles used different input ledgers")
    return record


def latest_checkpoint(root: Path) -> Path | None:
    pointer = root / "LATEST.json"
    if not pointer.exists():
        return None
    value = read_json(pointer)
    path = generation_path(root, value["generation"]) / "COMMITTED.json"
    if file_sha(path) != value["sha256"]:
        raise ValueError("Latest recovery pointer does not match its committed transaction")
    validate_commit(path, root=root)
    return path


def _retire_previous(root: Path, previous: dict | None, current: Path) -> None:
    if previous is None or previous["generation"] == current.name:
        return
    old = generation_path(root, previous["generation"])
    if not old.exists():
        raise ValueError("Previous recovery disappeared before rolling retention")
    if file_sha(old / "COMMITTED.json") != previous["sha256"]:
        raise ValueError("Refusing to delete a changed recovery generation")
    if read_json(root / "LATEST.json")["generation"] != current.name:
        raise ValueError("Refusing retention before the replacement was published")
    receipt = {
        "previous_generation": old.name, "previous_commit_sha256": previous["sha256"],
        "replacement_generation": current.name,
        "replacement_commit_sha256": file_sha(current / "COMMITTED.json"),
        "deletion_target": str(old.resolve()), "at_unix": time.time(),
        "owner": read_json(old / "OWNER.json"),
    }
    write_json(root / "retention" / f"{old.name}.json", {**receipt, "status": "deleting"})
    shutil.rmtree(old)
    write_json(root / "retention" / f"{old.name}.json", {**receipt, "status": "deleted"})


def retire_incomplete_generations(root: Path, current: Path, completed: int) -> None:
    if read_json(root / "LATEST.json")["generation"] != current.name:
        raise ValueError("Partial checkpoint cleanup requires a committed replacement")
    for old in sorted(root.iterdir()):
        if old == current or GENERATION.fullmatch(old.name) is None:
            continue
        old = generation_path(root, old.name)
        owner_path = old / "OWNER.json"
        receipt = root / "retention" / f"{old.name}.json"
        if owner_path.is_file():
            owner = read_json(owner_path)
        elif receipt.is_file():
            previous = read_json(receipt)
            if previous.get("status") != "deleting" or previous.get("deletion_target") != str(old.resolve()):
                raise ValueError("Interrupted retention journal does not own its remaining target")
            owner = previous["owner"]
        else:
            raise ValueError(f"Refusing to delete an unowned partial recovery directory: {old}")
        if (
            owner != {"schema": SCHEMA, "root": str(root), "generation": old.name,
                      "completed_rounds": int(old.name.split("-")[1])}
            or owner["completed_rounds"] > completed
        ):
            raise ValueError("Partial recovery ownership or rollback ordering changed")
        audit = {
            "status": "deleting", "reason": "superseded_partial_or_unpublished_generation",
            "owner": owner, "files": file_inventory(old, allow_empty=True), "replacement": current.name,
            "deletion_target": str(old.resolve()), "at_unix": time.time(),
        }
        write_json(receipt, audit)
        shutil.rmtree(old)
        write_json(receipt, {**audit, "status": "deleted"})


def commit_pair(root: Path, generation: Path, completed: int) -> bool:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "COMMIT.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        committed = generation / "COMMITTED.json"
        if committed.exists():
            validate_commit(committed, root=root)
            return True
        if not all((generation / f"{role}.json").is_file() for role in ROLES):
            return False
        records = {
            role: validate_role_record(generation / f"{role}.json", expected_round=completed, role=role)
            for role in ROLES
        }
        if records["editor"]["data_sha256"] != records["solver"]["data_sha256"]:
            raise ValueError("Cannot commit different Engineer/Solver input ledgers")
        previous = read_json(root / "LATEST.json") if (root / "LATEST.json").exists() else None
        if previous is not None and previous["completed_rounds"] >= completed:
            raise ValueError("Recovery cannot overwrite or regress committed training")
        record = {
            "schema": SCHEMA, "status": "committed", "completed_rounds": completed,
            "roles": {role: file_sha(generation / f"{role}.json") for role in ROLES},
            "previous": previous, "committed_at_unix": time.time(),
            "validation": "All shard sizes, small-file hashes, Adam counters, sampled fingerprints, "
                          "ValueNorm, schedule and committed data/metric cursors; not full large-tensor hashes",
            "sampling": "Resident serving is unchanged between saves; interrupted serving is restarted "
                        "and uncommitted asynchronous work is discarded, not bitwise rollout replay",
        }
        _immutable_json(committed, record)
        validate_commit(committed, root=root)
        write_json(root / "history" / f"{generation.name}.json", {
            **record, "role_metadata": records,
            "optimizer_fingerprints": {
                role: {
                    engine: [read_json(path) for path in sorted(
                        (generation / role / engine).glob("adam-rank*.json")
                    )] for engine in ("actor", "critic")
                } for role in ROLES
            },
        })
        write_json(root / "LATEST.json", {
            "generation": generation.name, "completed_rounds": completed,
            "sha256": file_sha(committed), "at_unix": time.time(),
        })
        _retire_previous(root, previous, generation)
        retire_incomplete_generations(root, generation, completed)
        return True


def restore_trainer(trainer, spec: dict) -> None:
    validate_recovery_spec(spec)
    options = spec["rolling_recovery"]
    start = spec.get("recovery_start_round", 0)
    if options.get("finalize_only"):
        raise ValueError("A completed final state must be finalized without launching another learner")
    if not start:
        return
    from areal.api import SaveLoadMeta, StepInfo

    committed = Path(options["resume"])
    record = validate_commit(committed, root=Path(options["root"]))
    role = read_json(committed.parent / f"{spec['role']}.json")
    if (
        record["completed_rounds"] != start or role["science_sha256"] != science_digest(spec)
        or role["data_sha256"] != file_sha(Path(spec["data"]))
    ):
        raise ValueError("Requested continuation differs from the committed scientific state")
    payload = committed.parent / spec["role"]
    for name in ("actor", "critic"):
        engine = getattr(trainer, name)
        engine.load(meta=SaveLoadMeta(path=str(payload / name), weight_format="dcp", with_optim=True))
        engine.set_version(start)
    runtime = read_json(payload / "controller.json")
    trainer.saver.load_state_dict(runtime["saver"])
    trainer.evaluator.load_state_dict(runtime["evaluator"])
    trainer.stats_logger.load_state_dict(runtime["stats_logger"])
    if trainer.config.train_dataset.shuffle or trainer.config.train_dataset.num_workers != 0:
        raise ValueError("Committed-cursor recovery requires the frozen ordered single-process data loader")
    iterator = iter(trainer.train_dataloader)
    for _ in range(start):
        next(iterator)
    trainer.train_dataloader.load_state_dict(trainer.train_dataloader.state_dict())
    trainer.recover_info = SimpleNamespace(last_step_info=StepInfo(
        epoch=0, epoch_step=start - 1, global_step=start - 1,
        steps_per_epoch=len(trainer.train_dataloader),
    ))
    trainer.rollout.pause()
    trainer.actor.update_weights(trainer.weight_update_meta.with_version(start))
    trainer.rollout.set_version(start)
    owner = getattr(trainer.rollout, "workflow_executor", trainer.rollout)
    owner.staleness_manager.on_version_recovered(start)
    trainer.rollout.resume()
    write_json(Path(spec["run"]) / "RECOVERY_LOADED.json", {
        "status": "restored", "completed_rounds": start, "checkpoint": str(committed),
        "checkpoint_sha256": file_sha(committed), "next_sample_index": start * spec["batch_size"],
        "adam_reset": False, "warmup_repeated": False, "bitwise_rollout_resume": False,
        "verified_optimizer_fingerprints": {
            name: [read_json(path) for path in sorted((payload / name).glob("adam-rank*.json"))]
            for name in ("actor", "critic")
        },
        "at_unix": time.time(),
    })


class PairedRecovery:
    def __init__(self, trainer, spec: dict):
        validate_recovery_spec(spec)
        self.trainer, self.spec = trainer, spec
        self.options = spec["rolling_recovery"]
        self.root = Path(self.options["root"])

    def after_commit(self, completed: int) -> None:
        if completed % self.options["interval_rounds"] and completed != self.spec["steps"]:
            return
        from areal.api import SaveLoadMeta

        trainer, spec = self.trainer, self.spec
        state = trainer.joint_schedule_state
        counts = update_counts(spec, completed)
        if state != {"round": completed, "actor_steps": counts["actor_steps"],
                     "critic_steps": counts["critic_steps"]}:
            raise ValueError("Paired recovery ran outside a committed native update boundary")
        started = time.time()
        generation = generation_path(self.root, f"round-{completed:04d}-{self.options['attempt']}")
        _immutable_json(generation / "OWNER.json", {
            "schema": SCHEMA, "root": str(self.root), "generation": generation.name,
            "completed_rounds": completed,
        })
        payload = generation / spec["role"]
        if payload.exists():
            raise ValueError("Never overwrite a partial or complete native recovery payload")
        payload.mkdir(parents=True)
        for name in ("actor", "critic"):
            engine = getattr(trainer, name)
            config = engine.config.optimizer
            if config.lr_scheduler_type != "constant" or config.warmup_steps_proportion != 0:
                raise ValueError("Native paired recovery requires the frozen constant learning rates")
            engine.save(meta=SaveLoadMeta(path=str(payload / name), weight_format="dcp", with_optim=True))
        rows = metric_rows(Path(spec["run"]) / "metrics.jsonl")
        if [row["global_step"] for row in rows] != list(range(completed)):
            raise ValueError("Checkpoint metrics were not committed through the same collection round")
        (payload / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        write_json(payload / "spec.json", spec)
        write_json(payload / "controller.json", {
            "saver": trainer.saver.state_dict(), "evaluator": trainer.evaluator.state_dict(),
            "stats_logger": trainer.stats_logger.state_dict(), "schedule": state,
            "committed_sample_cursor": completed * spec["batch_size"],
            "queued_uncommitted_rollouts_restored": False,
        })
        for path in [payload, *payload.rglob("*")]:
            path.chmod(path.stat().st_mode | (0o070 if path.is_dir() else 0o060))
        hf = Path(spec["run"]) / "engine/checkpoints"
        protected = [
            str(path.relative_to(Path(spec["run"])))
            for path in hf.glob(f"*/{spec['experiment']}/{spec['phase']}/*/epoch*globalstep*")
            if int(path.name.split("globalstep")[-1]) + 1 in self.options["hf_rounds"]
        ]
        record = {
            "schema": SCHEMA, "role": spec["role"], "completed_rounds": completed,
            "batch_size": spec["batch_size"], "committed_sample_cursor": completed * spec["batch_size"],
            "science_sha256": science_digest(spec), "data_sha256": file_sha(Path(spec["data"])),
            "source_run": spec["run"], "source_joint": spec["joint_root"],
            "protected_hf_relative_paths": protected, "files": file_inventory(payload),
            "save_seconds": time.time() - started, "at_unix": time.time(),
        }
        _immutable_json(generation / f"{spec['role']}.json", record)
        timeout = min(spec["training_deadline_unix"], time.time() + 1800)
        while not commit_pair(self.root, generation, completed):
            check_failures(Path(spec["joint_root"]))
            if time.time() >= timeout:
                raise TimeoutError("The partner did not commit its same-round native recovery state")
            time.sleep(1)
        write_json(Path(spec["run"]) / "RECOVERY_SAVED.json", {
            "checkpoint": str(generation / "COMMITTED.json"), "completed_rounds": completed,
            "save_and_pair_wait_seconds": time.time() - started, "at_unix": time.time(),
        })


def materialize_prefix(spec: dict) -> None:
    validate_recovery_spec(spec)
    start = spec.get("recovery_start_round", 0)
    if not start:
        return
    options = spec["rolling_recovery"]
    committed = Path(options["resume"])
    validate_commit(committed, root=Path(options["root"]))
    record = read_json(committed.parent / f"{spec['role']}.json")
    if record["science_sha256"] != science_digest(spec) or record["completed_rounds"] != start:
        raise ValueError("Attempt input does not match the selected paired recovery point")
    run, joint = Path(spec["run"]), Path(spec["joint_root"])
    run.mkdir(parents=True)
    shutil.copyfile(committed.parent / spec["role"] / "metrics.jsonl", run / "metrics.jsonl")
    for relative in record["protected_hf_relative_paths"]:
        source = (Path(record["source_run"]) / relative).resolve()
        destination = run / relative
        if not source.is_dir() or not Path(relative).is_relative_to("engine/checkpoints"):
            raise ValueError("Protected evaluation checkpoint is missing or outside its checkpoint subtree")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source, target_is_directory=True)
    joint.mkdir(parents=True, exist_ok=True)
    for version in range(start):
        source = (Path(record["source_joint"]) / f"round-{version:04d}").resolve()
        destination = joint / source.name
        if not source.is_dir():
            raise ValueError("A committed joint episode round disappeared")
        if destination.is_symlink():
            if destination.resolve() != source:
                raise ValueError("Recovery partners disagree about the committed joint prefix")
        else:
            try:
                destination.symlink_to(source, target_is_directory=True)
            except FileExistsError:
                if not destination.is_symlink() or destination.resolve() != source:
                    raise


def finalize_committed_role(spec: dict, release: dict, *, elapsed_seconds: float) -> dict:
    """Publish audited final receipts if termination followed the final paired save."""
    from .common import ensure_model_runtime_metadata
    from .joint_health import recover_checkpoints
    from .joint_schedule import accounting, consumed_row_ledger, validate_metric_row
    from .resources import research_resources_released

    validate_recovery_spec(spec)
    if not spec["rolling_recovery"].get("finalize_only") or not research_resources_released(release):
        raise ValueError("Final-state publication requires an explicitly completed pair and released node")
    materialize_prefix(spec)
    run, joint = Path(spec["run"]), Path(spec["joint_root"])
    rows = metric_rows(run / "metrics.jsonl")
    if len(rows) != spec["steps"]:
        raise ValueError("The committed final state omitted training metrics")
    for row in rows:
        validate_metric_row(spec, row)
    history = recover_checkpoints(spec, rows)
    if str(spec["steps"] - 1) not in history:
        raise ValueError("The final native checkpoint has no corresponding final HF evaluation weights")
    generated = sum(1 for _ in joint.glob(f"round-*/{spec['role']}-receipt-*.json"))
    if generated != spec["steps"] * spec["batch_size"]:
        raise ValueError("Final checkpoint publication lost committed role trajectories")
    counts = update_counts(spec, spec["steps"])
    final = history[str(spec["steps"] - 1)]
    ensure_model_runtime_metadata(Path(final["actor"]), Path(spec["model"]))
    result = {
        "status": "completed", "role": spec["role"], "steps": spec["steps"],
        "actor_updates": counts["actor_steps"], "critic_updates": counts["critic_steps"],
        "metrics_sha256": file_sha(run / "metrics.jsonl"), "checkpoints": final,
        "checkpoint_history": history, "joint_rounds": spec["steps"],
        "elapsed_seconds": elapsed_seconds, "elapsed_includes_all_attempts_and_waiting": True,
        "finished_at_unix": time.time(), "optimizer_resets_between_rounds": False,
        "generated_policy_trajectories": generated, "stop": None,
        "sampled_engineer_actions": sum(1 for _ in joint.glob("round-*/proposal-*.json")),
        "schedule": spec["schedule"],
        "checkpoint_update_counts": {step: update_counts(spec, int(step) + 1) for step in history},
        "finalized_from_complete_native_pair_without_new_training": spec["rolling_recovery"]["resume"],
        **accounting(spec, spec["steps"], generated=generated),
    }
    write_json(run / "consumed_rows.json", consumed_row_ledger(spec, read_json(Path(spec["data"])), len(rows)))
    _immutable_json(run.parent / "RELEASED.json", release)
    _immutable_json(run.parent / "COMPLETE.json", result)
    return result
