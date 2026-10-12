"""Persistent two-node domain campaign; no GPU work occurs on import.

Run ``python -m sao_bridge.domain_campaign baseline --protocol FILE`` on both
hosts to publish the assigned raw/SFT dev result, then ``worker --protocol FILE``.
Use ``baseline --domain searchqa`` to start ready SearchQA without reading any
Chess dataset. Later run ``baseline --domain chess --resume`` when it is ready.
``worker --domain searchqa --resume`` likewise runs only the ready domain, then
``worker --domain chess --resume`` completes the other domain. Protocol order
is unchanged; an explicitly scoped execution is recorded separately, and a
single completed domain never marks the whole campaign completed.
Add ``--resume`` to worker when baseline already created the run. Worker also
performs its baseline if absent. ``--resume`` permits only verified
phase-boundary recovery; an interrupted native optimizer invocation is terminal.
``status`` reads local/peer progress without allocating GPUs.
Run ``monitor --protocol FILE`` in a separately supervised CPU-only service.
It writes WATCHER_STATUS.json and, on failure/stall, WATCHER_ALERT.json, emits
the alert on stderr and exits 2 for its parent supervisor. It does not restart
partial optimizers or kill any GPU workload. Use ``--until baseline`` to watch
only baseline completion; otherwise it watches the complete two-domain campaign.
Fresh standalone baselines/reports use eight solver replicas; H-phase solvers
retain GPUs4-7 and previously started baselines keep their recorded four-GPU
layout. Solver ports start at18801, incrementing by10 per replica. Actual
allocation/health maps are recorded in ACTIVE_SERVERS.json and inference logs;
their GPU-seconds measure reserved wall time, not kernel utilization or FLOPs.

Each node owns its output tree. Cross-host barriers contain immutable hashed
reports; no shared writer, test-based decisions, or continuously resumed Adam.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
from collections import Counter
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit

from .analysis import difference_in_differences, paired_difference, paired_lower_bound
from .budget import initial_model_digest
from .common import file_sha, read_json, write_json
from .phase_recovery import cached_evaluation, editor_pair_audit

ARMS = {"sao", "ma_sao", "weight_only", "harness_only"}
HANDOFF = "HF actor/critic weights continued; optimizer moments reset at every phase"


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    write_json(path, value)
    path.chmod(0o644)


def immutable(path: Path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Immutable campaign record changed: {path}")
    else:
        save(path, value)
    return value


def public_tree(root: Path) -> None:
    # Do not follow model/dataset links or alter files outside the owned run.
    for base, dirs, files in os.walk(root, followlinks=False):
        Path(base).chmod(0o755)
        for name in files:
            path = Path(base) / name
            if not path.is_symlink():
                path.chmod(0o644)


def validate_protocol(protocol: dict) -> None:
    c, a, s = (protocol[k] for k in ("common", "algorithm", "stopping"))
    fixed = {
        "context_tokens": 16384, "solver_tokens": 4096, "editor_tokens": 2048,
        "editor_steps": 3, "editor_batch_size": 16, "solver_steps": 32,
        "solver_batch_size": 64, "editor_lr": 5e-6, "solver_lr": 2e-6,
        "critic_lr": 1e-5, "reward_questions": 32, "dev_questions": 256,
        "baseline_samples": 8, "selection_samples": 2, "report_samples": 8,
        "proposals": 4,
    }
    if any(c.get(k) != v for k, v in fixed.items()) or protocol["seed"] != 42:
        raise ValueError("Changed fixed campaign training/evaluation budget")
    if a != dict(central_critic=False, critic_warmup_steps=8,
                 warmup_recycled_per_row=2, solver_recycle_fraction=1.0,
                 replay_fraction=0.25, interaction_credit=0.0):
        raise ValueError("Changed matched recycling/replay protocol")
    if s != dict(minimum_blocks=3, patience=3, minimum_dev_gain=0.005, maximum_blocks=12):
        raise ValueError("Changed predeclared dev plateau rule")
    if set(protocol["domain_order"]) != {"chess", "searchqa"} or len(protocol["domain_order"]) != 2:
        raise ValueError("Campaign requires both domains exactly once")
    expected = {
        "dev-NDH200-026": (1008, "original", "sao", "weight_only"),
        "dev-NDH200-025": (1009, "sft1152", "ma_sao", "harness_only"),
    }
    if set(protocol["nodes"]) != set(expected):
        raise ValueError("Only assigned node3/node4 may participate")
    roots = []
    for host, values in expected.items():
        node = protocol["nodes"][host]
        if tuple(node[k] for k in ("uid", "baseline", "alternating", "single")) != values:
            raise ValueError("Node/UID/arm assignment changed")
        if node["peer"] not in expected or node["peer"] == host:
            raise ValueError("Invalid peer")
        roots.append(Path(node["root"]).resolve())
    if roots[0] == roots[1] or any(x in y.parents for x, y in (roots, roots[::-1])):
        raise ValueError("Node output roots must be disjoint")
    for name, count in (("chess", 256), ("searchqa", 700)):
        if protocol["domains"][name]["test_questions"] != count:
            raise ValueError("Changed held-out report count")
    if not 0 < protocol["runtime"]["safety_seconds"] <= 48 * 3600:
        raise ValueError("Campaign needs a bounded <=48h safety deadline")


def implementation_identity(domain: str | None = None) -> dict:
    directory = Path(__file__).parent
    if domain is None:
        return {path.name: file_sha(path) for path in sorted(directory.glob("*.py"))}
    names = {"domain_api.py", "evaluation.py", "launch.py", "resources.py",
             "program_harness.py", f"{domain}_domain.py"}
    if domain == "searchqa":
        names.add("searchqa_retrieval.py")
    result = {name: file_sha(directory / name) for name in sorted(names)}
    for name, symbols in (("common.py", {"file_sha", "read_json", "write_json"}),
                          ("retool.py", {"ContextExhausted"})):
        result.update(component_identity(directory / name, symbols))
    return result


def training_implementation_identity(domain: str) -> dict:
    other = "chess" if domain == "searchqa" else "searchqa"
    return {name: value for name, value in implementation_identity().items()
            if not name.startswith(other + "_") and name != other + ".py"}


def component_identity(path: Path, symbols: set[str]) -> dict:
    tree = ast.parse(path.read_text())
    nodes = {node.name: node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
             and node.name in symbols}
    if set(nodes) != symbols:
        raise ValueError(f"Missing frozen evaluation component in {path}")
    return {f"{path.name}:{name}": digest(ast.dump(node, include_attributes=False))
            for name, node in sorted(nodes.items())}


def requested_domains(protocol: dict, command: str, domain: str | None = None):
    if domain is not None:
        if command not in {"baseline", "worker"} or domain not in protocol["domain_order"]:
            raise ValueError("--domain is supported only for a declared baseline/worker domain")
        return [domain]
    return list(protocol["domain_order"])


def completed_domain_names(protocol: dict, root: Path) -> list[str]:
    return [domain for domain in protocol["domain_order"]
            if (root / domain / "DOMAIN_COMPLETE.json").is_file()]


def execution_completion(protocol: dict, root: Path, command: str, domains: list) -> dict:
    completed = completed_domain_names(protocol, root)
    whole = completed == protocol["domain_order"]
    return {
        "status": "baselines_ready" if command == "baseline" else "completed" if whole else "domains_completed",
        "requested_domains": domains, "completed_domains": completed, "campaign_complete": whole,
        "baseline_domains": [domain for domain in protocol["domain_order"]
                             if (root / domain / "BASELINE_READY.json").is_file()],
    }


def solver_server_layout(*, full_node=False, context_tokens=16384) -> dict:
    devices = list(range(8)) if full_node else list(range(4, 8))
    return {
        "allocation": "full_node" if full_node else "four_gpu_frozen_solver",
        "gpu_ids": devices, "data_parallel": len(devices), "tensor_parallel": 1,
        "context_tokens": context_tokens,
        "replicas": [
            {"gpu": gpu, "port": 18801 + 10 * index,
             "api_url": f"http://127.0.0.1:{18801 + 10 * index}/v1",
             "health_url": f"http://127.0.0.1:{18801 + 10 * index}/health"}
            for index, gpu in enumerate(devices)
        ],
    }


def baseline_server_layout(root: Path, context_tokens=16384) -> dict:
    """Preserve a started baseline's layout; old execution records imply four GPUs."""
    previous = [read_json(path).get("solver_server_layout") or solver_server_layout(
        context_tokens=context_tokens) for path in sorted(root.glob("BASELINE_EXECUTION_*.json"))]
    if previous:
        if any(layout != previous[0] for layout in previous):
            raise ValueError("Baseline allocation layout changed across execution records")
        return previous[0]
    return solver_server_layout(full_node=True, context_tokens=context_tokens)


def inference_ledger(root: Path) -> dict:
    records = [(path, read_json(path)) for path in sorted(root.glob("**/inference/*.json"))]
    return {
        "recorded_server_sessions": len(records),
        "incomplete_server_sessions": sum("allocated_gpu_seconds" not in row for _, row in records),
        "allocated_gpu_seconds": sum(row.get("allocated_gpu_seconds", 0) for _, row in records),
        "server_wall_seconds": sum(row.get("wall_seconds", 0) for _, row in records),
        "records_sha256": {str(path.relative_to(root)): file_sha(path) for path, _ in records},
        "measurement": "Requested GPU reservation from server startup through shutdown, including "
                       "loading/idle time; not GPU kernel utilization or FLOPs. Only instrumented "
                       "sessions count; older baseline work is not retrospectively metered.",
    }


def audit_identity(protocol: dict, domain: str) -> dict:
    from .domain_api import get_domain

    data = Path(protocol["domains"][domain]["data"])
    manifest = read_json(data / "manifest.json")
    splits, seen = {}, set()
    for name in ("train", "reward", "dev", "test"):
        path = data / f"{name}.json"
        sha = file_sha(path)
        if sha != manifest["splits"][name]["sha256"]:
            raise ValueError(f"{domain} {name} split hash mismatch")
        rows = read_json(path)
        ids = [row["id"] for row in rows]
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError("Duplicate/overlapping split question IDs")
        for identity in ids:
            int(identity[:8], 16)
        seen.update(ids)
        expected = {"reward": 128, "dev": 256,
                    "test": protocol["domains"][domain]["test_questions"]}
        if name in expected and len(rows) != expected[name]:
            raise ValueError(f"Wrong {domain} {name} split size")
        if name == "train" and len(rows) < 2048:
            raise ValueError("Train split cannot cover one unique actor phase")
        splits[name] = {"sha256": sha, "count": len(rows), "ids_sha256": digest(ids)}
    receipt = read_json(Path(protocol["sft_receipt"]))
    if (receipt["status"] != "completed" or not receipt["hf_cpu_reload_verified"]
            or receipt["checkpoint"] != protocol["models"]["sft1152"]
            or receipt["cumulative_train_records"] != 1152):
        raise ValueError("SFT initializer is not the verified 1152-record checkpoint")
    result = {
        "protocol_sha256": digest(protocol), "domain": domain, "seed": protocol["seed"],
        "manifest_sha256": file_sha(data / "manifest.json"), "splits": splits,
        "models": {key: initial_model_digest(Path(value))
                   for key, value in protocol["models"].items()},
        "initial_source_sha256": digest(get_domain(domain).INITIAL_SOURCE),
        "evaluation_implementation_sha256": implementation_identity(domain),
        "sft_receipt_sha256": file_sha(Path(protocol["sft_receipt"])),
    }
    if domain == "searchqa":
        result["retrieval_manifest_sha256"] = file_sha(data / "retrieval_manifest.json")
    return result


def evaluation_identity(identity, model_sha, source, split, rows, samples, seed, purpose):
    return {"campaign": digest(identity), "model_sha256": model_sha,
            "source_sha256": digest(source), "split": split, "rows_sha256": digest(rows),
            "samples": samples, "seed": seed, "purpose": purpose}


def verify_report(path: Path, rows: list, identity: dict) -> dict:
    report = cached_evaluation(path, rows, identity["samples"])
    if report.get("cache_identity") != identity:
        raise ValueError("Cached evaluation model/source/split/seed identity changed")
    expected = {(row["id"], sample,
                 identity["seed"] + int(row["id"][:8], 16) % 1000000 + 1009 * sample)
                for row in rows for sample in range(identity["samples"])}
    actual = [(row["id"], row.get("sample_index"), row.get("rollout_seed"))
              for row in report["results"]]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("Cached report sample counts or seed identity differ")
    rewards = [row["reward"] for row in report["results"]]
    if (any(v not in (0, 1) for v in rewards)
            or not math.isclose(report["accuracy"], sum(rewards) / len(rewards), abs_tol=1e-12)):
        raise ValueError("Cached report accuracy disagrees with binary receipts")
    return report


def require_dev(report: dict) -> None:
    if report.get("cache_identity", {}).get("split") != "dev":
        raise ValueError("Only DEV may select, extend or stop")


def choose_initializer(original: dict, sft: dict) -> str:
    for report in (original, sft):
        require_dev(report)
    paired_difference(original, sft, draws=100)
    return "sft1152" if sft["accuracy"] > original["accuracy"] else "original"


def harness_error_count(report: dict) -> int:
    errors = report.get("harness_error_episodes", 0)
    if type(errors) is not int or errors < 0:
        raise ValueError("Invalid runtime hook error count")
    return max(errors, sum(bool(row.get("harness_errors") or row.get("hook_errors"))
                           for row in report["results"]))


def select_harness(incumbent: str, control: dict, candidates: list, outcomes: list,
                   *, seed: int, proposals=4) -> dict:
    if len(candidates) != proposals or len(outcomes) != proposals:
        raise ValueError("Harness selection requires all predeclared proposal slots")
    require_dev(control)
    rejected, blocked = [], set()
    for slot, (candidate, outcome) in enumerate(zip(candidates, outcomes)):
        require_dev(outcome)
        errors = harness_error_count(outcome)
        if errors:
            source = candidate["source"] if candidate["valid"] else incumbent
            rejected.append({"slot": slot, "source_sha256": digest(source),
                             "harness_error_episodes": errors})
            blocked.add(source)
    scored, seen = [], {incumbent}
    for slot, (candidate, outcome) in enumerate(zip(candidates, outcomes)):
        if not candidate["valid"] or candidate["source"] in seen | blocked:
            continue
        source = candidate["source"]
        seen.add(source)
        scored.append({
            "source": source, "accuracy": outcome["accuracy"], "slot": slot,
            "lcb": paired_lower_bound(control, outcome, alpha=0.05 / proposals,
                                      seed=seed + 31337 + slot),
        })
    eligible = [record for record in scored if record["lcb"] > 0]
    chosen = max(eligible, key=lambda item: item["accuracy"]) if eligible else {
        "source": incumbent, "accuracy": control["accuracy"], "lcb": 0, "slot": None}
    return {
        "split": "dev", "selected": chosen, "candidates": scored,
        "runtime_rejected_slots": rejected,
        "alpha": 0.05 / proposals, "proposals": proposals,
        "dedup": "exact source text, including optional executable hooks; any error-bearing slot disqualifies its source",
        "report": "accepted harness only, independent of whether it changed",
    }


def phase_plan(arm: str, completed_blocks: int) -> tuple[str, ...]:
    if arm not in ARMS or completed_blocks < 0:
        raise ValueError("Invalid arm/block")
    return (("H",) if arm == "harness_only" else ("W",) if arm == "weight_only"
            else ("H", "W", "H") if completed_blocks == 0 else ("W", "H"))


def update_plateau(state: dict, snapshot: dict, report: dict, stopping: dict) -> dict:
    require_dev(report)
    state = copy.deepcopy(state)
    score = float(report["accuracy"])
    state["blocks"] += 1
    state["current"] = snapshot
    state["history"].append({"block": state["blocks"], "score": score,
                             "report": snapshot["dev_report"]})
    if score > state["best_score"]:
        state.update(best=copy.deepcopy(snapshot), best_score=score)
    if score >= state["anchor_score"] + stopping["minimum_dev_gain"]:
        state.update(anchor_score=score, stale=0)
    else:
        state["stale"] += 1
    state["status"] = (
        "converged" if state["blocks"] >= stopping["minimum_blocks"]
        and state["stale"] >= stopping["patience"]
        else "budget_exhausted" if state["blocks"] >= stopping["maximum_blocks"]
        else "active")
    return state


def extension_decision(baseline, sao, ma, left: dict, right: dict, stopping: dict) -> dict:
    for report in (baseline, sao, ma):
        require_dev(report)
    if left["blocks"] != right["blocks"]:
        raise ValueError("Peer block barriers do not match")
    paired_difference(baseline, sao, draws=100)
    paired = paired_difference(sao, ma)
    changes = difference_in_differences(baseline, sao, baseline, ma)
    ma_gain = ma["accuracy"] - baseline["accuracy"]
    advantage = ma_gain > 0 and changes["ma_change_minus_sao_change"] > 0
    blocks = left["blocks"]
    # The user-conditional gate is not evidence of a plateau.
    if not advantage:
        reason = "not_extended" if blocks == 1 else "extension_gate_closed"
    elif blocks >= stopping["minimum_blocks"] and min(left["stale"], right["stale"]) >= stopping["patience"]:
        reason = "converged"
    elif blocks >= stopping["maximum_blocks"]:
        reason = "budget_exhausted"
    else:
        reason = "continue"
    return {"decision": reason, "blocks": blocks, "ma_positive_gain": ma_gain,
            "ma_point_estimate_advantage": advantage, "paired_final": paired,
            "difference_in_differences": changes,
            "claim": "Single-seed exploratory dev comparison; no statistical-superiority claim",
            "ci_excludes_zero": changes["paired_question_bootstrap_95pct"][0] > 0}


def recycling_partition(pool: list, common: dict, algorithm: dict):
    warm_n = algorithm["critic_warmup_steps"] * common["solver_batch_size"] * algorithm["warmup_recycled_per_row"]
    mixed_n = int(common["solver_steps"] * common["solver_batch_size"] * algorithm["solver_recycle_fraction"])
    ids = [row["id"] for row in pool]
    if len(ids) != len(set(ids)) or len(ids) < warm_n + mixed_n:
        raise ValueError("Unique partner pool cannot cover disjoint warmup and actor mixing")
    warm, mixed = ids[:warm_n], ids[warm_n:warm_n + mixed_n]
    return warm, mixed, pool[:warm_n + mixed_n]


def recycling_namespace(identity: dict, run: Path, phase: str) -> str:
    return digest({"campaign_identity": identity, "reward_run": str(run.resolve()),
                   "reward_phase": phase})


def training_window(rows: list, count: int, offset: int, namespace: str):
    if count < 1 or len(rows) < count or offset < 0:
        raise ValueError("A solver phase requires distinct canonical questions within its window")
    selected, epochs = [], Counter()
    for index in range(offset, offset + count):
        epoch, source_index = divmod(index, len(rows))
        original = rows[source_index]
        episode_id = digest({"namespace": namespace, "position": index, "question": original["id"]})
        selected.append({
            **original, "id": episode_id, "campaign_question_id": original["id"],
            "campaign_episode_id": episode_id, "campaign_training_epoch": epoch,
            "campaign_source_index": source_index,
        })
        epochs[str(epoch)] += 1
    return selected, {
        "split": "train", "canonical_questions": len(rows), "global_offset": offset,
        "count": count, "epoch_counts": dict(epochs), "namespace": namespace,
        "canonical_ids_sha256": digest([row["campaign_question_id"] for row in selected]),
        "episode_ids_sha256": digest([row["id"] for row in selected]),
        "fresh_generation": True, "previous_episode_replay": False,
        "note": "Canonical train questions cycle across epochs; every phase generates new episodes. "
                "Native asynchronous prefetch is separately counted in the physical ledger.",
    }


def reward_window(h_index: int, arm: str, count=32) -> dict:
    if h_index < 0 or arm not in ARMS or count != 32:
        raise ValueError("Invalid predeclared reward window")
    return {
        "split": "reward", "offset": (h_index % 4) * count, "count": count,
        "rotation": h_index // 4, "unseen_question_claim": False,
        "may_have_entered_solver_optimizer": arm == "ma_sao" and h_index >= 4,
        "note": "Training-only reward windows rotate across 128 questions and repeat after "
                "four H phases. MA recycling can train W on these questions; not held-out evidence.",
    }


def completed_optimizer(root: Path, spec: dict) -> dict | None:
    """A launch intent is durable BEFORE spawning; ambiguous crashes never rerun."""
    intent, out = root / f"{spec['phase']}_optimizer_intent.json", Path(spec["run"])
    if not intent.exists() and not out.exists():
        return None
    if not intent.exists() or read_json(intent) != spec:
        raise ValueError("Optimizer launch identity missing or changed")
    if not (out / "trained.json").is_file():
        raise ValueError("Refusing partial optimizer resume; allocate a new campaign")
    result = read_json(out / "trained.json")
    metrics = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    warmup = spec["critic_warmup_steps"]
    if ([r["global_step"] for r in metrics] != list(range(spec["steps"]))
            or result["actor_steps"] != spec["steps"] - warmup
            or result["critic_only_warmup_steps"] != warmup
            or result["critic_steps"] != 2 * spec["steps"]
            or result.get("stage_optimizer_reset") is not True
            or result.get("central_critic") is not False):
        raise ValueError("Incomplete/mismatched native optimizer receipt")
    for role in ("actor", "critic"):
        directory = Path(result["checkpoints"][role])
        if not directory.is_dir() or not list(directory.glob("*.safetensors")):
            raise ValueError("Native optimizer checkpoint is missing")
    return result


def phase_evidence(run: Path, phase: str) -> dict:
    paths = [p for p in run.glob(f"{phase}_*") if p.is_file()
             and p.name != f"{phase}_COMPLETE.json"]
    paths += list((run / phase / "receipts").glob("*.json"))
    paths += list((run / f"{phase}_reward_evals").glob("*.json"))
    paths += [run / phase / "trained.json", run / phase / "metrics.jsonl"]
    paths += list((run / "inference").glob(f"{run.name}_{phase}_*.json"))
    if (run / f"{phase}.py").exists():
        paths.append(run / f"{phase}.py")
    return {str(path.relative_to(run)): file_sha(path) for path in sorted(paths)}


def training_feedback(report: dict) -> str:
    if report["cache_identity"]["split"] != "train":
        raise ValueError("Editor feedback must contain training-only traces, never dev/test")
    failures = [row for row in report["results"] if row["reward"] == 0]
    records = failures[:8] or report["results"][:2]
    # These are actual solver episode records, not a dev-score-derived instruction.
    fields = {"id", "reward", "messages", "failure_trace", "harness_errors", "hook_errors",
              "tool_errors", "boxed", "truncated", "stop_reason", "outcome"}
    traces = [{k: v for k, v in row.items() if k in fields} for row in records]
    return json.dumps({"split": "train", "failures": len(failures),
                       "sampled_episode_traces": traces}, ensure_ascii=False)[:24000]


class Monitor:
    def __init__(self, root: Path, seconds: int, safety: int, progress_timeout=14400):
        import psutil
        self.root, self.seconds = root, seconds
        self.progress_timeout = progress_timeout
        self.started = time.monotonic()
        self.deadline = self.started + safety
        self.lock, self.done = threading.Lock(), threading.Event()
        now = time.time()
        self.state = {"status": "starting", "host": os.uname().nodename, "pid": os.getpid(),
                      "pid_created_at": psutil.Process(os.getpid()).create_time(),
                      "progress_at": now, "progress_deadline_at": now + progress_timeout}
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def update(self, status: str, **fields):
        with self.lock:
            now = time.time()
            self.state.update(status=status, progress_at=now,
                              progress_deadline_at=fields.pop("progress_deadline_at", now + self.progress_timeout),
                              **fields)
            self._write()

    def _write(self):
        save(self.root / "STATUS.json", {**self.state, "heartbeat_at": time.time(),
                                         "elapsed_seconds": time.monotonic() - self.started})

    def _loop(self):
        while not self.done.wait(self.seconds):
            with self.lock:
                self._write()

    def check(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("48h campaign safety budget exhausted; NOT convergence")

    def __enter__(self):
        with self.lock:
            self._write()
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.done.set()
        self.thread.join()


def retrieval_health_url(url: str) -> str:
    parsed = urlsplit(url.rstrip("/"))
    if (parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment or parsed.path not in ("", "/retrieve")):
        raise ValueError("SAO_SEARCHQA_RETRIEVAL_URL must be the trusted HTTP root or /retrieve endpoint")
    return parsed._replace(path="/health").geturl()


def watcher_snapshot(protocol: dict, *, now=None, stale_seconds=180, grace_seconds=300,
                     domains=None) -> dict:
    """Read-only cross-host health; heartbeat freshness and real progress differ."""
    now = time.time() if now is None else now
    nodes, alerts = {}, []
    domains = protocol["domain_order"] if domains is None else domains
    for host, node in protocol["nodes"].items():
        root = Path(node["root"])
        path = root / "STATUS.json"
        state = read_json(path) if path.exists() else {"status": "not_started"}
        baselines = all((root / domain / "BASELINE_READY.json").is_file()
                        for domain in domains)
        record = {"worker": state, "baselines_ready": baselines, "arms": {},
                  "completed_domains": completed_domain_names(protocol, root)}
        for domain in protocol["domain_order"]:
            for arm in (node["alternating"], node["single"]):
                arm_path = root / domain / arm / "STATE.json"
                if arm_path.exists():
                    current = read_json(arm_path)
                    record["arms"][f"{domain}/{arm}"] = {
                        key: current[key] for key in ("blocks", "status", "best_score", "stale")
                    }
        nodes[host] = record
        status = state["status"]
        if status == "failed":
            alerts.append({"host": host, "kind": "worker_failed", "detail": state.get("error")})
        elif status not in {"completed", "domains_completed", "baselines_ready", "not_started"}:
            if now - state.get("heartbeat_at", 0) > stale_seconds:
                alerts.append({"host": host, "kind": "heartbeat_stale"})
            if now > state.get("progress_deadline_at",
                               state.get("progress_at", now) + protocol["runtime"]["phase_timeout_seconds"]) + grace_seconds:
                alerts.append({"host": host, "kind": "progress_stalled",
                               "phase": state.get("phase"), "status": status})
            if host == os.uname().nodename and state.get("pid_created_at") is not None:
                import psutil
                try:
                    process = psutil.Process(state["pid"])
                    alive = (process.create_time() == state["pid_created_at"]
                             and process.status() != psutil.STATUS_ZOMBIE
                             and process.uids().real == node["uid"])
                except psutil.Error:
                    alive = False
                if not alive:
                    alerts.append({"host": host, "kind": "worker_process_exited"})
    return {"checked_at": now, "nodes": nodes, "alerts": alerts,
            "campaign_complete": all(row["completed_domains"] == protocol["domain_order"]
                                     for row in nodes.values()),
            "baselines_complete": all(row["baselines_ready"] for row in nodes.values())}


def watch(protocol: dict, host: str, *, until="campaign", stale_seconds=180, interval=15,
          domain=None):
    root = Path(protocol["nodes"][host]["root"])
    root.mkdir(parents=True, mode=0o755, exist_ok=True)
    if root.stat().st_uid != os.getuid():
        raise PermissionError("Watcher must write only its own assigned node root")
    with (root / "watcher.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("A persistent watcher already owns this node") from exc
        started = time.monotonic()
        while True:
            try:
                state = watcher_snapshot(protocol, stale_seconds=stale_seconds,
                                         domains=[domain] if domain else None)
            except (OSError, ValueError, KeyError) as exc:
                state = {"checked_at": time.time(), "alerts": [
                    {"kind": "watcher_read_failure", "detail": repr(exc)}]}
            state.update(watcher_pid=os.getpid(), until=until, domain=domain)
            elapsed = time.monotonic() - started
            if not state["alerts"] and state.get(f"{'baselines' if until == 'baseline' else 'campaign'}_complete"):
                save(root / "WATCHER_STATUS.json", {**state, "status": "completed"})
                return 0
            if elapsed >= protocol["runtime"]["safety_seconds"]:
                state["alerts"].append({"kind": "watcher_safety_deadline", "converged": False})
            for name, node in state.get("nodes", {}).items():
                if (node["worker"]["status"] == "not_started"
                        and elapsed > protocol["runtime"]["peer_timeout_seconds"]):
                    state["alerts"].append({"host": name, "kind": "worker_never_started"})
            if state["alerts"]:
                state["status"] = "alert"
                save(root / "WATCHER_STATUS.json", state)
                save(root / "WATCHER_ALERT.json", state)
                print(json.dumps(state), file=sys.stderr, flush=True)
                return 2
            save(root / "WATCHER_STATUS.json", {**state, "status": "watching"})
            time.sleep(interval)


@contextmanager
def allocation(root: Path, protocol: dict, monitor: Monitor, domain: str):
    from .launch import Launcher
    from .resources import UNIT, stop_known
    import httpx

    monitor.check()
    if domain == "searchqa":
        url = retrieval_health_url(os.environ.get("SAO_SEARCHQA_RETRIEVAL_URL", ""))
        with httpx.Client(trust_env=False, timeout=10) as client:
            response = client.get(url)
            response.raise_for_status()
            health = response.json()
        data = Path(protocol["domains"][domain]["data"])
        expected = read_json(data / "retrieval_manifest.json")
        identity = read_json(root / domain / "identity.json")
        if (file_sha(data / "retrieval_manifest.json") != identity["retrieval_manifest_sha256"]
                or health.get("index_identity") != identity["retrieval_manifest_sha256"]
                or any(health.get(key) != expected[key]
                       for key in ("corpus_sha256", "passages", "top_k_ceiling"))):
            raise ValueError("Retrieval service differs from the frozen corpus/index identity")
        save(root / domain / "RETRIEVAL_HEALTH.json", {"url": url, "checked_at": time.time(), **health})
    event = stop_known(require_exclusive=True)
    save(root / f"resource_audit_{time.time_ns()}.json", event)
    if event.get("verified_unit"):
        save(root / "authorized_filler.json", {"unit": UNIT, "verified": True})
    launch = None
    try:
        launch = Launcher(root / "supervision" / f"{domain}_{time.time_ns()}")
        # Launcher inherits the environment; assign explicitly for nested workers.
        if domain == "searchqa":
            launch.env["SAO_SEARCHQA_RETRIEVAL_URL"] = os.environ["SAO_SEARCHQA_RETRIEVAL_URL"]
        yield launch
    finally:
        if launch is not None:
            launch.close()
        public_tree(root)
        pids = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            text=True).strip()
        restored = False
        if (not pids and protocol["runtime"].get("restore_authorized_filler")
                and (root / "authorized_filler.json").exists()):
            subprocess.run(["sudo", "-n", "systemctl", "start", UNIT], check=True, timeout=45)
            restored = True
        save(root / "RELEASED.json", {"time": time.time(), "remaining_gpu_pids": pids,
                                      "authorized_filler_restored": restored})


class Engine:
    """Reusable H/W phase functions backed exclusively by native SAO subprocesses."""
    def __init__(self, protocol, domain, root, identity, launch, monitor):
        self.protocol, self.domain, self.root = protocol, domain, root
        self.identity, self.launch, self.monitor = identity, launch, monitor
        self.c, self.a = protocol["common"], protocol["algorithm"]
        self.data = Path(protocol["domains"][domain]["data"])
        self.rows = {split: read_json(self.data / f"{split}.json")
                     for split in ("train", "dev")}

    @contextmanager
    def server(self, snapshot, name, *, full_node=False, ledger_root=None):
        layout = solver_server_layout(full_node=full_node, context_tokens=self.c["context_tokens"])
        path = (self.root if ledger_root is None else ledger_root) / "inference" / f"{name}_{time.time_ns()}.json"
        started = time.monotonic()
        record = {"name": name, "domain": self.domain, "model": snapshot["solver"],
                  "model_sha256": snapshot.get("solver_sha256"), "solver_server_layout": layout,
                  "started_at": time.time(), "status": "starting"}
        previous_layout = getattr(self, "active_server_layout", None)
        save(path, record)
        self.monitor.update("starting_frozen_solver", domain=self.domain, service=name,
                            solver_server_layout=layout, server_record=str(path))
        try:
            with self.launch.vllm(
                snapshot["solver"], port=18801, gpus=",".join(map(str, layout["gpu_ids"])),
                name=name, data_parallel=layout["data_parallel"],
                max_model_len=self.c["context_tokens"],
            ) as url:
                if url.split(",") != [replica["api_url"] for replica in layout["replicas"]]:
                    raise ValueError("Frozen solver endpoints differ from the declared allocation")
                self.active_server_layout = layout
                record.update(status="ready", ready_at=time.time())
                save(path, record)
                save(self.root / "ACTIVE_SERVERS.json", record)
                self.monitor.update("solver_ready", domain=self.domain, service=name,
                                    solver_server_layout=layout, solver_health_status="ready_at_start",
                                    server_record=str(path))
                yield url
        except BaseException:
            record["status"] = "failed"
            raise
        finally:
            self.active_server_layout = previous_layout
            elapsed = time.monotonic() - started
            record.update(status="failed" if record["status"] == "failed" else "closed",
                          closed_at=time.time(), wall_seconds=elapsed,
                          allocated_gpu_seconds=elapsed * len(layout["gpu_ids"]))
            save(path, record)
            save(self.root / "ACTIVE_SERVERS.json", record)
            self.monitor.update("solver_closed", domain=self.domain, service=name,
                                solver_health_status="closed", server_record=str(path))

    def evaluate(self, url, snapshot, split, samples, output, purpose, rows=None):
        from .evaluation import Sampler, evaluate
        self.monitor.check()
        rows = (self.rows[split] if split in self.rows else read_json(self.data / f"{split}.json")) if rows is None else rows
        seed = self.protocol["seed"] + (0 if purpose == "selection" else 5_000_003)
        source = Path(snapshot["harness"]).read_text()
        self.monitor.update("evaluating", domain=self.domain, split=split, purpose=purpose,
                            evaluation_output=str(output))
        identity = evaluation_identity(self.identity, snapshot["solver_sha256"], source,
                                       split, rows, samples, seed, purpose)
        if output.exists():
            return verify_report(output, rows, identity)

        async def run():
            sampler = Sampler(url, concurrency=self.c["eval_concurrency"])
            try:
                return await evaluate(sampler, rows, source, tokens=self.c["solver_tokens"],
                                      samples=samples, seed=seed, domain=self.domain)
            finally:
                await sampler.close()
        report = asyncio.run(run())
        save(output, {**report, "cache_identity": identity,
                      "domain_metric": self.protocol["domains"][self.domain]["metric"],
                      "benchmark_adaptations": self.protocol["domains"][self.domain].get("adaptations", {}),
                      "solver_server_layout": getattr(self, "active_server_layout", None)})
        self.launch.check()
        return verify_report(output, rows, identity)

    def train(self, run: Path, phase, role, snapshot, data, evaluator=None, recycle=None):
        self.monitor.check()
        warmup = self.a["critic_warmup_steps"] if recycle else 0
        spec = {
            "run": str(run / phase), "role": role, "domain": self.domain,
            "context_tokens": self.c["context_tokens"], "model": snapshot[role],
            "critic_model": snapshot[f"{role}_critic"], "phase": phase,
            "experiment": f"{self.protocol['campaign']}-{self.domain}-{run.parent.parent.name}-{run.name}",
            "steps": self.c[f"{role}_steps"] + warmup,
            "batch_size": self.c[f"{role}_batch_size"], "seed": self.protocol["seed"],
            "lr": self.c[f"{role}_lr"], "critic_lr": self.c["critic_lr"],
            "tokens": self.c[f"{role}_tokens"], "temperature": self.c["editor_temperature"] if role == "editor" else 1.0,
            "staleness": self.c["staleness"], "harness": snapshot["harness"],
            "data": str(data), "evaluator_url": evaluator, "central_context": None,
            "interaction_credit": 0.0, "weight_update_mode": self.c["weight_update_mode"],
            "actor_gpus": 2 if role == "editor" else 4,
            "rollout_gpus": 2 if role == "editor" else 4,
            "recycle_pool": str(recycle) if recycle else None, "critic_warmup_steps": warmup,
        }
        immutable(run / f"{phase}_spec.json", spec)
        previous = completed_optimizer(run, spec)
        if previous is None:
            immutable(run / f"{phase}_optimizer_intent.json", spec)
            self.monitor.update("training", domain=self.domain, phase=phase, run=str(run),
                                handoff=HANDOFF)
            process = self.launch.module("sao_bridge.large_train",
                                         ["--spec", run / f"{phase}_spec.json"],
                                         gpus="0,1,2,3" if role == "editor" else "0,1,2,3,4,5,6,7",
                                         name=f"{run.name}_{phase}_train")
            timeout = min(self.protocol["runtime"]["phase_timeout_seconds"],
                          self.monitor.deadline - time.monotonic())
            self.launch.wait(process, phase, timeout=timeout)
            previous = completed_optimizer(run, spec)
        snapshot = copy.deepcopy(snapshot)
        for label, key in (("actor", role), ("critic", f"{role}_critic")):
            snapshot[key] = previous["checkpoints"][label]
            snapshot[key + "_sha256"] = initial_model_digest(Path(snapshot[key]))
        return snapshot

    def phase(self, arm: str, role: str, run: Path, snapshot: dict):
        phase = ("h" if role == "H" else "w") + str(snapshot["h_count" if role == "H" else "w_count"] + 1)
        marker = run / f"{phase}_COMPLETE.json"
        before = digest(snapshot)
        immutable(run / f"{phase}_input.json", snapshot)
        if marker.exists():
            record = read_json(marker)
            if record["before_sha256"] != before:
                raise ValueError("Phase continuation does not match the saved E/W/H state")
            spec = read_json(run / f"{phase}_spec.json")
            completed_optimizer(run, spec)
            if phase_evidence(run, phase) != record["evidence_sha256"]:
                raise ValueError("Completed phase receipts/spec/data changed")
            verify_snapshot(record["after"])
            return record["after"]
        intent = run / f"{phase}_optimizer_intent.json"
        if intent.exists():
            completed_optimizer(run, read_json(intent))
        after = self.h_phase(arm, phase, run, snapshot) if role == "H" else self.w_phase(arm, phase, run, snapshot)
        immutable(marker, {"before_sha256": before, "after": after, "handoff": HANDOFF,
                           "evidence_sha256": phase_evidence(run, phase)})
        return after

    def h_phase(self, arm, phase, run, snapshot):
        from .experiment import propose
        from .phase_data import editor_dataset
        incumbent = Path(snapshot["harness"]).read_text()
        h_index = snapshot["h_count"]
        immutable(run / f"{phase}_reward_window.json",
                  reward_window(h_index, arm, self.c["reward_questions"]))
        with self.server(snapshot, f"{run.name}_{phase}_frozen_solver", ledger_root=run) as url:
            selection = self.evaluate(url, snapshot, "dev", self.c["selection_samples"],
                                      run / f"{phase}_before_select.json", "selection")
            feedback_report = self.evaluate(url, snapshot, "train", 1,
                                           run / f"{phase}_feedback.json", "training_feedback",
                                           rows=self.rows["train"][:self.c["feedback_questions"]])
            feedback = training_feedback(feedback_report)
            data = run / f"{phase}_editor_data.json"
            if not data.exists():
                replay = arm == "ma_sao" and snapshot.get("last_h")
                rows = editor_dataset(
                    incumbent, 0.0, (self.c["editor_steps"] + 1) * self.c["editor_batch_size"],
                    h_index, replay_dir=Path(snapshot["last_h"]["receipts"]) if replay else None,
                    replay_fraction=self.a["replay_fraction"] if replay else 0,
                    editor_step_offset=self.c["editor_steps"] if replay else 0,
                    domain=self.domain, failure_feedback=feedback,
                )
                immutable(data, rows)
            immutable(run / f"{phase}_data_sha.json", file_sha(data))
            out = run / phase
            intent = run / f"{phase}_optimizer_intent.json"
            if intent.exists() and not (out / "trained.json").exists():
                raise ValueError("Refusing partial optimizer resume")
            evaluator = None
            try:
                if not (out / "trained.json").exists():
                    evaluator = self.launch.module("sao_bridge.evaluation", [
                        "--domain", self.domain, "--url", url, "--data", self.data / "reward.json",
                        "--harness", snapshot["harness"], "--output", run / f"{phase}_reward_evals",
                        "--tokens", self.c["solver_tokens"], "--limit", self.c["reward_questions"],
                        "--offset", (h_index % 4) * self.c["reward_questions"],
                        "--port", 18802, "--record-tokens", "--concurrency", self.c["eval_concurrency"],
                    ], gpus="", name=f"{run.name}_{phase}_reward")
                    self.launch.ready(evaluator, "http://127.0.0.1:18802/health")
                after = self.train(run, phase, "editor", snapshot, data, "http://127.0.0.1:18802")
            finally:
                if evaluator is not None:
                    self.launch.stop(evaluator)
            audit = editor_pair_audit(run, phase, {**self.c, "batch_size": self.c["editor_batch_size"]})
            immutable(run / f"{phase}_pair_audit.json", audit)
            proposals_path = run / f"{phase}_proposals.json"
            if proposals_path.exists():
                candidates = read_json(proposals_path)
            else:
                self.monitor.update("proposing", domain=self.domain, phase=phase, run=str(run))
                with self.launch.vllm(after["editor"], port=18803, gpus="0",
                                      max_model_len=self.c["context_tokens"],
                                      name=f"{run.name}_{phase}_proposals") as editor_url:
                    candidates = asyncio.run(propose(
                        editor_url, incumbent, self.c["proposals"], self.c["editor_tokens"],
                        self.protocol["seed"] + 1000 + h_index * 100,
                        domain=self.domain, feedback=feedback,
                    ))
                immutable(proposals_path, candidates)
            if len(candidates) != self.c["proposals"]:
                raise ValueError("Incomplete fixed proposal slots")
            immutable(run / f"{phase}_proposals_sha.json", file_sha(proposals_path))
            outcomes = []
            for slot, candidate in enumerate(candidates):
                source = candidate["source"] if candidate["valid"] else incumbent
                harness = run / f"{phase}_candidate_{slot}.py"
                write_source(harness, source)
                trial = {**after, "harness": str(harness)}
                outcome = self.evaluate(url, trial, "dev", self.c["selection_samples"],
                                        run / f"{phase}_candidate_{slot}_select.json", "selection")
                outcomes.append(outcome)
            selection_record = select_harness(
                incumbent, selection, candidates, outcomes,
                seed=self.protocol["seed"], proposals=self.c["proposals"])
            after = self.report_harness_choice(run, phase, after, incumbent, selection,
                                               selection_record, url)
            after.update(h_count=h_index + 1,
                         last_h={"run": str(run), "phase": phase, "receipts": str(out / "receipts")})
        return after

    def report_harness_choice(self, run, phase, snapshot, incumbent, control, selection, url):
        """Commit H only after its full dev report is free of runtime hook errors."""
        selection = copy.deepcopy(selection)
        chosen = selection["selected"]
        proposed = run / f"{phase}_proposed.py"
        write_source(proposed, chosen["source"])
        output = run / f"{phase}_proposed_dev_report.json"
        reported = self.evaluate(url, {**snapshot, "harness": str(proposed)}, "dev",
                                 self.c["report_samples"], output, "dev_report")
        errors = harness_error_count(reported)
        validation = {
            "proposed_report": str(output), "proposed_report_sha256": file_sha(output),
            "harness_error_episodes": errors, "fallback_evaluation_rollouts": 0,
            "status": "accepted", "committed": False,
        }
        selection["report_validation"] = validation
        if errors:
            selection["rejected_after_report"] = copy.deepcopy(chosen)
            if chosen["source"] == incumbent:
                validation["status"] = "incumbent_runtime_error"
                immutable(run / f"{phase}_selection.json", selection)
                raise RuntimeError("Incumbent full-dev report has runtime hook errors; no safe H handoff")
            chosen = {"source": incumbent, "accuracy": control["accuracy"], "lcb": 0, "slot": None}
            selection["selected"] = chosen
            output = run / f"{phase}_incumbent_dev_report.json"
            reported = self.evaluate(url, snapshot, "dev", self.c["report_samples"], output, "dev_report")
            validation.update(
                status="retained_incumbent_after_runtime_rejection",
                incumbent_report=str(output), incumbent_report_sha256=file_sha(output),
                fallback_evaluation_rollouts=len(reported["results"]),
                note="Failed proposed report retained as diagnostic physical overhead; "
                     "only the validated incumbent report informs stopping/best state.",
            )
            if harness_error_count(reported):
                validation["status"] = "incumbent_runtime_error"
                immutable(run / f"{phase}_selection.json", selection)
                raise RuntimeError("Fallback incumbent full-dev report has runtime hook errors; no safe H handoff")
        harness = run / f"{phase}.py"
        write_source(harness, chosen["source"])
        validation["committed"] = True
        immutable(run / f"{phase}_selection.json", selection)
        return {**snapshot, "harness": str(harness), "harness_sha256": file_sha(harness),
                "dev_report": str(output), "dev_report_sha256": file_sha(output)}

    def w_phase(self, arm, phase, run, snapshot):
        from .phase_data import solver_dataset
        from .replay import recycle_pool
        count = self.c["solver_steps"] * self.c["solver_batch_size"]
        offset = snapshot["w_count"] * count
        rows = self.rows["train"]
        namespace = digest({"campaign_identity": self.identity, "solver_run": str(run.resolve()), "phase": phase})
        epoch_rows, window = training_window(rows, count, offset, namespace)
        source = run / f"{phase}_train_window.json"
        immutable(source, epoch_rows)
        immutable(run / f"{phase}_train_window_audit.json", window)
        recycle, warm, mixed = None, [], []
        if arm == "ma_sao":
            previous = snapshot["last_h"]
            h_run, h_phase = Path(previous["run"]), previous["phase"]
            audit = editor_pair_audit(h_run, h_phase, {**self.c, "batch_size": self.c["editor_batch_size"]})
            immutable(h_run / f"{h_phase}_pair_audit.json", audit)
            namespace = recycling_namespace(self.identity, h_run, h_phase)
            pool = recycle_pool(h_run / f"{h_phase}_reward_evals", context_tokens=self.c["context_tokens"],
                                request_indices=set(audit["committed_request_indices"]),
                                namespace=namespace)
            warm, mixed, selected = recycling_partition(pool, self.c, self.a)
            recycle = run / f"{phase}_recycle.json"
            immutable(recycle, selected)
            immutable(run / f"{phase}_recycling_audit.json", {
                "warmup_unique": len(warm), "actor_mixed_unique": len(mixed), "overlap": 0,
                "planned_max_reuse": 1, "available_committed": len(pool),
                "context_tokens": self.c["context_tokens"], "source_audit": audit,
                "namespace": namespace,
            })
        data = run / f"{phase}_solver_data.json"
        if not data.exists():
            solver_dataset(source, data, count, recycled_ids=mixed or None, warmup_ids=warm or None,
                           warmup_rows=self.a["critic_warmup_steps"] * self.c["solver_batch_size"] if warm else 0)
        immutable(run / f"{phase}_data_sha.json", file_sha(data))
        after = self.train(run, phase, "solver", snapshot, data, recycle=recycle)
        after["w_count"] += 1
        with self.server(after, f"{run.name}_{phase}_report", full_node=True, ledger_root=run) as url:
            report = run / f"{phase}_dev_report.json"
            self.evaluate(url, after, "dev", self.c["report_samples"], report, "dev_report")
        after["dev_report"], after["dev_report_sha256"] = str(report), file_sha(report)
        return after


def write_source(path: Path, source: str):
    if path.exists() and path.read_text() != source:
        raise ValueError("Saved harness source changed")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    path.write_text(source)
    path.chmod(0o644)


def verify_snapshot(snapshot):
    for role in ("editor", "solver", "editor_critic", "solver_critic"):
        if initial_model_digest(Path(snapshot[role])) != snapshot[f"{role}_sha256"]:
            raise ValueError("Saved phase checkpoint changed")
    if file_sha(Path(snapshot["harness"])) != snapshot["harness_sha256"]:
        raise ValueError("Saved accepted harness changed")
    if file_sha(Path(snapshot["dev_report"])) != snapshot["dev_report_sha256"]:
        raise ValueError("Saved development report changed")


def ledger(root: Path) -> dict:
    physical = Counter()
    optimizer = Counter()
    for path in root.glob("blocks/*/[hw]*/receipts/*.json"):
        row = read_json(path)
        role = row["role"]
        physical[f"{role}_{'replayed' if row.get('replayed') else 'generated'}"] += 1
        physical["training_generated_tokens"] += row.get("generated_tokens", 0)
        physical["recycled_episode_attachments_including_prefetch"] += len(row.get("recycled_partner_episodes", []))
    for path in root.glob("blocks/*/*_reward_evals/*.json"):
        row = read_json(path)
        physical["harness_scoring_rollouts"] += len(row["results"])
        physical["harness_scoring_tokens"] += row["tokens"]
    for path in root.glob("blocks/*/*.json"):
        row = read_json(path)
        if isinstance(row, dict) and "cache_identity" in row:
            purpose = row["cache_identity"]["purpose"]
            physical[purpose + "_rollouts"] += len(row["results"])
            physical[purpose + "_tokens"] += row["tokens"]
        elif path.name.endswith("_proposals.json"):
            physical["editor_proposal_rollouts"] += len(row)
            physical["editor_proposal_tokens"] += sum(
                (record.get("usage") or {}).get("completion_tokens", 0) for record in row)
    report = root / "test.json"
    if report.exists():
        result = read_json(report)
        physical["test_report_rollouts"] += len(result["results"])
        physical["test_report_tokens"] += result["tokens"]
    for path in root.glob("blocks/*/[hw]*/trained.json"):
        receipt = read_json(path)
        optimizer["actor_updates"] += receipt["actor_steps"]
        optimizer["critic_updates"] += receipt["critic_steps"]
        optimizer["critic_only_warmup_updates"] += receipt["critic_only_warmup_steps"]
        optimizer["training_seconds"] += receipt["elapsed_seconds"]
        phase_spec = read_json(path.parent.parent / f"{path.parent.name}_spec.json")
        optimizer["allocated_training_gpu_seconds"] += receipt["elapsed_seconds"] * (
            phase_spec["actor_gpus"] + phase_spec["rollout_gpus"])
        for line in (path.parent / "metrics.jsonl").read_text().splitlines():
            metrics = json.loads(line)["metrics"]
            optimizer["native_optimizer_sequences"] += metrics.get("ppo_actor/n_seqs", 0)
    result = {"physical_rollouts": dict(physical), "optimizer_compute": dict(optimizer),
              "inference_compute": inference_ledger(root),
              "handoff": HANDOFF,
              "limitation": "Completed receipts include prefetch; cancelled in-flight tokens unmetered. "
                            "Matched actor updates do NOT imply equal optimizer compute: MA adds critic warmup."}
    save(root / "LEDGERS.json", result)
    return result


def initial_state(snapshot: dict, score: float):
    return {"blocks": 0, "current": snapshot, "best": copy.deepcopy(snapshot), "best_score": score,
            "anchor_score": score, "stale": 0, "history": [], "status": "active"}


class Campaign:
    def __init__(self, protocol, host, monitor):
        self.p, self.host, self.monitor = protocol, host, monitor
        self.node = protocol["nodes"][host]
        self.root = Path(self.node["root"])
        self.peer = Path(protocol["nodes"][self.node["peer"]]["root"])

    def prepare(self, domain):
        root = self.root / domain
        identity = audit_identity(self.p, domain)
        immutable(root / "identity.json", identity)
        return root, identity

    def baseline(self, domain, root, identity):
        from .domain_api import get_domain
        model_key = self.node["baseline"]
        source = root / "initial.py"
        write_source(source, get_domain(domain).INITIAL_SOURCE)
        snapshot = {"solver": self.p["models"][model_key], "solver_sha256": identity["models"][model_key],
                    "harness": str(source)}
        path = root / f"baseline_{model_key}_dev.json"
        marker = root / "BASELINE_READY.json"
        if not marker.exists():
            layout = baseline_server_layout(root, self.p["common"]["context_tokens"])
            save(root / f"BASELINE_EXECUTION_{time.time_ns()}.json", {
                "host": self.host, "domain": domain, "model": model_key,
                "source_sha256": implementation_identity(),
                "evaluation_components": identity["evaluation_implementation_sha256"],
                "solver_server_layout": layout,
                "note": "Execution provenance only; training/controller files are not baseline cache identity.",
            })
            self.monitor.update("baseline", domain=domain, model=model_key)
            with allocation(self.root, self.p, self.monitor, domain) as launch:
                engine = Engine(self.p, domain, root, identity, launch, self.monitor)
                with engine.server(
                    snapshot, f"{domain}_{model_key}_baseline",
                    full_node=layout["allocation"] == "full_node", ledger_root=root / "baseline",
                ) as url:
                    engine.evaluate(url, snapshot, "dev", self.p["common"]["baseline_samples"], path, "dev_report")
            if implementation_identity(domain) != identity["evaluation_implementation_sha256"]:
                raise ValueError("Evaluation implementation changed while baseline was running")
            immutable(marker, {"identity": identity, "model": model_key, "report": str(path),
                               "report_sha256": file_sha(path)})
        verify_baseline(marker, identity, self.p)
        save(root / "BASELINE_GPU_LEDGER.json", inference_ledger(root / "baseline"))

    def wait(self, marker: Path, domain: str, single=None):
        deadline = time.monotonic() + self.p["runtime"]["peer_timeout_seconds"]
        wall_deadline = time.time() + self.p["runtime"]["peer_timeout_seconds"]
        while not marker.is_file():
            self.monitor.check()
            if time.monotonic() >= deadline:
                raise TimeoutError("Peer barrier deadline; local phases remain durable")
            peer_status = self.peer / "STATUS.json"
            if peer_status.exists() and read_json(peer_status)["status"] == "failed":
                raise RuntimeError("Peer failed; resume only from verified phase boundaries")
            if single is not None and self.single_step(*single):
                continue
            # No Launcher exists while waiting: no GPU reservation, filler restored.
            self.monitor.update("peer_barrier", domain=domain, waiting_for=str(marker),
                                progress_deadline_at=wall_deadline)
            time.sleep(self.p["runtime"]["heartbeat_seconds"])
        return marker

    def freeze_initializer(self, domain, root, identity):
        peer_marker = self.wait(self.peer / domain / "BASELINE_READY.json", domain)
        outcomes = [verify_baseline(path, identity, self.p)
                    for path in (root / "BASELINE_READY.json", peer_marker)]
        reports = {record["model"]: report for record, report in outcomes}
        if set(reports) != {"original", "sft1152"}:
            raise ValueError("Both raw and SFT baseline outcomes are required")
        winner = choose_initializer(reports["original"], reports["sft1152"])
        for key, report in reports.items():
            immutable(root / f"frozen_baseline_{key}_dev.json", report)
        freeze = {
            "identity": identity, "winner": winner, "model": self.p["models"][winner],
            "model_sha256": identity["models"][winner], "tie": "original",
            "baseline_reports": {key: digest(report) for key, report in reports.items()},
            "scores": {key: report["accuracy"] for key, report in reports.items()},
            "selection_split": "dev", "test_used": False,
        }
        immutable(root / "INITIALIZER_FROZEN.json", freeze)
        # Both peers independently derive identical content before any optimizer call.
        other = self.wait(self.peer / domain / "INITIALIZER_FROZEN.json", domain)
        if read_json(other) != freeze:
            raise ValueError("Peer did not freeze the same common E/W initializer")
        training = self.wait(self.peer / domain / "TRAINING_IMPLEMENTATION.json", domain)
        if read_json(training) != read_json(root / "TRAINING_IMPLEMENTATION.json"):
            raise ValueError("Peer training implementation differs")
        return freeze

    def arm_state(self, root, arm, freeze):
        path = root / arm / "STATE.json"
        if path.exists():
            state = read_json(path)
            if state["initializer_sha256"] != digest(freeze):
                raise ValueError("Arm initializer changed")
            return state
        harness = root / "initial.py"
        report = root / f"frozen_baseline_{freeze['winner']}_dev.json"
        snapshot = {"harness": str(harness), "harness_sha256": file_sha(harness),
                    "dev_report": str(report), "dev_report_sha256": file_sha(report),
                    "h_count": 0, "w_count": 0, "last_h": None}
        for role in ("editor", "solver", "editor_critic", "solver_critic"):
            snapshot[role] = freeze["model"]
            snapshot[role + "_sha256"] = freeze["model_sha256"]
        state = {**initial_state(snapshot, freeze["scores"][freeze["winner"]]),
                 "initializer_sha256": digest(freeze), "arm": arm}
        save(path, state)
        return state

    def block(self, domain, root, identity, freeze, arm):
        state = self.arm_state(root, arm, freeze)
        snapshot = state["current"]
        verify_snapshot(snapshot)
        run = root / arm / "blocks" / f"{state['blocks'] + 1:04d}"
        immutable(run / "input.json", state)
        self.monitor.update("block", domain=domain, arm=arm, block=state["blocks"] + 1)
        with allocation(self.root, self.p, self.monitor, domain) as launch:
            engine = Engine(self.p, domain, root, identity, launch, self.monitor)
            try:
                for role in phase_plan(arm, state["blocks"]):
                    snapshot = engine.phase(arm, role, run, snapshot)
            finally:
                ledger(root / arm)
        report = read_json(Path(snapshot["dev_report"]))
        updated = update_plateau(state, snapshot, report, self.p["stopping"])
        if arm in ("sao", "ma_sao"):
            updated["status"] = "active"  # Common paired rule, not unilateral stopping.
        immutable(run / "COMPLETE.json", updated)
        save(root / arm / "STATE.json", updated)
        ledger(root / arm)
        return updated

    def single_step(self, domain, root, identity, freeze):
        arm = self.node["single"]
        state = self.arm_state(root, arm, freeze)
        if state["status"] != "active":
            return False
        state = self.block(domain, root, identity, freeze, arm)
        if state["status"] != "active":
            self.finalize(root, arm, state)
        return True

    def finalize(self, root, arm, state):
        immutable(root / arm / "TRAINING_FROZEN.json", {
            "initializer_sha256": state["initializer_sha256"], "arm": arm,
            "reason": state["status"], "blocks": state["blocks"], "best": state["best"],
            "final": state["current"], "history": state["history"], "test_used": False,
            "handoff": HANDOFF,
        })

    def alternating(self, domain, root, identity, freeze):
        arm, peer_arm = self.node["alternating"], self.p["nodes"][self.node["peer"]]["alternating"]
        state = self.arm_state(root, arm, freeze)
        while state["status"] == "active":
            # An already-published block awaiting a peer must not run another optimizer.
            number = state["blocks"]
            own_marker = root / arm / f"BLOCK_{number:04d}.json"
            decision_path = root / arm / f"DECISION_{number:04d}.json"
            if decision_path.exists() and read_json(decision_path)["decision"] != "continue":
                state["status"] = read_json(decision_path)["decision"]
                save(root / arm / "STATE.json", state)
                self.finalize(root, arm, state)
                break
            if not number or decision_path.exists():
                state = self.block(domain, root, identity, freeze, arm)
                number = state["blocks"]
                own_marker = root / arm / f"BLOCK_{number:04d}.json"
                decision_path = root / arm / f"DECISION_{number:04d}.json"
            report_path = Path(state["current"]["dev_report"])
            immutable(own_marker, {"identity": identity, "initializer_sha256": digest(freeze),
                                  "arm": arm, "state": state, "report": str(report_path),
                                  "report_sha256": file_sha(report_path)})
            peer_path = self.peer / domain / peer_arm / own_marker.name
            self.wait(peer_path, domain, single=(domain, root, identity, freeze))
            peer = read_json(peer_path)
            if (peer["identity"] != identity or peer["initializer_sha256"] != digest(freeze)
                    or peer["arm"] != peer_arm or peer["state"]["blocks"] != number
                    or file_sha(Path(peer["report"])) != peer["report_sha256"]):
                raise ValueError("Peer block identity/report mismatch")
            reports = {arm: read_json(report_path), peer_arm: read_json(Path(peer["report"]))}
            states = {arm: state, peer_arm: peer["state"]}
            baseline = read_json(root / f"frozen_baseline_{freeze['winner']}_dev.json")
            decision = extension_decision(baseline, reports["sao"], reports["ma_sao"],
                                          states["sao"], states["ma_sao"], self.p["stopping"])
            immutable(decision_path, decision)
            if decision["decision"] != "continue":
                state["status"] = decision["decision"]
                save(root / arm / "STATE.json", state)
                self.finalize(root, arm, state)
        return state

    def report(self, domain, root, identity, freeze, arm):
        frozen = read_json(root / arm / "TRAINING_FROZEN.json")
        if frozen["initializer_sha256"] != digest(freeze) or frozen["test_used"]:
            raise ValueError("Reporting requires a matching dev-only final freeze")
        marker = root / arm / "COMPLETE.json"
        snapshot = frozen["best"]
        verify_snapshot(snapshot)
        test = root / arm / "test.json"
        if not marker.exists():
            with allocation(self.root, self.p, self.monitor, domain) as launch:
                engine = Engine(self.p, domain, root, identity, launch, self.monitor)
                with engine.server(snapshot, f"{domain}_{arm}_test", full_node=True,
                                   ledger_root=root / arm) as url:
                    result = engine.evaluate(url, snapshot, "test", 8, test, "test_report")
            immutable(marker, {"frozen_sha256": file_sha(root / arm / "TRAINING_FROZEN.json"),
                               "test_sha256": file_sha(test), "test_accuracy": result["accuracy"],
                               "metric": self.p["domains"][domain]["metric"],
                               "benchmark_adaptations": self.p["domains"][domain].get("adaptations", {}),
                               "test_question_count": result["question_count"], "test_samples": 8,
                               "reason": frozen["reason"], "test_decisions": False,
                               "ledgers": ledger(root / arm)})
        else:
            result = read_json(marker)
            if (result["frozen_sha256"] != file_sha(root / arm / "TRAINING_FROZEN.json")
                    or result["test_sha256"] != file_sha(test)):
                raise ValueError("Frozen result/report changed")

    def baseline_report(self, domain, root, identity, freeze):
        model = self.node["baseline"]
        snapshot = {"solver": self.p["models"][model],
                    "solver_sha256": identity["models"][model],
                    "harness": str(root / "initial.py")}
        marker, output = root / "BASELINE_REPORT.json", root / f"baseline_{model}_test.json"
        if not marker.exists():
            with allocation(self.root, self.p, self.monitor, domain) as launch:
                engine = Engine(self.p, domain, root, identity, launch, self.monitor)
                with engine.server(snapshot, f"{domain}_{model}_baseline_test", full_node=True,
                                   ledger_root=root / "baseline") as url:
                    report = engine.evaluate(url, snapshot, "test", 8, output, "test_report")
            baseline = read_json(root / f"frozen_baseline_{model}_dev.json")
            immutable(marker, {
                "initializer_sha256": digest(freeze), "model": model, "test_decisions": False,
                "test_sha256": file_sha(output), "test_accuracy": report["accuracy"],
                "physical_rollouts": {"baseline_dev": len(baseline["results"]),
                                      "baseline_test": len(report["results"])},
                "optimizer_updates": 0, "metric": self.p["domains"][domain]["metric"],
                "benchmark_adaptations": self.p["domains"][domain].get("adaptations", {}),
                "inference_compute": inference_ledger(root / "baseline"),
            })
        elif (read_json(marker)["test_sha256"] != file_sha(output)
              or read_json(marker)["initializer_sha256"] != digest(freeze)):
            raise ValueError("Frozen baseline report changed")

    def run_domain(self, domain):
        immutable(self.root / domain / "TRAINING_IMPLEMENTATION.json", {
            "domain": domain, "source_sha256": training_implementation_identity(domain),
            "protocol_sha256": digest(self.p), "handoff": HANDOFF,
        })
        root, identity = self.prepare(domain)
        self.baseline(domain, root, identity)
        freeze = self.freeze_initializer(domain, root, identity)
        self.alternating(domain, root, identity, freeze)
        while self.single_step(domain, root, identity, freeze):
            pass
        # Neither report is opened until both local arms have stopped on dev.
        for arm in (self.node["alternating"], self.node["single"]):
            state = self.arm_state(root, arm, freeze)
            self.finalize(root, arm, state)
            self.report(domain, root, identity, freeze, arm)
        self.baseline_report(domain, root, identity, freeze)
        immutable(root / "DOMAIN_COMPLETE.json", {
            "identity": identity, "initializer_sha256": digest(freeze),
            "baseline_report_sha256": file_sha(root / "BASELINE_REPORT.json"),
            "arms": {arm: file_sha(root / arm / "COMPLETE.json")
                     for arm in (self.node["alternating"], self.node["single"])},
        })


def verify_baseline(marker, identity, protocol):
    record = read_json(marker)
    if record["identity"] != identity or record["model"] not in protocol["models"]:
        raise ValueError("Peer baseline protocol/model/split identity differs")
    path = Path(record["report"])
    if file_sha(path) != record["report_sha256"]:
        raise ValueError("Published baseline report changed")
    from .domain_api import get_domain
    domain = identity["domain"]
    rows = read_json(Path(protocol["domains"][domain]["data"]) / "dev.json")
    cache = evaluation_identity(identity, identity["models"][record["model"]],
                                get_domain(domain).INITIAL_SOURCE, "dev", rows,
                                protocol["common"]["baseline_samples"], protocol["seed"] + 5_000_003,
                                "dev_report")
    return record, verify_report(path, rows, cache)


@contextmanager
def worker_claim(root: Path, protocol: dict, resume: bool):
    root.mkdir(parents=True, exist_ok=True, mode=0o755)
    if root.stat().st_uid != os.getuid():
        raise PermissionError("Output root must be owned by the assigned local UID")
    with (root / "worker.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another campaign process owns this node") from exc
        claim = root / "worker_claim.json"
        if claim.exists() and not resume:
            raise ValueError("Persistent campaign exists; use --resume for audited recovery")
        immutable(root / "protocol.json", protocol)
        save(claim, {"host": os.uname().nodename, "pid": os.getpid(), "started_at": time.time(),
                     "protocol_sha256": digest(protocol)})
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["baseline", "worker", "status", "monitor"])
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--domain", choices=["chess", "searchqa"],
                        help="baseline/worker: execute only this ready domain; monitor: scope --until baseline")
    parser.add_argument("--until", choices=["baseline", "campaign"], default="campaign",
                        help="monitor completion target; default waits through baseline handoff")
    parser.add_argument("--stale-seconds", type=float, default=180)
    parser.add_argument("--interval", type=float, default=15)
    args = parser.parse_args()
    protocol = read_json(args.protocol)
    validate_protocol(protocol)
    if args.domain and not (args.command in {"baseline", "worker"} or (
            args.command == "monitor" and args.until == "baseline")):
        parser.error("--domain requires baseline/worker or monitor --until baseline")
    if args.command == "status":
        for host, node in protocol["nodes"].items():
            path = Path(node["root"]) / "STATUS.json"
            print(json.dumps({"host": host, "status": read_json(path) if path.exists() else "not_started"}))
        return
    host = os.uname().nodename
    if host not in protocol["nodes"] or os.getuid() != protocol["nodes"][host]["uid"]:
        raise PermissionError("Campaign can run only under its assigned node3/node4 UID")
    os.umask(0o022)
    root = Path(protocol["nodes"][host]["root"])
    if args.command == "monitor":
        if args.stale_seconds <= 0 or args.interval <= 0:
            parser.error("Monitor intervals must be positive")
        raise SystemExit(watch(protocol, host, until=args.until,
                               stale_seconds=args.stale_seconds, interval=args.interval, domain=args.domain))

    def interrupted(signum, _):
        raise InterruptedError(f"Campaign interrupted by signal {signum}; phase-boundary resume only")
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    with worker_claim(root, protocol, args.resume), Monitor(
        root, protocol["runtime"]["heartbeat_seconds"], protocol["runtime"]["safety_seconds"],
        protocol["runtime"]["phase_timeout_seconds"],
    ) as monitor:
        campaign = Campaign(protocol, host, monitor)
        try:
            domains = requested_domains(protocol, args.command, args.domain)
            monitor.update("starting", requested_domains=domains,
                           completed_domains=completed_domain_names(protocol, root),
                           protocol_domain_order=protocol["domain_order"])
            if args.command == "worker":
                save(root / f"WORKER_EXECUTION_{time.time_ns()}.json", {
                    "host": host, "pid": os.getpid(), "requested_domains": domains,
                    "explicit_scope": args.domain, "protocol_domain_order": protocol["domain_order"],
                    "protocol_sha256": digest(protocol), "source_sha256": implementation_identity(),
                    "note": "Full execution provenance; matched training code is frozen separately "
                            "per domain, excluding unused other-domain modules.",
                })
            for domain in domains:
                if args.command == "baseline":
                    domain_root, identity = campaign.prepare(domain)
                    campaign.baseline(domain, domain_root, identity)
                else:
                    campaign.run_domain(domain)
            completion = execution_completion(protocol, root, args.command, domains)
            monitor.update(completion.pop("status"), **completion)
        except BaseException as exc:
            monitor.update("failed", error=repr(exc), resume="verified phase boundaries only")
            raise
        finally:
            public_tree(root)


if __name__ == "__main__":
    main()
