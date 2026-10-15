"""Opt-in training dependencies and numerical/speed checks; no research rollouts."""
from __future__ import annotations

import argparse
import importlib.metadata
import math
from pathlib import Path
import statistics
import time

from .common import read_json, write_json


def overlay_path(spec: dict) -> Path | None:
    value = spec.get("training_kernel_overlay")
    if value is None:
        return None
    path = Path(value).resolve()
    expected = spec.get("training_kernel_versions")
    if expected != {"flash-linear-attention": "0.5.2", "fla-core": "0.5.2", "triton": "3.7.1"}:
        raise ValueError("Training acceleration requires explicit pinned kernel versions")
    found = {
        dist.metadata["Name"].lower().replace("_", "-"): dist.version
        for dist in importlib.metadata.distributions(path=[str(path)])
    }
    if not path.is_dir() or found != expected:
        raise ValueError(f"Training kernel overlay differs from its declared packages: {found}")
    if spec.get("training_kernel_validation") is None:
        raise ValueError("Training acceleration requires a passed numerical/native validation")
    validation = Path(spec["training_kernel_validation"])
    from .common import file_sha
    if spec.get("pinned_files", {}).get(str(validation)) != file_sha(validation):
        raise ValueError("Training kernel validation is not pinned")
    if read_json(validation).get("status") != "passed":
        raise ValueError("Training kernel validation did not pass")
    return path


def relative_rms(actual, expected) -> float:
    import torch

    if actual.shape != expected.shape or not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError("Kernel output shape or finite-value contract failed")
    error = (actual.float() - expected.float()).square().mean().sqrt()
    scale = expected.float().square().mean().sqrt().clamp_min(1e-8)
    return float(error / scale)


def kernel_check(output: Path) -> dict:
    from .resources import gpu_release_audit, require_research_node, research_resources_released
    from .slurm_runtime import active

    require_research_node()
    before = gpu_release_audit()
    released = research_resources_released(before) if active() else (
        not before["foreign_gpu_pids"] and before["filler_unit_active"]
    )
    if not released:
        raise RuntimeError("Kernel benchmark requires released research GPUs")
    # Importing FLA queries CUDA capabilities and can create its own GPU context.
    import torch
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

    torch.cuda.set_device(0)
    torch.manual_seed(20261006)
    results = []
    for length in (257, 4096, 32768):
        shape = (1, length, 32, 128)
        tensors = [
            torch.randn(shape, device="cuda", dtype=torch.bfloat16)
            for _ in range(3)
        ]
        tensors += [
            -torch.rand((1, length, 32), device="cuda", dtype=torch.float32),
            torch.rand((1, length, 32), device="cuda", dtype=torch.bfloat16),
        ]
        cotangent = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        measurements, outputs, gradients = {}, {}, {}
        for name, function in (("torch", torch_chunk_gated_delta_rule), ("fla", chunk_gated_delta_rule)):
            timings = []
            for repetition in range(4):
                inputs = [value.detach().clone().requires_grad_(True) for value in tensors]
                torch.cuda.synchronize()
                started = time.perf_counter()
                result, _ = function(*inputs, use_qk_l2norm_in_kernel=True)
                grads = torch.autograd.grad(result, inputs, grad_outputs=cotangent)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                if repetition:
                    timings.append(elapsed)
                outputs[name] = result.detach().cpu()
                gradients[name] = [grad.detach().cpu() for grad in grads]
                del inputs, result, grads
            measurements[name] = statistics.median(timings)
        forward_error = relative_rms(outputs["fla"], outputs["torch"])
        gradient_errors = [relative_rms(a, b) for a, b in zip(gradients["fla"], gradients["torch"])]
        row = {
            "length": length, "heads": 32, "head_dim": 128, "dtype": "bfloat16",
            "forward_relative_rms": forward_error,
            "gradient_relative_rms_q_k_v_g_beta": gradient_errors,
            "median_forward_backward_seconds": measurements,
            "speedup": measurements["torch"] / measurements["fla"],
        }
        results.append(row)
        write_json(output / "KERNEL_PROGRESS.json", results)
        if forward_error > 0.02 or max(gradient_errors) > 0.05:
            raise AssertionError(f"Kernel numerical tolerance exceeded: {row}")
        del tensors, cotangent, outputs, gradients
        torch.cuda.empty_cache()
    result = {
        "status": "passed", "cases": results, "new_research_trajectories": 0,
        "scope": "Operator forward/backward equivalence and warmed throughput, not end-to-end training speed",
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "triton", "flash-linear-attention", "fla-core")},
    }
    write_json(output / "KERNEL_COMPLETE.json", result)
    return result


def compare_native(baseline: Path, accelerated: Path, output: Path) -> dict:
    import torch

    for root in (baseline, accelerated):
        if read_json(root / "result.json")["status"] != "passed":
            raise ValueError("Both full-context native optimizer checks must pass")
    # Snapshot symlinks may spell the same immutable model differently.
    scopes = [read_json(root / "scope.json") for root in (baseline, accelerated)]
    for scope in scopes:
        if "model" in scope:
            scope["model"] = str(Path(scope["model"]).resolve())
    if scopes[0] != scopes[1]:
        raise ValueError("Native workloads differ")
    errors, timings = {}, {}
    for role in ("reference", "proximal", "values"):
        per_rank = []
        for rank in range(4):
            a = torch.load(accelerated / f"{role}-rank{rank}.pt", weights_only=True)
            b = torch.load(baseline / f"{role}-rank{rank}.pt", weights_only=True)
            if len(a) != len(b):
                raise ValueError("Native output count differs")
            actual, expected = torch.cat(a, dim=-1), torch.cat(b, dim=-1)
            error = relative_rms(actual, expected)
            mean_absolute = float((actual.float() - expected.float()).abs().mean())
            per_rank.append({"relative_rms": error, "mean_absolute": mean_absolute})
            if error > 0.05 or (role != "values" and mean_absolute > 0.05):
                raise AssertionError(f"Native {role} numerical tolerance exceeded: {per_rank[-1]}")
        errors[role] = per_rank
    for name, root in (("baseline", baseline), ("accelerated", accelerated)):
        timings[name] = {}
        for role in ("actor", "critic"):
            by_rank = []
            for rank in range(4):
                events = [read_json_line(line) for line in (root / f"{role}-rank{rank}.jsonl").read_text().splitlines()]
                starts = [e["time"] for e in events if e["event"] == "ppo_begin"]
                ends = [e["time"] for e in events if e["event"] == "ppo_end"]
                if not starts or len(starts) != len(ends):
                    raise ValueError("Incomplete native update timings")
                durations = [end - start for start, end in zip(starts, ends)]
                if not all(math.isfinite(x) and x > 0 for x in durations):
                    raise ValueError("Invalid native update timing")
                by_rank.append(durations)
            timings[name][role] = [max(values) for values in zip(*by_rank)]
    # Exclude the first update's lazy optimizer/kernel initialization equally.
    sums = {
        name: statistics.median(times["actor"][1:]) + 2 * statistics.median(times["critic"][1:])
        for name, times in timings.items()
    }
    result = {
        "status": "passed", "initial_output_errors": errors, "native_update_seconds": timings,
        "actor_plus_two_critics_seconds": sums,
        "optimizer_speedup": sums["baseline"] / sums["accelerated"],
        "new_research_trajectories": 0,
        "scope": "Full32K synthetic Actor/Critic updates; not a claim of end-to-end rollout speed",
    }
    if result["optimizer_speedup"] < 1.10:
        raise AssertionError(f"Acceleration below the predeclared 10% threshold: {result}")
    write_json(output, result)
    return result


def read_json_line(line: str) -> dict:
    import json

    return json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        kernel_check(args.output)
    except BaseException as exc:
        write_json(args.output / "FAILED.json", {"error": repr(exc), "at_unix": time.time()})
        raise


if __name__ == "__main__":
    main()
