"""Prespecified fixed-h0 and contemporary-baseline checks for role credit."""
from pathlib import Path
import time

from .alfworld_core_extend import subset_report, validate_cohort
from .alfworld_matched_grpo import EVAL_PROTOCOL
from .alfworld_strong_phase import checked_native_report
from .alfworld_strong_select import run_matrix
from .alfworld_train_data import _immutable_json
from .analysis import paired_difference
from .common import ROOT, file_sha, read_json, write_json
from .launch import Launcher
from .resources import gpu_release_audit
from .validity_credit_study import wait_files


def contrasts(reports: dict, ids: list[int]) -> dict:
    r = {m: {p: subset_report(v, ids) for p, v in variants.items()}
         for m, variants in reports.items()}
    pairs = {
        "remeasured_system_gain": (r["control"]["selected"], r["treatment"]["selected"]),
        "fixed_h0_weight_gain": (r["control"]["h0"], r["treatment"]["h0"]),
        "treatment_patch_gain": (r["treatment"]["h0"], r["treatment"]["selected"]),
        "control_patch_gain": (r["control"]["h0"], r["control"]["selected"]),
        "versus_fresh_baseline": (r["baseline"]["h2"], r["treatment"]["selected"]),
        "no_patch_weight_gain": (r["control"]["no_patch"], r["treatment"]["no_patch"]),
    }
    return {k: paired_difference(*v, seed=20261050 + i) for i, (k, v) in enumerate(pairs.items())}


def run(study: Path) -> dict:
    root = study / "diagnostics"
    launch = None
    try:
        design = read_json(study / "DIAGNOSTIC_PROTOCOL.json")
        if design["study_protocol_sha256"] != file_sha(study / "protocol.json"):
            raise ValueError("Frozen credit-study design changed")
        wait_files(study, ["PRIMARY_RESULTS.json"], timeout=10 * 3600)
        primary = read_json(study / "PRIMARY_RESULTS.json")
        if primary["status"] != "primary_complete_diagnostics_pending":
            raise ValueError("Original primary endpoint unavailable")
        for arm in ("control", "treatment"):
            release = read_json(study / arm / "matched_RELEASED.json")
            if release["foreign_gpu_pids"] or not release["filler_unit_active"]:
                raise ValueError("Primary evaluation did not release GPU")
        audit = gpu_release_audit()
        if audit["foreign_gpu_pids"] or not audit["filler_unit_active"]:
            raise RuntimeError("Diagnostic node is not released")
        cohort = read_json(EVAL_PROTOCOL)
        validate_cohort(cohort)
        baseline = read_json(Path(cohort["configs"]["w1"]))
        models, harnesses, bound = {}, {}, {}
        for arm in ("control", "treatment"):
            endpoint_path = study / arm / "matched_COMPLETE.json"
            endpoint = read_json(endpoint_path)
            protocol = read_json(study / arm / "protocol.json")
            if endpoint["selection"]["rounds"] != 32:
                raise ValueError("Diagnostic model is not the fixed endpoint")
            models[arm] = endpoint["selection"]["pair"]["solver"]["actor"]
            harnesses[arm] = {"h0": protocol["incumbent_harness"],
                              "selected": endpoint["selection"]["harness"]}
            bound[str(endpoint_path)] = file_sha(endpoint_path)
        models["baseline"] = baseline["solver_model"]
        harnesses["baseline"] = {"h2": baseline["harnesses"]["h2"]}
        for name, model in models.items():
            weights = list(Path(model).glob("*.safetensors"))
            if not weights:
                raise ValueError("Missing model weights")
            for path in [Path(model) / "config.json", *weights,
                         *(Path(p) for p in harnesses[name].values())]:
                bound[str(path)] = file_sha(path)
        _immutable_json(root / "BOUND_INPUTS.json", {"models": models, "harnesses": harnesses, "sha256": bound})
        launch = Launcher(root / "supervision")
        reports, refs, episodes = {}, {}, 0
        for name, model in models.items():
            config = {"solver_model": model, "solver_version": f"credit-diagnostic-{name}",
                      "no_patch": str(ROOT / "examples/alfworld_no_patch.json"),
                      "harnesses": harnesses[name], "task_ids": cohort["all_task_ids"], "samples": 1,
                      **{k: cohort[k] for k in ("seed", "temperature", "solver_tokens")},
                      "eval_concurrency": 128, "environment_concurrency": 32, "matrix_concurrency": 3}
            matrix, _ = run_matrix(launch, config, root / name, 1)
            episodes += 200 * len({v["harness_sha256"] for v in matrix["variants"].values()})
            reports[name], refs[name] = {}, {}
            for patch, entry in matrix["variants"].items():
                report = checked_native_report(read_json(Path(entry["report"])), cohort["all_task_ids"],
                    {"samples": 1, "seed": cohort["seed"], "solver_tokens": cohort["solver_tokens"]})
                compact = {**report, "results": [{k: row[k] for k in
                    ("id", "task_id", "sample_index", "rollout_seed", "reward", "tokens")}
                    for row in report["results"]]}
                path = root / "reports" / f"{name}-{patch}.json"
                _immutable_json(path, compact)
                reports[name][patch] = compact
                refs[name][patch] = {"accuracy": compact["accuracy"], "path": str(path), "sha256": file_sha(path)}
        protocol = read_json(study / "control/protocol.json")
        split = read_json(Path(protocol["evaluation"]["task_manifest"]))
        groups = {"all200": cohort["all_task_ids"], "selection48": split["selection_task_ids"],
                  "report152": split["report_task_ids"]}
        if set(groups["selection48"]) & set(groups["report152"]) or set(groups["all200"]) != set(groups["selection48"] + groups["report152"]):
            raise ValueError("Invalid development partition")
        result = {"status": "completed", "original_primary_results": primary,
                  "scores": refs, "contrasts": {k: contrasts(reports, ids) for k, ids in groups.items()},
                  "environment_episodes": episodes, "new_training_trajectories": 0,
                  "diagnostic_protocol_sha256": file_sha(study / "DIAGNOSTIC_PROTOCOL.json"),
                  "limitations": ["Reused development data", "Single training seed",
                                  "Diagnostic retests do not replace original primary scores"],
                  "at_unix": time.time()}
    except BaseException as exc:
        write_json(root / "FAILED.json", {"error": repr(exc), "at_unix": time.time()})
        raise
    finally:
        if launch is not None:
            launch.close()
            write_json(root / "RELEASED.json", gpu_release_audit())
    _immutable_json(root / "COMPLETE.json", result)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.study.resolve()), flush=True)
