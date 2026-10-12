"""Matched native quick evaluation of the official Engineer and frozen H1 proposals."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from .alfworld_matched_engineer_baseline import OFFICIAL_LABEL, validate_inputs
from .alfworld_strong_select import generate_proposals, matrix_config, matrix_reports, run_matrix
from .alfworld_train_data import _immutable_json
from .common import file_sha, read_json, write_json
from .launch import Launcher
from .resources import gpu_release_audit, require_research_node


def summarize_proposals(reports: dict, proposals: dict) -> dict:
    result = {}
    for label, rows in proposals.items():
        if [row["sample_index"] for row in rows] != list(range(8)):
            raise ValueError("Quick comparison requires all eight ordered proposal slots")
        names = [f"{label}-{index:02d}" for index in range(8)]
        selected = min(names, key=lambda name: (-reports[name]["accuracy"], name))
        result[label] = {
            "proposal_mean_accuracy": sum(reports[name]["accuracy"] for name in names) / 8,
            "best_selection_accuracy": reports[selected]["accuracy"],
            "selected_label": selected,
            "valid_proposals": sum(row["valid"] for row in rows),
            "total_proposals": 8,
        }
    return result


def run(request_path: Path) -> dict:
    require_research_node()
    request = read_json(request_path)
    reference, reference_spec, _, _, _, _, _ = validate_inputs(Path(request["verified_reference_request"]))
    protocol = read_json(Path(request["protocol"]))
    state = read_json(Path(request["initial_state"]))
    root = Path(request["run"])
    if request["protocol_name"] != "alfworld-core-quick-v1":
        raise ValueError("Unknown core quick evaluation protocol")
    if protocol["evaluation"]["selection_count"] != 24 or protocol["evaluation"]["samples"] != 2:
        raise ValueError("Core table requires the declared 24-task mean@2 protocol")
    _immutable_json(root / "identity.json", {
        "request": request, "request_sha256": file_sha(request_path),
        "protocol_sha256": file_sha(Path(request["protocol"])),
        "initial_state_sha256": file_sha(Path(request["initial_state"])),
        "official_model": reference["official_model"],
        "solver": reference_spec["solver_model"],
    })
    audit = gpu_release_audit()
    if audit["foreign_gpu_pids"] or not audit["filler_unit_active"]:
        raise RuntimeError("Quick comparison requires released GPUs with filler active")
    launch = Launcher(root / "supervision")
    started = time.time()
    try:
        reference_root = Path(protocol["common_prefix"]["proposal_file"]).parent.parent
        frozen = generate_proposals(
            launch, protocol, state, {"incumbent": {"actor": state["engineer"]}}, reference_root,
        )["incumbent"]
        official = generate_proposals(
            launch, protocol, state,
            {OFFICIAL_LABEL: {"actor": reference["official_model"]}}, root,
        )[OFFICIAL_LABEL]
        if read_json(root / "generation_prompts.json") != read_json(reference_root / "generation_prompts.json"):
            raise ValueError("Official and H1 Engineer inputs differ")
        proposals = {OFFICIAL_LABEL: official, "sao_h1": frozen}
        harnesses = {"frozen_h1": state["harness"]}
        for label, rows in proposals.items():
            for row in rows:
                name = f"{label}-{row['sample_index']:02d}"
                path = Path(protocol["no_patch"])
                if row["valid"]:
                    path = root / "patches" / f"{name}.json"
                    _immutable_json(path, row["patch"])
                harnesses[name] = str(path)
        config = matrix_config(
            protocol, reference_spec["solver_model"], "w0-core-official",
            harnesses, root / "matrix",
        )
        matrix, output = run_matrix(launch, config, root / "matrix", 2)
        reports = matrix_reports(matrix, protocol)
        summaries = summarize_proposals(reports, proposals)
        selected = summaries[OFFICIAL_LABEL]["selected_label"]
        result = {
            "status": "completed", "question_count": 24, "samples_per_question": 2,
            "no_patch_accuracy": reports["no_patch"]["accuracy"],
            "frozen_h1_accuracy": reports["frozen_h1"]["accuracy"],
            "engineers": summaries, "official_selected_harness": harnesses[selected],
            "all_scores": {label: report["accuracy"] for label, report in reports.items()},
            "native_matrix": str(output), "elapsed_seconds": time.time() - started,
            "limitations": [
                "Selection-development scores, not an independent report or the paper's 500-task result.",
                "Both Engineers use identical historical W79 evidence, proposal seeds and decoding budgets.",
                "Eight invalid/duplicate slots retained; means and best-of-eight are separate metrics.",
                "H1 proposals are frozen before this comparison; historical training/search budgets differ.",
            ],
        }
        _immutable_json(root / "COMPLETE.json", result)
        return result
    except BaseException as exc:
        write_json(root / "FAILED.json", {"error": repr(exc), "failed_at_unix": time.time()})
        raise
    finally:
        launch.close()
        write_json(root / "RELEASED.json", gpu_release_audit())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    print(run(parser.parse_args().request), flush=True)
