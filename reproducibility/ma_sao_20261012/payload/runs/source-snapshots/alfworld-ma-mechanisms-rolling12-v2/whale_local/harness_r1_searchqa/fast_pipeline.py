"""Persistent matched raw/SFT/RL pilot with train triage and independent dev confirmation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from sao_bridge.common import file_sha, write_json
from sao_bridge.launch import Launcher
from .pipeline import verify_grpo_commit
from .selection import paired_gain
from .workflow import checkpoint_identity


def wait_unit(unit: str, receipt: Path, timeout: int = 3600) -> None:
    deadline = time.monotonic() + timeout
    while not receipt.is_file():
        state = subprocess.check_output(
            ["systemctl", "--user", "show", unit, "-p", "ActiveState", "--value"],
            text=True,
        ).strip()
        if state not in {"active", "activating"}:
            raise RuntimeError(f"{unit} ended without verified receipt: {state}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"{unit} receipt timeout")
        time.sleep(5)


def select_candidate(directory: Path) -> dict | None:
    candidates = []
    for path in sorted((directory / "samples").glob("*.json")):
        result = json.loads(path.read_text())
        reward = result.get("reward", {})
        if (result.get("clean_execution") is True and reward.get("hook_errors") == 0
                and reward.get("score", 0) > 0):
            candidates.append(result)
    if not candidates:
        return None
    best = max(candidates, key=lambda item: (item["reward"]["score"], item["patch_sha256"]))
    return {
        "patch": best["reward"]["merged_patch"], "generation_id": best["id"],
        "train_delta": best["reward"]["score"], "selection_split": "train",
    }


def record_results(readme: Path, results: dict) -> None:
    lines = [
        "\n## Latest automatically verified matched-control result\n",
        "This is a preliminary one-update pilot, not a convergence claim.\n",
        "| Control | Proposals | Valid | Clean | Dev mean | Accepted |\n",
        "|---|---:|---:|---:|---:|---|\n",
    ]
    for name, control in results["controls"].items():
        dev = control["dev"]
        mean = (f"{100 * dev['candidate_mean']:.4f}%"
                if "candidate_mean" in dev else "not evaluated: no clean positive candidate")
        lines.append(
            f"| {name} | {control['proposals']} | {control['valid']} | "
            f"{control['clean']} | {mean} | {dev['accepted']} |\n"
        )
    contrast = results["rl_vs_sft"]
    if contrast:
        lines.append(
            f"\nRL minus SFT: {100 * contrast['absolute_gain']:.4f} percentage points; "
            f"paired question-bootstrap 95% CI "
            f"{[100 * value for value in contrast['paired_bootstrap_ci95']]} pp.\n"
        )
    else:
        lines.append("\nRL versus SFT efficacy could not be established: missing qualified candidate.\n")
    lines.append(
        f"\nRL improvement demonstrated: **{results['rl_improvement_demonstrated']}**. "
        "Test split was not read. See the matched-control COMPLETE.json receipt.\n"
    )
    with readme.open("a") as stream:
        stream.writelines(lines)


def remote_raw_urls(ready: Path, roles_config: Path, model: Path) -> str:
    receipt = json.loads(ready.read_text())
    if receipt["roles_sha256"] != file_sha(roles_config):
        raise ValueError("Remote role configuration changed after serving readiness")
    roles = json.loads(roles_config.read_text())["roles"]
    engineer = next(role for role in roles if role["name"] == "engineer")
    if (Path(engineer["model"]).resolve() != model.resolve()
            or engineer.get("engineer_think_prefill") is not True):
        raise ValueError("Remote raw Engineer model/template differs from matched control")
    urls = receipt["endpoints"]["engineer"]
    if len(urls.split(",")) != len(engineer["gpus"].split(",")):
        raise ValueError("Remote raw Engineer replica count differs from role configuration")
    return urls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--grpo-unit", required=True)
    parser.add_argument("--grpo-output", type=Path, required=True)
    parser.add_argument("--sft-model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--excluded-packets", type=Path, nargs="+", required=True)
    parser.add_argument("--reward-config", type=Path, required=True)
    parser.add_argument("--profile-complete", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-engineer-ready", type=Path,
                        help="collect the raw control concurrently from existing node3 replicas")
    parser.add_argument("--raw-roles-config", type=Path)
    args = parser.parse_args()
    if bool(args.raw_engineer_ready) != bool(args.raw_roles_config):
        parser.error("--raw-engineer-ready and --raw-roles-config must be supplied together")
    root = Path("/mnt/3fs/xudong/coevolve")
    target_urls = ",".join(f"http://127.0.0.1:{port}/v1" for port in range(18901, 18932, 10))
    target_sha = "ed70e9c6825cc84b3c292455aa1e2c972da8b48b52f12d422cf7ccf61cbd25ae"
    args.output.mkdir(parents=True, exist_ok=False)
    status = args.output / "STATUS.json"
    launch = None
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")

    def execute(module: str, arguments: list[str]) -> None:
        subprocess.run([sys.executable, "-B", "-m", module, *arguments],
                       env=env, check=True, timeout=3600)

    def proposals(directory: Path, manifest: Path, urls: str) -> None:
        execute("harness_r1_searchqa.bootstrap", [
            "--data", str(args.data), "--engineer-urls", urls,
            "--engineer-manifest", str(manifest),
            "--reward-config", str(args.reward_config),
            "--output", str(directory / "proposals"), "--packet-count", "4",
            "--samples", "8", "--concurrency", "32",
            "--reward-concurrency", "8", "--minimum-examples", "256",
        ])

    try:
        if args.raw_engineer_ready:
            raw_model = root / "assets/Qwen3-4B-Instruct-2507"
            urls = remote_raw_urls(args.raw_engineer_ready, args.raw_roles_config, raw_model)
            directory = args.output / "raw4b"
            directory.mkdir()
            manifest = directory / "ENGINEER_IDENTITY.json"
            write_json(manifest, checkpoint_identity(raw_model))
            write_json(status, {
                "stage": "concurrent_node3_raw_control", "control": "raw4b",
                "proposals": 32, "optimizer_running_on": "node4",
            })
            proposals(directory, manifest, urls)
        write_json(status, {"stage": "waiting_for_optimizer_and_infra_receipts"})
        wait_unit(args.grpo_unit, args.grpo_output / "engine/checkpoint-000001/COMMITTED.json")
        verify_grpo_commit(args.grpo_output / "engine", 1)
        wait_unit("searchqa-harness-r1-eval-speed-node3-20260930-r2.service",
                  args.profile_complete)
        performance = json.loads(args.profile_complete.read_text())
        if not performance["no_op_acceptance_must_be_false"]:
            raise RuntimeError("Infrastructure no-op control was incorrectly accepted")
        # The completed optimizer service must release its GPUs before serving controls.
        deadline = time.monotonic() + 120
        while subprocess.check_output(
            ["systemctl", "--user", "show", args.grpo_unit, "-p", "ActiveState", "--value"],
            text=True,
        ).strip() in {"active", "activating"}:
            if time.monotonic() > deadline:
                raise TimeoutError("GRPO service did not release completed GPU workers")
            time.sleep(2)
        launch = Launcher(args.output / "processes")
        shared = args.run_root / "node3/shared_dev_baseline"
        controls = [
            ("raw4b", root / "assets/Qwen3-4B-Instruct-2507"),
            ("sft4b", args.sft_model),
            ("grpo4b", args.grpo_output / "engine/checkpoint-000001/actor"),
        ]
        results = {}
        for name, model in controls:
            directory = args.output / name
            precollected = name == "raw4b" and args.raw_engineer_ready is not None
            if not precollected:
                directory.mkdir()
            manifest = directory / "ENGINEER_IDENTITY.json"
            if precollected:
                if json.loads(manifest.read_text()) != checkpoint_identity(model):
                    raise ValueError("Raw control model changed during concurrent collection")
            else:
                write_json(manifest, checkpoint_identity(model))
            extra = ["--enable-prefix-caching"]
            if name == "raw4b":
                from transformers import AutoTokenizer
                from .sft_train import configure_engineer_template
                tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
                configure_engineer_template(tokenizer, True)
                template = directory / "template.jinja"
                template.write_text(tokenizer.chat_template)
                extra.extend(["--chat-template", str(template)])
            write_json(status, {
                "stage": "matched_engineer_proposals", "control": name,
                "packets": 4, "responses_per_packet": 8, "proposals": 32,
                "purpose": "same_prompt_seed_decode_rerun_budget",
            })
            if not precollected:
                with launch.vllm(
                    str(model), port=19101, gpus="0,1,2,3,4,5,6,7", name=name,
                    data_parallel=8, max_model_len=32768, server_args=extra,
                ) as urls:
                    proposals(directory, manifest, urls)
            selected = select_candidate(directory / "proposals")
            samples = [json.loads(path.read_text()) for path in
                       (directory / "proposals/samples").glob("*.json")]
            summary = {
                "proposals": len(samples),
                "valid": sum(sample.get("valid_patch") is True for sample in samples),
                "clean": sum(sample.get("clean_execution") is True for sample in samples),
                "selected": selected, "model_identity_sha256": file_sha(manifest),
            }
            if selected:
                patch = directory / "candidate.json"
                write_json(patch, selected["patch"])
                write_json(status, {"stage": "heldout_train_screen", "control": name,
                                    "purpose": "independent_train_generalization_not_final_acceptance"})
                execute("harness_r1_searchqa.rapid_screen", [
                    "--target-urls", target_urls, "--target-model-sha256", target_sha,
                    "--data-root", str(root / "assets/searchqa_sao_20260929"),
                    "--retrieval-url", "http://127.0.0.1:18731",
                    "--candidate", str(patch),
                    "--incumbent", str(root / "examples/searchqa_harness_r1_noop.json"),
                    "--excluded-packets", *map(str, args.excluded_packets),
                    "--output", str(directory / "train_screen"),
                ])
                write_json(status, {"stage": "independent_development_confirmation",
                                    "control": name, "episodes": 2048,
                                    "shared_baseline_reused": True})
                execute("harness_r1_searchqa.evaluate", [
                    "--target-urls", target_urls, "--target-model-sha256", target_sha,
                    "--data-root", str(root / "assets/searchqa_sao_20260929"),
                    "--retrieval-url", "http://127.0.0.1:18731", "--candidate", str(patch),
                    "--shared-baseline", str(shared), "--concurrency", "64",
                    "--output", str(directory / "dev"),
                ])
                summary["dev"] = json.loads((directory / "dev/COMPLETE.json").read_text())
            else:
                summary["dev"] = {"accepted": False, "reason": "no_clean_positive_train_candidate"}
            results[name] = summary
            write_json(args.output / "PARTIAL_RESULTS.json", results)
        contrast = None
        if all((args.output / name / "dev/candidate.json").is_file()
               for name in ("sft4b", "grpo4b")):
            before = json.loads((args.output / "sft4b/dev/candidate.json").read_text())
            after = json.loads((args.output / "grpo4b/dev/candidate.json").read_text())
            contrast = paired_gain(before["results"], after["results"])
        final = {
            "controls": results, "rl_vs_sft": contrast,
            "rl_improvement_demonstrated": bool(
                contrast and contrast["accepted"] and results["grpo4b"]["dev"]["accepted"]
            ),
            "preliminary": True, "grpo_updates": 1, "optimizer_steps": 4,
            "formal_scale_decision_requires_replication": True,
            "test_split_read": False, "performance": performance,
        }
        write_json(args.output / "COMPLETE.json", final)
        record_results(args.run_root / "README.md", final)
        write_json(status, {"stage": "small_scale_controls_completed",
                            "rl_improvement_demonstrated": final["rl_improvement_demonstrated"]})
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        write_json(status, {"stage": "blocked", "error": str(exc)})
        with (args.run_root / "README.md").open("a") as stream:
            stream.write(
                f"\n### Pipeline blocked ({args.output.name})\n\n"
                f"Error: `{type(exc).__name__}: {str(exc).replace('`', '')}`. "
                "No success or harness gain is claimed; inspect STATUS.json.\n"
            )
        raise
    finally:
        if launch is not None:
            launch.close()


if __name__ == "__main__":
    main()
