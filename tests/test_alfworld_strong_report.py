from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.alfworld_strong_report import compare, frozen_states, report_config, run, wait_reports
from sao_bridge.budget import contract_digest
from sao_bridge.common import file_sha, read_json, write_json
from sao_bridge.evaluation import stable_id_seed_offset


class StrongReportTests(unittest.TestCase):
    def fixture(self, root, *, extra_successes=7, noisy=False, identical_policy=False):
        manifest = root / "manifest.json"
        ids = list(range(300, 370))
        write_json(manifest, {"selection_task_ids": [240, 241], "report_task_ids": ids})
        empty = root / "empty.json"
        write_json(empty, {"benchmark": "alfworld", "actions": []})
        protocol = {
            "run": str(root), "no_patch": str(empty), "initial_solver": "w0", "max_phase_attempts": 2,
            "phase_orders": {"sequential": ["h2", "w1", "w2"], "ma": ["w1", "h2", "w2"]},
            "evaluation": {
                "task_manifest": str(manifest), "samples": 2, "seed": 42, "temperature": 1.0,
                "solver_tokens": 4096, "eval_concurrency": 128, "environment_concurrency": 32,
                "matrix_concurrency": 4, "minimum_ma_gain": 0.05, "bootstrap_seed": 17,
                "bootstrap_draws": 1000,
            },
        }
        protocol_path = root / "frozen_protocol.json"
        write_json(protocol_path, protocol)
        states = {}
        for arm in ("sequential", "ma"):
            state = {
                "solver": "same-w" if identical_policy else arm + "-w",
                "harness": str(empty), "harness_sha256": contract_digest(read_json(empty)),
                "protocol_sha256": file_sha(protocol_path),
            }
            states[arm] = state
            write_json(root / arm / "FINAL_STATE.json", state)
            write_json(root / arm / "ARM_FROZEN.json", {
                "status": "frozen", "report_opened": False, "protocol_sha256": file_sha(protocol_path),
                "final_state_sha256": file_sha(root / arm / "FINAL_STATE.json"),
                "solver_model": state["solver"],
                "solver_weights_sha256": "same-weights" if identical_policy else arm + "-weights",
                "harness_sha256": state["harness_sha256"], "phase_order": protocol["phase_orders"][arm],
            })
        freeze_hashes = {arm: file_sha(root / arm / "ARM_FROZEN.json") for arm in states}

        def report(candidate):
            rows = []
            for index, task_id in enumerate(ids):
                for sample in range(2):
                    if noisy:
                        reward = float(index >= 33) if candidate else float(index < 30)
                    else:
                        reward = float(index < extra_successes and sample == 0) if candidate else 0.0
                    rows.append({
                        "id": f"alfworld-{task_id}", "task_id": task_id, "sample_index": sample,
                        "rollout_seed": 42 + stable_id_seed_offset(f"alfworld-{task_id}") + 1009 * sample,
                        "reward": reward, "tokens": 10,
                    })
            return {
                "metric": "mean@2", "question_count": 70, "samples_per_question": 2,
                "accuracy": sum(row["reward"] for row in rows) / len(rows), "results": rows,
            }

        for arm in states:
            report_root = root / arm / "final_report"
            groups = {}
            for group in ([arm, "initial_w0"] if arm == "ma" else [arm]):
                values = {}
                for harness in ("no_patch", "common_h1", "sequential_final", "ma_final"):
                    path = report_root / group / f"{harness}.json"
                    write_json(path, report(group == "ma" and harness == "ma_final"))
                    values[harness] = {"path": str(path), "sha256": file_sha(path)}
                groups[group] = {
                    "solver": "w0" if group == "initial_w0" else states[group]["solver"],
                    "solver_weights_sha256": (
                        "initial" if group == "initial_w0" else
                        "same-weights" if identical_policy else group + "-weights"
                    ),
                    "reports": values,
                }
            write_json(report_root / "COMPLETE.json", {
                "status": "completed", "identity": {
                    "protocol_sha256": file_sha(protocol_path), "arm": arm, "arm_freezes": freeze_hashes,
                }, "report_groups": groups,
            })
            write_json(report_root / "RELEASED.json", {
                "foreign_gpu_pids": [], "remaining_gpu_pids": [], "verified_filler_pids": [],
                "filler_unit_active": True,
            })
        return protocol_path, protocol, states

    def test_report_waits_for_both_freezes_before_any_gpu_or_weight_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, _, _ = self.fixture(root)
            (root / "ma/ARM_FROZEN.json").unlink()
            with (
                patch("sao_bridge.alfworld_strong_report.require_research_node"),
                patch("sao_bridge.alfworld_strong_report.Launcher") as launch,
                patch("sao_bridge.alfworld_strong_report.model_digests") as model,
                self.assertRaisesRegex(FileNotFoundError, "both arms freeze"),
            ):
                run(path, "sequential")
            launch.assert_not_called()
            model.assert_not_called()
            self.assertFalse((root / "sequential/final_report/REPORT_OPENING.json").exists())

    def test_report_config_uses_only_seventy_report_questions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, protocol, states = self.fixture(root)
            config = report_config(protocol, states, root / "matrix", states["ma"]["solver"], "ma")
            self.assertEqual(config["task_ids"], list(range(300, 370)))
            self.assertEqual(set(config["harnesses"]), {"common_h1", "sequential_final", "ma_final"})
            self.assertEqual(config["samples"], 2)

    def test_comparison_wait_requires_both_release_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, _, _ = self.fixture(root)
            wait_reports(path, timeout=1)
            (root / "ma/final_report/RELEASED.json").unlink()
            with (
                patch("sao_bridge.alfworld_strong_report.time.monotonic", side_effect=[0, 2]),
                self.assertRaisesRegex(TimeoutError, "released GPUs"),
            ):
                wait_reports(path, timeout=1)

    def test_exact_five_point_threshold_and_paired_positive_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, _, _ = self.fixture(Path(temporary), extra_successes=7)
            result = compare(path)
            self.assertEqual(result["paired_final_system"]["candidate_minus_control"], 0.05)
            self.assertGreater(result["paired_final_system"]["paired_question_bootstrap_95pct"][0], 0)
            self.assertTrue(result["gate_passed"])
            self.assertEqual(set(result["report_matrix"]), {"sequential", "ma", "initial_w0"})
            self.assertTrue((Path(temporary) / "PAIRED_REPORT.json").is_file())
        with tempfile.TemporaryDirectory() as temporary:
            path, _, _ = self.fixture(Path(temporary), extra_successes=6)
            self.assertFalse(compare(path)["gate_passed"])
        with tempfile.TemporaryDirectory() as temporary:
            path, _, _ = self.fixture(Path(temporary), noisy=True)
            result = compare(path)
            self.assertGreater(result["paired_final_system"]["candidate_minus_control"], 0.05)
            self.assertLessEqual(result["paired_final_system"]["paired_question_bootstrap_95pct"][0], 0)
            self.assertFalse(result["gate_passed"])

    def test_freeze_and_report_hashes_are_not_relabelled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, _, _ = self.fixture(root)
            freeze = read_json(root / "ma/ARM_FROZEN.json")
            freeze["protocol_sha256"] = "changed"
            write_json(root / "ma/ARM_FROZEN.json", freeze)
            with self.assertRaisesRegex(ValueError, "freeze identity"):
                frozen_states(path)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, _, _ = self.fixture(root)
            complete_path = root / "ma/final_report/COMPLETE.json"
            complete = read_json(complete_path)
            complete["identity"]["arm_freezes"]["sequential"] = "changed"
            write_json(complete_path, complete)
            with self.assertRaisesRegex(ValueError, "protocol or completion"):
                compare(path)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, _, _ = self.fixture(root)
            record = root / "ma/final_report/ma/ma_final.json"
            report = read_json(record)
            report["accuracy"] = 1.0
            write_json(record, report)
            with self.assertRaisesRegex(ValueError, "after publication"):
                compare(path)

    def test_identical_policy_cannot_win_from_evaluation_variation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, _, _ = self.fixture(Path(temporary), extra_successes=7, identical_policy=True)
            result = compare(path)
            self.assertEqual(result["paired_final_system"]["candidate_minus_control"], 0.05)
            self.assertGreater(result["paired_final_system"]["paired_question_bootstrap_95pct"][0], 0)
            self.assertTrue(result["identical_final_policy"])
            self.assertFalse(result["gate_passed"])
            self.assertIn("evaluation variation", result["identity_interpretation"])


if __name__ == "__main__":
    unittest.main()
