from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.alfworld_strong_select import (
    checked_matrix, matrix_config, publish_selection_binding, run_matrix,
    select_engineer, selection_binding, supplement_runtime,
)
from sao_bridge.budget import contract_digest
from sao_bridge.common import read_json, write_json


class StrongSelectionTests(unittest.TestCase):
    @staticmethod
    def proposals():
        return {
            label: [{"sample_index": index, "valid": True} for index in range(8)]
            for label in ("incumbent", "step15", "step31")
        }

    @staticmethod
    def reports():
        return {
            "no_patch": {"accuracy": 0.7},
            "incumbent_harness": {"accuracy": 0.75},
            **{f"{label}-proposal{index:02d}": {"accuracy": accuracy}
               for label, accuracy in (("incumbent", 0.65), ("step15", 0.8), ("step31", 0.7))
               for index in range(8)},
        }

    def test_two_level_selection_and_incumbent_tie_policy(self):
        reports = self.reports()
        reports["step15-proposal00"]["accuracy"] = 0.9
        result = select_engineer(reports, self.proposals(), 0.02)
        self.assertEqual(result["selected_engineer"], "step15")
        self.assertEqual(result["selected_harness"], "step15-proposal00")
        self.assertTrue(result["useful_editing_evidence"])
        self.assertAlmostEqual(result["engineer_accuracies"]["step15"], 0.8125)
        reports = {key: {"accuracy": 0.75} for key in reports}
        result = select_engineer(reports, self.proposals(), 0.02)
        self.assertEqual(result["selected_engineer"], "incumbent")
        self.assertEqual(result["selected_harness"], "incumbent_harness")

    def test_invalid_slots_remain_and_abstention_is_not_useful_editing(self):
        proposals, reports = self.proposals(), self.reports()
        reports["no_patch"]["accuracy"] = 0.8
        reports["incumbent_harness"]["accuracy"] = 0.7
        for row in proposals["step15"]:
            row["valid"] = False
        result = select_engineer(reports, proposals, 0.02)
        self.assertEqual(result["selected_harness"], "no_patch")
        self.assertEqual(result["valid_proposals"]["step15"], 0)
        self.assertFalse(result["useful_editing_evidence"])
        proposals["step15"].pop()
        with self.assertRaisesRegex(ValueError, "proposal slots"):
            select_engineer(reports, proposals, 0.02)

    def test_unselected_engineers_cannot_supply_a_lucky_harness(self):
        reports = self.reports()
        reports["step31-proposal00"]["accuracy"] = 1.0
        for index in range(1, 8):
            reports[f"step31-proposal{index:02d}"]["accuracy"] = 0.4
        result = select_engineer(reports, self.proposals(), 0.02)
        self.assertEqual(result["selected_engineer"], "step15")
        self.assertTrue(result["selected_harness"].startswith("step15"))

    def test_metadata_supplementation_does_not_rewrite_completed_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("preprocessor_config.json", "processor_config.json"):
                write_json(root / name, {})
            receipt = root / "runtime_supplements.json"
            write_json(receipt, {"historical": "must remain unchanged"})
            before = receipt.read_bytes()
            with patch("sao_bridge.alfworld_strong_select.ensure_model_runtime_metadata") as supplement:
                supplement_runtime(root, root)
                supplement.assert_not_called()
            self.assertEqual(receipt.read_bytes(), before)
            (root / "preprocessor_config.json").unlink()
            with patch("sao_bridge.alfworld_strong_select.ensure_model_runtime_metadata") as supplement:
                supplement_runtime(root, root)
                supplement.assert_called_once_with(root, root)

    def test_selection_binding_rejects_counterpart_harness_and_decode_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            write_json(manifest, {"selection_task_ids": [240, 241], "report_task_ids": [300]})
            report = root / "selected_report.json"
            write_json(report, {"fixture": "native report"})
            evaluation = {
                "task_manifest": str(manifest), "samples": 2, "seed": 42, "temperature": 1,
                "solver_tokens": 4096, "eval_concurrency": 128, "environment_concurrency": 32,
                "matrix_concurrency": 4,
            }
            protocol = {"evaluation": evaluation, "no_patch": "empty"}
            state = {"solver": str(root / "w0"), "harness_sha256": "h1",
                     "selection_report": str(report), "protocol_sha256": "frozen"}
            publish_selection_binding(protocol, state)
            frozen = read_json(report.with_suffix(".binding.json"))
            self.assertEqual(frozen, selection_binding(protocol, state))
            self.assertNotEqual(frozen, selection_binding(protocol, {**state, "solver": str(root / "w1")}))
            self.assertNotEqual(frozen, selection_binding(protocol, {**state, "harness_sha256": "h2"}))
            changed = {**protocol, "evaluation": {**evaluation, "samples": 3}}
            self.assertNotEqual(frozen, selection_binding(changed, state))
            config = matrix_config(protocol, state["solver"], "w0", {"deployed": "h1"}, root / "matrix")
            self.assertEqual(config["task_ids"], [240, 241])
            self.assertNotIn(300, config["task_ids"])

    def test_cached_matrix_checks_all_aliases_and_preserves_failed_attempts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            empty = root / "empty.json"
            write_json(empty, {"benchmark": "alfworld", "actions": []})
            config = {"run": str(root / "matrix"), "no_patch": str(empty),
                      "harnesses": {"deployed": str(empty)}, "solver_model": "w0"}
            first = root / "matrix/attempt-01"
            first.mkdir(parents=True)
            write_json(first / "FAILED.json", {"error": "infrastructure"})
            second = root / "matrix/attempt-02"
            second.mkdir()
            write_json(second / "config.json", {**config, "run": str(second)})
            write_json(second / "COMPLETE.json", {"status": "completed"})
            write_json(second / "RELEASED.json", {
                "foreign_gpu_pids": [], "remaining_gpu_pids": [], "verified_filler_pids": [],
                "filler_unit_active": True,
            })
            matrix = {"variants": {
                label: {"harness_sha256": contract_digest(read_json(empty))}
                for label in ("no_patch", "deployed")
            }}
            write_json(second / "report.json", matrix)
            launch = Mock()
            result, output = run_matrix(launch, config, root / "matrix", 2)
            self.assertEqual(output, second)
            self.assertEqual(result, matrix)
            self.assertTrue((first / "FAILED.json").is_file())
            launch.module.assert_not_called()
            matrix["variants"].pop("deployed")
            write_json(second / "report.json", matrix)
            with self.assertRaisesRegex(ValueError, "logical"):
                checked_matrix(second, {**config, "run": str(second)})


if __name__ == "__main__":
    unittest.main()
