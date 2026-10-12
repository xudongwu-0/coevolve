import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.rewrite_audit import audit_rows
from sao_bridge.retool import INITIAL_SOURCE


class RewriteAuditTests(unittest.TestCase):
    targets = {
        "train_valid_teacher_records": 1, "dev_examples": 1, "test_examples": 1,
        "minimum_independent_training_contexts": 1,
        "harness0_max_train_fraction": 0.01, "retool_family_max_train_fraction": 0.1,
    }

    def example(self, split):
        key = split + "_prompt"
        return {
            "id": split, "family": split + "_family", "split": split, "format": "json",
            "instruction": "Change the prompt to the supplied verified instruction.",
            "source": json.dumps({key: "Before: {task}"}),
            "target": json.dumps({key: "After: {task}"}),
            "changes": {key: "After: {task}"},
        }

    def test_full_counts_and_disjoint_schemas(self):
        rows = [self.example(split) for split in ("train", "dev", "test")]
        result = audit_rows(rows, self.targets, full_budget=True)
        self.assertTrue(result["passed_structural_checks"])
        self.assertEqual(result["unique_canonical_sources"]["train"], 1)
        self.assertEqual(result["train_string_edit_fraction"], 1)

    def test_partial_snapshot_is_not_full_budget(self):
        rows = [self.example("train")]
        self.assertTrue(audit_rows(rows, self.targets, full_budget=False)["passed_structural_checks"])
        self.assertFalse(audit_rows(rows, self.targets, full_budget=True)["passed_structural_checks"])

    def test_schema_leak_detected_despite_renamed_family(self):
        train = self.example("train")
        dev = {**train, "id": "different", "family": "different_family", "split": "dev",
               "instruction": "A different instruction for the same source field schema."}
        result = audit_rows([train, dev], self.targets, full_budget=False)
        self.assertIn("Source-field schema overlaps between splits", result["errors"])

    def test_duplicate_identifier_or_equivalent_document_detected(self):
        train = self.example("train")
        duplicate = {**train, "id": "different",
                     "source": json.dumps(json.loads(train["source"]), indent=4)}
        result = audit_rows([train, duplicate], self.targets, full_budget=False)
        self.assertTrue(result["duplicate_ids_or_samples"])

    def test_harness0_and_numeric_only_concentration_rejected(self):
        row = self.example("train")
        row.update(format="python", source=INITIAL_SOURCE,
                   target=INITIAL_SOURCE.replace("TOOL_OUTPUT_LIMIT = 2048", "TOOL_OUTPUT_LIMIT = 1024"),
                   changes={"TOOL_OUTPUT_LIMIT": 1024})
        result = audit_rows([row], self.targets, full_budget=False)
        self.assertEqual(result["harness0_train_fraction"], 1)
        self.assertEqual(result["retool_train_fraction"], 1)
        self.assertGreaterEqual(len(result["errors"]), 3)


if __name__ == "__main__":
    unittest.main()
