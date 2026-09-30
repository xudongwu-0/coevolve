import argparse
import json
import sys
from pathlib import Path
import shutil
import unittest
import uuid

from .driver import (TYPES, baseline_command, checked_rows, classify_results,
                     freeze_baseline, patch_is_no_op, select_cohort, ASSETS, PYTHON)
from .driver import official_paths, preserve_paired_ids


class ControlsTest(unittest.TestCase):
    def test_cohort_deterministic_six_types(self):
        paths = [f"json_2.1.1/train/{kind}-X/trial/game.tw-pddl"
                 for _ in range(8) for kind in TYPES]
        ids = select_cohort(paths, 32)
        self.assertEqual(ids, select_cohort(paths, 32))
        self.assertEqual(len(set(ids)), 32)
        self.assertEqual({paths[i].split("/")[2].split("-")[0] for i in ids}, set(TYPES))
        for size in (8, 12):
            small = select_cohort(paths, size)
            self.assertEqual(len(small), size)
            self.assertEqual({paths[i].split("/")[2].split("-")[0] for i in small}, set(TYPES))

    def test_target_exact_and_cpu_cli(self):
        args = argparse.Namespace(python=PYTHON, run_id="test", size=32,
            concurrency=4, controller_port=15240, worker_port=15340,
            target_url="http://localhost:8110/v1", target_model="Qwen3.5-9B")
        cmd = baseline_command(args, ASSETS / "runs/test")
        self.assertEqual(cmd[cmd.index("--rollout-temperature") + 1], "0.0")
        self.assertEqual(cmd[cmd.index("--rollout-tool-choice") + 1], "auto")
        self.assertEqual(json.loads(cmd[cmd.index("--rollout-chat-template-kwargs") + 1]),
                         {"enable_thinking": False})
        self.assertNotIn("--harness-enabled", cmd)

    def test_infra_not_invalid_or_noop(self):
        results = [
            {"valid_patch": False, "patch_status": "engineer_request_failed_treated_as_no_patch"},
            {"valid_patch": True, "eval_status": "eval_failed_treated_as_no_patch"},
            {"valid_patch": False, "patch_status": "invalid_patch_treated_as_no_patch"},
            {"valid_patch": True, "eval_status": "ok", "adapter_no_op": True},
            {"valid_patch": True, "eval_status": "ok"},
        ]
        self.assertEqual(classify_results(results),
                         {"infra_failure": 2, "invalid_patch": 1, "no_op": 1, "valid_evaluated": 1})

    def test_official_compile_detects_noop_and_real_edit(self):
        task = {"alfworld": {"parameters": {"tools": [{"type": "function",
            "function": {"name": "take_action", "description": "Take an action."}}]}}}
        patch = {"benchmark": "alfworld", "actions": [
            {"type": "edit_tool_hint", "tool": "take_action",
             "operation": "replace", "text": "Take an action."}]}
        self.assertTrue(patch_is_no_op(patch, task))
        patch["actions"][0]["text"] = "Use only admissible actions."
        self.assertFalse(patch_is_no_op(patch, task))

    def test_strict_prefill_real_code_hook(self):
        _, examples = official_paths()
        sys.path.insert(0, str(examples))
        from eval_alfworld_patches import parse_patch
        patch = {"benchmark": "alfworld", "actions": [
            {"type": "add_code_hook", "hook": "make_pre_hint",
             "code": "def hook(ctx, nb):\n    return {'message': 'Use admissible actions.'}\n"}]}
        raw = json.dumps(patch)
        self.assertIsNone(parse_patch(raw, "prefill_think_patch")[0])
        parsed, error = parse_patch("Reasoning.</think>\n<patch>" + raw + "</patch>",
                                   "prefill_think_patch")
        self.assertEqual(error, "")
        self.assertIsNotNone(parsed)
        self.assertFalse(patch_is_no_op(parsed, {"alfworld": {"parameters": {}}}))

    def test_sparse_packet_identity_preservation(self):
        records = [{"metadata": {"baseline_rewards": {"8": 0, "27": 1}, "batch_size": 20}}]
        preserve_paired_ids(records, [8, 27], "immutable-hash")
        metadata = records[0]["metadata"]
        self.assertEqual(metadata["batch_size"], 2)
        self.assertEqual(metadata["task_ids"], [8, 27])
        self.assertEqual(metadata["adapter_baseline_identity_sha256"], "immutable-hash")
        with self.assertRaises(RuntimeError):
            preserve_paired_ids(records, [8, 28], "different")

    def test_coverage_and_baseline_identity(self):
        ASSETS.mkdir(exist_ok=True)
        root = ASSETS / ("test_scratch_" + uuid.uuid4().hex)
        root.mkdir()
        try:
            file = root / "runs.jsonl"
            file.write_text(json.dumps({"index": 8, "output": {
                "status": "completed", "result": {"reward": 0}}}) + "\n")
            self.assertEqual(len(checked_rows(root, [8])), 1)
            original = freeze_baseline(root)
            with self.assertRaises(RuntimeError):
                checked_rows(root, [8, 9])
            file.write_text(json.dumps({"index": 8, "output": {
                "status": "task_error", "result": {"reward": 0}}}) + "\n")
            with self.assertRaises(RuntimeError):
                checked_rows(root, [8])
            self.assertNotEqual(original, freeze_baseline(root))
        finally:
            shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
