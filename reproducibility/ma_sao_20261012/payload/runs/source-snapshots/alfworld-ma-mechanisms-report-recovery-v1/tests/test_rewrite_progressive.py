from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sao_bridge.common import write_json
from sao_bridge.rewrite_data import digest
from sao_bridge.rewrite_progressive import (
    accepted_rows, evaluation_args, execute, selection_gate, unseen_train, validate_control,
)


def comparison(gain=0.01, math_delta=0.0, lower=-0.02, legality=0.0):
    return {
        "edits": {"dev": {
            "legal": {"candidate_minus_control": legality},
            "instruction_following": {"candidate_minus_control": gain},
        }},
        "math_dev": {"candidate_minus_control": math_delta,
                     "paired_question_bootstrap_95pct": [lower, 0.02]},
        "harness_generation": {"candidate": {
            "legal_fraction": 1.0, "distinct_valid_configurations": 3,
        }},
    }


class ProgressiveTests(unittest.TestCase):
    def test_only_unseen_training_records_are_consumed(self):
        rows = [{"id": str(i), "split": "train"} for i in range(16)]
        rows.append({"id": "heldout", "split": "dev"})
        self.assertEqual([r["id"] for r in unseen_train(rows, {str(i) for i in range(8)})],
                         [str(i) for i in range(8, 16)])
        self.assertEqual(unseen_train(rows, {str(i) for i in range(16)}), [])
        with self.assertRaisesRegex(ValueError, "dropping"):
            unseen_train(rows[:-2], set())
        with self.assertRaisesRegex(ValueError, "disappeared"):
            unseen_train(rows, {"missing"})
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            unseen_train(rows + rows[:1], set())

    def test_gate_uses_development_gain_not_training_loss(self):
        result = selection_gate(comparison(), comparison())
        self.assertTrue(result["improved"])
        self.assertFalse(selection_gate(comparison(), comparison(gain=0.004))["improved"])
        self.assertFalse(selection_gate(comparison(), comparison(gain=0.005))["improved"])
        self.assertTrue(selection_gate(comparison(), comparison(gain=0, math_delta=0.01))["improved"])

    def test_regression_and_uncertain_retention_are_not_success(self):
        result = selection_gate(comparison(math_delta=-0.10, lower=-0.15), comparison())
        self.assertTrue(result["stop_for_regression"])
        self.assertFalse(result["eligible"])
        result = selection_gate(comparison(lower=-0.04), comparison())
        self.assertFalse(result["improved"])
        self.assertFalse(result["stop_for_regression"])
        self.assertTrue(selection_gate(comparison(legality=-0.02), comparison())["stop_for_regression"])
        self.assertTrue(selection_gate(comparison(gain=-0.02), comparison())["stop_for_regression"])

    def test_harness_collapse_and_nonfinite_metrics_fail(self):
        value = comparison()
        value["harness_generation"]["candidate"]["distinct_valid_configurations"] = 1
        self.assertFalse(selection_gate(value, comparison())["eligible"])
        value = comparison(gain=float("nan"))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            selection_gate(comparison(), value)

    def test_stage_evaluation_does_not_open_test(self):
        args = SimpleNamespace(model=Path("/base"), fixed_data=Path("/fixed"),
                               math_data=Path("/math"), math_questions=256,
                               math_samples=8, samples=8)
        self.assertIn("--dev-only", evaluation_args(args, Path("/model"), Path("/out")))
        self.assertNotIn("--dev-only", evaluation_args(args, Path("/model"), Path("/out"),
                                                       heldout=True))

    def test_accepted_responses_are_verified_and_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = root / "shard"
            shard.mkdir()
            request = {"shard": "shard", "instruction": "Original goal-driven edit",
                       "counts": {"train": 1, "dev": 0, "test": 0}}
            response = {
                "request_sha256": digest(request), "teacher": "test",
                "authorship": "fixture",
                "examples": [{"id": "one", "family": "train-one", "split": "train",
                              "format": "json", "instruction": "Set the timeout to five.",
                              "source": '{"timeout": 1}', "target": '{"timeout": 5}',
                              "changes": {"timeout": 5}}],
            }
            write_json(shard / "request.json", request)
            write_json(shard / "response.json", response)
            future = (shard / "response.json").stat().st_mtime + 10
            with patch("sao_bridge.rewrite_progressive.time.time", return_value=future):
                rows, hashes, pending = accepted_rows(root, {})
                self.assertEqual(len(rows), 1)
                self.assertEqual(pending, [])
                accepted_rows(root, hashes)
                response["authorship"] = "mutated"
                write_json(shard / "response.json", response)
                with self.assertRaisesRegex(ValueError, "changed"):
                    accepted_rows(root, hashes)
                response["request_sha256"] = "invalid"
                write_json(shard / "response.json", response)
                with self.assertRaisesRegex(ValueError, "digest"):
                    accepted_rows(root, {})

    def test_control_must_match_frozen_cohort_and_sample_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "COMPLETE.json", {})
            write_json(root / "spec.json", {"initial_model": "/wrong"})
            args = SimpleNamespace(model=Path("/base"), fixed_data=root / "COMPLETE.json",
                                   math_data=root / "COMPLETE.json", math_questions=256,
                                   samples=8, math_samples=8)
            with self.assertRaisesRegex(ValueError, "initial_model"):
                validate_control(root, args)

    def test_two_plateau_rounds_stop_with_no_repeated_training_or_test_peeking(self):
        commands, closed = [], []

        class FakeLauncher:
            def __init__(self, root):
                pass

            def module(self, module, values, **kwargs):
                commands.append((module, values))
                output = Path(values[values.index("--output") + 1])
                output.mkdir()
                if module == "torch.distributed.run":
                    (output / "checkpoint").mkdir()
                    (output / "training_state.pt").write_bytes(b"fixture")
                    write_json(output / "result.json", {
                        "status": "completed", "hf_cpu_reload_verified": True,
                        "examples_seen": 8, "training_seconds": 1,
                    })
                return object()

            def wait(self, *args, **kwargs):
                pass

            def close(self):
                closed.append(True)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "exit").write_text("0\n")
            write_json(root / "data.json", [])
            write_json(root / "campaign.json", {"targets": {}})
            now = datetime.now(timezone.utc)
            args = SimpleNamespace(
                run=root / "run", queue=root / "queue", fixed_data=root / "data.json",
                initial_training=None, initial_evaluation=None,
                control=root / "control", predecessor_exit=root / "exit", model=root / "base",
                campaign=root / "campaign.json", math_data=root / "math.json",
                train_until=(now + timedelta(hours=1)).isoformat(),
                finish_by=(now + timedelta(hours=2)).isoformat(),
                math_questions=256, math_samples=8, samples=8, minimum_new=8,
                patience=2, max_rounds=8, lr=2e-6, min_gain=0.005, math_margin=0.03,
            )
            rows = [{"id": str(i), "split": "train"} for i in range(16)]
            with patch("sao_bridge.rewrite_progressive.validate_control"), \
                    patch("sao_bridge.rewrite_progressive.stop_known"), \
                    patch("sao_bridge.rewrite_progressive.Launcher", FakeLauncher), \
                    patch("sao_bridge.rewrite_progressive.accepted_rows",
                          side_effect=[(rows[:8], {}, []), (rows, {}, [])]), \
                    patch("sao_bridge.rewrite_progressive.audit_rows",
                          return_value={"errors": [], "counts": {"train": 16}}), \
                    patch("sao_bridge.rewrite_progressive.compare",
                          return_value=comparison(gain=0)):
                execute(args)
            from sao_bridge.common import read_json
            result = read_json(args.run / "COMPLETE.json")
            self.assertEqual(result["trained_records"], 16)
            self.assertEqual(result["stop_reason"], "development_plateau")
            self.assertFalse(result["sao_ready"])
            self.assertEqual(len(commands), 4)
            self.assertNotIn("--resume-state", commands[0][1])
            self.assertIn("--resume-state", commands[2][1])
            self.assertTrue(all("--dev-only" in values for module, values in commands
                                if module == "sao_bridge.rewrite_trial"))
            first = read_json(args.run / "round_01/new_train.json")
            second = read_json(args.run / "round_02/new_train.json")
            self.assertFalse({r["id"] for r in first} & {r["id"] for r in second})
            self.assertEqual(closed, [True])


if __name__ == "__main__":
    unittest.main()
