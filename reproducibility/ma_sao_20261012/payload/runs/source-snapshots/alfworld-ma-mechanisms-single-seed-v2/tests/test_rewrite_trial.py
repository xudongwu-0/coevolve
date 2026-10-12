import asyncio
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import unittest
import tempfile
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.rewrite_trial import evaluate_edits, main
from sao_bridge.retool import INITIAL_SOURCE


class EditEvaluationTests(unittest.TestCase):
    def test_evaluation_only_routes_eight_samples_without_training_or_opening_test(self):
        allocations = []
        closed = []

        class FakeLauncher:
            def __init__(self, root):
                pass

            def module(self, *args, **kwargs):
                raise AssertionError("Evaluation-only mode must not train")

            @contextmanager
            def vllm(self, model, **kwargs):
                allocations.append((model, kwargs))
                yield "http://unused"

            def check(self):
                pass

            def close(self):
                closed.append(True)

        class FakeSampler:
            def __init__(self, *args, **kwargs):
                pass

            async def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("base", "trained"):
                (root / name).mkdir()
                (root / name / "model.safetensors").write_bytes(b"fixture")
            rows = [{
                "id": split, "family": split, "split": split, "format": "json",
                "instruction": "Change the timeout to 5.", "source": '{"timeout":2}',
                "target": '{"timeout":5}', "changes": {"timeout": 5},
            } for split in ("train", "dev", "test")]
            (root / "edits.json").write_text(json.dumps(rows))
            (root / "math.json").write_text(json.dumps([{"id": "q", "question": "q", "answer": "1"}]))
            argv = [
                "rewrite_trial", "--mode", "evaluate", "--model", str(root / "trained"),
                "--origin-model", str(root / "base"), "--data", str(root / "edits.json"),
                "--output", str(root / "output"), "--math-data", str(root / "math.json"),
                "--math-questions", "1", "--samples", "8", "--math-samples", "8",
                "--edit-tokens", "2048", "--math-tokens", "8192", "--server-context", "32768",
                "--evaluation-gpus", "0,1,2,3,4,5,6,7", "--dev-only",
            ]
            edits = AsyncMock(return_value={"results": [], "legal_fraction": 1.0})
            math = AsyncMock(return_value={"results": [], "accuracy": 1.0})
            proposals = AsyncMock(return_value=[{"valid": True, "source": INITIAL_SOURCE}])
            with patch("sys.argv", argv), patch("sao_bridge.rewrite_trial.stop_known"), \
                    patch("sao_bridge.rewrite_trial.Launcher", FakeLauncher), \
                    patch("sao_bridge.rewrite_trial.Sampler", FakeSampler), \
                    patch("sao_bridge.rewrite_trial.evaluate_edits", edits), \
                    patch("sao_bridge.rewrite_trial.evaluate", math), \
                    patch("sao_bridge.rewrite_trial.propose", proposals):
                main()
            self.assertEqual(edits.await_count, 1)
            self.assertEqual(edits.await_args.args[1][0]["split"], "dev")
            self.assertEqual(edits.await_args.kwargs, {"samples": 8, "max_tokens": 2048})
            self.assertEqual(math.await_args.kwargs["samples"], 8)
            self.assertEqual(math.await_args.kwargs["tokens"], 8192)
            self.assertEqual(allocations[0][1]["data_parallel"], 8)
            self.assertEqual(allocations[0][1]["max_model_len"], 32768)
            self.assertEqual(closed, [True])
            self.assertFalse((root / "output/test_edits.json").exists())
            self.assertFalse((root / "output/training").exists())
            spec = json.loads((root / "output/spec.json").read_text())
            self.assertEqual(spec["initial_model"], str(root / "base"))
            self.assertEqual(spec["input_model"], str(root / "trained"))
            self.assertTrue((root / "output/COMPLETE.json").exists())

    def test_missing_math_data_fails_before_allocating_gpus(self):
        row = {
            "id": "test-example", "family": "heldout", "split": "test", "format": "json",
            "instruction": "Change the timeout to 5.", "source": '{"timeout":2}',
            "target": '{"timeout":5}', "changes": {"timeout": 5},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.json"
            data.write_text(json.dumps([row]))
            argv = ["rewrite_trial", "--mode", "initial", "--data", str(data),
                    "--output", str(root / "output"), "--model", str(root),
                    "--math-data", str(root / "missing.json")]
            with patch("sys.argv", argv), patch("sao_bridge.rewrite_trial.stop_known") as allocate:
                with self.assertRaises(FileNotFoundError):
                    main()
                allocate.assert_not_called()
            self.assertFalse((root / "output").exists())

    def test_counts_repeated_samples_and_records_real_failures(self):
        row = {
            "id": "test-example", "family": "heldout", "split": "test", "format": "json",
            "instruction": "Change the timeout to 5, preserving retries.",
            "source": '{"timeout": 2, "retry": true}',
            "target": '{"timeout": 5, "retry": true}', "changes": {"timeout": 5},
        }
        instances = []

        class FakeSampler:
            def __init__(self, *args, **kwargs):
                self.generated_tokens = 0
                self.seeds = []
                self.limits = []
                self.closed = False
                instances.append(self)

            async def __call__(self, messages, tools, tokens, seed):
                self.seeds.append(seed)
                self.limits.append(tokens)
                self.generated_tokens += 10
                content = row["target"] if len(self.seeds) == 1 else row["source"]
                return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}

            async def close(self):
                self.closed = True

        with patch("sao_bridge.rewrite_trial.Sampler", FakeSampler):
            report = asyncio.run(evaluate_edits("http://unused", [row], samples=2))
        self.assertEqual(report["legal_fraction"], 1.0)
        self.assertEqual(report["instruction_following_fraction"], 0.5)
        self.assertEqual(report["tokens"], 20)
        self.assertEqual(instances[0].seeds[1] - instances[0].seeds[0], 1)
        self.assertTrue(instances[0].closed)
        self.assertEqual(len(report["results"]), 2)
        with patch("sao_bridge.rewrite_trial.Sampler", FakeSampler):
            report = asyncio.run(evaluate_edits("http://unused", [row], samples=8, max_tokens=2048))
        self.assertEqual(len(report["results"]), 8)
        self.assertEqual(report["instruction_following_fraction"], 1 / 8)
        self.assertEqual(report["max_tokens"], 2048)
        self.assertEqual(instances[1].limits, [2048] * 8)
        self.assertEqual(instances[1].seeds[-1] - instances[1].seeds[0], 7)
        self.assertTrue(instances[1].closed)

    def test_transport_failure_is_not_a_zero_accuracy_fallback(self):
        class BrokenSampler:
            def __init__(self, *args, **kwargs):
                pass

            async def __call__(self, *args, **kwargs):
                raise ConnectionError("Inference unavailable")

            async def close(self):
                pass

        row = {"id": "test", "instruction": "Change timeout to 5.", "source": '{"timeout":2}'}
        with patch("sao_bridge.rewrite_trial.Sampler", BrokenSampler):
            with self.assertRaisesRegex(ConnectionError, "Inference unavailable"):
                asyncio.run(evaluate_edits("http://unused", [row]))


if __name__ == "__main__":
    unittest.main()
