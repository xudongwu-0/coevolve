import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.alfworld_solver_monitor import native_metrics, snapshot, watch
from sao_bridge.common import file_sha, read_json, write_json


class AlfworldSolverMonitorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.spec = {
            "total_steps": 3, "actor_steps": 2, "critic_warmup_steps": 1,
            "batch_size": 8, "terminal_episodes": True, "critic_mode": "learned",
            "solver_trajectory_style": "concat", "save_interval_steps": 3,
            "critic_updates_per_step": 1,
        }

    def rows(self):
        result = []
        for step in range(3):
            metrics = {
                "ppo_critic/update/target_value/min": 0.0,
                "ppo_critic/update/target_value/max": 1.0,
            }
            if step:
                metrics.update({
                    "ppo_actor/update/grad_norm": 0.5, "ppo_actor/n_seqs": 8,
                })
            result.append({
                "global_step": step, "recorded_at_unix": float(step), "metrics": metrics,
            })
        return result

    def metrics(self, rows, suffix=b""):
        path = self.root / "metrics.jsonl"
        path.write_bytes(b"".join(
            json.dumps(row).encode() + b"\n" for row in rows
        ) + suffix)
        return path

    def test_metrics_ignore_only_unfinished_final_append(self):
        report = native_metrics(self.metrics(self.rows()[:2], b'{"global'), self.spec)
        self.assertEqual(report["optimizer_steps"], 2)
        self.assertEqual(report["actor_updates"], 1)
        self.assertEqual(report["critic_target_range"], [0, 1])
        self.assertTrue(report["metrics_write_in_progress"])
        with self.assertRaises(ValueError):
            native_metrics(self.metrics(self.rows(), b"invalid\n"), self.spec)

    def test_rejects_native_invariant_violations(self):
        cases = [
            (0, "ppo_actor/update/grad_norm", 0.1),
            (1, "ppo_actor/update/grad_norm", float("nan")),
            (1, "ppo_actor/n_seqs", 16),
            (1, "ppo_critic/update/target_value/max", 1.01),
            (1, "ppo_critic/update/target_value/max", 1.00001),
            (1, "ppo_critic/update/target_value/min", -0.00001),
            (1, "ppo_critic/update/target_value/min", float("nan")),
            (1, "ppo_critic/update/target_value/max", float("inf")),
        ]
        for step, name, value in cases:
            with self.subTest(name=name, value=value):
                rows = self.rows()
                rows[step]["metrics"][name] = value
                with self.assertRaises(ValueError):
                    native_metrics(self.metrics(rows), self.spec)
        rows = self.rows()
        rows[2]["global_step"] = 3
        with self.assertRaisesRegex(ValueError, "contiguous"):
            native_metrics(self.metrics(rows), self.spec)
        with self.assertRaisesRegex(ValueError, "budget"):
            native_metrics(self.metrics(self.rows() + self.rows()[:1]), self.spec)

    def test_accepts_fp32_roundoff_without_clipping_raw_targets(self):
        rows = self.rows()
        rows[1]["metrics"].update({
            "ppo_critic/update/target_value/min": -3.725290298461914e-9,
            "ppo_critic/update/target_value/max": 1.0000001192092896,
        })
        report = native_metrics(self.metrics(rows), self.spec)
        self.assertEqual(report["critic_target_range"], [-3.725290298461914e-9, 1.0000001192092896])
        self.assertEqual(report["critic_target_roundoff_atol"], 1e-6)

    def make_run(self, seed):
        root = self.root / f"seed-{seed}"
        write_json(root / "spec.json", {**self.spec, "seed": seed, "run": str(root)})
        module = Path(__file__).resolve().parents[1] / "whale_local/sao_bridge"
        write_json(root / "source_manifest.json", {"terminal_actor.py": file_sha(
            module / "terminal_actor.py",
        )})
        write_json(root / "STATUS.json", {"stage": "training"})
        metrics = root / "training/optimizer/metrics.jsonl"
        metrics.parent.mkdir(parents=True)
        metrics.write_bytes(self.metrics(self.rows()).read_bytes())
        return root

    def complete(self, root):
        history = {}
        for role in ("actor", "critic"):
            checkpoint = root / role
            checkpoint.mkdir()
            (checkpoint / "model.safetensors").write_bytes(b"fixture")
            history[role] = [{"global_step": 2, "path": str(checkpoint)}]
        write_json(root / "training/optimizer/trained.json", {
            "actor_steps": 2, "critic_only_warmup_steps": 1,
            "critic_steps": 3, "checkpoint_history": history,
        })
        write_json(root / "COMPLETE.json", {
            "status": "complete", "seed": read_json(root / "spec.json")["seed"],
            "harness": "fixed_no_patch", "test_split_opened": False,
            "selected_trained_checkpoint": True,
            "paired_report": {
                "candidate_minus_control": 0.04,
                "paired_question_bootstrap_95pct": [0.01, 0.07],
            },
        })
        write_json(root / "gpu_release_audit.json", {
            "foreign_gpu_pids": [], "filler_unit_active": True,
        })

    def test_completion_requires_native_checkpoints_and_gpu_release(self):
        root = self.make_run(42)
        self.assertFalse(snapshot(root, 10)["completed_and_released"])
        self.complete(root)
        self.assertTrue(snapshot(root, 10)["completed_and_released"])
        (root / "critic/model.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "checkpoint weights"):
            snapshot(root, 10)

    def test_source_or_data_drift_cannot_be_hidden_by_completion(self):
        root = self.make_run(42)
        self.complete(root)
        write_json(root / "source_manifest.json", {"terminal_actor.py": "0" * 64})
        with self.assertRaisesRegex(ValueError, "source changed"):
            snapshot(root, 10)

    def test_release_from_a_failed_attempt_does_not_complete_the_run(self):
        root = self.make_run(42)
        self.complete(root)
        os.utime(root / "gpu_release_audit.json", (1, 1))
        self.assertFalse(snapshot(root, 10)["completed_and_released"])
        write_json(root / "gpu_release_audit.json", {
            "foreign_gpu_pids": [123], "filler_unit_active": True,
        })
        with self.assertRaisesRegex(ValueError, "release GPUs"):
            snapshot(root, 10)

    def test_frozen_dataset_hash_is_checked_before_returning_status(self):
        root = self.make_run(42)
        data = root / "data.json"
        write_json(data, [{"task_id": 0}])
        spec = read_json(root / "spec.json")
        write_json(root / "spec.json", {
            **spec, "training_data": str(data), "training_data_sha256": file_sha(data),
        })
        self.assertEqual(snapshot(root, 10)["optimizer_steps"], 3)
        write_json(data, [{"task_id": 1}])
        with self.assertRaisesRegex(ValueError, "input changed"):
            snapshot(root, 10)

    def test_unrecovered_controller_failure_is_persisted_not_scored(self):
        roots = [self.make_run(seed) for seed in (42, 43)]
        write_json(roots[0] / "FAILED.json", {"error": "controller failed"})
        protocol = self.root / "protocol.json"
        write_json(protocol, {"required_seeds": [42, 43]})
        output = self.root / "pair.json"
        with self.assertRaisesRegex(RuntimeError, "did not recover"):
            watch(roots, protocol, output, failure_grace=0)
        self.assertFalse(output.exists())
        self.assertEqual(
            read_json(self.root / "pair_monitor.json")["health"], "attention_required",
        )

    def test_watch_persists_unchanged_pair_gate(self):
        roots = [self.make_run(seed) for seed in (42, 43)]
        for root in roots:
            self.complete(root)
        protocol = self.root / "protocol.json"
        write_json(protocol, {
            "required_seeds": [42, 43], "minimum_seed_delta": 0.0,
            "minimum_mean_delta": 0.03, "minimum_significant_seeds": 1,
        })
        output = self.root / "pair.json"
        with patch("builtins.print"):
            result = watch(roots, protocol, output)
        self.assertTrue(result["gate_passed"])
        self.assertEqual(read_json(output), result)
        self.assertEqual(read_json(self.root / "pair_monitor.json")["health"], "completed")
        write_json(roots[0] / "FAILED.json", {"error": "controller failed"})
        with self.assertRaisesRegex(ValueError, "completion and failure"):
            watch(roots, protocol, output)
        self.assertEqual(
            read_json(self.root / "pair_monitor.json")["health"], "attention_required",
        )

    def test_monitor_restart_cannot_relax_the_gate(self):
        roots = [self.make_run(seed) for seed in (42, 43)]
        for root in roots:
            self.complete(root)
        protocol = self.root / "protocol.json"
        gate = {
            "required_seeds": [42, 43], "minimum_seed_delta": 0.0,
            "minimum_mean_delta": 0.03, "minimum_significant_seeds": 1,
        }
        write_json(protocol, gate)
        output = self.root / "pair.json"
        with patch("builtins.print"):
            watch(roots, protocol, output)
        original = read_json(output)
        write_json(protocol, {**gate, "minimum_mean_delta": 0.01})
        with self.assertRaisesRegex(ValueError, "Cannot resume monitoring"):
            watch(roots, protocol, output)
        self.assertEqual(read_json(output), original)

    def test_single_seed_exploration_is_not_labeled_as_replication(self):
        root = self.make_run(42)
        self.complete(root)
        protocol = self.root / "single-seed.json"
        write_json(protocol, {
            "required_seeds": [42], "minimum_seed_delta": 0.03,
            "minimum_mean_delta": 0.03, "minimum_significant_seeds": 1,
        })
        with patch("builtins.print"):
            result = watch([root], protocol, self.root / "single-result.json")
        self.assertTrue(result["gate_passed"])
        self.assertIn("single-training-seed exploratory", result["interpretation"])
        self.assertNotIn("two-training-seed", result["interpretation"])

    def test_controller_retry_is_observed_without_restarting_or_scoring_early(self):
        roots = [self.make_run(seed) for seed in (42, 43)]
        failure = roots[0] / "FAILED.json"
        write_json(failure, {"error": "temporary startup failure"})
        protocol = self.root / "protocol.json"
        write_json(protocol, {
            "required_seeds": [42, 43], "minimum_seed_delta": 0.0,
            "minimum_mean_delta": 0.03, "minimum_significant_seeds": 1,
        })
        output = self.root / "pair.json"

        def recovered(_):
            self.assertFalse(output.exists())
            self.assertEqual(
                read_json(self.root / "pair_monitor.json")["health"],
                "waiting_for_controller_retry",
            )
            failure.unlink()
            for root in roots:
                self.complete(root)

        with patch("sao_bridge.alfworld_solver_monitor.time.sleep", recovered):
            with patch("builtins.print"):
                result = watch(roots, protocol, output)
        self.assertTrue(result["gate_passed"])


if __name__ == "__main__":
    unittest.main()
