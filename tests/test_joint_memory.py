from pathlib import Path
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/sao/vendor/AReaL"))

import torch
from sao_bridge.joint_critic import expand_joint_context
from sao_bridge.joint_memory_check import (
    ACTIONS, BATCH, LENGTH, MemoryActor, make_config, synthetic_batch, tensor_summary,
)


class JointMemoryTests(unittest.TestCase):
    def test_step_end_cache_release_is_only_for_joint_learner(self):
        from areal.engine.fsdp_engine import FSDPPPOActor
        from sao_bridge.terminal_actor import TerminalFSDPPPOActor

        engine = object.__new__(TerminalFSDPPPOActor)
        expected = object()
        for joint, optimizer, released in (
            ("1", object(), True), ("1", None, False), ("0", object(), False),
        ):
            engine.config = SimpleNamespace(optimizer=optimizer)
            with (
                patch.dict(os.environ, {"MA_SAO_JOINT_EPISODES": joint}),
                patch.object(FSDPPPOActor, "clear_batches", return_value=expected) as native,
                patch("sao_bridge.terminal_actor.current_platform") as platform,
            ):
                self.assertIs(engine.clear_batches(["shard"]), expected)
                native.assert_called_once_with(["shard"])
                self.assertEqual(platform.empty_cache.call_count, int(released))
        engine.config = SimpleNamespace(optimizer=object())
        with (
            patch.dict(os.environ, {"MA_SAO_JOINT_EPISODES": "1"}),
            patch.object(FSDPPPOActor, "clear_batches", side_effect=RuntimeError("cleanup failed")),
            patch("sao_bridge.terminal_actor.current_platform") as platform,
        ):
            with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                engine.clear_batches()
            platform.empty_cache.assert_not_called()

    def test_full_context_and_realistic_action_budget_are_not_shortened(self):
        rows = synthetic_batch(42)
        self.assertEqual(len(rows), BATCH)
        for row in rows:
            self.assertEqual(row["input_ids"].shape, (1, 32768))
            self.assertEqual(int(row["attention_mask"].sum()), LENGTH)
            self.assertEqual(int(row["loss_mask"].sum()), ACTIONS)
            self.assertEqual(int(row["loss_mask"].sum()), 4096)
            self.assertFalse(bool(row["loss_mask"][0, 0]))
            self.assertEqual(int(row["input_ids"][0, -1]), 42)
            expanded, prefix = expand_joint_context(row)
            self.assertEqual(prefix, 1024)
            self.assertEqual(expanded["input_ids"].shape[-1], 33792)
            self.assertEqual(int(expanded["loss_mask"].sum()), ACTIONS)
        self.assertFalse(torch.equal(rows[0]["input_ids"], rows[1]["input_ids"]))
        self.assertTrue(torch.equal(rows[0]["input_ids"], synthetic_batch(42)[0]["input_ids"]))

    def test_output_audit_is_deterministic_and_detects_changes(self):
        values = torch.tensor([[1.0, -2.0]])
        first = tensor_summary(values)
        self.assertTrue(first["finite"])
        self.assertEqual(first, tensor_summary(values.clone()))
        self.assertNotEqual(first["sha256_float32"], tensor_summary(values + 0.1)["sha256_float32"])

    def test_native_config_keeps_four_dp_and_colocated_reference(self):
        repo = Path(__file__).resolve().parents[1]
        if not (repo / "runs/alfworld_joint_async_6h_v5/solver/optimizer/trainer.yaml").exists():
            self.skipTest("Archived failed-run configuration is required for this integration test")
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"SAO_LOCAL_ADMIN_TOKEN": "unit-test"},
        ):
            config = make_config(repo, Path(directory))
            h0 = make_config(repo, Path(directory), model_role="h0", rounds=3)
            compact = make_config(repo, Path(directory), reference_storage_dtype="bfloat16")
            no_context = make_config(repo, Path(directory), critic_cooperative_context=False)
            no_norm = make_config(repo, Path(directory), value_normalization=False)
        self.assertEqual(MemoryActor.__module__, "sao_bridge.joint_memory_check")
        self.assertEqual(config.actor.backend, "fsdp:d4p1t1")
        self.assertEqual(config.actor.mb_spec.n_mbs, 2)
        self.assertEqual(config.actor.mb_spec.max_tokens_per_mb, 32768)
        self.assertIsNone(config.ref.optimizer)
        self.assertTrue(config.ref.scheduling_strategy.fork)
        self.assertFalse(config.critic.scheduling_strategy.fork)
        self.assertFalse(config.enable_offload)
        self.assertEqual(compact.ref.optimizer_dtype, "bfloat16")
        self.assertEqual(compact.actor.optimizer_dtype, config.actor.optimizer_dtype)
        self.assertEqual(compact.critic.optimizer_dtype, config.critic.optimizer_dtype)
        self.assertIsNone(compact.ref.optimizer)
        self.assertEqual(config.actor.path, str(repo / "assets/Harness-R1/target-agent-sft"))
        self.assertNotIn("MA_SAO_ASYNC_JOINT_ROOT", config.actor.scheduling_spec[0].env_vars)
        for key in ("AREAL_CACHE_DIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
            self.assertTrue(Path(config.actor.scheduling_spec[0].env_vars[key]).is_relative_to(
                Path(directory) / "runtime-cache"))
        self.assertEqual(h0.total_train_steps, 3)
        for role in ("actor", "critic", "ref"):
            selected = getattr(h0, role)
            self.assertEqual(selected.path, str(repo / "assets/Harness-R1/harness-r1"))
            self.assertEqual(selected.backend, "fsdp:d4p1t1")
            self.assertEqual(selected.mb_spec.n_mbs, 2)
        self.assertEqual(h0.actor.mb_spec.max_tokens_per_mb, 32768)
        self.assertEqual(h0.critic.mb_spec.max_tokens_per_mb, 34816)
        for configured, context, norm in ((config, "1", "1"), (no_context, "0", "1"), (no_norm, "1", "0")):
            env = configured.actor.scheduling_spec[0].env_vars
            self.assertEqual(env["MA_SAO_COOPERATIVE_CONTEXT"], context)
            self.assertEqual(env["MA_SAO_VALUE_NORM"], norm)
            self.assertEqual(configured.actor.mb_spec.max_tokens_per_mb, 32768)
            self.assertEqual(configured.gconfig.max_new_tokens, config.gconfig.max_new_tokens)


if __name__ == "__main__":
    unittest.main()
