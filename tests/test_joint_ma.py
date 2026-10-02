import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/sao/vendor/AReaL"))

import torch
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.infra.workflow_executor import check_trajectory_format
from areal.utils.data import batched_call
from sao_bridge.common import AREAL_REVISION, SAO_REVISION, read_json, verify_upstreams, write_json
from sao_bridge.joint_critic import attach_joint_context, expand_joint_context
from sao_bridge.joint_round import checked_behavior, install_version_publication, publish_version, record_failure, wait_versions
from sao_bridge.joint_workflow import JointVLLMConfig, JointWorkflow


class JointTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_joint_vllm_returns_actual_tempered_behavior_probabilities(self):
        from areal.api.cli_args import vLLMConfig
        from areal.infra.rpc.serialization import deserialize_value, serialize_value
        from vllm.config import ModelConfig

        self.assertEqual(ModelConfig.__dataclass_fields__["logprobs_mode"].default, "raw_logprobs")
        original = vLLMConfig(model="initial-policy", max_num_seqs=16)
        joint = JointVLLMConfig(**asdict(original))
        args = vLLMConfig.build_args(joint, tp_size=1, pp_size=1)
        command = vLLMConfig.build_cmd_from_args(args)
        self.assertEqual(args["logprobs_mode"], "processed_logprobs")
        self.assertEqual(command[command.index("--logprobs-mode") + 1], "processed_logprobs")
        self.assertEqual(args["max_num_seqs"], original.max_num_seqs)
        self.assertNotIn("logprobs_mode", asdict(original))
        restored = deserialize_value(serialize_value(joint))
        self.assertIsInstance(restored, JointVLLMConfig)
        self.assertEqual(asdict(restored), asdict(joint))
        with self.assertRaisesRegex(ValueError, "processed"):
            JointVLLMConfig(logprobs_mode="raw_logprobs")

    def test_first_failure_is_preserved_under_concurrent_peer_cancellation(self):
        original = {"error": "original environment failure"}
        record_failure(self.root, "solver", original)
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(
                lambda i: record_failure(self.root, "solver", {"error": f"peer cancellation {i}"}),
                range(32),
            ))
        self.assertEqual(read_json(self.root / "solver_FAILED.json"), original)
        self.assertFalse((self.root / "solver_FAILED.json.tmp").exists())

    def test_explicit_stop_cancellation_does_not_poison_checkpoint_recovery(self):
        write_json(self.root / "STOP.json", {"reason": "evaluation deadline"})
        record_failure(self.root, "solver", {"error": "worker terminated during shutdown"})
        self.assertFalse((self.root / "solver_FAILED.json").exists())
        self.assertEqual(
            read_json(self.root / "solver_CANCELLED.json")["stop"],
            {"reason": "evaluation deadline"},
        )
        write_json(self.root / "editor_FAILED.json", {"error": "earlier real failure"})
        record_failure(self.root, "editor", {"error": "later cancellation"})
        self.assertEqual(read_json(self.root / "editor_FAILED.json"), {"error": "earlier real failure"})

    def test_environment_uses_parent_ownership_and_never_closes_native_launcher(self):
        from sao_bridge.joint_environment import run
        spec = self.root / "spec.json"
        data = self.root / "data.json"
        write_json(spec, {"joint_root": str(self.root), "data": str(data), "steps": 1, "batch_size": 1})
        write_json(data, [{"joint_slot": 0, "task_id": 7}])
        current = self.root / "round-0000"
        write_json(current / "proposal-000.json", {"patch": {"benchmark": "alfworld", "actions": []}})
        write_json(current / "outcome-000.json", {"reward": 1})
        write_json(current / "environment/READY.json", {"ready": True})
        launch = Mock()
        run(spec, launch)
        launch.module.assert_called_once()
        launch.stop.assert_called_once_with(launch.module.return_value)
        launch.close.assert_not_called()
        self.assertTrue((current / "ENVIRONMENT_RELEASED.json").exists())

    def test_canonical_git_trust_does_not_depend_on_user_home(self):
        canonical = self.root / "canonical/sao"
        areal = canonical / "vendor/AReaL"
        areal.mkdir(parents=True)
        alias = self.root / "snapshot"
        alias.symlink_to(canonical, target_is_directory=True)
        with (
            patch("sao_bridge.common.SAO", alias),
            patch("sao_bridge.common.AREAL", alias / "vendor/AReaL"),
            patch("sao_bridge.common.subprocess.check_output", side_effect=[SAO_REVISION, AREAL_REVISION]) as rev,
            patch("sao_bridge.common.subprocess.run") as git,
            patch("sao_bridge.common.file_sha", return_value="digest"),
        ):
            result = verify_upstreams()
        self.assertEqual(result["sao"], SAO_REVISION)
        self.assertEqual(result["areal"], AREAL_REVISION)
        self.assertIn(f"safe.directory={canonical}", rev.call_args_list[0].args[0])
        self.assertIn(f"safe.directory={areal}", rev.call_args_list[1].args[0])
        for call in git.call_args_list:
            self.assertIn(f"safe.directory={areal}", call.args[0])
            self.assertNotIn("safe.directory=*", call.args[0])

    def test_joint_barrier_requires_both_versions_and_propagates_failures(self):
        publish_version(self.root, "editor", 0)
        with self.assertRaises(TimeoutError):
            asyncio.run(wait_versions(self.root, 0, timeout=0.01))
        publish_version(self.root, "solver", 0)
        asyncio.run(wait_versions(self.root, 0, timeout=0.01))
        write_json(self.root / "solver_FAILED.json", {"error": "infrastructure failure"})
        with self.assertRaisesRegex(RuntimeError, "partner failed"):
            asyncio.run(wait_versions(self.root, 1, timeout=1))

    def test_publication_follows_actual_rollout_version_update(self):
        trainer = SimpleNamespace(rollout=SimpleNamespace(set_version=Mock()))
        install_version_publication(trainer, self.root, "editor")
        self.assertTrue((self.root / "round-0000/editor_ready.json").exists())
        trainer.rollout.set_version(1)
        self.assertEqual(read_json(self.root / "round-0001/editor_ready.json"), {"role": "editor", "version": 1})
        trainer = SimpleNamespace(rollout=SimpleNamespace(set_version=Mock(side_effect=RuntimeError("weight update failed"))))
        install_version_publication(trainer, self.root, "solver")
        with self.assertRaises(RuntimeError):
            trainer.rollout.set_version(1)
        self.assertFalse((self.root / "round-0001/solver_ready.json").exists())

    def test_behavior_is_authentic_and_does_not_cross_partner_round(self):
        call = {"response_ids": [1, 2], "response_logprobs": [-0.1, -0.3], "versions": [3, 3]}
        checked_behavior([call], 3)
        for changed in (
            {**call, "versions": [2, 3]}, {**call, "response_logprobs": [-0.1]},
            {**call, "response_logprobs": [float("nan"), -0.3]},
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                checked_behavior([changed], 3)

    @staticmethod
    def trajectory():
        return {
            "input_ids": torch.tensor([[5, 6, 7, 8, 9, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0, 0]]).bool(),
            "loss_mask": torch.tensor([[0, 0, 0, 1, 1, 0, 0]]),
            "logprobs": torch.tensor([[0., 0., 0., -0.1, -0.2, 0., 0.]]),
            "versions": torch.tensor([[-1, -1, -1, 0, 0, -1, -1]]),
            "rewards": torch.tensor([1.]),
        }

    def test_context_is_critic_only_masks_prefix_and_survives_padding(self):
        row = self.trajectory()
        original = {key: value.clone() for key, value in row.items()}
        annotated = attach_joint_context(row, [0, 20, 30])
        self.assertTrue(check_trajectory_format(annotated))
        for key in original:
            self.assertTrue(torch.equal(annotated[key], original[key]))
        expanded, n = expand_joint_context(annotated)
        self.assertEqual(n, 3)
        self.assertEqual(expanded["input_ids"][0, :3].tolist(), [0, 20, 30])
        self.assertEqual(expanded["loss_mask"][0, :3].tolist(), [0, 0, 0])
        self.assertEqual(expanded["versions"][0, :3].tolist(), [-1, -1, -1])
        self.assertNotIn("joint_context_ids", expanded)
        self.assertEqual(int(expanded["loss_mask"].sum()), 2)
        self.assertEqual(expanded["rewards"].item(), 1)
        packed = batched_call(lambda data: data, [annotated, attach_joint_context(row, [4, 5])])
        self.assertEqual([expand_joint_context(item)[1] for item in packed], [3, 2])
        annotated["joint_context_ids"][0, 5] = 7
        with self.assertRaisesRegex(ValueError, "contiguous"):
            expand_joint_context(annotated)
        with self.assertRaisesRegex(ValueError, "token budget"):
            attach_joint_context(row, list(range(1025)))

    def test_two_live_role_workflows_share_one_outcome_and_separate_probabilities(self):
        for role in ("editor", "solver"):
            publish_version(self.root, role, 0)
        write_json(self.root / "round-0000/ENVIRONMENT_READY.json", {"version": 0})
        clients = []

        class Client:
            def __init__(client, **kwargs):
                client.role = len(clients)
                clients.append(client)
                client.chat = SimpleNamespace(completions=SimpleNamespace(create=client.create))
                client.interaction = InteractionWithTokenLogpReward(
                    model_response=SimpleNamespace(
                        input_tokens=list(range(100)), output_tokens=[101, 102],
                        output_logprobs=[-0.1, -0.2], output_versions=[0, 0],
                        stop_reason="stop", input_len=100, output_len=2,
                    ),
                    chat_template_type="concat",
                )

            async def create(client, **kwargs):
                return SimpleNamespace(
                    id=str(client.role),
                    choices=[SimpleNamespace(message=SimpleNamespace(content="model action"))],
                    model_dump=lambda: {"choices": []},
                )

            def set_reward(client, completion_id, reward):
                client.interaction.reward = reward

            def export_interactions(client, style):
                return {str(client.role): client.interaction}

        async def episode(row, harness, generate, **kwargs):
            await generate([], None, 4096, 42)
            return {"reward": 1., "tokens": 2, "task_id": row["task_id"]}

        data = {
            "id": "paired-0", "joint_version": 0, "joint_slot": 0,
            "task_id": 7, "task_context": "pick up cup", "critic_evidence": "Prior W79 missed a cup.",
            "messages": [{"role": "user", "content": "historical failure packet"}],
        }
        tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [4, 8, 12])
        with (
            patch("sao_bridge.joint_workflow.load_hf_tokenizer", return_value=tokenizer),
            patch("sao_bridge.joint_workflow.ArealOpenAI", Client),
            patch("sao_bridge.joint_workflow.parse_proposal", return_value=({"benchmark": "alfworld", "actions": []}, "invalid patch")),
            patch("sao_bridge.joint_workflow.domain.run_episode", side_effect=episode),
        ):
            workers = [
                JointWorkflow(role, "model", SimpleNamespace(temperature=1.), str(self.root))
                for role in ("editor", "solver")
            ]

            async def both():
                return await asyncio.gather(*(worker.arun_episode(None, data) for worker in workers))

            editor, solver = asyncio.run(both())
        self.assertEqual(editor["rewards"].item(), 1.)
        self.assertEqual(solver["rewards"].item(), 1.)
        self.assertEqual(int(editor["loss_mask"].sum()), 2)
        self.assertEqual(int(solver["loss_mask"].sum()), 2)
        for role in ("editor", "solver"):
            receipt = read_json(self.root / f"round-0000/{role}-receipt-000.json")
            self.assertEqual(receipt["calls"][0]["versions"], [0, 0])
            self.assertEqual(receipt["model_error"], "invalid patch")
            self.assertNotIn("reward", receipt["context"])
            self.assertNotIn("action", receipt["context"])
        self.assertEqual(read_json(self.root / "round-0000/outcome-000.json")["sample_id"], "paired-0")


if __name__ == "__main__":
    unittest.main()
