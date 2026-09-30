import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.central_critic import JointStateCritic, context_text, prepend_context
from sao_bridge.resources import SCRIPT, UNIT, is_filler
from sao_bridge.replay import interaction_reward, replay_interaction, replay_rows, validate_call
from sao_bridge.common import write_json
from sao_bridge.retool import INITIAL_SOURCE


class ResourceTests(unittest.TestCase):
    def test_only_exact_filler_identity(self):
        record = {"cmd": ["/usr/bin/python3", "-B", SCRIPT, "worker", "--gpu", "0"],
                  "cgroup": "0::/system.slice/" + UNIT}
        self.assertTrue(is_filler(record))
        for change in ({"cgroup": "0::/other.service"},
                       {"cmd": ["python", "real_training.py"]}):
            self.assertFalse(is_filler({**record, **change}))


class CentralCriticTests(unittest.TestCase):
    def test_native_value_outputs_return_to_actor_alignment(self):
        from areal.engine.fsdp_engine import FSDPPPOCritic
        critic = object.__new__(JointStateCritic)
        critic.context_ids = [10, 11]
        row = {"input_ids": torch.tensor([[4, 5, 6]]),
               "attention_mask": torch.ones((1, 3), dtype=torch.bool),
               "loss_mask": torch.tensor([[0, 1, 1]])}
        with patch.object(FSDPPPOCritic, "compute_values",
                          return_value=[torch.tensor([[10., 11., 1., 2., 3.]])]) as forward:
            values = critic.compute_values([row])
            self.assertEqual(values[0].tolist(), [[1., 2., 3.]])
            self.assertEqual(forward.call_args.args[0][0]["loss_mask"].tolist(),
                             [[0, 0, 0, 1, 1]])
        with patch.object(FSDPPPOCritic, "ppo_update") as update:
            critic.ppo_update([row])
            self.assertEqual(update.call_args.args[0][0]["loss_mask"].sum(), 2)

    def test_prefix_masks_and_alignment_without_actor_mutation(self):
        row = {
            "input_ids": torch.tensor([[4, 5, 6, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.bool),
            "loss_mask": torch.tensor([[0, 1, 1, 0]], dtype=torch.bool),
            "values": torch.tensor([[0., 1., 2., 0.]]),
            "returns": torch.tensor([[0., 3., 3., 0.]]),
        }
        original = copy.deepcopy(row)
        result = prepend_context(row, [10, 11])
        self.assertEqual(result["input_ids"].tolist(), [[10, 11, 4, 5, 6, 0]])
        self.assertEqual(result["loss_mask"].tolist(), [[False, False, False, True, True, False]])
        self.assertTrue(torch.equal(result["values"][:, 2:], row["values"]))
        for key in row:
            self.assertTrue(torch.equal(row[key], original[key]))

    def test_context_rejects_future_or_test_information(self):
        context = dict(phase="h1", actor_role="editor", partner_checkpoint="frozen",
                       harness_source="h0", calibration_accuracy=0.5,
                       calibration_split="dev", partner_version=0)
        self.assertIn("Joint pre-action", context_text(context))
        with self.assertRaises(ValueError):
            context_text({**context, "test_accuracy": 1.0})
        with self.assertRaises(ValueError):
            context_text({**context, "calibration_split": "test"})

    def test_native_value_results_are_sliced_back_to_actor_tokens(self):
        from sao_bridge.central_critic import JointStateCritic
        from areal.engine.fsdp_engine import FSDPPPOCritic
        critic = object.__new__(JointStateCritic)
        critic.context_ids = [10, 11]
        sample = {"input_ids": torch.tensor([[4, 5, 6]]),
                  "attention_mask": torch.ones(1, 3, dtype=torch.bool),
                  "loss_mask": torch.tensor([[False, True, True]])}
        with patch.object(FSDPPPOCritic, "compute_values",
                          return_value=[torch.tensor([[9., 8., 7., 6., 5.]])]) as native:
            result = critic.compute_values([sample])
        self.assertEqual(result[0].tolist(), [[7., 6., 5.]])
        self.assertEqual(native.call_args.args[0][0]["input_ids"].shape[-1], 5)
        with patch.object(FSDPPPOCritic, "ppo_update") as native_update:
            critic.ppo_update([sample])
        self.assertEqual(native_update.call_args.args[0][0]["loss_mask"].tolist(),
                         [[False, False, False, True, True]])


class ReplayTests(unittest.TestCase):
    def test_paired_interaction_preserves_invalid_penalty_and_zero_ablation(self):
        evaluation = {"valid": True, "reward": 0.25}
        self.assertEqual(interaction_reward(evaluation, -0.25, 0.5), (0.5, 0.5))
        self.assertEqual(interaction_reward(evaluation, -0.25, 0), (0.25, 0.5))
        self.assertEqual(interaction_reward({"valid": False, "reward": -1}, -1, 0.5), (-1, 0))
        with self.assertRaises(ValueError):
            interaction_reward(evaluation, -0.25, float("nan"))

    def call(self):
        return dict(prompt_ids=[1, 2], response_ids=[3, 4],
                    response_logprobs=[-0.5, -0.7], versions=[1, 1],
                    stop_reason="length", temperature=0.7)

    def test_original_probabilities_and_real_version_age_survive_refresh(self):
        row = {"id": "old-editor", "messages": [{"role": "user", "content": "old state"}],
               "replay": {"call": self.call(), "editor_step_offset": 4}}
        result = replay_interaction(row, None, 0.25)["old-editor"]
        self.assertEqual(result.reward, 0.25)
        self.assertEqual(result.model_response.output_logprobs, [-0.5, -0.7])
        self.assertEqual(result.model_response.output_versions, [-3, -3])
        self.assertEqual(result.model_response.input_tokens, [1, 2])
        tensor = result.to_tensor_dict()
        self.assertEqual(tensor["loss_mask"].tolist(), [[0, 0, 1, 1]])
        self.assertEqual(tensor["versions"].tolist(), [[-1, -1, -3, -3]])
        torch.testing.assert_close(
            tensor["logprobs"], torch.tensor([[0., 0., -0.5, -0.7]]))

    def test_reject_invented_or_misaligned_behavior(self):
        with self.assertRaises(ValueError):
            validate_call({**self.call(), "response_logprobs": [0.0]})
        with self.assertRaises(ValueError):
            validate_call({**self.call(), "response_logprobs": [float("nan"), -0.7]})

    def test_buffer_keeps_invalid_actions_and_does_not_filter_by_reward(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i, reward in enumerate([-1, 0.5]):
                write_json(root / f"{i}.json", dict(
                    role="editor", id=str(i), reward=reward, calls=[self.call()],
                    input_messages=[{"role": "user", "content": "state"}],
                    action_text="invalid" if reward < 0 else "candidate",
                    partner_version=0, harness_source=INITIAL_SOURCE, replayed=False,
                ))
            write_json(root / "already_replayed.json", {
                "role": "editor", "replayed": True, "reward": 0.25,
            })
            rows = replay_rows(root, 2, 4)
            original_ids = [row["id"] for row in rows]
            (root / "0.json").rename(root / "random-uuid-b.json")
            (root / "1.json").rename(root / "random-uuid-a.json")
            self.assertEqual([row["id"] for row in replay_rows(root, 2, 4)], original_ids)
            self.assertEqual(sorted(r["replay"]["old_reward"] for r in rows), [-1, 0.5])
            self.assertTrue(all(r["replay"]["reference_source"] == INITIAL_SOURCE for r in rows))


if __name__ == "__main__":
    unittest.main()
