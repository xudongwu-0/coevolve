import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.central_critic import JointStateCritic, context_text, prepend_context
from sao_bridge.adaptive_gpu_filler import (
    ControllerConfig,
    ExternalLoad,
    decide_policy,
    external_loads,
    parse_pmon,
    read_reservations,
)
from sao_bridge.joint_context import joint_context_audit
from sao_bridge.resources import (
    ADAPTIVE_MODULE,
    SCRIPT,
    UNIT,
    filler_kind,
    gpu_release_audit,
    is_filler,
    stop_known,
)
from sao_bridge.replay import interaction_reward, replay_interaction, replay_rows, validate_call
from sao_bridge.common import write_json
from sao_bridge.retool import INITIAL_SOURCE


class ResourceTests(unittest.TestCase):
    def test_only_exact_filler_identity(self):
        record = {"cmd": ["/usr/bin/python3", "-B", SCRIPT, "worker", "--gpu", "0"],
                  "cgroup": "0::/system.slice/" + UNIT}
        self.assertTrue(is_filler(record))
        adaptive = {
            "cmd": [
                "/usr/bin/python3", "-B", "-m", ADAPTIVE_MODULE, "worker",
                "--gpu", "7", "--matrix-size", "2048", "--control-fd", "4",
            ],
            "cgroup": "0::/system.slice/" + UNIT,
        }
        self.assertEqual(filler_kind(adaptive), "adaptive")
        self.assertEqual(filler_kind({
            "cmd": ["/usr/bin/python3", "-B", "-m", ADAPTIVE_MODULE, "supervisor"],
            "cgroup": "0::/system.slice/" + UNIT,
        }, worker=False), "adaptive")
        for change in ({"cgroup": "0::/other.service"},
                       {"cmd": ["python", "real_training.py"]}):
            self.assertFalse(is_filler({**record, **change}))

    def test_adaptive_filler_remains_active_for_training(self):
        supervisor = {
            "pid": 100,
            "uid": 200,
            "cmd": ["/usr/bin/python3", "-B", "-m", ADAPTIVE_MODULE, "supervisor"],
            "cgroup": "0::/system.slice/" + UNIT,
        }
        state = {
            "unit": {"ActiveState": "active"},
            "processes": [],
            "supervisor": supervisor,
            "verified_unit": True,
            "filler_kind": "adaptive",
        }
        with patch("sao_bridge.resources.require_research_node"), patch(
            "sao_bridge.resources.audit", return_value=state,
        ), patch("sao_bridge.resources.subprocess.run") as systemctl:
            result = stop_known(require_exclusive=True)
        systemctl.assert_not_called()
        self.assertFalse(result["stopped"])
        self.assertTrue(result["cooperative"])

    def test_adaptive_filler_does_not_hide_existing_foreign_workload(self):
        state = {
            "unit": {"ActiveState": "active"},
            "processes": [{"pid": 999, "verified_filler": False}],
            "supervisor": {},
            "verified_unit": True,
            "filler_kind": "adaptive",
        }
        with patch("sao_bridge.resources.require_research_node"), patch(
            "sao_bridge.resources.audit", return_value=state,
        ), self.assertRaisesRegex(RuntimeError, "non-filler GPU processes"):
            stop_known(require_exclusive=True)

    def test_release_audit_separates_filler_from_foreign_processes(self):
        state = {
            "unit": {"ActiveState": "active"},
            "processes": [
                {"pid": 10, "verified_filler": True},
                {"pid": 20, "verified_filler": False},
            ],
            "filler_kind": "adaptive",
            "vanished_gpu_pids": [30],
        }
        with patch("sao_bridge.resources.audit", return_value=state):
            result = gpu_release_audit()
        self.assertEqual(result["remaining_gpu_pids"], [10, 20])
        self.assertEqual(result["verified_filler_pids"], [10])
        self.assertEqual(result["foreign_gpu_pids"], [20])
        self.assertEqual(result["vanished_gpu_pids"], [30])


class AdaptiveFillerTests(unittest.TestCase):
    sample = {
        "utilization_pct": 0,
        "memory_used_mib": 0,
        "memory_total_mib": 143771,
        "temperature_c": 45,
    }

    def test_idle_gpu_ramps_gradually(self):
        decision = decide_policy(self.sample, ExternalLoad(), 0.0, 0)
        self.assertEqual(decision.mode, "idle")
        self.assertEqual(decision.duty, ControllerConfig().scale_up_step)

    def test_busy_training_yields_immediately(self):
        external = ExternalLoad((123,), 120_000, 92)
        decision = decide_policy(
            {**self.sample, "utilization_pct": 95, "memory_used_mib": 120_100},
            external,
            0.9,
            60,
        )
        self.assertEqual(decision.mode, "yielded")
        self.assertEqual(decision.duty, 0)
        self.assertEqual(decision.reason, "external_busy")

    def test_training_memory_pressure_releases_worker(self):
        external = ExternalLoad((123,), 140_000, 0)
        decision = decide_policy(
            {**self.sample, "memory_used_mib": 141_000},
            external,
            0.2,
            60,
        )
        self.assertEqual(decision.mode, "yielded")
        self.assertEqual(decision.reason, "memory_yield")

    def test_external_idle_uses_hysteresis_and_tiered_caps(self):
        external = ExternalLoad((123,), 120_000, 0)
        short = decide_policy(self.sample, external, 0.0, 2)
        medium = decide_policy(self.sample, external, 0.0, 35)
        long = decide_policy(self.sample, external, 0.0, 125)
        downsized = decide_policy(self.sample, external, 0.9, 35)
        partial = decide_policy(
            self.sample,
            ExternalLoad((123,), 120_000, 10),
            0.9,
            0,
        )
        self.assertEqual(short.mode, "yielded")
        self.assertEqual(short.duty, 0)
        self.assertEqual(partial.mode, "yielded")
        self.assertLessEqual(medium.duty, ControllerConfig().shared_medium_cap)
        self.assertEqual(medium.duty, ControllerConfig().scale_up_step)
        self.assertEqual(long.duty, ControllerConfig().scale_up_step)
        self.assertGreater(long.cap, medium.cap)
        self.assertEqual(downsized.duty, ControllerConfig().shared_medium_cap)

    def test_pmon_parser_keeps_process_sm_and_memory(self):
        text = """# gpu pid type sm mem enc dec jpg ofa fb ccpm command
    0 123 C 99 81 - - - - 122814 0 VLLM::Worker
    1 456 C 0 0 - - - - 512 0 python
"""
        self.assertEqual(parse_pmon(text), [
            {
                "gpu": 0, "pid": 123, "sm_pct": 99, "memory_pct": 81,
                "framebuffer_mib": 122814, "command": "VLLM::Worker",
            },
            {
                "gpu": 1, "pid": 456, "sm_pct": 0, "memory_pct": 0,
                "framebuffer_mib": 512, "command": "python",
            },
        ])

    def test_external_loads_exclude_owned_worker(self):
        samples = [
            {
                "index": 0,
                "uuid": "GPU-a",
                "utilization_pct": 90,
                "memory_used_mib": 1024,
                "memory_total_mib": 143771,
                "temperature_c": 45,
                "power_w": 300,
            },
        ]
        apps = [
            {"uuid": "GPU-a", "pid": 10, "memory_mib": 48},
            {"uuid": "GPU-a", "pid": 20, "memory_mib": 512},
        ]
        pmon = [
            {"gpu": 0, "pid": 10, "sm_pct": 70, "framebuffer_mib": 48},
            {"gpu": 0, "pid": 20, "sm_pct": 25, "framebuffer_mib": 512},
        ]
        load = external_loads(samples, apps, pmon, {10})[0]
        self.assertEqual(load.pids, (20,))
        self.assertEqual(load.memory_mib, 512)
        self.assertEqual(load.sm_pct, 25)

    def test_invalid_reservation_yields_all_gpus(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "reservation.json"
            write_json(path, {"reserved_gpus": [8]})
            reserved, error = read_reservations(path, 8)
        self.assertEqual(reserved, set(range(8)))
        self.assertIn("invalid GPU indices", error)


class AdaptiveFillerDeploymentTests(unittest.TestCase):
    repo = Path(__file__).resolve().parents[1]
    manager = repo / "scripts/systemd/manage-adaptive-gpu-filler.sh"

    def test_complete_unit_has_boot_and_crash_recovery(self):
        template = (
            self.repo / "scripts/systemd/bruli-h200-load.service.in"
        ).read_text()
        self.assertIn("WantedBy=multi-user.target", template)
        self.assertIn("Restart=always", template)
        self.assertIn("RestartSec=5", template)
        self.assertIn("KillMode=control-group", template)
        self.assertIn("ConditionPathExists=!@PAUSE_FILE@", template)
        self.assertIn("COEVOLVE_GPU_FILLER_RESERVATION=@RESERVATION_FILE@", template)

    def test_manager_renders_self_contained_unit(self):
        result = subprocess.run(
            [str(self.manager), "render"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertNotIn("@REPO_ROOT@", result.stdout)
        self.assertNotIn("@BASE_DIR@", result.stdout)
        self.assertIn(
            "ExecStart=/usr/bin/python3 -B -m "
            "sao_bridge.adaptive_gpu_filler supervisor",
            result.stdout,
        )
        self.assertIn(f"RequiresMountsFor={self.repo}", result.stdout)


class CentralCriticTests(unittest.TestCase):
    def test_native_value_outputs_return_to_actor_alignment(self):
        from areal.engine.fsdp_engine import FSDPPPOCritic
        critic = object.__new__(JointStateCritic)
        critic.context_ids = [10, 11]
        row = {"input_ids": torch.tensor([4, 5, 6]),
               "attention_mask": torch.ones(3, dtype=torch.bool),
               "loss_mask": torch.tensor([0, 1, 1])}
        with patch.object(FSDPPPOCritic, "compute_values",
                          return_value=[torch.tensor([10., 11., 1., 2., 3.])]) as forward:
            values = critic.compute_values([row])
            self.assertEqual(values[0].tolist(), [1., 2., 3.])
            self.assertEqual(forward.call_args.args[0][0]["loss_mask"].tolist(),
                             [0, 0, 0, 1, 1])
        with patch.object(FSDPPPOCritic, "ppo_update") as update:
            critic.ppo_update([row])
            self.assertEqual(update.call_args.args[0][0]["loss_mask"].tolist(),
                             [0, 0, 0, 1, 1])

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

    def test_context_audit_rejects_full_harness_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            context = Path(temporary) / "context.json"
            write_json(context, {
                "phase": "w", "actor_role": "solver",
                "partner_checkpoint": "frozen",
                "harness_source": "full source", "calibration_accuracy": 0.5,
                "calibration_split": "dev", "partner_version": 1,
            })
            tokenizer = Mock()
            tokenizer.encode.return_value = list(range(1025))
            with patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=tokenizer,
            ), self.assertRaisesRegex(ValueError, "stable harness identity"):
                joint_context_audit(context, Path(temporary))

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
