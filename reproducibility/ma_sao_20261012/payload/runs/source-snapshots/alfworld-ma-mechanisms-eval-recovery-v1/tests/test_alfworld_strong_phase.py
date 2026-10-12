import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.alfworld_strong_phase import (
    checked_native_report, checkpoint_pair, phase_configuration, train_calibration,
    training_evidence, execute_phase, PhaseAttemptsExhausted,
)
from sao_bridge.alfworld_train_data import solver_rows
from sao_bridge.budget import contract_digest
from sao_bridge.common import file_sha, read_json, write_json
from sao_bridge.evaluation import stable_id_seed_offset


class StrongPhaseTests(unittest.TestCase):
    @staticmethod
    def packet():
        return {
            "id": "packet", "messages": [{"role": "system", "content": "rules"},
                                        {"role": "user", "content": "training evidence"}],
            "replay": None,
            "metadata": {
                "batch_tag": "train", "split": "test_set", "start": 0, "end": 2,
                "batch_size": 2, "baseline_pass": 1, "baseline_rewards": {"0": 0, "1": 1},
                "reward_threshold": 1, "target_agent_name": "fixture", "target_model": "policy",
                "response_protocol": "prefill_think_patch", "solver_version": "w0",
            },
        }

    def test_report_audits_exact_seed_coverage_and_budget(self):
        evaluation = {"samples": 2, "seed": 42, "solver_tokens": 8}
        rows = [{
            "id": f"alfworld-{task}", "task_id": task, "sample_index": sample,
            "rollout_seed": 42 + stable_id_seed_offset(f"alfworld-{task}") + 1009 * sample,
            "reward": float(sample), "tokens": 8,
        } for task in (240, 241) for sample in range(2)]
        report = {
            "metric": "mean@2", "samples_per_question": 2, "question_count": 2,
            "accuracy": 0.5, "results": rows,
        }
        self.assertIs(checked_native_report(report, [240, 241], evaluation), report)
        for change in ({"tokens": 9}, {"reward": True}, {"rollout_seed": 0},
                       {"id": "different"}, {"sample_index": 1}):
            changed = copy.deepcopy(report)
            changed["results"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                checked_native_report(changed, [240, 241], evaluation)
        with self.assertRaisesRegex(ValueError, "accuracy"):
            checked_native_report({**report, "accuracy": 0.6}, [240, 241], evaluation)

    def test_matching_critic_and_train_only_calibration(self):
        trained = {"checkpoint_history": {
            role: [{"global_step": 15, "path": role + "15"},
                   {"global_step": 31, "path": role + "31"}]
            for role in ("actor", "critic")
        }}
        self.assertEqual(checkpoint_pair(trained, 15), {"actor": "actor15", "critic": "critic15"})
        trained["checkpoint_history"]["critic"] = []
        with self.assertRaisesRegex(ValueError, "critic"):
            checkpoint_pair(trained, 15)
        packet = self.packet()
        calibration = train_calibration([packet], "w0")
        self.assertEqual(calibration["accuracy"], 0.5)
        self.assertEqual(calibration["split"], "train")
        with self.assertRaisesRegex(ValueError, "repeats"):
            train_calibration([packet, packet], "w0")
        with self.assertRaisesRegex(ValueError, "different Solver"):
            train_calibration([packet], "w1")
        packet["metadata"].update(
            start=300, end=302, baseline_rewards={"300": 0, "301": 1},
        )
        with self.assertRaisesRegex(ValueError, "training tasks"):
            train_calibration([packet], "w0")

    def fixture(self, root):
        empty = root / "empty.json"
        write_json(empty, {"benchmark": "alfworld", "actions": []})
        packets_path = root / "packets.json"
        packets = [self.packet()]
        write_json(packets_path, packets)
        manifest_path = root / "manifest.json"
        write_json(manifest_path, {"selection_task_ids": [240, 241], "report_task_ids": [300, 301]})
        catalogue = root / "catalogue.json"
        games = tuple(f"json_2.1.1/train/train-goal-{index}/instance/game.tw-pddl" for index in range(16))
        write_json(catalogue, {"train": games})
        raw = root / "raw.json"
        write_json(raw, solver_rows(games, file_sha(catalogue), catalogue_path=catalogue))
        h_template = {
            "engineer_model": "e0", "engineer_critic": "ec0", "solver_model": "w0",
            "solver_version": "w0", "harness": str(empty), "data": str(raw), "tmpdir": str(root),
            "steps": 4, "batch_size": 2, "seed": 42, "phase": "h_fast", "experiment": "fixture",
            "critic_warmup_steps": 1, "terminal_episodes": True, "critic_mode": "learned",
            "critic_updates_per_step": 1, "learning_rate": 2e-6, "critic_learning_rate": 1e-5,
            "solver_tokens": 4096, "temperature": 1.0, "staleness": 1,
            "actor_gpus": 4, "rollout_gpus": 4, "save_interval_steps": 2,
            "solver_trajectory_style": "concat", "actor_gradient_checkpointing": True,
            "critic_gradient_checkpointing": True,
        }
        h_path = root / "h.json"
        write_json(h_path, h_template)
        w_path = root / "w.json"
        write_json(w_path, {
            **h_template, "training_data": str(raw), "training_catalogue": str(catalogue),
            "training_catalogue_sha256": file_sha(catalogue),
            "training_environment_concurrency": 1, "training_timeout_seconds": 100,
        })
        protocol = {
            "protocol": "fixture", "run": str(root), "no_patch": str(empty),
            "phase_orders": {"sequential": ["h2", "w1", "w2"], "ma": ["w1", "h2", "w2"]},
            "evaluation": {"task_manifest": str(manifest_path)},
            "h2": {"template": str(h_path), "steps": 4, "batch_size": 2,
                   "critic_warmup_steps": 1, "seed": 1000042, "sample_offset": 192},
            "w": {"template": str(w_path), "steps_per_phase": 4, "batch_size": 2,
                  "sample_offsets": {"w1": 0, "w2": 8}, "seeds": {"w1": 42, "w2": 1000042}},
        }
        write_json(root / "frozen_protocol.json", protocol)
        state = {
            "engineer": "e15", "engineer_critic": "ec15", "solver": "w0", "solver_critic": "w0",
            "solver_version": "w0", "solver_epoch_version": 0, "engineer_version": 1,
            "harness_version": 0, "harness": str(empty),
            "harness_sha256": contract_digest(read_json(empty)),
            "packets": str(packets_path), "packets_sha256": file_sha(packets_path),
            "packet_solver": "w0", "calibration": train_calibration(packets, "w0"),
            "protocol_sha256": file_sha(root / "frozen_protocol.json"),
        }
        paths = [f"json_2.1.1/train/legacy-{index}/instance/game.tw-pddl" for index in range(500)]
        return protocol, state, paths, games, raw

    def test_role_budgets_rows_and_actor_seeds_match_across_arms(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol, state, paths, games, raw = self.fixture(root)
            with (
                patch("sao_bridge.alfworld_strong_phase.evaluation_paths", return_value=paths),
                patch("sao_bridge.alfworld_strong_phase.training_catalogue", return_value=games),
                patch("sao_bridge.alfworld_strong_phase.training_capacity_audit", return_value={"checked": True}),
                patch("sao_bridge.alfworld_strong_phase.joint_context_audit", return_value={"tokens": 128}),
            ):
                h_seq = phase_configuration(protocol, "sequential", "h2", state, root / "h-seq")
                h_ma = phase_configuration(protocol, "ma", "h2", state, root / "h-ma")
                for key in ("steps", "batch_size", "seed", "critic_warmup_steps",
                            "engineer_model", "engineer_critic", "learning_rate", "critic_learning_rate"):
                    self.assertEqual(h_seq[key], h_ma[key])
                self.assertEqual(read_json(Path(h_seq["data"])), read_json(Path(h_ma["data"])))
                self.assertTrue(h_seq["terminal_episodes"])
                self.assertNotIn("central_context", h_seq)
                self.assertEqual(read_json(Path(h_ma["central_context"]))["calibration_split"], "train")
                rows = read_json(Path(h_seq["data"]))
                self.assertEqual(len(rows), 8)
                self.assertEqual(len({row["id"] for row in rows}), 8)
                self.assertTrue(all(not row["replay"] for row in rows))

                w_seq = phase_configuration(protocol, "sequential", "w1", state, root / "w-seq")
                w_ma = phase_configuration(protocol, "ma", "w1", state, root / "w-ma")
                self.assertEqual(read_json(Path(w_seq["data"])), read_json(Path(w_ma["data"])))
                w2 = phase_configuration(protocol, "sequential", "w2", state, root / "w2")
                self.assertEqual([row["task_id"] for row in read_json(Path(w_seq["data"]))], list(range(8)))
                self.assertEqual([row["task_id"] for row in read_json(Path(w2["data"]))], list(range(8, 16)))
                self.assertEqual(w_seq["solver_trajectory_style"], "concat")

                changed = copy.deepcopy(state)
                changed["calibration"]["split"] = "report"
                with self.assertRaisesRegex(ValueError, "calibration"):
                    phase_configuration(protocol, "ma", "h2", changed, root / "bad-calibration")
                changed = {**state, "packet_solver": "different"}
                with self.assertRaisesRegex(ValueError, "counterpart"):
                    phase_configuration(protocol, "ma", "h2", changed, root / "bad-partner")
                damaged = read_json(raw)
                damaged[0]["task_id"] = 9
                write_json(raw, damaged)
                with self.assertRaisesRegex(ValueError, "checksum-bound"):
                    phase_configuration(protocol, "sequential", "w1", state, root / "bad-data")

    def test_native_audit_rejects_warmup_updates_and_wrong_role_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {"run": str(root), "steps": 3, "critic_warmup_steps": 1,
                      "critic_updates_per_step": 1, "batch_size": 2, "phase": "h2"}
            write_json(root / "config.json", config)
            write_json(root / "COMPLETE.json", {"status": "completed"})
            write_json(root / "RELEASED.json", {
                "foreign_gpu_pids": [], "remaining_gpu_pids": [1], "verified_filler_pids": [1],
                "filler_unit_active": True,
            })
            trained = {
                "role": "editor", "actor_steps": 2, "critic_steps": 3,
                "critic_only_warmup_steps": 1, "terminal_episodes": True, "critic_mode": "learned",
            }
            write_json(root / "optimizer/trained.json", trained)
            rows = [{"global_step": index, "metrics": {
                "ppo_critic/update/target_value/min": -1.0,
                "ppo_critic/update/target_value/max": 0.5,
                **({"ppo_actor/update/grad_norm": 0.2, "ppo_actor/n_seqs": 2} if index else {}),
            }} for index in range(3)]
            path = root / "optimizer/metrics.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            _, audit = training_evidence(root, config)
            self.assertEqual(audit["nonzero_actor_gradients"], 2)
            rows[0]["metrics"]["ppo_actor/update/grad_norm"] = 0
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "warmup"):
                training_evidence(root, config)
            rows[0]["metrics"].pop("ppo_actor/update/grad_norm")
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            write_json(root / "optimizer/trained.json", {**trained, "role": "solver"})
            with self.assertRaisesRegex(ValueError, "reward range"):
                training_evidence(root, config)
            for row in rows:
                row["metrics"]["ppo_critic/update/target_value/min"] = -3.725290298461914e-9
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            _, audit = training_evidence(root, config)
            self.assertEqual(audit["critic_target_range"], [-3.725290298461914e-9, 0.5])
            self.assertEqual(audit["critic_target_roundoff_atol"], 1e-6)
            for low, high in ((-0.00001, 0.5), (0, 1.00001), (0.2, 0.1),
                              (float("nan"), 1), (0, float("inf"))):
                with self.subTest(low=low, high=high):
                    rows[0]["metrics"].update({
                        "ppo_critic/update/target_value/min": low,
                        "ppo_critic/update/target_value/max": high,
                    })
                    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                    with self.assertRaisesRegex(ValueError, "reward range"):
                        training_evidence(root, config)

    def test_phase_retries_are_persistent_bounded_and_do_not_repeat_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {
                "run": str(root / "attempt-01"), "steps": 3, "critic_warmup_steps": 1,
                "critic_updates_per_step": 1, "batch_size": 2, "phase": "h2",
                "phase_timeout_seconds": 100,
            }
            release = {"foreign_gpu_pids": [], "remaining_gpu_pids": [],
                       "verified_filler_pids": [], "filler_unit_active": True}
            for name, value in (("phase_config.json", config), ("data.json", []),
                                ("input_state.json", {})):
                write_json(root / name, value)
            write_json(root / "phase_identity.json", {
                "module": "sao_bridge.alfworld_h_train",
                "config_sha256": file_sha(root / "phase_config.json"),
                "data_sha256": file_sha(root / "data.json"),
                "state_sha256": file_sha(root / "input_state.json"),
            })
            launches = []

            def child(module, args, **kwargs):
                current = read_json(Path(args[1]))
                output = Path(current["run"])
                launches.append(output)
                output.mkdir()
                write_json(output / "config.json", current)
                if output.name == "attempt-02":
                    write_json(output / "COMPLETE.json", {"status": "completed"})
                    write_json(output / "RELEASED.json", release)
                    write_json(output / "optimizer/trained.json", {
                        "role": "editor", "actor_steps": 2, "critic_steps": 3,
                        "critic_only_warmup_steps": 1, "terminal_episodes": True, "critic_mode": "learned",
                    })
                    rows = [{"global_step": index, "metrics": {
                        "ppo_critic/update/target_value/min": -1,
                        "ppo_critic/update/target_value/max": 0,
                        **({"ppo_actor/update/grad_norm": 0.1, "ppo_actor/n_seqs": 2} if index else {}),
                    }} for index in range(3)]
                    (output / "optimizer/metrics.jsonl").write_text(
                        "".join(json.dumps(row) + "\n" for row in rows)
                    )
                return Mock()

            first = Mock(module=Mock(side_effect=child),
                         wait=Mock(side_effect=RuntimeError("native failure")))
            second = Mock(module=Mock(side_effect=child))
            cached = Mock()
            with (
                patch("sao_bridge.alfworld_strong_phase.Launcher", side_effect=[first, second, cached]),
                patch("sao_bridge.alfworld_strong_phase.gpu_release_audit", return_value=release),
            ):
                with self.assertRaisesRegex(RuntimeError, "native failure"):
                    execute_phase(root)
                self.assertEqual(len(read_json(root / "attempts.json")), 1)
                result = execute_phase(root)
                self.assertEqual(len(read_json(root / "attempts.json")), 2)
                self.assertEqual(result["native_audit"]["nonzero_actor_gradients"], 2)
                legacy = copy.deepcopy(result)
                legacy["native_audit"].pop("critic_target_range")
                legacy["native_audit"].pop("critic_target_roundoff_atol")
                write_json(root / "PHASE_COMPLETE.json", legacy)
                previous_sha = file_sha(root / "PHASE_COMPLETE.json")
                self.assertEqual(execute_phase(root), legacy)
                self.assertEqual(file_sha(root / "PHASE_COMPLETE.json"), previous_sha)
                cached.module.assert_not_called()
            self.assertEqual([path.name for path in launches], ["attempt-01", "attempt-02"])
            self.assertTrue((root / "attempt-01").exists())
            with (
                patch("sao_bridge.alfworld_strong_phase.Launcher") as factory,
                patch("sao_bridge.alfworld_strong_phase.gpu_release_audit", return_value=release),
                self.assertRaises(PhaseAttemptsExhausted),
            ):
                execute_phase(root, maximum_attempts=1)
            factory.return_value.module.assert_not_called()
            changed = copy.deepcopy(legacy)
            changed["native_audit"]["nonzero_actor_gradients"] = 1
            write_json(root / "PHASE_COMPLETE.json", changed)
            with (
                patch("sao_bridge.alfworld_strong_phase.Launcher") as factory,
                patch("sao_bridge.alfworld_strong_phase.gpu_release_audit", return_value=release),
                self.assertRaisesRegex(ValueError, "frozen"),
            ):
                execute_phase(root)
            factory.return_value.module.assert_not_called()
            write_json(root / "data.json", ["changed"])
            with (
                patch("sao_bridge.alfworld_strong_phase.Launcher") as factory,
                self.assertRaisesRegex(ValueError, "input changed"),
            ):
                execute_phase(root)
            factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
