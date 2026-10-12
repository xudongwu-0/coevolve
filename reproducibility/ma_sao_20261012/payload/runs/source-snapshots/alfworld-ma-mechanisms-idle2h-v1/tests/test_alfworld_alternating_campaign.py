from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge import alfworld_alternating_campaign as campaign
from sao_bridge.common import file_sha, read_json, write_json


class AlternatingCampaignTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        pinned = self.root / "pinned.json"
        write_json(pinned, {"source": "immutable"})
        self.protocol = {
            "run": str(self.root), "phase_orders": {"sequential": ["h2", "w2"], "ma": ["h2", "w2"]},
            "initial_solver_group": "initial_w1", "initial_solver": "W79",
            "evaluation": {"report_mode": "final_systems"}, "max_phase_attempts": 2,
            "autonomous": {
                "unit_prefix": "coevolve-alt8h-test", "source_worktree": str(campaign.ROOT),
                "started_at_unix": 1000, "handoff_deadline_unix": 1600,
                "common_deadline_unix": 4600, "training_deadline_unix": 26200,
                "deadline_unix": 29800, "pinned_files": {str(pinned): file_sha(pinned)},
                "prior_units": {"sequential": ["coevolve-strong-test-report.service"], "ma": []},
            },
        }
        self.path = self.root / "frozen_protocol.json"
        write_json(self.path, self.protocol)
        (self.root / "sequential").mkdir()

    def test_deadlines_order_and_inputs_are_frozen(self):
        self.assertEqual(campaign.validate_protocol(self.path), self.protocol)
        self.protocol["autonomous"]["deadline_unix"] += 60
        write_json(self.path, self.protocol)
        with self.assertRaisesRegex(ValueError, "eight-hour deadline"):
            campaign.validate_protocol(self.path)
        self.protocol["autonomous"]["deadline_unix"] -= 60
        self.protocol["phase_orders"]["ma"] = ["w2", "h2"]
        write_json(self.path, self.protocol)
        with self.assertRaisesRegex(ValueError, "matched order"):
            campaign.validate_protocol(self.path)

    def test_expired_stage_never_allocates_gpus(self):
        with (
            patch.object(campaign, "unit_status", return_value={"LoadState": "not-found"}),
            patch.object(campaign.time, "time", return_value=30),
            patch.object(campaign, "launch_stage") as launch,
        ):
            result = campaign.execute_stage(self.protocol, "sequential", "arm", [], 20, [self.root / "missing"])
        self.assertEqual(result["status"], "deadline")
        launch.assert_not_called()

    def test_restart_rejoins_active_unit_instead_of_duplicating_it(self):
        complete = self.root / "done.json"
        write_json(complete, {})
        with (
            patch.object(campaign, "unit_status", side_effect=[
                {"LoadState": "loaded", "ActiveState": "active"},
                {"LoadState": "loaded", "ActiveState": "inactive", "Result": "success", "ExecMainStatus": "0"},
            ]),
            patch.object(campaign.time, "time", return_value=10),
            patch.object(campaign.time, "sleep"),
            patch.object(campaign, "launch_stage") as launch,
        ):
            result = campaign.execute_stage(self.protocol, "sequential", "arm", [], 20, [complete])
        self.assertEqual(result["status"], "complete")
        launch.assert_not_called()

    def test_deadline_stops_only_its_explicit_stage(self):
        active = {"LoadState": "loaded", "ActiveState": "active"}
        with (
            patch.object(campaign, "unit_status", return_value=active),
            patch.object(campaign.time, "time", return_value=30),
            patch.object(campaign, "stop_unit") as stop,
        ):
            result = campaign.execute_stage(self.protocol, "sequential", "arm", [], 20, [self.root / "missing"])
        self.assertEqual(result["status"], "deadline")
        stop.assert_called_once_with(self.protocol, "sequential", "coevolve-alt8h-test-sequential-arm.service")
        with patch.object(campaign.subprocess, "run") as command:
            with self.assertRaisesRegex(ValueError, "explicit ownership"):
                campaign.stop_unit(self.protocol, "sequential", "bruli-h200-load.service")
            with self.assertRaisesRegex(ValueError, "explicit ownership"):
                campaign.stop_unit(self.protocol, "sequential", "bruli-h200-load.service", prior=True)
            command.assert_not_called()

    def test_systemd_expiry_is_absolute_and_independent_of_coordinator(self):
        with (
            patch.object(campaign, "unit_status", return_value={"LoadState": "not-found"}),
            patch.object(campaign.subprocess, "run") as command,
        ):
            campaign.install_expiry(self.protocol, "sequential", "arm", 26200)
        args = command.call_args.args[0]
        self.assertIn("--on-calendar=1970-01-01 07:16:40 UTC", args)
        self.assertEqual(args[-3:], ["/usr/bin/systemctl", "stop", "coevolve-alt8h-test-sequential-arm.service"])
        self.assertTrue(read_json(self.root / "sequential/arm_expiry.json")["filler_is_not_a_target"])

    def test_stage_launch_keeps_remaining_budget_and_whole_child_cgroup(self):
        with (
            patch.object(campaign.time, "time", return_value=10),
            patch.object(campaign.subprocess, "run") as command,
        ):
            campaign.launch_stage(self.protocol, "sequential", "arm", ["-m", "module"], 20)
        args = command.call_args.args[0]
        self.assertIn("--property=RuntimeMaxSec=10", args)
        self.assertIn("--property=KillMode=control-group", args)
        self.assertIn("--property=RestartPreventExitStatus=78", args)
        self.assertIn("--setenv=TMPDIR=/dev/shm", args)
        self.assertIn("--property=StandardOutput=append:" + str(self.root / "sequential/arm.service.log"), args)
        self.assertEqual(read_json(self.root / "sequential/arm_launches.json")[0]["deadline_unix"], 20)

    def test_budget_freeze_never_promotes_a_partial_checkpoint(self):
        state = {
            "solver": "W79", "solver_critic": "critic79", "engineer": "E15",
            "engineer_critic": "critic15", "harness": "accepted-h1", "harness_sha256": "h1",
            "protocol_sha256": file_sha(self.path), "solver_weights_sha256": "w79-weights",
        }
        write_json(self.root / "sequential/w2/partial-checkpoint.json", {"solver": "unvalidated-w2"})
        release = {"foreign_gpu_pids": [], "filler_unit_active": True}
        with (
            patch.object(campaign, "checked_selected_state", return_value=state),
            patch.object(campaign, "gpu_release_audit", return_value=release),
            patch.object(campaign, "model_digests", return_value=("metadata", "w79-weights")),
        ):
            result = campaign.freeze_available(self.path, "sequential", {"status": "deadline"})
            self.assertEqual(campaign.freeze_available(self.path, "sequential", {"status": "retry"}), result)
        self.assertFalse(result["all_phases_completed"])
        self.assertTrue(result["partial_training_checkpoints_not_promoted"])
        self.assertEqual(read_json(self.root / "sequential/FINAL_STATE.json"), state)
        self.assertEqual(result["phases"], [])

    def test_bootstrap_carries_w79_and_its_own_critic_not_sft(self):
        engineer_training, solver_training = self.root / "engineer-training", self.root / "solver-training"
        for root, step, actor, critic in (
            (engineer_training, 15, "E15", "critic15"),
            (solver_training, 79, "W79", "critic79"),
        ):
            write_json(root / "optimizer/trained.json", {"checkpoint_history": {
                "actor": [{"global_step": step, "path": actor}],
                "critic": [{"global_step": step, "path": critic}],
            }})
        empty = self.root / "empty.json"
        write_json(empty, {"benchmark": "alfworld", "actions": []})
        self.protocol.update(
            no_patch=str(empty), common_prefix={
                "engineer_training": str(engineer_training), "selected_checkpoint_step": 15,
            },
        )
        self.protocol["autonomous"].update(
            solver_training=str(solver_training), solver_checkpoint_step=79,
            solver_runtime_reference="sft-runtime-only",
        )
        write_json(self.path, self.protocol)
        with (
            patch.object(campaign, "require_research_node"),
            patch.object(campaign, "ensure_model_runtime_metadata"),
            patch.object(campaign, "model_digests", return_value=("metadata", "w79-weights")),
            patch.object(campaign, "Launcher", return_value=Mock()),
            patch.object(campaign, "gpu_release_audit", return_value={}),
            patch.object(campaign, "refresh_partner", return_value={"engineer_training_ready": False}),
            self.assertRaisesRegex(ValueError, "no authentic training failures"),
        ):
            campaign.bootstrap(self.path)
        state = read_json(self.root / "common/WARM_START_STATE.json")
        self.assertEqual((state["solver"], state["solver_critic"]), ("W79", "critic79"))
        self.assertEqual((state["engineer"], state["engineer_critic"]), ("E15", "critic15"))
        self.assertEqual(state["solver_epoch_version"], 1)


if __name__ == "__main__":
    unittest.main()
