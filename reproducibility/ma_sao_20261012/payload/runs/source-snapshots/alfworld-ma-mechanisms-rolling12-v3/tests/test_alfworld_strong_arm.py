from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.alfworld_strong_arm import main, run, wait_existing_phase, wait_stopped_unit
from sao_bridge.common import file_sha, read_json, write_json


class StrongArmTests(unittest.TestCase):
    def exercise(self, arm, *, external=False, all_pass=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = {
                "run": str(root),
                "phase_orders": {"sequential": ["h2", "w1", "w2"], "ma": ["w1", "h2", "w2"]},
            }
            protocol_path = root / "frozen_protocol.json"
            write_json(protocol_path, protocol)
            initial = {
                "solver": "w0", "packet_solver": "w0", "harness": "h1",
                "harness_sha256": "h1-sha", "protocol_sha256": file_sha(protocol_path),
            }
            write_json(root / "common/INITIAL_STATE.json", initial)
            events = []

            def training(protocol_path, chosen_arm, phase, output, *, state_path):
                events.append("train-" + phase)
                state = read_json(state_path)
                if chosen_arm == "ma" and phase == "h2":
                    self.assertEqual(state["packet_solver"], state["solver"])
                    self.assertEqual(state["solver"], "w1")
                write_json(output / "input_state.json", state)
                complete = {"native_audit": {"fixture": phase}}
                write_json(output / "PHASE_COMPLETE.json", complete)
                return complete

            def selection(protocol_path, chosen_arm, phase, training_root, output):
                events.append("select-" + phase)
                state = read_json(training_root / "input_state.json")
                if phase == "h2":
                    state.update(harness="h2", harness_sha256="h2-sha")
                else:
                    state["solver"] = phase
                write_json(output / "SELECTED_STATE.json", state)
                complete = {"selected_state": str(output / "SELECTED_STATE.json"),
                            "selected_state_sha256": file_sha(output / "SELECTED_STATE.json")}
                write_json(output / "COMPLETE.json", complete)
                return complete

            def refresh(protocol_path, state_path, output):
                events.append("refresh")
                state = read_json(state_path)
                self.assertNotEqual(state["packet_solver"], state["solver"])
                state["packet_solver"] = state["solver"]
                write_json(output / "REFRESHED_STATE.json", state)
                return {"engineer_training_ready": not all_pass,
                        "state": str(output / "REFRESHED_STATE.json")}

            if external:
                first = protocol["phase_orders"][arm][0]
                phase_root = root / arm / first
                write_json(phase_root / "input_state.json", initial)
                write_json(phase_root / "PHASE_COMPLETE.json", {"native_audit": {"fixture": first}})
                write_json(phase_root / "RELEASED.json", {
                    "foreign_gpu_pids": [], "remaining_gpu_pids": [], "verified_filler_pids": [],
                    "filler_unit_active": True,
                })
                write_json(root / arm / f"launch_receipt_{first}.json", {"unit": "coevolve-strong-fixture.service"})
            with (
                patch("sao_bridge.alfworld_strong_arm.require_research_node"),
                patch("sao_bridge.alfworld_strong_arm.checked_selected_state",
                      side_effect=lambda path, *args: read_json(path)),
                patch("sao_bridge.alfworld_strong_arm.run_phase", side_effect=training),
                patch("sao_bridge.alfworld_strong_arm.select_phase", side_effect=selection),
                patch("sao_bridge.alfworld_strong_arm.refresh_partner", side_effect=refresh),
                patch("sao_bridge.alfworld_strong_arm.model_digests", return_value=("model", "weights")),
            ):
                if all_pass:
                    with self.assertRaisesRegex(ValueError, "no training failures"):
                        run(protocol_path, arm)
                    self.assertTrue((root / arm / "ARM_FAILED.json").is_file())
                    self.assertFalse((root / arm / "ARM_FROZEN.json").exists())
                else:
                    result = run(protocol_path, arm)
                    self.assertFalse(result["report_opened"])
                    self.assertEqual(result["phase_order"], protocol["phase_orders"][arm])
                    self.assertEqual(len(result["phases"]), 3)
                    previous = list(events)
                    self.assertEqual(run(protocol_path, arm), result)
                    self.assertEqual(events, previous)
            return events

    def test_sequential_order_and_external_training_adoption(self):
        self.assertEqual(self.exercise("sequential", external=True), [
            "select-h2", "train-w1", "select-w1", "train-w2", "select-w2",
        ])

    def test_ma_refreshes_changed_partner_before_h2(self):
        self.assertEqual(self.exercise("ma"), [
            "train-w1", "select-w1", "refresh", "train-h2", "select-h2", "train-w2", "select-w2",
        ])

    def test_all_pass_refresh_does_not_fake_training_or_arm_completion(self):
        self.assertEqual(self.exercise("ma", all_pass=True), ["train-w1", "select-w1", "refresh"])

    def test_external_failed_unit_is_not_accepted_as_completed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt = root / "launch.json"
            write_json(receipt, {"unit": "coevolve-strong-fixture.service"})
            with (
                patch("sao_bridge.alfworld_strong_arm.subprocess.check_output", return_value="failed\n"),
                self.assertRaisesRegex(RuntimeError, "without complete evidence"),
            ):
                wait_existing_phase(root, receipt, timeout=1)

    def test_handoff_waits_through_restarts_and_shutdown(self):
        with (
            patch("sao_bridge.alfworld_strong_arm.subprocess.check_output",
                  side_effect=["active\n", "activating\n", "deactivating\n", "failed\n"]),
            patch("sao_bridge.alfworld_strong_arm.time.sleep") as sleep,
        ):
            self.assertEqual(wait_stopped_unit("coevolve-strong-previous.service"), "failed")
            self.assertEqual(sleep.call_count, 3)
        with self.assertRaisesRegex(ValueError, "previous"):
            wait_stopped_unit("unrelated.service")
        with (
            patch("sao_bridge.alfworld_strong_arm.subprocess.check_output", return_value="active\n"),
            patch("sao_bridge.alfworld_strong_arm.time.monotonic", side_effect=[0, 0, 2]),
            patch("sao_bridge.alfworld_strong_arm.time.sleep"),
            self.assertRaises(TimeoutError),
        ):
            wait_stopped_unit("coevolve-strong-previous.service", timeout=1)

    def test_cli_does_not_enter_arm_before_previous_unit_stops(self):
        events = []
        with (
            patch.object(sys, "argv", ["arm", "--protocol", "frozen.json", "--arm", "ma",
                                      "--after-unit", "coevolve-strong-previous.service"]),
            patch("sao_bridge.alfworld_strong_arm.wait_stopped_unit",
                  side_effect=lambda unit: events.append("wait")),
            patch("sao_bridge.alfworld_strong_arm.run",
                  side_effect=lambda protocol, arm: events.append("run")),
            patch("builtins.print"),
        ):
            main()
        self.assertEqual(events, ["wait", "run"])


if __name__ == "__main__":
    unittest.main()
