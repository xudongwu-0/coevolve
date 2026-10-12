"""Routing safety tests; these are not benchmark performance evidence."""
from pathlib import Path
import tempfile
import unittest
from failure_gate import decide, route_and_select, probe_indices


def trajectory(index=0, key="a", reward=0, harness="old", token=5, stop="wrong_move"):
    return {"sample_index": index, "task_key": key, "reward": reward,
            "rollout_seed": 42, "sampling_identity": {"weight_digest": "frozen"},
            "harness_sha256": harness, "stop_condition": "solved" if reward else stop,
            "policy_calls": [{"prompt_ids": [1], "response_ids": [token]}],
            "prompt": [], "completion": [],
            "harness_trace": [{"actor": "verifier", "info": {"legal_by_board": True,
                                                                       "legal_by_harness": True}}]}


class GateSafety(unittest.TestCase):
    def test_intervention_and_abstention(self):
        original = trajectory()
        self.assertEqual(decide(original)["route"], "uncertain")
        self.assertEqual(decide(original, trajectory(), trajectory(reward=1, harness="new"))["route"], "harness")
        self.assertEqual(decide(original, trajectory(), trajectory(harness="new"))["route"], "weight")
        self.assertEqual(decide(original, trajectory(token=6), trajectory(reward=1, harness="new"))["route"], "uncertain")
        self.assertEqual(decide(original, trajectory(), trajectory())["route"], "uncertain")
        self.assertEqual(decide(original, trajectory(), trajectory(harness="new", stop="malformed"))["route"], "uncertain")

    def test_stale_weights_fail_closed(self):
        stale = trajectory(harness="new")
        stale["sampling_identity"]["weight_digest"] = "changed"
        with self.assertRaises(AssertionError):
            decide(trajectory(), trajectory(), stale)

    def test_changed_control_prompt_abstains(self):
        control = trajectory()
        control["policy_calls"][0]["prompt_ids"] = [2]
        self.assertEqual(decide(trajectory(), control, trajectory(harness="new", reward=1))["route"], "uncertain")

    def test_budget_covers_positions_first(self):
        rows = [trajectory(0), trajectory(1), trajectory(2, key="b")]
        self.assertEqual(probe_indices(rows, 2), [0, 2])

    def test_only_verified_original_harness_targets_enter_sft(self):
        rows = [trajectory(0, "h"), trajectory(1, "w"),
                trajectory(2, "h", reward=1), trajectory(3, "w", reward=1)]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            probe = out / "probe.py"
            probe.write_text("# intervention")
            def replay(key, path, seed):
                is_probe = path == probe
                return trajectory(key=key, reward=int(is_probe and key == "h"),
                                  harness="new" if is_probe else "old")
            selected, report = route_and_select(rows, replay=replay, original_harness="old",
                probe_harness=probe, out=out, probe_budget=2)
            self.assertEqual([r["sample_index"] for r in selected], [3])
            self.assertEqual(report["extra_rollouts"], 4)
            self.assertTrue(report["harness_search_enabled"])
            self.assertEqual(report["weight_task_keys"], ["w"])


if __name__ == "__main__":
    unittest.main()
