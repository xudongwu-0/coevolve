import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from .packet_report import aggregate, paired_bootstrap


class PacketReportTest(unittest.TestCase):
    def test_empty_failure_packet_means_all_baselines_passed(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "driver"
            output.mkdir()
            (output / "failure_packets.jsonl").write_text("")
            complete = {
                "status": "completed",
                "results": [{
                    "pairing": "rl_agent_sft", "start": 10, "size": 2,
                    "selected_attempt": 1,
                    "attempts": [{
                        "output": str(output),
                        "summary": {"status": "baseline_all_pass_no_failures"},
                    }],
                }],
            }
            (root / "COMPLETE.json").write_text(json.dumps(complete))
            report = aggregate([root])
            self.assertEqual(report["baseline_pass"], 2)
            self.assertEqual(report["patched_pass"], 2)

    def test_invalid_packet_uses_baseline_fallback(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "driver"
            output.mkdir()
            (output / "failure_packets.jsonl").write_text(json.dumps({
                "metadata": {"baseline_rewards": {"0": 0.0, "1": 1.0}},
            }) + "\n")
            complete = {
                "status": "completed",
                "results": [{
                    "pairing": "rl_agent_sft", "start": 0, "size": 2,
                    "selected_attempt": 1,
                    "attempts": [{
                        "output": str(output),
                        "summary": {
                            "categories": {
                                "infra_failure": 0, "valid_evaluated": 0,
                                "invalid_patch": 1, "no_op": 0,
                            },
                            "paired_tasks": [],
                        },
                    }],
                }],
            }
            (root / "COMPLETE.json").write_text(json.dumps(complete))
            report = aggregate([root])
            self.assertEqual(report["baseline_pass"], 1)
            self.assertEqual(report["patched_pass"], 1)
            self.assertEqual(report["invalid_packets"], 1)

    def test_paired_bootstrap_accepts_uniform_gain(self):
        reference = {"task_rows": [
            {"task_id": index, "patched_reward": 0.0} for index in range(20)
        ]}
        candidate = {"task_rows": [
            {"task_id": index, "patched_reward": 1.0} for index in range(20)
        ]}
        result = paired_bootstrap(reference, candidate, "patched_reward", samples=100)
        self.assertTrue(result["accepted_1pp_rule"])
        self.assertEqual(result["one_sided_95pct_lower_bound"], 1.0)


if __name__ == "__main__":
    unittest.main()
