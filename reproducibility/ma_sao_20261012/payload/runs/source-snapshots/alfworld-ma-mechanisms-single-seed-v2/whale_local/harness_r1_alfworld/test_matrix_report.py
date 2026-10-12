import json
from pathlib import Path
import tempfile
import unittest

from .matrix_report import aggregate, PAIRINGS


class MatrixReportTest(unittest.TestCase):
    def test_aggregate_matched_pairings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = []
            for replica in range(4):
                for index, name in enumerate(PAIRINGS):
                    baseline = float(index < 2)
                    output = root / f"{name}-{replica}"
                    output.mkdir()
                    (output / "failure_packets.jsonl").write_text(json.dumps({
                        "metadata": {"baseline_rewards": {str(replica): baseline}},
                    }) + "\n")
                    summary = {
                        "categories": {
                            "infra_failure": 0, "invalid_patch": 0,
                            "no_op": 0, "valid_evaluated": 1,
                        },
                        "official_summary": {
                            "total_tasks": 1, "baseline_pass": int(baseline),
                            "patched_pass": 1, "delta_pass": int(not baseline),
                            "valid_patches": 1,
                        },
                        "rescued_tasks": int(not baseline), "regressed_tasks": 0,
                        "paired_tasks": [],
                    }
                    results.append({
                        "pairing": name, "start": replica, "size": 1,
                        "selected_attempt": 1,
                        "attempts": [{"summary": summary, "output": str(output)}],
                    })
            (root / "COMPLETE.json").write_text(json.dumps({
                "status": "completed", "completed": 16,
                "elapsed_seconds": 1.0, "results": results,
            }))
            report = aggregate([root])
            self.assertEqual(report["wave_count"], 1)
            self.assertEqual(report["baseline_consistency"]["agent_sft"]["mismatch_count"], 0)
            self.assertEqual(report["pairings"]["rl_original"]["delta_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
