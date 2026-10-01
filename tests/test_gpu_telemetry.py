from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.common import write_json
from sao_bridge.gpu_telemetry import (
    parse_pmon,
    summarize,
    summarize_active_training,
)


class GpuTelemetryTests(unittest.TestCase):
    def test_pmon_parser(self):
        rows = parse_pmon(
            "# gpu pid type sm mem enc dec jpg ofa command\n"
            "0 123 C 74 8 - - - - python3\n"
            "1 456 C - 2 - - - - python3\n"
            "2 - - - - - - - - -\n"
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["sm_percent"], 74)
        self.assertIsNone(rows[1]["sm_percent"])
        self.assertEqual(rows[1]["gpu"], 1)

    def test_summary_keeps_filler_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "samples.jsonl"
            records = [
                {
                    "mean_total_sm_percent": 90,
                    "mean_experiment_sm_percent": 70,
                    "mean_filler_sm_percent": 20,
                    "filler_unit_active": True,
                },
                {
                    "mean_total_sm_percent": 80,
                    "mean_experiment_sm_percent": 75,
                    "mean_filler_sm_percent": 5,
                    "filler_unit_active": True,
                },
                {"error": "transient"},
            ]
            path.write_text("\n".join(__import__("json").dumps(row) for row in records))
            result = summarize(path)
        self.assertEqual(result["valid_samples"], 2)
        self.assertEqual(result["error_samples"], 1)
        self.assertEqual(result["mean_experiment_sm_percent"], 72.5)
        self.assertEqual(result["experiment_active_sample_fraction"], 1.0)
        self.assertEqual(
            result["mean_filler_sm_percent_when_experiment_active"], 12.5
        )
        self.assertTrue(result["filler_remained_active"])

    def test_summary_reports_peak_memory_and_mean_power(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "samples.jsonl"
            records = []
            for memory, power in ((100, 200.0), (150, 300.0)):
                records.append({
                    "mean_total_sm_percent": 90,
                    "mean_experiment_sm_percent": 90,
                    "mean_filler_sm_percent": 0,
                    "filler_unit_active": True,
                    "gpus": [
                        {
                            "memory_used_mib": memory + gpu,
                            "power_watts": power + gpu,
                        }
                        for gpu in range(8)
                    ],
                })
            path.write_text("\n".join(
                __import__("json").dumps(row) for row in records
            ))
            result = summarize(path)
        self.assertEqual(result["peak_memory_used_mib"], 157)
        self.assertEqual(result["peak_memory_used_mib_by_gpu"][0], 150)
        self.assertEqual(result["mean_power_watts_by_gpu"][0], 250.0)

    def test_active_training_summary_filters_by_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            telemetry = root / "samples.jsonl"
            marker = root / "active.json"
            records = [
                {
                    "sampled_at": sampled_at,
                    "mean_total_sm_percent": experiment,
                    "mean_experiment_sm_percent": experiment,
                    "mean_filler_sm_percent": 0,
                    "filler_unit_active": True,
                }
                for sampled_at, experiment in ((1, 10), (2, 50), (3, 70))
            ]
            telemetry.write_text("\n".join(
                __import__("json").dumps(row) for row in records
            ))
            marker.write_text(__import__("json").dumps({
                "status": "completed",
                "started_at_unix": 2,
                "finished_at_unix": 3,
            }))
            result = summarize_active_training(telemetry, marker)
        self.assertEqual(result["samples"], 2)
        self.assertEqual(result["mean_experiment_sm_percent"], 60)


if __name__ == "__main__":
    unittest.main()
