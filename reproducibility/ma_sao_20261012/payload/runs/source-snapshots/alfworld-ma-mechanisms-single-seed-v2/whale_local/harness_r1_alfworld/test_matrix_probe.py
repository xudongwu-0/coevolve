import unittest
from pathlib import Path

from .matrix_probe import command, infrastructure_failures, jobs
from .packet_probe import packet_jobs


class MatrixProbeTest(unittest.TestCase):
    def test_infrastructure_summary_requires_retry(self):
        self.assertEqual(infrastructure_failures({
            "categories": {"infra_failure": 1, "valid_evaluated": 0},
        }), 1)
        self.assertEqual(infrastructure_failures({
            "categories": {"infra_failure": 0, "invalid_patch": 1},
        }), 0)
        self.assertEqual(infrastructure_failures({
            "status": "baseline_all_pass_no_failures",
        }), 0)

    def test_matched_matrix_uses_every_replica(self):
        planned = jobs("test", [0, 10, 20, 30], 10, 4)
        self.assertEqual(len(planned), 16)
        self.assertEqual({row["pairing"] for row in planned}, {
            "rl_agent_sft", "cold_sft_agent_sft", "rl_original", "cold_sft_original",
        })
        self.assertEqual(
            sorted(row["target_port"] for row in planned if row["target"] == "agent_sft"),
            sorted((19301, 19311, 19321, 19331) * 2),
        )
        self.assertEqual(
            sorted(row["engineer_port"] for row in planned if row["engineer"] == "rl"),
            sorted((19201, 19211, 19221, 19231) * 2),
        )
        self.assertEqual(len({row["controller_port"] for row in planned}), 16)

    def test_retry_changes_output_and_ports(self):
        job = jobs("test", [0, 10, 20, 30], 10, 4)[0]
        first, second = command(job, 1), command(job, 2)
        self.assertNotEqual(first[first.index("--run-id") + 1],
                            second[second.index("--run-id") + 1])
        self.assertEqual(
            int(second[second.index("--controller-port") + 1])
            - int(first[first.index("--controller-port") + 1]),
            1000,
        )

    def test_packet_jobs_round_robin_all_replicas(self):
        planned = packet_jobs(
            "round", [0, 10, 20, 30, 40], 10, 4,
            [101, 102], [201, 202], Path("/tmp/target.json"), "w1",
            Path("/tmp/engineer.json"), Path("/tmp/runs"),
        )
        self.assertEqual(
            [job["target_port"] for job in planned], [101, 102, 101, 102, 101])
        self.assertEqual(
            [job["engineer_port"] for job in planned], [201, 202, 201, 202, 201])
        self.assertEqual([job["start"] for job in planned], [0, 10, 20, 30, 40])


if __name__ == "__main__":
    unittest.main()
