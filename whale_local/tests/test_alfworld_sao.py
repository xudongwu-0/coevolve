import json
from pathlib import Path
import tempfile
import unittest

from sao_bridge.alfworld_domain import (
    _running, _terminal_reward, harness_spec,
)
from sao_bridge.alfworld_env_service import task_definition
from sao_bridge.alfworld_global_harness import (
    accepted_candidate, complete_without_candidates,
)
from sao_bridge.alfworld_phase_data import (
    TRAINING_VALIDATOR_REMINDER, assignments_from_matrix, editor_dataset,
    packet_rows, solver_dataset,
)
from sao_bridge.alfworld_h_train import training_spec
from sao_bridge.alfworld_prepare import partition_rows
from sao_bridge.alfworld_w_train import (
    assignment_schedule_audit, training_spec as solver_training_spec,
)
from sao_bridge.alfworld_reward_service import infrastructure_failure, validate_metadata
from sao_bridge.alfworld_runtime import prepare_agentbench_runtime
from sao_bridge.alfworld_solver_selection import paired_solver_report
from sao_bridge.budget import model_weight_digest
from sao_bridge.common import ensure_model_runtime_metadata, file_sha, write_json
from sao_bridge.gpu_report import LAYOUTS


class AlfworldSaoTest(unittest.TestCase):
    def test_global_harness_requires_declared_gain_and_positive_lcb(self):
        threshold = 0.01
        report = {"gain": 0.009, "one_sided_95pct_lower_bound": 0.001}
        self.assertFalse(accepted_candidate(report, threshold))
        self.assertTrue(accepted_candidate(
            {**report, "gain": threshold},
            threshold,
        ))
        self.assertFalse(accepted_candidate(
            {**report, "gain": 0.02, "one_sided_95pct_lower_bound": 0.0},
            threshold,
        ))
        with self.assertRaisesRegex(ValueError, "minimum_absolute_gain"):
            accepted_candidate(report, 0.0)

    def test_solver_selection_is_paired_and_uses_the_same_gate(self):
        report = paired_solver_report(
            {task_id: 0.0 for task_id in range(10)},
            {task_id: 1.0 for task_id in range(10)},
            1000,
            42,
            0.01,
        )
        self.assertTrue(report["accepted"])
        self.assertEqual(report["gain"], 1.0)
        self.assertEqual(report["one_sided_95pct_lower_bound"], 1.0)
        with self.assertRaisesRegex(ValueError, "task identity"):
            paired_solver_report({0: 0.0}, {1: 1.0}, 1000, 42, 0.01)

    def test_no_valid_global_candidate_is_a_rejected_model_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            no_patch = run / "no-patch.json"
            no_patch.write_text('{"benchmark":"alfworld","actions":[]}')
            complete_without_candidates(
                run,
                {
                    "phase": "h2-dev",
                    "no_patch": str(no_patch),
                    "selection_split": "dev-300-399",
                },
                [{"batch_size": 10}, {"batch_size": 10}],
                0.01,
            )
            result = json.loads((run / "COMPLETE.json").read_text())
            released = json.loads((run / "RELEASED.json").read_text())
        self.assertFalse(result["accepted"])
        self.assertEqual(result["candidate_count"], 0)
        self.assertEqual(result["tasks"], 20)
        self.assertEqual(result["reason"], "no_valid_harness_candidates")
        self.assertFalse(released["gpu_resources_claimed"])

    def test_runtime_metadata_supplements_do_not_replace_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, checkpoint = root / "source", root / "checkpoint"
            source.mkdir()
            checkpoint.mkdir()
            (checkpoint / "model.safetensors").write_bytes(b"weights")
            for name in ("preprocessor_config.json", "processor_config.json"):
                (source / name).write_text(f'{{"name":"{name}"}}')
            before = file_sha(checkpoint / "model.safetensors")
            weight_digest = model_weight_digest(checkpoint)
            receipt = ensure_model_runtime_metadata(checkpoint, source)
            self.assertEqual(before, file_sha(checkpoint / "model.safetensors"))
            self.assertEqual(weight_digest, model_weight_digest(checkpoint))
            self.assertTrue(receipt["weights_unchanged"])
            self.assertEqual(
                receipt["files"]["preprocessor_config.json"]["status"], "copied")
            repeated = ensure_model_runtime_metadata(checkpoint, source)
            self.assertEqual(
                repeated["files"]["preprocessor_config.json"]["status"],
                "already_present",
            )

    def test_no_patch_harness_and_task_definition(self):
        harness = harness_spec('{"actions":[],"benchmark":"alfworld"}')
        self.assertEqual(harness["source"],
                         '{"actions":[],"benchmark":"alfworld"}\n')
        task = task_definition("external", [0, 8], 2, harness["patch"])
        self.assertEqual(set(task), {"external"})
        self.assertEqual(task["external"]["parameters"]["task_ids"], [0, 8])

    def test_agentbench_runtime_is_run_local(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = prepare_agentbench_runtime(Path(temporary) / "AgentBench")
            self.assertTrue((runtime / "src").is_symlink())
            self.assertTrue((runtime / "scripts").is_symlink())
            self.assertTrue((runtime / "data/alfworld").is_symlink())
            self.assertTrue((runtime / "configs/generated").is_dir())
            marker = runtime / "configs/generated/owned-by-this-run"
            marker.write_text("ok")
            task = task_definition(
                "external", [0, 8], 2,
                {"benchmark": "alfworld", "actions": []},
                runtime,
            )
            self.assertEqual(
                task["external"]["parameters"]["data_path"],
                str(runtime / "data/alfworld"),
            )

    def test_status_and_reward(self):
        self.assertTrue(_running("SampleStatus.RUNNING"))
        self.assertFalse(_running("SampleStatus.COMPLETED"))
        self.assertEqual(_terminal_reward({"reward": 1}), 1.0)
        self.assertEqual(_terminal_reward({"metrics": {"score": 0.5}}), 0.5)
        with self.assertRaises(RuntimeError):
            _terminal_reward({"metrics": {}})

    def test_reward_metadata_and_infrastructure_classification(self):
        metadata = {
            "batch_tag": "w0-b0", "split": "test_set", "start": 10, "end": 12,
            "batch_size": 2, "baseline_pass": 1,
            "baseline_rewards": {"10": 1.0, "11": 0.0},
            "reward_threshold": 1.0, "target_agent_name": "qwen35-9b-nothink",
            "target_model": "policy", "response_protocol": "prefill_think_patch",
            "solver_version": "w0",
        }
        validate_metadata(metadata, "w0")
        with self.assertRaises(ValueError):
            validate_metadata(dict(metadata, solver_version="w1"), "w0")
        self.assertFalse(infrastructure_failure(
            {"eval_status": "invalid_patch_treated_as_no_patch"}))
        self.assertTrue(infrastructure_failure(
            {"eval_status": "eval_timeout_treated_as_no_patch"}))

    def test_phase_datasets_preserve_packet_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "packet-output"
            output.mkdir()
            metadata = {
                "start": 0, "end": 2, "batch_size": 2,
                "baseline_rewards": {"0": 0.0, "1": 1.0},
            }
            (output / "failure_packets.jsonl").write_text(json.dumps({
                "prompt": [
                    {"role": "system", "content": "engineer contract"},
                    {"role": "user", "content": "packet"},
                ],
                "metadata": metadata,
            }) + "\n")
            matrix = root / "matrix"
            write_json(matrix / "COMPLETE.json", {
                "status": "completed",
                "results": [{
                    "pairing": "rl_agent_sft", "selected_attempt": 1,
                    "attempts": [{"output": str(output), "summary": {
                        "categories": {"infra_failure": 0},
                    }}],
                }],
            })
            packets = packet_rows([matrix], "w0")
            self.assertEqual(packets[0]["metadata"]["solver_version"], "w0")
            encoded = editor_dataset(packets, 3, 0)
            self.assertEqual(len(encoded), 3)
            self.assertNotIn("metadata", encoded[0])
            self.assertIn(
                TRAINING_VALIDATOR_REMINDER,
                encoded[0]["messages"][0]["content"],
            )
            self.assertEqual(json.loads(encoded[0]["metadata_json"])["solver_version"], "w0")
            replay_dir = root / "receipts"
            replay_dir.mkdir()
            write_json(replay_dir / "one.json", {
                "role": "editor", "replayed": False, "id": "old",
                "input_messages": packets[0]["messages"], "action_text": "{}",
                "reward": 0.0, "partner_version": 0,
                "harness_source": '{"benchmark":"alfworld","actions":[]}\n',
                "metadata": dict(packets[0]["metadata"], solver_version="old-w"),
                "calls": [{
                    "prompt_ids": [1], "response_ids": [2],
                    "response_logprobs": [-0.1], "versions": [0],
                    "stop_reason": "stop", "temperature": 0.7,
                }],
            })
            replayed = editor_dataset(
                packets, 4, 1, replay_dir=replay_dir,
                replay_fraction=0.25, editor_step_offset=8,
            )
            self.assertIsNotNone(replayed[0]["replay"])
            self.assertEqual(
                json.loads(replayed[0]["metadata_json"])["solver_version"], "w0")
            no_patch = root / "h0.json"
            no_patch.write_text('{"benchmark":"alfworld","actions":[]}')
            patch_dir = output / "patched/baseline"
            patch_dir.mkdir(parents=True)
            (patch_dir / "patch.json").write_text(
                '{"benchmark":"alfworld","actions":[]}')
            matrix_record = json.loads((matrix / "COMPLETE.json").read_text())
            matrix_record["results"][0]["start"] = 0
            matrix_record["results"][0]["size"] = 2
            matrix_record["results"][0]["attempts"][0]["summary"]["categories"].update(
                valid_evaluated=1)
            write_json(matrix / "COMPLETE.json", matrix_record)
            assignments, names = assignments_from_matrix([matrix], no_patch, 0)
            self.assertEqual(assignments[0]["task_ids"], [0, 1])
            rows = solver_dataset([0, 1], 3, 1, 0, names)
            self.assertEqual([row["task_id"] for row in rows], [1, 0, 1])
            self.assertEqual(rows[0]["task_name"], "alfworld-h0-b000")
            self.assertEqual(rows[0]["id"], "alfworld-task-1-h0-sample1")

            interleaved = solver_dataset(
                [0, 1, 10, 11, 20, 21],
                6,
                0,
                0,
                {
                    0: "packet-0", 1: "packet-0",
                    10: "packet-1", 11: "packet-1",
                    20: "packet-2", 21: "packet-2",
                },
            )
            self.assertEqual(
                [row["task_id"] for row in interleaved],
                [0, 10, 20, 1, 11, 21],
            )

    def test_h_train_spec_uses_native_alfworld_workflow(self):
        config = {
            "engineer_model": "/models/e0", "solver_model": "/models/w0",
            "phase": "h1", "experiment": "campaign", "steps": 2, "batch_size": 6,
            "actor_gpus": 3, "rollout_gpus": 1,
            "harness": "/run/h0.json", "data": "/run/h1.json",
        }
        spec = training_spec(config, Path("/run"), "http://127.0.0.1:28600")
        self.assertEqual(spec["domain"], "alfworld")
        self.assertEqual((spec["actor_gpus"], spec["rollout_gpus"]), (3, 1))
        self.assertEqual(spec["gdn_prefill_backend"], "triton")
        self.assertEqual(spec["evaluator_url"], "http://127.0.0.1:28600")
        self.assertEqual(spec["microbatch_tokens"], 20000)
        self.assertEqual(LAYOUTS["h"], {
            "actor": [0, 1, 2], "rollout": [3],
            "frozen_solver": [4, 5, 6, 7],
        })

    def test_w_train_spec_uses_all_eight_gpus(self):
        config = {
            "solver_model": "/models/w0", "phase": "w1", "experiment": "campaign",
            "steps": 4, "batch_size": 16, "harness": "/run/h1.json",
            "data": "/run/w1.json",
        }
        spec = solver_training_spec(config, Path("/run"))
        self.assertEqual(spec["domain"], "alfworld")
        self.assertEqual((spec["actor_gpus"], spec["rollout_gpus"]), (4, 4))
        self.assertEqual(spec["gdn_prefill_backend"], "triton")
        self.assertIsNone(spec["evaluator_url"])

    def test_w_train_rejects_same_worker_collisions(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary) / "data.json"
            config = {"data": str(data), "steps": 1, "batch_size": 2}
            write_json(data, [
                {"task_id": 0, "task_name": "packet-0"},
                {"task_id": 1, "task_name": "packet-0"},
            ])
            with self.assertRaisesRegex(ValueError, "reuses an AgentBench assignment"):
                assignment_schedule_audit(config)
            write_json(data, [
                {"task_id": 0, "task_name": "packet-0"},
                {"task_id": 10, "task_name": "packet-1"},
            ])
            audit = assignment_schedule_audit(config)
            self.assertEqual(audit["batches"][0]["task_ids"], [0, 10])
            self.assertEqual(audit["active_window"], 4)
            config.update(steps=2, batch_size=2)
            write_json(data, [
                {"task_id": 0, "task_name": "packet-0"},
                {"task_id": 10, "task_name": "packet-1"},
                {"task_id": 1, "task_name": "packet-0"},
                {"task_id": 20, "task_name": "packet-2"},
            ])
            with self.assertRaisesRegex(ValueError, "active window"):
                assignment_schedule_audit(config)

    def test_campaign_partitions_cover_all_tasks(self):
        paths = [
            f"json_2.1.1/train/pick_and_place_simple-X-{index}/trial/game.tw-pddl"
            for index in range(500)
        ]
        data = {
            "train": {"start": 0, "end": 240, "count": 240},
            "reward": {"start": 240, "end": 300, "count": 60},
            "dev": {"start": 300, "end": 400, "count": 100},
            "test": {"start": 400, "end": 500, "count": 100},
        }
        splits = partition_rows(paths, data)
        self.assertEqual([len(splits[name]) for name in ("train", "reward", "dev", "test")],
                         [240, 60, 100, 100])
        self.assertEqual(splits["test"][0]["task_id"], 400)


if __name__ == "__main__":
    unittest.main()
