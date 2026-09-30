import json
from pathlib import Path
import tempfile
import unittest

from sao_bridge.alfworld_domain import (
    _running, _terminal_reward, harness_spec,
)
from sao_bridge.alfworld_env_service import task_definition
from sao_bridge.alfworld_global_harness import (
    accepted_candidate, complete_without_candidates, packet_metadata,
    normalize_patched_rewards, paired_patch_improvement, receipt_candidates,
    seed_reward_cache,
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
from harness_r1_alfworld.matrix_probe import command as packet_command
from harness_r1_alfworld.packet_probe import packet_jobs


class AlfworldSaoTest(unittest.TestCase):
    def test_packet_retry_ports_do_not_overlap_full_train_first_attempt(self):
        jobs = packet_jobs(
            "phase", list(range(0, 240, 10)), 10, 4,
            [19701, 19702, 19703, 19704],
            [19601, 19602, 19603, 19604],
            Path("/target.json"), "w0", Path("/engineer.json"), Path("/runs"),
        )
        first_attempt_ports = {
            int(packet_command(job, 1)[
                packet_command(job, 1).index(flag) + 1
            ])
            for job in jobs
            for flag in ("--controller-port", "--worker-port")
        }
        retry_ports = {
            int(packet_command(job, 2)[
                packet_command(job, 2).index(flag) + 1
            ])
            for job in jobs
            for flag in ("--controller-port", "--worker-port")
        }
        self.assertTrue(first_attempt_ports.isdisjoint(retry_ports))

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

    def test_harness_round_compares_against_the_incumbent_not_no_patch(self):
        incumbent = {
            "task_rows": [
                {"task_id": task_id, "patched_reward": 1.0}
                for task_id in range(10)
            ],
        }
        candidate = {
            "task_rows": [
                {"task_id": task_id, "patched_reward": float(task_id != 0)}
                for task_id in range(10)
            ],
        }
        improvement = paired_patch_improvement(candidate, incumbent, 1000, 42)
        self.assertEqual(improvement["gain"], -0.1)
        self.assertFalse(accepted_candidate(improvement, 0.01))

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

    def test_global_candidates_use_only_consumed_optimizer_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            patch_value = {"benchmark": "alfworld", "actions": []}
            source = "reason</think><patch>" + json.dumps(patch_value) + "</patch>"
            base = {
                "evaluation": {"valid": True},
                "action_text": source,
                "calls": [{"versions": [0]}],
            }
            write_json(root / "consumed.json", base)
            write_json(
                root / "prefetched.json",
                {**base, "calls": [{"versions": [1]}]},
            )
            write_json(
                root / "invalid.json",
                {**base, "evaluation": {"valid": False}},
            )
            candidates = receipt_candidates([root], 0)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["patch"], patch_value)
        self.assertEqual(candidates[0]["candidate_source"], "optimizer_receipt")
        self.assertEqual(candidates[0]["maximum_behavior_version"], 0)

    def test_global_metadata_normalizes_an_empty_all_pass_packet(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            probe = root / "probe"
            all_pass = root / "all-pass"
            failed = root / "failed"
            probe.mkdir()
            all_pass.mkdir()
            failed.mkdir()
            (all_pass / "failure_packets.jsonl").write_text("")
            (failed / "failure_packets.jsonl").write_text(json.dumps({
                "metadata": {
                    "start": 310,
                    "end": 320,
                    "batch_size": 10,
                    "baseline_pass": 9,
                    "baseline_rewards": {
                        str(task_id): float(task_id != 311)
                        for task_id in range(310, 320)
                    },
                },
            }) + "\n")
            write_json(probe / "COMPLETE.json", {
                "results": [
                    {
                        "start": 300,
                        "size": 10,
                        "selected_attempt": 1,
                        "attempts": [{
                            "output": str(all_pass),
                            "summary": {
                                "status": "baseline_all_pass_no_failures",
                            },
                        }],
                    },
                    {
                        "start": 310,
                        "size": 10,
                        "selected_attempt": 1,
                        "attempts": [{
                            "output": str(failed),
                            "summary": {"categories": {"valid_evaluated": 1}},
                        }],
                    },
                ],
            })
            packets = packet_metadata(root, "w1")
        self.assertEqual(len(packets), 2)
        self.assertEqual(packets[0]["baseline_pass"], 10)
        self.assertEqual(set(packets[0]["baseline_rewards"]), {
            str(task_id) for task_id in range(300, 310)
        })
        self.assertEqual(packets[0]["solver_version"], "w1")

    def test_global_rewards_map_exact_packet_local_ids_to_protocol_ids(self):
        metadata = {"start": 390, "end": 400, "batch_size": 10}
        local = {str(task_id): float(task_id % 2) for task_id in range(10)}
        normalized = normalize_patched_rewards(local, metadata)
        self.assertEqual(set(normalized), set(range(390, 400)))
        self.assertEqual(normalized[393], 1.0)
        global_ids = {str(task_id): 1.0 for task_id in range(390, 400)}
        self.assertEqual(
            normalize_patched_rewards(global_ids, metadata),
            {task_id: 1.0 for task_id in range(390, 400)},
        )
        with self.assertRaisesRegex(ValueError, "neither global task IDs"):
            normalize_patched_rewards({"1": 1.0, "10": 0.0}, metadata)

    def test_global_reward_cache_seed_requires_matching_evaluation_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            previous = root / "previous"
            cache = previous / "reward_cache"
            cache.mkdir(parents=True)
            (cache / "result.json").write_text("{}", encoding="utf-8")
            (cache / ".lock").write_text("", encoding="utf-8")
            failed = cache / "failed"
            failed.mkdir()
            (failed / "reward.json").write_text(json.dumps({
                "eval_status": "eval_failed_treated_as_no_patch",
            }), encoding="utf-8")
            identity = {
                "packet_refresh": "/packets",
                "candidate_receipt_roots": ["/receipts"],
                "maximum_behavior_version": 0,
                "no_patch": "/no-patch.json",
                "solver_model": "/solver",
                "solver_version": "w0",
            }
            (previous / "config.json").write_text(
                json.dumps(identity), encoding="utf-8",
            )
            run = root / "retry"
            run.mkdir()
            seed_reward_cache(run, {
                **identity, "seed_reward_cache": str(cache),
            })
            self.assertEqual(
                (run / "reward_cache/result.json").read_text(encoding="utf-8"),
                "{}",
            )
            self.assertFalse((run / "reward_cache/.lock").exists())
            self.assertFalse((run / "reward_cache/failed").exists())
            manifest = json.loads(
                (run / "reward_cache_seed.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["files"], 1)
            self.assertEqual(
                manifest["discarded_infrastructure_entries"], ["failed"],
            )
            with self.assertRaisesRegex(ValueError, "identity changed"):
                seed_reward_cache(root / "other", {
                    **identity, "solver_version": "w1",
                    "seed_reward_cache": str(cache),
                })

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
            refreshed_solver_rows = solver_dataset(
                sorted(names), 16, 8, 1, names,
            )
            self.assertEqual(len(refreshed_solver_rows), 16)
            self.assertEqual(
                len({row["id"] for row in refreshed_solver_rows}),
                16,
            )
            self.assertTrue(all(
                row["partner_version"] == 1 for row in refreshed_solver_rows
            ))

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
