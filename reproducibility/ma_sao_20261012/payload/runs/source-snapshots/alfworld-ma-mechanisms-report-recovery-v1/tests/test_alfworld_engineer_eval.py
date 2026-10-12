import inspect
import json
from contextlib import contextmanager
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.alfworld_engineer_eval import (
    Jobs, _model_options, checked_scores, generation_args,
    normalize_rewards, official_modules, proposal_report, reserve_ports, run, run_job,
)
from sao_bridge.analysis import paired_difference
from sao_bridge.budget import contract_digest
from sao_bridge.common import read_json, write_json
from sao_bridge.launch import Launcher


class AlfworldEngineerEvalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_sparse_reward_mapping_is_exact_and_never_fills_missing_tasks(self):
        ids = [240, 247, 290]
        expected = {240: 1.0, 247: 0.0, 290: 1.0}
        self.assertEqual(normalize_rewards({"0": 1, "1": 0, "2": 1}, ids), expected)
        self.assertEqual(normalize_rewards({"240": 1, "247": 0, "290": 1}, ids), expected)
        for values in ({"0": 1}, {"1": 1, "2": 0, "3": 1}, {"0": 1, "1": 0.5, "2": 1}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                normalize_rewards(values, ids)

    def test_checked_rollouts_reject_infrastructure_status_and_missing_reward(self):
        output = self.root / "rollout"
        output.mkdir()
        rows = [
            {"index": index, "output": {"status": "completed", "result": {"reward": float(index == 0)}}}
            for index in range(2)
        ]
        path = output / "runs.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.assertEqual(checked_scores(output, [240, 247]), {240: 1.0, 247: 0.0})
        rows[1]["output"]["status"] = "task_error"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaises(RuntimeError):
            checked_scores(output, [240, 247])
        rows[1]["output"] = {"status": "completed", "result": {}}
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "omitted"):
            checked_scores(output, [240, 247])

    def proposals(self):
        return [
            {"sample_index": index, "decode_seed": 42 + index,
             "valid": False, "patch_sha256": None}
            for index in range(8)
        ]

    def test_invalid_and_duplicate_proposals_keep_their_matched_weight(self):
        baseline = {300: 0.0, 305: 1.0}
        control = proposal_report(self.proposals(), {}, [300, 305], baseline)
        proposals = self.proposals()
        proposals[0].update(valid=True, patch_sha256="same-patch")
        proposals[1].update(valid=True, patch_sha256="same-patch")
        candidate = proposal_report(
            proposals, {"same-patch": {300: 1.0, 305: 1.0}}, [300, 305], baseline,
        )
        self.assertEqual(candidate["samples_per_question"], 8)
        self.assertEqual(len(candidate["results"]), 16)
        self.assertEqual(candidate["accuracy"], 0.625)
        difference = paired_difference(control, candidate, draws=100)
        self.assertEqual(difference["candidate_minus_control"], 0.125)
        with self.assertRaisesRegex(ValueError, "all eight"):
            proposal_report(proposals[:2], {}, [300, 305], baseline)

    def test_generation_continues_the_same_prefill_with_an_explicit_decode_seed(self):
        args = generation_args({"temperature": 0.7, "tokens": 4096, "decode_seed": 5000045, "url": "http://local/v1"})
        self.assertEqual(args.engineer_assistant_prefill, "<think>\n")
        self.assertEqual(args.engineer_extra_body, {"add_generation_prompt": False, "seed": 5000045})
        self.assertFalse(args.engineer_use_env_proxy)
        options = _model_options("model", port=19000, gpus="0", name="e0", replicas=1, context=16384)
        inspect.signature(Launcher.vllm).bind(None, **options)

    def test_generation_transport_failure_is_not_a_no_patch_model_result(self):
        request = {
            "kind": "generate", "temperature": 0.7, "tokens": 4096,
            "decode_seed": 42, "url": "http://local/v1", "messages": [],
        }
        job = self.root / "job.json"
        output = self.root / "attempt"
        write_json(job, {"request": request, "output": str(output), "attempt": 1})
        evaluator = Mock()
        evaluator.call_chat.side_effect = RuntimeError("HTTP failure")
        with patch("sao_bridge.alfworld_engineer_eval.official_modules", return_value=(evaluator, Mock())):
            with self.assertRaisesRegex(RuntimeError, "HTTP failure"):
                run_job(job)
        self.assertTrue((output / "FAILED.json").exists())
        self.assertFalse((output / "result.json").exists())

    def test_jobs_retry_only_failure_and_reuse_completed_proposals(self):
        launch = Mock()
        processes = []

        def module(module_name, args, **kwargs):
            job = read_json(Path(args[1]))
            process = Mock()
            attempt = job["attempt"]
            process.poll.return_value = 1 if attempt == 1 else 0
            process.returncode = process.poll.return_value
            if attempt == 2:
                write_json(Path(job["output"]) / "result.json", {
                    "status": "complete", "request_sha256": contract_digest(job["request"]),
                    "valid": False, "model_failure": "invalid patch",
                })
            processes.append(process)
            return process

        launch.module.side_effect = module
        jobs = Jobs(self.root / "jobs", launch, 1)
        request = {"name": "proposal-0", "kind": "generate"}
        jobs.add(request)
        with patch("sao_bridge.alfworld_engineer_eval.time.sleep"):
            with patch("builtins.print"):
                result = jobs.wait()
        self.assertFalse(result["proposal-0"]["valid"])
        self.assertEqual(len(processes), 2)
        cached = Jobs(self.root / "jobs", launch, 1)
        cached.add(request)
        self.assertEqual(cached.wait(), result)
        self.assertEqual(len(processes), 2)
        with self.assertRaisesRegex(ValueError, "frozen"):
            Jobs(self.root / "jobs", launch, 1).add({**request, "kind": "score"})

    def test_sparse_official_patch_call_has_exact_task_ids_and_rejects_its_fallback(self):
        request = {
            "name": "score-fixture", "kind": "score", "task_ids": [240, 247],
            "port_index": 0, "baseline": {"240": 0, "247": 1},
            "patch": {"benchmark": "alfworld", "actions": []},
            "url": "http://local/v1", "concurrency": 2,
        }
        job = self.root / "job.json"
        write_json(job, {"request": request, "output": str(self.root / "attempt"), "attempt": 1})
        evaluator = Mock()
        evaluator.evaluate_patch.return_value = {
            "eval_status": "eval_failed_treated_as_no_patch", "patched_rewards": {},
        }
        with (
            patch("sao_bridge.alfworld_engineer_eval.official_modules", return_value=(evaluator, Mock())),
            patch("sao_bridge.alfworld_engineer_eval.prepare_agentbench_runtime", return_value=self.root),
            patch("sao_bridge.alfworld_engineer_eval.cpu_env", return_value={}),
            patch("sao_bridge.alfworld_engineer_eval.reserve_ports", return_value=(12000, 22000, [])),
        ):
            with self.assertRaisesRegex(RuntimeError, "not a model score"):
                run_job(job)
        metadata = evaluator.evaluate_patch.call_args.args[1]["metadata"]
        self.assertEqual(metadata["task_ids"], [240, 247])
        self.assertEqual(metadata["batch_size"], 2)
        self.assertEqual(metadata["baseline_rewards"], {240: 0.0, 247: 1.0})

    def test_real_port_allocator_excludes_ephemeral_and_locks_parallel_jobs(self):
        _, reward = official_modules()
        job = {
            "request": {"name": "same-seed"}, "attempt": 1,
            "port_locks": str(self.root / "ports"),
        }
        import os
        with patch.dict(os.environ, {}):
            left = reserve_ports(reward, job)
            try:
                right = reserve_ports(reward, job)
                try:
                    self.assertFalse(set(left[:2]) & set(right[:2]))
                    self.assertTrue(all(
                        not reward._port_is_ephemeral(port) for port in (*left[:2], *right[:2])
                    ))
                finally:
                    reward._release_port_locks(right[2])
            finally:
                reward._release_port_locks(left[2])

    def test_infrastructure_retry_credits_apply_only_to_named_jobs(self):
        write_json(self.root / "infrastructure_recovery.json", {
            "additional_attempts": {"affected": 2},
            "preserved_file_sha256": {},
        })
        jobs = Jobs(self.root / "jobs", Mock(), 1)
        self.assertEqual(jobs.retry_credits.get("affected", 0), 2)
        self.assertEqual(jobs.retry_credits.get("new-report", 0), 0)
        write_json(self.root / "infrastructure_recovery.json", {
            "additional_attempts": {"affected": 100},
            "preserved_file_sha256": {},
        })
        with self.assertRaises(ValueError):
            Jobs(self.root / "jobs", Mock(), 1)

    def controller_fixture(self):
        root = self.root / "study/evaluation"
        training = root.parent / "training"
        manifest_path = self.root / "manifest.json"
        solver_identity = self.root / "solver.json"
        write_json(manifest_path, {})
        write_json(solver_identity, {})
        write_json(root.parent / "TRAINING_NATIVE_AUDIT.json", {
            "status": "complete", "source_manifest_matches": True,
        })
        write_json(training / "optimizer/trained.json", {
            "checkpoint_history": {"actor": [
                {"global_step": step, "path": str(self.root / f"step-{step}")}
                for step in (15, 31)
            ]},
        })
        spec = {
            "run": str(root), "training_root": str(training),
            "data_manifest": str(manifest_path), "solver_identity_receipt": str(solver_identity),
            "initial_engineer": "initial", "solver_model": "solver",
            "checkpoint_steps": [15, 31], "seed": 42,
            "engineer_temperature": 0.7, "engineer_tokens": 4096,
            "target_rollout_concurrency": 16, "candidate_evaluation_concurrency": 8,
            "selection_min_delta": 0.02, "report_min_delta": 0.03,
            "require_positive_report_lcb": True, "bootstrap_draws": 100,
        }
        manifest = {"selection_task_ids": list(range(240, 264)), "report_task_ids": list(range(300, 370))}
        prompts = [{"id": f"packet-{index}", "messages": []} for index in range(8)]
        return root, spec, manifest, prompts

    def execute_controller(self, *, cached=False):
        root, spec, manifest, prompts = self.controller_fixture()
        if cached:
            for label in ("e0", "step15", "step31"):
                for sample in range(8):
                    write_json(root / "jobs" / f"generate-{label}-{sample}" / "result.json", {
                        "status": "complete",
                    })
        requests = {}
        jobs = Mock()
        jobs.add.side_effect = lambda request: requests.update({request["name"]: request})

        def results():
            return {
                name: (
                    {"scores": {task_id: 0.0 for task_id in request["task_ids"]}}
                    if request["kind"] == "baseline" else
                    {
                        **request, "valid": request["model_label"] != "e0",
                        "patch_sha256": "improved" if request["model_label"] != "e0" else None,
                    }
                )
                for name, request in requests.items()
            }

        jobs.wait.side_effect = results
        launch = Mock()

        @contextmanager
        def vllm(**options):
            yield ",".join(f"http://127.0.0.1:{options['port'] + index}/v1" for index in range(options["data_parallel"]))

        launch.vllm.side_effect = vllm

        def reports(root, launch, spec, proposals, labels, task_ids, baseline, urls, stage):
            result = {}
            for label in labels:
                result[label] = proposal_report(
                    proposals[label], {"improved": {index: 1.0 for index in task_ids}},
                    task_ids, baseline,
                )
            return result

        release = {"foreign_gpu_pids": [], "filler_kind": "adaptive", "filler_unit_active": True}
        with (
            patch("sao_bridge.alfworld_engineer_eval.validate_spec", return_value=(spec, manifest, prompts)),
            patch("sao_bridge.alfworld_engineer_eval.gpu_release_audit", return_value=release),
            patch("sao_bridge.alfworld_engineer_eval.ensure_model_runtime_metadata"),
            patch("sao_bridge.alfworld_engineer_eval.monitor_gpu_processes"),
            patch("sao_bridge.alfworld_engineer_eval.Launcher", return_value=launch),
            patch("sao_bridge.alfworld_engineer_eval.Jobs", return_value=jobs),
            patch("sao_bridge.alfworld_engineer_eval._score_models", side_effect=reports) as score,
        ):
            complete = run(self.root / "spec.json")
        return root, complete, score, launch

    def test_controller_selects_mean_proposal_quality_and_only_then_opens_report(self):
        root, complete, score, launch = self.execute_controller()
        self.assertTrue(complete["gate_passed"])
        self.assertTrue(complete["report_opened"])
        self.assertEqual(complete["selected"]["global_step"], 15)
        self.assertEqual(complete["paired_report"]["question_count"], 70)
        self.assertEqual(complete["paired_report"]["metric"], "mean@8")
        self.assertEqual(score.call_count, 2)
        self.assertEqual(score.call_args.args[4], ["e0", "step15"])
        self.assertEqual(read_json(root / "COMPLETE.json"), complete)
        launch.close.assert_called_once()
        self.assertTrue((root / "RELEASED.json").is_file())

    def test_completed_proposals_do_not_reload_any_engineer_model(self):
        _, complete, _, launch = self.execute_controller(cached=True)
        self.assertTrue(complete["gate_passed"])
        self.assertEqual(
            [call.kwargs["name"] for call in launch.vllm.call_args_list],
            ["fixed_solver_5", "fixed_solver_extra_3"],
        )


if __name__ == "__main__":
    unittest.main()
