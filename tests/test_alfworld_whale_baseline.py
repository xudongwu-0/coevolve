"""CPU tests; scratch files stay inside the project and no Launcher is started."""
import asyncio
import copy
from contextlib import contextmanager
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "whale_local"))
from sao_bridge import alfworld_whale_baseline as whale
from sao_bridge.common import read_json, write_json
from sao_bridge.evaluation import stable_id_seed_offset


class WhaleTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / "runs" / ("whale-cpu-test-" + uuid.uuid4().hex)
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root)
        self.config = read_json(whale.DEFAULT_CONFIG)

    def artifact(self, name):
        path = self.root / "harnesses" / name / "harness.py"
        write_json(path, {"benchmark": "alfworld", "actions": []})
        return path

    def example(self, reward=1.0):
        row = {"id": "alfworld-sao_train_v1-3", "task_id": 3, "task_split": "sao_train_v1"}
        call = {"prompt_ids": [1, 2], "response_ids": [3, 4], "temperature": 1.0}
        record = dict(row, reward=reward, sample_index=0,
                      rollout_seed=42 + stable_id_seed_offset(row["id"]), calls=[call],
                      tokens=2, turns=1, status="completed", truncated=False, messages=[])
        return row, record

    def test_real_frozen_cpu_preflight_matches_joint_prefix_without_eval_tasks(self):
        result = whale.preparation(whale.DEFAULT_CONFIG, verify_weights=False)
        self.assertEqual(result["status"], "prepared_cpu_only_gpu_not_validated")
        self.assertEqual(result["planned_training_episodes"], 640)
        self.assertEqual(result["planned_proposals"], 30)
        self.assertEqual(result["maximum_development_episodes"], 768)
        self.assertEqual(len(result["final_evaluation"]["all200"]), 200)
        self.assertEqual(len(result["final_evaluation"]["strict197"]), 197)
        self.assertFalse(result["model_weights_verified"])
        joint = read_json(ROOT / "runs/alfworld_joint_h0w0_8h_v1/data.json")
        self.assertEqual(result["plan"]["train_record_indices"], list(range(640)))
        self.assertEqual(result["plan"]["train_ids"], [row["task_id"] for row in joint[:640]])
        self.assertEqual(result["plan"]["train_record_ids"], [row["id"] for row in joint[:640]])
        self.assertEqual(result["plan"]["unique_training_tasks"], 240)
        self.assertEqual(result["plan"]["train_all200_task_overlap"], 0)
        self.assertEqual(result["plan"]["train_strict197_goal_overlap"], 0)
        self.assertEqual(result["plan"]["train_development_goal_overlap"], 0)
        self.assertEqual(result["joint_data_reference"]["sha256"],
                         "6e8a952f388e42d167e4388f638d5b5e94043f94bfaa158c4fbf420b11c6c12a")
        self.assertEqual(result["joint_data_reference"]["ma_1024_total_solver_records"], 512)
        self.assertEqual(result["gpu_smoke_contract"]["forward_input_tokens"], 32768)
        self.assertEqual(result["gpu_smoke_contract"]["response_mask_tokens"], 4096)
        self.assertEqual(result["gpu_smoke_contract"]["updates_before_export"], 2)
        self.assertFalse(result["gpu_smoke_contract"]["short_smoke_can_authorize_training"])

    def test_joint_prefix_rejects_eval_tasks_changed_context_and_reordered_records(self):
        from sao_bridge.alfworld_train_data import evaluation_paths, training_catalogue
        rows = read_json(ROOT / "runs/alfworld_joint_h0w0_8h_v1/data.json")
        games = evaluation_paths()
        dev = read_json(ROOT / "assets/alfworld_solver_train_only_v1/solver_train.json")
        catalogue = ROOT / "assets/alfworld_solver_train_only_v1/sao_train_v1.json"
        dev_games = training_catalogue(catalogue, whale.file_sha(catalogue))
        protocol = read_json(ROOT / self.config["evaluation_protocol"])
        bad_eval = copy.deepcopy(rows)
        bad_eval[0]["task_id"] = 240
        bad_context = copy.deepcopy(rows)
        bad_context[0]["task_context"] = games[240]
        bad_order = copy.deepcopy(rows)
        bad_order[0], bad_order[1] = bad_order[1], bad_order[0]
        for records in (bad_eval, bad_context, bad_order):
            with self.subTest(kind=records[0]["task_id"]), self.assertRaises(ValueError):
                whale.split_plan(records, games, self.config, development_rows=dev,
                                 development_games=dev_games, protocol=protocol)

    def test_training_projection_preserves_repeats_but_never_uses_historical_messages(self):
        rows = [{"id": f"record-{i}", "task_id": 8, "task_split": "test_set",
                 "messages": [{"role": "assistant", "content": "historical proposer prompt"}],
                 "critic_evidence": "historical evidence"} for i in range(2)]
        projected = whale.training_episode_rows(rows, [0, 1])
        self.assertEqual([row["id"] for row in projected], ["record-0", "record-1"])
        self.assertEqual([row["task_id"] for row in projected], [8, 8])
        self.assertEqual([row["source_record_index"] for row in projected], [0, 1])
        self.assertTrue(all("messages" not in row and "critic_evidence" not in row for row in projected))

    def test_changed_freeze_rejected(self):
        config = copy.deepcopy(self.config)
        config["pinned_files"]["WHALE/README.md"] = "0" * 64
        path = self.root / "config.json"
        write_json(path, config)
        with self.assertRaisesRegex(ValueError, "Frozen source/input changed"):
            whale.preparation(path, verify_weights=False)

    def test_frozen_config_relocates_with_source_root_not_preparation_main(self):
        snapshot = self.root / "snapshot"
        config = snapshot / "configs/alfworld_whale_baseline.json"
        write_json(config, self.config)
        frozen = {"config_relative_path": "configs/alfworld_whale_baseline.json",
                  "config_path": "/untrusted/mutable-main/configs/old.json"}
        with patch.object(whale, "ROOT", snapshot):
            self.assertEqual(whale.frozen_config_path(frozen), config)
            for path in ("/absolute/config.json", "../outside.json"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    whale.frozen_config_path({"config_relative_path": path})
        with self.assertRaisesRegex(ValueError, "v6"):
            whale.frozen_config_path({"config_path": str(whale.DEFAULT_CONFIG)})

    def test_snapshot_run_outputs_use_shared_runs_link(self):
        snapshot = self.root / "snapshot"
        snapshot.mkdir()
        (snapshot / "runs").symlink_to(ROOT / "runs", target_is_directory=True)
        output = self.root / "shared-output"
        with patch.object(whale, "ROOT", snapshot):
            self.assertEqual(whale.run_output_path(output), output)
            with self.assertRaises(ValueError):
                whale.run_output_path(ROOT / "configs")

    def test_execution_audit_rejects_foreign_local_import_and_source_mutation(self):
        fake = SimpleNamespace(__file__="/foreign/main/whale_local/sao_bridge/common.py")
        with patch.dict(sys.modules, {"foreign_local_helper": fake}), \
                self.assertRaisesRegex(ValueError, "Foreign mutable local-code module"):
            whale.execution_source_audit()
        with self.assertRaisesRegex(ValueError, "dependency changed"):
            whale.execution_source_audit({"source_dependency_sha256": {
                "whale_local/sao_bridge/alfworld_whale_baseline.py": "0" * 64}})

    def test_venv_main_tree_fallback_is_pruned_and_child_modules_are_wrapped(self):
        snapshot = self.root / "snapshot"
        paths = [str(snapshot / "whale_local"), "/foreign/main/whale_local", "/venv/site-packages"]
        kept, removed = whale.isolate_local_paths(paths, snapshot)
        self.assertEqual(removed, ["/foreign/main/whale_local"])
        self.assertEqual(kept, [paths[0], paths[2]])
        from unittest.mock import Mock
        launch = Mock()
        with patch("sao_bridge.launch.Launcher", return_value=launch):
            created = whale.create_launcher(self.root, budget=None)
        created.module("sao_bridge.alfworld_env_service", ["--run", self.root],
                       gpus="", name="environment")
        argv = launch.start.call_args.args[0]
        self.assertEqual(argv[2:6], ["-m", "sao_bridge.alfworld_whale_baseline", "exec-local",
                                     "sao_bridge.alfworld_env_service"])

    def test_static_source_manifest_tracks_runtime_closure_not_git_head(self):
        files = whale.local_source_manifest()
        for name in ("sao_bridge/alfworld_whale_baseline.py", "sao_bridge/alfworld_env_service.py",
                     "sao_bridge/launch.py", "harness_r1_searchqa/grpo_train.py", "math_reward.py"):
            self.assertIn("whale_local/" + name, files)
        self.assertFalse(any(".git" in name or name.startswith("docs/") for name in files))

    def test_config_cannot_replace_rsft_with_grpo_or_change_initialization(self):
        for key, value in [("score_threshold", 0), ("sft_epochs", 2),
                           ("initial_solver", "trained-w1"), ("temperature", 0.7),
                           ("sft_micro_batch_size_per_gpu", 2),
                           ("training_rows", "assets/alfworld_solver_train_only_v1/solver_train.json")]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                whale.validate_config(dict(self.config, **{key: value}))

    def test_official_mh_success_first_turn_tiebreak_and_incumbent_tie(self):
        for name in ("h0", "h1", "h2"):
            self.artifact(name)
        selected, _ = whale.select_harness([("h0", .75, 12), ("h1", .8, 30), ("h2", .8, 20)], self.root)
        self.assertEqual(selected, "h2")
        selected, _ = whale.select_harness([("h0", .8, 20), ("h1", .8, 20)], self.root)
        self.assertEqual(selected, "h0")
        with self.assertRaisesRegex(RuntimeError, "missing artifact"):
            whale.select_harness([("missing", 1.0, 1)], self.root)

    def test_official_mh_loader_never_imports_remote_proposer(self):
        function = whale.official_function(whale.MH / "benchmark.py", "compute_pareto_frontier")
        self.assertNotIn("AsyncOpenAI", function.__globals__)
        self.assertNotIn("claude_wrapper", function.__globals__)
        self.assertEqual(function([("h0", .5, 2), ("h1", .75, 1)]), [("h1", .75, 1)])

    def test_reject_nonlocal_inference(self):
        for url in ("https://example.org/v1", "http://10.0.0.1/v1", "http://user@localhost/v1",
                    "http://localhost/v1?proxy=yes"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                whale.loopback_urls(url)
        self.assertEqual(whale.loopback_urls("http://127.0.0.1:1234/v1"),
                         "http://127.0.0.1:1234/v1")

    def test_exact_episode_verification_and_success_only_rsft(self):
        row, record = self.example()
        whale.validate_episodes([record], [row], 42, 4096, exact_tokens=True)
        failed = dict(record, reward=0, status="task limit reached")
        whale.validate_episodes([failed], [row], 42, 4096, exact_tokens=True)
        self.assertEqual(whale.accepted_calls([record, failed]), record["calls"])
        self.assertEqual(whale.accepted_calls([failed]), [])
        for change in ({"rollout_seed": 1}, {"reward": float("nan")}, {"tokens": 3},
                       {"calls": []}, {"reward": 1.0, "status": "token_limit"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                whale.validate_episodes([dict(record, **change)], [row], 42, 4096, exact_tokens=True)
        with self.assertRaisesRegex(RuntimeError, "infrastructure"):
            whale.validate_episodes([dict(record, status="task error")], [row], 42, 4096, exact_tokens=True)

    def test_proposals_use_fresh_dev_traces_and_archive_not_old_packet_failures(self):
        incumbent = self.artifact("h0")
        _, record = self.example(0)
        record["messages"] = [{"role": "user", "content": "FRESH_W2_FAILURE_ONLY"}]
        messages = whale.proposal_messages(self.config, incumbent,
                                          [{"name": "h1", "success_rate": .7}], {"results": [record]}, 2)
        text = messages[1]["content"]
        self.assertIn("FRESH_W2_FAILURE_ONLY", text)
        self.assertIn('"name": "h1"', text)
        self.assertNotIn("## Trace case", text)
        self.assertEqual(messages[-1], {"role": "assistant", "content": "<think>\n"})

    def test_search_feedback_reaches_next_iteration_not_independent_best_of_n(self):
        config = dict(self.config, mh_iterations=2, proposals_per_iteration=1)
        initial = self.artifact("initial")
        prompts = []

        async def proposal(urls, messages, config, seed):
            prompts.append(messages[1]["content"])
            return {"valid": True, "patch": {"benchmark": "alfworld", "actions": [],
                    "description": f"hypothesis-{len(prompts)}"}, "usage": {}, "text": ""}

        def score(launch, model, harness, rows, root, config, seed):
            name = root.name
            return {"accuracy": {"h0": .1, "h1": .7, "h2": .6}[name],
                    "mean_turn_count": 10, "results": [{
                        "task_id": 1, "reward": 0, "turns": 10,
                        "messages": [{"role": "user", "content": f"fresh-{name}-failure"}]}]}

        @contextmanager
        def serving(*args):
            yield "http://127.0.0.1/v1"

        with patch.object(whale, "propose", side_effect=proposal), \
                patch.object(whale, "score", side_effect=score), \
                patch.object(whale, "serving", serving):
            selected = whale.harness_search(None, self.root / "frozen-solver", initial, [],
                                            self.root / "search", config, 0)
        self.assertIn("hypothesis-1", prompts[1])
        self.assertIn('"success_rate": 0.7', prompts[1])
        self.assertIn("fresh-h1-failure", prompts[1])
        self.assertNotIn("fresh-h0-failure", prompts[1])
        self.assertEqual(selected.parent.name, "h1")

    def test_parse_uses_official_code_hook_validation(self):
        source = '</think><patch>{"benchmark":"alfworld","description":"brief","actions":[{"type":"add_code_hook","hook":"make_pre_hint","code":"def hook(ctx, nb):\\n    return {\\\"message\\\": \\\"Finish the requested subgoal.\\\"}"}]}</patch>'
        self.assertEqual(whale.parse_candidate(source)["actions"][0]["type"], "add_code_hook")
        with self.assertRaises(ValueError):
            whale.parse_candidate('</think><patch>{"benchmark":"alfworld","actions":[]}</patch>')

    def test_collect_records_actual_verifier_results_and_sampled_ids(self):
        row, record = self.example()

        async def native_episode(row, spec, generate, **kwargs):
            await generate([], [], 4096, kwargs["seed"])
            return {key: value for key, value in record.items()
                    if key not in {"calls", "rollout_seed", "sample_index"}}

        sampler = SimpleNamespace(__call__=AsyncMock(), close=AsyncMock())

        class FakeSampler:
            async def __call__(self, *args):
                return {"record": record["calls"][0]}

            async def close(self):
                await sampler.close()

        harness = self.artifact("h0")
        with patch("sao_bridge.evaluation.Sampler", return_value=FakeSampler()), \
                patch("sao_bridge.alfworld_domain.run_episode", side_effect=native_episode) as native:
            report = asyncio.run(whale.collect("http://127.0.0.1:1234/v1", [row], harness,
                                               self.config, 42, training=True))
        native.assert_called_once()
        sampler.close.assert_awaited_once()
        self.assertEqual(report["accuracy"], 1)
        self.assertEqual(report["results"][0]["calls"], record["calls"])

    def test_native_rsft_cpu_gradient_is_token_weighted_not_sequence_weighted(self):
        import torch
        sys.path.insert(0, str(whale.NATIVE))
        from verl.workers.actor.dp_actor import DataParallelPPOActor
        calls = [
            {"prompt_ids": [10], "response_ids": [1], "temperature": 1.0},
            {"prompt_ids": [20, 21], "response_ids": [2, 2, 2], "temperature": 1.0},
        ]
        parameter = torch.nn.Parameter(torch.tensor(0.0))
        gradients = []

        class Actor:
            actor_module = SimpleNamespace(train=lambda: None)
            actor_optimizer = torch.optim.SGD([parameter], lr=0.1)
            config = SimpleNamespace(ppo_mini_batch_size=64, ppo_micro_batch_size_per_gpu=1,
                                     use_dynamic_bsz=False)
            use_prefix_grouper = False
            scaler = None

            def _forward_micro_batch(self, batch, temperature, calculate_entropy):
                targets = batch["responses"].float()
                return {"log_probs": -torch.square(parameter - targets)}

            def _optimizer_step(self):
                gradients.append(float(parameter.grad))
                self.actor_optimizer.step()
                return torch.tensor(abs(gradients[-1]))

        method = DataParallelPPOActor.update_sft_policy
        batch = whale.native_batch(calls, pad=0, mini_batch=64)
        self.assertEqual(batch.batch["response_mask"].sum().item(), 4)
        self.assertEqual(batch.batch["attention_mask"].sum().item(), 7)
        with patch.dict(method.__globals__, {"get_device_id": lambda: "cpu"}):
            metrics = method(Actor(), batch)
        self.assertAlmostEqual(gradients[0], -3.5)
        self.assertAlmostEqual(float(parameter.detach()), .35)
        self.assertEqual(metrics["actor/sft_updates"], 1)
        self.assertEqual(metrics["actor/sft_token_count"], 4)

    def test_run_rejects_unassigned_host_before_launcher(self):
        frozen = self.root / "frozen.json"
        write_json(frozen, {"config": self.config})
        with patch.object(whale.os, "uname", return_value=SimpleNamespace(nodename="login")), \
                self.assertRaisesRegex(RuntimeError, "assigned research node"):
            whale.run(frozen, "dev-NDH200-025")

    def test_gpu_smoke_inputs_cover_full_context_and_4096_action_mask(self):
        tokenizer = SimpleNamespace(
            eos_token_id=99,
            apply_chat_template=lambda messages, **kwargs: [1, 2, 3],
            encode=lambda value, **kwargs: [4, 5, 6],
        )
        for template in ([1, 2, 3], {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}):
            with self.subTest(template_type=type(template).__name__):
                tokenizer.apply_chat_template = lambda messages, **kwargs: template
                calls = whale.smoke_calls(tokenizer)
                self.assertEqual(len(calls), 2)
                self.assertTrue(all(len(call["prompt_ids"]) == 28672 for call in calls))
                self.assertTrue(all(len(call["response_ids"]) == 4096 for call in calls))
                self.assertTrue(all(call["response_ids"][-1] == 99 for call in calls))
                self.assertTrue(all("task_id" not in call and "reward" not in call for call in calls))

    def test_full_context_forward_really_uses_32768_not_32767_inputs(self):
        import torch
        observed = {}

        def model(**kwargs):
            observed.update(kwargs)
            return SimpleNamespace(logits=torch.zeros((1, 4097, 8), requires_grad=True))

        call = {"prompt_ids": [1] * 28672, "response_ids": [2] * 4096}
        logprobs = whale.full_context_logprobs(model, call, torch.device("cpu"), 1.0)
        self.assertEqual(observed["input_ids"].shape, (1, 32768))
        self.assertEqual(observed["logits_to_keep"], 4097)
        self.assertFalse(observed["use_cache"])
        self.assertEqual(logprobs.shape, (4096,))
        (-logprobs.mean()).backward()

    def test_short_smoke_cannot_authorize_long_training(self):
        short = {"status": "passed", "purpose": "synthetic_gpu_smoke",
                 "forward_backward_export": True, "exact_fp32_and_adam_restore": True}
        with self.assertRaisesRegex(ValueError, "full-context"):
            whale.require_full_context_smoke(short)

    def test_optimizer_fingerprint_detects_state_mutation_on_cpu(self):
        import torch
        state = {"step": torch.tensor(1.), "exp_avg": torch.tensor([.1, .2])}
        original = whale.state_fingerprint(state)
        self.assertEqual(original, whale.state_fingerprint(copy.deepcopy(state)))
        state["exp_avg"][0] += .1
        self.assertNotEqual(original, whale.state_fingerprint(state))

    def test_queue_requires_completed_best_and_clean_release(self):
        complete, released = self.root / "best_COMPLETE.json", self.root / "best_RELEASED.json"
        self.assertFalse(whale.queue_prerequisites(complete, released))
        write_json(complete, {"status": "completed"})
        write_json(released, {"foreign_gpu_pids": [123], "filler_kind": "adaptive",
                              "filler_unit_active": True})
        with self.assertRaisesRegex(ValueError, "release"):
            whale.queue_prerequisites(complete, released)
        write_json(released, {"foreign_gpu_pids": [], "filler_kind": "adaptive",
                              "filler_unit_active": True})
        self.assertTrue(whale.queue_prerequisites(complete, released))
        write_json(complete, {"status": "unavailable", "reason": "No eligible best checkpoint"})
        self.assertTrue(whale.queue_prerequisites(complete, released))
        write_json(complete, {"status": "failed"})
        with self.assertRaisesRegex(ValueError, "completed or unavailable"):
            whale.queue_prerequisites(complete, released)

    def test_queue_failure_receipt_is_checked_before_completion_exists(self):
        complete, released = self.root / "best_COMPLETE.json", self.root / "best_RELEASED.json"
        write_json(self.root / "best_FAILED.json", {"error": "upstream report failure"})
        with self.assertRaisesRegex(RuntimeError, "prerequisite failed"):
            whale.queue_prerequisites(complete, released)
        (self.root / "best_FAILED.json").unlink()
        write_json(self.root / "FAILED.json", {"error": "global prerequisite failure"})
        with self.assertRaisesRegex(RuntimeError, "prerequisite failed"):
            whale.queue_prerequisites(complete, released)

    def test_queue_persists_wait_state_without_allocating_foreign_gpus(self):
        config_path = self.root / "config.json"
        write_json(config_path, self.config)
        frozen_path = self.root / "FROZEN.json"
        write_json(frozen_path, {
            "config_relative_path": str(config_path.relative_to(ROOT)), "config_sha256": whale.file_sha(config_path),
            "adapter_sha256": whale.file_sha(Path(whale.__file__)),
        })
        complete, released = self.root / "best_COMPLETE.json", self.root / "best_RELEASED.json"
        release = {"foreign_gpu_pids": [], "filler_kind": "adaptive", "filler_unit_active": True}
        write_json(complete, {"status": "completed"})
        write_json(released, release)
        with patch.object(whale.os, "uname", return_value=SimpleNamespace(nodename="dev-NDH200-026")), \
                patch.object(whale.os, "getuid", return_value=1008), \
                patch.object(whale.os, "getgid", return_value=1008), \
                patch("sao_bridge.resources.gpu_release_audit",
                      return_value=dict(release, foreign_gpu_pids=[987])), \
                patch.object(whale, "run_gpu_smoke") as smoke, \
                patch.object(whale.time, "sleep", side_effect=InterruptedError("test stop")), \
                self.assertRaises(InterruptedError):
            whale.queued_run(frozen_path, whale.file_sha(frozen_path), "dev-NDH200-026",
                             self.root / "queue", self.root / "smoke", complete, released, 86400)
        smoke.assert_not_called()
        waiting = read_json(self.root / "queue/WAITING.json")
        self.assertEqual(waiting["purpose"], "queue_wait_not_training")
        self.assertEqual(waiting["live_release"]["foreign_gpu_pids"], [987])
        self.assertFalse((self.root / "queue/STARTED.json").exists())

    def test_run_wait_flags_route_to_persistent_smoke_then_baseline_queue(self):
        frozen_path = self.root / "FROZEN.json"
        write_json(frozen_path, {"config": self.config})
        joint = self.root / "joint"
        with patch.object(whale, "queued_run", return_value={"status": "completed"}) as queued:
            result = whale.run(frozen_path, "dev-NDH200-026", wait_for_joint=joint,
                               wait_label="best", frozen_sha256="frozen-external-digest")
        self.assertEqual(result["status"], "completed")
        queued.assert_called_once_with(
            frozen_path, "frozen-external-digest", "dev-NDH200-026",
            self.root / "queue", self.root / "gpu-smoke-queued",
            joint / "best_COMPLETE.json", joint / "best_RELEASED.json", 86400)

    def test_queue_failed_or_short_smoke_cannot_start_baseline(self):
        config_path = self.root / "config.json"
        write_json(config_path, self.config)
        frozen = {"config": self.config, "config_relative_path": str(config_path.relative_to(ROOT)),
                  "config_sha256": whale.file_sha(config_path),
                  "adapter_sha256": whale.file_sha(Path(whale.__file__))}
        frozen_path = self.root / "FROZEN.json"
        write_json(frozen_path, frozen)
        complete, released = self.root / "best_COMPLETE.json", self.root / "best_RELEASED.json"
        release = {"foreign_gpu_pids": [], "filler_kind": "adaptive", "filler_unit_active": True}
        write_json(complete, {"status": "unavailable"})
        write_json(released, release)
        for status in ("failed", "passed"):
            queue = self.root / ("queue-" + status)
            with self.subTest(status=status), \
                    patch.object(whale.os, "uname", return_value=SimpleNamespace(nodename="dev-NDH200-026")), \
                    patch.object(whale.os, "getuid", return_value=1008), \
                    patch.object(whale.os, "getgid", return_value=1008), \
                    patch("sao_bridge.resources.gpu_release_audit", return_value=release), \
                    patch.object(whale, "run_gpu_smoke", return_value={
                        "status": status, "frozen_sha256": whale.file_sha(frozen_path)}), \
                    patch.object(whale, "run") as baseline, \
                    self.assertRaisesRegex((RuntimeError, ValueError), "smoke"):
                whale.queued_run(frozen_path, whale.file_sha(frozen_path), "dev-NDH200-026",
                                 queue, self.root / "smoke", complete, released, 86400)
            baseline.assert_not_called()
            state = read_json(queue / "STATUS.json")
            self.assertEqual(state["stage"], "failed")
            self.assertFalse(state["baseline_started"])

    def test_full_cost_includes_dev_and_rank_duplication(self):
        for name, count in (("cycle-00-rsft-00", 3), ("mh-00/h0", 2), ("mh-00/h1", 2),
                            ("final200", 4)):
            write_json(self.root / name / "report.json",
                       {"results": [{}] * count, "tokens": count * 10})
        write_json(self.root / "cycle-00-rsft-00/optimizer/COMMITTED.json", {
            "accepted_calls": 5, "response_tokens": 20, "metrics": {"actor/sft_updates": 1},
        })
        write_json(self.root / "mh-00/proposals/h1.json",
                   {"valid": True, "usage": {"prompt_tokens": 40, "completion_tokens": 8}})
        costs = whale.cost_report(self.root, self.config)
        self.assertEqual(costs["actual_solver_episodes"], 11)
        self.assertEqual(costs["development_reward_episodes"], 4)
        self.assertEqual(costs["logical_rsft_response_tokens"], 20)
        self.assertEqual(costs["rank_replicated_rsft_response_token_positions"], 160)
        self.assertEqual(costs["rank_replicated_rsft_forward_calls"], 40)

    def test_gpu_smoke_orchestration_restores_after_two_steps_without_episodes(self):
        frozen = {"config": self.config, "config_relative_path": str(whale.DEFAULT_CONFIG.relative_to(ROOT)),
                  "adapter_sha256": whale.file_sha(Path(whale.__file__)),
                  "model_weights_verified": True}
        frozen_path = self.root / "FROZEN.json"
        write_json(frozen_path, frozen)
        output = self.root / "gpu-smoke"
        requests = []

        class FakeLauncher:
            def __init__(self, root, budget=None):
                pass

            def start(self, command, **kwargs):
                request = read_json(Path(command[-1]))
                index = len(requests)
                requests.append(request)
                count = 4 if index == 0 else 2
                steps = [[], [], [1], [1]] if index == 0 else [[2], [2]]
                proof = {"steps_before": [] if index == 0 else [2],
                         "steps_after": [2 if index == 0 else 3], "restored_exactly": index == 1,
                         "forward_evidence": [{"input_tokens": 32768, "response_mask_tokens": 4096,
                                              "optimizer_steps": step} for step in steps]}
                path = Path(request["output"])
                write_json(path / "COMMITTED.json", {
                    "checkpoint": str(path / "actor"), "purpose": "synthetic_gpu_smoke",
                    "accepted_episodes": 0, "synthetic_calls": count, "response_tokens": count * 4096,
                    "smoke_proofs": [dict(proof, rank=rank) for rank in range(8)],
                })
                return object()

            def wait(self, *args, **kwargs):
                pass

            def close(self):
                pass

        @contextmanager
        def fake_serving(*args):
            yield "http://127.0.0.1/v1"

        release = {"foreign_gpu_pids": [], "filler_kind": "adaptive", "filler_unit_active": True}
        with patch.object(whale.os, "uname", return_value=SimpleNamespace(nodename="dev-NDH200-026")), \
                patch.object(whale.os, "getuid", return_value=1008), \
                patch.object(whale.os, "getgid", return_value=1008), \
                patch.object(whale, "preparation", return_value=frozen), \
                patch("sao_bridge.resources.gpu_release_audit", return_value=release), \
                patch("sao_bridge.launch.Launcher", FakeLauncher), \
                patch.object(whale, "score", side_effect=AssertionError("No episodes in GPU smoke")), \
                patch.object(whale, "serving", fake_serving):
            result = whale.run_gpu_smoke(frozen_path, "dev-NDH200-026", output)
        self.assertEqual(result["research_episodes"], 0)
        self.assertEqual(result["rank_replicated_forward_calls"], 48)
        self.assertEqual(result["logical_optimizer_steps"], 3)
        self.assertEqual(result["full_context_tokens"], 32768)
        self.assertEqual(result["response_mask_tokens"], 4096)
        self.assertEqual(result["logical_synthetic_response_tokens"], 24576)
        self.assertEqual(result["rank_replicated_synthetic_response_positions"], 196608)
        self.assertTrue(result["full_context_memory_validated"])
        self.assertEqual(Path(requests[0]["model"]), ROOT / self.config["initial_solver"])
        self.assertEqual(requests[1]["resume"], requests[0]["output"])
        self.assertTrue(all("report" not in request for request in requests))
        self.assertTrue((output / "SMOKE_COMPLETE.json").is_file())

    def test_complete_alternation_wiring_and_final_selection_precede_eval(self):
        config = dict(self.config, cycles=2, rsft_batches_per_cycle=2, episodes_per_batch=1,
                      pinned_files={})
        rows = [{"id": f"joint-{i}", "task_id": i % 2,
                 "task_split": "test_set"} for i in range(4)]
        data = self.root / "rows.json"
        write_json(data, rows)
        config["training_rows"] = str(data)
        dev_data = self.root / "dev.json"
        write_json(dev_data, [{"id": "dev-4", "task_id": 4, "task_split": "sao_train_v1"}])
        config["development_rows"] = str(dev_data)
        config_path = self.root / "config.json"
        write_json(config_path, config)
        frozen = {"config": config, "config_relative_path": str(config_path.relative_to(ROOT)),
                  "config_sha256": whale.file_sha(config_path),
                  "adapter_sha256": whale.file_sha(Path(whale.__file__)),
                  "model_weights_verified": True,
                  "plan": {"train_ids": [0, 1, 0, 1], "train_record_indices": [0, 1, 2, 3],
                           "development_ids": [0]},
                  "final_evaluation": {"all200": [240, 241], "strict197": [240], "seed": 6000045}}
        frozen_path = self.root / "FROZEN.json"
        write_json(frozen_path, frozen)
        events, resumes, trained_record_ids = [], [], []

        class FakeLauncher:
            def __init__(self, root, budget=None):
                pass

            def start(self, command, **kwargs):
                request = read_json(Path(command[-1]))
                resumes.append(request["resume"])
                output = Path(request["output"])
                write_json(output / "COMMITTED.json", {
                    "checkpoint": str(output / "actor"), "accepted_calls": 1,
                    "response_tokens": 1, "metrics": {"actor/sft_updates": 1},
                })
                events.append("rsft")
                return object()

            def wait(self, *args, **kwargs):
                pass

            def close(self):
                events.append("close")

        def fake_score(launch, model, harness, selected, root, cfg, seed, training=False):
            if not training:
                self.assertTrue((self.root / "execution/FINAL_SELECTION.json").exists())
                self.assertEqual(events.count("mh"), 2)
                self.assertEqual([row["task_id"] for row in selected], [240, 241])
            else:
                self.assertTrue(all(row["task_split"] == "test_set" for row in selected))
                trained_record_ids.extend(row["id"] for row in selected)
            report = {"accuracy": 1.0, "tokens": len(selected),
                      "results": [dict(row, reward=1.0, calls=[{}], tokens=1) for row in selected]}
            write_json(root / "report.json", report)
            events.append("train-rollout" if training else "final-eval")
            return report

        def fake_search(launch, model, harness, selected, root, cfg, cycle):
            self.assertEqual([row["task_id"] for row in selected], [4])
            events.append("mh")
            return self.artifact(f"selected{cycle}")

        @contextmanager
        def fake_serving(*args):
            events.append("reload")
            yield "http://127.0.0.1/v1"

        release = {"foreign_gpu_pids": [], "filler_kind": "adaptive", "filler_unit_active": True}
        with patch.object(whale.os, "uname", return_value=SimpleNamespace(nodename="dev-NDH200-025")), \
                patch.object(whale.os, "getuid", return_value=1009), \
                patch.object(whale.os, "getgid", return_value=1008), \
                patch.object(whale, "preparation", return_value=frozen), \
                patch("sao_bridge.resources.gpu_release_audit", return_value=release), \
                patch("sao_bridge.launch.Launcher", FakeLauncher), \
                patch.object(whale, "score", side_effect=fake_score), \
                patch.object(whale, "harness_search", side_effect=fake_search), \
                patch.object(whale, "serving", fake_serving):
            result = whale.run(frozen_path, "dev-NDH200-025")
        self.assertEqual(result["weight_update_batches"], 4)
        self.assertEqual(events, ["train-rollout", "rsft", "reload"] * 2 + ["mh"] +
                         ["train-rollout", "rsft", "reload"] * 2 + ["mh", "final-eval", "close"])
        self.assertIsNone(resumes[0])
        self.assertTrue(all(resumes[1:]))
        self.assertEqual(trained_record_ids, [row["id"] for row in rows])
        self.assertTrue((self.root / "execution/RELEASED.json").is_file())


if __name__ == "__main__":
    unittest.main()
