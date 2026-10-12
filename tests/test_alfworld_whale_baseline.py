"""CPU tests; scratch files stay inside the project and no Launcher is started."""
import asyncio
import copy
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
import uuid

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "whale_local"))
from sao_bridge import alfworld_whale_baseline as whale
from sao_bridge.common import read_json, write_json
from sao_bridge.evaluation import stable_id_seed_offset


def _fsdp_restore_cpu_worker(rank, root):
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2,
                            init_method=(root / "rendezvous").as_uri())
    try:
        def build():
            inner = FSDP(torch.nn.Linear(8, 8), device_id=torch.device("cpu"), use_orig_params=True)
            return FSDP(torch.nn.Sequential(inner, torch.nn.Linear(8, 4)),
                        device_id=torch.device("cpu"), use_orig_params=True)

        torch.manual_seed(42)
        model = build()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, foreach=False)
        inputs = torch.arange(16, dtype=torch.float32).reshape(2, 8)
        for _ in range(2):
            model(inputs).square().mean().backward()
            optimizer.step()
            optimizer.zero_grad()
        parameters = whale.state_fingerprint(dict(model.named_parameters()))
        adam = whale.state_fingerprint(optimizer.state_dict())
        path = root / f"rank-{rank}.pt"
        torch.save({"actor_state": whale.actor_checkpoint_state(model),
                    "optimizer": optimizer.state_dict()}, path)
        restored = build()
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=.001, foreach=False)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        restored_optimizer.load_state_dict(saved["optimizer"])
        whale.restore_actor_checkpoint(restored, saved["actor_state"])
        assert whale.state_fingerprint(dict(restored.named_parameters())) == parameters
        assert whale.state_fingerprint(restored_optimizer.state_dict()) == adam
        assert whale.optimizer_steps(restored_optimizer) == [2]
        restored(inputs).square().mean().backward()
        restored_optimizer.step()
        assert whale.optimizer_steps(restored_optimizer) == [3]
        bad = {**saved["actor_state"], "state_dict": {
            **saved["actor_state"]["state_dict"], "unexpected_weight": torch.zeros(1)}}
        try:
            whale.restore_actor_checkpoint(restored, bad)
        except RuntimeError as exc:
            assert "unexpected_weight" in str(exc)
        else:
            raise AssertionError("Sharded restore must reject unexpected keys")
        write_json(root / f"rank-{rank}.json",
                   {"exact_parameters": True, "exact_adam": True, "steps": 3, "strict_load": True})
    finally:
        dist.destroy_process_group()


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
        with self.assertRaisesRegex(ValueError, "v7"):
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
        for status in ("AGENT_INVALID_ACTION", "SampleStatus.AGENT_INVALID_ACTION", "agent invalid action"):
            invalid = dict(record, reward=0.0, status=status)
            whale.validate_episodes([invalid], [row], 42, 4096, exact_tokens=True)
            self.assertEqual(whale.accepted_calls([invalid]), [])
            with self.assertRaisesRegex(ValueError, "Nonterminal"):
                whale.validate_episodes([dict(invalid, reward=1.0)], [row], 42, 4096, exact_tokens=True)
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
        config = dict(self.config, mh_iterations=2, proposals_per_iteration=1,
                      reuse_solver_for_search=True)
        initial = self.artifact("initial")
        prompts, reused = [], []

        async def proposal(urls, messages, config, seed):
            prompts.append(messages[1]["content"])
            return {"valid": True, "patch": {"benchmark": "alfworld", "actions": [],
                    "description": f"hypothesis-{len(prompts)}"}, "usage": {}, "text": ""}

        def score(launch, model, harness, rows, root, config, seed, *, urls=None):
            name = root.name
            reused.append(urls)
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
        self.assertEqual(reused, [None, "http://127.0.0.1/v1", "http://127.0.0.1/v1"])
        self.assertEqual(read_json(self.root / "search/SELECTED.json")["accuracy"], .7)

    def test_native_overlay_is_opt_in_and_validated(self):
        launch = SimpleNamespace(env={"PYTHONPATH": "/source/whale_local"})
        self.assertEqual(whale.native_environment(self.config, launch), {})
        config = dict(self.config, training_kernel_overlay="/missing")
        with self.assertRaisesRegex(ValueError, "explicit pinned"):
            whale.native_environment(config, launch)

    def test_larger_online_batches_preserve_both320_episode_cycles(self):
        config = dict(self.config, cycle_batch_episode_counts=[
            [32, 32, 32, 32, 96, 96], [160, 160]])
        whale.validate_config(config)
        self.assertEqual([sum(cycle) for cycle in whale.rsft_batch_layout(config)], [320, 320])
        for layout in ([[320], [160, 160]], [[32, 128], [160, 160]],
                       [[32, 32, 32, 32, 96, 96], [160, False]]):
            with self.subTest(layout=layout), self.assertRaises(ValueError):
                whale.validate_config(dict(config, cycle_batch_episode_counts=layout))

    def test_persistent_search_loads_two_isolated_models_once(self):
        config = dict(self.config, persistent_search_services=True)
        events = []

        @contextmanager
        def serving(launch, model, name, **kwargs):
            events.append((name, kwargs))
            yield "http://127.0.0.1:19801/v1" if kwargs["port"] == 19801 else "http://127.0.0.1:19701/v1"

        with patch.object(whale, "serving", serving), \
                patch.object(whale, "_harness_search", return_value=self.root / "selected") as search:
            result = whale.harness_search(None, self.root / "solver", self.root / "harness",
                                          [{"task_id": 3}], self.root / "mh", config, 0)
        self.assertEqual(result, self.root / "selected")
        self.assertEqual(len(events), 2)
        self.assertTrue(all(event[1]["memory_fraction"] == .4 for event in events))
        self.assertEqual(search.call_args.kwargs, {
            "proposer_urls": "http://127.0.0.1:19801/v1", "solver_urls": "http://127.0.0.1:19701/v1"})

    def test_shared_environment_preserves_native_catalogue_and_independent_harnesses(self):
        rows = [{"id": "dev-3", "task_id": 3, "task_split": "sao_train_v1",
                 "task_catalogue": "/frozen/train.json", "task_catalogue_sha256": "frozen"}]
        harnesses = {name: self.artifact(name) for name in ("h1", "h2")}
        calls = []
        process = SimpleNamespace(poll=lambda: None)

        def module(module_name, args, **kwargs):
            calls.append((module_name, args, kwargs))
            write_json(self.root / "environment/READY.json", {
                "health": {"indices": {"h1": [3], "h2": [3]}}})
            return process

        launch = SimpleNamespace(module=module, ready=lambda *args: None,
                                 stop=lambda child: calls.append(child))
        with patch("sao_bridge.alfworld_solver_eval.bind_environment"):
            with whale.environment_matrix(launch, self.root, harnesses, rows, self.config) as assigned:
                self.assertEqual(assigned["h1"], [dict(rows[0], task_name="h1")])
                self.assertEqual(assigned["h2"], [dict(rows[0], task_name="h2")])
        self.assertIn("--harness-matrix", calls[0][1])
        assignments = read_json(self.root / "assignments.json")
        self.assertEqual([row["name"] for row in assignments], ["h1", "h2"])
        self.assertEqual([row["catalogue_sha256"] for row in assignments], ["frozen", "frozen"])
        self.assertEqual(calls[-1], process)

    def test_parallel_candidate_scores_keep_order_seeds_and_bound_concurrency(self):
        harnesses = {name: self.artifact(name) for name in ("h1", "h2", "h3")}
        rows = [{"id": "same-seed-identity", "task_id": 3}]
        active, maximum, calls = 0, 0, []

        @contextmanager
        def environments(launch, work, patches, actual_rows, config):
            self.assertEqual(actual_rows, rows)
            yield {name: [dict(rows[0], task_name=name)] for name in patches}

        async def collect(urls, assigned, harness, config, seed, *, training, report_path):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            calls.append((assigned, seed, training, report_path))
            await asyncio.sleep(.01 if harness.parent.name == "h1" else 0)
            active -= 1
            return {"accuracy": {"h1": .1, "h2": .2, "h3": .3}[harness.parent.name]}

        config = dict(self.config, search_evaluation_concurrency=2)
        with patch.object(whale, "environment_matrix", environments), \
                patch.object(whale, "collect", side_effect=collect):
            result = whale.score_search_candidates(None, "http://127.0.0.1/v1",
                                                   harnesses, rows, self.root, config)
        self.assertEqual(list(result), ["h1", "h2", "h3"])
        self.assertEqual([x["accuracy"] for x in result.values()], [.1, .2, .3])
        self.assertEqual(maximum, 2)
        self.assertTrue(all(x[0][0]["id"] == rows[0]["id"] and x[1] == config["development_seed"]
                            and x[2] is False and x[3].name == "report.json" for x in calls))

    def test_parallel_candidate_failure_cancels_other_scores_without_fallback(self):
        harnesses = {name: self.artifact(name) for name in ("h1", "h2")}
        cancelled = []

        @contextmanager
        def environments(*args):
            yield {name: [{"task_name": name}] for name in harnesses}

        async def collect(urls, rows, harness, *args, **kwargs):
            if harness.parent.name == "h1":
                await asyncio.sleep(.01)
                raise RuntimeError("explicit environment failure")
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.append(harness.parent.name)
                raise

        config = dict(self.config, search_evaluation_concurrency=2)
        with patch.object(whale, "environment_matrix", environments), \
                patch.object(whale, "collect", side_effect=collect):
            with self.assertRaises(ExceptionGroup) as error:
                whale.score_search_candidates(None, "http://127.0.0.1/v1",
                                               harnesses, [{"task_id": 3}], self.root, config)
        self.assertIn("explicit environment failure", str(error.exception.exceptions[0]))
        self.assertEqual(cancelled, ["h2"])

    def test_parallel_search_keeps_invalid_slots_and_updates_only_after_the_iteration(self):
        config = dict(self.config, mh_iterations=1, proposals_per_iteration=3,
                      reuse_solver_for_search=True, search_evaluation_concurrency=2)
        initial = self.artifact("initial")
        generated = [
            {"valid": True, "patch": {"benchmark": "alfworld", "actions": [], "description": "first"}},
            {"valid": False, "error": "invalid syntax"},
            {"valid": True, "patch": {"benchmark": "alfworld", "actions": [], "description": "third"}},
        ]
        initial_report = {"accuracy": .1, "mean_turn_count": 10,
                          "results": [{"task_id": 3, "reward": 0, "turns": 10, "messages": []}]}
        scored = []

        @contextmanager
        def serving(*args):
            yield "http://127.0.0.1/v1"

        def score_candidates(launch, urls, harnesses, rows, root, actual_config):
            scored.extend(harnesses)
            return {name: {**initial_report, "accuracy": .7 if name == "h1" else .6}
                    for name in harnesses}

        with patch.object(whale, "propose", AsyncMock(side_effect=generated)), \
                patch.object(whale, "score", return_value=initial_report) as singleton, \
                patch.object(whale, "score_search_candidates", side_effect=score_candidates), \
                patch.object(whale, "serving", serving):
            result = whale.harness_search(None, self.root / "solver", initial, [{"task_id": 3}],
                                          self.root / "search", config, 0)
        self.assertEqual(scored, ["h1", "h3"])
        self.assertEqual(singleton.call_count, 1)
        self.assertEqual(result.parent.name, "h1")
        archive = read_json(self.root / "search/iteration-00.json")["archive"]
        self.assertEqual([row["name"] for row in archive], ["h0", "h1", "h2", "h3"])
        self.assertFalse(archive[2]["valid"])

    def test_search_and_early_stop_settings_are_validated(self):
        for change in ({"reuse_solver_for_search": "yes"}, {"proposal_concurrency": 0},
                       {"proposal_concurrency": 4}, {"early_stop_development_drop": 0},
                       {"early_stop_development_drop": True}, {"search_evaluation_concurrency": 0},
                       {"search_evaluation_concurrency": 9}, {"search_evaluation_concurrency": True},
                       {"search_evaluation_concurrency": 2}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                whale.validate_config(dict(self.config, **change))
        self.assertTrue(whale.development_regressed(.7, .5, .2))
        self.assertFalse(whale.development_regressed(.7, .6, .2))
        self.assertFalse(whale.development_regressed(.7, .8, .2))
        with self.assertRaises(ValueError):
            whale.development_regressed(.7, float("nan"), .2)
        with patch.object(whale.time, "time", return_value=100):
            self.assertFalse(whale.training_time_exhausted({}))
            self.assertFalse(whale.training_time_exhausted({"training_deadline_unix": 101}))
            self.assertTrue(whale.training_time_exhausted({"training_deadline_unix": 100}))

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
        record.update(status="task error", reward=0)
        raw = self.root / "failed-collect/report.json"
        with patch("sao_bridge.evaluation.Sampler", return_value=FakeSampler()), \
                patch("sao_bridge.alfworld_domain.run_episode", side_effect=native_episode), \
                self.assertRaisesRegex(RuntimeError, "infrastructure"):
            asyncio.run(whale.collect("http://127.0.0.1:1234/v1", [row], harness,
                                     self.config, 42, training=True, report_path=raw))
        self.assertEqual(read_json(raw)["results"][0]["status"], "task error")

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

    def test_fsdp_original_parameter_shards_and_warm_adam_restore_strictly_on_cpu(self):
        import torch.multiprocessing as mp
        with patch.dict(whale.os.environ, {"GLOO_SOCKET_IFNAME": "lo"}):
            mp.spawn(_fsdp_restore_cpu_worker, args=(self.root,), nprocs=2, join=True)
        for rank in range(2):
            self.assertEqual(read_json(self.root / f"rank-{rank}.json"), {
                "exact_parameters": True, "exact_adam": True, "steps": 3, "strict_load": True})
        with self.assertRaisesRegex(ValueError, "legacy local-state"):
            whale.restore_actor_checkpoint(None, {"format": "local-state"})

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

    def test_queue_smoke_failure_propagates_without_overwriting_baseline_failure(self):
        config_path = self.root / "config.json"
        write_json(config_path, self.config)
        frozen = {"config": self.config, "config_relative_path": str(config_path.relative_to(ROOT)),
                  "config_sha256": whale.file_sha(config_path),
                  "adapter_sha256": whale.file_sha(Path(whale.__file__))}
        complete, released = self.root / "best_COMPLETE.json", self.root / "best_RELEASED.json"
        release = {"foreign_gpu_pids": [], "filler_kind": "adaptive", "filler_unit_active": True}
        write_json(complete, {"status": "unavailable"})
        write_json(released, release)
        for outcome in ("failed", "passed", "exception", "existing_baseline_failure"):
            case = self.root / outcome
            frozen_path = case / "FROZEN.json"
            write_json(frozen_path, frozen)
            queue = case / "queue"
            execution_failure = case / "execution/FAILED.json"
            if outcome == "existing_baseline_failure":
                write_json(execution_failure, {"error": "original baseline failure", "consumed_episodes": 128})
                original = execution_failure.read_bytes()
            with self.subTest(outcome=outcome), \
                    patch.object(whale.os, "uname", return_value=SimpleNamespace(nodename="dev-NDH200-026")), \
                    patch.object(whale.os, "getuid", return_value=1008), \
                    patch.object(whale.os, "getgid", return_value=1008), \
                    patch("sao_bridge.resources.gpu_release_audit", return_value=release), \
                    patch.object(whale, "run_gpu_smoke", return_value={
                        "status": outcome, "frozen_sha256": whale.file_sha(frozen_path)},
                        side_effect=RuntimeError("GPU smoke failed") if outcome in (
                            "exception", "existing_baseline_failure") else None), \
                    patch.object(whale, "run") as baseline, \
                    self.assertRaisesRegex((RuntimeError, ValueError), "smoke"):
                whale.queued_run(frozen_path, whale.file_sha(frozen_path), "dev-NDH200-026",
                                 queue, case / "smoke", complete, released, 86400)
            baseline.assert_not_called()
            state = read_json(queue / "STATUS.json")
            self.assertEqual(state["stage"], "failed")
            self.assertFalse(state["baseline_started"])
            if outcome == "existing_baseline_failure":
                self.assertEqual(execution_failure.read_bytes(), original)
            else:
                failure = read_json(execution_failure)
                self.assertEqual(failure["status"], "failed")
                self.assertFalse(failure["baseline_started"])
                self.assertEqual(failure["stage"], "gpu_smoke")
                self.assertEqual(failure["originating_queue_receipt"], str(queue / "FAILED.json"))
                self.assertEqual(read_json(queue / "FAILED.json")["error"], failure["error"])
            self.assertFalse((case / "execution/COMPLETE.json").exists())
            self.assertFalse((case / "execution/RELEASED.json").exists())

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
        self._smoke_orchestration(False)

    def test_mixed_native_cost_counts_real_rank_calls_and_masked_padding(self):
        write_json(self.root / "cycle-00-rsft-00/optimizer/COMMITTED.json", {
            "accepted_calls": 5, "response_tokens": 20, "metrics": {"actor/sft_updates": 1},
        })
        write_json(self.root / "cycle-00-rsft-01/optimizer/COMMITTED.json", {
            "accepted_calls": 9, "response_tokens": 37, "metrics": {"actor/sft_updates": 1},
            "replicated_data_across_ranks": 1, "physical_rsft_forward_calls": 16,
            "physical_rsft_response_token_positions": 44, "masked_padding_calls": 7,
        })
        result = whale.cost_report(self.root, self.config)
        self.assertEqual(result["logical_rsft_forward_calls"], 14)
        self.assertEqual(result["rank_replicated_rsft_forward_calls"], 56)
        self.assertEqual(result["rank_replicated_rsft_response_token_positions"], 204)
        self.assertTrue(result["mixed_native_execution_modes"])
        self.assertIsNone(result["replicated_rank_multiplier"])
        self.assertEqual(result["masked_native_padding_calls"], 7)

    def test_native_prefix_rejects_learning_changes_and_changed_saved_evidence(self):
        origin = self.root / "original"
        write_json(origin / "FROZEN.json", {"config": self.config})
        commit = origin / "execution/cycle-00-rsft-01/optimizer/COMMITTED.json"
        write_json(commit, {"checkpoint": "frozen"})
        pause = origin / "THROUGHPUT_PAUSED.json"
        write_json(pause, {
            "status": "coordinator_paused_after_native_commit",
            "training_episodes": 2 * self.config["episodes_per_batch"],
            "native_commit_sha256": whale.file_sha(commit),
        })
        prefix = {
            "root": str(origin), "batches": 2, "frozen_sha256": whale.file_sha(origin / "FROZEN.json"),
            "pause_sha256": whale.file_sha(pause), "pins": {
                str(commit.relative_to(origin)): whale.file_sha(commit)}}
        config = dict(self.config, parallel_rsft_samples=True, parallel_checkpoint_hashing=True,
                      defer_reload_to_use=True, native_prefix=prefix)
        self.assertEqual(whale.checked_prefix(config), (origin, self.config))
        with self.assertRaisesRegex(ValueError, "learning/data"):
            whale.checked_prefix(dict(config, learning_rate=self.config["learning_rate"] * 2))
        write_json(commit, {"checkpoint": "changed"})
        with self.assertRaisesRegex(ValueError, "paused checkpoint"):
            whale.checked_prefix(config)

    def test_sharded_gpu_smoke_exercises_full_context_on_every_rank(self):
        self._smoke_orchestration(True)

    def _smoke_orchestration(self, sharded):
        config = dict(self.config, parallel_rsft_samples=True) if sharded else self.config
        frozen = {"config": config, "config_relative_path": str(whale.DEFAULT_CONFIG.relative_to(ROOT)),
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
                count = (8 if sharded else 2) * (2 if index == 0 else 1)
                steps = ([[], [1]] if index == 0 else [[2]]) if sharded else (
                    [[], [], [1], [1]] if index == 0 else [[2], [2]])
                proof = {"steps_before": [] if index == 0 else [2],
                         "steps_after": [2 if index == 0 else 3], "restored_exactly": index == 1,
                         "forward_evidence": [{"input_tokens": 32768, "response_mask_tokens": 4096,
                                              "optimizer_steps": step} for step in steps]}
                path = Path(request["output"])
                write_json(path / "COMMITTED.json", {
                    "checkpoint": str(path / "actor"), "purpose": "synthetic_gpu_smoke",
                    "accepted_episodes": 0, "synthetic_calls": count, "response_tokens": count * 4096,
                    "physical_rsft_forward_calls": count * (1 if sharded else 8),
                    "physical_rsft_response_token_positions": count * 4096 * (1 if sharded else 8),
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
        self.assertEqual(result["rank_replicated_forward_calls"], 24 if sharded else 48)
        self.assertEqual(result["logical_optimizer_steps"], 3)
        self.assertEqual(result["full_context_tokens"], 32768)
        self.assertEqual(result["response_mask_tokens"], 4096)
        self.assertEqual(result["logical_synthetic_response_tokens"], 98304 if sharded else 24576)
        self.assertEqual(result["rank_replicated_synthetic_response_positions"], 98304 if sharded else 196608)
        self.assertIs(result["sample_sharded_rsft"], sharded)
        self.assertTrue(result["full_context_memory_validated"])
        self.assertEqual(Path(requests[0]["model"]), ROOT / self.config["initial_solver"])
        self.assertEqual(requests[1]["resume"], requests[0]["output"])
        self.assertTrue(all("report" not in request for request in requests))
        self.assertTrue((output / "SMOKE_COMPLETE.json").is_file())

    def test_complete_alternation_wiring_and_final_selection_precede_eval(self):
        self._alternation_wiring(False)

    def test_reload_can_be_verified_by_actual_use_without_a_second_model_start(self):
        self._alternation_wiring(True)

    def _alternation_wiring(self, defer):
        config = dict(self.config, cycles=2, rsft_batches_per_cycle=2, episodes_per_batch=1,
                      pinned_files={}, defer_reload_to_use=defer)
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
        batch_events = ["train-rollout", "rsft"] + ([] if defer else ["reload"])
        self.assertEqual(events, batch_events * 2 + ["mh"] +
                         batch_events * 2 + ["mh", "final-eval", "close"])
        self.assertIsNone(resumes[0])
        self.assertTrue(all(resumes[1:]))
        self.assertEqual(trained_record_ids, [row["id"] for row in rows])
        self.assertTrue((self.root / "execution/RELEASED.json").is_file())
        self.assertTrue(all(read_json(self.root / "execution" / f"cycle-{cycle:02d}-rsft-{step:02d}"
                                      / "RELOAD_VERIFIED.json")["rollout_reload_verified"]
                            for cycle in range(2) for step in range(2)))


if __name__ == "__main__":
    unittest.main()
