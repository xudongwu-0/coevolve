import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.alfworld_domain import run_episode
from sao_bridge.alfworld_env_service import task_definition
from sao_bridge.alfworld_h_train import reward_config
from sao_bridge.alfworld_runtime import prepare_agentbench_runtime
from sao_bridge.alfworld_reward_service import (
    create_app, official_args, reward_data_identity, validate_metadata,
)
from sao_bridge.alfworld_solver_validation import (
    _run_evaluation, _run_training, no_patch_assignments, validate_spec,
)
from sao_bridge.alfworld_train_data import (
    TRAIN_SPLIT, VALIDATION_SPLITS, build_catalogue, evaluation_partitions,
    goal_group, prepare,
    solver_rows, task_count, training_catalogue,
)
from sao_bridge.alfworld_w_train import assignment_schedule_audit
from sao_bridge.common import file_sha, read_json, write_json


def game(root, split, goal, trial="trial-0"):
    path = root / "json_2.1.1" / split / goal / trial / "game.tw-pddl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("fixture")
    return path.relative_to(root).as_posix()


def reserved_fixture(root):
    paths = [
        f"json_2.1.1/train/eval-goal-{index}/trial-0/game.tw-pddl"
        for index in range(500)
    ]
    write_json(root / "test_set.json", {"mixed": paths})
    for split in VALIDATION_SPLITS:
        game(root, split, f"reserved-{split}")


def large_catalogue_fixture(root, size=600):
    reserved_fixture(root)
    paths = [
        f"json_2.1.1/train/new-goal-{index}/trial-0/game.tw-pddl"
        for index in range(size)
    ]
    path = root / f"{TRAIN_SPLIT}.json"
    write_json(path, {"train": paths})
    return tuple(paths), file_sha(path)


class AlfworldEngineerCatalogueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data_root = self.root / "shared_data"
        self.paths, digest = large_catalogue_fixture(self.data_root)
        self.catalogue = self.root / f"{TRAIN_SPLIT}.json"
        (self.data_root / self.catalogue.name).replace(self.catalogue)
        for module in ("alfworld_train_data", "alfworld_runtime", "alfworld_reward_service"):
            override = patch(f"sao_bridge.{module}.ALFWORLD_DATA", self.data_root)
            override.start()
            self.addCleanup(override.stop)
        self.config = {
            "cache_root": str(self.root / "cache"),
            "receipts": str(self.root / "receipts"),
            "port_locks": str(self.root / "locks"),
            "agentbench_dir": str(self.root / "runtime/AgentBench"),
            "target_urls": ["http://127.0.0.1:19501/v1"],
            "solver_version": "w0", "solver_weights_sha256": "a" * 64,
            "split": TRAIN_SPLIT, "catalogue": str(self.catalogue),
            "catalogue_sha256": digest,
        }
        self.metadata = {
            "batch_tag": "train-512", "split": TRAIN_SPLIT, "start": 512, "end": 514,
            "batch_size": 2, "baseline_pass": 1,
            "baseline_rewards": {"512": 0.0, "513": 1.0}, "reward_threshold": 1.0,
            "target_agent_name": "qwen35-9b-nothink", "target_model": "policy",
            "response_protocol": "prefill_think_patch", "solver_version": "w0",
            "solver_weights_sha256": "a" * 64,
            "task_catalogue": str(self.catalogue), "task_catalogue_sha256": digest,
        }

    def validate(self, metadata):
        validate_metadata(metadata, "w0", **{
            name: self.config[name] for name in (
                "split", "catalogue", "catalogue_sha256", "solver_weights_sha256",
            )
        })

    def test_train_packet_identity_and_dynamic_bounds(self):
        self.validate(self.metadata)
        for key, value in (
            ("split", "test_set"),
            ("task_catalogue_sha256", "b" * 64),
            ("task_catalogue", str(self.root / "wrong.json")),
            ("solver_weights_sha256", "b" * 64),
            ("solver_version", "w1"),
            ("end", 601),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate({**self.metadata, key: value})
        for value in (float("nan"), float("inf"), -1, None):
            with self.subTest(reward=value), self.assertRaises(ValueError):
                self.validate({
                    **self.metadata, "baseline_rewards": {"512": value, "513": 1.0},
                })

    def test_official_cache_is_scoped_to_catalogue_and_solver_weights(self):
        args = official_args(self.config, self.root / "runtime")
        self.assertEqual(args.harness_r1_alfworld_split, TRAIN_SPLIT)
        changed = official_args(
            {**self.config, "solver_weights_sha256": "b" * 64}, self.root / "runtime",
        )
        self.assertNotEqual(
            args.harness_r1_reward_cache_protocol, changed.harness_r1_reward_cache_protocol,
        )
        with self.assertRaisesRegex(ValueError, "Solver weights"):
            reward_data_identity({**self.config, "solver_weights_sha256": None})
        legacy = {
            key: value for key, value in self.config.items()
            if key not in {"split", "catalogue", "catalogue_sha256", "solver_weights_sha256"}
        }
        self.assertIsNone(reward_data_identity(legacy))
        self.assertFalse(hasattr(
            official_args(legacy, self.root / "runtime"), "harness_r1_reward_cache_protocol",
        ))

    def test_engineer_launcher_forwards_the_exact_reward_catalogue_identity(self):
        config = {
            "tmpdir": str(self.root / "tmp"), "batch_size": 6, "solver_version": "w0",
            "training_split": TRAIN_SPLIT, "training_catalogue": str(self.catalogue),
            "training_catalogue_sha256": self.config["catalogue_sha256"],
            "solver_weights_sha256": self.config["solver_weights_sha256"],
        }
        reward = reward_config(config, self.root / "run", self.config["target_urls"])
        self.assertEqual(reward_data_identity(reward), reward_data_identity(self.config))
        self.assertEqual(reward["target_urls"], self.config["target_urls"])
        with self.assertRaises(ValueError):
            reward_config({**config, "training_catalogue_sha256": "0" * 64}, self.root, [])

    def test_live_adapter_rechecks_catalogue_before_cached_reward_calls(self):
        backend_calls = []

        async def reward_func(args, samples):
            backend_calls.append((args, samples))
            return [{"score": 0.25, "valid_patch": True, "eval_status": "ok"}]

        backend = ModuleType("examples.harness_r1.reward_alfworld_patch")
        backend.reward_func = reward_func
        types = ModuleType("relax.utils.types")
        types.Sample = SimpleNamespace
        with (
            patch.dict(sys.modules, {
                "examples.harness_r1.reward_alfworld_patch": backend,
                "relax.utils.types": types,
            }),
            patch.object(sys, "path", list(sys.path)),
            patch.dict(os.environ, {}),
        ):
            app = create_app(self.config)
            with TestClient(app) as client:
                self.assertEqual(client.get("/health").json()["split"], TRAIN_SPLIT)
                overlay = Path(self.config["agentbench_dir"]) / "data/alfworld"
                self.assertEqual((overlay / self.catalogue.name).resolve(), self.catalogue)
                self.assertFalse((self.data_root / self.catalogue.name).exists())
                request = {
                    "source": "<patch>{}</patch>", "sample_id": "same-packet",
                    "metadata": self.metadata,
                }
                response = client.post("/score", json=request)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["reward"], 0.25)
                write_json(self.catalogue, {"train": list(reversed(self.paths))})
                response = client.post("/score", json=request)
                self.assertEqual(response.status_code, 422, response.text)
                self.assertIn("checksum changed", response.json()["detail"])
            self.assertEqual(len(backend_calls), 1)
            write_json(self.catalogue, {"train": list(self.paths)})
            with self.assertRaisesRegex(ValueError, "different identity"):
                create_app({**self.config, "solver_weights_sha256": "b" * 64})


class AlfworldTrainDataTests(unittest.TestCase):
    def test_catalogue_excludes_goal_siblings_and_all_validation_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reserved_fixture(root)
            game(root, "train", "eval-goal-0", "different-trial")
            for split in VALIDATION_SPLITS:
                game(root, "train", f"reserved-{split}", "different-trial")
            for goal in ("new-a", "new-b", "new-c"):
                game(root, "train", goal)
            game(root, "train", "new-a", "trial-1")
            catalogue = build_catalogue(root)
            self.assertEqual(catalogue, build_catalogue(root))
        self.assertEqual(len(catalogue["train"]), 4)
        self.assertEqual(len({goal_group(p) for p in catalogue["train"][:3]}), 3)
        self.assertEqual(
            {goal_group(p) for p in catalogue["train"]}, {"new-a", "new-b", "new-c"},
        )

    def test_prepare_is_immutable_and_records_instance_and_goal_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reserved_fixture(root)
            game(root, "train", "new-a")
            game(root, "train", "new-a", "trial-1")
            game(root, "train", "new-b")
            output = root / "prepared"
            result = prepare(output, root)
            self.assertEqual(prepare(output, root), result)
            self.assertEqual(result["game_instances"], 3)
            self.assertEqual(result["distinct_goal_groups"], 2)
            self.assertEqual(result["reserved_goal_group_overlap"], 0)
            game(root, "train", "new-c")
            with self.assertRaisesRegex(ValueError, "Refusing to change frozen"):
                prepare(output, root)

    def test_evaluation_partitions_drop_goal_siblings_without_opening_reserved_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reserved_fixture(root)
            catalogue = read_json(root / "test_set.json")
            catalogue["mixed"][250] = (
                "json_2.1.1/train/eval-goal-249/another-trial/game.tw-pddl"
            )
            catalogue["mixed"][310] = (
                "json_2.1.1/train/eval-goal-245/another-trial/game.tw-pddl"
            )
            write_json(root / "test_set.json", catalogue)
            partitions = evaluation_partitions(root)
            self.assertNotIn(250, partitions["selection_task_ids"])
            self.assertNotIn(310, partitions["report_task_ids"])
            self.assertEqual(len(partitions["selection_task_ids"]), 59)
            self.assertEqual(len(partitions["report_task_ids"]), 99)
            self.assertTrue(all(i < 400 for i in partitions["report_task_ids"]))

    def test_cached_report_cannot_bypass_evaluation_catalogue_freeze(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reserved_fixture(root)
            digest = file_sha(root / "test_set.json")
            write_json(root / "evaluations/cached/COMPLETE.json", {"status": "complete"})
            write_json(root / "evaluations/cached/report.json", {"cached": True})
            kwargs = {
                "name": "cached", "model": "/unused", "version": "w0", "task_ids": [300],
            }
            with patch("sao_bridge.alfworld_train_data.ALFWORLD_DATA", root):
                self.assertEqual(_run_evaluation(
                    root, Mock(), {"evaluation_catalogue_sha256": digest}, **kwargs,
                ), {"cached": True})
                write_json(root / "test_set.json", {"mixed": []})
                with self.assertRaisesRegex(ValueError, "evaluation catalogue changed"):
                    _run_evaluation(
                        root, Mock(), {"evaluation_catalogue_sha256": digest}, **kwargs,
                    )

    def test_cached_catalogue_still_rejects_file_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            games, digest = large_catalogue_fixture(root)
            path = root / f"{TRAIN_SPLIT}.json"
            self.assertEqual(training_catalogue(path, digest, data_root=root), games)
            write_json(path, {"train": list(games[:-1])})
            with self.assertRaisesRegex(ValueError, "checksum changed"):
                training_catalogue(path, digest, data_root=root)

    def test_catalogue_rejects_reserved_groups_duplicates_and_nontrain_games(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reserved_fixture(root)
            for paths in (
                ["json_2.1.1/train/eval-goal-0/other-trial/game.tw-pddl"],
                ["json_2.1.1/valid_seen/new-goal/trial/game.tw-pddl"],
                ["json_2.1.1/train/new-goal/trial/game.tw-pddl"] * 2,
            ):
                with self.subTest(paths=paths):
                    path = root / f"{TRAIN_SPLIT}.json"
                    write_json(path, {"train": paths})
                    with self.assertRaises(ValueError):
                        training_catalogue(path, file_sha(path), data_root=root)

    def test_missing_reserved_data_and_noncanonical_paths_are_not_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reserved_fixture(root)
            write_json(root / "test_set.json", {"mixed": ["incomplete"]})
            with self.assertRaisesRegex(ValueError, "full official 500"):
                build_catalogue(root)
        for path in (
            "/json_2.1.1/train/goal/trial/game.tw-pddl",
            "json_2.1.1/train/../goal/trial/game.tw-pddl",
            "json_2.1.1/train/goal/./trial/game.tw-pddl",
            "json_2.1.1/train/goal//trial/game.tw-pddl",
        ):
            with self.subTest(path=path), self.assertRaises(ValueError):
                goal_group(path)

    def test_new_split_requires_checksum_and_legacy_limit_is_unchanged(self):
        self.assertEqual(task_count("test_set"), 500)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            task_count(TRAIN_SPLIT)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            task_count("valid_unseen")

    def test_environment_definition_and_row_schedule_support_more_than500_games(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            games, digest = large_catalogue_fixture(root)
            catalogue_path = root / f"{TRAIN_SPLIT}.json"
            with patch("sao_bridge.alfworld_train_data.ALFWORLD_DATA", root):
                definition = task_definition(
                    "scaled", [499, 500, 599], 1,
                    {"benchmark": "alfworld", "actions": []},
                    split=TRAIN_SPLIT, catalogue_sha256=digest, catalogue=catalogue_path,
                )
                with self.assertRaises(ValueError):
                    task_definition(
                        "legacy", [500], 1, {"benchmark": "alfworld", "actions": []},
                    )
                with self.assertRaises(ValueError):
                    task_definition(
                        "scaled", [600], 1,
                        {"benchmark": "alfworld", "actions": []},
                        split=TRAIN_SPLIT, catalogue_sha256=digest, catalogue=catalogue_path,
                    )
            self.assertEqual(definition["scaled"]["parameters"]["split"], TRAIN_SPLIT)
            rows = solver_rows(games, digest, catalogue_path=catalogue_path)
            data = root / "rows.json"
            write_json(data, rows)
            audit = assignment_schedule_audit({
                "data": str(data), "steps": 32, "batch_size": 16, "staleness": 1,
            })
            self.assertEqual(audit["required_rows"], 512)
            assignments = no_patch_assignments(root / "no-patch.json", training_rows=rows)
            self.assertEqual(len(assignments), 32)
            self.assertEqual(
                {task_id for entry in assignments for task_id in entry["task_ids"]},
                set(range(600)),
            )
            self.assertTrue(all(entry["split"] == TRAIN_SPLIT for entry in assignments))

    def test_validation_distinguishes_catalogue_ids_from_evaluation_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            games, digest = large_catalogue_fixture(root)
            catalogue_path = root / f"{TRAIN_SPLIT}.json"
            model = root / "model"
            model.mkdir()
            (model / "model.safetensors").write_bytes(b"weights")
            no_patch = root / "no-patch.json"
            write_json(no_patch, {"benchmark": "alfworld", "actions": []})
            data = root / "rows.json"
            write_json(data, solver_rows(games, digest, catalogue_path=catalogue_path))
            spec = {
                "run": str(root / "run"),
                "initial_solver": str(model),
                "no_patch": str(no_patch),
                "training_data": str(data),
                "training_split": TRAIN_SPLIT,
                "training_catalogue": str(catalogue_path),
                "training_catalogue_sha256": digest,
                "seed": 42, "actor_steps": 128, "critic_warmup_steps": 0,
                "batch_size": 4, "learning_rate": 1e-6, "critic_learning_rate": 5e-6,
                "actor_gpus": 4, "rollout_gpus": 4, "save_interval_steps": 16,
            }
            with patch("sao_bridge.alfworld_train_data.ALFWORLD_DATA", root):
                normalized = validate_spec(spec)
                self.assertEqual(normalized["training_data_sha256"], file_sha(data))
                self.assertEqual(normalized["total_steps"] * normalized["batch_size"], 512)
                test = self

                class Launch:
                    def module(self, module, args, **kwargs):
                        test.assertEqual(module, "sao_bridge.alfworld_w_train")
                        config = read_json(args[1])
                        consumed = read_json(Path(config["data"]))
                        test.assertEqual(len(consumed), 512)
                        test.assertNotEqual(config["data"], str(data))
                        assigned = {
                            index
                            for assignment in read_json(Path(config["assignments"]))
                            for index in assignment["task_ids"]
                        }
                        test.assertEqual(assigned, {row["task_id"] for row in consumed})
                        write_json(Path(config["run"]) / "optimizer/trained.json", {
                            "status": "mock-training-complete",
                        })
                        return object()

                    def wait(self, *args, **kwargs):
                        return None

                _run_training(Path(normalized["run"]), Launch(), normalized)
                with self.assertRaises(ValueError):
                    validate_spec({**spec, "training_split": "test_set"})
                evaluation = read_json(root / "test_set.json")
                evaluation["mixed"][300] = (
                    "json_2.1.1/train/eval-goal-240/another-trial/game.tw-pddl"
                )
                write_json(root / "test_set.json", evaluation)
                with self.assertRaisesRegex(ValueError, "distinct goal groups"):
                    validate_spec(spec)
                rows = solver_rows(games, digest, catalogue_path=catalogue_path)
                rows[0]["task_catalogue_sha256"] = "0" * 64
                write_json(data, rows)
                with self.assertRaisesRegex(ValueError, "catalogue identity"):
                    validate_spec(spec)

    def test_domain_routes_large_task_id_without_model_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            games, digest = large_catalogue_fixture(root)
            catalogue_path = root / f"{TRAIN_SPLIT}.json"
            requests = []

            def handle(request):
                requests.append(json.loads(request.content))
                return httpx.Response(200, json={
                    "status": "SampleStatus.COMPLETED", "reward": 1.0,
                    "messages": [], "tools": [],
                }, headers={"session_id": "fixture"})

            async def generate(*args):
                self.fail("A terminal environment must not request model generation")

            client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
            with (
                patch("sao_bridge.alfworld_train_data.ALFWORLD_DATA", root),
                patch.dict(os.environ, {
                    "SAO_ALFWORLD_CONTROLLER_URL": "http://127.0.0.1:28000/api",
                }),
                patch("sao_bridge.alfworld_domain.httpx.AsyncClient", return_value=client),
            ):
                result = asyncio.run(run_episode(
                    solver_rows(games, digest, catalogue_path=catalogue_path)[512],
                    {}, generate,
                ))
            self.assertEqual(requests[0]["index"], 512)
            self.assertEqual(result["task_split"], TRAIN_SPLIT)
            self.assertEqual(result["reward"], 1.0)

    def test_runtime_overlays_do_not_modify_or_shadow_shared_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            reserved_fixture(data)
            catalogue = root / "train.json"
            write_json(catalogue, {"train": ["fixture"]})
            with patch("sao_bridge.alfworld_runtime.ALFWORLD_DATA", data):
                runtime = prepare_agentbench_runtime(
                    root / "runtime", catalogues={TRAIN_SPLIT: catalogue},
                )
                overlay = runtime / "data/alfworld"
                self.assertFalse(overlay.is_symlink())
                self.assertEqual(
                    (overlay / f"{TRAIN_SPLIT}.json").resolve(), catalogue.resolve(),
                )
                self.assertEqual(
                    (overlay / "test_set.json").resolve(), (data / "test_set.json").resolve(),
                )
                self.assertFalse((data / f"{TRAIN_SPLIT}.json").exists())
                with self.assertRaisesRegex(ValueError, "shadow"):
                    prepare_agentbench_runtime(
                        root / "invalid", catalogues={"test_set": catalogue},
                    )
                legacy = prepare_agentbench_runtime(root / "legacy")
                with self.assertRaisesRegex(RuntimeError, "shared ALFWorld data symlink"):
                    prepare_agentbench_runtime(
                        legacy, catalogues={TRAIN_SPLIT: catalogue},
                    )


if __name__ == "__main__":
    unittest.main()
