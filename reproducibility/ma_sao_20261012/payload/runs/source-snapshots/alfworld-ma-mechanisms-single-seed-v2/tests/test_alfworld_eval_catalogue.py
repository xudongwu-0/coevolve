import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))

from sao_bridge.alfworld_env_service import task_definition
from sao_bridge.alfworld_runtime import prepare_agentbench_runtime
from sao_bridge.alfworld_train_data import (
    EVAL_SPLIT, evaluation_catalogue, task_count, training_catalogue,
)
from sao_bridge.common import file_sha, read_json, write_json


class HeldoutCatalogueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        write_json(self.data / "test_set.json", {"legacy": [
            f"json_2.1.1/train/legacy-{index}/trial/game.tw-pddl" for index in range(500)
        ]})
        self.games = [self.game("valid_seen", f"new-{index}") for index in range(3)]
        self.catalogue = self.root / "heldout.json"
        self.freeze({"valid_seen": self.games})

    def game(self, split, group, trial="trial"):
        path = self.data / "json_2.1.1" / split / group / trial / "game.tw-pddl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic catalogue fixture")
        return path.relative_to(self.data).as_posix()

    def freeze(self, value):
        write_json(self.catalogue, value)
        self.digest = file_sha(self.catalogue)

    def load(self):
        return evaluation_catalogue(self.catalogue, self.digest, data_root=self.data)

    def test_explicit_reserved_games_are_supported_without_reinterpreting_training(self):
        self.assertEqual(self.load(), tuple(self.games))
        with patch("sao_bridge.alfworld_train_data.ALFWORLD_DATA", self.data):
            self.assertEqual(task_count(EVAL_SPLIT, self.digest, catalogue_path=self.catalogue), 3)
            self.assertEqual(task_count("test_set"), 500)
        with self.assertRaisesRegex(ValueError, "only the train"):
            training_catalogue(self.catalogue, self.digest, data_root=self.data)

    def test_catalogue_requires_checksum_and_cache_cannot_hide_changes(self):
        self.load()
        old_digest = self.digest
        self.freeze({"valid_seen": self.games[:-1]})
        with self.assertRaisesRegex(ValueError, "checksum changed"):
            evaluation_catalogue(self.catalogue, old_digest, data_root=self.data)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            evaluation_catalogue(self.catalogue, None, data_root=self.data)

    def test_training_mixed_and_unknown_splits_are_rejected(self):
        for value in (
            {"train": self.games}, {"valid_train": self.games},
            {"valid_seen": self.games, "valid_unseen": []}, {"valid_seen": []},
        ):
            with self.subTest(value=value):
                self.freeze(value)
                with self.assertRaises(ValueError):
                    self.load()

    def test_duplicate_instances_and_goal_siblings_are_rejected(self):
        for games in (
            self.games + [self.games[0]],
            self.games + [self.game("valid_seen", "new-0", "different-trial")],
        ):
            self.freeze({"valid_seen": games})
            with self.assertRaisesRegex(ValueError, "repeats"):
                self.load()

    def test_legacy_goal_overlap_is_rejected_even_for_a_new_instance(self):
        self.freeze({"valid_seen": [self.game("valid_seen", "legacy-23")]})
        with self.assertRaisesRegex(ValueError, "legacy500"):
            self.load()

    def test_missing_mislabelled_and_escaping_games_are_rejected(self):
        for game in (
            "json_2.1.1/valid_seen/missing/trial/game.tw-pddl",
            self.game("valid_unseen", "different-split"),
            "../valid_seen/new/trial/game.tw-pddl",
        ):
            self.freeze({"valid_seen": [game]})
            with self.assertRaises(ValueError):
                self.load()
        target = self.data / self.games[0]
        outside = self.root / "outside.tw-pddl"
        outside.write_text("not a validation game")
        target.unlink()
        target.symlink_to(outside)
        self.freeze({"valid_seen": [self.games[0]]})
        with self.assertRaisesRegex(ValueError, "outside"):
            self.load()

    def test_native_runtime_and_task_definition_use_the_exact_overlay(self):
        with (
            patch("sao_bridge.alfworld_train_data.ALFWORLD_DATA", self.data),
            patch("sao_bridge.alfworld_runtime.ALFWORLD_DATA", self.data),
        ):
            runtime = prepare_agentbench_runtime(
                self.root / "runtime", catalogues={EVAL_SPLIT: self.catalogue},
            )
            linked = runtime / f"data/alfworld/{EVAL_SPLIT}.json"
            self.assertEqual(linked.resolve(), self.catalogue.resolve())
            self.assertEqual(read_json(linked), {"valid_seen": self.games})
            definition = task_definition(
                "heldout", [0, 2], 2, {"benchmark": "alfworld", "actions": []},
                runtime, split=EVAL_SPLIT, catalogue=self.catalogue,
                catalogue_sha256=self.digest,
            )
            self.assertEqual(definition["heldout"]["parameters"]["split"], EVAL_SPLIT)
            with self.assertRaises(ValueError):
                task_definition(
                    "heldout", [3], 2, {"benchmark": "alfworld", "actions": []},
                    runtime, split=EVAL_SPLIT, catalogue=self.catalogue,
                    catalogue_sha256=self.digest,
                )


if __name__ == "__main__":
    unittest.main()
