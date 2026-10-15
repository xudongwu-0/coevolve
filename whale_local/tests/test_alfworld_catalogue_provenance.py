import asyncio
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

from sao_bridge import alfworld_domain as domain
from sao_bridge.alfworld_train_data import task_count
from sao_bridge.common import file_sha


class CatalogueProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.catalogue = self.root / "test_set.json"
        self.catalogue.write_text(json.dumps({"train": [
            f"json_2.1.1/train/goal-{index}/trial/game.tw-pddl"
            for index in range(500)
        ]}))
        self.row = {
            "id": "alfworld-0", "task_id": 0, "task_name": "external",
            "task_split": "test_set", "task_catalogue": str(self.catalogue),
            "task_catalogue_sha256": file_sha(self.catalogue),
        }
        self.root_patch = patch.object(domain, "ALFWORLD_DATA", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def test_verified_official_provenance_is_not_an_override(self):
        self.assertEqual(domain._row_task_count(self.row), 500)
        with self.assertRaisesRegex(ValueError, "cannot be overridden"):
            task_count("test_set", self.row["task_catalogue_sha256"],
                       catalogue_path=self.catalogue)

    def test_unpinned_official_rows_keep_existing_behavior(self):
        self.assertEqual(domain._row_task_count({"task_id": 0}), 500)

    def test_same_bytes_at_another_path_are_rejected(self):
        alternate = self.root / "alternate.json"
        alternate.write_bytes(self.catalogue.read_bytes())
        with self.assertRaisesRegex(ValueError, "must identify the official"):
            domain._row_task_count({**self.row, "task_catalogue": str(alternate)})

    def test_changed_official_bytes_are_rejected(self):
        self.catalogue.write_text("{}")
        with self.assertRaisesRegex(ValueError, "checksum changed"):
            domain._row_task_count(self.row)

    def test_incomplete_provenance_is_rejected(self):
        for absent in ("task_catalogue", "task_catalogue_sha256"):
            with self.subTest(absent=absent):
                row = {key: value for key, value in self.row.items() if key != absent}
                with self.assertRaisesRegex(ValueError, "requires both"):
                    domain._row_task_count(row)

    def test_malformed_official_catalogue_is_rejected(self):
        self.catalogue.write_text(json.dumps({"train": ["invalid"] * 500}))
        row = {**self.row, "task_catalogue_sha256": file_sha(self.catalogue)}
        with self.assertRaisesRegex(ValueError, "duplicate games"):
            domain._row_task_count(row)

    def test_custom_split_keeps_catalogue_validation(self):
        row = {**self.row, "task_split": "sao_train_v1"}
        with patch.object(domain, "task_count", return_value=17) as count:
            self.assertEqual(domain._row_task_count(row), 17)
            count.assert_called_once_with(
                "sao_train_v1", row["task_catalogue_sha256"],
                catalogue_path=row["task_catalogue"],
            )

    def test_actual_episode_accepts_verified_producer_fields(self):
        client = AsyncMock()
        client.__aenter__.return_value = client
        response = Mock()
        response.headers = {"session_id": "session-1"}
        response.json.return_value = {"status": "COMPLETED", "reward": 1.0, "messages": []}
        client.post.return_value = response
        with (
            patch.dict(os.environ, {"SAO_ALFWORLD_CONTROLLER_URL": "http://127.0.0.1:28000/api"}),
            patch.object(domain.httpx, "AsyncClient", return_value=client),
        ):
            result = asyncio.run(domain.run_episode(self.row, {}, AsyncMock()))
        self.assertEqual(result["reward"], 1.0)
        client.post.assert_awaited_once_with(
            "http://127.0.0.1:28000/api/start_sample",
            json={"name": "external", "index": 0},
        )


if __name__ == "__main__":
    unittest.main()
