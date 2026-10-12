import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.rewrite_data import (
    EditExample, assemble, create_app, grade_output, parse_document, teacher_token, validate_example,
)
from sao_bridge.resources import require_research_node


def example(split="train"):
    return {
        "id": split, "family": split + "_family", "split": split, "format": "json",
        "instruction": "Set the timeout to 5 and preserve the other settings.",
        "source": '{"timeout": 2, "retry": true}',
        "target": '{"timeout": 5, "retry": true}', "changes": {"timeout": 5},
    }


class RewriteDataTests(unittest.TestCase):
    def test_private_token_can_resume_but_is_never_silently_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private/access.token"
            token = teacher_token(path)
            self.assertGreaterEqual(len(token), 32)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(teacher_token(path, resume=True), token)
            with self.assertRaises(FileExistsError):
                teacher_token(path)
            path.chmod(0o644)
            with self.assertRaises(PermissionError):
                teacher_token(path, resume=True)
            self.assertEqual(path.read_text(), token)
            path.chmod(0o600)
            link = Path(directory) / "link.token"
            link.symlink_to(path)
            with self.assertRaises(OSError):
                teacher_token(link, resume=True)
            path.write_text("too-short")
            with self.assertRaisesRegex(ValueError, "invalid"):
                teacher_token(path, resume=True)

    def test_compute_host_authorization(self):
        for host in ("dev-NDH200-025", "dev-NDH200-026"):
            with patch("sao_bridge.resources.os.uname", return_value=SimpleNamespace(nodename=host)):
                require_research_node()
        with patch("sao_bridge.resources.os.uname", return_value=SimpleNamespace(nodename="dev-NDH200-022")):
            with self.assertRaisesRegex(RuntimeError, "only on node3/node4"):
                require_research_node()

    def test_strict_target_and_instruction_score(self):
        row = example()
        validate_example(EditExample(**row))
        self.assertTrue(grade_output(row, row["target"])["instruction_following"])
        self.assertFalse(grade_output(row, row["source"])["instruction_following"])
        self.assertFalse(grade_output(row, '{"timeout": true, "retry": true}')["legal"])
        self.assertFalse(grade_output(row, '```json\n' + row["target"] + '\n```')["legal"])
        with self.assertRaises(ValueError):
            parse_document('{"a": 1, "a": 2}', "json")

    def test_python_structure_and_placeholders_preserved(self):
        row = example()
        row.update(
            format="python",
            source='from example import Harness\nclass Task(Harness):\n    TEXT = "{task}"\n    LIMIT = 2\n',
            target='from example import Harness\nclass Task(Harness):\n    TEXT = "Read: {task}"\n    LIMIT = 2\n',
            changes={"TEXT": "Read: {task}"},
        )
        validate_example(EditExample(**row))
        row["target"] = row["target"].replace("{task}", "{input}")
        row["changes"] = {"TEXT": "Read: {input}"}
        with self.assertRaises(ValueError):
            validate_example(EditExample(**row))

    def test_authenticated_immutable_teacher_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "queue"
            client = TestClient(create_app(root, "s" * 40))
            self.assertEqual(client.get("/health").status_code, 401)
            headers = {"Authorization": "Bearer " + "s" * 40}
            request = {"shard": "test", "instruction": "Author examples.",
                       "counts": {"train": 1, "dev": 1, "test": 1}}
            receipt = client.post("/jobs", json=request, headers=headers).json()
            response = {"request_sha256": receipt["request_sha256"],
                        "teacher": "test fixture, not a real model",
                        "authorship": "active_agent_authored",
                        "examples": [example(split) for split in ("train", "dev", "test")]}
            for row in response["examples"]:
                row["instruction"] += " Case " + row["split"] + "."
            self.assertEqual(client.post("/jobs/test/response", json=response, headers=headers).status_code, 200)
            self.assertEqual(client.post("/jobs/test/response", json=response, headers=headers).status_code, 409)
            manifest = assemble(root, Path(directory) / "dataset")
            self.assertEqual(manifest["counts"], request["counts"])

    def test_cross_split_family_leak_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "queue"
            client = TestClient(create_app(root, "s" * 40))
            headers = {"Authorization": "Bearer " + "s" * 40}
            request = {"shard": "test", "instruction": "Author examples.",
                       "counts": {"train": 1, "dev": 1, "test": 1}}
            receipt = client.post("/jobs", json=request, headers=headers).json()
            rows = [example(split) for split in ("train", "dev", "test")]
            for row in rows:
                row["family"] = "leaky"
                row["instruction"] += " Case " + row["split"] + "."
            response = {"request_sha256": receipt["request_sha256"], "teacher": "fixture",
                        "authorship": "active_agent_authored", "examples": rows}
            self.assertEqual(client.post("/jobs/test/response", json=response, headers=headers).status_code, 200)
            with self.assertRaisesRegex(ValueError, "family leaked"):
                assemble(root, Path(directory) / "dataset")

    def test_single_split_shards_and_invalid_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            client = TestClient(create_app(Path(directory) / "queue", "s" * 40))
            headers = {"Authorization": "Bearer " + "s" * 40}
            request = {"shard": "train_only", "instruction": "Author training examples.",
                       "counts": {"train": 1, "dev": 0, "test": 0}}
            receipt = client.post("/jobs", json=request, headers=headers)
            self.assertEqual(receipt.status_code, 200)
            response = {
                "request_sha256": receipt.json()["request_sha256"], "teacher": "fixture",
                "authorship": "active_agent_authored", "examples": [example()],
            }
            self.assertEqual(client.post("/jobs/train_only/response", json=response,
                                         headers=headers).status_code, 200)
            pending = {**request, "shard": "pending"}
            self.assertEqual(client.post("/jobs", json=pending, headers=headers).status_code, 200)
            root = Path(directory) / "queue"
            with self.assertRaisesRegex(ValueError, "completed"):
                assemble(root, Path(directory) / "all")
            packed = assemble(root, Path(directory) / "selected", ["train_only"])
            self.assertEqual(packed["selected_shards"], ["train_only"])
            self.assertEqual(packed["counts"], request["counts"])
            for names in ([], ["train_only", "train_only"], ["../train_only"], ["pending"]):
                with self.assertRaises(ValueError):
                    assemble(root, Path(directory) / "invalid_selection", names)
            for counts in ({"train": 0, "dev": 0, "test": 0},
                           {"train": 2, "dev": -1, "test": 0},
                           {"train": 1, "dev": 0}):
                invalid = {**request, "shard": "invalid", "counts": counts}
                self.assertEqual(client.post("/jobs", json=invalid, headers=headers).status_code, 400)


if __name__ == "__main__":
    unittest.main()
