from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.common import read_json, write_json
from sao_bridge.phase_data import solver_dataset
from sao_bridge.replay import recycle_pool, recycled_interactions
from sao_bridge.train import skip_actor_updates


def call(prompt, response):
    return {"prompt_ids": prompt, "response_ids": response,
            "response_logprobs": [-0.1 * (i + 1) for i in range(len(response))],
            "temperature": 1.0, "stop_reason": "stop"}


class RecycleTests(unittest.TestCase):
    def write_eval(self, root, name, source, rewards):
        write_json(root / f"{name}.json", {
            "harness_source": source,
            "results": [{"id": f"q{i}", "reward": r,
                         "calls": [call([1, 2], [3, 4]), call([1, 2, 3, 4, 9], [5])]}
                        for i, r in enumerate(rewards)],
        })

    def test_pool_order_is_outcome_independent_and_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_eval(root, "a", "source-a", [1.0, 0.0])
            self.write_eval(root, "b", "source-b", [0.0, 1.0])
            pool = recycle_pool(root)
            self.assertEqual(len(pool), 4)
            self.assertEqual([e["id"] for e in pool], sorted(e["id"] for e in pool))
            flipped = root / "flipped"
            flipped.mkdir()
            self.write_eval(flipped, "a", "source-a", [0.0, 1.0])
            self.write_eval(flipped, "b", "source-b", [1.0, 0.0])
            self.assertEqual([e["id"] for e in recycle_pool(flipped)], [e["id"] for e in pool])

    def test_missing_token_receipts_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "a.json", {"harness_source": "s",
                                         "results": [{"id": "q", "reward": 1.0, "calls": []}]})
            with self.assertRaisesRegex(ValueError, "token receipts"):
                recycle_pool(root)

    def test_fresh_phases_have_distinct_recycle_namespaces(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_eval(root, "a", "unchanged-harness", [1.0, 0.0])
            old = {row["id"] for row in recycle_pool(root, namespace="cycle1/h1")}
            new = {row["id"] for row in recycle_pool(root, namespace="cycle2/h1")}
            self.assertFalse(old & new)
            self.assertEqual(old, {row["id"] for row in recycle_pool(root, namespace="cycle1/h1")})

    def test_turns_become_native_sequences_with_behavior_logprobs(self):
        episode = {"id": "e", "reward": 1.0,
                   "calls": [call([1, 2], [3, 4]), call([1, 2, 3, 4, 9], [5])]}
        result = recycled_interactions(episode, None)
        self.assertEqual(len(result), 2)
        first, second = (v.to_tensor_dict() for v in result.values())
        self.assertEqual(first["loss_mask"].tolist(), [[0, 0, 1, 1]])
        self.assertEqual(first["versions"].tolist(), [[-1, -1, 0, 0]])
        torch.testing.assert_close(first["logprobs"], torch.tensor([[0., 0., -0.1, -0.2]]))
        self.assertEqual(second["loss_mask"].tolist(), [[0, 0, 0, 0, 0, 1]])
        self.assertEqual(float(second["rewards"]), 1.0)

    def test_every_batch_gets_the_same_recycled_share(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "train.json", [{"id": str(i)} for i in range(10)])
            rows = solver_dataset(root / "train.json", root / "out.json", 4, 2,
                                  recycled_ids=[f"r{i}" for i in range(8)])
            self.assertEqual([len(r["recycled_ids"]) for r in rows], [2, 2, 2, 2])
            self.assertEqual(rows[0]["id"], "2")
            plain = solver_dataset(root / "train.json", root / "plain.json", 4)
            self.assertNotIn("recycled_ids", plain[0])
            self.assertEqual(read_json(root / "out.json")[1]["recycled_ids"], ["r1", "r5"])

    def test_warmup_rows_precede_fresh_rows_with_identical_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "train.json", [{"id": str(i), "question": "q", "answer": "1"}
                                             for i in range(6)])
            rows = solver_dataset(root / "train.json", root / "out.json", 4,
                                  warmup_ids=[f"w{i}" for i in range(6)], warmup_rows=2)
            self.assertEqual([r["warmup"] for r in rows], [True, True, False, False, False, False])
            self.assertEqual({frozenset(r) for r in rows}, {frozenset(rows[0])})
            self.assertEqual(rows[0]["question"], "")
            self.assertEqual(rows[0]["recycled_ids"], ["w0", "w2", "w4"])
            self.assertEqual(rows[2]["id"], "0")
            with self.assertRaises(ValueError):
                solver_dataset(root / "train.json", root / "x.json", 4, warmup_ids=["w"], warmup_rows=2)

    def test_critic_warmup_skips_only_the_first_actor_updates(self):
        class Actor:
            def __init__(self):
                self.updates, self.schedules = [], 0
            def ppo_update(self, batch):
                self.updates.append(batch)
            def step_lr_scheduler(self):
                self.schedules += 1
        actor = Actor()
        original = actor.ppo_update
        state = skip_actor_updates(actor, 2)
        for step in range(5):
            actor.ppo_update(step)
            actor.step_lr_scheduler()
        self.assertEqual(state["skipped"], 2)
        self.assertEqual(original.__self__.updates, [2, 3, 4])
        self.assertEqual(original.__self__.schedules, 3)


if __name__ == "__main__":
    unittest.main()
