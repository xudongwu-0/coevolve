import asyncio
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.common import read_json
from sao_bridge.evaluation import Sampler, create_app
from sao_bridge.retool import INITIAL_SOURCE


class FixedRewardBudgetTests(unittest.TestCase):
    def test_every_action_gets_a_fresh_paired_evaluation_even_if_duplicate_or_invalid(self):
        calls = []
        async def fake_evaluate(sampler, rows, source, *, tokens, samples=1, seed=42, domain="math"):
            self.assertEqual(domain, "math")
            calls.append((source, seed))
            return {"accuracy": seed % 11 / 10, "results": [], "question_count": len(rows),
                    "samples_per_question": samples, "metric": f"mean@{samples}"}

        async def run(root):
            sampler = type("S", (), {"generated_tokens": 0})()
            with patch("sao_bridge.evaluation.evaluate", side_effect=fake_evaluate):
                app = create_app(sampler, [{"id": "q"}], INITIAL_SOURCE, root)
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    first = (await client.post("/score", json={"source": INITIAL_SOURCE, "sample_id": "a"})).json()
                    second = (await client.post("/score", json={"source": INITIAL_SOURCE, "sample_id": "b"})).json()
                    invalid = (await client.post("/score", json={"source": "not python", "sample_id": "c"})).json()
            return first, second, invalid

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second, invalid = asyncio.run(run(root))
            self.assertEqual(len(calls), 6)  # candidate + reference for all 3 actions
            seeds = [x[1] for x in calls]
            self.assertEqual(seeds[0::2], seeds[1::2])
            self.assertEqual(len(set(seeds[0::2])), 3)
            self.assertEqual(len(list(root.glob("*.json"))), 6)
            self.assertEqual((first["request_index"], second["request_index"]), (0, 1))
            self.assertFalse(invalid["valid"])
            self.assertEqual(invalid["reward"], -1)
            records = [read_json(p) for p in root.glob("*.json")]
            self.assertEqual({r["pair_role"] for r in records}, {"candidate", "reference"})


if __name__ == "__main__":
    unittest.main()
