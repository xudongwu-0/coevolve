import asyncio
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx

from sao_bridge.domain_api import get_domain
from sao_bridge.evaluation import create_app, evaluate, stable_id_seed_offset
from sao_bridge.experiment import propose
from sao_bridge.large_train import (
    configure, validate_capacity_audit, validate_training_rows,
)
from sao_bridge.phase_data import editor_dataset
from sao_bridge import retool


class DomainIntegrationTests(unittest.TestCase):
    def test_evaluation_seed_supports_semantic_ids(self):
        self.assertEqual(stable_id_seed_offset("00000001-row"), 1)
        self.assertEqual(
            stable_id_seed_offset("alfworld-1"),
            stable_id_seed_offset("alfworld-1"),
        )
        self.assertNotEqual(
            stable_id_seed_offset("alfworld-1"),
            stable_id_seed_offset("alfworld-2"),
        )

    def domain(self):
        def parse(source):
            if source not in {"h0", "h1"}:
                raise ValueError("invalid candidate")
            return {"source": source}

        async def episode(row, spec, generate, total_tokens, seed):
            return {
                "id": row["id"], "reward": float(spec["source"] == "h1"),
                "tokens": 3, "tool_calls": 1, "tool_errors": 0,
                "truncated": False, "boxed": True, "calls": [],
                "messages": [], "elapsed_seconds": 0,
            }

        return SimpleNamespace(harness_spec=parse, run_episode=episode,
                               EDITOR_SYSTEM="Domain hook editing contract")

    def test_registry_keeps_math_default_and_rejects_unknown_domains(self):
        self.assertIs(get_domain(), retool)
        with self.assertRaisesRegex(ValueError, "Unsupported domain"):
            get_domain("unregistered")

    def test_domain_evaluation_preserves_sample_ids_and_accounting(self):
        rows = [{"id": "00000001"}, {"id": "00000002"}]
        with patch("sao_bridge.evaluation.get_domain", return_value=self.domain()):
            result = asyncio.run(evaluate(None, rows, "h1", tokens=4096,
                                          samples=8, seed=42, domain="chess"))
        self.assertEqual(result["accuracy"], 1)
        self.assertEqual(result["tokens"], 48)
        self.assertEqual(result["formatted_fraction"], 1)
        self.assertEqual(result["domain"], "chess")
        self.assertEqual(len({(r["id"], r["sample_index"]) for r in result["results"]}), 16)
        self.assertEqual(result["results"][1]["rollout_seed"], 42 + 1 + 1009)

    def test_domain_evaluation_limits_whole_episode_concurrency(self):
        active = 0
        peak = 0

        async def episode(row, spec, generate, total_tokens, seed):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return {
                "id": row["id"], "reward": 1.0, "tokens": 1,
                "tool_calls": 1, "tool_errors": 0, "truncated": False,
                "boxed": True,
            }

        domain = self.domain()
        domain.run_episode = episode
        rows = [{"id": f"{index:08x}"} for index in range(6)]
        with patch("sao_bridge.evaluation.get_domain", return_value=domain):
            asyncio.run(evaluate(
                None,
                rows,
                "h1",
                episode_concurrency=2,
            ))
        self.assertEqual(peak, 2)

    def test_reward_service_dispatches_domain_and_preserves_invalid_penalty(self):
        async def run(root):
            sampler = SimpleNamespace(generated_tokens=0)
            with patch("sao_bridge.evaluation.get_domain", return_value=self.domain()):
                app = create_app(sampler, [{"id": "00000001"}], "h0", root, domain="searchqa")
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    good = await client.post("/score", json={"source": "h1", "sample_id": "good"})
                    bad = await client.post("/score", json={"source": "bad", "sample_id": "bad"})
            return good.json(), bad.json()

        with tempfile.TemporaryDirectory() as temporary:
            good, bad = asyncio.run(run(Path(temporary)))
            self.assertEqual(good["reward"], 1)
            self.assertEqual(bad["reward"], -1)
            self.assertFalse(bad["valid"])
            self.assertEqual(len(list(Path(temporary).glob("*.json"))), 4)

    def test_domain_proposal_uses_hook_contract_and_training_feedback(self):
        seen = []

        class FakeSampler:
            def __init__(self, *args, **kwargs):
                pass

            async def __call__(self, messages, tools, tokens, seed):
                seen.extend(messages)
                return {"choices": [{"message": {"content": "h1"}, "finish_reason": "stop"}],
                        "usage": {"completion_tokens": 10}}

            async def close(self):
                pass

        with (patch("sao_bridge.experiment.get_domain", return_value=self.domain()),
              patch("sao_bridge.experiment.Sampler", FakeSampler)):
            output = asyncio.run(propose(
                "http://test", "h0", 1, 2048, 42, domain="chess",
                feedback="Training trace: response omitted the required move tag."))
        self.assertTrue(output[0]["valid"])
        self.assertEqual(seen[0]["content"], "Domain hook editing contract")
        self.assertIn("executable hooks", seen[1]["content"])
        self.assertIn("Training trace:", seen[1]["content"])
        self.assertNotIn("six allowed literal values", seen[1]["content"])

    def test_runtime_broken_hook_is_an_invalid_editor_action(self):
        task = self.domain()
        original = task.run_episode

        async def broken_episode(row, spec, generate, total_tokens, seed):
            result = await original(row, spec, generate, total_tokens, seed)
            if spec["source"] == "h1":
                result.update(reward=0, harness_errors=[{"hook": "format_observation",
                                                         "error": "missing public key"}])
            return result

        task.run_episode = broken_episode

        async def run(root):
            with patch("sao_bridge.evaluation.get_domain", return_value=task):
                app = create_app(SimpleNamespace(generated_tokens=0),
                                 [{"id": "00000001"}], "h0", root, domain="chess")
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    response = await client.post("/score", json={"source": "h1", "sample_id": "bad-hook"})
            return response.json()

        with tempfile.TemporaryDirectory() as temporary:
            result = asyncio.run(run(Path(temporary)))
        self.assertFalse(result["valid"])
        self.assertEqual(result["reward"], -1)
        self.assertIn("Executable harness failed", result["error"])

    def test_editor_training_and_generation_share_domain_contract(self):
        with patch("sao_bridge.phase_data.get_domain", return_value=self.domain()):
            rows = editor_dataset("h0", 0.25, 4, 2, domain="chess",
                                  failure_feedback="training-failure-123")
        for row in rows:
            self.assertEqual(row["messages"][0]["content"], "Domain hook editing contract")
            self.assertIn("training-failure-123", row["messages"][1]["content"])
            self.assertIn("CandidateHarness", row["messages"][1]["content"])
            self.assertNotIn("development accuracy", row["messages"][1]["content"])
        legacy = editor_dataset(retool.INITIAL_SOURCE, 0.25, 1, 0)[0]
        self.assertEqual(legacy["messages"][0]["content"], retool.EDITOR_SYSTEM)
        self.assertTrue(legacy["messages"][1]["content"].endswith(
            "Return only the complete allowed CandidateEnv class module."))

    def test_native_training_context_and_retrieval_environment(self):
        spec = {
            "model": "/model", "role": "solver", "steps": 1, "batch_size": 2,
            "phase": "w1", "experiment": "domains", "seed": 42,
            "lr": 2e-6, "tokens": 4096, "context_tokens": 16384,
        }
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {
                "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                "SAO_SEARCHQA_RETRIEVAL_URL": "http://127.0.0.1:19631",
            }
        ):
            config = configure(spec, Path(temporary))
            self.assertEqual(config["gconfig"]["max_tokens"], 16384)
            self.assertEqual(config["vllm"]["max_model_len"], 16384)
            self.assertEqual(config["actor"]["mb_spec"]["max_tokens_per_mb"], 16384)
            limited = configure({**spec, "microbatch_tokens": 8192}, Path(temporary))
            self.assertEqual(limited["actor"]["mb_spec"]["max_tokens_per_mb"], 8192)
            self.assertEqual(limited["critic"]["mb_spec"]["max_tokens_per_mb"], 10240)
            self.assertEqual(config["actor"]["scheduling_spec"][0]["env_vars"][
                "SAO_SEARCHQA_RETRIEVAL_URL"], "http://127.0.0.1:19631")
            self.assertEqual(config["actor"]["scheduling_spec"][0]["env_vars"][
                "PYTORCH_ALLOC_CONF"], "expandable_segments:True")
            self.assertTrue(config["actor"]["use_decoupled_loss"])
            self.assertEqual(config["gconfig"]["n_samples"], 1)
            optimized = configure({
                **spec,
                "critic_updates_per_step": 1,
                "actor_gradient_checkpointing": False,
                "critic_gradient_checkpointing": False,
                "solver_trajectory_style": "concat",
            }, Path(temporary))
            self.assertFalse(optimized["actor"]["gradient_checkpointing"])
            self.assertFalse(optimized["critic"]["gradient_checkpointing"])
            actor_only = configure(
                {**spec, "critic_mode": "none"},
                Path(temporary),
            )
            self.assertIsNone(actor_only["critic"])
            with self.assertRaisesRegex(ValueError, "centralized critic"):
                configure({
                    **spec,
                    "critic_mode": "none",
                    "central_context": "/context.json",
                }, Path(temporary))
            with self.assertRaisesRegex(ValueError, "trajectory style"):
                configure({
                    **spec,
                    "solver_trajectory_style": "invalid",
                }, Path(temporary))
            with self.assertRaisesRegex(ValueError, "require concat"):
                configure({**spec, "terminal_episodes": True}, Path(temporary))
            with self.assertRaisesRegex(ValueError, "must be boolean"):
                configure({**spec, "terminal_episodes": 1}, Path(temporary))
            with self.assertRaisesRegex(ValueError, "leave room"):
                configure({**spec, "context_tokens": 4096}, Path(temporary))
            with self.assertRaisesRegex(ValueError, "data-parallel size"):
                configure({**spec, "batch_size": 1}, Path(temporary))
            with self.assertRaisesRegex(ValueError, "divisible"):
                configure(
                    {**spec, "batch_size": 4, "actor_gpus": 3},
                    Path(temporary),
                )
            audit = {
                "role": "solver", "microbatch_tokens": 16384,
                "required_microbatch_tokens": 16384,
            }
            validate_capacity_audit(spec, audit)
            with self.assertRaisesRegex(ValueError, "longest possible trajectory"):
                validate_capacity_audit(
                    {**spec, "microbatch_tokens": 12000},
                    {**audit, "microbatch_tokens": 12000},
                )
            validate_training_rows(
                spec,
                [{"id": "one"}, {"id": "two"}],
            )
            with self.assertRaisesRegex(ValueError, "distinct rows"):
                validate_training_rows(spec, [{"id": "one"}])
            with self.assertRaisesRegex(ValueError, "duplicate training IDs"):
                validate_training_rows(
                    spec,
                    [{"id": "one"}, {"id": "one"}],
                )
            with self.assertRaisesRegex(ValueError, "fresh complete"):
                validate_training_rows(
                    {**spec, "terminal_episodes": True},
                    [{"id": "one", "recycled_ids": ["past"]}, {"id": "two"}],
                )


if __name__ == "__main__":
    unittest.main()
