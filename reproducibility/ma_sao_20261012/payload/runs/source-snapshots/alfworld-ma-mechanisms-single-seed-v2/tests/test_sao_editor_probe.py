import asyncio
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import httpx

from sao_bridge.common import file_sha, read_json, write_json
from sao_bridge.editor_probe import (
    choose_candidate, load_rows, proposal_summary, require_complete_output,
    reuse_completed_proposals, run_probe, validate_origin,
)
from sao_bridge.evaluation import Sampler
from sao_bridge.experiment import propose
from sao_bridge.launch import Launcher
from sao_bridge.retool import INITIAL_SOURCE


CHANGED = INITIAL_SOURCE.replace("MAX_TURNS = 4", "MAX_TURNS = 3")


def result(rows, reward, samples=2):
    return {"accuracy": reward, "question_count": len(rows), "samples_per_question": samples,
            "metric": f"mean@{samples}",
            "results": [{"id": row["id"], "reward": reward} for row in rows for _ in range(samples)]}


class EditorProbeTests(unittest.TestCase):
    def test_semantic_diversity_ignores_formatting_and_invalid_slots(self):
        proposals = [
            {"valid": True, "source": INITIAL_SOURCE.replace('"', "'")},
            {"valid": True, "source": CHANGED},
            {"valid": True, "source": CHANGED.replace('"', "'")},
            {"valid": False, "finish_reason": "length"},
        ]
        summary = proposal_summary(proposals, INITIAL_SOURCE)
        self.assertEqual(summary, dict(slots=4, valid=3, distinct_valid_configurations=2,
                                       modified_slots=2, distinct_modified_configurations=1,
                                       length_limited_slots=1))
        rows = [{"id": str(i)} for i in range(8)]
        baseline = result(rows, 0)
        chosen = choose_candidate(proposals, [result(rows, x) for x in (1, 0.5, 1, 1)],
                                  baseline, INITIAL_SOURCE, seed=42, total_slots=16)
        self.assertEqual(chosen["best"]["slot"], 1)
        self.assertEqual(chosen["best"]["selection_accuracy"], 0.5)
        self.assertEqual(len(chosen["candidates"]), 1)

    def test_no_modified_candidate_is_not_a_successful_rewrite(self):
        rows = [{"id": str(i)} for i in range(8)]
        proposals = [{"valid": False}, {"valid": True, "source": INITIAL_SOURCE}]
        chosen = choose_candidate(proposals, [result(rows, 1)] * 2, result(rows, 0),
                                  INITIAL_SOURCE, seed=42, total_slots=8)
        self.assertIsNone(chosen["best"])
        self.assertFalse(chosen["any_passes_original_gate"])

    def test_thinking_controls_and_reasoning_are_separate_from_final_code(self):
        async def run(options, tokens=768):
            requests = []

            def handle(request):
                requests.append(json.loads(request.content))
                return httpx.Response(200, json={
                    "choices": [{"message": {"content": CHANGED, "reasoning": "analysis"},
                                 "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 100},
                })

            sampler = Sampler("http://test/v1", chat_template_kwargs=options)
            await sampler.client.aclose()
            sampler.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
            with patch("sao_bridge.experiment.Sampler", return_value=sampler):
                proposals = await propose("http://test/v1", INITIAL_SOURCE, 1, tokens, 1142,
                                          chat_template_kwargs=options)
            self.assertTrue(proposals[0]["valid"])
            self.assertEqual(proposals[0]["reasoning_characters"], 8)
            return requests[0]

        self.assertNotIn("chat_template_kwargs", asyncio.run(run(None)))
        for thinking in (False, True):
            request = asyncio.run(run({"enable_thinking": thinking}))
            self.assertEqual(request["chat_template_kwargs"], {"enable_thinking": thinking})
            self.assertEqual(request["max_tokens"], 768)
        full_request = asyncio.run(run({"enable_thinking": True}, None))
        self.assertNotIn("max_tokens", full_request)
        self.assertNotIn("max_completion_tokens", full_request)

    def test_uncapped_context_retry_and_long_request_timeout(self):
        async def run():
            requests = []

            def handle(request):
                body = json.loads(request.content)
                requests.append((request.url.path, body))
                if len(requests) == 1:
                    return httpx.Response(400, text="maximum context length")
                if request.url.path == "/tokenize":
                    return httpx.Response(200, json={"max_model_len": 262144, "count": 200})
                return httpx.Response(200, json={
                    "choices": [{"message": {"content": CHANGED}, "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 100},
                })

            sampler = Sampler("http://test/v1", request_timeout=5400)
            self.assertEqual(sampler.client.timeout.read, 5400)
            await sampler.client.aclose()
            sampler.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
            try:
                await sampler([], None, None, 42)
                self.assertEqual(sampler.context_clipped, 1)
                self.assertNotIn("max_tokens", requests[0][1])
                self.assertEqual(requests[-1][1]["max_tokens"], 262144 - 200)
            finally:
                await sampler.close()

        asyncio.run(run())

    def test_new_model_server_flags_are_forwarded_without_changing_allocation(self):
        launcher = object.__new__(Launcher)
        launcher.module, launcher.ready, launcher.stop = Mock(), Mock(), Mock()
        with launcher.vllm("model", port=19803, gpus="0", name="probe",
                           server_args=["--reasoning-parser", "qwen3", "--language-model-only"]) as url:
            self.assertEqual(url, "http://127.0.0.1:19803/v1")
        args = launcher.module.call_args.args[1]
        self.assertEqual(args[-3:], ["--reasoning-parser", "qwen3", "--language-model-only"])
        self.assertEqual(launcher.module.call_args.kwargs["gpus"], "0")
        launcher.stop.assert_called_once()

    def fixture(self, root):
        data, frozen = root / "data", root / "frozen"
        dev, train = [{"id": "select"}, {"id": "report"}], [{"id": "trained"}, {"id": "tail"}]
        for split, rows in (("dev", dev), ("train", train)):
            write_json(data / f"{split}.json", rows)
        write_json(data / "manifest.json", {"splits": {
            split: {"sha256": file_sha(data / f"{split}.json")} for split in ("dev", "train")}})
        write_json(frozen / "h2_selected_report.json", result([dev[1], train[1]], 0))
        write_json(frozen / "w1_solver_data.json", [train[0]])
        write_json(frozen / "w1_recycle.json", [{"question_id": "reward_only"}])
        incumbent = root / "incumbent.py"
        incumbent.write_text(INITIAL_SOURCE)
        solver = str(root / "solver")
        write_json(frozen / "FROZEN.json", {"stages": [{"solver": solver, "harness": str(incumbent)}]})
        write_json(root / "provenance.json", {"repo": "test", "revision": "pinned"})
        return {
            "data": str(data), "frozen_run": str(frozen), "solver": solver,
            "incumbent": str(incumbent), "run": str(root / "probe"),
            "new_model_provenance": str(root / "provenance.json"),
            "select_questions": 1, "report_questions": 1, "report_train_tail": 1,
            "seed": 42, "proposal_seed": 1142, "proposals": 2, "editor_temperature": 0.5,
            "solver_tokens": 2048, "select_samples": 2, "report_samples": 8,
            "arms": [{"name": "editor", "model": str(root / "editor"), "tokens": 768, "thinking": False}],
        }

    def test_report_data_cannot_overlap_solver_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = self.fixture(Path(tmp))
            select, report = load_rows(spec)
            self.assertEqual((len(select), len(report)), (1, 2))
            write_json(Path(spec["frozen_run"]) / "w1_solver_data.json", [{"id": "tail"}])
            with self.assertRaisesRegex(ValueError, "overlaps"):
                load_rows(spec)

    def first_h1(self, spec):
        spec["first_h1"] = True
        spec["proposals"] = 1
        previous = Path(spec["frozen_run"])
        write_json(previous / "spec.json", {"model": spec["solver"]})
        (previous / "h0.py").write_text(INITIAL_SOURCE)
        spec["incumbent"] = str(previous / "h0.py")
        arm = spec["arms"][0]
        arm.update(thinking=True, tokens=None, max_model_len=262144, request_timeout=5400)
        write_json(Path(arm["model"]) / "config.json",
                   {"text_config": {"max_position_embeddings": 262144}})

    def test_first_h1_requires_one_naturally_finished_uncapped_proposal(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = self.fixture(Path(tmp))
            self.first_h1(spec)
            validate_origin(spec)
            with self.assertRaisesRegex(ValueError, "best-of-N"):
                validate_origin({**spec, "proposals": 8})
            with self.assertRaisesRegex(ValueError, "original SFT"):
                validate_origin({**spec, "solver": "different_weights"})
            spec["arms"][0]["tokens"] = 4096
            with self.assertRaisesRegex(ValueError, "no short token cap"):
                validate_origin(spec)
        require_complete_output([{"finish_reason": "stop"}])
        with self.assertRaisesRegex(ValueError, "truncated output"):
            require_complete_output([{"finish_reason": "length", "valid": True}])

    def test_runtime_recovery_reuses_only_unchanged_complete_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            previous, new = root / "old", root / "new"
            spec = {"run": str(previous), "proposals": 2, "seed": 42, "arms": [
                {"name": "control", "model": "four_b", "tokens": 768},
                {"name": "new_model", "model": "nine_b", "tokens": 768},
            ]}
            hashes = {"four_b": "a", "nine_b": "b"}
            write_json(previous / "spec.json", spec)
            write_json(previous / "STATUS.json", {"phase": "failed"})
            write_json(previous / "initialization.json", {"models": hashes})
            proposals = [{"id": i, "valid": True, "text": CHANGED, "source": CHANGED} for i in range(2)]
            write_json(previous / "control_proposals.json", proposals)
            resumed = {**spec, "run": str(new), "resume_from": str(previous),
                       "arms": [spec["arms"][0], {**spec["arms"][1], "server_args": ["--gdn-prefill-backend", "triton"]}]}
            self.assertEqual(reuse_completed_proposals(resumed, hashes, new), {"control": proposals})
            self.assertEqual(read_json(new / "control_proposals.json"), proposals)
            with self.assertRaisesRegex(ValueError, "digests"):
                reuse_completed_proposals(resumed, {**hashes, "four_b": "changed"}, new)
            resumed["arms"][0] = {**spec["arms"][0], "server_args": ["changed"]}
            with self.assertRaisesRegex(ValueError, "completed editor"):
                reuse_completed_proposals(resumed, hashes, new)

    def test_end_to_end_probe_uses_inference_only_and_persists_verified_result(self):
        class FakeLauncher:
            def __init__(self, root):
                pass

            @contextmanager
            def vllm(self, model, **kwargs):
                yield "http://test"

            def close(self):
                pass

        async def proposals(*args, **kwargs):
            return [{"id": i, "valid": True, "source": CHANGED, "finish_reason": "stop"}
                    for i in range(args[2])]

        async def evaluate(sampler, rows, source, *, tokens, samples, seed):
            return result(rows, int(source == CHANGED), samples)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = self.fixture(root)
            path = root / "spec.json"
            write_json(path, spec)
            with (patch("sao_bridge.editor_probe.require_research_node"),
                  patch("sao_bridge.editor_probe.stop_known"),
                  patch("sao_bridge.editor_probe.initial_model_digest", return_value="unchanged"),
                  patch("sao_bridge.editor_probe.Launcher", FakeLauncher),
                  patch("sao_bridge.editor_probe.propose", side_effect=proposals),
                  patch("sao_bridge.editor_probe.evaluate", side_effect=evaluate),
                  patch("sao_bridge.editor_probe.subprocess.check_output", return_value=""),
                  patch("sao_bridge.editor_probe.subprocess.run")):
                run_probe(path)
            complete = read_json(Path(spec["run"]) / "COMPLETE.json")
            self.assertEqual(complete["optimizer_updates"], 0)
            self.assertTrue(complete["models_unchanged"])
            self.assertFalse(complete["test_evaluated"])
            self.assertEqual(complete["arms"]["editor"]["report_comparison"]["candidate_minus_control"], 1)
            self.assertEqual(complete["physical_completed_samples"],
                             {"editor_proposals": 2, "solver_selection": 6, "solver_reporting": 32})

    def test_first_h1_runtime_uses_native_context_and_no_output_cap(self):
        class FakeLauncher:
            calls = []

            def __init__(self, root):
                pass

            @contextmanager
            def vllm(self, model, **kwargs):
                self.calls.append((model, kwargs))
                yield "http://test"

            def close(self):
                pass

        async def proposals(url, source, count, tokens, seed, **kwargs):
            self.assertEqual(count, 1)
            self.assertIsNone(tokens)
            self.assertTrue(kwargs["chat_template_kwargs"]["enable_thinking"])
            self.assertEqual(kwargs["request_timeout"], 5400)
            return [{"id": 0, "valid": True, "source": CHANGED, "finish_reason": "stop"}]

        async def evaluate(sampler, rows, source, *, tokens, samples, seed):
            self.assertEqual(tokens, 2048)
            return result(rows, int(source == CHANGED), samples)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = self.fixture(root)
            self.first_h1(spec)
            path = root / "spec.json"
            write_json(path, spec)
            with (patch("sao_bridge.editor_probe.require_research_node"),
                  patch("sao_bridge.editor_probe.stop_known"),
                  patch("sao_bridge.editor_probe.initial_model_digest", return_value="unchanged"),
                  patch("sao_bridge.editor_probe.Launcher", FakeLauncher),
                  patch("sao_bridge.editor_probe.propose", side_effect=proposals),
                  patch("sao_bridge.editor_probe.evaluate", side_effect=evaluate),
                  patch("sao_bridge.editor_probe.subprocess.check_output", return_value=""),
                  patch("sao_bridge.editor_probe.subprocess.run")):
                run_probe(path)
            self.assertEqual(FakeLauncher.calls[0][1]["max_model_len"], 262144)
            complete = read_json(Path(spec["run"]) / "COMPLETE.json")
            self.assertTrue(complete["first_h1"])
            self.assertEqual(complete["optimizer_updates"], 0)
            self.assertEqual(complete["physical_completed_samples"]["editor_proposals"], 1)


if __name__ == "__main__":
    unittest.main()
