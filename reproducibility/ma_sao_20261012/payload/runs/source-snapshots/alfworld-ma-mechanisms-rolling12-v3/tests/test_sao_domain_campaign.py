"""CPU-only controller tests. Scratch directories stay inside this repository."""
import copy
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import json
from io import StringIO
import os
from pathlib import Path
import shutil
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import uuid

from sao_bridge.common import file_sha, read_json
from sao_bridge.domain_campaign import (
    Campaign, Engine, Monitor, audit_identity, baseline_server_layout, choose_initializer,
    completed_optimizer, component_identity,
    digest, evaluation_identity, execution_completion, extension_decision, immutable,
    implementation_identity, initial_state, main,
    inference_ledger, ledger, phase_plan, recycling_namespace, recycling_partition, requested_domains,
    retrieval_health_url, reward_window,
    save, select_harness, solver_server_layout, training_feedback, training_implementation_identity,
    training_window, update_plateau, validate_protocol,
    verify_report, watch, watcher_snapshot, worker_claim,
)


ROOT = Path(__file__).resolve().parents[1]


def protocol():
    return read_json(ROOT / "examples/searchqa_chess_campaign_20260929.json")


def report(score, split="dev", samples=8, questions=8):
    results = []
    for question in range(questions):
        for sample in range(samples):
            results.append({
                "id": f"{question:064x}", "reward": int(sample < round(score * samples)),
                "sample_index": sample, "rollout_seed": 42 + question + 1009 * sample,
                "failure_trace": [{"sampled_action": "actual model action", "error": "illegal move"}],
            })
    return {"accuracy": sum(row["reward"] for row in results) / len(results),
            "question_count": questions, "samples_per_question": samples,
            "metric": f"mean@{samples}", "tokens": 10 * len(results), "results": results,
            "cache_identity": {"split": split, "purpose": "test_report" if split == "test" else "dev_report"}}


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / "tests" / f".domain-campaign-{os.getpid()}-{uuid.uuid4().hex}"
        self.root.mkdir()

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_protocol_fixed_budgets_and_nodes(self):
        p = protocol()
        validate_protocol(p)
        self.assertEqual(p["domain_order"], ["chess", "searchqa"])
        self.assertIn("NOT paid WHALE", p["domains"]["searchqa"]["metric"])
        self.assertIn("BM25", p["domains"]["searchqa"]["adaptations"]["retrieval"])
        self.assertIn("e5", p["domains"]["searchqa"]["adaptations"]["retrieval"])
        self.assertFalse(p["domains"]["searchqa"]["adaptations"]["exact_paper_reproduction"])
        for section, key, value in (
            ("common", "context_tokens", 8192), ("common", "solver_tokens", 8192),
            ("algorithm", "interaction_credit", 1.0), ("algorithm", "central_critic", True),
            ("stopping", "minimum_blocks", 1),
        ):
            altered = copy.deepcopy(p)
            altered[section][key] = value
            with self.assertRaises(ValueError):
                validate_protocol(altered)

    def test_baselines_freeze_evaluation_but_not_unfinished_training_sources(self):
        baseline = implementation_identity("searchqa")
        self.assertIn("searchqa_domain.py", baseline)
        self.assertIn("searchqa_retrieval.py", baseline)
        self.assertIn("program_harness.py", baseline)
        self.assertIn("evaluation.py", baseline)
        self.assertNotIn("retool.py", baseline)
        self.assertIn("retool.py:ContextExhausted", baseline)
        self.assertNotIn("chess_domain.py", baseline)
        self.assertNotIn("large_train.py", baseline)
        self.assertNotIn("domain_campaign.py", baseline)
        training = implementation_identity()
        self.assertIn("large_train.py", training)
        self.assertIn("domain_campaign.py", training)

    def test_unrelated_math_edits_do_not_change_frozen_evaluation_components(self):
        path = self.root / "math_helper.py"
        path.write_text("class ContextExhausted(RuntimeError):\n    pass\n\ndef math_solver():\n    return 1\n")
        before = component_identity(path, {"ContextExhausted"})
        path.write_text("# unrelated math edit\nclass ContextExhausted(RuntimeError):\n    pass\n\ndef math_solver():\n    return 2\n")
        self.assertEqual(before, component_identity(path, {"ContextExhausted"}))
        path.write_text("class ContextExhausted(ValueError):\n    pass\n")
        self.assertNotEqual(before, component_identity(path, {"ContextExhausted"}))

    def test_ready_searchqa_baseline_can_skip_unavailable_chess(self):
        p = protocol()
        self.assertEqual(requested_domains(p, "baseline", "searchqa"), ["searchqa"])
        self.assertEqual(requested_domains(p, "baseline", "chess"), ["chess"])
        self.assertEqual(requested_domains(p, "baseline"), ["chess", "searchqa"])
        self.assertEqual(requested_domains(p, "worker"), ["chess", "searchqa"])
        self.assertEqual(requested_domains(p, "worker", "searchqa"), ["searchqa"])
        with self.assertRaisesRegex(ValueError, "--domain"):
            requested_domains(p, "status", "searchqa")

    def test_training_freeze_does_not_depend_on_pending_other_domain_code(self):
        sources = {"domain_campaign.py": "controller", "large_train.py": "native",
                   "chess_domain.py": "chess1", "chess_prepare.py": "chess-data",
                   "searchqa_domain.py": "searchqa", "searchqa_retrieval.py": "retrieval"}
        with patch("sao_bridge.domain_campaign.implementation_identity", return_value=sources):
            before = training_implementation_identity("searchqa")
            self.assertNotIn("chess_domain.py", before)
            self.assertNotIn("chess_prepare.py", before)
            self.assertIn("large_train.py", before)
            self.assertIn("domain_campaign.py", before)
            sources["chess_domain.py"] = "chess2"
            self.assertEqual(before, training_implementation_identity("searchqa"))
            self.assertNotIn("searchqa_retrieval.py", training_implementation_identity("chess"))

    def test_scoped_completion_accumulates_without_claiming_whole_campaign(self):
        p = protocol()
        save(self.root / "searchqa/BASELINE_READY.json", {})
        baseline = execution_completion(p, self.root, "baseline", ["searchqa"])
        self.assertEqual(baseline["completed_domains"], [])
        self.assertEqual(baseline["baseline_domains"], ["searchqa"])
        save(self.root / "searchqa/DOMAIN_COMPLETE.json", {})
        first = execution_completion(p, self.root, "worker", ["searchqa"])
        self.assertEqual(first["status"], "domains_completed")
        self.assertEqual(first["completed_domains"], ["searchqa"])
        self.assertFalse(first["campaign_complete"])
        save(self.root / "chess/DOMAIN_COMPLETE.json", {})
        second = execution_completion(p, self.root, "worker", ["chess"])
        self.assertEqual(second["status"], "completed")
        self.assertEqual(second["requested_domains"], ["chess"])
        self.assertEqual(second["completed_domains"], ["chess", "searchqa"])
        self.assertTrue(second["campaign_complete"])
        self.assertEqual(p["domain_order"], ["chess", "searchqa"])

    def test_worker_domain_cli_executes_only_searchqa_and_publishes_partial_status(self):
        campaign = self.campaign()
        p = campaign.p
        path = self.root / "protocol.json"
        save(path, p)
        fake = MagicMock()
        fake.run_domain.side_effect = lambda domain: save(campaign.root / domain / "DOMAIN_COMPLETE.json", {})
        monitor = MagicMock()

        @contextmanager
        def claim(*_):
            yield

        with patch("sao_bridge.domain_campaign.sys.argv",
                   ["domain_campaign", "worker", "--protocol", str(path), "--domain", "searchqa", "--resume"]), \
                patch("sao_bridge.domain_campaign.os.uname", return_value=SimpleNamespace(nodename=campaign.host)), \
                patch("sao_bridge.domain_campaign.os.getuid", return_value=1008), \
                patch("sao_bridge.domain_campaign.signal.signal"), \
                patch("sao_bridge.domain_campaign.worker_claim", claim), \
                patch("sao_bridge.domain_campaign.Campaign", return_value=fake), \
                patch("sao_bridge.domain_campaign.Monitor") as monitor_class:
            monitor_class.return_value.__enter__.return_value = monitor
            main()
        fake.run_domain.assert_called_once_with("searchqa")
        fake.baseline.assert_not_called()
        args, kwargs = monitor.update.call_args
        self.assertEqual(args, ("domains_completed",))
        self.assertEqual(kwargs["completed_domains"], ["searchqa"])
        self.assertFalse(kwargs["campaign_complete"])
        execution, = campaign.root.glob("WORKER_EXECUTION_*.json")
        self.assertEqual(read_json(execution)["protocol_domain_order"], ["chess", "searchqa"])
        self.assertEqual(read_json(execution)["requested_domains"], ["searchqa"])

    def test_gpu_layouts_default_h_to_four_and_standalone_to_eight(self):
        frozen = solver_server_layout()
        full = solver_server_layout(full_node=True)
        self.assertEqual(frozen["gpu_ids"], [4, 5, 6, 7])
        self.assertEqual([row["port"] for row in frozen["replicas"]], [18801, 18811, 18821, 18831])
        self.assertEqual(full["gpu_ids"], list(range(8)))
        self.assertEqual(full["data_parallel"], 8)
        self.assertEqual(full["replicas"][-1]["health_url"], "http://127.0.0.1:18871/health")
        self.assertEqual(frozen["context_tokens"], full["context_tokens"])
        self.assertEqual(baseline_server_layout(self.root), full)

    def test_started_baseline_keeps_legacy_four_gpu_layout_on_resume(self):
        save(self.root / "BASELINE_EXECUTION_1.json", {"source_sha256": {}})
        self.assertEqual(baseline_server_layout(self.root), solver_server_layout())
        save(self.root / "BASELINE_EXECUTION_2.json",
             {"solver_server_layout": solver_server_layout()})
        self.assertEqual(baseline_server_layout(self.root), solver_server_layout())
        save(self.root / "BASELINE_EXECUTION_3.json",
             {"solver_server_layout": solver_server_layout(full_node=True)})
        with self.assertRaisesRegex(ValueError, "layout changed"):
            baseline_server_layout(self.root)

    def test_winner_uses_dev_only_and_tie_prefers_original(self):
        a, b = report(0.5), report(0.75)
        self.assertEqual(choose_initializer(a, b), "sft1152")
        self.assertEqual(choose_initializer(b, a), "original")
        self.assertEqual(choose_initializer(a, a), "original")
        for x, y in ((report(0.0, "test"), b), (a, report(1.0, "test"))):
            with self.assertRaisesRegex(ValueError, "Only DEV"):
                choose_initializer(x, y)

    def test_runtime_invalid_proposal_never_wins_even_with_positive_lcb(self):
        candidates = [{"valid": True, "source": source} for source in
                      ("broken_best", "valid_better", "incumbent", "other_broken")]
        outcomes = [report(.875), report(.5), report(.125), report(1)]
        outcomes[0]["harness_error_episodes"] = 1
        outcomes[3]["results"][0]["hook_errors"] = [{"error": "invalid hook output"}]
        selected = select_harness("incumbent", report(.125), candidates, outcomes, seed=42)
        self.assertEqual(selected["selected"]["source"], "valid_better")
        self.assertEqual({row["slot"] for row in selected["runtime_rejected_slots"]}, {0, 3})
        self.assertNotIn("broken_best", {row["source"] for row in selected["candidates"]})
        self.assertEqual(selected["proposals"], 4)
        self.assertEqual(selected["alpha"], .05 / 4)

    def test_duplicate_source_with_any_runtime_error_is_not_adopted(self):
        candidates = [{"valid": True, "source": source} for source in
                      ("duplicate", "duplicate", "incumbent", "other")]
        outcomes = [report(1), report(.875), report(.125), report(.125)]
        outcomes[1]["harness_error_episodes"] = 1
        selected = select_harness("incumbent", report(.125), candidates, outcomes, seed=42)
        self.assertEqual(selected["selected"]["source"], "incumbent")
        self.assertNotIn("duplicate", {row["source"] for row in selected["candidates"]})
        outcomes[0]["cache_identity"]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "Only DEV"):
            select_harness("incumbent", report(.125), candidates, outcomes, seed=42)

    def report_engine(self, invalid_proposal=False, invalid_incumbent=False):
        p = protocol()
        data = self.root / "data"
        save(data / "train.json", [])
        save(data / "dev.json", [])
        p["domains"]["chess"]["data"] = str(data)
        incumbent = self.root / "incumbent.py"
        incumbent.write_text("incumbent")
        snapshot = {"editor": "trained-E", "solver": "saved-W", "harness": str(incumbent)}
        engine = Engine(p, "chess", self.root, {}, MagicMock(), MagicMock())
        generated = []

        def evaluate(url, state, split, samples, output, purpose):
            source = Path(state["harness"]).read_text()
            if output.exists():
                result = read_json(output)
                self.assertEqual(result["cache_identity"]["source_sha256"], digest(source))
                return result
            generated.append(source)
            result = report(.875 if source == "candidate" else .25)
            result["harness_error_episodes"] = int(
                invalid_proposal if source == "candidate" else invalid_incumbent)
            result["cache_identity"]["source_sha256"] = digest(source)
            save(output, result)
            return result

        engine.evaluate = evaluate
        selection = {"selected": {"source": "candidate", "accuracy": .875, "lcb": .5, "slot": 0}}
        return engine, snapshot, selection, generated

    def test_full_report_hook_failure_reverts_before_handoff_and_preserves_resume(self):
        engine, snapshot, selection, generated = self.report_engine(invalid_proposal=True)
        run = self.root / "arm/blocks/0001"
        result = engine.report_harness_choice(run, "h1", snapshot, "incumbent",
                                               report(.25), selection, "fake")
        self.assertEqual(Path(result["harness"]).read_text(), "incumbent")
        self.assertEqual(result["editor"], "trained-E")
        self.assertEqual(read_json(Path(result["dev_report"]))["accuracy"], .25)
        self.assertEqual(read_json(run / "h1_proposed_dev_report.json")["accuracy"], .875)
        receipt = read_json(run / "h1_selection.json")
        self.assertEqual(receipt["selected"]["source"], "incumbent")
        self.assertEqual(receipt["report_validation"]["status"], "retained_incumbent_after_runtime_rejection")
        self.assertTrue(receipt["report_validation"]["committed"])
        self.assertEqual(generated, ["candidate", "incumbent"])
        again = engine.report_harness_choice(run, "h1", snapshot, "incumbent",
                                             report(.25), selection, "fake")
        self.assertEqual(result, again)
        self.assertEqual(generated, ["candidate", "incumbent"])
        self.assertEqual(ledger(self.root / "arm")["physical_rollouts"]["dev_report_rollouts"], 128)
        engine.launch.module.assert_not_called()

    def test_valid_full_report_is_committed_without_duplicate_reporting(self):
        engine, snapshot, selection, generated = self.report_engine()
        run = self.root / "arm/blocks/0001"
        result = engine.report_harness_choice(run, "h1", snapshot, "incumbent",
                                               report(.25), selection, "fake")
        self.assertEqual(Path(result["harness"]).read_text(), "candidate")
        self.assertEqual(generated, ["candidate"])
        self.assertEqual(ledger(self.root / "arm")["physical_rollouts"]["dev_report_rollouts"], 64)
        self.assertTrue(read_json(run / "h1_selection.json")["report_validation"]["committed"])

    def test_invalid_full_incumbent_report_stops_instead_of_committing_bad_h(self):
        for selected_source in ("candidate", "incumbent"):
            engine, snapshot, selection, _ = self.report_engine(
                invalid_proposal=True, invalid_incumbent=True)
            selection["selected"]["source"] = selected_source
            run = self.root / selected_source
            with self.assertRaisesRegex(RuntimeError, "no safe H handoff"):
                engine.report_harness_choice(run, "h1", snapshot, "incumbent",
                                             report(.25), selection, "fake")
            self.assertFalse((run / "h1.py").exists())
            self.assertFalse(read_json(run / "h1_selection.json")["report_validation"]["committed"])

    def test_server_records_actual_ports_layout_and_reserved_gpu_seconds(self):
        engine, snapshot, _, _ = self.report_engine()
        observed = []

        @contextmanager
        def vllm(model, **kwargs):
            observed.append(kwargs)
            yield ",".join(f"http://127.0.0.1:{18801 + 10 * i}/v1"
                           for i in range(kwargs["data_parallel"]))

        engine.launch.vllm.side_effect = vllm
        arm = self.root / "arm"
        for full, seconds in ((False, 10), (True, 20)):
            with patch("sao_bridge.domain_campaign.time.monotonic", side_effect=[100, 100 + seconds]):
                with engine.server(snapshot, f"server_{full}", full_node=full, ledger_root=arm) as url:
                    active = read_json(self.root / "ACTIVE_SERVERS.json")
                    self.assertEqual(active["status"], "ready")
                    self.assertEqual(active["solver_server_layout"]["data_parallel"], 8 if full else 4)
                    self.assertEqual(len(url.split(",")), 8 if full else 4)
        self.assertEqual(observed[0]["gpus"], "4,5,6,7")
        self.assertEqual(observed[1]["gpus"], "0,1,2,3,4,5,6,7")
        self.assertTrue(all(row["max_model_len"] == 16384 for row in observed))
        result = inference_ledger(arm)
        self.assertEqual(result["recorded_server_sessions"], 2)
        self.assertEqual(result["incomplete_server_sessions"], 0)
        self.assertEqual(result["allocated_gpu_seconds"], 200)
        self.assertEqual(result["server_wall_seconds"], 30)
        self.assertEqual(read_json(self.root / "ACTIVE_SERVERS.json")["status"], "closed")

    def test_phase_plans_never_cross_single_role_boundaries(self):
        for block in range(12):
            self.assertEqual(phase_plan("harness_only", block), ("H",))
            self.assertEqual(phase_plan("weight_only", block), ("W",))
        for arm in ("sao", "ma_sao"):
            self.assertEqual(phase_plan(arm, 0), ("H", "W", "H"))
            self.assertEqual(phase_plan(arm, 1), ("W", "H"))
        with self.assertRaises(ValueError):
            phase_plan("custom_ppo", 0)

    def test_plateau_keeps_best_saved_checkpoint_and_enforces_minimum(self):
        stopping = protocol()["stopping"]
        state = initial_state({"editor": "e0", "solver": "w0", "harness": "h0"}, 0.5)
        for block in range(1, 4):
            snapshot = {"editor": f"e{block}", "solver": f"w{block}",
                        "harness": f"h{block}", "dev_report": f"dev{block}"}
            state = update_plateau(state, snapshot, report(0.5), stopping)
            self.assertEqual(state["status"], "converged" if block == 3 else "active")
        self.assertEqual(state["best"]["editor"], "e0")
        self.assertEqual(state["current"]["editor"], "e3")
        with self.assertRaisesRegex(ValueError, "Only DEV"):
            update_plateau(state, snapshot, report(1.0, "test"), stopping)

    def test_plateau_meaningful_gain_anchor_and_fixed_best(self):
        state = initial_state({"solver": "initial"}, 0.5)
        stopping = protocol()["stopping"]
        for score in (0.503, 0.506, 0.508, 0.506, 0.507):
            result = report(0.5)
            result["accuracy"] = score
            state = update_plateau(state, {"solver": str(score), "dev_report": str(score)},
                                   result, stopping)
        self.assertEqual(state["anchor_score"], 0.506)
        self.assertEqual(state["best_score"], 0.508)
        self.assertEqual(state["stale"], 3)
        self.assertEqual(state["status"], "converged")

    def test_cap_is_budget_exhausted_not_converged(self):
        state = initial_state({"solver": "initial"}, 0.0)
        for block in range(1, 13):
            result = report(0.0)
            result["accuracy"] = block / 100
            state = update_plateau(state, {"solver": str(block), "dev_report": str(block)},
                                   result, protocol()["stopping"])
        self.assertEqual(state["status"], "budget_exhausted")

    def test_ma_gate_requires_positive_absolute_and_relative_dev_gain(self):
        stopping = protocol()["stopping"]
        a = b = {"blocks": 1, "stale": 0}
        decision = extension_decision(report(0.5), report(0.5), report(0.75), a, b, stopping)
        self.assertEqual(decision["decision"], "continue")
        self.assertTrue(decision["ma_point_estimate_advantage"])
        for left, right in ((0.75, 0.75), (0.5, 0.5), (0.125, 0.375)):
            result = extension_decision(report(0.5), report(left), report(right), a, b, stopping)
            self.assertEqual(result["decision"], "not_extended")
        with self.assertRaisesRegex(ValueError, "Only DEV"):
            extension_decision(report(0.5), report(0.5), report(0.75, "test"), a, b, stopping)

    def test_paired_gate_common_patience_and_budget_are_explicit(self):
        stopping = protocol()["stopping"]
        a, b = {"blocks": 4, "stale": 3}, {"blocks": 4, "stale": 2}
        self.assertEqual(extension_decision(report(0), report(.25), report(.5),
                                            a, b, stopping)["decision"], "continue")
        b["stale"] = 3
        self.assertEqual(extension_decision(report(0), report(.25), report(.5),
                                            a, b, stopping)["decision"], "converged")
        a = b = {"blocks": 12, "stale": 0}
        self.assertEqual(extension_decision(report(0), report(.25), report(.5),
                                            a, b, stopping)["decision"], "budget_exhausted")
        self.assertEqual(extension_decision(report(0), report(.5), report(.5),
                                            a, b, stopping)["decision"], "extension_gate_closed")

    def test_matched_recycle_pool_uses_1024_plus_2048_disjoint(self):
        p = protocol()
        pool = [{"id": f"episode-{i}"} for i in range(4096)]
        warm, actor, selected = recycling_partition(pool, p["common"], p["algorithm"])
        self.assertEqual((len(warm), len(actor), len(selected)), (1024, 2048, 3072))
        self.assertFalse(set(warm) & set(actor))
        self.assertEqual(len(set(warm + actor)), 3072)
        self.assertEqual(selected, pool[:3072])
        for broken in (pool[:3000], pool + [pool[0]]):
            with self.assertRaisesRegex(ValueError, "disjoint"):
                recycling_partition(broken, p["common"], p["algorithm"])

    def test_unchanged_harness_fresh_phase_recycling_cannot_collide(self):
        from sao_bridge.replay import recycle_pool

        runs = [self.root / "ma_sao/blocks" / number for number in ("0001", "0002")]
        record = {
            "request_index": 0, "harness_source": "unchanged CandidateHarness",
            "results": [{"id": "same-question", "reward": 0.0, "calls": [{
                "prompt_ids": [1] * 8200, "response_ids": [2],
                "response_logprobs": [-0.1], "temperature": 1.0, "stop_reason": "stop",
            }]}],
        }
        pools, namespaces = [], []
        for run in runs:
            directory = run / "h1_reward_evals"
            save(directory / "000000-candidate-same-source.json", record)
            namespace = recycling_namespace({"domain": "chess", "seed": 42}, run, "h1")
            namespaces.append(namespace)
            pool = recycle_pool(directory, context_tokens=16384, request_indices={0},
                                namespace=namespace)
            self.assertEqual(pool, recycle_pool(directory, context_tokens=16384,
                                                request_indices={0}, namespace=namespace))
            pools.append(pool)
            with self.assertRaisesRegex(ValueError, "context"):
                recycle_pool(directory, context_tokens=8192, namespace=namespace)
        self.assertNotEqual(namespaces[0], namespaces[1])
        self.assertNotEqual(pools[0][0]["id"], pools[1][0]["id"])
        self.assertNotEqual(namespaces[0], recycling_namespace(
            {"domain": "chess", "seed": 42}, runs[0], "h2"))

    def test_training_feedback_has_real_failure_trace_and_no_dev_test_path(self):
        result = report(0.5, "train")
        for row in result["results"]:
            row["reference_answer"] = "PRIVATE_REFERENCE_SENTINEL"
            row["harness_errors"] = [{"error": "actual hook error"}]
        text = training_feedback(result)
        self.assertIn("actual model action", text)
        self.assertIn("illegal move", text)
        self.assertIn("actual hook error", text)
        self.assertNotIn("PRIVATE_REFERENCE_SENTINEL", text)
        self.assertEqual(json.loads(text)["split"], "train")
        for split in ("test", "dev"):
            with self.assertRaisesRegex(ValueError, "training-only"):
                training_feedback(report(0.5, split))

    def test_cache_rejects_wrong_seed_count_identity_and_score(self):
        rows = [{"id": f"{i:08x}" + "0" * 56, "question": str(i)} for i in range(3)]
        identity = evaluation_identity({"id": "campaign"}, "model", "source", "dev", rows, 2, 42, "selection")
        value = {
            "question_count": 3, "samples_per_question": 2, "metric": "mean@2", "accuracy": .5,
            "cache_identity": identity, "results": [
                {"id": row["id"], "sample_index": index,
                 "rollout_seed": 42 + int(row["id"][:8], 16) + 1009 * index, "reward": index}
                for row in rows for index in range(2)],
        }
        path = self.root / "report.json"
        save(path, value)
        self.assertEqual(verify_report(path, rows, identity)["accuracy"], .5)
        for mutation in ("seed", "count", "score", "model", "duplicate"):
            changed = copy.deepcopy(value)
            if mutation == "seed":
                changed["results"][0]["rollout_seed"] += 1
            elif mutation == "count":
                changed["results"].pop()
            elif mutation == "score":
                changed["accuracy"] = 1.0
            elif mutation == "model":
                changed["cache_identity"]["model_sha256"] = "wrong"
            else:
                changed["results"][0] = changed["results"][1]
            save(path, changed)
            with self.assertRaises(ValueError, msg=mutation):
                verify_report(path, rows, identity)

    def fake_optimizer(self):
        out = self.root / "w1"
        spec = {"phase": "w1", "run": str(out), "steps": 40, "critic_warmup_steps": 8}
        for role in ("actor", "critic"):
            directory = out / role
            directory.mkdir(parents=True)
            (directory / "model.safetensors").write_bytes(b"test weights, no GPU")
        save(self.root / "w1_optimizer_intent.json", spec)
        save(out / "trained.json", {
            "actor_steps": 32, "critic_only_warmup_steps": 8, "critic_steps": 80,
            "stage_optimizer_reset": True, "central_critic": False,
            "checkpoints": {role: str(out / role) for role in ("actor", "critic")},
        })
        (out / "metrics.jsonl").write_text("".join(
            json.dumps({"global_step": i}) + "\n" for i in range(40)))
        return spec

    def test_optimizer_launch_intent_is_never_repeated_after_partial_crash(self):
        spec = {"phase": "w1", "run": str(self.root / "w1"), "steps": 40, "critic_warmup_steps": 8}
        self.assertIsNone(completed_optimizer(self.root, spec))
        immutable(self.root / "w1_optimizer_intent.json", spec)
        with self.assertRaisesRegex(ValueError, "partial optimizer resume"):
            completed_optimizer(self.root, spec)

    def test_optimizer_completed_boundary_requires_all_steps_and_checkpoints(self):
        spec = self.fake_optimizer()
        result = completed_optimizer(self.root, spec)
        self.assertEqual(result["actor_steps"], 32)
        (self.root / "w1/metrics.jsonl").write_text('{"global_step": 0}\n')
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            completed_optimizer(self.root, spec)

    def test_optimizer_cannot_silently_change_spec_or_resume_moments(self):
        spec = self.fake_optimizer()
        changed = {**spec, "steps": 41}
        with self.assertRaisesRegex(ValueError, "changed"):
            completed_optimizer(self.root, changed)
        value = read_json(self.root / "w1/trained.json")
        value["stage_optimizer_reset"] = False
        save(self.root / "w1/trained.json", value)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            completed_optimizer(self.root, spec)

    def test_immutable_records_and_cross_node_file_permissions(self):
        path = self.root / "nested/identity.json"
        immutable(path, {"value": 1})
        immutable(path, {"value": 1})
        self.assertEqual(path.stat().st_mode & 0o777, 0o644)
        self.assertEqual(path.parent.stat().st_mode & 0o777, 0o755)
        with self.assertRaisesRegex(ValueError, "Immutable"):
            immutable(path, {"value": 2})

    def test_worker_claim_prevents_duplicate_concurrent_launch(self):
        with worker_claim(self.root, {"campaign": "test"}, False):
            with self.assertRaisesRegex(RuntimeError, "Another campaign"):
                with worker_claim(self.root, {"campaign": "test"}, True):
                    self.fail("Concurrent lock acquired")
        with self.assertRaisesRegex(ValueError, "--resume"):
            with worker_claim(self.root, {"campaign": "test"}, False):
                self.fail("Undeclared resume accepted")
        with worker_claim(self.root, {"campaign": "test"}, True):
            pass

    def test_retrieval_health_accepts_root_and_endpoint_not_unrelated_paths(self):
        for url in ("http://127.0.0.1:18790", "http://127.0.0.1:18790/retrieve"):
            self.assertEqual(retrieval_health_url(url), "http://127.0.0.1:18790/health")
        for url in ("", "https://remote.invalid", "http://host/wrong", "http://user:secret@host"):
            with self.assertRaisesRegex(ValueError, "trusted HTTP"):
                retrieval_health_url(url)

    def test_training_epochs_wrap_with_distinct_new_episode_ids(self):
        rows = [{"id": f"{i:064x}", "question": str(i)} for i in range(6)]
        first, one = training_window(rows, count=4, offset=4, namespace="phase-1")
        second, two = training_window(rows, count=4, offset=10, namespace="phase-2")
        self.assertEqual([r["campaign_question_id"] for r in first],
                         [rows[i]["id"] for i in (4, 5, 0, 1)])
        self.assertEqual(one["epoch_counts"], {"0": 2, "1": 2})
        self.assertEqual(two["epoch_counts"], {"1": 2, "2": 2})
        self.assertFalse({r["id"] for r in first} & {r["id"] for r in second})
        self.assertEqual(first, training_window(rows, 4, 4, "phase-1")[0])
        self.assertFalse(one["previous_episode_replay"])
        self.assertTrue(one["fresh_generation"])
        self.assertEqual({r["id"] for r in first}, {r["campaign_episode_id"] for r in first})
        for row in first + second:
            int(row["id"], 16)
        with self.assertRaisesRegex(ValueError, "distinct"):
            training_window(rows, 7, 0, "too-many")

    def test_reward_windows_disclose_reuse_and_solver_training_exposure(self):
        windows = [reward_window(index, "ma_sao") for index in range(6)]
        self.assertEqual([row["offset"] for row in windows], [0, 32, 64, 96, 0, 32])
        self.assertEqual([row["rotation"] for row in windows], [0, 0, 0, 0, 1, 1])
        self.assertFalse(any(row["unseen_question_claim"] for row in windows))
        self.assertFalse(windows[3]["may_have_entered_solver_optimizer"])
        self.assertTrue(windows[4]["may_have_entered_solver_optimizer"])
        self.assertFalse(reward_window(4, "harness_only")["may_have_entered_solver_optimizer"])

    def test_heartbeat_and_phase_status_writes_are_serialized(self):
        active, maximum = 0, 0
        counter_lock = threading.Lock()
        original_save = save

        def slow_save(path, value):
            nonlocal active, maximum
            with counter_lock:
                active += 1
                maximum = max(active, maximum)
            time.sleep(.001)
            original_save(path, value)
            with counter_lock:
                active -= 1

        with patch("sao_bridge.domain_campaign.save", slow_save):
            with Monitor(self.root, .001, 60) as monitor:
                with ThreadPoolExecutor(max_workers=4) as executor:
                    list(executor.map(lambda i: monitor.update("phase", phase=i), range(24)))
        self.assertEqual(maximum, 1)
        self.assertIn("heartbeat_at", read_json(self.root / "STATUS.json"))

    def test_heartbeat_does_not_disguise_lack_of_progress(self):
        monitor = Monitor(self.root, 15, 60)
        monitor.update("training")
        first = read_json(self.root / "STATUS.json")
        with patch("sao_bridge.domain_campaign.time.time", return_value=first["heartbeat_at"] + 100):
            with monitor.lock:
                monitor._write()
        last = read_json(self.root / "STATUS.json")
        self.assertEqual(last["progress_at"], first["progress_at"])
        self.assertEqual(last["progress_deadline_at"], first["progress_deadline_at"])
        self.assertGreater(last["heartbeat_at"], first["heartbeat_at"])

    def test_watcher_detects_failure_dead_heartbeat_and_live_heartbeat_stall(self):
        campaign = self.campaign()
        p = campaign.p
        roots = [Path(node["root"]) for node in p["nodes"].values()]
        for root in roots:
            save(root / "STATUS.json", {"status": "training", "heartbeat_at": 100,
                                       "progress_deadline_at": 150})
        self.assertFalse(watcher_snapshot(p, now=110, stale_seconds=20, grace_seconds=0)["alerts"])
        save(roots[0] / "STATUS.json", {"status": "failed", "error": "native optimizer failed"})
        result = watcher_snapshot(p, now=160, stale_seconds=20, grace_seconds=0)
        self.assertEqual({row["kind"] for row in result["alerts"]},
                         {"worker_failed", "heartbeat_stale", "progress_stalled"})
        for root in roots:
            save(root / "STATUS.json", {"status": "training", "heartbeat_at": 1000,
                                       "progress_deadline_at": 150})
        self.assertEqual({row["kind"] for row in watcher_snapshot(
            p, now=1000, stale_seconds=20, grace_seconds=0)["alerts"]}, {"progress_stalled"})

    def test_watcher_persists_alert_and_notifies_supervisor_with_nonzero_exit(self):
        campaign = self.campaign()
        save(campaign.root / "STATUS.json", {"status": "failed", "error": "native phase failed"})
        with patch("sao_bridge.domain_campaign.sys.stderr", new_callable=StringIO) as stderr:
            code = watch(campaign.p, campaign.host, interval=.001)
        self.assertEqual(code, 2)
        self.assertIn("worker_failed", stderr.getvalue())
        self.assertEqual(read_json(campaign.root / "WATCHER_ALERT.json")["status"], "alert")
        self.assertEqual(read_json(campaign.root / "WATCHER_STATUS.json")["status"], "alert")

    def test_baseline_monitor_completes_but_campaign_monitor_waits_for_workers(self):
        campaign = self.campaign()
        for node in campaign.p["nodes"].values():
            root = Path(node["root"])
            save(root / "STATUS.json", {"status": "baselines_ready", "heartbeat_at": 1})
            for domain in campaign.p["domain_order"]:
                save(root / domain / "BASELINE_READY.json", {})
        snapshot = watcher_snapshot(campaign.p, now=10000)
        self.assertFalse(snapshot["alerts"])
        self.assertTrue(snapshot["baselines_complete"])
        self.assertFalse(snapshot["campaign_complete"])
        self.assertEqual(watch(campaign.p, campaign.host, until="baseline"), 0)
        self.assertEqual(read_json(campaign.root / "WATCHER_STATUS.json")["status"], "completed")

    def test_searchqa_only_monitor_does_not_require_chess_exports(self):
        campaign = self.campaign()
        for node in campaign.p["nodes"].values():
            root = Path(node["root"])
            save(root / "STATUS.json", {"status": "baselines_ready"})
            save(root / "searchqa/BASELINE_READY.json", {})
        self.assertTrue(watcher_snapshot(campaign.p, domains=["searchqa"])["baselines_complete"])
        self.assertFalse(watcher_snapshot(campaign.p)["baselines_complete"])
        self.assertEqual(watch(campaign.p, campaign.host, until="baseline", domain="searchqa"), 0)

    def test_watcher_does_not_complete_or_flag_idle_after_only_one_domain(self):
        campaign = self.campaign()
        for node in campaign.p["nodes"].values():
            root = Path(node["root"])
            save(root / "STATUS.json", {"status": "domains_completed", "heartbeat_at": 1,
                                        "completed_domains": ["searchqa"]})
            save(root / "searchqa/DOMAIN_COMPLETE.json", {})
        state = watcher_snapshot(campaign.p, now=10000)
        self.assertFalse(state["campaign_complete"])
        self.assertFalse(state["alerts"])
        for node in campaign.p["nodes"].values():
            save(Path(node["root"]) / "chess/DOMAIN_COMPLETE.json", {})
        self.assertTrue(watcher_snapshot(campaign.p)["campaign_complete"])

    def campaign(self):
        p = protocol()
        p["nodes"]["dev-NDH200-026"]["root"] = str(self.root / "node4")
        p["nodes"]["dev-NDH200-025"]["root"] = str(self.root / "node3")
        return Campaign(p, "dev-NDH200-026", MagicMock())

    def test_single_state_restarts_from_winner_not_alternating_final(self):
        campaign = self.campaign()
        root = self.root / "domain"
        root.mkdir()
        (root / "initial.py").write_text("INITIAL")
        save(root / "frozen_baseline_sft1152_dev.json", report(.5))
        freeze = {"winner": "sft1152", "model": "winner", "model_sha256": "winner-sha",
                  "scores": {"sft1152": .5}}
        alternating = campaign.arm_state(root, "sao", freeze)
        alternating["current"]["editor"] = "trained-editor"
        alternating["current"]["solver"] = "trained-solver"
        save(root / "sao/STATE.json", alternating)
        single = campaign.arm_state(root, "weight_only", freeze)
        self.assertEqual(single["current"]["solver"], "winner")
        self.assertEqual(single["current"]["editor"], "winner")

    def test_continuation_consumes_saved_editor_solver_harness_not_initial(self):
        campaign = self.campaign()
        root = self.root / "domain"
        report_path = self.root / "dev.json"
        save(report_path, report(.5))
        current = {"editor": "saved-E", "solver": "saved-W", "harness": "saved-H",
                   "dev_report": str(report_path)}
        state = {**initial_state(current, .5), "blocks": 1, "arm": "sao"}
        save(root / "sao/STATE.json", state)
        seen = []

        class FakeEngine:
            def __init__(self, *args):
                pass

            def phase(self, arm, role, run, snapshot):
                seen.append((role, snapshot["editor"], snapshot["solver"], snapshot["harness"]))
                return snapshot

        @contextmanager
        def allocate(*args):
            yield None

        with patch.object(campaign, "arm_state", return_value=state), \
                patch("sao_bridge.domain_campaign.verify_snapshot"), \
                patch("sao_bridge.domain_campaign.allocation", allocate), \
                patch("sao_bridge.domain_campaign.Engine", FakeEngine):
            campaign.block("chess", root, {}, {}, "sao")
        self.assertEqual(seen, [
            ("W", "saved-E", "saved-W", "saved-H"),
            ("H", "saved-E", "saved-W", "saved-H"),
        ])
        self.assertEqual(read_json(root / "sao/STATE.json")["blocks"], 2)

    def test_resume_does_not_repeat_block_when_terminal_decision_already_saved(self):
        campaign = self.campaign()
        root = self.root / "domain"
        state = {"status": "active", "blocks": 1}
        save(root / "sao/DECISION_0001.json", {"decision": "not_extended"})
        with patch.object(campaign, "arm_state", return_value=state), \
                patch.object(campaign, "block") as block, \
                patch.object(campaign, "finalize") as finalize:
            result = campaign.alternating("chess", root, {}, {})
        block.assert_not_called()
        finalize.assert_called_once()
        self.assertEqual(result["status"], "not_extended")

    def test_ledger_separates_physical_work_from_extra_critic_compute(self):
        run = self.root / "arm/blocks/0001"
        save(run / "w1/receipts/a.json", {
            "role": "solver", "replayed": False, "generated_tokens": 100,
            "recycled_partner_episodes": ["x"]})
        save(run / "w1/trained.json", {
            "actor_steps": 32, "critic_steps": 80, "critic_only_warmup_steps": 8,
            "elapsed_seconds": 10})
        save(run / "w1_spec.json", {"actor_gpus": 4, "rollout_gpus": 4})
        (run / "w1/metrics.jsonl").write_text(json.dumps({"metrics": {"ppo_actor/n_seqs": 2}}) + "\n")
        save(self.root / "arm/test.json", report(.5, "test"))
        result = ledger(self.root / "arm")
        self.assertEqual(result["physical_rollouts"]["solver_generated"], 1)
        self.assertEqual(result["physical_rollouts"]["test_report_rollouts"], 64)
        self.assertEqual(result["optimizer_compute"]["actor_updates"], 32)
        self.assertEqual(result["optimizer_compute"]["critic_only_warmup_updates"], 8)
        self.assertEqual(result["optimizer_compute"]["allocated_training_gpu_seconds"], 80)

    def test_identity_audits_test_hash_and_cross_split_overlap(self):
        data = self.root / "data"
        p = protocol()
        p["domains"]["chess"]["data"] = str(data)
        receipt = self.root / "sft.json"
        p["sft_receipt"] = str(receipt)
        save(receipt, {"status": "completed", "hf_cpu_reload_verified": True,
                       "checkpoint": p["models"]["sft1152"], "cumulative_train_records": 1152})
        splits = {}
        for index, (split, count) in enumerate((("train", 2048), ("reward", 128), ("dev", 256), ("test", 256))):
            rows = [{"id": f"{index * 10000 + k:064x}"} for k in range(count)]
            save(data / f"{split}.json", rows)
            splits[split] = {"sha256": file_sha(data / f"{split}.json")}
        save(data / "manifest.json", {"splits": splits})
        with patch("sao_bridge.domain_campaign.initial_model_digest", return_value="weights"), \
                patch("sao_bridge.domain_api.get_domain", return_value=MagicMock(INITIAL_SOURCE="initial")):
            self.assertEqual(audit_identity(p, "chess")["splits"]["test"]["count"], 256)
            rows = read_json(data / "test.json")
            rows[0]["id"] = read_json(data / "dev.json")[0]["id"]
            save(data / "test.json", rows)
            with self.assertRaisesRegex(ValueError, "test split hash"):
                audit_identity(p, "chess")
            splits["test"]["sha256"] = file_sha(data / "test.json")
            save(data / "manifest.json", {"splits": splits})
            with self.assertRaisesRegex(ValueError, "overlapping"):
                audit_identity(p, "chess")

    def test_native_train_api_and_completed_resume_never_spawn_twice(self):
        p = protocol()
        data = self.root / "data"
        save(data / "train.json", [{"id": "00000000", "question": "training"}])
        save(data / "dev.json", [])
        p["domains"]["chess"]["data"] = str(data)
        monitor = MagicMock()
        monitor.deadline = float("inf")
        launch = MagicMock()
        snapshot = {"solver": "saved-W", "solver_critic": "saved-C",
                    "editor": "saved-E", "harness": "saved-H"}
        source = self.root / "solver_data.json"
        save(source, [])

        def module(module_name, arguments, **kwargs):
            self.assertEqual(module_name, "sao_bridge.large_train")
            spec = read_json(arguments[1])
            self.assertEqual(spec["domain"], "chess")
            self.assertEqual(spec["context_tokens"], 16384)
            self.assertEqual(spec["tokens"], 4096)
            self.assertEqual(spec["steps"], 40)
            self.assertEqual(spec["critic_warmup_steps"], 8)
            self.assertEqual(spec["critic_lr"], 1e-5)
            self.assertEqual(spec["model"], "saved-W")
            self.assertEqual(spec["critic_model"], "saved-C")
            self.assertIsNone(spec["central_context"])
            out = Path(spec["run"])
            for role in ("actor", "critic"):
                (out / role).mkdir(parents=True)
                (out / role / "model.safetensors").write_bytes(b"test")
            save(out / "trained.json", {
                "actor_steps": 32, "critic_only_warmup_steps": 8, "critic_steps": 80,
                "stage_optimizer_reset": True, "central_critic": False,
                "checkpoints": {role: str(out / role) for role in ("actor", "critic")}})
            (out / "metrics.jsonl").write_text("".join(
                json.dumps({"global_step": i}) + "\n" for i in range(40)))
            return MagicMock()

        launch.module.side_effect = module
        engine = Engine(p, "chess", self.root, {}, launch, monitor)
        launch.vllm.return_value.__enter__.return_value = ",".join(
            row["api_url"] for row in solver_server_layout()["replicas"])
        with engine.server(snapshot, "frozen_solver_context"):
            pass
        self.assertEqual(launch.vllm.call_args.kwargs["max_model_len"], 16384)
        with patch("sao_bridge.domain_campaign.initial_model_digest", return_value="checkpoint-hash"):
            after = engine.train(self.root, "w1", "solver", snapshot, source,
                                 recycle=self.root / "recycle.json")
            again = engine.train(self.root, "w1", "solver", snapshot, source,
                                 recycle=self.root / "recycle.json")
        launch.module.assert_called_once()
        self.assertEqual(after, again)
        self.assertEqual(after["editor"], "saved-E")
        self.assertEqual(after["harness"], "saved-H")
        self.assertNotEqual(after["solver"], "saved-W")

    def test_two_node_barriers_singles_and_resumed_campaign_without_gpus(self):
        for advantage in (False, True):
            p = protocol()
            scenario = self.root / str(advantage)
            p["runtime"]["heartbeat_seconds"] = .001
            for host, node in p["nodes"].items():
                node["root"] = str(scenario / host)
            campaigns = [Campaign(p, host, MagicMock()) for host in p["nodes"]]
            calls = []

            def prepare(campaign, domain):
                root = campaign.root / domain
                root.mkdir(parents=True, exist_ok=True)
                (root / "initial.py").write_text("initial harness")
                return root, {"domain": domain, "models": {"original": "raw-hash", "sft1152": "sft-hash"}}

            def baseline(campaign, domain, root, identity):
                model = campaign.node["baseline"]
                path = root / f"baseline_{model}_dev.json"
                save(path, report(.5))
                immutable(root / "BASELINE_READY.json", {
                    "model": model, "report": str(path), "identity": identity})

            def verify(path, *_):
                record = read_json(path)
                return record, read_json(Path(record["report"]))

            @contextmanager
            def allocate(*_):
                yield None

            class FakeEngine:
                def __init__(self, *_):
                    pass

                def phase(self, arm, role, run, snapshot):
                    phase = "h_count" if role == "H" else "w_count"
                    after = copy.deepcopy(snapshot)
                    calls.append((str(run.parent.parent), int(run.name), role,
                                  snapshot["editor"], snapshot["solver"], snapshot["harness"]))
                    after[phase] += 1
                    if role == "H":
                        after["editor"] = f"saved-E-{after[phase]}"
                        after["harness"] = f"saved-H-{after[phase]}"
                    else:
                        after["solver"] = f"saved-W-{after[phase]}"
                    score = (.75 if arm == "ma_sao" else .625) if advantage else .5
                    path = run / f"{role}_{after[phase]}_dev.json"
                    save(path, report(score))
                    after["dev_report"] = str(path)
                    return after

                @contextmanager
                def server(self, *_, **kwargs):
                    yield "fake://never-launched"

                def evaluate(self, url, snapshot, split, samples, output, purpose):
                    value = report(1.0, split, samples)
                    save(output, value)
                    return value

            with patch.object(Campaign, "prepare", prepare), \
                    patch.object(Campaign, "baseline", baseline), \
                    patch("sao_bridge.domain_campaign.verify_baseline", verify), \
                    patch("sao_bridge.domain_campaign.allocation", allocate), \
                    patch("sao_bridge.domain_campaign.verify_snapshot"), \
                    patch("sao_bridge.domain_campaign.Engine", FakeEngine):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [executor.submit(c.run_domain, "chess") for c in campaigns]
                    for future in futures:
                        future.result(timeout=15)
                count = len(calls)
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [executor.submit(c.run_domain, "chess") for c in campaigns]
                    for future in futures:
                        future.result(timeout=15)
                self.assertEqual(len(calls), count, "Completed resume performed another phase")
            for campaign in campaigns:
                root = campaign.root / "chess"
                alternating = read_json(root / campaign.node["alternating"] / "STATE.json")
                single = read_json(root / campaign.node["single"] / "STATE.json")
                self.assertEqual(alternating["blocks"], 4 if advantage else 1)
                self.assertEqual(alternating["status"], "converged" if advantage else "not_extended")
                self.assertEqual(single["status"], "converged")
                self.assertTrue((root / "DOMAIN_COMPLETE.json").is_file())
                self.assertTrue((root / "TRAINING_IMPLEMENTATION.json").is_file())
                self.assertFalse((campaign.root / "TRAINING_IMPLEMENTATION.json").exists())
                if advantage:
                    second = [entry for entry in calls
                              if Path(entry[0]).name == campaign.node["alternating"] and entry[1] == 2]
                    self.assertEqual([entry[2] for entry in second], ["W", "H"])
                    self.assertEqual(second[0][3:], ("saved-E-2", "saved-W-1", "saved-H-2"))


if __name__ == "__main__":
    unittest.main()
