from contextlib import contextmanager
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge import alfworld_matched_engineer_baseline as matched
from sao_bridge.budget import contract_digest
from sao_bridge.common import file_sha, read_json, write_json


class MatchedEngineerBaselineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.reference = self.root / "reference" / "evaluation"
        self.model = self.root / "assets" / "agent-sft-harness-r1"
        self.reference_spec = self.root / "reference" / "evaluation_protocol.json"
        self.prompts = [{"id": f"prompt-{index}", "messages": []} for index in range(8)]
        prompts_path = self.root / "prompts.json"
        write_json(prompts_path, self.prompts)
        self.spec = {
            "run": str(self.reference), "checkpoint_steps": [15, 31], "seed": 42,
            "initial_engineer": str(self.root / "assets" / "harness-r1"),
            "solver_model": str(self.root / "assets" / "target-agent-sft"),
            "evaluation_prompts": str(prompts_path), "engineer_temperature": 0.7,
            "engineer_tokens": 4096, "target_rollout_concurrency": 16,
            "candidate_evaluation_concurrency": 8, "bootstrap_draws": 100,
        }
        write_json(self.reference_spec, self.spec)
        self.manifest = {"report_task_ids": [300, 301]}
        write_json(self.reference / "COMPLETE.json", {
            "status": "complete", "report_opened": True,
            "selected": {"label": "step15", "global_step": 15},
        })
        write_json(self.reference / "identity.json", {"spec": self.spec})
        write_json(self.reference / "models.json", {"step15": "frozen-sao-engineer"})
        sao_patch = {"kind": "sao"}
        self.proposals = [{
            "sample_index": index, "prompt_id": self.prompts[index]["id"],
            "decode_seed": 5_000_045 + index, "model_label": "step15",
            "valid": True, "patch": sao_patch, "patch_sha256": contract_digest(sao_patch),
        } for index in range(8)]
        self.proposals_path = self.reference / "proposals" / "step15.json"
        write_json(self.proposals_path, self.proposals)
        for subfolder in ("harness-r1", "target-agent-sft", "agent-sft-harness-r1"):
            write_json(self.root / "assets" / subfolder / "provenance.json", {
                "model": "released-repo", "revision": "pinned-release", "subfolder": subfolder,
                "files": [{"path": "model.safetensors", "size": 7, "verified": True}],
            })
        (self.model / "model.safetensors").write_bytes(b"weights")
        self.status_path = self.model.parent / "agent-sft-harness-r1_DOWNLOAD_STATUS.json"
        write_json(self.status_path, {
            "status": "verified_complete", "receipt_sha256": file_sha(self.model / "provenance.json"),
            "model_path": str(self.model),
        })
        self.config = {
            "protocol": "alfworld-matched-official-engineer-v1",
            "run": str(self.root / "comparison"), "reference_spec": str(self.reference_spec),
            "official_model": str(self.model),
            "reference_sha256": {
                "protocol": file_sha(self.reference_spec),
                "completion": file_sha(self.reference / "COMPLETE.json"),
                "identity": file_sha(self.reference / "identity.json"),
                "models": file_sha(self.reference / "models.json"),
                "proposals": file_sha(self.proposals_path),
            },
        }
        self.path = self.root / "protocol.json"
        write_json(self.path, self.config)
        validated = patch.object(
            matched.evaluation, "validate_spec", return_value=(self.spec, self.manifest, self.prompts),
        )
        validated.start()
        self.addCleanup(validated.stop)

    def test_inputs_bind_exact_reference_and_target_specific_release(self):
        request, spec, manifest, prompts, label, proposals, identity = matched.validate_inputs(self.path)
        self.assertEqual(request, self.config)
        self.assertEqual(spec, self.spec)
        self.assertEqual(manifest["report_task_ids"], [300, 301])
        self.assertEqual(prompts, self.prompts)
        self.assertEqual(label, "step15")
        self.assertEqual(proposals, self.proposals)
        self.assertEqual(identity["reference_sha256"], self.config["reference_sha256"])
        self.assertEqual(identity["official_provenance_sha256"], file_sha(self.model / "provenance.json"))

    def test_changed_reference_and_mismatched_proposal_slots_are_rejected(self):
        self.proposals[0]["decode_seed"] += 1
        write_json(self.proposals_path, self.proposals)
        with self.assertRaisesRegex(ValueError, "reference evidence changed"):
            matched.validate_inputs(self.path)
        self.config["reference_sha256"]["proposals"] = file_sha(self.proposals_path)
        write_json(self.path, self.config)
        with self.assertRaisesRegex(ValueError, "slots, patches or decode seeds"):
            matched.validate_inputs(self.path)

    def test_wrong_engineer_release_or_partial_download_cannot_be_a_baseline(self):
        path = self.model / "provenance.json"
        provenance = read_json(path)
        for field, value in (("subfolder", "harness-r1"), ("revision", "different-release")):
            with self.subTest(field=field):
                write_json(path, {**provenance, field: value})
                with self.assertRaisesRegex(ValueError, "target-specific checkpoint"):
                    matched.validate_inputs(self.path)
        write_json(path, provenance)
        (self.model / "model.safetensors").write_bytes(b"partial")
        status = read_json(self.status_path)
        write_json(self.status_path, {**status, "status": "downloading"})
        with self.assertRaisesRegex(ValueError, "target-specific checkpoint"):
            matched.validate_inputs(self.path)
        write_json(self.status_path, status)
        (self.model / "model.safetensors").write_bytes(b"short")
        with self.assertRaisesRegex(ValueError, "target-specific checkpoint"):
            matched.validate_inputs(self.path)

    def test_reference_must_be_selected_and_new_outputs_must_be_separate(self):
        path = self.reference / "COMPLETE.json"
        complete = read_json(path)
        write_json(path, {**complete, "selected": None})
        with self.assertRaisesRegex(ValueError, "already selected"):
            matched.validate_inputs(self.path)
        write_json(path, complete)
        for root in (self.reference, self.reference / "new-baseline", self.reference.parent):
            with self.subTest(root=root):
                write_json(self.path, {**self.config, "run": str(root)})
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    matched.validate_inputs(self.path)

    def test_prior_report_requires_both_success_and_release(self):
        root = self.root / "prior-report"
        write_json(root / "COMPLETE.json", {"status": "completed"})
        with self.assertRaises(FileNotFoundError):
            matched.require_report_release(root)
        write_json(root / "RELEASED.json", {"foreign_gpu_pids": [123]})
        with self.assertRaisesRegex(ValueError, "release"):
            matched.require_report_release(root)
        write_json(root / "RELEASED.json", {"foreign_gpu_pids": []})
        matched.require_report_release(root)
        write_json(root / "COMPLETE.json", {"status": "failed"})
        with self.assertRaisesRegex(ValueError, "finish"):
            matched.require_report_release(root)

    def test_exhausted_attempts_fail_before_gpu_allocation(self):
        output = Path(self.config["run"])
        write_json(output / "jobs" / "generate-official_target_specific-0" / "attempts.json", [{}, {}])
        with patch.object(matched, "Launcher") as launcher:
            with self.assertRaises(matched.evaluation.EvaluationAttemptsExhausted):
                matched.run(self.path)
            launcher.assert_not_called()

    def test_complete_report_reuses_sao_proposals_and_counts_all_eight_slots(self):
        requests = []
        servers = []
        launch = Mock()

        @contextmanager
        def vllm(**options):
            servers.append(options)
            yield ",".join(
                f"http://127.0.0.1:{options['port'] + 10 * index}/v1"
                for index in range(options["data_parallel"])
            )

        launch.vllm.side_effect = vllm

        class Jobs:
            def __init__(self, root, launch, concurrency):
                self.results = {}

            def add(self, request):
                requests.append(request)
                if request["kind"] == "generate":
                    valid = request["sample_index"] < 2
                    proposal = {"kind": "official"} if valid else None
                    result = {
                        **request, "valid": valid, "patch": proposal,
                        "patch_sha256": contract_digest(proposal) if valid else None,
                    }
                else:
                    result = {"scores": {
                        index: (
                            float(index % 2) if request["kind"] == "baseline"
                            else float(request["patch"]["kind"] == "sao")
                        )
                        for index in request["task_ids"]
                    }}
                self.results[request["name"]] = result

            def tick(self):
                pass

            def wait(self):
                return self.results

        release = {"foreign_gpu_pids": [], "filler_kind": "adaptive", "filler_unit_active": True}
        with (
            patch.object(matched, "Launcher", return_value=launch),
            patch.object(matched.evaluation, "Jobs", Jobs),
            patch.object(matched, "gpu_release_audit", return_value=release),
            patch.object(matched, "require_research_node"),
            patch.object(matched.threading, "Thread"),
        ):
            complete = matched.run(self.path)
            self.assertEqual(matched.run(self.path), complete)
        self.assertEqual(complete["status"], "complete")
        self.assertEqual(complete["system_accuracies"], {"official_target_specific": 0.375, "step15": 1.0})
        self.assertEqual(complete["sao_minus_official"]["candidate_minus_control"], 0.625)
        self.assertEqual(complete["valid_proposals"], {"official_target_specific": 2, "step15": 8})
        self.assertEqual(complete["samples_per_question"], 8)
        self.assertFalse(complete["sealed_test_opened"])
        generated = [request for request in requests if request["kind"] == "generate"]
        self.assertEqual(len(generated), 8)
        self.assertEqual([request["decode_seed"] for request in generated], list(range(5_000_045, 5_000_053)))
        self.assertEqual({request["model"] for request in generated}, {str(self.model)})
        report = read_json(Path(self.config["run"]) / "report" / "step15.json")
        self.assertEqual(len(report["results"]), 16)
        self.assertEqual(len(servers), 3)
        launch.close.assert_called_once()
        self.assertTrue((Path(self.config["run"]) / "RELEASED.json").is_file())


if __name__ == "__main__":
    unittest.main()
