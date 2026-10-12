from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.budget import (
    ComputeBudget, claim_formal_baseline, freeze_baseline, initial_model_digest, initial_role_models,
    load_contract, validate_trial,
)
from sao_bridge.common import file_sha, read_json, write_json
from sao_bridge.freeze import SAMPLE_BUDGET_KEYS, contract_from_spec


class BudgetTests(unittest.TestCase):
    def test_separate_role_models_are_locked_and_legacy_contract_cannot_be_bypassed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            solver, editor, data = root / "solver", root / "editor", root / "data"
            for model in (solver, editor):
                model.mkdir()
                (model / "model.safetensors").write_bytes(model.name.encode())
            splits = {}
            for split in ("train", "reward", "dev", "test"):
                write_json(data / f"{split}.json", [])
                splits[split] = {"sha256": file_sha(data / f"{split}.json")}
            write_json(data / "manifest.json", {"splits": splits})
            spec = dict(model=str(solver), editor_model=str(editor), data=str(data), seed=42,
                        method="independent", formal_baseline=True, lr=1e-6)
            self.assertEqual(initial_role_models(spec), {"solver": str(solver), "editor": str(editor)})
            self.assertEqual(initial_role_models({"model": str(solver)})["editor"], str(solver))
            with self.assertRaisesRegex(ValueError, "Shared-critic"):
                initial_role_models({**spec, "central_critic": True})
            contract = {
                **self.contract(), "initial_actor_sha256": initial_model_digest(solver),
                "data_manifest_sha256": file_sha(data / "manifest.json"),
                "evaluation_protocol": {"split": "test", "samples": 8},
            }
            legacy = root / "legacy.json"
            freeze_baseline(legacy, contract)
            with self.assertRaisesRegex(ValueError, "editor"):
                validate_trial(legacy, spec)
            current = root / "current.json"
            freeze_baseline(current, {**contract, "initial_editor_sha256": initial_model_digest(editor)})
            validate_trial(current, spec)
            validate_trial(current, {**spec, "formal_baseline": False, "method": "ma_sao"})
            tuned = {**{key: 1 for key in SAMPLE_BUDGET_KEYS}, **spec,
                     "solver_lr": 1e-6, "editor_lr": 2e-6, "editor_temperature": 0.7,
                     "weight_update_mode": "xccl"}
            generated = contract_from_spec(tuned, 1.0)
            self.assertEqual(generated["initial_actor_sha256"], initial_model_digest(solver))
            self.assertEqual(generated["initial_editor_sha256"], initial_model_digest(editor))
            with self.assertRaisesRegex(ValueError, "editor"):
                validate_trial(current, {**spec, "editor_model": str(solver)})
            (editor / "model.safetensors").write_bytes(b"altered editor weights")
            with self.assertRaisesRegex(ValueError, "editor"):
                validate_trial(current, spec)

    def test_formal_run_verifies_initial_state_seed_and_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, data = root / "model", root / "data"
            model.mkdir()
            data.mkdir()
            (model / "model.safetensors").write_bytes(b"fixture weights")
            splits = {}
            for split in ("train", "reward", "dev", "test"):
                write_json(data / f"{split}.json", [])
                splits[split] = {"sha256": file_sha(data / f"{split}.json")}
            write_json(data / "manifest.json", {"splits": splits})
            contract = self.contract()
            contract.update(
                initial_actor_sha256=initial_model_digest(model),
                data_manifest_sha256=file_sha(data / "manifest.json"),
                evaluation_protocol={"split": "test", "samples": 8, "solver_tokens": 4096},
            )
            path = root / "contract.json"
            freeze_baseline(path, contract)
            spec = dict(model=str(model), data=str(data), seed=42, solver_tokens=4096,
                        formal_baseline=True, method="independent", lr=1e-6)
            validate_trial(path, spec)
            for change in ({"seed": 43}, {"solver_tokens": 8192}, {"lr": 2e-6},
                           {"method": "ma_sao"}):
                with self.assertRaises(ValueError):
                    validate_trial(path, {**spec, **change})
            validate_trial(path, {**spec, "formal_baseline": False, "method": "ma_sao", "lr": 2e-6})
            write_json(data / "train.json", ["altered"])
            with self.assertRaisesRegex(ValueError, "data file"):
                validate_trial(path, spec)
            write_json(data / "train.json", [])
            (model / "model.safetensors").write_bytes(b"different weights")
            with self.assertRaisesRegex(ValueError, "initial"):
                validate_trial(path, spec)

    def contract(self):
        return dict(
            hardware="NVIDIA H200", gpu_count=8, budget_gpu_seconds=800,
            initial_actor_sha256="model", data_manifest_sha256="data",
            evaluation_protocol={"samples": 8, "split": "held_out"},
            baseline_hyperparameters={"lr": 1e-6}, training_seed=42, formal_baseline_runs=1,
            selection_metric="dev_report_accuracy_at_fixed_sample_budget",
        )

    def test_baseline_is_immutable_and_tampering_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            freeze_baseline(path, self.contract())
            with self.assertRaises(FileExistsError):
                freeze_baseline(path, self.contract())
            record = read_json(path)
            record["contract"]["budget_gpu_seconds"] *= 2
            write_json(path, record)
            with self.assertRaisesRegex(ValueError, "modified"):
                load_contract(path)

    def test_charge_all_reserved_gpu_time_and_enforce_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            freeze_baseline(root / "baseline.json", self.contract())
            now = [1000.0]
            budget = ComputeBudget(root / "baseline.json", root / "ledger.json",
                                   hardware="NVIDIA H200", gpu_count=8,
                                   clock=lambda: now[0])
            now[0] += 50
            self.assertEqual(budget.spent, 400)
            self.assertEqual(budget.remaining, 400)
            budget.record("h1_development_reward")
            now[0] += 51
            with self.assertRaises(TimeoutError):
                budget.check()
            self.assertFalse(read_json(root / "ledger.json")["within_budget"])

    def test_reject_different_hardware(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            freeze_baseline(root / "baseline.json", self.contract())
            with self.assertRaises(ValueError):
                ComputeBudget(root / "baseline.json", root / "ledger.json",
                              hardware="A100", gpu_count=8)

    def test_reference_compute_is_reported_not_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "contract.json"
            digest = freeze_baseline(path, self.contract())
            reference = root / "baseline"
            write_json(reference / "COMPLETE.json", {"method": "independent"})
            write_json(reference / "FROZEN.json", {"spec": {"formal_baseline": True, "seed": 42}})
            write_json(reference / "compute.json", {
                "contract_sha256": digest, "within_budget": True,
                "spent_gpu_seconds": 400, "events": [{"phase": "training_frozen"}],
            })
            now = [0.0]
            budget = ComputeBudget(
                path, root / "ma.json", hardware="NVIDIA H200", gpu_count=8,
                clock=lambda: now[0], reference_run=reference, trial_seed=42,
            )
            now[0] = 60
            budget.check()
            budget.record("h1")
            ledger = read_json(root / "ma.json")
            self.assertEqual(ledger["reference_gpu_seconds"], 400)
            self.assertEqual(ledger["spent_gpu_seconds"], 480)

    def test_only_one_formal_baseline_can_be_claimed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = root / "baseline.json"
            freeze_baseline(contract, self.contract())
            claim_formal_baseline(contract, root / "formal_control", 42)
            with self.assertRaises(FileExistsError):
                claim_formal_baseline(contract, root / "second_control", 42)


if __name__ == "__main__":
    unittest.main()
