"""Immutable baseline contract and hardware-matched per-trial compute accounting."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

from .common import file_sha, read_json, write_json

REQUIRED_CONTRACT = {
    "hardware", "gpu_count", "budget_gpu_seconds", "initial_actor_sha256",
    "data_manifest_sha256", "evaluation_protocol", "baseline_hyperparameters",
    "training_seed", "selection_metric", "formal_baseline_runs",
}
OPTIONAL_CONTRACT = {"initial_editor_sha256"}


def initial_role_models(spec: dict) -> dict[str, str]:
    solver = Path(spec["model"]).resolve()
    editor = Path(spec.get("editor_model", spec["model"])).resolve()
    if (spec.get("method") == "central" or spec.get("central_critic")) and editor != solver:
        raise ValueError("Shared-critic transfer with distinct role initializations is not validated")
    for model in {solver, editor}:
        if not model.is_dir() or not list(model.glob("*.safetensors")):
            raise FileNotFoundError(f"Initial role model has no local safetensors checkpoint: {model}")
    return {"solver": str(solver), "editor": str(editor)}


def _seeds(contract: dict) -> list:
    """A contract may lock one seed or a set of paired seeds (each arm runs every seed)."""
    seed = contract["training_seed"]
    return list(seed) if isinstance(seed, list) else [seed]


def contract_digest(contract: dict) -> str:
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def freeze_baseline(path: Path, contract: dict) -> str:
    if not REQUIRED_CONTRACT.issubset(contract) or set(contract) - REQUIRED_CONTRACT - OPTIONAL_CONTRACT:
        raise ValueError("Baseline contract is incomplete or has undeclared fields")
    if (
        contract["selection_metric"] != "dev_report_accuracy_at_fixed_sample_budget"
        or contract["gpu_count"] < 1 or contract["budget_gpu_seconds"] <= 0
        or not all(type(seed) is int for seed in _seeds(contract))
        or contract["formal_baseline_runs"] != len(_seeds(contract))
    ):
        raise ValueError("Invalid fixed-compute performance protocol")
    digest = contract_digest(contract)
    path.parent.mkdir(parents=True, exist_ok=True)
    # O_EXCL prevents a later trial from silently replacing the control.
    with path.open("x") as stream:
        json.dump({"contract": contract, "sha256": digest}, stream, indent=2)
        stream.write("\n")
    return digest


def load_contract(path: Path) -> tuple[dict, str]:
    record = read_json(path)
    digest = contract_digest(record["contract"])
    if digest != record["sha256"]:
        raise ValueError("Frozen baseline contract was modified")
    return record["contract"], digest


def claim_formal_baseline(contract_path: Path, run: Path, seed: int) -> Path:
    """Permit exactly one formal control per predeclared seed."""
    contract, digest = load_contract(contract_path)
    if seed not in _seeds(contract):
        raise ValueError("Formal baseline seed is not predeclared")
    claim = contract_path.with_name(contract_path.stem + f".formal-claim-seed{seed}.json")
    with claim.open("x") as stream:
        json.dump({"contract_sha256": digest, "seed": seed,
                   "run": str(run.resolve())}, stream, indent=2)
        stream.write("\n")
    return claim


def initial_model_digest(model: Path) -> str:
    files = sorted(p for p in model.iterdir() if p.is_file() and p.suffix in {
        ".safetensors", ".json", ".txt", ".jinja",
    } and p.name != "provenance.json")
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError("Initial model has no safetensors weights")
    return contract_digest({p.name: file_sha(p) for p in files})


def validate_trial(contract_path: Path, spec: dict) -> str:
    contract, digest = load_contract(contract_path)
    models = initial_role_models(spec)
    solver_digest = initial_model_digest(Path(models["solver"]))
    if solver_digest != contract["initial_actor_sha256"]:
        raise ValueError("Trial does not start from the frozen initial weights/tokenizer")
    editor_digest = (solver_digest if models["editor"] == models["solver"]
                     else initial_model_digest(Path(models["editor"])))
    if editor_digest != contract.get("initial_editor_sha256", contract["initial_actor_sha256"]):
        raise ValueError("Trial editor does not start from the frozen initial weights/tokenizer")
    data = Path(spec["data"])
    if file_sha(data / "manifest.json") != contract["data_manifest_sha256"]:
        raise ValueError("Trial data protocol differs from the frozen control")
    splits = read_json(data / "manifest.json")["splits"]
    if set(splits) != {"train", "reward", "dev", "test"}:
        raise ValueError("Frozen dataset requires all four declared splits")
    for split, record in splits.items():
        if file_sha(data / f"{split}.json") != record["sha256"]:
            raise ValueError(f"Frozen data file was modified: {split}")
    if spec["seed"] not in _seeds(contract):
        raise ValueError("Trial training seed differs from the frozen control")
    protocol = contract["evaluation_protocol"]
    if protocol["split"] != "test" or protocol["samples"] != 8:
        raise ValueError("Unsupported frozen held-out evaluation protocol")
    for key, value in protocol.items():
        if key not in {"split", "samples"} and spec.get(key) != value:
            raise ValueError(f"Trial violates frozen evaluation/budget setting: {key}")
    if spec.get("formal_baseline"):
        if spec["method"] != "independent":
            raise ValueError("Only independent SAO can claim the formal baseline")
        for key, value in contract["baseline_hyperparameters"].items():
            if spec.get(key) != value:
                raise ValueError(f"Formal baseline differs from tuned setting: {key}")
    return digest


class ComputeBudget:
    """Charge reserved identical GPUs, including generation, waits, and saving.

    All H1/W/H2 activity, including development reward evaluation, is charged.
    Final held-out evaluation runs separately with the identical frozen protocol.
    """

    def __init__(self, contract_path: Path, output: Path, *, hardware: str,
                 gpu_count: int, clock=time.monotonic, reference_run: Path | None = None,
                 trial_seed: int | None = None):
        self.contract, self.digest = load_contract(contract_path)
        if (hardware, gpu_count) != (self.contract["hardware"], self.contract["gpu_count"]):
            raise ValueError("Trial hardware allocation differs from the frozen baseline")
        if output.exists():
            raise FileExistsError("Use a fresh compute ledger, not a reset trial budget")
        self.limit = self.contract["budget_gpu_seconds"]
        self.reference = None
        self.reference_gpu_seconds = None
        if reference_run is not None:
            complete = read_json(reference_run / "COMPLETE.json")
            frozen = read_json(reference_run / "FROZEN.json")
            ledger = read_json(reference_run / "compute.json")
            if (complete["method"] != "independent" or not frozen["spec"]["formal_baseline"]
                    or trial_seed is None or frozen["spec"]["seed"] != trial_seed
                    or ledger["contract_sha256"] != self.digest or not ledger["within_budget"]
                    or ledger["events"][-1]["phase"] != "training_frozen"):
                raise ValueError("Reference must be a completed, frozen formal control")
            # The matched quantity is the locked sample budget (validate_trial); GPU time is
            # reported against the control, not used to truncate a method that trains longer.
            self.reference = str(reference_run.resolve())
            self.reference_gpu_seconds = ledger["spent_gpu_seconds"]
        self.output = output
        self.clock = clock
        self.start = clock()
        self.gpu_count = gpu_count
        self.events = []
        self.record("start")

    @property
    def spent(self) -> float:
        return (self.clock() - self.start) * self.gpu_count

    @property
    def remaining(self) -> float:
        return max(0.0, self.limit - self.spent)

    def record(self, phase: str) -> None:
        used = self.spent
        self.events.append({"phase": phase, "gpu_seconds": used})
        write_json(self.output, {
            "contract_sha256": self.digest, "accounting": "reserved_identical_gpu_seconds",
            "budget_gpu_seconds": self.contract["budget_gpu_seconds"],
            "effective_budget_gpu_seconds": self.limit,
            "reference_formal_run": self.reference,
            "reference_gpu_seconds": self.reference_gpu_seconds,
            "spent_gpu_seconds": used, "within_budget": used <= self.limit,
            "events": self.events,
            "tuning_cost_is_separate": True,
        })

    def check(self) -> None:
        if self.remaining <= 0:
            self.record("budget_exhausted")
            raise TimeoutError("Frozen baseline compute budget exhausted")
