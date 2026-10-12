from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/sao/vendor/AReaL"))
from sao_bridge.joint_continuation import convergence_status, handoff, optimizer_fingerprint
from sao_bridge.common import read_json


def test_real_native_dcp_restores_adam_and_next_update(tmp_path):
    from areal.engine.fsdp_engine import FSDPEngine
    engine = object.__new__(FSDPEngine)
    engine.model = torch.nn.Linear(3, 2)
    engine.optimizer = torch.optim.Adam(engine.model.parameters(), lr=0.01)
    x = torch.tensor([[1., 2., 3.]])

    def update():
        engine.optimizer.zero_grad()
        engine.model(x).square().sum().backward()
        engine.optimizer.step()

    for _ in range(3):
        update()
    before = optimizer_fingerprint(engine)
    engine._save_to_dcp(str(tmp_path / "state"), True)
    update()
    expected = optimizer_fingerprint(engine)
    engine._load_from_dcp(str(tmp_path / "state"), True)
    assert optimizer_fingerprint(engine) == before
    update()
    assert optimizer_fingerprint(engine) == expected


def test_native_rank_shards_preserve_lazy_unused_parameters(tmp_path):
    import torch.distributed as dist
    from areal.engine.fsdp_engine import FSDPEngine

    assert not dist.is_initialized()
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    try:
        engine = object.__new__(FSDPEngine)
        engine.model = torch.nn.ModuleDict({"text": torch.nn.Linear(3, 2), "visual": torch.nn.Linear(3, 2)})
        engine.optimizer = torch.optim.Adam(engine.model.parameters(), lr=0.01)
        engine.world_size = 1
        engine._cpu_group = dist.group.WORLD
        engine._initialized = True

        def update():
            engine.optimizer.zero_grad()
            engine.model["text"](torch.ones(1, 3)).square().sum().backward()
            engine.optimizer.step()

        update()
        original = optimizer_fingerprint(engine)
        engine._save_to_dcp(str(tmp_path / "saved"), False)
        engine._save_optimizer_state(str(tmp_path / "saved"))
        update()
        expected = optimizer_fingerprint(engine)
        engine._load_from_dcp(str(tmp_path / "saved"), False)
        engine._load_optimizer_state(str(tmp_path / "saved"))
        assert optimizer_fingerprint(engine) == original
        assert all(p not in engine.optimizer.state for p in engine.model["visual"].parameters())
        update()
        assert optimizer_fingerprint(engine) == expected
    finally:
        dist.destroy_process_group()


def test_handoff_is_opt_in_and_preserves_both_engines(tmp_path):
    trainer = Mock()
    handoff(trainer, {}, tmp_path, loading=True)
    assert not trainer.mock_calls
    cfg = SimpleNamespace(optimizer=SimpleNamespace(lr_scheduler_type="constant", warmup_steps_proportion=0))
    trainer.actor.config = trainer.critic.config = cfg
    spec = {"preserve_optimizer_state": True, "joint_root": "/joint"}
    handoff(trainer, spec, tmp_path, loading=True)
    assert read_json(tmp_path / "OPTIMIZER_LOAD.json")["mode"] == "initial_fresh_adam"
    handoff(trainer, spec, tmp_path, loading=False)
    for name in ("actor", "critic"):
        meta = getattr(trainer, name).save.call_args.kwargs["meta"]
        assert meta.with_optim and meta.weight_format == "dcp"
    spec["optimizer_state_input"] = {"actor": "/a", "critic": "/c"}
    handoff(trainer, spec, tmp_path, loading=True)
    assert trainer.actor.load.call_args.kwargs["meta"].path == "/a"
    assert trainer.critic.load.call_args.kwargs["meta"].path == "/c"
    cfg.optimizer.lr_scheduler_type = "cosine"
    with pytest.raises(ValueError, match="constant"):
        handoff(trainer, spec, tmp_path, loading=True)


def test_plateau_is_not_short_run_or_decline_or_oscillation():
    def points(values):
        return [{"rounds": 48 * (i + 1), "accuracy": x} for i, x in enumerate(values)]
    assert not convergence_status(points([.935, .94, .935, .94]))["operational_plateau"]
    assert convergence_status(points([.935, .94, .935, .94, .94]))["operational_plateau"]
    assert not convergence_status(points([.935, .92, .88, .84, .80]))["operational_plateau"]
    assert not convergence_status(points([.935, .80, .94, .80, .94]))["operational_plateau"]
    assert not convergence_status(points([.935, .94, .945, .95, .955]))["operational_plateau"]
    with pytest.raises(ValueError):
        convergence_status([{"rounds": 48, "accuracy": float("nan")}])


def test_curve_evaluation_rejects_undeclared_checkpoint(tmp_path):
    from sao_bridge.common import write_json
    from sao_bridge.joint_campaign import evaluate
    write_json(tmp_path / "protocol.json", {"curve_rounds": [48, 96]})
    with pytest.raises(ValueError, match="exact label"):
        evaluate(tmp_path, "matched", checkpoint_rounds=48)
    with pytest.raises(ValueError, match="not declared"):
        evaluate(tmp_path, "round0012", checkpoint_rounds=12)


def test_peak_resume_plateau_requires_192_new_rounds():
    points = [{"rounds": 144 + 48 * i, "accuracy": 0.97} for i in range(4)]
    assert not convergence_status(points)["operational_plateau"]
    points.append({"rounds": 336, "accuracy": 0.97})
    assert convergence_status(points)["operational_plateau"]
