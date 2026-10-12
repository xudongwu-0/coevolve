import importlib.util
from pathlib import Path

import pytest

from sao_bridge.common import read_json, write_json
from sao_bridge.joint_anchor import build_rows
from sao_bridge.joint_schedule import actor_updates, total_consumed
from test_joint_schedule import original_rows, spec_for


def save_native(parent, role, actor_steps):
    optimizer = parent / role / "optimizer"
    paths = {component: str(optimizer / "optimizer_state" / component) for component in ("actor", "critic")}
    write_json(optimizer / "OPTIMIZER_SAVED.json", {"mode": "native_dcp_with_adam", "paths": paths})
    for component, path in paths.items():
        steps = actor_steps * (2 if component == "critic" else 1)
        for rank in range(4):
            write_json(Path(path) / f"adam-rank{rank}.json", {"min_adam_step": steps, "max_adam_step": steps})
        (Path(path) / ".metadata").write_bytes(b"native-metadata")
    return paths


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    parent, template, root = (tmp_path / name for name in ("peak", "template", "candidate"))
    for path in (parent, template, root):
        path.mkdir()
    driver_path = Path(__file__).resolve().parents[1] / "runs/alfworld_ma_simultaneous_from97_v1/driver.py"
    module_spec = importlib.util.spec_from_file_location("peak_campaign", driver_path)
    module = importlib.util.module_from_spec(module_spec)
    design = {
        "parent": str(parent), "parent_endpoint": "round0096", "source": str(tmp_path / "source"),
        "input_template_parent": str(template), "initial_source_offset": 768, "inherited_lineage_consumed": 1920,
        "inherited_optimizer_steps": {
            "editor": {"actor": 64, "critic": 128}, "solver": {"actor": 96, "critic": 192},
        },
    }
    from sao_bridge import common
    original_read = common.read_json
    monkeypatch.setattr(common, "read_json", lambda path: design if Path(path) == driver_path.parent / "DESIGN.json"
                        else original_read(path))
    module_spec.loader.exec_module(module)
    module.ROOT = root
    write_json(root / "STARTED.json", {"deadline_unix": 9999999999})
    write_json(root / "DESIGN.json", design)
    write_json(root / "MEMORY_COMPLETE.json", {"status": "passed"})
    validation = tmp_path / "kernel.json"
    write_json(validation, {})
    write_json(parent / "INITIAL_STATE.json", {"initialization": "old starting-point description"})
    pair = {role: {"actor": f"/peak/{role}/actor", "critic": f"/peak/{role}/critic"} for role in ("editor", "solver")}
    write_json(parent / "round0096_COMPLETE.json", {"selection": {"pair": pair}})
    for variant in ("joint_harness", "no_patch"):
        write_json(parent / f"round0096_{variant}_report.json", {"accuracy": .97 if variant == "joint_harness" else .925})
    write_json(template / "schedule_source_data.json", original_rows()[:256])
    write_json(template / "protocol.json", {"budget": {}, "anchor_mode": "sampled_control"})
    for role in ("editor", "solver"):
        spec = spec_for(tmp_path, role)
        spec.update(training_kernel_validation=str(validation), anchor_mode="sampled_control",
                    anchor_harness="/old/anchor", anchor_harness_sha256="old", reference_model=f"/reference/{role}")
        write_json(parent / f"{role}_spec.json", spec)
        save_native(parent, role, design["inherited_optimizer_steps"][role]["actor"])
    monkeypatch.setattr(module, "configure", lambda spec, run: {
        "actor": {"optimizer": {"lr": 5e-7}},
        "ref": {"optimizer_dtype": "bfloat16", "path": spec["reference_model"]},
    })
    return module, parent, root, pair


def test_peak_resume_uses_next_inputs_both_roles_and_original_references(campaign):
    module, parent, _, pair = campaign
    block = module.prepare_block(1, parent, "/peak/harness.json")
    rows = read_json(block / "data.json")
    source = read_json(block / "schedule_source_data.json")
    assert source[0]["id"] == "continuation-source-000768-task48"
    assert source[-1]["id"] == "continuation-source-001279-task79"
    old_order = build_rows(source, cycles=32, batch_size=8)
    assert len(rows) == 768 and len({row["id"] for row in rows}) == 768
    for key in ("task_id", "messages", "task_context", "critic_evidence", "joint_version", "joint_slot"):
        assert [row[key] for row in rows] == [row[key] for row in old_order]
    assert all("anchor_round" not in row for row in rows)
    for role in ("editor", "solver"):
        spec = read_json(block / f"{role}_spec.json")
        assert spec["schedule"] == "simultaneous" and "anchor_mode" not in spec
        assert all(actor_updates(spec, step) for step in range(96))
        assert total_consumed(spec, 96) == 1536
        assert spec["reference_model"] == f"/reference/{role}"
        assert (spec["model"], spec["critic_model"]) == (pair[role]["actor"], pair[role]["critic"])
        assert spec["optimizer_state_input"] == read_json(parent / role / "optimizer/OPTIMIZER_SAVED.json")["paths"]
        assert spec["inherited_optimizer_steps"] == module.DESIGN["inherited_optimizer_steps"][role]
    assert read_json(block / "protocol.json")["budget"]["common_executed_trajectories"] == 1920
    assert read_json(block / "INITIAL_STATE.json")["harness"] == "/peak/harness.json"


def test_second_block_keeps_peak_offsets_and_continues_source_stream(campaign):
    module, parent, _, pair = campaign
    first = module.prepare_block(1, parent, "/peak/harness.json")
    write_json(first / "round0096_COMPLETE.json", {"selection": {"pair": pair}})
    for variant in ("joint_harness", "no_patch"):
        write_json(first / f"round0096_{variant}_report.json", {})
    for role in ("editor", "solver"):
        save_native(first, role, module.DESIGN["inherited_optimizer_steps"][role]["actor"] + 96)
    second = module.prepare_block(2, first, "/peak/harness.json")
    assert read_json(second / "schedule_source_data.json")[0]["id"] == "continuation-source-001280-task80"
    for role in ("editor", "solver"):
        spec = read_json(second / f"{role}_spec.json")
        assert spec["optimizer_state_input"] == read_json(first / role / "optimizer/OPTIMIZER_SAVED.json")["paths"]
        assert spec["inherited_optimizer_steps"] == module.DESIGN["inherited_optimizer_steps"][role]
    assert read_json(second / "protocol.json")["budget"]["common_executed_trajectories"] == 3456


def test_peak_resume_refuses_missing_native_critic(campaign):
    module, parent, _, _ = campaign
    path = parent / "editor/optimizer/OPTIMIZER_SAVED.json"
    saved = read_json(path)
    del saved["paths"]["critic"]
    write_json(path, saved)
    with pytest.raises(ValueError, match="complete native Actor/Critic"):
        module.prepare_block(1, parent, "/peak/harness.json")
