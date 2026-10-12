import importlib.util
from pathlib import Path

from sao_bridge.common import read_json, write_json
from sao_bridge.joint_anchor import build_rows
from sao_bridge.joint_schedule import actor_updates, total_consumed
from test_joint_schedule import original_rows, spec_for


def test_preparation_keeps_input_order_and_activates_both_roles(tmp_path, monkeypatch):
    parent, root = tmp_path / "parent", tmp_path / "candidate"
    parent.mkdir(); root.mkdir()
    driver_path = Path(__file__).resolve().parents[1] / "runs/alfworld_ma_simultaneous_convergence_v1/driver.py"
    module_spec = importlib.util.spec_from_file_location("simultaneous_campaign", driver_path)
    module = importlib.util.module_from_spec(module_spec)
    design = {"parent": str(parent), "source": str(tmp_path / "source")}
    from sao_bridge import common
    original_read = common.read_json
    monkeypatch.setattr(common, "read_json", lambda path: design if Path(path) == driver_path.parent / "DESIGN.json"
                        else original_read(path))
    module_spec.loader.exec_module(module)
    module.ROOT = root
    write_json(root / "STARTED.json", {"deadline_unix": 9999999999})
    write_json(root / "DESIGN.json", design)
    write_json(root / "MEMORY_COMPLETE.json", {"status": "passed"})
    validation = tmp_path / "kernel.json"; write_json(validation, {})
    write_json(parent / "INITIAL_STATE.json", {})
    pair = {role: {"actor": f"/weights/{role}", "critic": f"/critics/{role}"} for role in ("editor", "solver")}
    write_json(parent / "matched_COMPLETE.json", {"selection": {"pair": pair}})
    for variant in ("joint_harness", "no_patch"):
        write_json(parent / f"matched_{variant}_report.json", {})
    source = original_rows()[:256]
    write_json(parent / "schedule_source_data.json", source)
    write_json(parent / "protocol.json", {"budget": {}, "anchor_mode": "sampled_control"})
    for role in ("editor", "solver"):
        spec = spec_for(tmp_path, role)
        spec.update(training_kernel_validation=str(validation), anchor_mode="sampled_control",
                    anchor_harness="/old/anchor", anchor_harness_sha256="old", reference_model=f"/reference/{role}")
        write_json(parent / f"{role}_spec.json", spec)
    monkeypatch.setattr(module, "configure", lambda spec, run: {
        "actor": {"optimizer": {"lr": 5e-7}},
        "ref": {"optimizer_dtype": "bfloat16", "path": spec["reference_model"]},
    })
    block = module.prepare_block(1, parent, "/incumbent.json")
    rows = read_json(block / "data.json")
    old_order = build_rows(read_json(block / "schedule_source_data.json"), cycles=32, batch_size=8)
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
        assert spec["model"] == pair[role]["actor"]
    assert read_json(block / "protocol.json")["schedule"] == "simultaneous"
