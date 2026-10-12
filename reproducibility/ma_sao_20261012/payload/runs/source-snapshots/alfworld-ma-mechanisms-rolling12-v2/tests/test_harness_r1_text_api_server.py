"""Mock-only wrapper regressions. Parent runs with CUDA_VISIBLE_DEVICES=''.

No test imports vLLM or Torch, unmasks physical devices, or starts a process.
"""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "whale_local"))


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.delenv("HARNESS_R1_REGISTER_VLLM_TEXT", raising=False)
    return importlib.import_module("harness_r1_searchqa.text_api_server")


@pytest.fixture
def text_model(tmp_path):
    model = tmp_path / "text-checkpoint"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5_text", "architectures": ["Qwen3_5ForCausalLM"]}))
    return model


def test_native_api_flags_and_forwarded_policy(server, text_model):
    args, argv = server.prepare_argv([
        "--model", str(text_model), "--host", "127.0.0.1",
        "--dtype", "bfloat16", "--max-model-len", "32768",
        "--tensor-parallel-size", "2", "--served-model-name", "engineer-control",
        "--port", "18990", "--gpu-memory-utilization", "0.8"])
    assert args.tensor_parallel_size == 2
    assert argv[argv.index("--gdn-prefill-backend") + 1] == "triton"
    assert argv[argv.index("--worker-extension-cls") + 1] == server.WORKER_EXTENSION
    assert argv[argv.index("--model-impl") + 1] == "vllm"
    assert argv[argv.index("--served-model-name") + 1] == "engineer-control"
    assert argv[argv.index("--port") + 1] == "18990"


@pytest.mark.parametrize("option,value", [
    ("--gdn-prefill-backend", "flashinfer"),
    ("--worker-extension-cls", "foreign.worker.Extension"),
    ("--model-impl", "transformers"),
    ("--host", "0.0.0.0"),
])
def test_unsafe_or_non_native_overrides_rejected(server, text_model, option, value):
    with pytest.raises(SystemExit):
        server.prepare_argv(["--model", str(text_model), "--max-model-len", "32768",
                             option, value])


def test_raw_vlm_rejected_without_importing_vllm(server, text_model):
    (text_model / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"]}))
    with pytest.raises(ValueError, match="text-only"):
        server.prepare_argv(["--model", str(text_model), "--max-model-len", "32768"])


def test_gpu_selection_validation_does_not_mutate_visibility(server):
    before = os.environ["CUDA_VISIBLE_DEVICES"]
    server.validate_gpu_selection("GPU-mocked-a,GPU-mocked-b", 2)
    for selection, tp in [("", 1), (None, 1), ("-1", 1),
                           ("GPU-mocked-a,GPU-mocked-a", 2), ("GPU-mocked-a", 2)]:
        with pytest.raises(ValueError):
            server.validate_gpu_selection(selection, tp)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == before == ""


def test_parent_registration_precedes_standard_api_without_gpu_imports(
        server, text_model, monkeypatch):
    events = []
    original_argv = sys.argv

    def check_selection(visible, tp):
        assert visible == "" and tp == 1
        events.append("audited-selection-mocked")

    def registration():
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
        assert os.environ["HARNESS_R1_REGISTER_VLLM_TEXT"] == "1"
        events.append("native-registration-mocked")

    def api(module, **kwargs):
        assert events == ["audited-selection-mocked", "native-registration-mocked"]
        assert module == server.API_MODULE
        assert kwargs == {"run_name": "__main__", "alter_sys": False}
        assert os.environ["VLLM_PLUGINS"] == ""
        assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
        assert server.WORKER_EXTENSION in sys.argv
        events.append("standard-api-mocked")

    monkeypatch.setattr(server, "validate_gpu_selection", check_selection)
    monkeypatch.setattr(server, "_register_native_model", registration)
    monkeypatch.setattr(server.runpy, "run_module", api)
    for key in ("HARNESS_R1_REGISTER_VLLM_TEXT", "VLLM_PLUGINS",
                "VLLM_WORKER_MULTIPROC_METHOD"):
        monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv(key, "")
    server.main(["--model", str(text_model), "--max-model-len", "32768",
                 "--served-model-name", "engineer-control"])
    assert events[-1] == "standard-api-mocked"
    assert sys.argv is original_argv


def test_spawn_import_registration_uses_existing_native_hook(server, monkeypatch):
    calls = []
    fake_workflow = SimpleNamespace(register_text_engine=lambda: calls.append("registered"))
    monkeypatch.setitem(sys.modules, "harness_r1_searchqa.workflow", fake_workflow)
    monkeypatch.setenv("HARNESS_R1_REGISTER_VLLM_TEXT", "1")
    importlib.reload(server)
    assert calls == ["registered"]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
