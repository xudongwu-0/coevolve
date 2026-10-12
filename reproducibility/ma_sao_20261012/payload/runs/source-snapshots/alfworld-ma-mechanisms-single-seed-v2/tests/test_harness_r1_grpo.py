"""CPU-only algorithm/orchestration tests; never start a real Engineer job."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "whale_local"))

from harness_r1_searchqa.grpo_objective import (
    group_advantages, grpo_loss, response_logprobs,
)
from harness_r1_searchqa.grpo_train import (
    _generate, _load_fsdp, _logprobs, packet_batch, train_update, validate_config,
)
from harness_r1_searchqa.workflow import (
    atomic_json, checkpoint_identity, register_text_engine, thinking_prompt,
    validate_rewards, validate_rollouts,
)


def config():
    return json.loads((ROOT / "configs/searchqa_engineer_grpo.json").read_text())


def upstream_function(relative_path, name):
    source = ROOT / "third_party/harness-r1-upstream/code/Relax" / relative_path
    node = next(node for node in ast.parse(source.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"torch": torch, "Any": Any}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[name]


def test_pinned_relax_policy_tis_and_sample_mean():
    torch.manual_seed(7)
    new = torch.randn(11, requires_grad=True)
    old, rollout = torch.randn(11), torch.randn(11)
    advantage = torch.tensor(-0.37)
    compute = upstream_function("relax/utils/training/ppo_utils.py", "compute_policy_loss")
    tis = upstream_function("relax/backends/megatron/loss.py", "vanilla_tis_function")
    pg, _ = compute(old - new, advantage, .2, .28)
    corrected, _, _ = tis(
        SimpleNamespace(tis_clip=2., tis_clip_low=0.),
        pg_loss=pg, train_log_probs=[old], rollout_log_probs=[rollout],
        loss_masks=[torch.ones_like(old)])
    actual, _ = grpo_loss(new, old, rollout, advantage)
    torch.testing.assert_close(actual, corrected.mean())
    expected_gradient = torch.autograd.grad(corrected.mean(), new, retain_graph=True)[0]
    torch.testing.assert_close(torch.autograd.grad(actual, new)[0], expected_gradient)


def test_group_unbiased_std_and_constant_groups():
    rewards = torch.tensor([0., 0., 0., 0., 0., 0., 0., 1.] + [.5] * 8)
    actual, constants = group_advantages(rewards)
    centered = rewards[:8] - rewards[:8].mean()
    torch.testing.assert_close(actual[:8], centered / (centered.std() + 1e-6))
    assert constants.tolist() == [False, True]
    assert actual[8:].eq(0).all()
    with pytest.raises(ValueError, match="nonfinite"):
        group_advantages(torch.tensor([float("nan")] * 8))
    with pytest.raises(ValueError, match="complete"):
        group_advantages(torch.ones(7))


def test_kl_fixed_reference_matches_relax():
    compute = upstream_function("relax/utils/training/ppo_utils.py", "compute_approx_kl")
    new = torch.tensor([-.1, -.7, -3.], requires_grad=True)
    old = new.detach().clone()
    ref = torch.tensor([-.9, -.8, -1.])
    loss, _ = grpo_loss(new, old, old, torch.tensor(0.),
                        reference_logprobs=ref, kl_coef=.03)
    expected = compute(new, ref, kl_loss_type="low_var_kl").mean() * .03
    torch.testing.assert_close(loss, expected)
    with pytest.raises(ValueError, match="reference required"):
        grpo_loss(new, old, old, torch.tensor(1.), kl_coef=.03)


def test_nonfinite_and_clipping():
    old = torch.zeros(2)
    new = torch.tensor([1., -1.], requires_grad=True)
    loss, _ = grpo_loss(new, old, old, torch.tensor(1.))
    torch.testing.assert_close(loss, -torch.tensor([1.28, torch.exp(torch.tensor(-1.))]).mean())
    loss.backward()
    assert new.grad[0] == 0
    with pytest.raises(ValueError, match="nonfinite"):
        grpo_loss(torch.tensor([float("inf"), 1.]), old, old, torch.tensor(1.))


def test_chunked_temperature_logprobs_gradient():
    assert not torch.cuda.is_available()
    logits = torch.randn(5, 7, requires_grad=True)
    tokens = torch.tensor([0, 1, 2, 3, 4])
    actual = response_logprobs(logits, tokens, chunk_size=2)
    expected = torch.log_softmax(logits / .7, -1).gather(1, tokens[:, None]).flatten()
    torch.testing.assert_close(actual, expected)
    expected_grad = torch.autograd.grad(expected.sum(), logits, retain_graph=True)[0]
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), logits)[0], expected_grad)


class ToyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(7, 5)
        self.head = torch.nn.Linear(5, 7)

    def forward(self, input_ids, use_cache=False, logits_to_keep=0):
        logits = self.head(self.embedding(input_ids))
        return SimpleNamespace(logits=logits[:, -logits_to_keep:])


class CountingAdamW(torch.optim.AdamW):
    calls = 0

    def step(self, *args, **kwargs):
        self.calls += 1
        return super().step(*args, **kwargs)


def toy_records(model):
    records = []
    for i in range(32):
        record = {"prompt_ids": [0, 1, 2], "response_ids": [3, 4],
                  "reward": float(i % 8 == 0)}
        record["rollout_logprobs"] = _logprobs(model, record, "cpu", .7).detach().tolist()
        records.append(record)
    return records


def test_four_optimizer_steps_finite_and_persistent_reload(tmp_path):
    torch.manual_seed(12)
    model = ToyLM()
    records = toy_records(model)
    optimizer = CountingAdamW(model.parameters(), lr=1e-6, betas=(.9, .98), weight_decay=.1)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    metrics = train_update(model, optimizer, records, config(), "cpu")
    assert metrics["optimizer_steps"] == optimizer.calls == 4
    assert metrics["constant_reward_groups"] == [False] * 4
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert any(not torch.equal(before[name], value) for name, value in model.state_dict().items())
    path = tmp_path / "tiny-actor.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "metrics": metrics}, path)
    saved = torch.load(path, weights_only=True)
    reloaded = ToyLM()
    reloaded.load_state_dict(saved["model"])
    reloaded_optimizer = CountingAdamW(reloaded.parameters(), lr=1e-6)
    reloaded_optimizer.load_state_dict(saved["optimizer"])
    for a, b in zip(model.parameters(), reloaded.parameters(), strict=True):
        torch.testing.assert_close(a, b)
    assert all(int(state["step"]) == 4 for state in reloaded_optimizer.state.values())
    torch.testing.assert_close(_logprobs(model, records[0], "cpu", .7),
                               _logprobs(reloaded, records[0], "cpu", .7))


def test_all_constant_groups_are_explicit_and_not_skipped():
    model = ToyLM()
    records = toy_records(model)
    for record in records:
        record["reward"] = 0.
    optimizer = CountingAdamW(model.parameters(), lr=1e-6, weight_decay=.1)
    result = train_update(model, optimizer, records, config(), "cpu")
    assert optimizer.calls == 4
    assert result["all_groups_constant"]
    assert all(metric["loss"] == 0 for metric in result["iterations"])


def test_exact_thinking_prefill_and_no_metadata_in_prompt():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {"tokenize": False, "add_generation_prompt": True,
                              "enable_thinking": True}
            assert "metadata" not in str(messages)
            return "<|im_start|>assistant\n<think>\n"

        def encode(self, text, add_special_tokens):
            assert not add_special_tokens
            return [1, 2, 3]

    assert thinking_prompt(Tokenizer(), [{"role": "user", "content": "public failure"}]) == [1, 2, 3]
    class BadTokenizer(Tokenizer):
        def apply_chat_template(self, *args, **kwargs):
            return "<|im_start|>assistant\n"
    with pytest.raises(ValueError, match="thinking prefill"):
        thinking_prompt(BadTokenizer(), [{"role": "user", "content": "public failure"}])


def test_delta_reward_and_infrastructure_failure():
    assert validate_rewards([{"score": 0., "delta_score": 0.,
                              "eval_status": "invalid_patch_treated_as_no_patch"}], 1) == [0.]
    with pytest.raises(RuntimeError, match="infrastructure"):
        validate_rewards([{"score": 0., "delta_score": 0., "eval_status": "http_error"}], 1)
    with pytest.raises(ValueError, match="without validity bonus"):
        validate_rewards([{"score": .1, "delta_score": 0., "eval_status": "ok"}], 1)


def test_rollout_alignment_and_stale_actor():
    rows = [{"metadata": {"train": i}} for i in range(4)]
    records = [{"group": i // 8, "metadata": rows[i // 8]["metadata"],
                "actor_sha256": "fresh", "prompt_ids": [1], "response_ids": [2],
                "rollout_logprobs": [-.1]} for i in range(32)]
    validate_rollouts(records, rows, "fresh")
    with pytest.raises(ValueError, match="stale"):
        validate_rollouts(records, rows, "old")
    records[0]["rollout_logprobs"] = []
    with pytest.raises(ValueError, match="misaligned"):
        validate_rollouts(records, rows, "fresh")


def test_actor_identity_stable_and_disk_sync_job(tmp_path, monkeypatch):
    actor = tmp_path / "actor"
    actor.mkdir()
    (actor / "model.safetensors").write_bytes(b"tiny checkpoint identity")
    (actor / "config.json").write_text("{}")
    identity = checkpoint_identity(actor)
    atomic_json(actor / "actor_identity.json", identity)
    assert checkpoint_identity(actor) == identity
    args = SimpleNamespace(output=tmp_path, rollout_gpus="", seed=42)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "_lazy_init",
        lambda *args, **kwargs: pytest.fail("CUDA initialization is forbidden in CPU validation"))
    monkeypatch.setenv("RANK", "0")
    captured = []
    def child(command, *, env, start_new_session):
        assert env["CUDA_VISIBLE_DEVICES"] == "" and "RANK" not in env
        assert env["VLLM_PLUGINS"] == ""
        assert start_new_session
        job = json.loads(Path(command[-1]).read_text())
        assert job["verify_only"]
        assert checkpoint_identity(Path(job["actor"]))["sha256"] == job["actor_sha256"]
        captured.append(job)
        atomic_json(Path(job["result"]), {"actor_sha256": job["actor_sha256"],
                                         "engine_reloaded_from_disk": True})
        return SimpleNamespace(wait=lambda timeout: 0)
    monkeypatch.setattr("harness_r1_searchqa.grpo_train.subprocess.Popen", child)
    _generate(args, config(), {}, [], actor, identity["sha256"], 1, verify_only=True)
    (actor / "model.safetensors").write_bytes(b"updated actor")
    updated = checkpoint_identity(actor)
    _generate(args, config(), {}, [], actor, updated["sha256"], 2, verify_only=True)
    assert captured[0]["actor_sha256"] != captured[1]["actor_sha256"]


def test_config_and_packet_selection():
    validate_config(config())
    assert config()["rollout_gdn_prefill_backend"] == "triton"
    wrong = config()
    wrong["optimizer_iterations"] = 1
    with pytest.raises(ValueError, match="optimizer_iterations"):
        validate_config(wrong)
    wrong = config()
    wrong["rollout_gdn_prefill_backend"] = "flashinfer"
    with pytest.raises(ValueError, match="rollout_gdn_prefill_backend"):
        validate_config(wrong)
    rows = [{"metadata": i} for i in range(5)]
    assert packet_batch(rows, 0, 42) == packet_batch(rows, 0, 42)
    assert len({row["metadata"] for row in packet_batch(rows, 1, 42)}) == 4


def test_native_text_engine_registration_is_lazy_and_idempotent(monkeypatch):
    registered = {}
    fake = SimpleNamespace(
        ModelRegistry=SimpleNamespace(
            get_supported_archs=lambda: set(registered),
            register_model=lambda name, path: registered.update({name: path})))
    monkeypatch.setitem(sys.modules, "vllm", fake)
    register_text_engine()
    register_text_engine()
    assert registered == {
        "Qwen3_5ForCausalLM": "vllm.model_executor.models.qwen3_5:Qwen3_5ForCausalLM"}


def test_missing_checkpoint_weights_fail_before_fsdp_gpu_initialization(monkeypatch):
    from transformers import AutoConfig, AutoModelForCausalLM
    monkeypatch.setattr(
        AutoConfig, "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(model_type="qwen3_5_text"))
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        monkeypatch.setattr(
            AutoModelForCausalLM, "from_pretrained",
            lambda *args, **kwargs: (ToyLM(), {key: ["invalid checkpoint tensor"]}))
        with pytest.raises(ValueError, match="load strictly"):
            _load_fsdp(Path("not-used"), torch.device("cpu"))
