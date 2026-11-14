"""CPU-only contracts; the toy shop is explicitly not the WebShop benchmark."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import socket
from types import SimpleNamespace

import pytest
import torch

from whale_local.webshop.core import (
    Episode, Protocol, Sample, Turn, collect_episode, compare_reports, digest,
    evaluation_report, parse_action, prompt_for, successful_context, training_rows,
    wilson_interval, bootstrap_mean_interval,
)
from whale_local.webshop.objective import policy_loss
from whale_local.webshop.runtime import (
    HFPolicy, WebShopEnvironment, check_lease, model_identity, response_log_probs,
)
from whale_local.webshop.__main__ import main, optimize_group


class Codec:
    def render(self, system, user):
        return f"SYSTEM:{system}\nUSER:{user}\nASSISTANT:"

    def encode(self, text):
        return list(text.encode())

    def privileged(self, prompt, feedback):
        return feedback + "\n" + prompt


class ScriptedPolicy(Codec):
    def __init__(self, texts):
        self.texts = iter(texts)

    def sample(self, prompt_ids, max_tokens, seed, temperature):
        text = next(self.texts)
        ids = self.encode(text)
        return Sample(text, ids, [-1.0] * len(ids))


class Shop:
    def __init__(self, terminal_score=1.0):
        self.actions = []
        self.terminal_score = terminal_score

    def reset(self, task_id):
        self.actions = []
        return "Search page", "Buy a red mug", {"id": task_id, "goal": "red mug", "seed": 233}

    def available_actions(self):
        return {"has_search_bar": True, "clickables": ["Buy Now"]}

    def step(self, action):
        self.actions.append(action)
        done = action == "click[Buy Now]"
        return "Purchased" if done else "Product page", self.terminal_score if done else 0.0, done, {}


def small_protocol(**kwargs):
    return Protocol(train_ids=(500,), eval_ids=(0, 1), eval_seeds=(7, 8),
                    group_size=2, **kwargs)


def episode(success=True, seed=10, split="train", task_id=500):
    text = "<action>click[Buy Now]</action>"
    sample = Sample(text, list(text.encode()), [-1.0] * len(text))
    result = Episode(task_id, split, seed, {"id": task_id, "goal_seed": 233})
    result.turns = [Turn("student sees CURRENT state", [2, 3], sample, "An item", "click[Buy Now]",
                         "Done", False)]
    result.terminated = True
    result.task_score = 1.0 if success else 0.5
    result.stop_reason = "terminal"
    return result


@pytest.mark.parametrize("changes", [
    {"train_ids": (0,)}, {"eval_ids": (500,)}, {"eval_ids": (0, 0)},
    {"eval_seeds": ()}, {"group_size": 1}, {"max_steps": 0},
    {"learning_rate": float("nan")}, {"distill_coefficient": -1},
    {"max_context_tokens": 512}, {"history_turns": -1}, {"eval_ids": (True,)},
])
def test_protocol_rejects_leakage_and_invalid_budgets(changes):
    with pytest.raises(ValueError):
        Protocol(**changes)


@pytest.mark.parametrize(("response", "expected"), [
    ("<think>Example search[wrong]</think><action>search[red mug]</action>", "search[red mug]"),
    ("click[Buy Now]", "click[Buy Now]"),
    ("<action>click[red]</action><action>click[Buy Now]</action>", None),
    ("Thought: click[Buy Now]", None),
    ("search[]", None),
    ("search[  ]", None),
    ("<think>search[x]</think>", None),
    ("click[a[b]]", None),
])
def test_action_extraction_does_not_execute_thoughts(response, expected):
    assert parse_action(response) == expected


def test_collection_budget_invalid_response_and_terminal_success():
    policy = ScriptedPolicy(["bad response", "<action>search[red mug]</action>", "click[Buy Now]"])
    env = Shop()
    result = collect_episode(env, policy, small_protocol(max_steps=3), 500, "train", 1)
    assert env.actions == ["invalid", "search[red mug]", "click[Buy Now]"]
    assert result.success and result.turns[0].invalid and len(result.turns) == 3
    assert result.turns[2].sample.token_ids == list(b"click[Buy Now]")
    assert result.turns[2].sample.log_probs == [-1] * len(b"click[Buy Now]")


def test_partial_reward_is_not_success_and_exhaustion_not_terminal():
    partial = collect_episode(Shop(0.9), ScriptedPolicy(["click[Buy Now]"]),
                              small_protocol(max_steps=1), 0, "eval", 7)
    assert partial.terminated and not partial.success and partial.task_score == 0.9
    exhausted = collect_episode(Shop(), ScriptedPolicy(["search[red]"]),
                                small_protocol(max_steps=1), 0, "eval", 7)
    assert not exhausted.terminated and not exhausted.success and exhausted.stop_reason == "max_steps"


def test_context_overflow_never_drops_current_observation_or_makes_fake_turn():
    protocol = small_protocol(max_context_tokens=600, max_new_tokens=512)
    result = collect_episode(Shop(), ScriptedPolicy([]), protocol, 500, "train", 1)
    assert result.turns == [] and result.stop_reason == "context_exhausted"


def test_context_history_is_trimmed_at_whole_turn_boundary():
    codec = Codec()
    past = episode().turns[0]
    past.observation = "HISTORY " * 200
    protocol = small_protocol(max_context_tokens=1300, max_new_tokens=512)
    text, ids = prompt_for("GOAL", "CURRENT", {"clickables": ["Buy Now"]}, [past], codec, protocol)
    assert "HISTORY" not in text and "CURRENT" in text and "GOAL" in text
    assert len(ids) + protocol.max_new_tokens <= protocol.max_context_tokens


def test_only_successful_train_targets_and_exact_student_token_alignment():
    good, bad = episode(), episode(False, seed=11)
    rows, stats = training_rows([good, bad], Codec(), small_protocol())
    assert rows[0].teacher_prompt_ids is None  # leave-one-out, no self-target
    assert rows[1].teacher_prompt_ids is not None
    assert rows[1].response_ids == bad.turns[0].sample.token_ids
    assert rows[1].student_prompt_ids == bad.turns[0].prompt_ids
    teacher = bytes(rows[1].teacher_prompt_ids).decode()
    assert "Terminal task score: 1.0" in teacher and "CURRENT state" in teacher
    assert stats["teacher_turns"] == 1
    assert rows[0].advantage > 0 and rows[1].advantage < 0
    with pytest.raises(ValueError, match="successful training"):
        successful_context(episode(False))
    with pytest.raises(ValueError, match="successful training"):
        successful_context(episode(split="eval", task_id=0))


def test_no_success_means_no_distillation_not_failure_imitation():
    rows, stats = training_rows([episode(False), episode(False, seed=11)], Codec(), small_protocol())
    assert all(r.teacher_prompt_ids is None and r.advantage == 0 for r in rows)
    assert stats["zero_reward_variance"] and stats["teacher_turns"] == 0


def test_success_with_invalid_actions_is_not_a_teacher_example():
    good, bad = episode(), episode(False, seed=11)
    good.turns[0].invalid = True
    rows, statistics = training_rows([good, bad], Codec(), small_protocol())
    assert statistics["successes"] == 1
    assert all(row.teacher_prompt_ids is None for row in rows)


def test_success_context_compacts_without_truncating_student_state():
    good, bad = episode(), episode(False, seed=11)
    good.turns[0].observation = "LONG OBSERVATION " * 1000
    rows, stats = training_rows([good, bad], Codec(), small_protocol())
    assert stats["teacher_context_compacted"] == 1 and stats["teacher_context_overflow"] == 0
    assert rows[1].student_prompt_ids == [2, 3]
    assert b"Action: click[Buy Now]" in bytes(rows[1].teacher_prompt_ids)


@pytest.mark.parametrize("mutation", ["eval", "manifest", "version", "seed"])
def test_group_identity_and_split_isolation(mutation):
    a, b = episode(), episode(False, seed=11)
    if mutation == "eval":
        b.split = "eval"
    elif mutation == "manifest":
        b.task_manifest["goal_seed"] = 234
    elif mutation == "version":
        b.policy_version = 1
    else:
        b.seed = a.seed
    with pytest.raises(ValueError):
        training_rows([a, b], Codec(), small_protocol())


def test_masked_nan_tokens_have_zero_gradient_teacher_and_behavior_are_detached():
    student = torch.tensor([[-1.0, float("nan"), -2.0]], requires_grad=True)
    old = torch.tensor([[-1.0, float("nan"), -2.0]], requires_grad=True)
    teacher = torch.tensor([[-0.5, float("nan"), -3.0]], requires_grad=True)
    loss, stats = policy_loss(
        student, old, torch.tensor([[1, 0, 1]]), torch.tensor([0.0]),
        teacher_log_probs=teacher, teacher_mask=torch.tensor([[1, 0, 1]]),
        coefficient=0.5,
    )
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(student.grad).all()
    assert student.grad[0, 1] == 0
    assert student.grad[0, 0] < 0 and student.grad[0, 2] > 0
    assert teacher.grad is None and old.grad is None
    assert stats["teacher_tokens"] == 2 and stats["action_tokens"] == 2


def test_episode_balancing_not_turn_or_length_weighting():
    # One short trajectory and one long trajectory receive equal total mass.
    student = torch.tensor([[-1.0, 0, 0], [-1.0, -1.0, -1.0]], requires_grad=True)
    old = student.detach().clone()
    loss, _ = policy_loss(student, old, torch.tensor([[1, 0, 0], [1, 1, 1]]),
                          torch.tensor([1.0, 1.0]), coefficient=0)
    loss.backward()
    assert torch.allclose(student.grad.sum(dim=1), torch.tensor([-0.5, -0.5]))


def test_clipping_and_zero_signal_have_expected_gradients():
    student = torch.tensor([[-0.1]], requires_grad=True)
    loss, _ = policy_loss(student, torch.tensor([[-2.0]]), torch.ones(1, 1),
                          torch.ones(1), coefficient=0)
    loss.backward()
    assert student.grad.item() == 0
    zero = torch.tensor([[-1.0]], requires_grad=True)
    loss, _ = policy_loss(zero, zero.detach(), torch.ones(1, 1), torch.zeros(1))
    loss.backward()
    assert zero.grad.item() == 0


def test_fixed_reference_kl_is_shared_and_detached():
    student = torch.tensor([[-2.0, float("nan")]], requires_grad=True)
    reference = torch.tensor([[-1.0, float("nan")]], requires_grad=True)
    loss, stats = policy_loss(
        student, student.detach(), torch.tensor([[1, 0]]), torch.zeros(1),
        reference_log_probs=reference, reference_kl_coefficient=0.01,
    )
    loss.backward()
    assert torch.isfinite(loss) and stats["reference_kl"] > 0
    assert student.grad[0, 0] < 0 and student.grad[0, 1] == 0
    assert reference.grad is None


@pytest.mark.parametrize("changes", [
    {"coefficient": float("nan")}, {"gap_clip": float("inf")},
    {"reference_kl_coefficient": 0.01}, {"ppo_clip": -0.5},
])
def test_loss_rejects_invalid_coefficients_and_missing_reference(changes):
    with pytest.raises(ValueError):
        policy_loss(torch.tensor([[-1.0]]), torch.tensor([[-1.0]]),
                    torch.ones(1, 1), torch.ones(1), **changes)


def test_causal_response_log_probs_align_only_generated_token_targets():
    class TinyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.table = torch.nn.Parameter(torch.arange(49, dtype=torch.float).reshape(7, 7) / 10)

        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(logits=self.table[input_ids])

    model = TinyLM()
    actual = response_log_probs(model, [1, 2, 3], [4, 5], grad=True)
    expected = torch.stack([model.table[3].log_softmax(0)[4], model.table[4].log_softmax(0)[5]])
    assert torch.allclose(actual, expected)
    (-actual.mean()).backward()
    assert model.table.grad[1].abs().sum() == 0
    assert model.table.grad[2].abs().sum() == 0
    assert model.table.grad[3].abs().sum() > 0
    assert model.table.grad[4].abs().sum() > 0


def test_real_optimizer_updates_tiny_policy_with_frozen_teacher_targets():
    class TinyPolicy(Codec):
        def __init__(self):
            self.model = torch.nn.Embedding(2, 2)
            with torch.no_grad():
                self.model.weight.copy_(torch.tensor([[0.0, 0.0], [2.0, -2.0]]))

        def encode(self, text):
            return [1 if "Verified successful example" in text else 0]

        def log_probs(self, prompt, response, grad=False):
            context = torch.enable_grad() if grad else torch.no_grad()
            with context:
                distribution = self.model(torch.tensor(prompt[-1:])).log_softmax(-1)[0]
                return distribution[torch.tensor(response)]

        def reference_log_probs(self, prompt, response):
            return torch.full((len(response),), -0.6931471805599453)

    policy = TinyPolicy()
    good, bad = episode(), episode(False, seed=11)
    for e, token in ((good, 0), (bad, 1)):
        e.turns[0].prompt_ids = [0]
        e.turns[0].sample = Sample("click[Buy Now]", [token], [-0.6931471805599453])
    optimizer = torch.optim.SGD(policy.model.parameters(), lr=0.1)
    before = policy.model.weight.detach().clone()
    stats = optimize_group(policy, [good, bad], small_protocol(), "success_sd", optimizer)
    assert not torch.equal(before[0], policy.model.weight.detach()[0])
    assert torch.equal(before[1], policy.model.weight.detach()[1])
    assert stats["gradient_norm"] > 0 and stats["teacher_turns"] == 1
    assert stats["optimizer_steps"] == 1 and stats["generated_action_tokens"] == 2


def test_actual_transformer_generation_logprobs_eos_alignment_and_lora_on_cpu():
    from transformers import GPT2Config, GPT2LMHeadModel
    from peft import get_peft_model, LoraConfig

    torch.manual_seed(17)
    model = get_peft_model(GPT2LMHeadModel(GPT2Config(
        n_layer=1, n_head=2, n_embd=16, vocab_size=32, n_positions=64,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    )), LoraConfig(task_type="CAUSAL_LM", r=2, target_modules=["c_attn"],
                  fan_in_fan_out=True, lora_dropout=0.0))
    model.eval()
    policy = HFPolicy.__new__(HFPolicy)
    policy.model, policy.device, policy.guard = model, torch.device("cpu"), lambda: None
    policy.tokenizer = SimpleNamespace(
        eos_token_id=2, pad_token_id=0,
        decode=lambda ids, **kwargs: " ".join(str(int(x)) for x in ids),
    )
    sample = policy.sample([1, 3, 4], max_tokens=4, seed=23, temperature=1.0)
    sample.validate()
    scores = policy.log_probs([1, 3, 4], sample.token_ids)
    assert torch.allclose(scores, torch.tensor(sample.log_probs), atol=1e-5)
    reference = policy.reference_log_probs([1, 3, 4], sample.token_ids)
    assert not reference.requires_grad
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1)
    before = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
    loss = -policy.log_probs([1, 3, 4], sample.token_ids, grad=True).mean()
    loss.backward()
    optimizer.step()
    assert any(not torch.equal(before[n], p.detach())
               for n, p in model.named_parameters() if p.requires_grad)


def test_environment_adapter_clears_only_its_own_stale_session_and_flags_invalid_clicks():
    class RawEnv(Shop):
        def __init__(self):
            super().__init__()
            self.server = SimpleNamespace(
                goals=[{"instruction_text": "red mug"}] * 501,
                user_sessions={"500": {"done": True, "reward": 1}, "peer": {"keep": True}},
            )

        def reset(self, session):
            assert str(session) not in self.server.user_sessions
            self.server.user_sessions[str(session)] = {"done": False}
            return "Search page", None

        def get_instruction_text(self):
            return "red mug"

        def get_available_actions(self):
            return self.available_actions()

    adapter = WebShopEnvironment.__new__(WebShopEnvironment)
    adapter.env, adapter.protocol = RawEnv(), small_protocol()
    adapter.prices_digest, adapter.assets = "prices", {"data": "sha"}
    _observation, _instruction, manifest = adapter.reset(500)
    assert adapter.env.server.user_sessions["peer"] == {"keep": True}
    assert manifest["task_id"] == 500 and manifest["goal_seed"] == 233
    assert adapter.step("click[unseen]")[3]["invalid"]
    assert not adapter.step("click[Buy Now]")[3]["invalid"]


def reports():
    protocol = small_protocol()
    episodes = [
        episode(success=(task == 0), seed=seed, task_id=task, split="eval")
        for task in protocol.eval_ids for seed in protocol.eval_seeds
    ]
    baseline = evaluation_report(episodes, protocol, "initial", {"products": "abc"})
    baseline["base_model_identity"] = {"weights": "toy-fixture"}
    baseline["implementation_identity"] = {"core.py": "toy-fixture"}
    improved = deepcopy(baseline)
    e = improved["episodes"][2]
    e["success"], e["task_score"] = True, 1.0
    improved["model"] = "improved"
    return baseline, improved


def test_paired_report_recomputes_counts_and_does_not_trust_aggregate_score():
    baseline, improved = reports()
    improved["metrics"]["success_rate"] = 10000
    result = compare_reports(baseline, improved)
    assert result["baseline_successes"] == 2 and result["improved_successes"] == 3
    assert result["success_rate_delta"] == 0.25
    assert result["rescued"] == 1 and result["regressed"] == 0
    assert result["mcnemar_exact_two_sided"] is None
    assert not result["subset_evidence_gate"]["passed"]


def test_confidence_intervals_have_correct_scale_and_deterministic_resampling():
    lower, upper = wilson_interval(0, 32)
    assert lower < 1e-10 and 0.10 < upper < 0.11
    assert wilson_interval(32, 32)[1] > 0.999
    assert bootstrap_mean_interval([0.25] * 8, 11) == [0.25, 0.25]
    assert bootstrap_mean_interval([0, 1, 0.5], 11) == bootstrap_mean_interval([0, 1, 0.5], 11)


def test_prespecified_subset_gate_requires_real_teacher_signal_and_matched_training():
    protocol = Protocol(train_ids=(500, 501, 502, 503), eval_ids=tuple(range(32)),
                        eval_seeds=(0,), group_size=2, updates=1)
    a_episodes = [episode(task_id=task, split="eval", seed=0, success=task < 16)
                  for task in protocol.eval_ids]
    b_episodes = [episode(task_id=task, split="eval", seed=0, success=task < 22)
                  for task in protocol.eval_ids]
    a = evaluation_report(a_episodes, protocol, "toy-control", {})
    b = evaluation_report(b_episodes, protocol, "toy-improved", {})
    for report in (a, b):
        report["base_model_identity"] = {"weights": "toy-fixture"}
        report["implementation_identity"] = {"code": "toy-fixture"}
        report["training"] = {
            "base_model": "toy", "base_model_identity": report["base_model_identity"],
            "implementation_identity": report["implementation_identity"],
            "protocol": protocol.as_dict(), "assets": {}, "trainable_parameters": 4,
            "metrics": [{"teacher_tokens": 1, "teacher_gap_abs_mean": 0.1, "gradient_norm": 0.5}],
        }
    comparison = compare_reports(a, b)
    assert comparison["mcnemar_exact_two_sided"] == 0.03125
    assert comparison["subset_evidence_gate"]["passed"]
    b["training"]["metrics"][0]["teacher_tokens"] = 0
    assert not compare_reports(a, b)["subset_evidence_gate"]["passed"]
    b["training"] = None
    assert not compare_reports(a, b)["subset_evidence_gate"]["passed"]


@pytest.mark.parametrize("mutation", ["assets", "protocol", "goal", "duplicate", "missing", "score"])
def test_pair_comparison_rejects_nonmatching_or_invalid_evidence(mutation):
    baseline, improved = reports()
    if mutation == "assets":
        improved["assets"]["products"] = "changed"
    elif mutation == "protocol":
        improved["protocol"]["max_steps"] += 1
    elif mutation == "goal":
        improved["episodes"][0]["task_manifest"]["goal_seed"] += 1
    elif mutation == "duplicate":
        improved["episodes"].append(improved["episodes"][0])
    elif mutation == "missing":
        improved["episodes"].pop()
    else:
        improved["episodes"][0]["success"] = False
    with pytest.raises(ValueError):
        compare_reports(baseline, improved)


def test_evaluation_seed_does_not_change_goal_identity():
    protocol = small_protocol()
    episodes = [episode(seed=seed, split="eval", task_id=task)
                for task in protocol.eval_ids for seed in protocol.eval_seeds]
    episodes[-1].task_manifest["goal_seed"] += 1
    with pytest.raises(ValueError, match="Goal identity"):
        evaluation_report(episodes, protocol, "model", {})


def test_preflight_is_cpu_only_and_persists_blockers(tmp_path):
    output = tmp_path / "preflight.json"
    code = main(["preflight", "--env-root", str(tmp_path / "absent"), "--output", str(output)])
    result = json.loads(output.read_text())
    assert code == 2 and not result["ready"]
    assert result["missing_files"] and not result["gpu_used"] and not result["model_loaded"]
    with pytest.raises(ValueError, match="already exists"):
        main(["preflight", "--env-root", str(tmp_path / "absent"), "--output", str(output)])


def test_train_cli_refuses_before_environment_or_model_initialization(tmp_path, monkeypatch):
    path, lease, _now = lease_file(tmp_path, monkeypatch)
    lease["primary_sft_released"] = False
    path.write_text(json.dumps(lease))
    config = tmp_path / "config.json"
    config.write_text(json.dumps(small_protocol().as_dict()))
    with pytest.raises(ValueError, match="Primary SFT"):
        main([
            "train", "--config", str(config), "--env-root", str(tmp_path / "noenv"),
            "--model", str(tmp_path / "nomodel"), "--lease", str(path),
            "--method", "success_sd", "--output", str(tmp_path / "output"),
        ])
    assert not (tmp_path / "output").exists()


def test_environment_check_requires_hidden_gpus_before_loading_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ValueError, match="CPU environment check"):
        main(["environment-check", "--config", str(tmp_path / "no-config"),
              "--env-root", str(tmp_path), "--output", str(tmp_path / "receipt.json")])


def test_cpu_environment_check_writes_deterministic_transition_receipt(tmp_path, monkeypatch):
    import importlib
    cli = importlib.import_module("whale_local.webshop.__main__")

    class LocalEnv(Shop):
        def __init__(self, path, protocol):
            super().__init__()
            self.assets = {"products": "test-fixture"}

        def close(self):
            pass

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(cli, "WebShopEnvironment", LocalEnv)
    monkeypatch.setattr(cli, "preflight", lambda *args: {"ready": True})
    config = tmp_path / "config.json"
    config.write_text(json.dumps(small_protocol().as_dict()))
    output = tmp_path / "environment.json"
    assert main(["environment-check", "--config", str(config), "--env-root", str(tmp_path),
                 "--action", "search[shoes]", "--output", str(output)]) == 0
    receipt = json.loads(output.read_text())
    assert receipt["healthy"] and receipt["deterministic_repeats"] == 2
    assert not receipt["model_used"] and not receipt["gpu_used"] and not receipt["http_service_started"]


def lease_file(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    now = datetime.now(timezone.utc)
    lease = {
        "purpose": "webshop-side-project", "hostname": socket.gethostname(),
        "cuda_visible_devices": "3", "expires_at": (now + timedelta(minutes=10)).isoformat(),
        "primary_sft_released": True, "modes": ["train", "evaluate"], "max_episodes": 128,
    }
    path = tmp_path / "explicit-lease.json"
    path.write_text(json.dumps(lease))
    return path, lease, now


@pytest.mark.parametrize("mutation", ["sft", "expired", "host", "devices", "mode", "budget", "naive"])
def test_resource_gate_fails_closed_before_importing_model(tmp_path, monkeypatch, mutation):
    path, lease, now = lease_file(tmp_path, monkeypatch)
    if mutation == "sft":
        lease["primary_sft_released"] = False
    elif mutation == "expired":
        lease["expires_at"] = (now - timedelta(seconds=1)).isoformat()
    elif mutation == "host":
        lease["hostname"] = "another-host"
    elif mutation == "devices":
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,4")
    elif mutation == "mode":
        lease["modes"] = ["evaluate"]
    elif mutation == "budget":
        lease["max_episodes"] = 1
    else:
        lease["expires_at"] = "2099-01-01T00:00:00"
    path.write_text(json.dumps(lease))
    with pytest.raises(ValueError):
        check_lease(path, mode="train", episodes=16, now=now)


def test_resource_gate_accepts_only_explicit_exact_scope(tmp_path, monkeypatch):
    path, lease, now = lease_file(tmp_path, monkeypatch)
    assert check_lease(path, mode="train", episodes=128, now=now) == lease


def test_canonical_goal_hash_supports_webshop_attribute_sets():
    assert digest({"attributes": {"red", "small"}}) == digest({"attributes": {"small", "red"}})


def test_model_identity_hashes_content_not_just_path_or_size(tmp_path):
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"first")
    before = model_identity(tmp_path)
    weight.write_bytes(b"other")
    assert model_identity(tmp_path) != before


def test_full_cli_train_save_evaluate_compare_roundtrip_on_cpu_toy_shop(tmp_path, monkeypatch):
    import importlib
    cli = importlib.import_module("whale_local.webshop.__main__")

    class LocalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([[0.0, 0.0], [1.0, -1.0]]))

        def save_pretrained(self, path):
            path.mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), path / "pytorch_model.bin")

    class LocalPolicy(Codec):
        def __init__(self, path, seed, *, train=False, adapter=None, guard):
            self.guard, self.model = guard, LocalModel()
            self.model_identity = model_identity(path)
            self.tokenizer = SimpleNamespace(save_pretrained=lambda path: None)
            if adapter:
                self.model.load_state_dict(torch.load(adapter / "pytorch_model.bin", weights_only=True))

        def encode(self, text):
            return [1 if "Verified successful example" in text else 0]

        def log_probs(self, prompt, response, grad=False):
            self.guard()
            with torch.enable_grad() if grad else torch.no_grad():
                return self.model.weight[prompt[-1]].log_softmax(-1)[torch.tensor(response)]

        def reference_log_probs(self, prompt, response):
            return torch.full((len(response),), -0.6931471805599453)

        def sample(self, prompt, max_tokens, seed, temperature):
            self.guard()
            logits = self.model.weight[0].detach() / temperature
            token = int(torch.multinomial(logits.softmax(-1), 1,
                                         generator=torch.Generator().manual_seed(seed)))
            return Sample("click[Buy Now]" if token == 0 else "search[red mug]", [token],
                          [float(logits.log_softmax(-1)[token])])

    class LocalEnv(Shop):
        def __init__(self, root, protocol):
            super().__init__()
            self.assets = {"products": "toy-only-not-webshop"}

        def close(self):
            pass

    monkeypatch.setattr(cli, "HFPolicy", LocalPolicy)
    monkeypatch.setattr(cli, "WebShopEnvironment", LocalEnv)
    monkeypatch.setattr(cli, "preflight", lambda *args: {"ready": True})
    lease, _value, _now = lease_file(tmp_path, monkeypatch)
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"toy base weights identity")
    (model / "config.json").write_text("{}")
    config = tmp_path / "protocol.json"
    config.write_text(json.dumps(small_protocol(updates=1, max_steps=1).as_dict()))
    common = ["--config", str(config), "--env-root", str(tmp_path),
              "--model", str(model), "--lease", str(lease)]
    for method in ("grpo", "success_sd"):
        training = tmp_path / method
        evaluation = tmp_path / (method + "_eval")
        assert main(["train", *common, "--method", method, "--output", str(training)]) == 0
        assert (training / "adapter" / "pytorch_model.bin").exists()
        assert main(["evaluate", *common, "--adapter", str(training / "adapter"),
                     "--output", str(evaluation)]) == 0
        report = json.loads((evaluation / "REPORT.json").read_text())
        assert report["metrics"]["episodes"] == 4
        assert report["training"]["metrics"][0]["optimizer_steps"] == 1
    output = tmp_path / "comparison.json"
    assert main([
        "compare", "--baseline", str(tmp_path / "grpo_eval" / "REPORT.json"),
        "--improved", str(tmp_path / "success_sd_eval" / "REPORT.json"),
        "--output", str(output),
    ]) == 0
    paired = json.loads(output.read_text())
    assert paired["paired_episodes"] == 4 and paired["matched_training_budget"]
