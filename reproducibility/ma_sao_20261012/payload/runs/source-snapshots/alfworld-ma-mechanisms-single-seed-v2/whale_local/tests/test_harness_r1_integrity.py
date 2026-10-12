"""CPU-only integrity tests. Run pytest with --basetemp inside the workspace."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness_r1_searchqa import patch, reward, sandbox
from harness_r1_searchqa.data import build_rl_rows, public_failure, load_train_rows
from harness_r1_searchqa.identity import baseline_config, digest, runtime_source_identity
from harness_r1_searchqa.manifest import load_manifest_splits
from harness_r1_searchqa.runtime import SearchQAHarnessRuntime
from sao_bridge.searchqa_prepare import ASSETS, sha256


def candidate(code='def hook(ctx, nb):\n    return {"message": "Search entities."}',
              hook="make_pre_hint"):
    return {"benchmark": "searchqa", "actions": [
        {"type": "add_code_hook", "hook": hook, "code": code}]}


def response(value):
    return "Reusable diagnosis.</think><patch>" + json.dumps(value) + "</patch>"


def runtime(value):
    return SearchQAHarnessRuntime(value, question="A public question?", max_steps=2,
                                  max_tool_calls=1, top_k=1, document_char_limit=800)


def hint(value):
    return value.pre_hint(step=0, remaining_steps=2, previous_queries=[], last_observation="")


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    monkeypatch.delenv("SAO_SEARCHQA_RETRIEVAL_URL", raising=False)
    rows = load_manifest_splits(ASSETS, splits=("train",))["train"][:2]
    root = tmp_path / "data"
    root.mkdir()
    train = root / "train.json"
    train.write_text(json.dumps(rows))
    original = json.loads((ASSETS / "manifest.json").read_text())
    original["splits"]["train"] = {
        "file": "train.json", "sha256": sha256(train), "count": len(rows),
        "ids_sha256": hashlib.sha256("\n".join(row["id"] for row in rows).encode()).hexdigest(),
    }
    # These intentionally do not exist: training/reward must never open them.
    for split in ("reward", "dev", "test"):
        original["splits"][split]["file"] = f"forbidden-{split}.json"
    (root / "manifest.json").write_text(json.dumps(original))
    (root / "retrieval_manifest.json").write_bytes((ASSETS / "retrieval_manifest.json").read_bytes())
    incumbent = patch.base_patch()
    args = SimpleNamespace(
        harness_r1_searchqa_data=str(root),
        harness_r1_target_model_sha256="frozen-target",
        harness_r1_target_temperature=0.3, harness_r1_rollout_max_tokens=4096,
        harness_r1_target_urls=["http://127.0.0.1:8110/v1"],
        harness_r1_retrieval_url="http://127.0.0.1:8120",
        harness_r1_eval_root=str(tmp_path / "cache"), harness_r1_rollout_concurrency=1,
    )
    config = baseline_config(
        target_model_sha256="frozen-target", target_temperature=0.3, total_tokens=4096,
        retrieval_manifest=root / "retrieval_manifest.json",
        data_manifest=root / "manifest.json", incumbent_patch=incumbent)
    results = [{"task_id": row["id"], "question": row["question"], "reward": 0.0,
                "rollout_seed": index + 1, "messages": [
                    {"role": "user", "content": row["question"]},
                    {"role": "assistant", "content": "<answer>wrong</answer>"}],
                "formatted": True, "tool_errors": 0}
               for index, row in enumerate(rows)]
    report = {"split": "train", "baseline_config": config, "results": results}
    metadata = build_rl_rows(report, incumbent_patch=incumbent,
                             target_model_sha256="frozen-target",
                             data_manifest=root / "manifest.json", packet_size=2)[0]["metadata"]
    return args, metadata, report, root


@pytest.mark.parametrize("text", [
    'prefix {"benchmark":"searchqa"}', '{"benchmark":"searchqa"} suffix',
    '{"benchmark":"searchqa","benchmark":"searchqa"}',
    '{"value":NaN}', '{"value":1e999}', '[]',
])
def test_strict_json(text):
    with pytest.raises(patch.PatchValidationError):
        patch.extract_patch(text, "json_patch")


@pytest.mark.parametrize("suffix", [" trailing", "<patch>{}</patch>", "<think>again</think>"])
def test_rejects_extra_response_blocks(suffix):
    with pytest.raises(patch.PatchValidationError):
        patch.extract_and_normalize_patch(response(candidate()) + suffix)


@pytest.mark.parametrize("change", [
    {"schema_version": "other"}, {"unknown": True}, {"benchmark": "SearchQA"},
    {"actions": [{"type": "add_code_hook", "hook": [], "code": "bad"}]},
])
def test_strict_patch_schema(change):
    with pytest.raises(patch.PatchValidationError):
        patch.normalize_patch({**candidate(), **change})


def test_cumulative_merge_and_canonical_identity():
    init = candidate('def hook(ctx, nb):\n    return {"tool_hint":"Use evidence."}', "on_init")
    merged = patch.merge_patches(init, candidate())
    assert {action["hook"] for action in merged["actions"]} == {"on_init", "make_pre_hint"}
    reordered = copy.deepcopy(merged)
    reordered["actions"].reverse()
    assert patch.patch_digest(merged) == patch.patch_digest(reordered)
    assert patch.patch_digest(patch.base_patch()) != patch.patch_digest(
        {**patch.base_patch(), "description": "Distinct incumbent"})


def test_runtime_identity_uses_bounded_inventory_not_training_utilities(monkeypatch):
    from harness_r1_searchqa import identity
    observed = []
    def source_hash(path):
        observed.append(Path(path).name)
        return "a" * 64
    monkeypatch.setattr(identity, "sha256", source_hash)
    first = runtime_source_identity()
    assert {"patch.py", "sandbox.py", "episode.py", "reward.py", "data.py"} <= set(observed)
    assert not {"sft.py", "grpo.py", "bootstrap.py", "serve.py"} & set(observed)
    monkeypatch.setattr(Path, "glob", lambda *args: (_ for _ in ()).throw(
        AssertionError("Runtime identity must not glob unrelated utilities")))
    assert runtime_source_identity() == first


def test_real_sandbox_notebook_and_episode_separation():
    value = candidate('def hook(ctx, nb):\n'
                      '    nb["count"] = nb.get("count", 0) + 1\n'
                      '    return {"message": str(nb["count"])}')
    first, second = runtime(value), runtime(value)
    assert hint(first) == "1"
    assert hint(first) == "2"
    assert hint(second) == "1"
    assert first.trace[-1]["status"] == "ok"


def test_empty_effect_is_official_noop_and_keeps_notebook():
    value = runtime(candidate('def hook(ctx, nb):\n    nb["state"]=1\n    return {}'))
    assert hint(value) == ""
    assert value.notebook == {"state": 1} and value.trace[-1]["status"] == "ok"


@pytest.mark.parametrize("code,error", [
    ('def hook(ctx, nb):\n    nb["partial"]=True\n    return 1 / 0', "ZeroDivisionError"),
    ('def hook(ctx, nb):\n    nb["large"]="x" * 20000\n    return None', "byte limit"),
    ('def hook(ctx, nb):\n    nb["cycle"]=nb\n    return None', "limit"),
    ('def hook(ctx, nb):\n    nb["bad"]=set([1])\n    return None', "JSON"),
    ('def hook(ctx, nb):\n    nb["huge"]="x" * 1000000000\n    return None', "MemoryError"),
    ('def hook(ctx, nb):\n    for i in range(1000000000):\n        nb["i"]=i\n    return None', "budget"),
    ('def hook(ctx, nb):\n    return {"kind":"rewrite_action"}', "invalid hook effect"),
])
def test_real_candidate_failures_are_traced_and_rollback(code, error):
    value = runtime(candidate(code))
    assert hint(value) == ""
    assert value.notebook == {}
    assert value.trace[-1]["status"] == "candidate_error"
    assert error in value.trace[-1]["error"]


def test_compile_defaults_are_process_contained(monkeypatch):
    value = candidate('def helper(x="x" * 1000000000):\n    return x\n'
                      'def hook(ctx, nb):\n    return None')
    def forbidden_host_compile(*args, **kwargs):
        raise AssertionError("Untrusted compilation must not run on the host")
    monkeypatch.setattr(patch.CODE_RUNNER, "compile_hook", forbidden_host_compile)
    with pytest.raises(patch.PatchValidationError, match="simple scalar constants"):
        patch.normalize_patch(value)
    assert patch.normalize_patch(candidate())["benchmark"] == "searchqa"


def test_infrastructure_failure_is_traced_and_raised(monkeypatch):
    value = runtime(candidate())
    def fail(*args, **kwargs):
        raise sandbox.SandboxInfrastructureError("bwrap unavailable")
    monkeypatch.setattr(patch, "invoke", fail)
    with pytest.raises(sandbox.SandboxInfrastructureError):
        hint(value)
    assert value.trace[-1]["status"] == "infrastructure_error"


def test_missing_hooks_preserve_exact_base_behavior():
    value = runtime(patch.base_patch())
    assert value.initialize() == ([], "")
    assert hint(value) == ""
    before = value.before_query("Original Query", step=0, remaining_steps=2,
                               previous_queries=[], last_observation="")
    assert before.query == "Original Query" and before.effect is None and not before.blocked
    assert value.post_step(query=before.query, observation="Public evidence.", step=0,
                           remaining_steps=1, previous_queries=[before.query]) == ""
    assert value.trace == [] and value.notebook == {}


def test_base_episode_exact_equivalence_to_legacy_noop(monkeypatch):
    from harness_r1_searchqa import episode
    from harness_r1_searchqa.runtime import BeforeQuery
    class Sampler:
        async def __call__(self, messages, tools, max_tokens, seed):
            return {"usage": {"completion_tokens": 4}, "choices": [
                {"message": {"role": "assistant", "content": "<answer>Paris</answer>"}}]}
    class LegacyNoopRuntime:
        def __init__(self, *args, **kwargs):
            self.trace, self.notebook = [], {}
            self.compiled = {"hooks": {}}
        def initialize(self):
            return [], ""
        def pre_hint(self, **kwargs):
            return ""
        def before_query(self, query, **kwargs):
            return BeforeQuery(query, False, "", None)
        def post_step(self, **kwargs):
            return ""
    def forbidden_sandbox(*args, **kwargs):
        raise AssertionError("Base episodes must not start a sandbox")
    monkeypatch.setattr(patch, "invoke", forbidden_sandbox)
    row = {"id": "public-test", "question": "Capital of France?", "answers": ["Paris"]}
    actual = asyncio.run(episode.evaluate_one(
        row, sampler=Sampler(), patch=patch.base_patch(), total_tokens=64, rollout_seed=42))
    monkeypatch.setattr(episode, "SearchQAHarnessRuntime", LegacyNoopRuntime)
    expected = asyncio.run(episode.evaluate_one(
        row, sampler=Sampler(), patch=patch.base_patch(), total_tokens=64, rollout_seed=42))
    assert actual == expected


def test_isolation_mounts_limits_and_output_bounds(monkeypatch):
    command = sandbox.isolation._sandbox_command()
    assert "--unshare-all" in command and "--clearenv" in command
    assert "--as=268435456" in command
    assert not any("/mnt/" in item for item in command if item != str(sandbox.isolation._BWRAP))
    monkeypatch.setattr(sandbox.isolation, "_sandbox_command", lambda: [
        "/usr/bin/python3", "-c",
        "import sys;sys.stdout.write('HARNESS_R1_READY\\n'+'x'*1000000);sys.stdout.flush()"])
    # _transport replaces the final argument with its bootstrap; replace bootstrap
    # for this pipe-exhaustion test, not candidate execution.
    monkeypatch.setattr(sandbox, "_WORKER",
                        "import sys;sys.stdout.write('HARNESS_R1_READY\\n'+'x'*1000000);sys.stdout.flush()")
    output = sandbox._transport(b"{}")
    assert len(output) < 1000 and b"output limit" in output


def test_train_loader_does_not_read_unrequested_splits(frozen):
    _, _, _, root = frozen
    assert len(load_manifest_splits(root, splits=("train",))["train"]) == 2
    assert len(load_train_rows(root)) == 2
    with pytest.raises(FileNotFoundError):
        load_manifest_splits(root, splits=("test",))
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["splits"]["train"]["file"] = "../escape.json"
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="escapes"):
        load_manifest_splits(root, splits=("train",))


def test_invalid_patch_only_after_validated_provenance(frozen):
    args, metadata, _, _ = frozen
    result = asyncio.run(reward.reward_func(args, SimpleNamespace(metadata=metadata, response="bad")))
    assert result["score"] == 0 and result["valid_patch"] is False
    for field, value in (("task_ids", ["test-task", metadata["task_ids"][1]]),
                         ("split", "test"), ("source_baseline_sha256", "missing"),
                         ("batch_tag", "../escape"), ("batch_tag", "/absolute")):
        bad = {**metadata, field: value}
        if field == "task_ids":
            bad["baseline_rewards"] = {identity: 0.0 for identity in value}
        with pytest.raises(ValueError):
            asyncio.run(reward.reward_func(args, SimpleNamespace(metadata=bad, response="bad")))


@pytest.mark.parametrize("field,value", [
    ("harness_r1_target_urls", ["not-a-url"]),
    ("harness_r1_retrieval_url", "https://127.0.0.1:8120"),
    ("harness_r1_eval_root", ""),
    ("harness_r1_rollout_concurrency", 0),
])
def test_invalid_infrastructure_config_is_not_zero(frozen, field, value):
    args, metadata, _, _ = frozen
    setattr(args, field, value)
    with pytest.raises(ValueError):
        asyncio.run(reward.reward_func(args, SimpleNamespace(metadata=metadata, response="bad")))


@pytest.mark.parametrize("field,value", [
    ("harness_r1_target_temperature", 0.8), ("harness_r1_rollout_max_tokens", 2048),
    ("harness_r1_target_model_sha256", "changed"),
])
def test_changed_frozen_config_rejected_even_for_invalid_patch(frozen, field, value):
    args, metadata, _, _ = frozen
    setattr(args, field, value)
    with pytest.raises(ValueError):
        asyncio.run(reward.reward_func(args, SimpleNamespace(metadata=metadata, response="bad")))


def test_changed_retriever_runtime_incumbent_and_train_rejected(frozen, monkeypatch):
    args, metadata, _, root = frozen
    changed = copy.deepcopy(metadata)
    changed["baseline_config"]["runtime_source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="provenance"):
        reward._validate_metadata(args, changed)
    changed = copy.deepcopy(metadata)
    changed["baseline_config"]["retrieval_manifest_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="provenance"):
        reward._validate_metadata(args, changed)
    changed = copy.deepcopy(metadata)
    changed["incumbent_patch"] = candidate()
    changed["incumbent_patch_sha256"] = patch.patch_digest(changed["incumbent_patch"])
    with pytest.raises(ValueError, match="provenance"):
        reward._validate_metadata(args, changed)
    reward._rows_by_id(str(root))
    (root / "train.json").write_text("[]")
    with pytest.raises(ValueError, match="split changed"):
        reward._rows_by_id(str(root))


def test_reward_cumulative_cache_binds_all_identities(frozen, monkeypatch):
    args, metadata, _, _ = frozen
    calls = []
    class Sampler:
        def __init__(self, *args, **kwargs):
            pass
        async def close(self):
            pass
    async def evaluate(rows, **kwargs):
        calls.append(kwargs)
        return {"results": [
            {"task_id": row["id"], "reward": 1.0, "rollout_seed": seed, "harness_trace": []}
            for row, seed in zip(rows, kwargs["rollout_seeds"])]}
    monkeypatch.setattr(reward, "Sampler", Sampler)
    monkeypatch.setattr(reward, "evaluate_rows", evaluate)
    sample = SimpleNamespace(metadata=metadata, response=response(candidate()))
    first = asyncio.run(reward.reward_func(args, sample))
    second = asyncio.run(reward.reward_func(args, sample))
    assert first == second and first["score"] == 1 and len(calls) == 1
    args.harness_r1_rollout_http_timeout = 300
    third = asyncio.run(reward.reward_func(args, sample))
    assert third["cache_identity"] != first["cache_identity"] and len(calls) == 2
    metadata["baseline_rewards"][metadata["task_ids"][0]] = 1.0
    fourth = asyncio.run(reward.reward_func(args, sample))
    assert fourth["score"] == 0.5 and len(calls) == 3
    metadata["source_baseline_sha256"] = "f" * 64
    fifth = asyncio.run(reward.reward_func(args, sample))
    assert fifth["cache_identity"] != fourth["cache_identity"] and len(calls) == 4


def test_hook_stats_aggregate_candidate_errors_and_do_not_swallow_infrastructure():
    stats = reward._hook_stats([
        {"task_id": "train-a", "harness_trace": [
            {"hook": "on_init", "status": "ok"},
            {"hook": "make_pre_hint", "status": "candidate_error", "error": "AttributeError"}]},
        {"task_id": "train-b", "harness_trace": [
            {"hook": "make_pre_hint", "status": "ok"}]},
        {"task_id": "train-c", "harness_trace": []},
    ])
    assert stats["hook_calls"] == 3 and stats["hook_errors"] == 1
    assert stats["hook_status_counts"] == {"ok": 2, "candidate_error": 1,
                                         "infrastructure_error": 0}
    assert stats["hook_runtime_errors"][0]["task_id"] == "train-a"
    assert reward._hook_stats([])["hook_calls"] == 0
    for trace in ([{"status": "infrastructure_error"}], [{"status": "unknown"}],
                  [{"status": []}], "malformed"):
        with pytest.raises(RuntimeError):
            reward._hook_stats([{"task_id": "train-a", "harness_trace": trace}])


def test_positive_reward_exposes_real_string_observation_hook_errors(frozen, monkeypatch):
    args, metadata, _, _ = frozen
    invocations = []
    class Sampler:
        def __init__(self, *args, **kwargs):
            pass
        async def close(self):
            pass
    async def evaluate(rows, **kwargs):
        results = []
        for row, seed in zip(rows, kwargs["rollout_seeds"]):
            value = runtime(kwargs["patch"])
            assert hint(value) == ""  # Official no-effect fallback remains unchanged.
            assert value.notebook == {}
            invocations.append(value.trace)
            results.append({"task_id": row["id"], "reward": 1.0, "rollout_seed": seed,
                            "harness_trace": value.trace})
        return {"results": results}
    monkeypatch.setattr(reward, "Sampler", Sampler)
    monkeypatch.setattr(reward, "evaluate_rows", evaluate)
    bad = candidate('def hook(ctx, nb):\n'
                    '    return {"message": ctx.last_observation.get("message", "")}')
    sample = SimpleNamespace(metadata=metadata, response=response(bad))
    result = asyncio.run(reward.reward_func(args, sample))
    assert result["score"] == result["delta_score"] == 1.0
    assert result["valid_patch"] is True and result["eval_status"] == "ok"
    assert result["hook_calls"] == 2 and result["hook_errors"] == 2
    assert result["hook_status_counts"] == {"ok": 0, "candidate_error": 2,
                                          "infrastructure_error": 0}
    assert all("AttributeError" in event["error"] for event in result["hook_runtime_errors"])
    assert not (result["score"] > 0 and result["hook_errors"] == 0)
    assert asyncio.run(reward.reward_func(args, sample)) == result
    assert len(invocations) == 2


@pytest.mark.parametrize("failure", ["duplicate", "seed", "reward", "infrastructure"])
def test_reward_rejects_bad_result_identities_without_cache(frozen, monkeypatch, failure):
    args, metadata, _, _ = frozen
    class Sampler:
        def __init__(self, *args, **kwargs):
            pass
        async def close(self):
            pass
    async def evaluate(rows, **kwargs):
        results = [{"task_id": row["id"], "reward": 1.0, "rollout_seed": seed}
                   for row, seed in zip(rows, kwargs["rollout_seeds"])]
        if failure == "duplicate":
            results[1] = results[0]
        elif failure == "seed":
            results[0]["rollout_seed"] += 1
        elif failure == "reward":
            results[0]["reward"] = 0.5
        else:
            results[0]["harness_trace"] = [{"status": "infrastructure_error"}]
        return {"results": results}
    monkeypatch.setattr(reward, "Sampler", Sampler)
    monkeypatch.setattr(reward, "evaluate_rows", evaluate)
    with pytest.raises(RuntimeError):
        asyncio.run(reward.reward_func(args, SimpleNamespace(
            metadata=metadata, response=response(candidate()))))
    assert not list(Path(args.harness_r1_eval_root).rglob("*.json"))
    assert not list(Path(args.harness_r1_eval_root).rglob("*.lock"))


def test_baseline_builder_validates_entire_source(frozen):
    args, metadata, report, root = frozen
    for field, value in (("task_id", "test-task"), ("question", "different"),
                         ("reward", 0.5), ("rollout_seed", "1")):
        bad = copy.deepcopy(report)
        bad["results"][0][field] = value
        with pytest.raises(ValueError):
            build_rl_rows(bad, incumbent_patch=metadata["incumbent_patch"],
                          target_model_sha256="frozen-target",
                          data_manifest=root / "manifest.json", packet_size=1)
    assert metadata["source_baseline_sha256"] == digest(report)
    assert metadata["packet_selection"]["omitted_failures"] == 0


def test_public_failure_explicit_bounding(frozen):
    _, _, report, _ = frozen
    result = copy.deepcopy(report["results"][0])
    result["messages"] = [{"role": "assistant", "content": "x" * 3000}] * 65
    result["truncated"] = True
    public = public_failure(result)
    assert public["trajectory_bounds"] == {
        "original_messages": 65, "retained_messages": 64,
        "messages_truncated": True, "source_episode_truncated": True}
    assert public["trajectory_messages"][0]["content_truncation"]["original_chars"] == 3000
    assert "task_id" not in json.dumps(public)


def test_exact_prompt_token_budget_rejects_overflow_without_clipping(frozen):
    _, metadata, report, root = frozen
    kwargs = dict(incumbent_patch=metadata["incumbent_patch"],
                  target_model_sha256="frozen-target",
                  data_manifest=root / "manifest.json", packet_size=2)
    seen = []
    def counter(messages):
        seen.append(copy.deepcopy(messages))
        return 100
    rows = build_rl_rows(report, **kwargs, prompt_token_limit=100, prompt_token_counter=counter)
    assert rows[0]["prompt"] == seen[0]
    assert rows[0]["metadata"]["prompt_bounds"] == {
        "token_budget_checked": True, "token_count": 100, "token_limit": 100}
    with pytest.raises(ValueError, match="never silently clip"):
        build_rl_rows(report, **kwargs, prompt_token_limit=99, prompt_token_counter=counter)
    assert seen[0] == seen[1]
    with pytest.raises(ValueError, match="supplied together"):
        build_rl_rows(report, **kwargs, prompt_token_limit=100)
    with pytest.raises(ValueError, match="positive integer"):
        build_rl_rows(report, **kwargs, prompt_token_limit=100,
                      prompt_token_counter=lambda messages: None)


def test_prompt_literal_output_example_passes_strict_official_contract():
    import ast
    from harness_r1_searchqa.prompt import OUTPUT_SHAPE_EXAMPLE
    normalized = patch.extract_and_normalize_patch(OUTPUT_SHAPE_EXAMPLE)
    assert len(normalized["actions"]) == 1
    action = normalized["actions"][0]
    assert set(action) == {"type", "hook", "code"}
    assert action["type"] == "add_code_hook" and action["hook"] == "make_pre_hint"
    tree = ast.parse(action["code"])
    assert len(tree.body) == 1
    assert isinstance(tree.body[0], ast.FunctionDef) and tree.body[0].name == "hook"
    assert [arg.arg for arg in tree.body[0].args.args] == ["ctx", "nb"]
    assert "task_id" not in action["code"] and "ctx.question" not in action["code"]
    assert OUTPUT_SHAPE_EXAMPLE.count("</think>") == 1
    assert OUTPUT_SHAPE_EXAMPLE.count("<patch>") == 1
    assert OUTPUT_SHAPE_EXAMPLE.endswith("</patch>") and "```" not in OUTPUT_SHAPE_EXAMPLE


def test_prompt_emphasizes_actual_response_shape_and_rejects_pilot_mistakes():
    from harness_r1_searchqa.prompt import build_messages, OUTPUT_SHAPE_EXAMPLE
    messages = build_messages([{"question": "A public question?"}])
    contract = " ".join(messages[1]["content"].split())
    assert OUTPUT_SHAPE_EXAMPLE in messages[1]["content"]
    for phrase in (
        "exactly </think><patch>", "Both closing tags are mandatory",
        "Do not return bare JSON", "function named exactly def hook(ctx, nb):",
        "never write def on_post_step", "ONLY the three keys type, hook, code",
        "do not add name, description", "nested function definitions",
        "Do not write Markdown",
    ):
        assert phrase in contract
    assert "do not emit another opening <think>" in messages[0]["content"]


def test_prompt_documents_string_observation_and_fixed_one_search_budget():
    from harness_r1_searchqa.prompt import PATCH_SCHEMA
    contract = " ".join(PATCH_SCHEMA.split())
    for phrase in (
        "max_steps=2, max_tool_calls=1, top_k=1",
        "remaining_steps is NOT remaining tool calls",
        "ctx.observation and ctx.last_observation are strings",
        "JSON-encoded public search observation",
        "inspect them as text, not a documents dictionary",
        "There is no ctx.documents or ctx.retrieved_documents field",
        "only ONE search call in total",
        "do NOT demand another search, retry",
        "force_action cannot increase this budget",
        'ctx.task["max_tool_calls"] (always 1)',
        "ctx.task has no remaining_tool_calls field",
        "Never call .get(...) on ctx.last_observation or ctx.observation",
    ):
        assert phrase in contract
