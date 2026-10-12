"""CPU/Gloo coverage plus an opt-in two-H200 native numerical-equivalence test.

GPU smoke (only after external node3/node4 allocation):
  CUDA_VISIBLE_DEVICES=0,1 REWRITE_SFT_TWO_GPU=1 PYTHONPATH=whale_local:tests \
    .venv-ppo/bin/python -m unittest \
    test_rewrite_sft_distributed.DistributedSFTTests.test_opt_in_two_gpu_native_equivalence

Neither test nor trainer manages utilization services. CPU tests spawn two
torchrun workers, using only project-local test files and localhost rendezvous.
"""
import contextlib
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "whale_local"))
from sao_bridge import rewrite_sft_train as sft


def calls(count=6):
    return [{
        "id": str(i), "split": "train", "prompt_ids": [3, 4, 5][:1 + i % 3],
        "response_ids": [6 + i % 5] * (1 if i % 2 == 0 else 10) + [2],
        "temperature": 1.0,
    } for i in range(count)]


class DistributedSFTTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / "tests" / (".rewrite-distributed-" + uuid.uuid4().hex)
        self.directory.mkdir()
        self.addCleanup(shutil.rmtree, self.directory)

    def test_single_node_rank_environment(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(sft.DistributedRun.from_environment(), sft.DistributedRun())
        base = {"RANK": "1", "LOCAL_RANK": "1", "WORLD_SIZE": "8", "LOCAL_WORLD_SIZE": "8"}
        with patch.dict(os.environ, base, clear=True):
            run = sft.DistributedRun.from_environment()
            self.assertEqual((run.rank, run.local_rank, run.world_size), (1, 1, 8))
        for update in ({"WORLD_SIZE": "9", "LOCAL_WORLD_SIZE": "9"},
                       {"LOCAL_WORLD_SIZE": "4"}, {"RANK": "2"}, {"GROUP_RANK": "1"}):
            with self.subTest(update=update), patch.dict(os.environ, {**base, **update}, clear=True):
                with self.assertRaisesRegex(ValueError, "single-node"):
                    sft.DistributedRun.from_environment()
        with patch.dict(os.environ, {"RANK": "0"}, clear=True):
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                sft.DistributedRun.from_environment()

    def test_production_and_calibration_have_exact_global_coverage(self):
        for count, updates in ((512, 4), (28672, 224)):
            data = calls(count)
            expected = None
            for world in (1, 2, 4, 8):
                plan = sft.plan_batches(data, 1, 128, 42, 32, world)
                if expected is None:
                    expected = plan
                self.assertEqual(plan, expected)
                self.assertEqual(len(plan), updates)
                visited = []
                for group in plan:
                    shards = [sft.rank_indices(group["indices"], r, world) for r in range(world)]
                    self.assertEqual([len(shard) for shard in shards], [128 // world] * world)
                    visited.extend(i for shard in shards for i in shard)
                self.assertEqual(sorted(visited), list(range(count)))
                self.assertEqual(len(set(visited)), count)

    def test_uneven_rank_groups_are_explicitly_rejected_but_divisible_remainders_work(self):
        for count, batch, world in ((5, 4, 2), (6, 3, 2), (6, 4, 8), (8, 8, 9)):
            with self.subTest(count=count, batch=batch, world=world):
                with self.assertRaisesRegex(ValueError, "divisible"):
                    sft.plan_batches(calls(count), 1, batch, 42, 32, world)
        plan = sft.plan_batches(calls(6), 1, 4, 42, 32, 2)
        self.assertEqual([len(group["indices"]) for group in plan], [4, 2])
        with self.assertRaises(ValueError):
            sft.rank_indices([0], 0, 2)

    def run_workers(self, mode):
        env = {
            **os.environ, "PYTHONPATH": str(ROOT / "whale_local"),
            "TMPDIR": str(self.directory), "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        }
        if mode == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        command = [
            sys.executable, "-B", "-m", "torch.distributed.run",
            "--standalone", "--local-addr=127.0.0.1", "--nnodes=1",
            "--nproc-per-node=2", "--max-restarts=0",
            str(Path(__file__).resolve()), "--worker", mode, str(self.directory),
        ]
        with subprocess.Popen(command, cwd=ROOT, env=env, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
            try:
                stdout, stderr = process.communicate(timeout=180)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    stdout, stderr = process.communicate(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, stderr = process.communicate()
                self.fail("CPU/GPU torchrun timed out:\n" + stdout + "\n" + stderr)
        self.assertEqual(process.returncode, 0, stdout + "\n" + stderr)
        self.assertIn(f"DISTRIBUTED_{mode.upper()}_OK", stdout)
        report = json.loads((self.directory / f"{mode}_equivalence.json").read_text())
        self.assertEqual(report["world_size"], 2)
        self.assertTrue(report["unequal_rank_token_counts"])
        self.assertTrue(report["parameters_and_fp32_moments_match"])

    def test_cpu_torchrun_native_equivalence_failure_propagation_and_checkpoint(self):
        self.run_workers("cpu")

    @unittest.skipUnless(os.environ.get("REWRITE_SFT_TWO_GPU") == "1",
                         "Opt-in only; requires two externally allocated GPUs on node3/node4")
    def test_opt_in_two_gpu_native_equivalence(self):
        from sao_bridge.resources import require_research_node

        require_research_node()
        self.run_workers("gpu")


def numerical_equivalence(run, directory, device, mode):
    import torch
    from transformers import AutoModelForCausalLM, Qwen3Config

    class TokenLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.table = torch.nn.Embedding(32, 32)

        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(logits=self.table(input_ids))

    torch.manual_seed(100)
    if mode != "cpu":
        reference = AutoModelForCausalLM.from_config(Qwen3Config(
            vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            max_position_embeddings=64, eos_token_id=2, pad_token_id=0,
            attention_dropout=0.0,
        ), attn_implementation="sdpa").to(device)
        reference.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        reference = TokenLM()
    model = copy.deepcopy(reference)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-5, foreach=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, foreach=False)
    reference_actor = sft.make_actor(reference, reference_optimizer, 4)
    actor = sft.make_actor(model, optimizer, 2)
    reference_gradients, gradients = [], []

    def capture(target_actor, snapshots):
        original = target_actor._optimizer_step

        def step():
            snapshots.append([p.grad.detach().clone() for p in target_actor.actor_module.parameters()])
            return original()

        target_actor._optimizer_step = step

    capture(reference_actor, reference_gradients)
    capture(actor, gradients)
    data = calls()
    # Deliberately unbalanced token counts per rank, including a divisible remainder.
    plan = [{"epoch": 1, "indices": [0, 1, 2, 3]}, {"epoch": 1, "indices": [4, 5]}]
    expected = sft.run_updates(reference_actor, reference_optimizer, data, plan, 0, 32)
    actual = sft.run_updates(actor, optimizer, data, plan, 0, 32, run=run)
    max_gradient_delta, max_parameter_delta = 0.0, 0.0
    # BF16 backward rounds the native local token normalization before FP32
    # reduction. FP32 is checked tightly; BF16 uses a whole-gradient relative
    # norm bound rather than meaningless relative errors near zero elements.
    tolerance = dict(rtol=2e-6, atol=2e-7)
    gradient_relative_errors = []

    def relative_error(left, right):
        baseline = torch.cat([value.detach().float().reshape(-1) for value in left])
        candidate = torch.cat([value.detach().float().reshape(-1) for value in right])
        return float((baseline - candidate).norm() / baseline.norm().clamp_min(1e-12))

    for baseline, distributed in zip(reference_gradients, gradients, strict=True):
        gradient_relative_errors.append(relative_error(baseline, distributed))
        assert gradient_relative_errors[-1] < (0.02 if mode != "cpu" else 2e-6)
        for left, right in zip(baseline, distributed, strict=True):
            max_gradient_delta = max(max_gradient_delta, (left - right).abs().max().item())
            if mode == "cpu":
                torch.testing.assert_close(left, right, **tolerance)
    for left, right in zip(reference.parameters(), model.parameters(), strict=True):
        max_parameter_delta = max(max_parameter_delta, (left - right).abs().max().item())
        torch.testing.assert_close(left, right, rtol=0, atol=3e-5 if mode != "cpu" else 1e-7)
        baseline, distributed = reference_optimizer.state[left], optimizer.state[right]
        assert distributed["step"].item() == 2
        for key in ("exp_avg", "exp_avg_sq"):
            assert distributed[key].dtype == torch.float32
            if mode == "cpu":
                torch.testing.assert_close(baseline[key], distributed[key], **tolerance)
    for key in ("exp_avg", "exp_avg_sq"):
        error = relative_error(
            [reference_optimizer.state[p][key] for p in reference.parameters()],
            [optimizer.state[p][key] for p in model.parameters()],
        )
        assert error < (0.03 if mode != "cpu" else 2e-6), (key, error)
    assert [record["response_tokens"] for record in actual] == [record["response_tokens"] for record in expected]
    assert sum(record["examples"] for record in actual) == len(data)
    assert actual[0]["rank_metrics"][0]["response_tokens"] != actual[0]["rank_metrics"][1]["response_tokens"]
    for baseline, distributed in zip(expected, actual, strict=True):
        assert abs(baseline["native"]["actor/sft_loss"] - distributed["native"]["actor/sft_loss"]) < 0.02
    run.agree("updated replica parameters", [
        sft.hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest() for p in model.parameters()
    ])
    run.root_phase("equivalence report", lambda: sft._write(directory / f"{mode}_equivalence.json", {
        "world_size": run.world_size, "mode": mode, "unequal_rank_token_counts": True,
        "parameters_and_fp32_moments_match": True, "max_gradient_abs_delta": max_gradient_delta,
        "max_parameter_abs_delta": max_parameter_delta, "fp32_elementwise_tolerance": tolerance,
        "gradient_relative_l2_errors": gradient_relative_errors,
        "bf16_gradient_relative_l2_limit": 0.02,
    }))
    return model, optimizer


def failure_checks(run, model):
    import torch

    def fail_on_root():
        raise OSError("injected rank-zero write failure")

    try:
        run.root_phase("write failure test", fail_on_root)
    except RuntimeError as exc:
        assert "rank 0" in str(exc) and "injected rank-zero write failure" in str(exc)
    else:
        raise AssertionError("All ranks must observe a rank-zero write failure")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, foreach=False)
    actor = sft.make_actor(model, optimizer, 2)
    original_forward = actor._forward_micro_batch

    def maybe_fail(*args, **kwargs):
        if run.rank == 1:
            raise RuntimeError("injected rank-one forward failure")
        return original_forward(*args, **kwargs)

    actor._forward_micro_batch = maybe_fail
    original_step = actor._optimizer_step
    started = time.monotonic()
    try:
        sft.run_updates(actor, optimizer, calls(4), [{"epoch": 1, "indices": [0, 1, 2, 3]}],
                        0, 32, run=run)
    except RuntimeError as exc:
        assert "rank 1" in str(exc) and "injected rank-one forward failure" in str(exc)
    else:
        raise AssertionError("All ranks must observe a pre-reduction backward failure")
    assert time.monotonic() - started < 15
    assert actor._optimizer_step == original_step
    assert not optimizer.state
    run.agree("both ranks recovered from coordinated failure", True)

    actor._forward_micro_batch = original_forward
    original_reduce = sft.reduce_token_weighted_gradients

    def tiny_buckets(module, local_tokens, global_tokens):
        return original_reduce(module, local_tokens, global_tokens, bucket_bytes=60)

    with patch.object(sft, "reduce_token_weighted_gradients", tiny_buckets):
        sft.run_updates(actor, optimizer, calls(4), [{"epoch": 1, "indices": [0, 1, 2, 3]}],
                        0, 32, run=run)
    run.agree("replicas after small multi-bucket reduction", [
        sft.hashlib.sha256(p.detach().numpy().tobytes()).hexdigest() for p in model.parameters()
    ])

    # A missing gradient must fail collectively before any gradient all_reduce.
    model.unused = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, foreach=False)
    actor = sft.make_actor(model, optimizer, 2)
    original_forward = actor._forward_micro_batch

    def asymmetric_unused(*args, **kwargs):
        result = original_forward(*args, **kwargs)
        if run.rank == 0:
            result["log_probs"] = result["log_probs"] + model.unused * 0
        return result

    actor._forward_micro_batch = asymmetric_unused
    try:
        sft.run_updates(actor, optimizer, calls(4), [{"epoch": 1, "indices": [0, 1, 2, 3]}],
                        0, 32, run=run)
    except RuntimeError as exc:
        assert "rank 1" in str(exc) and "FP32 gradient" in str(exc)
    else:
        raise AssertionError("Missing-gradient mismatch must fail on every rank")
    assert not optimizer.state


def cpu_main_integration(run, directory):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Qwen3Config
    from test_rewrite_sft_train import Tokenizer, example

    model_path, data_path = directory / "input", directory / "data.json"
    config = Qwen3Config(
        vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=64, eos_token_id=2, pad_token_id=0,
    )

    def setup():
        model_path.mkdir()
        config.save_pretrained(model_path)
        sft._write(data_path, [example("train_" + str(i)) for i in range(6)] +
                   [example("dev", "dev"), example("test", "test")])

    run.root_phase("fixture creation", setup)
    original_write = sft._write

    def rank_checked_write(*args):
        assert run.rank == 0, "Only rank zero may persist shared output"
        return original_write(*args)

    @contextlib.contextmanager
    def reuse_group(current_run, *args):
        current_run.control_group = run.control_group
        yield

    for fail_save in (False, True):
        torch.manual_seed(10)
        model = AutoModelForCausalLM.from_config(config, attn_implementation="sdpa")
        output = directory / ("failed_output" if fail_save else "output")
        arguments = [
            "--model", str(model_path), "--data", str(data_path), "--output", str(output),
            "--batch-size", "4", "--epochs", "2", "--max-length", "32",
        ]
        original_verify = sft.verify_saved_probes

        def verify(*args):
            if fail_save:
                raise RuntimeError("injected checkpoint reload mismatch")
            original_verify(*args)

        with patch.object(AutoConfig, "from_pretrained", return_value=config), \
                patch.object(AutoTokenizer, "from_pretrained", return_value=Tokenizer()), \
                patch.object(AutoModelForCausalLM, "from_pretrained", return_value=model), \
                patch.object(model, "to", return_value=model), \
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}), \
                patch("torch.cuda.device_count", return_value=2), \
                patch("torch.cuda.is_bf16_supported", return_value=True), \
                patch("torch.cuda.set_device"), patch("torch.cuda.manual_seed_all"), \
                patch("torch.cuda.reset_peak_memory_stats"), \
                patch("torch.cuda.max_memory_allocated", return_value=0), \
                patch("sao_bridge.resources.require_research_node"), \
                patch.object(sft, "distributed_process_group", reuse_group), \
                patch.object(sft, "_write", rank_checked_write), \
                patch.object(sft, "verify_saved_probes", verify):
            try:
                sft.main(arguments)
            except RuntimeError as exc:
                assert fail_save and "injected checkpoint reload mismatch" in str(exc)
            else:
                assert not fail_save

        def validate_output():
            result = json.loads((output / "result.json").read_text())
            if fail_save:
                assert result["status"] == "failed" and not result["updated"]
                return
            assert result["world_size"] == 2 and result["optimizer_updates"] == 4
            assert result["examples_seen"] == 12 and result["response_tokens"] == 36
            assert result["replica_probes_agree"] and result["saved_probes_verified"]
            assert result["hf_cpu_reload_verified"] and result["tokenizer_reload_verified"]
            assert result["training_seconds"] > 0 and result["examples_per_training_second"] > 0
            provenance = json.loads((output / "provenance.json").read_text())
            assert provenance["writer_rank"] == 0 and provenance["dropped_records"] == 0
            metrics = json.loads((output / "metrics.json").read_text())
            for group in metrics:
                assert sorted(sum(group["rank_ids"], [])) == sorted(group["ids"])
            saved = AutoModelForCausalLM.from_pretrained(output / "checkpoint", local_files_only=True)
            for left, right in zip(saved.parameters(), model.parameters(), strict=True):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            assert (output / "checkpoint/tokenizer_config.json").exists()
            row = example("train_0")
            with patch.object(AutoTokenizer, "from_pretrained", return_value=Tokenizer()):
                sft.verify_hf_reload(output / "checkpoint", sft.parameter_probes(model), row,
                                     sft.tokenize_training_rows([row], Tokenizer(), 32)[0], 0)

        run.root_phase("persisted output validation", validate_output)
    assert not torch.cuda.is_initialized()


def worker(mode, directory):
    if mode == "gpu":
        from sao_bridge.resources import require_research_node

        require_research_node()
    import torch
    from sao_bridge.legacy_update import DataParallelPPOActor

    assert DataParallelPPOActor.update_sft_policy
    torch.set_num_threads(1)
    run = sft.DistributedRun.from_environment()
    assert run.world_size == 2
    if mode == "gpu":
        assert torch.cuda.device_count() == 2
        torch.cuda.set_device(run.local_rank)
        assert torch.cuda.is_bf16_supported()
        device, backend = torch.device("cuda", run.local_rank), "nccl"
    else:
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
        device, backend = torch.device("cpu"), "gloo"
    with sft.distributed_process_group(run, timeout_seconds=60, backend=backend):
        with patch("verl.workers.actor.dp_actor.get_device_id", return_value=device), \
                patch("verl.utils.torch_functional.FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE", False):
            model, _ = numerical_equivalence(run, directory, device, mode)
            if mode == "cpu":
                numerical_equivalence(run, directory, device, "cpu_bf16")
                failure_checks(run, model)
                cpu_main_integration(run, directory)
        if run.rank == 0:
            print(f"DISTRIBUTED_{mode.upper()}_OK", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(sys.argv[2], Path(sys.argv[3]))
    else:
        unittest.main()
