"""Optimizer continuation, integrity checks and Gloo failure propagation.

Run CPU tests with CUDA_VISIBLE_DEVICES='' before importing the native actor;
its dependencies cache accelerator availability and may inspect CUDA at import.

Opt-in tiny Qwen GPU continuation (one externally allocated GPU on node3/node4):
  CUDA_VISIBLE_DEVICES=0 REWRITE_SFT_RESUME_GPU=1 PYTHONPATH=whale_local:tests \
    TMPDIR=/dev/shm PYTHONDONTWRITEBYTECODE=1 /dev/shm/sao-venv-ppo/bin/python -B \
    -m unittest test_rewrite_sft_resume.ResumeSFTTests.test_opt_in_gpu_two_stage_native_continuity

The GPU smoke compares three continuous updates with two updates, a real HF and
CPU optimizer-state save/reload, and one resumed update. No production weights,
teacher data, external evaluation service or utilization service are used.
"""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "whale_local"))
from sao_bridge import rewrite_sft_train as sft
from test_rewrite_sft_train import Tokenizer, example


class UnsafeState:
    def __setstate__(self, state):
        raise AssertionError("Untrusted pickle must never execute")


def tiny_model():
    import torch

    class TokenLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.table = torch.nn.Embedding(16, 16)

        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(logits=self.table(input_ids))

    return TokenLM()


def calls(count=6):
    return [{
        "id": str(i), "split": "train", "prompt_ids": [3, 4][:1 + i % 2],
        "response_ids": [6 + i % 5] * (1 + i % 3) + [2], "temperature": 1.0,
    } for i in range(count)]


def compatibility(world_size=1):
    return sft.training_compatibility(
        SimpleNamespace(batch_size=2, max_length=32), sft.DistributedRun(world_size=world_size),
    )


def save_checkpoint(model, checkpoint):
    from safetensors.torch import save_file

    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model_type": "test"}\n')
    save_file(model.state_dict(), str(checkpoint / "model.safetensors"))


@contextlib.contextmanager
def native_cpu():
    import torch
    from sao_bridge.legacy_update import DataParallelPPOActor

    assert DataParallelPPOActor.update_sft_policy
    with patch("torch.distributed.get_rank", return_value=0), \
            patch("verl.workers.actor.dp_actor.get_device_id", return_value=torch.device("cpu")), \
            patch("verl.utils.torch_functional.FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE", False):
        yield


class ResumeSFTTests(unittest.TestCase):
    def setUp(self):
        import torch

        torch.set_num_threads(1)
        torch.manual_seed(71)
        self.directory = ROOT / "tests" / (".rewrite-resume-" + uuid.uuid4().hex)
        self.directory.mkdir()
        self.addCleanup(shutil.rmtree, self.directory)

    def fixture(self):
        import torch

        model = tiny_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False)
        with native_cpu():
            sft.run_updates(sft.make_actor(model, optimizer, 2), optimizer, calls(2),
                            [{"epoch": 1, "indices": [0, 1]}], 0, 32)
        checkpoint = self.directory / "checkpoint"
        save_checkpoint(model, checkpoint)
        path = self.directory / "training_state.pt"
        sft.save_training_state(path, checkpoint, model, optimizer, compatibility(), 1, ["0", "1"])
        return model, optimizer, checkpoint, path

    def rewrite_state(self, path, payload):
        import torch

        torch.save(payload, path)
        path.with_name(path.name + ".sha256").write_text(sft._sha(path) + "\n")

    def test_cpu_state_roundtrip_does_not_initialize_cuda_even_if_reported_available(self):
        import torch

        model = tiny_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False)
        for parameter in model.parameters():
            optimizer.state[parameter] = {
                "step": torch.tensor(1.0), "exp_avg": torch.full_like(parameter, 0.01),
                "exp_avg_sq": torch.full_like(parameter, 0.001),
            }
        restored = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False)
        checkpoint, path = self.directory / "checkpoint", self.directory / "training_state.pt"
        with patch.object(torch.cuda, "is_available", return_value=True), \
                patch.object(torch.cuda, "_lazy_init",
                             side_effect=AssertionError("CPU state path initialized CUDA")) as initialize:
            save_checkpoint(model, checkpoint)
            sft.save_training_state(path, checkpoint, model, optimizer, compatibility(), 1, ["old"])
            self.assertEqual(
                sft.restore_training_state(path, checkpoint, model, restored, compatibility(), ["new"]),
                (1, ["old"]),
            )
            initialize.assert_not_called()
        for values in restored.state.values():
            self.assertTrue(all(value.device.type == "cpu" for value in values.values()))
        self.assertFalse(torch.cuda.is_initialized())

    def hf_continuity_oracle(self, device):
        import torch
        from transformers import AutoModelForCausalLM, Qwen3Config

        config = Qwen3Config(
            vocab_size=16, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            max_position_embeddings=64, eos_token_id=2, pad_token_id=0,
            attention_dropout=0.0, use_cache=False,
        )
        continuous = AutoModelForCausalLM.from_config(config, attn_implementation="sdpa").to(device)
        continuous.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        first = copy.deepcopy(continuous)
        expected_optimizer = torch.optim.AdamW(continuous.parameters(), lr=2e-6, foreach=False)
        first_optimizer = torch.optim.AdamW(first.parameters(), lr=2e-6, foreach=False)
        data = calls()
        plan = [{"epoch": 1, "indices": [i, i + 1]} for i in (0, 2, 4)]
        sft.run_updates(sft.make_actor(continuous, expected_optimizer, 2),
                        expected_optimizer, data, plan, 0, 32)
        sft.run_updates(sft.make_actor(first, first_optimizer, 2),
                        first_optimizer, data, plan[:2], 0, 32)
        checkpoint, path = self.directory / "checkpoint", self.directory / "training_state.pt"
        first.save_pretrained(checkpoint, safe_serialization=True)
        sft.save_training_state(path, checkpoint, first, first_optimizer,
                                compatibility(), 2, ["0", "1", "2", "3"])
        payload = torch.load(path, map_location="cpu", weights_only=True)
        for values in payload["optimizer"]["state"].values():
            self.assertTrue(all(value.device.type == "cpu" for value in values.values()))
        resumed = AutoModelForCausalLM.from_pretrained(
            checkpoint, local_files_only=True, dtype=torch.float32, attn_implementation="sdpa",
        ).to(device)
        resumed.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        optimizer = torch.optim.AdamW(resumed.parameters(), lr=2e-6, foreach=False)
        updates, ids = sft.restore_training_state(
            path, checkpoint, resumed, optimizer, compatibility(), ["4", "5"],
        )
        self.assertEqual((updates, ids), (2, ["0", "1", "2", "3"]))
        for parameter in resumed.parameters():
            for key in ("exp_avg", "exp_avg_sq"):
                self.assertEqual(optimizer.state[parameter][key].device, parameter.device)
        metrics = sft.run_updates(
            sft.make_actor(resumed, optimizer, 2), optimizer, data[4:],
            [{"epoch": 1, "indices": [0, 1]}], 0, 32, initial_updates=updates,
        )
        self.assertEqual(metrics[0]["optimizer_step"], 3)
        for actual, expected in zip(resumed.parameters(), continuous.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(optimizer.state[actual][key],
                                           expected_optimizer.state[expected][key], rtol=2e-6, atol=2e-7)

    def test_tiny_hf_two_stage_continuity_on_cpu(self):
        import torch

        with native_cpu():
            self.hf_continuity_oracle("cpu")
        self.assertFalse(torch.cuda.is_initialized())

    @unittest.skipUnless(os.environ.get("REWRITE_SFT_RESUME_GPU") == "1",
                         "Opt-in only; requires one externally allocated GPU on node3/node4")
    def test_opt_in_gpu_two_stage_native_continuity(self):
        import torch
        from sao_bridge.resources import require_research_node

        require_research_node()
        self.assertEqual(torch.cuda.device_count(), 1, "Select exactly one externally allocated GPU")
        torch.cuda.set_device(0)
        self.assertTrue(torch.cuda.is_bf16_supported())
        with sft.single_rank_process_group():
            self.hf_continuity_oracle("cuda")

    def test_two_stage_updates_and_fp32_moments_equal_continuous_native_oracle(self):
        import torch
        from safetensors.torch import load_file

        continuous = tiny_model()
        first = copy.deepcopy(continuous)
        expected_optimizer = torch.optim.AdamW(continuous.parameters(), lr=0.01, foreach=False)
        first_optimizer = torch.optim.AdamW(first.parameters(), lr=0.01, foreach=False)
        data = calls()
        plan = [{"epoch": 1, "indices": [i, i + 1]} for i in (0, 2, 4)]
        with native_cpu():
            sft.run_updates(sft.make_actor(continuous, expected_optimizer, 2),
                            expected_optimizer, data, plan, 0, 32)
            sft.run_updates(sft.make_actor(first, first_optimizer, 2),
                            first_optimizer, data, plan[:2], 0, 32)
            checkpoint = self.directory / "checkpoint"
            save_checkpoint(first, checkpoint)
            path = self.directory / "training_state.pt"
            sft.save_training_state(path, checkpoint, first, first_optimizer,
                                    compatibility(), 2, ["0", "1", "2", "3"])
            payload = torch.load(path, weights_only=True, map_location="cpu")
            for values in payload["optimizer"]["state"].values():
                self.assertTrue(all(value.device.type == "cpu" for value in values.values()))
            resumed = tiny_model()
            resumed.load_state_dict(load_file(str(checkpoint / "model.safetensors")))
            optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.01, foreach=False)
            for source in (path, self.directory):
                with patch.object(torch, "load", wraps=torch.load) as load:
                    updates, ids = sft.restore_training_state(
                        source, checkpoint, resumed, optimizer, compatibility(), ["4", "5"],
                    )
                self.assertEqual(load.call_args.kwargs, {"map_location": "cpu", "weights_only": True})
                self.assertEqual((updates, ids), (2, ["0", "1", "2", "3"]))
            metrics = sft.run_updates(
                sft.make_actor(resumed, optimizer, 2), optimizer, data[4:],
                [{"epoch": 1, "indices": [0, 1]}], 0, 32, initial_updates=updates,
            )
        self.assertEqual((metrics[0]["update"], metrics[0]["optimizer_step"]), (1, 3))
        for actual, expected in zip(resumed.parameters(), continuous.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(optimizer.state[actual][key],
                                           expected_optimizer.state[expected][key], rtol=0, atol=0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_corruption_and_unsafe_pickle_fail_without_reset_or_unsafe_load(self):
        import pickle
        import torch

        model, _, checkpoint, path = self.fixture()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False)
        original = path.read_bytes()
        path.write_bytes(original[:100] + bytes([original[100] ^ 1]) + original[101:])
        with patch.object(torch, "load", wraps=torch.load) as load:
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                sft.restore_training_state(path, checkpoint, model, optimizer, compatibility(), ["new"])
            load.assert_not_called()
        self.assertFalse(optimizer.state)
        unsafe = UnsafeState()
        unsafe.value = 1
        self.rewrite_state(path, unsafe)
        with patch.object(torch, "load", wraps=torch.load) as load:
            with self.assertRaises(pickle.UnpicklingError):
                sft.restore_training_state(path, checkpoint, model, optimizer, compatibility(), ["new"])
            self.assertEqual(load.call_count, 1)
            self.assertTrue(load.call_args.kwargs["weights_only"])
        self.assertFalse(optimizer.state)
        path.with_name(path.name + ".sha256").unlink()
        with self.assertRaises(FileNotFoundError):
            sft.restore_training_state(path, checkpoint, model, optimizer, compatibility(), ["new"])

    def test_checkpoint_model_hyperparameters_and_consumed_ids_must_match(self):
        import torch

        model, _, checkpoint, path = self.fixture()
        for key, value in (("world_size", 2), ("global_batch_size", 4), ("max_length", 64),
                           ("grad_clip", 0.5), ("master_dtype", "bfloat16")):
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "hyperparameters/world_size"):
                sft.restore_training_state(path, checkpoint, model, optimizer,
                                           {**compatibility(), key: value}, ["new"])
            self.assertFalse(optimizer.state)
        for kwargs in ({"lr": 0.02}, {"betas": (0.8, 0.999)}, {"eps": 1e-6},
                       {"weight_decay": 0}, {"foreach": True}):
            optimizer = torch.optim.AdamW(model.parameters(), **{"lr": 0.01, "foreach": False, **kwargs})
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "parameter groups"):
                sft.restore_training_state(path, checkpoint, model, optimizer, compatibility(), ["new"])
            self.assertFalse(optimizer.state)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False)
        with self.assertRaisesRegex(ValueError, "already consumed"):
            sft.restore_training_state(path, checkpoint, model, optimizer, compatibility(), ["1"])
        different = torch.nn.Linear(16, 16)
        with self.assertRaisesRegex(ValueError, "parameter identity"):
            sft.restore_training_state(path, checkpoint, different, optimizer, compatibility(), ["new"])
        with torch.no_grad():
            model.table.weight.add_(0.01)
        with self.assertRaisesRegex(ValueError, "checkpoint probes"):
            sft.restore_training_state(path, checkpoint, model, optimizer, compatibility(), ["new"])
        (checkpoint / "config.json").write_text('{"model_type": "changed"}')
        with self.assertRaisesRegex(ValueError, "checkpoint identity"):
            sft.restore_training_state(path, checkpoint, model, optimizer, compatibility(), ["new"])
        self.assertFalse(optimizer.state)

    def test_invalid_state_schema_counters_and_moments_are_rejected(self):
        import torch

        model, _, checkpoint, path = self.fixture()
        original = torch.load(path, map_location="cpu", weights_only=True)
        mutations = [
            lambda p: p.update(format="unknown"),
            lambda p: p.update(completed_updates=2),
            lambda p: p.update(completed_updates=True),
            lambda p: p.update(trained_ids=["0", "0"]),
            lambda p: p.update(probes={}),
            lambda p: p["optimizer"]["state"].clear(),
            lambda p: p["optimizer"]["state"][0].pop("exp_avg"),
            lambda p: p["optimizer"]["state"][0].update(step=torch.tensor(0.0)),
            lambda p: p["optimizer"]["state"][0].update(exp_avg=torch.zeros(1)),
            lambda p: p["optimizer"]["state"][0].update(
                exp_avg=p["optimizer"]["state"][0]["exp_avg"].double()),
            lambda p: p["optimizer"]["state"][0]["exp_avg"].fill_(float("nan")),
            lambda p: p["optimizer"]["state"][0]["exp_avg_sq"].fill_(-1),
        ]
        for index, mutate in enumerate(mutations):
            payload = copy.deepcopy(original)
            mutate(payload)
            self.rewrite_state(path, payload)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False)
            with self.subTest(index=index), self.assertRaises(ValueError):
                sft.restore_training_state(path, checkpoint, model, optimizer, compatibility(), ["new"])
            self.assertFalse(optimizer.state)

    def test_cli_paths_one_epoch_and_opt_in_defaults(self):
        required = ["--model", "model", "--data", "data", "--output", "new"]
        defaults = sft.parse_args(required)
        self.assertFalse(defaults.save_training_state)
        self.assertIsNone(defaults.resume_state)
        self.assertEqual(sft.parse_args([*required, "--epochs", "3"]).epochs, 3)
        for flags in (["--save-training-state"], ["--resume-state", "old"]):
            self.assertEqual(sft.parse_args([*required, *flags]).epochs, 1)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                sft.parse_args([*required, *flags, "--epochs", "2"])
        _, _, checkpoint, state = self.fixture()
        data = self.directory / "data.json"
        data.write_text("[]")
        for source in (state, self.directory):
            args = sft.parse_args([
                "--model", str(checkpoint), "--data", str(data),
                "--output", str(self.directory / "new"), "--resume-state", str(source),
            ])
            sft.validate_paths(args)
            self.assertEqual(args.resume_state, state)
        args.resume_state = self.directory / "missing.pt"
        with self.assertRaisesRegex(FileNotFoundError, "Missing training state"):
            sft.validate_paths(args)
        args.resume_state = Path("/dev/shm/outside-shared-state.pt")
        with self.assertRaisesRegex(ValueError, "shared storage"):
            sft.validate_paths(args)

    def test_main_two_stages_load_real_hf_checkpoint_and_persist_cumulative_identity(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config

        config = Qwen3Config(
            vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            max_position_embeddings=64, eos_token_id=2, pad_token_id=0,
        )
        initial = AutoModelForCausalLM.from_config(config, attn_implementation="sdpa")
        input_model = self.directory / "input"
        initial.save_pretrained(input_model, safe_serialization=True)
        original_to = type(initial).to

        def cpu_to(model, *args, **kwargs):
            return original_to(model, *("cpu" if arg == "cuda" else arg for arg in args), **kwargs)

        with native_cpu(), patch.object(type(initial), "to", cpu_to), \
                patch.object(AutoTokenizer, "from_pretrained", return_value=Tokenizer()), \
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), \
                patch("torch.cuda.device_count", return_value=1), \
                patch("torch.cuda.is_bf16_supported", return_value=True), \
                patch("torch.cuda.set_device"), patch("torch.cuda.manual_seed_all"), \
                patch("torch.cuda.reset_peak_memory_stats"), \
                patch("torch.cuda.max_memory_allocated", return_value=0), \
                patch("sao_bridge.resources.require_research_node"), \
                patch.object(sft, "single_rank_process_group", return_value=contextlib.nullcontext()), \
                contextlib.redirect_stdout(io.StringIO()):
            previous = None
            for stage in (1, 2):
                data = self.directory / f"data{stage}.json"
                data.write_text(json.dumps([example(f"{stage}-{i}") for i in range(3)]))
                output = self.directory / f"stage{stage}"
                arguments = [
                    "--model", str(input_model), "--data", str(data), "--output", str(output),
                    "--batch-size", "2", "--max-length", "32", "--save-training-state",
                ]
                if previous is not None:
                    arguments += ["--resume-state", str(previous)]
                sft.main(arguments)
                result = json.loads((output / "result.json").read_text())
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["optimizer_updates"], 2)
                self.assertEqual(result["cumulative_optimizer_updates"], 2 * stage)
                self.assertEqual(result["cumulative_train_records"], 3 * stage)
                self.assertEqual(result["optimizer_reset"], stage == 1)
                metrics = json.loads((output / "metrics.json").read_text())
                self.assertEqual([row["optimizer_step"] for row in metrics],
                                 [2 * stage - 1, 2 * stage])
                payload = torch.load(output / "training_state.pt", map_location="cpu", weights_only=True)
                self.assertEqual(len(payload["trained_ids"]), 3 * stage)
                self.assertEqual(payload["completed_updates"], 2 * stage)
                self.assertEqual(payload["checkpoint_files_sha256"],
                                 sft._checkpoint_identity(output / "checkpoint"))
                previous, input_model = output, output / "checkpoint"
            with self.assertRaisesRegex(ValueError, "already consumed"):
                sft.main([
                    "--model", str(input_model), "--data", str(data),
                    "--output", str(self.directory / "duplicate"), "--resume-state", str(previous),
                    "--batch-size", "2", "--max-length", "32",
                ])
        failed = json.loads((self.directory / "duplicate/result.json").read_text())
        self.assertEqual(failed["status"], "failed")
        self.assertFalse((self.directory / "duplicate/checkpoint").exists())
        self.assertFalse(torch.cuda.is_initialized())

    def test_cpu_gloo_state_failures_propagate_to_every_rank(self):
        env = {
            **os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(ROOT / "whale_local") + ":" + str(ROOT / "tests"),
            "OMP_NUM_THREADS": "1",
        }
        command = [
            sys.executable, "-B", "-m", "torch.distributed.run", "--standalone",
            "--local-addr=127.0.0.1", "--nnodes=1", "--nproc-per-node=2", "--max-restarts=0",
            str(Path(__file__).resolve()), "--worker", str(self.directory),
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
                self.fail("Gloo state test timed out:\n" + stdout + "\n" + stderr)
        self.assertEqual(process.returncode, 0, stdout + "\n" + stderr)
        self.assertIn("DISTRIBUTED_RESUME_OK", stdout)


def distributed_worker(directory):
    import torch
    from sao_bridge.legacy_update import DataParallelPPOActor

    assert DataParallelPPOActor.update_sft_policy
    run = sft.DistributedRun.from_environment()
    torch.manual_seed(17)
    torch.set_num_threads(1)
    with sft.distributed_process_group(run, 60, backend="gloo"), \
            patch("verl.workers.actor.dp_actor.get_device_id", return_value=torch.device("cpu")), \
            patch("verl.utils.torch_functional.FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE", False):
        model = tiny_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False)
        actor = sft.make_actor(model, optimizer, 1)
        plan = [{"epoch": 1, "indices": [0, 1]}]
        sft.run_updates(actor, optimizer, calls(2), plan, 0, 32, run=run)
        checkpoint, path = directory / "checkpoint", directory / "training_state.pt"
        run.root_phase("checkpoint", lambda: save_checkpoint(model, checkpoint))
        run.root_phase("state save", lambda: sft.save_training_state(
            path, checkpoint, model, optimizer, compatibility(2), 1, ["0", "1"],
        ))
        resumed = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False)
        updates, ids = run.phase("state restore", lambda: sft.restore_training_state(
            path, checkpoint, model, resumed, compatibility(2), ["2", "3"],
        ))
        assert updates == 1 and ids == ["0", "1"]
        for left, right in zip(optimizer.state.values(), resumed.state.values(), strict=True):
            for key in left:
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
        metrics = sft.run_updates(sft.make_actor(model, resumed, 1), resumed,
                                  calls(4)[2:], plan, 0, 32, run=run, initial_updates=updates)
        assert metrics[0]["optimizer_step"] == 2
        run.agree("resumed replicas", [
            sft.hashlib.sha256(p.detach().numpy().tobytes()).hexdigest() for p in model.parameters()
        ])
        # Restore failures must be collective even when only one rank is incompatible.
        saved_model = tiny_model()
        from safetensors.torch import load_file

        saved_model.load_state_dict(load_file(str(checkpoint / "model.safetensors")))
        fresh = torch.optim.AdamW(saved_model.parameters(), lr=0.01, foreach=False)
        try:
            run.phase("state restore", lambda: sft.restore_training_state(
                path, checkpoint, saved_model, fresh, compatibility(2 if run.rank == 0 else 8), ["new"],
            ))
        except RuntimeError as exc:
            assert "rank 1" in str(exc) and "hyperparameters/world_size" in str(exc)
        else:
            raise AssertionError("Resume mismatch must reach every rank")
        with patch.object(torch, "save", side_effect=OSError("injected state write failure")):
            try:
                run.root_phase("state save", lambda: sft.save_training_state(
                    directory / "failed.pt", checkpoint, saved_model, optimizer, compatibility(2),
                    1, ["0", "1"],
                ))
            except RuntimeError as exc:
                assert "rank 0" in str(exc) and "injected state write failure" in str(exc)
            else:
                raise AssertionError("State write failure must reach every rank")
        assert not torch.cuda.is_initialized()
        if run.rank == 0:
            print("DISTRIBUTED_RESUME_OK", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        distributed_worker(Path(sys.argv[2]))
    else:
        unittest.main()
