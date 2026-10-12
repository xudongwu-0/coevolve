"""CPU-only checks; native loss/backward is exercised without a CUDA context."""
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


def example(identifier="train", split="train", format="json"):
    row = {
        "id": identifier, "family": split + "_family", "split": split, "format": format,
        "instruction": "Set the timeout to 5 and preserve the other settings.",
        "source": '{"timeout": 2, "retry": true}',
        "target": '{"timeout": 5, "retry": true}', "changes": {"timeout": 5},
    }
    if format == "python":
        row.update(
            source="from service import Settings\nclass Config(Settings):\n    timeout = 2\n    retry = True\n",
            target="from service import Settings\nclass Config(Settings):\n    timeout = 5\n    retry = True\n",
        )
    return row


class Tokenizer:
    eos_token_id = 2
    pad_token_id = 0
    all_special_ids = [0, 2]

    def __init__(self):
        self.messages, self.targets = [], []

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": True, "add_generation_prompt": True,
            "return_dict": False, "truncation": False,
        }
        self.messages.append(messages)
        return [4, 5, 6]

    def encode(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False, "truncation": False}
        self.targets.append(text)
        return [7, 8]

    def save_pretrained(self, directory):
        (Path(directory) / "tokenizer_config.json").write_text('{"fixture": true}')


def call(identifier, prompt, response, split="train"):
    return {"id": identifier, "split": split, "prompt_ids": prompt,
            "response_ids": response, "temperature": 1.0}


class RewriteSFTTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / "tests" / (".rewrite-sft-test-" + uuid.uuid4().hex)
        self.directory.mkdir()
        self.addCleanup(shutil.rmtree, self.directory)

    def write_data(self, rows):
        path = self.directory / "data.json"
        path.write_text(json.dumps(rows))
        return path

    def test_import_and_help_do_not_import_torch_or_initialize_cuda(self):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(ROOT / "whale_local")}
        code = (
            "import sys; from sao_bridge import rewrite_sft_train as s; "
            "assert 'torch' not in sys.modules; "
            "assert len(s.format_messages({'instruction':'x','source':'y'})) == 2"
        )
        subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, env=env,
                       check=True, capture_output=True, text=True)
        result = subprocess.run(
            [sys.executable, "-B", "-m", "sao_bridge.rewrite_sft_train", "--help"],
            cwd=ROOT, env=env, check=True, capture_output=True, text=True,
        )
        self.assertIn("--max-length", result.stdout)

    def test_exact_messages_preserve_whitespace_and_exclude_answers(self):
        row = example()
        row.update(source=" \n" + row["source"] + "\n", instruction="  Edit literally.\n")
        messages = sft.format_messages(row)
        self.assertEqual(messages, [
            {"role": "system", "content": (
                "Apply the requested edit to the supplied document. Return only the complete "
                "edited document, without Markdown fences or explanation. Preserve every "
                "unrequested setting and the original declaration structure."
            )},
            {"role": "user", "content": "Instruction:\n  Edit literally.\n\n\nDocument:\n" + row["source"]},
        ])
        row["target"], row["changes"] = "HIDDEN ANSWER", {"secret": 1}
        self.assertEqual(sft.format_messages(row), messages)

    def test_valid_python_and_json_with_train_only_tokenization(self):
        rows = [example(), example("py", format="python"),
                example("dev", "dev"), example("test", "test")]
        checked, digest = sft.load_records(self.write_data(rows))
        self.assertEqual(checked, rows)
        self.assertEqual(len(digest), 64)
        tokenizer = Tokenizer()
        calls = sft.tokenize_training_rows(checked, tokenizer, 6)
        self.assertEqual([c["id"] for c in calls], ["train", "py"])
        self.assertEqual(tokenizer.targets, [row["target"] for row in rows[:2]])
        self.assertEqual(calls[0]["response_ids"], [7, 8, 2])
        self.assertEqual(tokenizer.messages, [sft.format_messages(row) for row in rows[:2]])

    def test_invalid_rows_are_rejected_not_filtered_or_coerced(self):
        bad = []
        for update in [
            {"split": "validation"}, {"format": "yaml"}, {"changes": []}, {"id": 1},
            {"target": '{"timeout": 6, "retry": true}'},
            {"target": '{"timeout": 5, "retry": false}'},
            {"target": '{"timeout": 5, "retry": true, "new": 1}'},
            {"target": '```json\n{"timeout": 5, "retry": true}\n```'},
            {"source": '{"timeout": 2, "timeout": 3}'},
            {"instruction": ""}, {"source": ""}, {"extra": "unexpected"},
        ]:
            bad.append([dict(example(), **update)])
        bad.extend([
            [], {}, [example(), example()], [example("dev", "dev")],
            [example(), dict(example("dev", "dev"), family="train_family")],
        ])
        for rows in bad:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                sft.load_records(self.write_data(rows))
        path = self.directory / "raw.json"
        for text in ('[{"id": "first", "id": "second"}]', '[NaN]', '[Infinity]'):
            path.write_text(text)
            with self.assertRaises(ValueError):
                sft.load_records(path)

    def test_heldout_documents_are_not_semantically_validated_or_tokenized(self):
        from sao_bridge.rewrite_data import validate_example

        train = example()
        rows = [train, example("dev", "dev"), example("test", "test")]
        for row in rows[1:]:
            row["source"], row["target"] = "opaque held-out source", "opaque held-out target"
        with patch("sao_bridge.rewrite_data.validate_example", wraps=validate_example) as validate:
            checked, _ = sft.load_records(self.write_data(rows))
        validate.assert_called_once()
        self.assertEqual(validate.call_args.args[0].split, "train")
        tokenizer = Tokenizer()
        self.assertEqual(len(sft.tokenize_training_rows(checked, tokenizer, 100)), 1)
        self.assertEqual(tokenizer.targets, [train["target"]])
        rows[-1]["target"] = 3
        with self.assertRaisesRegex(ValueError, "Invalid data row"):
            sft.load_records(self.write_data(rows))

    def test_token_bounds_empty_responses_and_reserved_tokens(self):
        tokenizer = Tokenizer()
        with self.assertRaisesRegex(ValueError, "truncation is forbidden"):
            sft.tokenize_training_rows([example()], tokenizer, 5)
        for tokens in ([], [2], [-1], [True]):
            with self.subTest(tokens=tokens), patch.object(tokenizer, "encode", return_value=tokens):
                with self.assertRaises(ValueError):
                    sft.tokenize_training_rows([example()], tokenizer, 100)
        tokenizer.eos_token_id = None
        with self.assertRaisesRegex(ValueError, "EOS"):
            sft.tokenize_training_rows([example()], tokenizer, 100)

    def test_planning_is_seeded_complete_and_keeps_remainder(self):
        calls = [call(str(i), [4], [5, 2]) for i in range(9)]
        plan = sft.plan_batches(calls, 2, 4, 42, 10)
        self.assertEqual(plan, sft.plan_batches(calls, 2, 4, 42, 10))
        self.assertNotEqual(plan, sft.plan_batches(calls, 2, 4, 43, 10))
        self.assertEqual([len(group["indices"]) for group in plan], [4, 4, 1, 4, 4, 1])
        for epoch in (1, 2):
            self.assertEqual(sorted(i for g in plan if g["epoch"] == epoch for i in g["indices"]),
                             list(range(9)))
        with self.assertRaisesRegex(ValueError, "Only train"):
            sft.plan_batches([call("heldout", [4], [2], "test")], 1, 1, 42, 10)
        with self.assertRaisesRegex(ValueError, "padded group"):
            sft.plan_batches([call("a", [4] * 8, [2]), call("b", [4], [5] * 7 + [2])],
                             1, 2, 42, 10)

    def test_cli_defaults_and_invalid_bounds(self):
        required = ["--model", "model", "--data", "data", "--output", "new"]
        args = sft.parse_args(required)
        self.assertEqual((args.lr, args.epochs, args.batch_size, args.seed, args.max_length),
                         (1e-5, 1, 16, 42, 4096))
        for flag, value in [("--lr", "nan"), ("--lr", "inf"), ("--lr", "0"),
                            ("--epochs", "0"), ("--batch-size", "-1"),
                            ("--max-length", "1"), ("--seed", "-1")]:
            with self.subTest(flag=flag, value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    sft.parse_args([*required, flag, value])

    def test_shared_paths_and_fresh_output(self):
        model = self.directory / "model"
        model.mkdir()
        data = self.write_data([example()])
        args = SimpleNamespace(model=model, data=data, output=self.directory / "fresh")
        sft.validate_paths(args)
        args.output.mkdir()
        with self.assertRaises(FileExistsError):
            sft.validate_paths(args)
        args.output = model / "nested"
        with self.assertRaisesRegex(ValueError, "inside the input model"):
            sft.validate_paths(args)
        args.output = self.directory / "fresh_again"
        args.model = Path("/dev/shm/not-a-shared-model")
        with self.assertRaisesRegex(ValueError, "shared storage"):
            sft.validate_paths(args)

    def test_rendezvous_is_local_and_cleanup_runs_on_failure(self):
        import torch.distributed as dist

        with patch.object(dist, "is_initialized", side_effect=[False, True]), \
                patch.object(dist, "init_process_group") as initialize, \
                patch.object(dist, "destroy_process_group") as destroy, \
                patch.object(Path, "unlink") as unlink:
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                with sft.single_rank_process_group():
                    raise RuntimeError("training failed")
            kwargs = initialize.call_args.kwargs
            self.assertTrue(kwargs["init_method"].startswith("file:///dev/shm/rewrite-sft-"))
            self.assertEqual((kwargs["rank"], kwargs["world_size"]), (0, 1))
            destroy.assert_called_once()
            unlink.assert_called_once_with(missing_ok=True)
        with patch.object(dist, "is_initialized", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "existing process group"):
                with sft.single_rank_process_group():
                    self.fail("Should not enter")

    def test_native_batch_response_mask_and_eos_when_pad_equals_eos(self):
        import torch

        calls = [call("a", [3, 4, 5], [8, 2]), call("b", [4], [6, 7, 2])]
        batch = sft.make_sft_batch(calls, pad=2, max_length=6)
        self.assertEqual(batch.batch["input_ids"].tolist(), [[3, 4, 5, 8, 2, 2], [2, 2, 4, 6, 7, 2]])
        self.assertEqual(batch.batch["attention_mask"].tolist(), [[1, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1]])
        self.assertEqual(batch.batch["position_ids"].tolist(), [[0, 1, 2, 3, 4, 4], [0, 0, 0, 1, 2, 3]])
        self.assertEqual(batch.batch["response_mask"].tolist(), [[1, 1, 0], [1, 1, 1]])
        self.assertEqual(batch.meta_info["sft_mini_batch_size"], 2)
        self.assertEqual(batch.meta_info["sft_micro_batch_size_per_gpu"], 1)
        self.assertEqual(batch.meta_info["sft_epochs"], 1)
        self.assertEqual(batch.batch["input_ids"].dtype, torch.int64)
        calls[0]["split"] = "dev"
        with self.assertRaisesRegex(ValueError, "Only train"):
            sft.make_sft_batch(calls, pad=2, max_length=6)

    def test_main_saves_verified_hf_weights_and_auditable_train_only_outputs(self):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Qwen3Config
        from sao_bridge.legacy_update import DataParallelPPOActor  # Import before mocking CUDA visibility.

        self.assertTrue(DataParallelPPOActor.update_sft_policy)
        config = Qwen3Config(
            vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            max_position_embeddings=64, eos_token_id=2, pad_token_id=0,
        )
        model = AutoModelForCausalLM.from_config(config, attn_implementation="sdpa")
        model_dir = self.directory / "input_model"
        model_dir.mkdir()
        config.save_pretrained(model_dir)
        data = self.write_data([example("t" + str(i)) for i in range(5)] +
                               [example("dev", "dev"), example("test", "test")])
        output = self.directory / "output"
        arguments = [
            "--model", str(model_dir), "--data", str(data), "--output", str(output),
            "--epochs", "2", "--batch-size", "2", "--max-length", "32",
        ]
        with patch.object(AutoConfig, "from_pretrained", return_value=config), \
                patch.object(AutoTokenizer, "from_pretrained", return_value=Tokenizer()), \
                patch.object(AutoModelForCausalLM, "from_pretrained", return_value=model) as load, \
                patch.object(model, "to", return_value=model) as to_device, \
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), \
                patch("torch.cuda.device_count", return_value=1), \
                patch("torch.cuda.is_bf16_supported", return_value=True), \
                patch("torch.cuda.set_device"), patch("torch.cuda.manual_seed_all"), \
                patch("torch.cuda.reset_peak_memory_stats"), \
                patch("torch.cuda.max_memory_allocated", return_value=0), \
                patch("sao_bridge.resources.require_research_node") as authorize, \
                patch("sao_bridge.resources.stop_known") as stop_filler, \
                patch("sao_bridge.resources.audit") as audit, \
                patch.object(sft, "single_rank_process_group", return_value=contextlib.nullcontext()), \
                patch("torch.distributed.get_rank", return_value=0), \
                patch("verl.workers.actor.dp_actor.get_device_id", return_value=torch.device("cpu")), \
                patch("verl.utils.torch_functional.FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE", False), \
                contextlib.redirect_stdout(io.StringIO()):
            sft.main(arguments)
        authorize.assert_called_once_with()
        stop_filler.assert_not_called()
        audit.assert_not_called()
        self.assertEqual(load.call_args.kwargs["dtype"], torch.float32)
        self.assertEqual(load.call_args.kwargs["attn_implementation"], "sdpa")
        to_device.assert_called_once_with("cuda")
        result = json.loads((output / "result.json").read_text())
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["optimizer_updates"], 6)
        self.assertEqual(result["response_tokens"], 30)
        self.assertTrue(result["saved_probes_verified"])
        self.assertEqual(result["heldout_acceptance"], "not evaluated")
        metrics = json.loads((output / "metrics.json").read_text())
        self.assertEqual([len(row["ids"]) for row in metrics], [2, 2, 1, 2, 2, 1])
        self.assertEqual({key for row in metrics for key in row["ids"]}, {"t" + str(i) for i in range(5)})
        provenance = json.loads((output / "provenance.json").read_text())
        self.assertEqual(provenance["optimizer_splits"], ["train"])
        self.assertEqual(provenance["split_counts"], {"train": 5, "dev": 1, "test": 1})
        self.assertTrue(provenance["source_sha256"])
        self.assertTrue((output / "checkpoint/tokenizer_config.json").is_file())
        saved = AutoModelForCausalLM.from_pretrained(output / "checkpoint", local_files_only=True)
        for actual, expected in zip(saved.parameters(), model.parameters(), strict=True):
            self.assertEqual(actual.dtype, torch.float32)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_cli_requires_external_gpu_selection_before_loading_weights_or_creating_output(self):
        import torch
        from transformers import AutoModelForCausalLM

        model = ROOT / "assets/Qwen3-4B-Instruct-2507"
        if not model.is_dir():
            self.skipTest("Local Qwen tokenizer assets unavailable")
        output = self.directory / "no_gpu_run"
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}), \
                patch("sao_bridge.resources.require_research_node"), \
                patch.object(AutoModelForCausalLM, "from_pretrained") as load:
            with self.assertRaisesRegex(RuntimeError, "exactly one allocated GPU"):
                sft.main([
                    "--model", str(model), "--data", str(self.write_data([example()])),
                    "--output", str(output),
                ])
            load.assert_not_called()
        self.assertFalse(output.exists())
        self.assertFalse(torch.cuda.is_initialized())

    def test_cli_rejects_unauthorized_node_before_gpu_work(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = self.directory / "model"
        model.mkdir()
        output = self.directory / "forbidden_run"
        arguments = [
            "--model", str(model), "--data", str(self.write_data([example()])),
            "--output", str(output),
        ]
        with patch("sao_bridge.resources.os.uname",
                   return_value=SimpleNamespace(nodename="dev-NDH200-022")), \
                patch("torch.cuda.device_count") as devices, \
                patch.object(AutoTokenizer, "from_pretrained") as tokenizer, \
                patch.object(AutoModelForCausalLM, "from_pretrained") as load:
            with self.assertRaisesRegex(RuntimeError, "only on node3/node4"):
                sft.main(arguments)
        devices.assert_not_called()
        tokenizer.assert_not_called()
        load.assert_not_called()
        self.assertFalse(output.exists())
        self.assertFalse(torch.cuda.is_initialized())

    def test_native_sft_token_weighting_masks_prompts_and_steps_once_per_group(self):
        import torch
        from sao_bridge.legacy_update import DataParallelPPOActor

        class TinyLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(16, 8)
                self.head = torch.nn.Linear(8, 16, bias=False)

            def forward(self, input_ids, **kwargs):
                return SimpleNamespace(logits=self.head(self.embedding(input_ids)))

        torch.manual_seed(3)
        torch.set_num_threads(1)
        model = TinyLM()
        oracle = copy.deepcopy(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False)
        expected_optimizer = torch.optim.AdamW(oracle.parameters(), lr=0.01, foreach=False)
        calls = [call("short", [3, 4, 5], [8, 2]), call("long", [4], [6, 7, 2])]
        batch = sft.make_sft_batch(calls, 0, 10)
        labels, mask = batch.batch["responses"], batch.batch["response_mask"].bool()
        total = mask.sum()
        # Independent oracle: only shifted response positions, never prompt/padding.
        for i in range(2):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                logits = oracle(batch.batch["input_ids"][i:i + 1]).logits[:, -4:-1, :]
                logp = logits.log_softmax(-1).gather(-1, labels[i:i + 1, :, None]).squeeze(-1)
            (-(logp * mask[i:i + 1]).sum() / total).backward()
        torch.nn.utils.clip_grad_norm_(oracle.parameters(), 1.0)
        expected_optimizer.step()
        before = sft.parameter_probes(model)
        with patch("torch.distributed.get_rank", return_value=0), \
                patch("verl.workers.actor.dp_actor.get_device_id", return_value=torch.device("cpu")), \
                patch("verl.utils.torch_functional.FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE", False):
            actor = sft.make_actor(model, optimizer, 16)
            self.assertIsInstance(actor, DataParallelPPOActor)
            self.assertEqual(actor.param_dtype, torch.bfloat16)
            plan = [{"epoch": 1, "indices": [0, 1]}]
            result = sft.run_updates(actor, optimizer, calls, plan, 0, 10)
        self.assertEqual(result[0]["response_tokens"], 5)
        self.assertEqual(result[0]["native"]["actor/sft_updates"], 1)
        self.assertEqual({state["step"].item() for state in optimizer.state.values()}, {1})
        for actual, expected in zip(model.parameters(), oracle.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=2e-3, atol=1e-4)
        sft.check_parameter_deltas(model, before)
        self.assertFalse(torch.cuda.is_initialized())

    def test_deltas_and_persisted_probes_detect_unchanged_nonfinite_and_corruption(self):
        import torch
        from safetensors.torch import save_file

        model = torch.nn.Linear(4, 3)
        before = sft.parameter_probes(model)
        with self.assertRaisesRegex(RuntimeError, "no checked parameter change"):
            sft.check_parameter_deltas(model, before)
        with torch.no_grad():
            model.weight.add_(0.01)
        deltas, after = sft.check_parameter_deltas(model, before)
        self.assertGreater(deltas["weight"]["changed_elements"], 0)
        save_file(model.state_dict(), str(self.directory / "model.safetensors"))
        sft.verify_saved_probes(self.directory, after)
        with torch.no_grad():
            model.weight.add_(1)
        save_file(model.state_dict(), str(self.directory / "model.safetensors"))
        with self.assertRaisesRegex(RuntimeError, "probe mismatch"):
            sft.verify_saved_probes(self.directory, after)
        with torch.no_grad():
            model.weight[0, 0] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "Nonfinite"):
            sft.check_parameter_deltas(model, before)

    def test_real_qwen_tokenizer_preserves_exact_target(self):
        from transformers import AutoTokenizer
        import torch

        model = ROOT / "assets/Qwen3-4B-Instruct-2507"
        if not model.is_dir():
            self.skipTest("Local Qwen tokenizer assets unavailable")
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        rows = [example(), example("python", format="python")]
        calls = sft.tokenize_training_rows(rows, tokenizer, 4096)
        for row, item in zip(rows, calls, strict=True):
            self.assertEqual(tokenizer.decode(item["response_ids"][:-1]), row["target"])
            self.assertEqual(item["response_ids"][-1], tokenizer.eos_token_id)
            self.assertTrue(tokenizer.decode(item["prompt_ids"]).endswith("<|im_start|>assistant\n"))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
