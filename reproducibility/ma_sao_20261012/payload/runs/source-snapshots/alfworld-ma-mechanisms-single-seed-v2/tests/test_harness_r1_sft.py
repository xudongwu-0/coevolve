"""CPU-only SFT checks. Tiny numerical fixtures are never formal training data."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
import uuid
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "whale_local"))
from harness_r1_searchqa import sft_train as sft


def row(identifier="a"):
    return {
        "id": identifier, "split": "train",
        "messages": [
            {"role": "system", "content": "Harness Engineer."},
            {"role": "user", "content": "Diagnose recurring search failures."},
            {"role": "assistant", "content": 'Reason about repeated queries.\n</think>\n<patch>{"actions":[]}</patch>'},
        ],
    }


def tiny_model():
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM

    config = Qwen3_5TextConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=8,
        max_position_embeddings=128, full_attention_interval=2,
        rope_parameters={"rope_type": "default", "rope_theta": 10000,
                         "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 0]},
    )
    config._attn_implementation = "sdpa"
    return Qwen3_5ForCausalLM(config)


def tiny_native_model():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=8, max_position_embeddings=128, tie_word_embeddings=True,
                         eos_token_id=31)
    config._attn_implementation = "sdpa"
    return Qwen3ForCausalLM(config)


class HarnessSFTTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / "tests" / (".harness-sft-test-" + uuid.uuid4().hex)
        self.directory.mkdir()
        self.addCleanup(shutil.rmtree, self.directory)

    def write_rows(self, rows, suffix=".json"):
        path = self.directory / ("rows" + suffix)
        payload = (json.dumps(rows) if suffix == ".json" else
                   "\n".join(json.dumps(r) for r in rows)).encode()
        path.write_bytes(payload)
        return path, hashlib.sha256(payload).hexdigest()

    def test_help_and_import_do_not_import_torch(self):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(ROOT / "whale_local")}
        command = [
            sys.executable, "-c",
            "import sys; from harness_r1_searchqa import sft_train as s; "
            "assert 'torch' not in sys.modules; s.parse_args(['--help'])",
        ]
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--data-sha256", result.stdout)
        self.assertIn("--inspect-base-only", result.stdout)

    def test_cpu_environment_hides_gpus_without_initializing_cuda(self):
        import torch
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")
        self.assertFalse(torch.cuda.is_initialized())

    def test_default_full_sft_settings_and_dataset_gate(self):
        args = sft.parse_args(["--data", "real.json", "--data-sha256", "0" * 64, "--output", "fresh"])
        self.assertEqual((args.lr, args.epochs, args.global_batch_size, args.max_length),
                         (1e-5, 2, 24, 32768))
        self.assertEqual((args.world_size, args.pilot_minimum_examples, args.engineer_think_prefill),
                         (8, 256, False))
        with self.assertRaises(SystemExit):
            sft.parse_args(["--output", "fresh"])
        for flags in (["--global-batch-size", "23"], ["--max-length", "32769"],
                      ["--lr", "nan"], ["--loss-chunk-size", "0"]):
            with self.assertRaises(SystemExit):
                sft.parse_args(["--inspect-base-only", *flags])

    def test_explicit_native_pilot_cli_and_four_rank_plan(self):
        pin = "d260f1ccb31bf321a758a880503776fb41abeeaf1bb0da21b0a999b243716495"
        args = sft.parse_args(["--model", str(sft.PILOT_BASE), "--base-receipt-sha256", pin,
                               "--pilot-minimum-examples", "32", "--engineer-think-prefill",
                               "--world-size", "4", "--global-batch-size", "4",
                               "--epochs", "1", "--preflight-only", "--data", "silver.jsonl",
                               "--data-sha256", "a" * 64])
        self.assertEqual((args.world_size, args.global_batch_size, args.pilot_minimum_examples),
                         (4, 4, 32))
        plan = sft.plan_batches(32, 1, 4, 4, 42)
        self.assertEqual(len(plan), 8)
        self.assertEqual(sorted(i for _, group in plan for i in group), list(range(32)))
        for flags in (
            ["--pilot-minimum-examples", "32"],
            ["--model", str(sft.PILOT_BASE)],
            ["--model", str(sft.PILOT_BASE), "--base-receipt-sha256", pin,
             "--preflight-only", "--data", "d", "--data-sha256", "a" * 64],
        ):
            with self.assertRaises(SystemExit):
                sft.parse_args(["--inspect-base-only", *flags] if "--preflight-only" not in flags else flags)

    def test_exact_schema_hash_and_jsonl(self):
        for suffix in (".json", ".jsonl"):
            path, sha = self.write_rows([row()], suffix)
            rows, actual = sft.load_records(path, sha)
            self.assertEqual(rows, [row()])
            self.assertEqual(actual, sha)
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                sft.load_records(path, "0" * 64)
        for change in (
            {"split": "dev"}, {"messages": row()["messages"][:2]}, {"id": ""},
            {"answer": "leak"}, {"messages": [*row()["messages"][:2],
                                           {"role": "user", "content": "wrong"}]},
        ):
            bad = {**row(), **change}
            path, sha = self.write_rows([bad])
            with self.assertRaises(ValueError):
                sft.load_records(path, sha)
        path, sha = self.write_rows([row(), row()])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            sft.load_records(path, sha)
        for payload in (b'[{"id":"a","id":"b"}]', b'[{"id":NaN}]'):
            path.write_bytes(payload)
            with self.assertRaises(ValueError):
                sft.load_records(path, hashlib.sha256(payload).hexdigest())

    def test_continuation_rejects_wrong_protocol(self):
        valid = row()["messages"][-1]["content"]
        sft.validate_continuation(valid)
        for content in (
            "<think>\n" + valid, valid.replace("</think>", ""), valid + "</think>",
            valid + "<patch>{}</patch>", valid.replace("Reason about repeated queries.", ""),
            "reason</think><patch>[]</patch>", "reason</think><patch>{bad}</patch>",
            valid + "\nextra", 'reason</think><patch>{"x":1,"x":2}</patch>',
        ):
            with self.assertRaises(ValueError, msg=content):
                sft.validate_continuation(content)

    def test_real_qwen_tokenizer_exact_prefill_mask_and_eos(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(sft.BASE, local_files_only=True)
        example = row()
        example["messages"][-1]["content"] = '  Preserve literal whitespace.\n\n</think>\n<patch>\n{}\n</patch>'
        call = sft.tokenize_rows([example], tokenizer, 32768)[0]
        boundary = call["labels"].index(next(v for v in call["labels"] if v != -100))
        prompt = tokenizer.decode(call["input_ids"][:boundary], skip_special_tokens=False)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n<think>\n"))
        self.assertEqual(tokenizer.decode(call["input_ids"][boundary:-1], skip_special_tokens=False),
                         example["messages"][-1]["content"])
        self.assertEqual(call["input_ids"][-1], tokenizer.eos_token_id)
        self.assertEqual(call["labels"][:boundary], [-100] * boundary)
        self.assertEqual(call["labels"][boundary:], call["input_ids"][boundary:])
        self.assertEqual(call["target_tokens"], len(call["input_ids"]) - boundary)
        with self.assertRaisesRegex(ValueError, "truncation forbidden"):
            sft.tokenize_rows([example], tokenizer, len(call["input_ids"]) - 1)
        bad = row()
        bad["messages"][-1]["content"] = "bad<|im_end|></think><patch>{}</patch>"
        with self.assertRaisesRegex(ValueError, "reserved tokens"):
            sft.tokenize_rows([bad], tokenizer, 32768)

    def test_native_template_generation_only_export_eos_and_original_assets_unchanged(self):
        from transformers import AutoTokenizer
        before = {name: sft._sha(sft.PILOT_BASE / name)
                  for name in ("provenance.json", "tokenizer_config.json", "tokenizer.json")}
        tokenizer = AutoTokenizer.from_pretrained(sft.PILOT_BASE, local_files_only=True)
        original_template = tokenizer.chat_template
        messages = row()["messages"][:2]
        plain = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        generation = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        self.assertTrue(generation.endswith("<|im_start|>assistant\n"))
        identity = sft.configure_engineer_template(tokenizer, True)
        self.assertTrue(identity["engineer_think_prefill"])
        self.assertNotEqual(identity["original_sha256"], identity["exported_sha256"])
        self.assertTrue(tokenizer.chat_template.startswith(original_template))
        self.assertEqual(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False), plain)
        self.assertEqual(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                         generation + "<think>\n")
        call = sft.tokenize_rows([row()], tokenizer, 32768)[0]
        self.assertEqual(call["input_ids"][-1], 151645)
        self.assertEqual(call["labels"][-1], 151645)
        supervised = [label for label in call["labels"] if label != -100]
        self.assertEqual(tokenizer.decode(supervised[:-1], skip_special_tokens=False),
                         row()["messages"][-1]["content"])
        exported = self.directory / "tokenizer-export"
        tokenizer.save_pretrained(exported)
        restored = AutoTokenizer.from_pretrained(exported, local_files_only=True)
        self.assertEqual(restored.chat_template, tokenizer.chat_template)
        self.assertEqual(restored.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                         generation + "<think>\n")
        self.assertEqual(before, {name: sft._sha(sft.PILOT_BASE / name) for name in before})

    def test_deterministic_eight_rank_batch_plan_no_repeats_or_drops(self):
        plan = sft.plan_batches(48, 2, 24, 8, 42)
        self.assertEqual(plan, sft.plan_batches(48, 2, 24, 8, 42))
        for epoch in (1, 2):
            consumed = [i for e, indices in plan if e == epoch for i in indices]
            self.assertEqual(sorted(consumed), list(range(48)))
        for _, indices in plan:
            per_rank = [indices[r::8] for r in range(8)]
            self.assertEqual([len(items) for items in per_rank], [3] * 8)
            self.assertEqual(len(set(i for items in per_rank for i in items)), 24)
        for count, batch, world in ((24, 23, 8), (0, 24, 8), (24, 24, 0)):
            with self.assertRaises(ValueError):
                sft.plan_batches(count, 2, batch, world, 42)

    def test_arbitrary_final_group_has_equal_collectives_and_no_extra_supervision(self):
        for count in (1, 256, 257, 263):
            plan = sft.plan_batches(count, 2, 24, 8, 42)
            for epoch in (1, 2):
                supervised = []
                for e, indices in plan:
                    if e != epoch:
                        continue
                    slots = [sft.rank_slots(indices, rank, 8) for rank in range(8)]
                    self.assertEqual(len({len(items) for items in slots}), 1)
                    supervised.extend(i for items in slots for i in items if i is not None)
                self.assertEqual(sorted(supervised), list(range(count)))
        tail = sft.plan_batches(257, 1, 24, 8, 42)[-1][1]
        self.assertEqual(len(tail), 17)
        self.assertEqual(sum(i is None for rank in range(8) for i in sft.rank_slots(tail, rank, 8)), 7)

    def write_bootstrap(self, count=256):
        rows = []
        for index in range(count):
            rows.append({
                "messages": row()["messages"],
                "provenance": {
                    "generation_id": f"packet-{index}-sample-0",
                    "engineer_manifest_sha256": "a" * 64,
                    "task_ids": ["verified-train-task"],
                    "reward_delta": 0.25,
                    "patch_sha256": "b" * 64,
                    "split": "train",
                },
            })
        path, sha = self.write_rows(rows, ".jsonl")
        complete = {
            "stage": "fresh_base_silver_collection", "usable_for_sft": True,
            "dataset_sha256": sha, "examples": count, "minimum_examples": 256,
            "input_identity": {"engineer_manifest_sha256": "a" * 64},
        }
        (self.directory / "COMPLETE.json").write_text(json.dumps(complete))
        return rows, path, sha, complete

    def test_verified_bootstrap_format_normalizes_without_prompt_provenance(self):
        rows, path, sha, _ = self.write_bootstrap()
        loaded, actual = sft.load_records(path, sha)
        self.assertEqual(actual, sha)
        self.assertEqual(len(loaded), 256)
        self.assertEqual(set(loaded[0]), {"id", "split", "messages"})
        self.assertEqual(loaded[0]["id"], rows[0]["provenance"]["generation_id"])
        self.assertEqual(loaded[0]["messages"], rows[0]["messages"])
        self.assertNotIn("provenance", loaded[0]["messages"][-1]["content"])

    def test_bootstrap_requires_positive_train_provenance_and_verified_completion(self):
        for field, bad in (("reward_delta", 0), ("reward_delta", True), ("split", "test"),
                           ("task_ids", []), ("task_ids", ["a", "a"]),
                           ("patch_sha256", "not-a-hash"), ("engineer_manifest_sha256", "c" * 64)):
            rows, _, _, complete = self.write_bootstrap()
            rows[0]["provenance"][field] = bad
            path, sha = self.write_rows(rows, ".jsonl")
            complete["dataset_sha256"] = sha
            (self.directory / "COMPLETE.json").write_text(json.dumps(complete))
            with self.assertRaises(ValueError):
                sft.load_records(path, sha)
        _, path, sha, complete = self.write_bootstrap(255)
        with self.assertRaisesRegex(ValueError, "at least 256"):
            sft.load_records(path, sha)
        _, path, sha, complete = self.write_bootstrap()
        complete["usable_for_sft"] = False
        (self.directory / "COMPLETE.json").write_text(json.dumps(complete))
        with self.assertRaisesRegex(ValueError, "at least 256"):
            sft.load_records(path, sha)
        (self.directory / "COMPLETE.json").unlink()
        with self.assertRaises(FileNotFoundError):
            sft.load_records(path, sha)

    def test_bootstrap_32_requires_explicit_pilot_override_but_teacher_hash_remains_9b(self):
        _, path, sha, complete = self.write_bootstrap(32)
        complete["minimum_examples"] = 32
        complete["preliminary"] = True
        (self.directory / "COMPLETE.json").write_text(json.dumps(complete))
        with self.assertRaisesRegex(ValueError, "256"):
            sft.load_records(path, sha)
        loaded, _ = sft.load_records(path, sha, minimum_examples=32)
        self.assertEqual(len(loaded), 32)
        self.assertEqual(sft.verify_bootstrap_completion(path, sha, 32, 32)[0], "a" * 64)
        complete["examples"] = 31
        (self.directory / "COMPLETE.json").write_text(json.dumps(complete))
        with self.assertRaisesRegex(ValueError, "32"):
            sft.load_records(path, sha, minimum_examples=32)

    def test_chunked_ce_matches_full_loss_and_all_gradients(self):
        import torch
        import torch.nn.functional as F

        torch.manual_seed(7)
        hidden = torch.randn(2, 9, 8, requires_grad=True)
        head = torch.nn.Linear(8, 23, bias=False)
        reference_hidden = hidden.detach().clone().requires_grad_()
        reference_head = copy.deepcopy(head)
        labels = torch.randint(0, 23, (2, 9))
        labels[:, :4] = -100
        loss = sft.chunked_loss(hidden, labels, head, 2)
        expected = F.cross_entropy(reference_head(reference_hidden[:, :-1]).float().reshape(-1, 23),
                                   labels[:, 1:].reshape(-1), reduction="sum")
        loss.backward()
        expected.backward()
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(hidden.grad, reference_hidden.grad)
        torch.testing.assert_close(head.weight.grad, reference_head.weight.grad)
        self.assertTrue(torch.equal(hidden.grad[:, :3], torch.zeros_like(hidden.grad[:, :3])))

    def test_global_token_weighting_matches_reference(self):
        import torch

        parameter = torch.tensor(0.7, requires_grad=True)
        counts = [1, 3, 4, 2, 5, 1, 2, 3]
        total = sum(counts)
        gradients = []
        for count in counts:
            value = parameter.detach().clone().requires_grad_()
            (count * value.square() * 8 / total).backward()
            gradients.append(value.grad)
        (parameter.square()).backward()
        torch.testing.assert_close(torch.stack(gradients).mean(), parameter.grad)

    def test_idle_forward_has_zero_supervision_and_zero_parameter_gradients(self):
        import torch
        model = tiny_model()
        trainer = sft.loss_module(model, 3)
        ids = torch.tensor([[4, 5, 6, 7, 8, 9]])
        labels = ids.clone()
        labels[:, :3] = -100
        (trainer(ids, labels) * 0.0).backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.equal(parameter.grad, torch.zeros_like(parameter.grad)), name)

    def make_upstream_checkpoint(self, model):
        from transformers import Qwen3_5Config
        from safetensors.torch import save_file

        Qwen3_5Config(text_config=model.config.to_dict()).save_pretrained(self.directory)
        tensors = {
            ("model.language_model." + k[len("model."):] if k.startswith("model.") else k):
            v.detach().clone() for k, v in model.state_dict().items()
        }
        # Explicitly unused upstream towers, not random missing language tensors.
        import torch
        tensors["model.visual.unused"] = torch.ones(2)
        tensors["mtp.unused"] = torch.ones(2)
        save_file(tensors, self.directory / "model.safetensors")
        (self.directory / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {k: "model.safetensors" for k in tensors},
        }))
        return tensors

    def make_native_checkpoint(self, model, include_head=False):
        from safetensors.torch import save_file
        model.config.save_pretrained(self.directory)
        tensors = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()
                   if include_head or name != "lm_head.weight"}
        save_file(tensors, self.directory / "model.safetensors")
        (self.directory / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {name: "model.safetensors" for name in tensors},
        }))
        return tensors

    def test_native_tied_strict_load_full_bf16_update_and_hf_export_reload(self):
        import torch
        from transformers import AutoModelForCausalLM
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
        from functools import partial
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            apply_activation_checkpointing, checkpoint_wrapper, CheckpointImpl,
        )

        original = tiny_native_model()
        self.make_native_checkpoint(original)
        meta = sft.make_text_model(self.directory)
        self.assertIs(meta.get_input_embeddings().weight, meta.get_output_embeddings().weight)
        mapping, ignored = sft.checkpoint_schema(self.directory, meta)
        self.assertEqual(ignored, [])
        self.assertNotIn("lm_head.weight", mapping)
        loaded = sft.load_text_weights(self.directory, meta, mapping)
        self.assertIs(loaded.get_input_embeddings().weight, loaded.get_output_embeddings().weight)
        for name, tensor in original.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[name], tensor)
        before = {name: parameter.detach().clone() for name, parameter in loaded.named_parameters()}
        trainer = sft.loss_module(loaded, 3)
        apply_activation_checkpointing(
            trainer,
            checkpoint_wrapper_fn=partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
            check_fn=lambda module: isinstance(module, Qwen3DecoderLayer),
        )
        ids = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]])
        labels = ids.clone()
        labels[:, :4] = -100
        optimizer = torch.optim.AdamW(trainer.parameters(), lr=1e-5, weight_decay=0.0)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = trainer(ids, labels)
        loss.backward()
        for name, parameter in loaded.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        optimizer.step()
        for name, parameter in loaded.named_parameters():
            self.assertFalse(torch.equal(parameter, before[name.replace("._checkpoint_wrapped_module", "")]), name)
        sft.validate_tied_state(loaded.state_dict())
        saved = self.directory / "native-export"
        loaded.save_pretrained(saved, safe_serialization=True)
        shapes = {name: tuple(tensor.shape) for name, tensor in loaded.state_dict().items()}
        sft.verify_saved_checkpoint(saved, shapes, tied_embeddings=True)
        with self.assertRaisesRegex(ValueError, "coverage incomplete"):
            sft.verify_saved_checkpoint(saved, shapes, tied_embeddings=False)
        restored, info = AutoModelForCausalLM.from_pretrained(
            saved, local_files_only=True, output_loading_info=True,
        )
        self.assertFalse(info["missing_keys"])
        self.assertFalse(info["unexpected_keys"])
        self.assertIs(restored.get_input_embeddings().weight, restored.get_output_embeddings().weight)
        self.assertTrue(all(parameter.requires_grad for parameter in restored.parameters()))
        restored.eval()
        with torch.no_grad():
            self.assertTrue(torch.isfinite(restored(ids).logits).all())

    def test_native_schema_rejects_missing_embedding_and_disagreeing_duplicate_head(self):
        import torch
        from safetensors.torch import save_file
        original = tiny_native_model()
        tensors = self.make_native_checkpoint(original, include_head=True)
        tensors["lm_head.weight"][0, 0] += 1
        save_file(tensors, self.directory / "model.safetensors")
        meta = sft.make_text_model(self.directory)
        mapping, _ = sft.checkpoint_schema(self.directory, meta)
        with self.assertRaisesRegex(ValueError, "tensors disagree"):
            sft.load_text_weights(self.directory, meta, mapping)
        self.make_native_checkpoint(original)
        index_path = self.directory / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        del index["weight_map"]["model.embed_tokens.weight"]
        index_path.write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError, "Missing text tensors"):
            sft.checkpoint_schema(self.directory, original)
        self.make_native_checkpoint(original)
        tensors = {name: tensor for name, tensor in tensors.items() if name != "lm_head.weight"}
        tensors["model.embed_tokens.weight"] = torch.zeros(1, 1)
        save_file(tensors, self.directory / "model.safetensors")
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            sft.checkpoint_schema(self.directory, original)

    def test_meta_materialization_preserves_shared_native_weights_and_rotary_buffers(self):
        import torch
        self.make_native_checkpoint(tiny_native_model())
        meta = sft.make_text_model(self.directory)
        cache = {}
        shared_id = id(meta.get_input_embeddings().weight)
        buffers = {name: buffer.clone() for name, buffer in meta.named_buffers()}
        for module in meta.modules():
            sft.materialize_meta_module(module, torch.device("cpu"), cache, {shared_id})
        self.assertEqual(len(cache), 1)
        self.assertIs(meta.get_input_embeddings().weight, meta.get_output_embeddings().weight)
        self.assertTrue(all(not parameter.is_meta for parameter in meta.parameters()))
        for name, buffer in meta.named_buffers():
            torch.testing.assert_close(buffer, buffers[name])

    def test_actual_loading_remaps_strictly_and_tiny_adam_updates_all_text_parameters(self):
        import torch

        torch.manual_seed(9)
        original = tiny_model()
        self.make_upstream_checkpoint(original)
        meta = sft.make_text_model(self.directory)
        self.assertTrue(all(p.is_meta and p.dtype == torch.float32 for p in meta.parameters()))
        self.assertTrue(all(not b.is_meta for b in meta.buffers()))
        mapping, ignored = sft.checkpoint_schema(self.directory, meta)
        self.assertEqual(len(ignored), 2)
        loaded = sft.load_text_weights(self.directory, meta, mapping)
        for name, tensor in original.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[name], tensor)
        before = {n: p.detach().clone() for n, p in loaded.named_parameters()}
        initial_probes = sft.text_probes(loaded.state_dict())
        trainer = sft.loss_module(loaded, 3)
        from functools import partial
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            apply_activation_checkpointing, checkpoint_wrapper, CheckpointImpl,
        )
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
        apply_activation_checkpointing(
            trainer,
            checkpoint_wrapper_fn=partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
            check_fn=lambda module: isinstance(module, Qwen3_5DecoderLayer),
        )
        ids = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]])
        labels = ids.clone()
        labels[:, :4] = -100
        optimizer = torch.optim.AdamW(trainer.parameters(), lr=1e-5)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = trainer(ids, labels) / 4
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, parameter in loaded.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        optimizer.step()
        self.assertTrue(all(not torch.equal(p, before[n.replace("._checkpoint_wrapped_module", "")])
                            for n, p in loaded.named_parameters()))
        self.assertTrue(any(p["changed"] for p in
                            sft.verify_changed_probes(initial_probes, loaded.state_dict()).values()))
        saved = self.directory / "export"
        loaded.save_pretrained(saved, safe_serialization=True)
        shapes = {k: tuple(v.shape) for k, v in loaded.state_dict().items()}
        hashes = sft.verify_saved_checkpoint(saved, shapes)
        self.assertIn("model.safetensors", hashes)
        from transformers import AutoModelForCausalLM
        restored, info = AutoModelForCausalLM.from_pretrained(
            saved, local_files_only=True, output_loading_info=True,
        )
        self.assertFalse(info["missing_keys"])
        self.assertFalse(info["unexpected_keys"])
        restored.eval()
        with torch.no_grad():
            self.assertTrue(torch.isfinite(restored(ids).logits).all())

    def test_missing_language_tensor_and_wrong_shape_are_rejected(self):
        model = tiny_model()
        tensors = self.make_upstream_checkpoint(model)
        index_path = self.directory / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        del index["weight_map"]["lm_head.weight"]
        index_path.write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError, "Missing text tensors"):
            sft.checkpoint_schema(self.directory, model)
        self.make_upstream_checkpoint(model)
        import torch
        from safetensors.torch import save_file
        tensors["lm_head.weight"] = torch.zeros(1, 1)
        save_file(tensors, self.directory / "model.safetensors")
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            sft.checkpoint_schema(self.directory, model)

    def test_atomic_completion_manifest_hash_and_fresh_output(self):
        staging = self.directory / "staging"
        staging.mkdir()
        (staging / "config.json").write_text("{}")
        output = self.directory / "published"
        sft.publish_checkpoint(staging, output, {"status": "VERIFIED COMPLETE"})
        self.assertFalse(staging.exists())
        self.assertEqual((output / "COMPLETE").read_text().splitlines(),
                         ["VERIFIED COMPLETE", sft._sha(output / "manifest.json")])
        with self.assertRaises(FileExistsError):
            sft.publish_checkpoint(self.directory / "missing", output, {})

    def test_saved_nonfinite_tensor_cannot_be_verified(self):
        import torch
        from safetensors.torch import save_file
        save_file({"weight": torch.tensor([float("nan")])}, self.directory / "model.safetensors")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            sft.verify_saved_checkpoint(self.directory, {"weight": (1,)})

    def test_noop_parameter_probes_cannot_be_certified(self):
        model = tiny_model()
        with self.assertRaisesRegex(ValueError, "no-op"):
            sft.verify_changed_probes(sft.text_probes(model.state_dict()), model.state_dict())

    def test_export_generation_stops_at_trained_chat_eos(self):
        from transformers import AutoTokenizer
        model = tiny_model()
        tokenizer = AutoTokenizer.from_pretrained(sft.BASE, local_files_only=True)
        sft.configure_generation(model, tokenizer)
        self.assertEqual(model.config.eos_token_id, 248046)
        self.assertEqual(model.generation_config.eos_token_id, 248046)
        self.assertEqual(model.generation_config.pad_token_id, tokenizer.pad_token_id)

    def test_base_receipt_is_verified_not_merely_copied(self):
        files = {}
        for name in ("config.json", "model.safetensors.index.json", "tokenizer.json",
                     "tokenizer_config.json", "chat_template.jinja", "model.safetensors"):
            content = ('{"weight_map":{"x":"model.safetensors"}}' if name.endswith("index.json") else "{}")
            (self.directory / name).write_text(content)
            files[name] = sft._sha(self.directory / name)
        entries = [{"path": name, "verified": True, "sha256": sha,
                    "source": "https://huggingface.co/Qwen/Qwen3.5-9B/resolve/revision/" + name}
                   for name, sha in files.items()]
        receipt_path = self.directory / "download_receipt.json"
        receipt_path.write_text(json.dumps({"files": entries}))
        receipt_sha = sft._sha(receipt_path)
        verified = sft.verify_base(self.directory, receipt_sha)
        self.assertEqual(verified["files_sha256"], files)
        self.assertEqual(verified["tree_sha256"], sft.digest_map(files))
        (self.directory / "config.json").write_text('{"modified":true}')
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            sft.verify_base(self.directory, receipt_sha)

    def test_native_provenance_receipt_requires_official_revision_and_cli_pin(self):
        files = {}
        for name in ("config.json", "model.safetensors.index.json", "tokenizer.json",
                     "tokenizer_config.json", "model.safetensors"):
            content = ('{"weight_map":{"x":"model.safetensors"}}' if name.endswith("index.json") else "{}")
            (self.directory / name).write_text(content)
            files[name] = sft._sha(self.directory / name)
        receipt = {
            "repository": "Qwen/Qwen3-4B-Instruct-2507", "revision": "original",
            "files": [{"path": name, "sha256": sha,
                       "url": "https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/resolve/original/" + name}
                      for name, sha in files.items()],
        }
        path = self.directory / "provenance.json"
        path.write_text(json.dumps(receipt))
        pin = sft._sha(path)
        with patch.object(sft, "PILOT_BASE", self.directory):
            identity = sft.verify_base(self.directory, pin)
            self.assertEqual(identity["receipt_file"], "provenance.json")
            self.assertEqual(identity["repository"], "Qwen/Qwen3-4B-Instruct-2507")
            with self.assertRaisesRegex(ValueError, "receipt SHA256"):
                sft.verify_base(self.directory, "0" * 64)
            receipt["repository"] = "other/trained-editor"
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "original official"):
                sft.verify_base(self.directory, sft._sha(path))


if __name__ == "__main__":
    unittest.main()
