"""Local-only runtime. No model initialization occurs before an explicit lease."""
from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import shutil

from .core import Sample, digest, parse_action


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    with partial.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(path)


def file_sha256(path: Path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def model_identity(path: Path):
    files = sorted(path.glob("*.safetensors")) or sorted(path.glob("pytorch_model*.bin"))
    if not files:
        raise ValueError("No local model weight files found")
    files += [p for p in (path / "config.json", path / "tokenizer.json",
                         path / "tokenizer_config.json", path / "adapter_config.json")
              if p.is_file()]
    return {p.name: file_sha256(p) for p in files}


def implementation_identity():
    return {p.name: file_sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def process_identity():
    stat = Path("/proc/self/stat").read_text().rsplit(") ", 1)[1].split()
    return {
        "pid": os.getpid(), "uid": os.getuid(), "gid": os.getgid(),
        "start_ticks": int(stat[19]), "hostname": socket.gethostname(),
        "process_group": os.getpgrp(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def preflight(env_root: Path, model: Path | None = None):
    required = [
        "web_agent_site/envs/web_agent_text_env.py",
        "data/items_shuffle_1000.json", "data/items_ins_v2_1000.json",
        "data/items_human_ins.json",
    ]
    missing = [str(env_root / name) for name in required if not (env_root / name).is_file()]
    index = env_root / "search_engine" / "indexes_1k"
    if not any(index.glob("segments*")):
        missing.append(str(index / "segments*"))
    missing_modules = [
        name for name in ("gym", "bs4", "spacy", "pyserini", "jnius", "torch", "transformers",
                          "peft", "cleantext", "flask", "rich", "rank_bm25", "faiss", "selenium")
        if importlib.util.find_spec(name) is None
    ]
    missing_tools = [name for name in ("javac", "java") if shutil.which(name) is None]
    if importlib.util.find_spec("en_core_web_sm") is None:
        missing_modules.append("en_core_web_sm")
    if model is not None and not (model / "config.json").is_file():
        missing.append(str(model / "config.json"))
    if model is not None and not (
        any(model.glob("*.safetensors")) or any(model.glob("pytorch_model*.bin"))
    ):
        missing.append(str(model / "*.safetensors (or pytorch_model*.bin)"))
    return {
        "schema": "coevolve_webshop_preflight_v1",
        "environment_root": str(env_root), "model": str(model) if model else None,
        "missing_files": missing, "missing_modules": missing_modules, "missing_tools": missing_tools,
        "ready": not missing and not missing_modules and not missing_tools,
        "gpu_used": False, "model_loaded": False, "network_used": False,
        "interpretation": "Dependency/file check only; not a model or environment validation.",
    }


def check_lease(path: Path, *, mode: str, episodes: int, now=None):
    lease = json.loads(path.read_text())
    required = {"purpose", "hostname", "cuda_visible_devices", "expires_at",
                "primary_sft_released", "modes", "max_episodes"}
    if not required <= lease.keys():
        raise ValueError("Incomplete explicit resource lease")
    if (type(lease["max_episodes"]) is not int or lease["max_episodes"] < 1
            or not isinstance(lease["modes"], list)
            or not set(lease["modes"]) <= {"train", "evaluate"}):
        raise ValueError("Invalid resource lease budget/modes")
    if lease["purpose"] != "webshop-side-project" or lease["primary_sft_released"] is not True:
        raise ValueError("Primary SFT must explicitly release this resource")
    if lease["hostname"] != socket.gethostname():
        raise ValueError("Lease is for a different host")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible or "," in visible or visible != lease["cuda_visible_devices"]:
        raise ValueError("Exactly one explicitly leased CUDA device must be visible")
    expiry = datetime.fromisoformat(lease["expires_at"].replace("Z", "+00:00"))
    current = now or datetime.now(timezone.utc)
    if expiry.tzinfo is None or expiry <= current:
        raise ValueError("Resource lease is expired or has no timezone")
    if mode not in lease["modes"] or episodes > lease["max_episodes"]:
        raise ValueError("Run exceeds leased mode or episode budget")
    return lease


class WebShopEnvironment:
    def __init__(self, root: Path, protocol):
        if protocol.num_products != 1000:
            raise ValueError("This adapter requires the explicitly fingerprinted 1k product corpus")
        self.root, self.protocol = root.resolve(), protocol
        assets = {}
        for name in ("items_shuffle_1000.json", "items_ins_v2_1000.json", "items_human_ins.json"):
            assets["data/" + name] = file_sha256(self.root / "data" / name)
        for path in sorted((self.root / "search_engine" / "indexes_1k").rglob("*")):
            if path.is_file() and path.name != "write.lock":
                assets[str(path.relative_to(self.root))] = file_sha256(path)
        # Source identity is part of evaluation parity, not merely the product JSON.
        sources = list((self.root / "web_agent_site").rglob("*.py"))
        sources += list((self.root / "search_engine").rglob("*.py"))
        for path in sorted(sources):
            assets[str(path.relative_to(self.root))] = file_sha256(path)
        self.assets = assets
        sys.path.insert(0, str(self.root))
        from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
        self.env = WebAgentTextEnv(
            observation_mode="text", seed=protocol.goal_seed, num_products=1000,
            file_path=str(self.root / "data" / "items_shuffle_1000.json"),
            attr_path=str(self.root / "data" / "items_ins_v2_1000.json"),
        )
        needed = max((*protocol.train_ids, *protocol.eval_ids))
        if needed >= len(self.env.server.goals):
            self.env.close()
            raise ValueError(f"Task {needed} absent from goal catalog of size {len(self.env.server.goals)}")
        self.prices_digest = digest(self.env.server.product_prices)

    def reset(self, task_id):
        # Upstream reset reuses the session dict and leaves prior done/reward keys.
        self.env.server.user_sessions.pop(str(task_id), None)
        observation, _info = self.env.reset(session=task_id)
        instruction = self.env.get_instruction_text()
        manifest = {
            "task_id": task_id, "goal_seed": self.protocol.goal_seed,
            "goal_sha256": digest(self.env.server.goals[task_id]),
            "instruction_sha256": digest(instruction),
            "product_prices_sha256": self.prices_digest,
            "assets_sha256": digest(self.assets),
        }
        return observation, instruction, manifest

    def available_actions(self):
        return self.env.get_available_actions()

    def step(self, action):
        parsed = parse_action(action)
        clickables = self.available_actions().get("clickables", [])
        invalid = parsed is None or (
            action.startswith("click[")
            and action[6:-1].lower() not in {label.lower() for label in clickables}
        )
        observation, score, done, info = self.env.step(action)
        return observation, float(score), bool(done), {**dict(info or {}), "invalid": invalid}

    def close(self):
        self.env.close()


class HFPolicy:
    def __init__(self, model_path: Path, seed: int, *, train: bool = False,
                 adapter: Path | None = None, guard, enable_thinking: bool = False):
        self.guard = guard
        self.guard()
        self.model_identity = model_identity(model_path)
        self.guard()
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
        torch.manual_seed(seed)
        self.device = torch.device("cuda:0")
        config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        model_cls = (
            AutoModelForImageTextToText if config.model_type in {"qwen3_5", "qwen3_5_moe"}
            else AutoModelForCausalLM
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False,
        )
        self.model = model_cls.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, device_map={"": "cuda:0"},
        )
        if adapter is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter, is_trainable=False)
        elif train:
            from peft import LoraConfig, get_peft_model
            self.model = get_peft_model(self.model, LoraConfig(
                task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.0,
                target_modules="all-linear",
            ))
        self.model.eval()
        self.model.config.use_cache = True
        self.prompts = {}
        self.enable_thinking = enable_thinking

    def render(self, system, user):
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=self.enable_thinking,
        )
        self.prompts[prompt] = (system, user)
        return prompt

    def privileged(self, prompt, feedback):
        system, user = self.prompts[prompt]
        return self.render(system, "Training-only hindsight:\n" + feedback + "\n\n" + user)

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def sample(self, prompt_ids, max_tokens, seed, temperature):
        self.guard()
        import torch
        from transformers import GenerationConfig
        ids = torch.tensor([prompt_ids], device=self.device)
        eos = self.tokenizer.eos_token_id
        generation = GenerationConfig(
            max_new_tokens=max_tokens, do_sample=True, temperature=temperature,
            top_k=0, top_p=1.0, repetition_penalty=1.0,
            eos_token_id=eos, pad_token_id=(
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos
            ),
            return_dict_in_generate=True, output_scores=True, use_cache=True,
        )
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(seed)
            output = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                         generation_config=generation)
            response = output.sequences[0, len(prompt_ids):]
            probabilities = self.model.compute_transition_scores(
                output.sequences, output.scores, normalize_logits=True,
            )[0]
        return Sample(self.tokenizer.decode(response, skip_special_tokens=True),
                      response.tolist(), probabilities.float().tolist())

    def log_probs(self, prompt_ids, response_ids, *, grad=False):
        self.guard()
        return response_log_probs(self.model, prompt_ids, response_ids, grad=grad)

    def reference_log_probs(self, prompt_ids, response_ids):
        self.guard()
        with self.model.disable_adapter():
            return response_log_probs(self.model, prompt_ids, response_ids)

    def runtime_metrics(self):
        import torch
        if self.device.type != "cuda":
            return {"device": str(self.device)}
        return {
            "device": str(self.device), "gpu_name": torch.cuda.get_device_name(self.device),
            "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
            "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
        }


def response_log_probs(model, prompt_ids, response_ids, *, grad=False):
    """Causal alignment excludes all prompt and observation targets."""
    import torch
    if not prompt_ids or not response_ids:
        raise ValueError("Both prompt and generated response must be nonempty")
    device = next(model.parameters()).device
    ids = torch.tensor([prompt_ids + response_ids], device=device)
    with nullcontext() if grad else torch.no_grad():
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        start = len(prompt_ids) - 1
        logits = output.logits[:, start:start + len(response_ids)].float()
        labels = torch.tensor([response_ids], device=device)
        return logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1).squeeze(0)
