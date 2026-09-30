"""Tool-using episodes and counterpart-refresh replay for native SAO."""
from __future__ import annotations

from pathlib import Path
import time
import uuid
import json

import httpx
from areal.api import RolloutWorkflow
from areal.experimental.openai import ArealOpenAI
from areal.utils.hf_utils import load_hf_tokenizer

from .common import write_json
from .domain_api import get_domain
from .replay import interaction_reward, replay_interaction


class MathWorkflow(RolloutWorkflow):
    def __init__(self, role, tokenizer, gconfig, harness_path, receipts_dir,
                 evaluator_url=None, total_tokens=4096, interaction_credit=0.0,
                 recycle_pool=None, domain="math", context_tokens=8192):
        self.role = role
        self.tokenizer = load_hf_tokenizer(tokenizer)
        self.gconfig = gconfig
        self.domain = get_domain(domain)
        self.context_tokens = context_tokens
        self.harness = self.domain.harness_spec(Path(harness_path).read_text())
        self.receipts = Path(receipts_dir)
        self.evaluator_url = evaluator_url
        self.total_tokens = total_tokens
        self.interaction_credit = interaction_credit
        self.recycled = {}
        if recycle_pool:
            import json
            self.recycled = {e["id"]: e for e in json.loads(Path(recycle_pool).read_text())}

    async def score(self, source, sample_id, reference_source=None, metadata=None):
        if not self.evaluator_url:
            raise ValueError("Editor workflow requires current counterpart evaluator")
        payload = {"source": source, "reference_source": reference_source,
                   "sample_id": sample_id}
        if metadata is not None:
            payload["metadata"] = metadata
        timeout = getattr(self.domain, "EVALUATOR_TIMEOUT", 1800)
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            response = await client.post(self.evaluator_url + "/score", json=payload)
            if not response.is_success:
                raise RuntimeError(
                    f"Evaluator returned HTTP {response.status_code}: {response.text[:2000]}")
            return response.json()

    async def arun_episode(self, engine, data):
        started = time.monotonic()
        metadata = data.get("metadata")
        if data.get("metadata_json") is not None:
            if metadata is not None:
                raise ValueError("Workflow row has duplicate metadata encodings")
            metadata = json.loads(data["metadata_json"])
        if self.role == "solver" and data.get("warmup"):
            # Critic warm-up batch: authentic partner-evaluation episodes, no new samples.
            from .replay import recycled_interactions
            result = {}
            for episode_id in data["recycled_ids"]:
                result.update(recycled_interactions(self.recycled[episode_id], self.tokenizer))
            if not result:
                raise ValueError("Warm-up row carries no recycled partner episodes")
            write_json(self.receipts / (uuid.uuid4().hex + ".json"), {
                "id": data["id"], "role": "solver_warmup", "replayed": True,
                "recycled_partner_episodes": data["recycled_ids"],
                "generated_tokens": 0, "elapsed_seconds": time.monotonic() - started,
            })
            return result
        if self.role == "editor" and data.get("replay"):
            evaluation = await self.score(
                data["replay"]["source"], data["id"],
                data["replay"]["reference_source"], metadata,
            )
            refreshed = float(evaluation["reward"])
            reward, interaction_gain = interaction_reward(
                evaluation, float(data["replay"]["old_reward"]), self.interaction_credit)
            result = replay_interaction(data, self.tokenizer, reward)
            write_json(self.receipts / (uuid.uuid4().hex + ".json"), {
                "id": data["id"], "role": "editor", "replayed": True,
                "reward": reward, "old_reward": data["replay"]["old_reward"],
                "evaluation": evaluation, "origin": data["replay"]["origin"],
                "elapsed_seconds": time.monotonic() - started,
                "behavior_probabilities_reused": True, "reward_recomputed": True,
                "refreshed_delta": refreshed, "cross_role_interaction_gain": interaction_gain,
                "interaction_credit_coefficient": self.interaction_credit,
                "historical_negative_epoch_versions_excluded_from_native_lag_metrics": True,
            })
            return result

        client = ArealOpenAI(engine=engine, tokenizer=self.tokenizer,
                             # Qwen3-2507 emits hermes JSON tool calls; AReaL's "qwen" maps to
                             # the Qwen3-Coder XML parser, which silently parsed none of them.
                             chat_template_type="concat",
                             tool_call_parser=getattr(self.domain, "TOOL_CALL_PARSER", "hermes"),
                             reasoning_parser=getattr(self.domain, "REASONING_PARSER", "qwen3"),
                             engine_max_tokens=self.context_tokens)
        completions = []

        async def generate(messages, tools, max_tokens, seed):
            kwargs = dict(messages=messages, max_completion_tokens=max_tokens,
                          temperature=self.gconfig.temperature, top_p=1.0)
            extra_body = getattr(self.domain, "SOLVER_EXTRA_BODY", None)
            if extra_body:
                kwargs["extra_body"] = extra_body
            if tools:
                kwargs["tools"] = tools
            try:
                completion = await client.chat.completions.create(**kwargs)
            except ValueError as exc:
                if "exceeds engine_max_tokens" not in str(exc):
                    raise
                from .retool import ContextExhausted
                raise ContextExhausted(str(exc)) from exc
            completions.append(completion)
            return completion.model_dump()

        if self.role == "editor":
            messages = list(data["messages"])
            prefill = getattr(self.domain, "EDITOR_ASSISTANT_PREFILL", None)
            if prefill is not None:
                messages.append({"role": "assistant", "content": prefill})
            completion = await client.chat.completions.create(
                messages=messages, max_completion_tokens=self.total_tokens,
                temperature=self.gconfig.temperature, top_p=1.0,
                extra_body=getattr(self.domain, "EDITOR_EXTRA_BODY", None),
            )
            completions.append(completion)
            action = completion.choices[0].message.content or ""
            evaluation = await self.score(action, data["id"], metadata=metadata)
            reward = float(evaluation["reward"])
            episode = {"evaluation": evaluation, "action_text": action,
                       "input_messages": messages}
        else:
            episode = await self.domain.run_episode(
                data, self.harness, generate, total_tokens=self.total_tokens, seed=42)
            reward = float(episode["reward"])
        if not completions:
            raise RuntimeError("Episode generated no policy action")
        client.set_reward(completions[-1].id, reward)
        client.apply_reward_discount(turn_discount=1.0)
        calls = []
        for interaction in client.export_interactions(style="individual").values():
            response = interaction.model_response
            if response is None:
                raise RuntimeError("Missing native rollout tokens")
            calls.append({
                "prompt_ids": response.input_tokens, "response_ids": response.output_tokens,
                "response_logprobs": response.output_logprobs,
                "versions": response.output_versions, "stop_reason": response.stop_reason,
                "temperature": self.gconfig.temperature,
            })
        recycled_ids = list(data.get("recycled_ids") or [])
        write_json(self.receipts / (uuid.uuid4().hex + ".json"), {
            **episode, "id": data["id"], "role": self.role, "reward": reward,
            "calls": calls, "replayed": False,
            "partner_version": data.get("partner_version", 0),
            "metadata": metadata,
            "harness_source": self.harness["source"],
            "elapsed_seconds": time.monotonic() - started,
            "generated_tokens": sum(len(call["response_ids"]) for call in calls),
            "recycled_partner_episodes": recycled_ids,
        })
        # One exact sequence per turn: re-rendered tool-call JSON need not reproduce the
        # sampled tokens, so concatenating turns could misalign behavior log-probabilities.
        result = client.export_interactions(style="individual" if self.role == "solver" else "concat")
        if recycled_ids:
            from .replay import recycled_interactions
            for episode_id in recycled_ids:
                result.update(recycled_interactions(self.recycled[episode_id], self.tokenizer))
        return result
