"""AReaL rollout adapters; policy losses and optimizer loops remain upstream."""
from __future__ import annotations

import uuid
from pathlib import Path

import httpx
from areal.api import RolloutWorkflow
from areal.experimental.openai import ArealOpenAI

from .common import (
    RETRY_MESSAGE, binary_reward, harness_spec, solver_messages, write_json,
)


class CoevolveWorkflow(RolloutWorkflow):
    def __init__(
        self, role, tokenizer, gconfig, harness_path, receipts_dir,
        total_tokens=512, evaluator_url=None,
    ):
        from areal.utils.hf_utils import load_hf_tokenizer
        if role not in {"editor", "solver"}:
            raise ValueError(role)
        self.role = role
        self.tokenizer = load_hf_tokenizer(tokenizer)
        self.gconfig = gconfig.new(n_samples=1)
        self.spec = harness_spec(Path(harness_path).read_text())
        self.receipts_dir = Path(receipts_dir)
        self.total_tokens = total_tokens
        self.evaluator_url = evaluator_url

    async def arun_episode(self, engine, data):
        client = ArealOpenAI(engine=engine, tokenizer=self.tokenizer,
                             chat_template_type="concat")
        messages = [dict(m) for m in data["messages"]] if self.role == "editor" else solver_messages(
            self.spec, data["question"]
        )
        remaining = self.total_tokens
        completions = []
        turn_limit = 1 if self.role == "editor" else self.spec["MAX_TURNS"]
        for _ in range(turn_limit):
            completion = await client.chat.completions.create(
                messages=messages, max_completion_tokens=remaining,
                temperature=self.gconfig.temperature, top_p=1.0,
            )
            text = completion.choices[0].message.content or ""
            completions.append(completion)
            messages.append({"role": "assistant", "content": text})
            if completion.usage is None or completion.usage.completion_tokens <= 0:
                raise RuntimeError("Missing generation token accounting")
            remaining -= completion.usage.completion_tokens
            if self.role == "editor" or remaining <= 0 or "\\boxed{" in text:
                break
            messages.append({"role": "user", "content": RETRY_MESSAGE})

        last_text = completions[-1].choices[0].message.content or ""
        if self.role == "editor":
            if not self.evaluator_url:
                raise ValueError("Editor requires a frozen solver evaluator URL")
            async with httpx.AsyncClient(timeout=600, trust_env=False) as session:
                response = await session.post(
                    self.evaluator_url + "/score", json={"source": last_text}
                )
                response.raise_for_status()
                evaluation = response.json()
            reward = float(evaluation["reward"])
        else:
            reward = binary_reward(last_text, data["answer"])
            evaluation = {"correct": bool(reward)}
        # Concat export uses the terminal reward and masks all observations.
        client.set_reward(completions[-1].id, reward)
        client.apply_reward_discount(turn_discount=1.0)
        trajectory = client.export_interactions(style="concat")
        write_json(self.receipts_dir / (uuid.uuid4().hex + ".json"), {
            "id": data["id"], "role": self.role, "reward": reward,
            "evaluation": evaluation, "messages": messages,
            "completion_ids": [c.id for c in completions],
            "tokens": self.total_tokens - remaining,
            "harness_source": self.spec["source"],
        })
        return trajectory
