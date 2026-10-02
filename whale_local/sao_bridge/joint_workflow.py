"""One native action trajectory per role, sharing a cooperative terminal outcome."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

import torch
from areal.api import RolloutWorkflow
from areal.api.cli_args import vLLMConfig
from areal.experimental.openai import ArealOpenAI
from areal.utils.hf_utils import load_hf_tokenizer

from . import alfworld_domain as domain
from .alfworld_train_data import _immutable_json
from .common import ROOT, read_json, write_json
from .joint_critic import attach_joint_context
from .joint_round import checked_behavior, record_failure, round_path, wait_record, wait_versions


@dataclass
class JointVLLMConfig(vLLMConfig):
    logprobs_mode: str = "processed_logprobs"

    def __post_init__(self):
        if self.logprobs_mode != "processed_logprobs":
            raise ValueError("Joint rollouts require processed behavior logprobs")


def parse_proposal(text: str) -> tuple[dict, str | None]:
    sys.path.insert(0, str(ROOT / "third_party/harness-r1-upstream/code/Relax"))
    from examples.harness_r1 import reward_alfworld_patch as reward

    patch, error = reward._parse_patch(
        text, response_protocol="prefill_think_patch", require_code_hook_only=True,
    )
    if patch is not None:
        error = reward._alfworld_runtime_incompatible_reason(patch)
        if error:
            patch = None
    if patch is None:
        return {"benchmark": "alfworld", "actions": []}, error or "invalid patch"
    return patch, None


class JointWorkflow(RolloutWorkflow):
    def __init__(self, role, tokenizer, gconfig, joint_root, total_tokens=4096,
                 context_tokens=32768, max_lag=0):
        if role not in ("editor", "solver"):
            raise ValueError("Unknown joint role")
        self.role, self.root = role, Path(joint_root)
        self.tokenizer = load_hf_tokenizer(tokenizer)
        self.gconfig = gconfig
        self.tokens, self.context_tokens = total_tokens, context_tokens
        if type(max_lag) is not int or max_lag not in (0, 1):
            raise ValueError("Only verified zero/one-round joint staleness is supported")
        self.max_lag = max_lag

    async def arun_episode(self, engine, data):
        version, slot = data["joint_version"], data["joint_slot"]
        root = round_path(self.root, version)
        await wait_versions(self.root, max(0, version - self.max_lag))
        started = time.monotonic()
        client = ArealOpenAI(
            engine=engine, tokenizer=self.tokenizer, chat_template_type="concat",
            tool_call_parser=domain.TOOL_CALL_PARSER, reasoning_parser=domain.REASONING_PARSER,
            engine_max_tokens=self.context_tokens,
        )
        completions = []
        identity = {"sample_id": data["id"], "version": version, "slot": slot}
        try:
            if self.role == "editor":
                messages = [*data["messages"], {"role": "assistant", "content": domain.EDITOR_ASSISTANT_PREFILL}]
                completion = await client.chat.completions.create(
                    messages=messages, max_completion_tokens=self.tokens,
                    temperature=self.gconfig.temperature, top_p=1.0,
                    extra_body=domain.EDITOR_EXTRA_BODY,
                )
                completions.append(completion)
                action = completion.choices[0].message.content or ""
                patch, error = parse_proposal(action)
                proposal = {
                    **identity, "patch": patch, "model_error": error, "action": action,
                    "generated_at_unix": time.time(),
                }
                _immutable_json(root / f"proposal-{slot:03d}.json", proposal)
                episode = await wait_record(root / f"outcome-{slot:03d}.json", self.root)
                if any(episode[key] != value for key, value in identity.items()):
                    raise ValueError("Joint terminal outcome belongs to a different episode")
                context = {
                    "joint_version": version, "role": self.role,
                    "maximum_behavior_version_lag": self.max_lag,
                    "solver_task_before_action": data["task_context"],
                    "solver_observation_protocol": "ALFWorld admissible actions; frozen joint-version Solver.",
                }
            else:
                proposal = await wait_record(root / f"proposal-{slot:03d}.json", self.root)
                if any(proposal[key] != value for key, value in identity.items()):
                    raise ValueError("Joint harness belongs to a different episode")
                await wait_record(root / "ENVIRONMENT_READY.json", self.root)

                async def generate(messages, tools, max_tokens, seed):
                    kwargs = {
                        "messages": messages, "max_completion_tokens": max_tokens,
                        "temperature": self.gconfig.temperature, "top_p": 1.0,
                        "extra_body": domain.SOLVER_EXTRA_BODY,
                    }
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

                row = {**data, "task_name": f"joint-{version}-{slot}"}
                episode = await domain.run_episode(
                    row, domain.harness_spec(json.dumps(proposal["patch"])), generate,
                    total_tokens=self.tokens, seed=42,
                )
                episode = {**episode, **identity}
                context = {
                    "joint_version": version, "role": self.role,
                    "maximum_behavior_version_lag": self.max_lag,
                    "engineer_pre_action_evidence": data["critic_evidence"],
                    "engineer_action_status": "valid" if proposal["model_error"] is None else "invalid_no_patch",
                    "solver_task_before_action": data["task_context"],
                }
            if not completions:
                raise RuntimeError("Joint role produced no policy action")
            reward = float(episode["reward"])
            if reward not in (0.0, 1.0):
                raise ValueError("Joint team outcome must be binary ALFWorld success")
            client.set_reward(completions[-1].id, reward)
            calls = []
            for interaction in client.export_interactions(style="individual").values():
                response = interaction.model_response
                if response is None:
                    raise RuntimeError("Joint rollout omitted authentic behavior tokens")
                calls.append({
                    "prompt_ids": response.input_tokens, "response_ids": response.output_tokens,
                    "response_logprobs": response.output_logprobs,
                    "versions": response.output_versions, "stop_reason": response.stop_reason,
                })
            checked_behavior(calls, version, self.max_lag)
            interactions = client.export_interactions(style="concat")
            if len(interactions) != 1:
                raise ValueError("Each joint role must supply one complete concatenated trajectory")
            trajectory = next(iter(interactions.values())).to_tensor_dict()
            expected = sum(len(call["response_ids"]) for call in calls)
            if int(trajectory["loss_mask"].sum()) != expected:
                raise ValueError("Joint concat lost policy actions from its loss mask")
            context_ids = self.tokenizer.encode(
                "Privileged pre-action cooperative state:\n" + json.dumps(context, sort_keys=True),
                add_special_tokens=False,
            )
            trajectory = attach_joint_context(trajectory, context_ids)
            trajectory["joint_round"] = torch.tensor([version], dtype=torch.long)
            trajectory["joint_slot"] = torch.tensor([slot], dtype=torch.long)
            write_json(root / f"{self.role}-receipt-{slot:03d}.json", {
                **identity, "role": self.role, "reward": reward, "calls": calls,
                "recorded_at_unix": time.time(),
                "context": context, "context_tokens": len(context_ids),
                "patch": proposal["patch"], "model_error": proposal["model_error"],
                "generated_tokens": expected, "elapsed_seconds": time.monotonic() - started,
                "trajectory_style": "concat", "optimizer_resets_between_rounds": False,
            })
            if self.role == "solver":
                _immutable_json(root / f"outcome-{slot:03d}.json", episode)
            return trajectory
        except BaseException as exc:
            record_failure(self.root, self.role, {
                **identity, "error": repr(exc), "failed_at_unix": time.time(),
            })
            raise
