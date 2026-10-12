"""Local dataset/rubric adapter for the release's missing ReTool env.py loader."""
import importlib.util
import inspect
from pathlib import Path
import sys

import pandas as pd
from datasets import Dataset
import verifiers as vf
from environments.retool.base_harness import ReToolEnv
from math_reward import binary_reward, last_assistant, native


def correct_answer(completion, answer, **kwargs):
    return binary_reward(last_assistant(completion), answer)


def format_reward(completion, **kwargs):
    return float(native.last_boxed_only_string(last_assistant(completion)) is not None)


def correct_reward(completion, answer, **kwargs):
    return correct_answer(completion, answer, **kwargs)


def assistant_tokens(state, **kwargs):
    return float(ReToolEnv._assistant_tokens_used(state))


def load_environment(harness_path=None, eval_dataset_path=None, num_eval_examples=-1, **kwargs):
    cls = ReToolEnv
    if harness_path:
        path = Path(harness_path).resolve()
        canonical = Path(inspect.getfile(ReToolEnv)).resolve()
        # Native h0 is a byte-for-byte copy containing its own class definition.
        if path.read_bytes() != canonical.read_bytes():
            spec = importlib.util.spec_from_file_location("retool_candidate_" + str(abs(hash(str(path)))), path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            cls = module.CandidateEnv
    assert issubclass(cls, ReToolEnv)
    tools = cls.TOOLS
    assert len(tools) == 1 and tools[0].__name__ == "code_interpreter"
    assert next(iter(inspect.signature(tools[0]).parameters)) == "code"
    raw = pd.read_parquet(eval_dataset_path).to_dict("records")
    if num_eval_examples > 0:
        raw = raw[:num_eval_examples]
    # Native template contains literal LaTeX braces as well as {question}.
    rows = [{"prompt": [{"role": "user", "content": cls.USER_PROMPT_TEMPLATE.replace("{question}", r["extra_info"]["question"])}],
             "answer": r["reward_model"]["ground_truth"]["target"][0]} for r in raw]
    data = Dataset.from_list(rows)
    rubric = vf.Rubric()
    rubric.add_reward_func(correct_answer)
    rubric.add_metric(format_reward)
    rubric.add_metric(correct_reward)
    rubric.add_metric(assistant_tokens)
    return cls(dataset=data, eval_dataset=data, system_prompt=cls.SYSTEM_PROMPT,
               tools=tools, rubric=rubric, max_turns=min(16, cls.MAX_TURNS))
